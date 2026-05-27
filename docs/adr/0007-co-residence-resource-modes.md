# ADR-0007: Co-residence Resource Modes (the VRAM-tradeoff policy layer)

- **Status**: Accepted
- **Date**: 2026-05-27
- **Deciders**: Memex core team
- **Tags**: stack, gpu, models, resources, architecture

## Context

Memex targets a single consumer GPU (reference: RTX 4070, 12 GB). The answering
stack co-resides on that one card:

- the **orchestrator** (Qwen3-8B-AWQ) runs out-of-process in vLLM (ADR-0001),
  reserving `gpu_memory_utilization × VRAM` upfront — that reservation holds
  the model weights (~5.7 GB) AND the KV cache (everything left over). The KV
  cache size, divided by `--max-model-len`, sets how much context a request can
  hold and how many requests run concurrently.
- the **embedder** + **reranker** (retrieval) load in the answering process
  (webui / MCP / CLI) — bf16 on the GPU by default (ADR-0006), ~0.8 GB + ~2 GB.

On 12 GB this is a zero-sum game: ~3 GB of "swing" VRAM goes EITHER to the
orchestrator's KV cache / context window OR to keeping the reranker on the GPU.
You cannot max both. The reranker is the load-time-OOM culprit when the
orchestrator runs at full util.

Commit `717054e` exposed the two halves of the lever as raw knobs:
`ModelSettings.{embedder,reranker}_device` (cuda/cpu, app-side) and the
orchestrator's `MEMEX_VLLM_GPU_FRACTION` / `MEMEX_VLLM_MAX_MODEL_LEN`
(serve-script env, orchestrator-side). But that is **three scattered env vars
across two processes that the operator must set by hand and keep mutually
consistent** — and the meaningful operating points (fast vs full-context) are
implicit folklore, not a named, discoverable thing.

Two further forces shaped this ADR:

1. The operating points are **capabilities, not just VRAM postures.** Moving the
   reranker to the CPU is not merely "fitting" — it reallocates the freed GPU
   into the orchestrator's **context window** (`--max-model-len`, 6144 → ~24k),
   which unlocks a *different answering mode*: feeding a whole document into one
   prompt for long-form synthesis, instead of top-k retrieval. So the choice
   spans resources AND product behaviour.
2. This is explicitly the **precursor to a dynamic VRAM manager** — a future
   component that picks the posture automatically from live free-VRAM and load.
   We want the manual, named version now, architected so the dynamic version is
   a drop-in replacement of one function's body.

## Decision Drivers

- A single, discoverable, named knob instead of three coupled env vars.
- One source of truth for the tradeoff, read identically by every consumer
  (registry, bootstrap VRAM estimate, daemon launch, CLI, webui).
- A clean seam the dynamic VRAM manager replaces without touching callers.
- Backward compatibility: existing deployments keep working unchanged.
- The module-boundary rules (`src/memex/CLAUDE.md`) — `core/` has no Memex deps;
  `daemon/ → core/`; the answering layer must not own daemon lifecycle.

## Decisions

### 1. A "mode" is a first-class named bundle resolving to a `ResourceProfile`

`CoResidenceMode = Literal["fast", "full", "gpu_only", "manual"]`
(`ModelSettings.co_residence_mode`, default `manual`). Each non-manual mode is a
**curated bundle** of the whole tradeoff, calibrated for the 12 GB reference
tier:

| mode | embedder | reranker | gpu_fraction | max_model_len | character |
|---|---|---|---|---|---|
| `fast` | cuda | cuda | 0.60 | 6144 | low-latency top-k RAG; tighter context |
| `full` | cuda | **cpu** | 0.80 | **24576** | whole-document context for long-form synthesis; slower (CPU) reranking |
| `gpu_only` | cuda | cuda | 0.72 | 6144 | all-GPU at full util — for >12 GB cards / orchestrator not co-resident |
| `manual` | explicit | explicit | (as launched) | (as launched) | honors the raw device knobs; **default**, backward-compatible |

### 2. One pure resolver is the single policy seam (`core/resources.py`)

`resolve_profile(mode, *, total_vram_gb=None, embedder_device=, reranker_device=) -> ResourceProfile`
is the ONLY place the tradeoff is encoded. `ResourceProfile` carries the
concrete posture: `embedder_device`, `reranker_device`,
`orchestrator_gpu_fraction`, `orchestrator_max_model_len`, plus human-facing
`label`/`summary`/`expected_latency`/`context_window`. `effective_devices(mode, …)`
is the registry's entry point. The module is **pure** (stdlib + pydantic +
`core/errors` only) — callers pass settings *primitives*, never a
`MemexSettings`, so there is no cycle with `core/config` (config may import
this; this never imports config). No consumer hardcodes the tradeoff.

### 3. The profile spans BOTH processes; `manual` is the floor

- **App-side**: `models/registry.py::_do_load` places the embedder/reranker via
  `effective_devices(...)`; `cli/bootstrap.py::_estimated_vram_gb` excludes a
  CPU-placed model from the GPU budget.
- **Orchestrator-side**: the daemon (`daemon/supervisor.py::restart`) launches
  vLLM with the profile's `orchestrator_gpu_fraction` /
  `orchestrator_max_model_len` (via the existing `MEMEX_VLLM_*` env the serve
  script already reads). `manual` leaves the orchestrator as launched (`None`).
- `manual` (default) echoes the explicit device knobs → a deployment that never
  sets a mode behaves exactly as before `717054e`.

### 4. Modes are capabilities, not just VRAM postures

`full` is not "the same RAG with a bigger KV cache." Its raised `max_model_len`
is the lever for a distinct answering path — full-document synthesis
(`agents/synthesize.py`, free-form summary baseline first; structured/grounded
output is the deferred end goal). A mode therefore couples a resource posture to
an answering capability. (See "Expanding … horizontally" for how new
capability-modes are added.)

### 5. The resolver is the dynamic-VRAM-manager precursor

`resolve_profile` is a pure function of `(mode, total_vram_gb)`. Today the
non-manual modes return curated constants and `total_vram_gb` is reserved. The
dynamic VRAM manager is delivered by **replacing the resolver's body** — compute
the profile from live free-VRAM + current load (+ card tier) — leaving every
caller (registry, bootstrap, daemon, CLI, webui) untouched. A user-triggered
*live hot-switch* (quiesce answering → swap retrieval device via
`registry.unload` → restart the orchestrator via the daemon → poll readiness)
**shipped 2026-05-27** (webui `POST /resources/mode` + `_apply_mode`; Chrome-e2e'd:
Apply `fast` restarted the daemon 24,576→6,144 live) on this same seam; the
auto-deciding policy is the increment after.

## Expanding the mode system

This is the load-bearing part of the ADR: the system is designed to grow along
two independent axes without churn.

### Vertically — deepen the resolver / a posture's fidelity (callers unchanged)

The seam is `resolve_profile`. Anything that makes a profile *smarter or finer*
lives behind it:

1. **VRAM-aware computation (→ the dynamic manager).** Replace the curated
   constants with a computation over `total_vram_gb` (then live free-VRAM +
   in-flight load): derive `gpu_fraction` so weights + a target KV fit, set
   `max_model_len` from the resulting KV, decide reranker placement from the
   leftover. Callers still just read `ResourceProfile` — no signature change.
2. **Per-card-tier calibration.** Add tier tables (8 / 12 / 16 / 24 GB); the
   resolver selects by `total_vram_gb`. A 24 GB card runs `full` all-GPU at a
   high util; 8 GB pushes both retrieval models to CPU even in `fast`.
3. **More knobs on `ResourceProfile`.** New fields (rerank batch size, KV-cache
   dtype, orchestrator quant tier, embedder device independent of reranker) are
   added to the model + set per mode + consumed where relevant. Additive: an
   unset field defaults to today's behaviour.
4. **Runtime transitions (SHIPPED 2026-05-27).** The coordinator that applies a
   profile *live* (the hot-switch) is the webui's `_apply_mode` / `POST
   /resources/mode`: it mutates `settings.models.co_residence_mode` (the registry
   shares that object), `registry.unload`s the embedder + reranker (per-model
   `asyncio.Lock` makes use-vs-unload safe — the quiesce), and `daemon.restart`s the
   orchestrator. It did NOT change the resolver or the profile shape — a pure
   vertical deepening of "apply a profile." (Adds the documented `webui → daemon`
   + `webui → models.registry` edges.)

### Horizontally — add new modes / capabilities / resource axes

1. **A new mode** = add a value to `CoResidenceMode`, a curated branch in
   `core/resources.py::_curated` (+ `_CURATED_ORDER` for display), and — if it
   unlocks a capability — the answering path it enables. Candidates:
   `throughput`/`batch` (high `max-num-seqs`, both retrieval on CPU, for eval
   sweeps / multi-client), `scan_ocr` (the VLM co-resident, for scan-heavy
   ingest), `index_only` (no orchestrator; all GPU to embedding/indexing). The
   registry/daemon/UI consume the new profile automatically.
2. **A new resource axis** = a new model/process to place (e.g. a second small
   model, a separate summarizer) → a new `ResourceProfile` field + the consumer
   that honors it (a registry slot, a daemon arg). The resolver assigns it per
   mode.
3. **A new capability tier on an existing posture.** `full` today is free-form
   summary; `full` + *structured/grounded* output (the deferred end goal) is a
   horizontal capability layered on the same resource posture — the answering
   path branches on the requested output shape, not on a new resource bundle.

**Invariant for both axes:** the orchestrator never silently CPU-offloads its
*weights* (ADR-0001 — that trades per-token latency catastrophically); modes
only move the small *retrieval* models and resize the orchestrator's KV /
context. A mode may make answering slower or narrower, never unsafe.

## Consequences

### Positive

- One discoverable, named knob (`memex mode set full`) replaces three coupled
  env vars; `memex mode show` + the webui panel make the posture observable
  (ADR-0004 spirit).
- The dynamic VRAM manager is now a body-swap of one pure function, not a
  cross-cutting rewrite.
- `full` mode makes long-context document synthesis possible on a 12 GB card —
  a capability the fixed RAG posture could not express.
- Backward-compatible: `manual` default = pre-`717054e` behaviour.

### Negative / Trade-offs

- A mode spans two processes, so a full apply (esp. the orchestrator's util /
  max-model-len) requires the orchestrator to be **daemon-managed** (`memex
  daemon start`, which owns the PID); a hand-launched `serve-vllm.sh` is changed
  only by relaunching it with the prescribed env. This is a documented
  precondition, not transparent.
- Curated constants are 12 GB-calibrated; other tiers are conservative until the
  vertical VRAM-aware computation lands.
- `full`'s CPU reranking adds ~20 s/query (a per-query, not per-token, cost).

### Neutral

- The mode is a thin policy layer; the underlying device + util knobs remain
  usable directly via `manual`.
- `total_vram_gb` is plumbed through the resolver now but unused — an
  intentional, documented seam.

## Alternatives in Detail

### Keep the three raw env vars (no mode abstraction)
Rejected: not discoverable, easy to set inconsistently across two processes, and
gives the dynamic manager nowhere to plug in.

### A full free-form config matrix (expose every knob, no named bundles)
Rejected for the default surface: maximally flexible but pushes the
VRAM-arithmetic onto the operator. `manual` already provides the escape hatch
for that audience; named modes are the curated default.

### Jump straight to an auto-only dynamic VRAM manager (no named modes)
Rejected as the first step: the auto policy needs a baseline (esp. whether
single-shot long-context summarization holds up vs map-reduce) before it can
decide well, and named modes remain the right user-facing vocabulary even once
auto-selection exists (an override / an explanation of what the manager chose).

## Revisit When

- The dynamic VRAM manager lands (the resolver becomes VRAM-aware) — fold its
  policy into §5 here.
- A second card tier is calibrated, or a non-Qwen3-8B orchestrator changes the
  weight/KV arithmetic.
- The structured-summary capability ships (it may add a capability tier and/or a
  resource axis per "Expanding horizontally").

## References

- ADR-0001 (vLLM as sole inference engine; no CPU weight offload for the orchestrator)
- ADR-0006 (dtype dispatch; bf16-on-GPU default + the `717054e` retrieval device override)
- `src/memex/core/resources.py` (the resolver), `models/registry.py`,
  `cli/bootstrap.py`, `daemon/supervisor.py`, `docs/deploy/hardware-tiers.md`
