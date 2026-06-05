# ADR-0019: Frictionless browser ingestion — exclusive-GPU mode via CLI child subprocesses

**Status:** Accepted (v1 shipped 2026-06-04, merge `31ab97a`; hardened `7ce735d`; deferred-backlog + live-pass polish 2026-06-05, merge `587edaa`)
**Extends:** [ADR-0007](0007-co-residence-resource-modes.md) (co-residence / dynamic VRAM), [ADR-0015](0015-qwen35-4b-unified-orchestrator.md) (the orchestrator serve-env bridge), [ADR-0017](0017-audio-asr-ingestion-route.md) (the media 2 GiB cap)
**Spec:** [`docs/specs/ui-ingestion.md`](../specs/ui-ingestion.md)
**Tags:** architecture, ux, resource-management

## Context

Before this, ingestion was **CLI-only**: `memex ingest <file>` (parse → VLM/chart-OCR/ASR → index), then a separate `memex enrich <doc_id>`. The web UI could browse, ask, and summarize, but had **no upload→pipeline path** — a new document needed a terminal and a manual two-step. This was the last "captive to the CLI" surface, in tension with VISION principle 5 (composable, not captive).

Two hard constraints shape the design. **(1) The module boundary** (`src/memex/CLAUDE.md`) forbids `webui/` from importing `parse`/`index`/`enrich`/`ingest` — the webui is an `agents + vault + core` consumer, not a pipeline driver. **(2) GPU coherence**: the parse pipeline needs *exclusive* GPU. The parse-time VLM serve (~7.4 GB, ADR-0006 §4) OOMs co-resident with the webui's resident embedder/reranker ([[webui-gpu-contention-2026-06-04]]); and while the pipeline holds `pause_vllm_for_gpu` the orchestrator is *down*, so the RAG surfaces would 404 mid-answer regardless of VRAM. The CLI ingest already runs as a fresh process that pauses the orchestrator and serves the VLM — exactly the CUDA semantics the pipeline assumes — so the webui must reproduce that, not contend with it.

A naive "import parse and run it in-process" both breaks the boundary and re-creates the contention the CLI fresh-process model avoids.

## Decision Drivers

- **Module-boundary integrity** — `webui/` must not gain parse/index/enrich import edges.
- **GPU coherence** — exclusive-GPU during ingest; no co-residence OOM, no mid-answer 404.
- **Fresh-process CUDA semantics** — the pipeline assumes a clean process that pauses vLLM and serves the VLM (the per-doc chart-OCR pass restarts the orchestrator between docs).
- **Reuse, not reinvent** — `pause_vllm_for_gpu`, the manifest stages, the long-poll progress widget, the `auto` dynamic VRAM manager all already exist.
- **Single-user localhost reality** — one in-flight ingest; no multi-worker / queue infra warranted.
- **Observability** — the user watches the doc traverse the whole pipeline + live VRAM.

## Considered Options

1. **Import `parse`/`index`/`enrich` into `webui/` and run the pipeline in-process** — breaks the boundary; recreates the GPU contention.
2. **The webui spawns the existing CLI as CHILD SUBPROCESSES** (`memex ingest` then `memex enrich`), one file per `memex ingest` — *chosen*.
3. **A separate long-running ingest worker service + a job queue** (Redis/RQ-style).

## Decision

We chose **Option 2**: the webui drives ingestion by spawning the existing CLI as **child subprocesses** through a boundary-clean driver (`webui/ingest_driver.py`), inside an **exclusive-GPU "ingestion mode"**. This keeps the module boundary intact (the driver imports only `asyncio`/`json`/`os` + `core`/`webui.progress`), gives the pipeline the fresh-process CUDA it assumes, and surfaces the GPU exclusivity `pause_vllm_for_gpu` already enforces instead of silently contending with it.

Load-bearing decisions:

1. **Child-subprocess driver, one file per `memex ingest`.** `run_ingest` spawns `memex ingest <file>`, `run_enrich` spawns `memex enrich <doc_id>`. One file per invocation is also the fix for the multi-doc VLM `VRAMExhausted` (the per-doc chart-OCR pass restarts the orchestrator between docs). **Stream contract (load-bearing):** structlog milestones go to **stderr**, the `IngestResult` JSON to **stdout** at exit; both pipes are drained CONCURRENTLY (`asyncio.gather`) — awaiting one while the other fills its OS buffer would deadlock the child. The subprocess runs in its own session/group so a cancelled task SIGKILLs the whole tree (no orphaned GPU-holder).

2. **Exclusive-GPU RAG lock (single-flight).** A per-app `_IngestState` is set synchronously at `POST /ingest` (a 2nd concurrent upload is rejected). While active, the five GPU POSTs (`/ask`, `/documents/{id}/summarize`, `/chat`, `/expert`, `/bridge`) return an honest "answering paused" fragment; **browsing already-ingested docs stays fully open** (CPU-only). The lock has two load-bearing reasons, not just OOM: a racing retrieval reload OOMs the VLM serve, AND the orchestrator is down during the pause so an answer would 404 — the lock converts a broken answer into an honest "paused."

3. **The webui releases its own GPU during ingest.** `_run_ingest` `registry.unload_all()`s the resident embedder/reranker before the subprocess takes the GPU — reconciled with `auto`-mode placement (ADR-0007 P4.4). For ingestion-heavy work the webui may also run retrieval on CPU.

4. **Orchestrator serve-env injection + ACTIVE reconcile (the "D3" fix).** The subprocess's post-parse vLLM restart must bring up the *configured* orchestrator, so `run_ingest` injects `orchestrator_serve_env` (else the serve-script default 8B comes up and every later `/ask` 404s — the ADR-0015 silent-404 class). And because the subprocess's own `pause_vllm_for_gpu` restart is **unreliable on a daemon-managed rig** (the `serve-vllm.sh` fallback can't fit under the webui's residual VRAM → it hangs ~245s then leaves the orchestrator DOWN), `_run_ingest` does NOT passively wait: it **actively** `daemon_restart`s a down orchestrator (fast-path skip when already reachable — the supervisor `stop` is a no-op against a stray it doesn't own) before clearing the lock.

5. **Manifest-stage status honesty + hardening.** Status is derived from the manifest stages: 0 chunks ⇒ *browsable-but-not-searchable* (B12, not "fully consumed"); a half-doc (`ingest` but no `index`) is detected + logged on startup (B19); an in-flight RAG answer is drained (bounded) before `unload_all` (B18); a hung child is reaped by a phase-aware **silence watchdog** (a separate generous budget during ASR, which is silent for its whole duration — `ingest.{silence,asr_silence}_timeout_s`); the "ingesting" banner + the RAG-paused notice **self-clear** via a 3s self-refresh (not a one-shot OOB, which would be premature because the lock releases in a `finally` after the done-render); a pre-stream 2 GiB `Content-Length` cap rejects an over-cap upload before the body streams to disk (B11).

**All of this is GPU-lifecycle / presentation only ⇒ HARD-gate-neutral by construction** — the agent's `/ask` graph, grounding, and refusal path are never touched; an ingested doc enters the *same* parse/index/enrich/answer path as a CLI-ingested one.

## Consequences

### Positive

- The last CLI-captive surface is closed (VISION principle 5); a doc lands fully consumed (chunks + FTS + table store + entity graph + companion) and immediately askable + browsable, zero terminal.
- The boundary holds (zero new `webui → parse/index/enrich` import edges); the driver is the single sanctioned subprocess seam.
- Reuses the fresh-process CUDA the pipeline already assumes; reuses `pause_vllm_for_gpu`, the manifest stages, the long-poll progress widget, and is the user-facing payoff of the `auto` dynamic VRAM manager (ADR-0007 P4.4).
- The exclusive-GPU mode is *visible* (live VRAM panel + per-process holders), not silent contention.

### Negative / Trade-offs

- **Single-flight, by contract:** a 2nd concurrent ingest is rejected (409); answering pauses for the ingest's duration. A multi-upload queue is a deliberate v1 non-goal.
- The subprocess-spawn model adds process-lifecycle surface (orphan-kill, the silence watchdog, the orchestrator reconcile) — hardened, but real.
- The exclusive lock + the active orchestrator reconcile assume a **single GPU on localhost**; they do not scale to multi-user / multi-GPU.

### Neutral

- Page-image preview, Related-docs, and companion chips all work on a browser-ingested doc identically to a CLI-ingested one (same derived state).

## Alternatives in Detail

### Option 1: in-process `parse`/`index`/`enrich`

Rejected: breaks the `webui → parse/index/enrich` boundary (a precedent that would erode the whole layering), and re-creates the GPU contention the CLI fresh-process model exists to avoid (the VLM serve would OOM against the webui's resident models / a long-lived CUDA context).

### Option 3: a separate ingest worker service + queue

Over-engineered for a single-user localhost tool. The CLI-subprocess model already gives process isolation + fresh CUDA + a natural single-flight; a queue/worker adds operational surface (a 2nd long-lived process, IPC, a broker) for a concurrency the single-GPU rig can't exploit anyway (only one ingest can hold the GPU at a time). The multi-upload *queue* is recorded as the one deferred design item if the constraint ever changes.

## Revisit When

- **Multi-user or multi-GPU.** The exclusive-GPU lock + single-flight + the active orchestrator reconcile all assume one GPU on localhost; a second concurrent GPU consumer breaks the model.
- **A real need to queue concurrent uploads** (the deferred multi-upload queue) — then a job model + a per-job GPU scheduler is warranted.
- **The module boundary relaxes** to allow an in-process ingest path (it won't without a strong reason — fresh-process CUDA is the whole point).
- **Image-file ingestion (`.png`/`.jpg`)** lands (a future one-page scan-VLM route) — additive to this path, likely its own short ADR.

## References

- Spec: [`docs/specs/ui-ingestion.md`](../specs/ui-ingestion.md)
- Frontend conventions: `src/memex/webui/CLAUDE.md` §"Document ingestion"
- ADR-0007 (co-residence / dynamic VRAM), ADR-0015 (orchestrator serve-env), ADR-0017 (media 2 GiB cap), ADR-0006 §4 (parse-time VLM serve)
- Memory: `[[ui-ingestion-2026-06-05]]`, `[[ui-ingestion-livetest-2026-06-05]]`, `[[ui-ingestion-deferred-done-2026-06-05]]`, `[[webui-gpu-contention-2026-06-04]]`
- ROADMAP §"Beyond this phase — frictionless UI ingestion" + the 2026-06-05 changelog entry
