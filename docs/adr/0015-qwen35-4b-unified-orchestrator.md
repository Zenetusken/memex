# ADR-0015: Qwen3.5-4B as the Unified Orchestrator (+ VLM) Model

- **Status**: Accepted — orchestrator role UNIFIED on the 4B; **VLM-role unification ATTEMPTED + REVERTED 2026-06-01** (the dedicated `Qwen3-VL-8B` is retained as the doc-VLM — **partial unification is the terminal state**). See §"VLM-role unification: attempted, reverted" below.
- **Date**: 2026-06-01
- **Deciders**: Memex core team
- **Tags**: models, vllm, orchestrator, vlm, reasoning, co-residence, vram

> Amends the "Revisit When" trigger of [ADR-0007](0007-co-residence-resource-modes.md)
> ("a non-Qwen3-8B orchestrator changes the weight/KV arithmetic") and the
> VL-serve provisions of [ADR-0006](0006-cuda-dispatch-and-dtype.md) §VLM-via-vLLM.
> **Does NOT reverse [ADR-0001](0001-vllm-as-sole-inference-engine.md)** — this is a
> vLLM *model* swap, not an engine change.

## Context

Memex ran TWO separate 8 B models: a persistent **orchestrator** daemon
(`Qwen/Qwen3-8B-AWQ`, non-reasoning) on port 8000 for the grounded `/ask` +
`summarize` paths, and a short-lived parse-time **doc-VLM**
(`cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit`) for diagram transcription.

`cyankiwi/Qwen3.5-4B-AWQ-4bit` — a **unified vision-language, hybrid-reasoning**
model (Gated-DeltaNet + sparse-MoE + a vision tower; Apache-2.0; 262 K ctx;
compressed-tensors W4A16) — can serve **both** roles from **one checkpoint**,
adding a reasoning-capable base and an 8,192-token window. Two deep-research
workflows (the candidate landscape; the swap orchestration) plus hands-on live
verification settled the decision.

## Decision Drivers

- **The HARD gate is non-negotiable.** The grounded-only contract
  (`refusal_cf=1.0`, 0 hallucinations) must survive a model change.
- **Local-first / 12 GB.** Any posture must co-reside with the GPU embedder +
  reranker on the reference RTX 4070 under desktop-peak load.
- **Reversibility.** The swap must be a one-flip rollback to the validated 8 B.
- **Reuse, not a parallel stack** — the existing daemon + serve-env + co-residence
  machinery, extended, not replaced.

## Decision

Adopt `cyankiwi/Qwen3.5-4B-AWQ-4bit` as the **orchestrator**, served by the
persistent daemon, gated behind a full multi-run re-baseline and a one-flip
kill-switch. The **doc-VLM role** unifies onto the **same checkpoint** as a
sequenced follow-on (it needs a model-conditional thinking-disable + a vault
re-parse; kept off the inference-only orchestrator swap to avoid bundling
derived-state churn).

End-state: **one checkpoint, two serve lifecycles** — the persistent
orchestrator daemon (text, strict guided-JSON) and the short-lived parse-time
VLM serve (vision). They are NOT one process: the index/embed phase OOMs the
embedder co-resident with any vLLM, so parse still pauses the orchestrator
(`pause_vllm_for_gpu`); the unification buys one checkpoint + a reasoning base +
the larger window, **NOT** VRAM (the 4 B is ~6.3 GB — comparable to the 8 B-AWQ,
not half: the vision tower / linear-attn / MoE-router / MTP head stay fp16).

### Load-bearing findings (live-verified 2026-06-01)

- **The serve-env bridge is the prerequisite** (`daemon/supervisor.orchestrator_serve_env`).
  `serve-vllm.sh` hardcoded the 8 B + `awq_marlin` + `fp8_e5m2` as env defaults
  while the client sends `settings.models.orchestrator`; a config-only swap
  silently 404'd. The bridge exports `MEMEX_VLLM_{MODEL,QUANTIZATION,KV_CACHE_DTYPE}`
  from config. **`compressed_tensors` → omit `--quantization` (empty string, not
  unset)** so vLLM auto-detects the W4A16 kernel.
- **The 4 B rejects `fp8_e5m2` KV** ("fp8_e5m2 kv-cache is not supported with fp8
  checkpoints") → it MUST use `--kv-cache-dtype auto` (fp16 KV). At 0.62 util /
  8192 window this still yields ~35 k tokens of KV (~4.3× concurrency) — ample
  for top-k RAG.
- **Co-residence FITS with room**: at 0.62 util the daemon is ~7.0 GB; a 6-query
  burst with the GPU embedder + CPU reranker peaked at **7.99 GB (4.3 GB slack)**
  — not the tight regime the 8 B-at-0.72 lived in.
- **Schema-conformance holds**: strict xgrammar guided-JSON fully enforces our
  pydantic schemas on the hybrid-reasoning model — valid JSON from token 1, **no
  `<think>` leak** (strict JSON suppresses the CoT by construction), `maxItems` /
  enums / nested all bounded; counterfactuals refuse with valid JSON.
- **The 4 B's vision is accurate** but, on a FREE-FORM (non-guided-JSON) VLM call,
  the hybrid model leaks its CoT/analysis into the transcription. **`enable_thinking=false`
  (chat_template_kwargs) restores clean Markdown** — hence the VLM-role
  unification is *possible*, wired conditionally in the follow-on.

## Consequences

### Positive
- One checkpoint to manage; a reasoning-capable orchestrator base; an 8,192-token
  window (vs 6,144). The reasoning is latent on the strict-JSON grounded path and
  the genuine lever for the ungrounded expert surface ([ADR-0013](0013-ungrounded-reasoning-expert-mode.md)).

### Negative / Trade-offs
- A NEW model on the validated grounded gate — mitigated by the staged re-baseline
  + kill-switch, never committed before the gate holds.
- No VRAM saving (the unification win is operational, not memory).
- The parse pause/serve/teardown dance survives (retiring it is a separate,
  VRAM-measurement-gated investigation, not part of this decision).

### Neutral / Rollback
- The swap touches **zero derived state** — chunk_ids are content-addressed;
  embeddings / FTS5 / RyuGraph / `tables.sqlite` are orchestrator-agnostic. Rollback
  is inference-only: revert `models.orchestrator` (+ quant) to `Qwen/Qwen3-8B-AWQ`
  and `memex daemon restart`; verify `GET /v1/models` + a counterfactual smoke.

## VLM-role unification: attempted, reverted (2026-06-01)

The follow-on — point `models.vlm` at the same 4B so one checkpoint serves both
roles — was **built, validated, and REVERTED**. The 4B is a hybrid-reasoning
model, so a FREE-FORM (non-guided-JSON) transcription leaks CoT; the fix
(`models.vlm_disable_thinking` → `extra_body={"chat_template_kwargs":
{"enable_thinking": False}}` in `parse/vlm_backend._vllm_transcribe`, + a
defensive `_strip_think_block`) worked cleanly (a smoke + all 17 re-parsed docs
showed **zero CoT leak**). But the focused re-baseline (the 5 VLM corpora, N=3,
on the 4B-VL-re-transcribed vault) **failed the gate**:

- ✅ `cr350-diagrams` 3/3 PASS (ANS=11, exact match); `ccna-multidoc` 3/3 PASS (ANS 7→8).
- ❌ **`cr350-multidoc` 3/3 FAIL** — `cr350-xref-15` (a true false-premise counterfactual:
  the Cyber Kill Chain has 7 phases, "what is the 8th?") **hallucinates an 8th phase**.
  The transcribed content is *correct* (7 phases, verified vs the 8B-VL backup) — the
  4B-VL **re-chunking** reliably tips this adversarial near-miss bait where the 8B-VL
  chunking refused it 3/3.
- ◐ `slide-decks` ANS 14→11 (−3) and `handwritten` ANS 5→3 (−2) — false-refuses
  (HARD-gate-safe, but real answerability loss on chart/scan-heavy pages).

**Root cause**: the dedicated **`Qwen3-VL-8B`** (a larger, vision-specialised model)
is simply stronger at demanding diagram/scan transcription than the 4B's unified
vision; and a 4B-VL re-transcription **re-chunks** the doc, which on its own can tip
a borderline counterfactual. **Decision: KEEP the 8B-VL as the doc-VLM** — the
gate-determined fallback the orchestrator-swap plan explicitly allowed. The
revert was clean (`git checkout` of the uncommitted code/config/eval-anchors +
restore the 17 docs' `.md`/manifests/`vlm_cache` from a pre-attempt backup +
`reindex --force`); the orchestrator unification is unaffected. The
`vlm_disable_thinking` mechanism is **not** retained (reverted) — a future hybrid
doc-VLM would re-add it. **Do not retry the 4B as the doc-VLM without a stronger
result**; revisit only if a 4B-class model's vision measurably matches the 8B-VL.

## Amendment (2026-06-06): the CLI-restart silent-404 gap is closed

The serve-env bridge was wired into `daemon.supervisor.start()`/`restart()` and the
webui INGEST path, but NOT the CLI parse-time vLLM restart. A `memex parse`/`index`/
`reindex` whose chart-OCR or VLM-escalation pass entered
`parse/pipeline.py::pause_vllm_for_gpu` restarted vLLM via `_vllm_restart`, which
spawned `serve-vllm.sh` with **no serve-env** → the script's hardcoded 8B default
came up while the client kept requesting the configured 4B → **every later `/ask`
404'd "model does not exist"**, and the restarted vLLM was an untracked stray that
blocked `memex daemon restart` (needed a manual `kill -9`).

**Fix:** `orchestrator_serve_env` + the daemon `daemon_dir`/`daemon_pid_file` helpers
were hoisted from `daemon/supervisor` to **`core/model_serving.py`** (so `parse/` can
reuse them without a forbidden `parse/ → daemon/` import edge; the supervisor
re-exports them). `_vllm_restart` now (a) injects the serve-env so the restart serves
the configured orchestrator, and (b) writes `daemon_pid_file` so the restarted daemon
is supervisor-TRACKED.

**Two load-bearing spawn lessons** (each independently produced an un-killable orphan):

1. Spawn the script as `["/usr/bin/env","bash",script]` (the `start()` shape), **not
   `nohup script`** — under `nohup` the `uv run … vllm serve` group leader exited right
   after spawning the vLLM child, so the written PID was a corpse and `daemon stop`'s
   `os.getpgid(pid)` raised `ProcessLookupError`, skipping the kill and orphaning the
   port-blocking vLLM.
2. Use a SYNC **`subprocess.Popen`**, **not `asyncio.create_subprocess_exec`** —
   asyncio's subprocess transport calls `_proc.kill()` on its child when the event loop
   closes, so an asyncio-spawned detached daemon is SIGKILLed the instant the CLI's
   `asyncio.run()` returns (same dead-leader orphan; `start_new_session` can't save it).

Mirroring `start()`'s `subprocess.Popen` makes the parse-restarted daemon byte-identical
to a `memex daemon start`ed one. Live-validated: `memex index` → restart serves the 4B
(`doctor orchestrator_match=True`), the leader survives the CLI exit, and a subsequent
`memex daemon restart` cleanly takes over (no `pid_reused_clearing_file`, no stray).
Pinned by `tests/unit/test_vram_lifecycle.py` + `tests/unit/test_orchestrator_serve_env.py`.

## Revisit When

- The grounded re-baseline regresses on any corpus (a counterfactual flips to
  answered on a majority of N≥3, any hallucination, a `<think>`/non-JSON leak, a
  co-residence OOM, or a citation-precision drop beyond tolerance) → STOP, roll back.
- vLLM 0.22 lands as a separately-gated bump IF `--language-model-only` proves
  load-bearing for the fit (it is not today — the 4 B fits on 0.21.0).
- The parse-dance-retirement investigation clears its VRAM gate.

## References

- [ADR-0001](0001-vllm-as-sole-inference-engine.md), [ADR-0006](0006-cuda-dispatch-and-dtype.md),
  [ADR-0007](0007-co-residence-resource-modes.md), [ADR-0008](0008-document-summarization.md)
  (the `inference_override` swap-in seam), [ADR-0013](0013-ungrounded-reasoning-expert-mode.md)
- `[[qwen-migration-research]]` — the candidate landscape + fit/conformance verification
- The swap-orchestration deep-research workflow (2026-06-01) — the staged plan + red-team
