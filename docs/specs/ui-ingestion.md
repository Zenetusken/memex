# Spec: Frictionless browser ingestion (exclusive-GPU mode)

**Status:** SHIPPED — v1 2026-06-04 (merge `31ab97a`), hardened 2026-06-05 (`7ce735d`), deferred-backlog + live-pass polish 2026-06-05 (`587edaa`)
**ADR:** [ADR-0019](../adr/0019-frictionless-ui-ingestion.md)
**Code:** `src/memex/webui/ingest_driver.py` + `src/memex/webui/app.py` (the `/ingest*` routes + `_run_ingest`) + `webui/progress.py` (`INGEST_PHASES`) + `templates/ingest.html`/`_ingest_done.html`/`_ingesting.html`/`_ingesting_banner.html`/`_ask_ready.html`
**Frontend conventions:** `src/memex/webui/CLAUDE.md` §"Document ingestion" (the authoritative, line-level record)

The browser upload → full pipeline → askable+browsable surface. This spec is the build-level design; the decision rationale is ADR-0019.

## 1. The subprocess driver (`webui/ingest_driver.py`)

The module boundary forbids `webui/ → parse/index/enrich/ingest`, so the webui spawns the CLI as a child:

- `run_ingest(file_path, *, on_phase, extra_env, spawn, silence_timeout_s, asr_silence_timeout_s) -> IngestOutcome` runs `memex ingest <file>`.
- `run_enrich(doc_id, *, on_phase, extra_env, spawn, silence_timeout_s) -> int` runs `memex enrich <doc_id>`.
- **One file per `memex ingest`** (the multi-doc VLM `VRAMExhausted` fix — the per-doc chart-OCR pass restarts the orchestrator between docs).
- Imports only `asyncio`/`json`/`os`/`signal`/`time` + `memex.webui.progress` — **zero new boundary exceptions**.

**Stream contract (load-bearing):** structlog milestones → **stderr**, the `IngestResult` JSON → **stdout** at exit. Both pipes are drained CONCURRENTLY (`asyncio.gather` over two `_pump` readline loops) — awaiting one while the other fills its OS pipe buffer deadlocks the child. `_build_env` forces `MEMEX_OBSERVABILITY__LOG_JSON=true` + `PYTHONUNBUFFERED=1` (else progress arrives in one burst at exit) + `MEMEX_INGEST__MAX_BYTES=2GiB` + `MEMEX_PARSE__DISABLE_VLM=false`. The child runs `start_new_session=True`; a cancelled/cut-short task SIGKILLs the whole process GROUP (`_terminate` → `killpg`) so a GPU-holding parse-time vLLM is never orphaned. An over-`_PIPE_LIMIT` line (a serialized traceback) is skipped, not fatal.

`IngestOutcome{accepted, exit_code, doc_id, rejection_reason, chunk_count}` — `succeeded` iff accepted + clean exit + a doc_id. `chunk_count` is captured off the `index.done` structlog event (drives the 0-chunk gate).

## 2. Routes (`webui/app.py`)

- `GET /ingest` — the upload page; if an ingest is in flight (you navigated away and back), RESUME its live progress in the pane (`_IngestState.cid`, B7/B8).
- `POST /ingest` — stream the upload into `mkdtemp()/safe_name` (`Path(filename).name`, NOT a temp name, so the doc is titled correctly), set the per-app `_IngestState` SYNCHRONOUSLY (single-flight; a 2nd concurrent upload → 409), schedule `_run_ingest` as a background task, return the shared `_progress.html` long-polling `GET /ingest/{cid}/status`. A no-file submit → a friendly `_ingest_done.html` error fragment (B18, not a 422). An upload while `mode_switch_lock` is held → rejected (the two GPU-orchestrating ops can't race).
- `GET /ingest/{cid}/status?v=N` — the long-poll; renders `_progress.html` while running, `_ingest_done.html` on completion (3 branches: ingested-and-askable / partially-ingested-browsable / failed).
- `GET /ingest/banner` + `GET /ingest/lock` — the **self-clearing** banner + RAG-paused notice (see §6).
- A pre-stream **`_UploadSizeLimitMiddleware`** (ASGI pure-middleware) rejects an over-2-GiB `Content-Length` BEFORE the body streams to disk (B11).

`INGEST_PHASES = (Parsing, Transcribing, Indexing, Enriching)`; `ingest_phase_for(event, page)` maps the subprocess's structlog events → phase labels (`vlm.start`→"Transcribing · page N", `chart_ocr.*`→"chart OCR", `asr.transcribe.start`→"audio", `index.start`→Indexing, `enrich.start`→Enriching, `vllm.restart.start`→"Indexing · restoring the orchestrator").

## 3. The GPU-coherence body (`_run_ingest`)

1. **Drain in-flight RAG** (`_drain_inflight_rag`, bounded 8s) — await OTHER-cid answer tasks before unloading, so an answer already running when the ingest started isn't yanked out mid-run (B18).
2. **`registry.unload_all()`** (`try/except ModelNotConfigured`) — free the webui's resident embedder/reranker so the parse-time VLM serve fits.
3. **`run_ingest(extra_env=orchestrator_serve_env(settings))`** — the serve-env injection is load-bearing: the subprocess's post-parse vLLM restart must bring up the *configured* orchestrator, not the serve-script default 8B (the ADR-0015 silent-404).
4. On success: `_set_entry_doc_id` BEFORE the now-best-effort enrich (a failed enrich still renders a browsable doc) → **`_reconcile_orchestrator`** → `run_enrich`.
5. `finally`: `rmtree` the temp dir + `_reconcile_orchestrator` + clear the lock in an INNER try/finally (the lock ALWAYS clears, even if the reconcile raises).

**`_reconcile_orchestrator` (the "D3" fix):** the subprocess's own `pause_vllm_for_gpu`→restart is UNRELIABLE on a daemon-managed rig (no `systemctl` unit → the `serve-vllm.sh` fallback can't fit under the webui's residual VRAM → it hangs then leaves the orchestrator DOWN, 404ing every later `/ask`). So: if already reachable (a short retry probe), done; else ACTIVELY `daemon_restart(...)` at the current mode's posture. The fast-path skip is REQUIRED — the supervisor `stop` is a no-op against a stray vLLM it doesn't own, so restarting over a live one refuses on the bound port.

## 4. Exclusive-GPU RAG lock

`_ingest_guard(request)` returns `_ingesting.html` as the FIRST statement of every GPU POST (`/ask`, `/documents/{id}/summarize`, `/chat/{id}/turn`, `/expert`, `/bridge`) — while the subprocess holds `pause_vllm_for_gpu` the orchestrator is DOWN, so those would 404 regardless of VRAM; the lock converts a broken answer into an honest "paused." `base.html` shows the `.ingesting-banner` (amber, `role="status"`) via the `ingesting_active()` jinja global. **Browsing stays open** (`/documents`, `/entity`, `/graph`, `/resources`, source-preview — no GPU).

## 5. Status honesty (manifest stages)

- **0 chunks ⇒ browsable-not-searchable** (B12): `IngestOutcome.chunk_count == 0` → "ingested but no searchable text was extracted — browsable, but won't appear in answers"; enrich skipped.
- **Half-doc detect** (B19): `_scan_half_docs` logs a manifest with `ingest` but no `index` on startup (detect+log only, never auto-deletes; per-file `suppress` so one corrupt manifest can't hide the rest).
- **Browse-link gating:** the partial-ingest "Open document (browsable)" link is gated on the canonical `.md` actually existing (no dead 404).

## 6. Self-clearing UI

The "ingesting" banner and the RAG-paused notice live OUTSIDE the long-poll swap target, so a one-shot `hx-swap-oob` clear at the done-render would be PREMATURE — `ingesting.active` flips False in `_run_ingest`'s `finally` AFTER `progress.finish()` and the reconcile (the 0-chunk/partial paths hold the lock ~40s past "done"). So both SELF-REFRESH (the `every 5s` VRAM-panel pattern): `_ingesting_banner.html` (included by `base.html`) and `_ingesting.html` poll `GET /ingest/banner` / `GET /ingest/lock` every 3s while active; once the lock clears the routes return a trigger-less fragment (an empty banner / `_ask_ready.html` "re-run your request") that stops the poll. They clear on their own, any tab, every path, no reload.

## 7. The silence watchdog (hung-child reaper)

A hung subprocess (a wedged GPU / deadlocked VLM serve that escapes the parse workers' own `asyncio.wait_for` timeouts) emits no output for a long stretch — the webui would then wait forever, the RAG lock never releasing. `_silence_watchdog` SIGKILLs the child's process group after `ingest.silence_timeout_s` (1800s) of no output on EITHER pipe. **Phase-aware** (the load-bearing safety): ASR transcription is silent for its whole duration (faster-whisper runs the file through one blocking call), so the stderr sink toggles `_Activity.in_asr` on `asr.transcribe.start`/`.done` and the watchdog applies a separate generous `ingest.asr_silence_timeout_s` (~8h) during ASR — else legit long media is false-killed AND, since ASR caches only on success, the re-transcribe loops. A hang is reported distinctly from an OOM signal-kill (`activity.timed_out` → "no output for Ns … appears to have hung").

## 8. Live VRAM

`ingest.html` embeds the reused `/resources/vram` fragment (`hx-trigger` refresh) — the read-only nvidia-smi probe is safe mid-ingest and shows the parse model as the live GPU holder, closing the loop with the exclusive-GPU mode.

## Testing

`tests/integration/test_webui.py` (`_patch_ingest`/`_patch_daemon_reachable` fakes + `_ingest_to_completion`: streaming byte-equality, the 5-POST lock matrix + 4-GET banner, crash/stream-failure lock-release, serve-env injection asserting `MEMEX_VLLM_MODEL`, temp-dir-no-leak, the 0-chunk gate, the in-flight-RAG drain, the half-doc scan, the self-clearing banner/lock, the size-limit middleware) + `tests/integration/test_ingest_driver.py` (fake-spawn phase sequence + a real-subprocess >64KiB concurrent-drain + the silence watchdog: kill / survives-active / exempts-silent-ASR) + `tests/unit/test_progress.py` (`ingest_phase_view` + the `vllm.restart.start` mapping).

## Deferred (v1 non-goal)

A **multi-upload QUEUE** — v1 rejects a 2nd concurrent ingest with a 409 by the single-flight `_IngestState` contract. Queueing is a design change (a job model + per-job GPU scheduling), not a hardening fix, and the single-GPU rig can only run one ingest at a time anyway. Revisit on multi-user / multi-GPU (ADR-0019 "Revisit When").
