# Memex Roadmap

**Last updated:** 2026-05-21 (end-of-session — entire P1 backlog shipped + P2.1 reranker infrastructure landed)

The blueprint in [`IMPLEMENTATION-PLAN.md`](IMPLEMENTATION-PLAN.md) is the architectural design — module signatures, cross-cutting concerns, build order. This document is the **operational view**: what is shipped today, what is measured, and what comes next.

---

## Status at a glance

| Phase | Goal | Status |
|---|---|---|
| 0 | Answering loop end-to-end on a hand-built vault | ✅ **Done** (2026-05-19) |
| 1 | Real documents in, queryable out | ✅ **Done** (2026-05-19) |
| 2 | The vault gets smarter — enrich + graph + watcher | ✅ **Done** (2026-05-20) |
| 3 | External interfaces — MCP + web UI | ✅ **Done** (2026-05-20) |
| 4 | Production hardening | ✅ **Done** (2026-05-20) |
| v1.x | Citation reasoning, daemon supervisor, sandboxed Docling, partial re-indexing | ✅ **All shipped** (2026-05-20) |
| Bug-hunt audit | Four-agent fan-out → ~70 findings → 30+ fixes | ✅ **Shipped + verified** (2026-05-20) |
| End-to-end on real rig | Install + Qwen3-8B-AWQ + 109-page PDF + load test + OCR A/B | ✅ **Verified** (2026-05-20) |
| **P1.1** | **PyMuPDF4LLM pre-filter with tiered routing classifier** | ✅ **Shipped + live-verified** (2026-05-21) |
| **P1.6** | **Chunker tuning + rerank-batch default for dense PyMuPDF chunks** | ✅ **Shipped + live-verified** (2026-05-21) |
| **P1.2** | **MCP HTTP bearer-token auth + safe-bind default** | ✅ **Shipped + live-verified** (2026-05-21) |
| **P1.4** | **Annotation UI 409 conflict surface + vault CAS** | ✅ **Shipped + live-verified** (2026-05-21) |
| **P1.5** | **Cross-process vault lock (fcntl.LOCK_EX)** | ✅ **Shipped + live-verified** (2026-05-21) |
| **P2.1-infra** | **Qwen3-Reranker backend wired behind a feature flag** | ✅ **Infrastructure shipped** (2026-05-21); quality A/B and VRAM-fit on 12 GB rig still pending |
| **P3.2** | **Daemon process model templates (systemd + launchd)** | ✅ **Shipped + live-verified** (2026-05-21) — vLLM + web + MCP all covered |

**File count:** 83 Python files in `src/memex/` + `tests/` + `scripts/`, all parse-clean. 7 ADRs. 8 audit reports under `docs/audits/`. 42 commits on `main` (public at `github.com/Zenetusken/memex`).

**Test suite:** 144/144 green on the reference rig (141 pre-existing + 3 rerank-dispatch unit tests). Linux + pyseccomp; 5 tests skip on Windows.

---

## Verified production baseline (RTX 4070, 12 GB)

Measured live, not estimated. Numbers are from the 2026-05-20 E2E + load + OCR audits in `docs/audits/05`–`07`.

### Reference stack (in `pyproject.toml` + `scripts/serve-vllm.sh`)

| Component | Pinned |
|---|---|
| Python | 3.12+ |
| Torch | `2.11.0+cu129` (`pytorch-cu129` index) |
| Torchvision | `0.26.0+cu129` (direct dep so source override applies) |
| Transformers | `>=4.57.4,<5` (v5 breaks `from transformers import PreTrainedModel`) |
| Sentence-transformers | `5.5.1` |
| vLLM | `0.21.0+cu129` (`wheels.vllm.ai/0.21.0/cu129`) |
| Langchain | `>=0.3` (required by `langfuse.langchain.CallbackHandler`) |
| flash-attn | Optional, post-install: `uv pip install flash-attn --no-build-isolation` |
| pyseccomp | Linux-only, `[parse]` extra |
| HTMX | `1.9.10` (vendored via `scripts/vendor-frontend.sh`, SHA-384 verified) |
| Tailwind | Hand-curated utility subset at `webui/static/tailwind.css` (no CDN) |

### Reference model mix

| Model | Used as | Resident VRAM |
|---|---|---|
| `Qwen/Qwen3-8B-AWQ` | Orchestrator (vLLM) — `awq_marlin` kernel | 5.7 GB weights + ~2 GB KV cache |
| `google/embeddinggemma-300m` | Embedder (lazy load in process) | ~0.6 GB BF16 |
| `BAAI/bge-reranker-v2-m3` | Reranker (lazy load in process) | ~2 GB BF16 |
| `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` | VLM, **disabled by default** | ~5 GB AWQ-Int4 (off the 12 GB budget) |

### Tuned defaults

- vLLM: `--gpu-memory-utilization 0.62 --max-model-len 4096 --kv-cache-dtype fp8_e5m2 --quantization awq_marlin --enable-prefix-caching --max-num-seqs 8`
- Parse: `docling_timeout_s=1200`, `do_ocr=False` (`MEMEX_PARSE_DOCLING_OCR=1` to opt in for scanned docs)
- Index: `embed_batch=32` (`MEMEX_INDEX_EMBED_BATCH=...` to push throughput on bigger rigs)
- Observability: `langfuse_enabled=False` by default (local-first, opt-in tracing)
- Vault: `~/.memex/vault` by default (or `$XDG_DATA_HOME/memex/vault`), mode 0700

### Measured performance

| Workload | Cost | Notes |
|---|---|---|
| Cold vLLM boot to reachable | **40 s** | Qwen3-8B-AWQ + awq_marlin |
| Single `memex ask` (cold path, Docling baseline) | **~21 s** | retrieve → expand_graph → rerank → assess → answer → verify → compose, 6699 tokens |
| Single `memex ask` (cold path, PyMuPDF chunks) | **~14 s** | top_k=4, batch_size=1; 1250 + 1791 + 1741 tokens |
| 5 concurrent `/ask` (web server) | **24.5 s wall** | 4/5 answered + 1/5 refused; 100% GPU utilisation; peak VRAM 11.7 GB |
| Ingest 109-page slide deck (Docling, OCR off) | **~96 s** parse + ~6 s index | 163 chunks, 245 figures, 13 tables |
| Ingest 109-page slide deck (PyMuPDF pre-filter) | **~33 s** parse + ~5 s index | 31 chunks (denser), 67 KB markdown (+67%), chart text captured natively |
| Re-index unchanged doc | **89 ms** | Diff-based partial re-index (audit fix) |
| Enrich 5 small docs parallel | **3.7 s** | 49 entities, 60 MENTIONS, 1 CITES |

### OCR off vs on (settled — see `docs/audits/07-ocr-ab.md`)

For born-digital PDFs: **OCR off**. 10.8× faster, zero query-outcome changes on the test deck. OCR on is the right choice only for scanned content (`MEMEX_PARSE_DOCLING_OCR=1`).

### Parse routing (settled in P1.1 — see `src/memex/parse/pipeline.py:_classify`)

PDFs now run through a tiered classifier before Docling. Producer metadata is the gold signal: a PowerPoint export goes to PyMuPDF4LLM (10-20× faster); an ABBYY-OCR'd PDF goes straight to Docling with OCR forced on; broken-encoding extractions fall through to Docling for re-extraction. The classifier logs the full attribution dict on every routing decision (`parse.pymupdf.classified`) so future eval-data collection is free. Defaults tuned to the canonical CUDA deck so it correctly stays on PyMuPDF (the deck's image-text is already in the PDF text layer, captured natively without OCR).

---

## What's next — prioritised

The entire P1 code-work backlog from the start-of-session prioritisation is now shipped (PyMuPDF pre-filter, chunker tuning, MCP HTTP auth, annotation UI 409, cross-process vault lock). The bottleneck is **P0 — eval corpus**. Every remaining P2 quality-A/B item depends on it, and the P1.6 chunker tradeoff + P2.1-infra Qwen3-Reranker quality claim both have measurements waiting on it.

### P0 — blocking quality measurement (load-bearing)

1. **Eval corpus assembly** — `docs/eval-corpus-plan.md` calls for 125 docs across 7 categories. **No other P2 item can ship a real verdict without this.** Multi-week curator effort end-to-end; a single-session bootstrap is feasible: source 3–5 docs in one category, label their queries + golden answers + chunk citations in the JSON query-set format, run the eval harness with `memex eval` against the live indexed vault, capture the baseline. Three pending decisions blocked by the absence of this corpus:
   - **P1.6**: is `chunk_target_tokens=400` the right default, or should it be 500–600 with a longer model context?
   - **P2.1**: does Qwen3-Reranker actually beat bge on Memex's workload, justifying its ~equal memory cost?
   - **P2.2 / P2.3**: do the orchestrator + VLM swaps clear their respective answer-quality floors?

### P2 — eval-gated stack swaps (after P0)

2. ~~**Qwen3-Reranker-0.6B infrastructure**~~ — ✅ **Infra shipped 2026-05-21** (commits `714dd32`, `b485748`). Backend flag wired; live verification revised the published "0.6 GB" footprint up to **~2.1 GB live** (autoregressive activations cost more than parameter count suggests). The swap is a quality play, not a memory play. Quality verdict pending P0.
3. **Granite 4.1-8B-Instruct** vs `Qwen/Qwen3-8B-AWQ`. Apache-2.0 license + native OpenAI-tool-calling in the chat template (vs Qwen's prompt-template tool path). Same vLLM serving infrastructure; the swap is a model-name change in `scripts/serve-vllm.sh`. Eval-gated.
4. **Qwen3-VL-8B-Instruct** vs `Qwen2.5-VL-7B`. +6–14 OCR points published; native 256 K context. Same AWQ-Int4 footprint (~5 GB). Currently disabled by default on 12 GB; the swap doesn't unblock 12 GB users (`disable_vlm=True` is still the right default there) but improves quality for the larger-VRAM tier. Eval-gated.

### P3 — infrastructure (no eval needed)

5. ~~**Daemon process model**~~ — ✅ **Shipped 2026-05-21** (commits below). Pure docs+templates: [`docs/deploy/systemd.md`](deploy/systemd.md) (Linux user unit, the recommended path), [`docs/deploy/launchd.md`](deploy/launchd.md) (macOS dev), and **nine** sibling artefacts — three pairs of `(unit, env)` for systemd (`memex-vllm.service`, `memex-web.service`, `memex-mcp.service`) plus three launchd plists (`com.memex.vllm.plist`, `com.memex.web.plist`, `com.memex.mcp.plist`). The web + MCP units carry soft `Wants=memex-vllm.service` dependencies so the full stack boots together; failures in any one service don't cascade. Live-verified on the reference rig: all three units enabled + reachable (vLLM 15 s, web 1 s, MCP 1 s); `kill -9` on the web's `uv run` parent triggered `Failed with result 'signal'` → automatic respawn 5 s later with `/healthz` back to 200. Logs flow into journald via `journalctl --user -u memex-vllm -u memex-web -u memex-mcp -f`; rotation is now systemd's job.
6. **Real-mode benchmark nightly CI** — `scripts/benchmark.py --real` measures cold start + first-token + embedding throughput. Needs a GPU runner: cloud (Lambda, RunPod, Modal) or a dedicated rig. Workflow template already in `.github/workflows/`. Decision blocker is **cost + ops**, not code.

### P4 — design decisions still owed (low urgency)

7. **Wikilink format** (per ADR-0003) — `[[doc_id]]` is committed; `[[doc_id#heading]]` for sub-document precision is unresolved. Punt until a real cross-doc citation needs it.
8. **8 GB GPU tier** — ADR-0001 says "no first-class CPU fallback." Should there be a documented 8 GB profile (Qwen3-4B-AWQ + reranker-base + no VLM)? Decision can wait until a user actually has that rig. **Note**: a smaller orchestrator would also unblock Qwen3-Reranker's quality test (P2.1) on tighter rigs.
9. **Trace retention** — `EventBus` has 30-day prune; Langfuse self-host wiring is open. Match retention windows when self-host lands.

### Filler — minor hardening (pickable in idle sessions)

When bored, working from `docs/audits/00-synthesis.md`:

- **N1** — LanceDB concurrent-search smoke test
- **N2** — `FTSStore` explicit `asyncio.Lock` for future `transaction()` use
- **N3** — `configure_client` should close prior `_client` on re-call (httpx pool linger)
- **N4** — Verify `GraphStore.close()` is no-op-safe on current ryugraph (Kuzu lineage needed it)
- **N5** — `_pid_alive` returning True on EPERM (alive-but-unkillable — document or fix)
- **N6** — `_COMPILED_GRAPH` thread-safety on first compile
- **N7** — `pypdfium2.to_pil` lifetime vs `doc.close()` ordering
- **N8** — Docling breaker `lambda` coroutine handling — verify the `await` path
- **N9** — FastMCP startup logs `PydanticUserError: get_graph_neighborsOutput is not fully defined` (forward-ref to `GraphNeighbor` in `mcp/server.py`'s `get_graph_neighbors` return type). Cosmetic — the tool still works — but worth a `from memex.index.graph_store import GraphNeighbor` at module level to silence it.

---

## Open architectural questions

Snapshot from IMPLEMENTATION-PLAN.md §5; current state.

| # | Question | Status |
|---|---|---|
| 5.1 | Chunking strategy | ✅ Shipped — header-aware ~600-tok windows, 100-tok overlap |
| 5.2 | Embedding dimensionality | ✅ Pinned — EmbeddingGemma native 768 |
| 5.3 | Prompt versioning storage | ✅ Shipped — `prompts/<name>/v<N>.md` + env-var pin |
| 5.4 | Refusal UI surface | ✅ Shipped — refused panel with retrieved chunks (web + CLI) |
| 5.5 | MCP HTTP auth | ✅ Shipped — `MEMEX_MCP__AUTH_TOKEN` + bearer middleware ([`docs/deploy/mcp-http.md`](deploy/mcp-http.md)) |
| 5.6 | Orchestrator placement | ✅ Settled — remote vLLM only; registry owns embedder/reranker/VLM |
| 5.7 | Trace retention | ⌗ Open — P4 above |

---

## Phase narrative (compressed)

Detailed per-phase log lives in git history + `docs/audits/`. The compressed version:

- **Phase 0** (2026-05-19): primitives, vault, agent skeleton, hybrid retrieval, CLI bootstrap.
- **Phase 1** (2026-05-19): real ingest (validation + Docling + crash breaker), per-page manifest, daemon stubs.
- **Phase 2** (2026-05-20): enrich + entity graph, watcher with sha-compare, VLM cascade, eval skeleton, real `daemon`/`graph`/`doctor`/`watch`/`eval` CLIs.
- **Phase 3** (2026-05-20): MCP FastMCP server (4 tools + transports) + FastAPI/HTMX web UI (5 routes).
- **Phase 4** (2026-05-20): graph view, PDF side-by-side, annotation correction, benchmark harness + CI gate, mkdocs site.
- **v1.x backlog** (2026-05-20): citation resolution + wikilinks, daemon supervisor real, watcher → bus, subprocess-sandboxed Docling, citation-graph reasoning, incremental partial re-indexing, network-egress sandbox.
- **Multi-agent bug-hunt audit** (2026-05-20): 4 specialist agents in parallel → ~70 findings → ~30 fixes. Reports at `docs/audits/00-synthesis.md` through `04-wiring.md`.
- **E2E + production tuning** (2026-05-20): cu128 → cu129 migration, Qwen3-8B-AWQ + awq_marlin tuning, OCR-off default, chunk-dedupe in upsert, `/ask` MemexError catch, langfuse-default off, vault-path default, ready banner. Reports at `docs/audits/05–07`.
- **P1.1 — PyMuPDF4LLM pre-filter** (2026-05-21): subprocess-sandboxed pymupdf worker, rich-signal collection (producer metadata, char distribution, image area, mojibake ratio, markdown structure), tiered classifier (`_classify` in `parse/pipeline.py`) with mixed-content force-OCR routing. `<br>` normalisation in the worker so chart-extracted runs become paragraph breaks. `MEMEX_RERANK_TOP_K` + `MEMEX_RERANK_BATCH_SIZE` env knobs for tight-rig context fit. 18 new tests (12 classifier unit + 6 pipeline integration).
- **P1.6 — Chunker tuning + rerank-batch default** (2026-05-21): added `IndexSettings` exposing `chunk_target_tokens` (default 400) and `chunk_overlap_tokens` (default 60), both env-tunable. Stripped PyMuPDF's structural metadata in the worker (image-omitted lines deleted; picture-text boundaries compacted to `[chart-text]` tags) — 21 % markdown shrinkage. Lowered default `MEMEX_RERANK_BATCH_SIZE` from 64 → 8 to fit 12 GB rigs running the 8B-AWQ orchestrator. Default pipeline now runs end-to-end at `top_k=10` without env workarounds; the quality side of the chunker tradeoff is eval-corpus-gated.
- **P1.2 — MCP HTTP bearer-token auth** (2026-05-21): new `McpSettings.auth_token` (`MEMEX_MCP__AUTH_TOKEN`, `SecretStr`), `mcp/auth.py` with `BearerAuthMiddleware` (constant-time, never logs the token) + `validate_bind` (refuses non-loopback bind without a token), `serve_http` branches on token presence, new `memex mcp generate-token` CLI helper, deploy doc at `docs/deploy/mcp-http.md`. 22 unit + 2 integration tests; stdio transport unaffected.
- **P1.4 — Annotation UI 409 conflict surface** (2026-05-21): new `StaleDocumentError` + `expected_sha` parameter on `write_document` (constant-time CAS inside the per-doc lock). `/review` now catches stale-sha submits, rolls back the anticipated-manifest update, and renders `_review_conflict.html` with HTTP 409 — unified diff (stdlib `difflib`) + "discard mine & reload" + "overwrite anyway" buttons. HTMX `responseHandling` configured in `base.html` to swap 4xx into the target so the panel actually shows up. Diff CSS in `style.css` (emerald-300 / red-300 / sky-300 / zinc-400). 4 unit + 3 integration tests.
- **P1.5 — Cross-process vault lock** (2026-05-21): new `vault/_file_lock.py` carries `doc_file_lock` (an async context manager that holds `fcntl.LOCK_EX` on `.memex/locks/{doc_id}.lock`) + `cleanup_lock_file` for delete-document. The flock syscall runs in the default executor with a 0.1 s polling loop so the event loop stays responsive while a structured `vault.lock.contended` / `vault.lock.acquired` audit trail is emitted on long waits. Linux + macOS get real locking; Windows falls back to no-op + one-time WARN (matches the pyseccomp Linux-only pattern). 1 unit + 3 cross-process integration tests.
- **P2.1-infra — Qwen3-Reranker backend wired** (2026-05-21): new `ModelSettings.reranker_backend` literal (`cross_encoder` default, `qwen3` opt-in), new `Qwen3RerankerHandle` + `_load_reranker_qwen3` in the registry, new `_score_qwen3` in `retrieve/rerank.py` (chat-template prompt → forward pass → softmax over cached yes/no token logits at the last position), backend-aware VRAM budget keys in `cli/bootstrap.py`. Live verification corrected the published "0.6 GB" estimate to **~2.1 GB live**: the autoregressive forward pass holds significantly more activation memory than the parameter count suggests, so the swap remains memory-neutral with the 8B orchestrator on a 12 GB rig. Quality A/B still gated on P0 (eval corpus).
- **P3.2 — Daemon templates** (2026-05-21): two waves. First wave shipped the vLLM unit + plist + deploy guides; second wave added `memex-web.service` + `memex-mcp.service` (+ matching launchd plists + env files) so the full Memex stack can boot together as user-level services. The web + MCP units carry soft `Wants=memex-vllm.service` so they cluster but tolerate vLLM downtime (document-browser / search / get_document don't need an LLM; only `/ask` does). Live-verified all three units end-to-end on the reference rig — including a `kill -9` on the web service's `uv run` parent, which triggered a clean systemd-level respawn with `/healthz` back to 200 within 6 s. Logs flow into journald (rotation is the OS's job now). No code change.

For the full per-commit log: `git log --oneline` (42 commits as of 2026-05-21).

---

## What the audit reports cover (`docs/audits/`)

| Report | Topic |
|---|---|
| `00-synthesis.md` | Cross-cutting overlap matrix from the 4-agent bug hunt |
| `01-resource-concurrency.md` | 15 findings: data-loss, leaks, races |
| `02-error-edge.md` | 23 findings: path traversal, SQL injection, exception semantics |
| `03-quality.md` | 25 findings: type discipline, doc drift, test quality |
| `04-wiring.md` | 15 findings: signature mismatches, boundary violations, broken commands |
| `05-e2e-loadtest.md` | Live verification of every audit fix on RTX 4070 |
| `06-8b-loadtest.md` | Post-tuning verification of the 8B-AWQ production target |
| `07-ocr-ab.md` | Empirical OCR off vs on settling the parse-default question |

## New env knobs (2026-05-21)

Reference for the env-tunable settings landed alongside P1.1:

| Variable | Default | When to set |
|---|---|---|
| `MEMEX_PARSE__PYMUPDF_ENABLED` | `true` | `false` disables the pre-filter entirely (all PDFs go to Docling). |
| `MEMEX_PARSE__PYMUPDF_MIN_CONFIDENCE` | `0.5` | Lower (0.3) for more aggressive PyMuPDF routing; higher (0.7) to prefer Docling on borderline cases. |
| `MEMEX_PARSE__PYMUPDF_MIXED_CONTENT_IMAGE_AREA_THRESHOLD` | `0.35` | Lower (0.20) to force-OCR more docs with image-embedded text; higher to be more conservative. |
| `MEMEX_PARSE__PYMUPDF_MIXED_CONTENT_MIN_IMAGE_HEAVY_PAGES` | `0.30` | Companion gate; both image-area AND image-heavy fractions must trip for mixed-content to fire. |
| `MEMEX_INDEX__CHUNK_TARGET_TOKENS` | `400` | Word-count target for the chunker. Default 400 ≈ 520 transformer tokens; raises to 600 on rigs with `max-model-len >= 8192` for pre-P1.6 chunk granularity. |
| `MEMEX_INDEX__CHUNK_OVERLAP_TOKENS` | `60` | Word-count overlap between chunks. Scales with `chunk_target_tokens`. |
| `MEMEX_RERANK_BATCH_SIZE` | `8` | bge-reranker pair-batch size. Empirical 12 GB-rig floor with 8B-AWQ resident; bump to 32 or 64 on bigger rigs / smaller orchestrators for ~2-8× rerank throughput. |
| `MEMEX_RERANK_TOP_K` | `10` | Reranked chunks fed to the agent. Drop to 4-5 if your chunks are large enough that 10 chunks overflow `max-model-len`. |
| `MEMEX_MCP__AUTH_TOKEN` | _(unset)_ | When set, the HTTP transport requires `Authorization: Bearer <token>` on every request (constant-time check). When unset, non-loopback binds are refused at startup. Generate via `memex mcp generate-token`. |
| `MEMEX_MODELS__RERANKER_BACKEND` | `cross_encoder` | `qwen3` flips the reranker dispatch to load `Qwen/Qwen3-Reranker-0.6B` (or whatever `MEMEX_MODELS__RERANKER` points at) via `transformers.AutoModelForCausalLM` and score via softmax-over-yes/no logits. Empirically ~2.1 GB live on the 12 GB rig — comparable to bge, not the parameter-count-implied savings. Quality A/B pending P0. |
