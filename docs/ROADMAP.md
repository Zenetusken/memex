# Memex Roadmap

**Last updated:** 2026-05-21 (P1.1 PyMuPDF4LLM pre-filter shipped + live-verified on RTX 4070)

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

**File count:** 79 Python files in `src/memex/` + `tests/` + `scripts/`, all parse-clean. 7 ADRs. 8 audit reports under `docs/audits/`. 34 commits on `main` (public at `github.com/Zenetusken/memex`).

**Test suite:** 130/130 green on the reference rig (106 pre-existing + 22 mcp-auth unit + 2 mcp-server integration). Linux + pyseccomp; 128/130 + 2 seccomp-skipped on environments without privileged seccomp.

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

### P0 — blocking quality measurement

1. **Eval corpus assembly** — `docs/eval-corpus-plan.md` calls for 125 docs across 7 categories. Bootstrap with the first 10-15 to land a real baseline that the next stack swap can measure against. **Multi-week curator effort**; no other roadmap item should ship without it eventually.

### P1 — high-leverage code work (next sessions)

2. ~~**PyMuPDF4LLM pre-filter**~~ — ✅ **Shipped 2026-05-21** (commits `9e02042`, `3773801`). 2.9× wall-clock speedup on the canonical 109-page deck (33 s vs 96 s), 67% more markdown extracted (chart text captured via PDF text operators, no OCR needed), tiered classifier with rich-signal routing, mixed-content force-OCR path for scanned/image-text docs. Default-tuned to avoid false-positive OCR on born-digital decks. See "Parse routing" above + `src/memex/parse/pymupdf_backend.py` + `pymupdf_worker.py`.
3. ~~**MCP HTTP auth model**~~ — ✅ **Shipped 2026-05-21** (commits `32b811d`, `bec5e57`, `78d9518`). `McpSettings.auth_token` (`MEMEX_MCP__AUTH_TOKEN`) gates every HTTP request via constant-time bearer-token comparison; non-loopback bind is refused at startup without a token; loopback bind without a token still runs with a clear WARN line (developer affordance). New `memex mcp generate-token` CLI prints a fresh `secrets.token_urlsafe(32)`. Full deployment story: [`docs/deploy/mcp-http.md`](deploy/mcp-http.md).
4. **Annotation UI 409 conflict surface** — `/review` currently last-write-wins on stale-sha edits. Add a 409 + inline diff UI so concurrent edits don't silently clobber each other. (Audit fix landed the manifest-before-write race; this is the user-facing surfacing.)
5. **Vault concurrency hardening** — per-doc `asyncio.Lock` shipped in `vault/store.py`, but cross-process `fcntl.LOCK_EX` for users running `memex` in two terminals simultaneously is still TBD.
6. ~~**Chunker tuning for dense PyMuPDF output**~~ — ✅ **Shipped 2026-05-21** (commits `bd948e4`, `e514d7c`, `fdf4d00`). `IndexSettings.chunk_target_tokens` defaults to 400 (was 600); `MEMEX_INDEX__CHUNK_TARGET_TOKENS` env-tunable. PyMuPDF marker stripping: `==> picture [...] intentionally omitted <==` lines deleted, `Start/End of picture text` boundaries compacted to `[chart-text]` / `[/chart-text]` tags — 21 % markdown shrinkage on the canonical deck. Default `MEMEX_RERANK_BATCH_SIZE` lowered 64 → 8 (12 GB rig empirical floor). **Infra fix complete** — default pipeline runs end-to-end at `top_k=10` with no env workarounds. **Quality tradeoff is now eval-corpus-gated**: smaller chunks improve retrieval precision but reduce per-chunk context, which surfaces refusal-rate changes on broad "what is this about" queries. P0 (eval corpus) is the right tool to settle the optimal `chunk_target_tokens` value across a real workload; until then, rigs with `max-model-len >= 8192` can bump back to 600 via env for the pre-P1.6 retrieval behaviour.

### P2 — eval-gated stack swaps (after P0)

6. **Qwen3-Reranker-0.6B** vs `BAAI/bge-reranker-v2-m3`. Memory: 0.6 GB vs 2 GB — frees ~1.5 GB on the 12 GB rig. Quality: +8 MMTEB-R on the published benchmark; autoregressive ⇒ higher per-pair latency. Gate on the `<15 s` answer-latency floor.
7. **Granite 4.1-8B-Instruct** vs `Qwen/Qwen3-8B-AWQ`. Apache-2.0 license + native OpenAI-tool-calling in the chat template (vs Qwen's prompt-template tool path). Run A/B against the eval corpus.
8. **Qwen3-VL-8B-Instruct** vs `Qwen2.5-VL-7B`. +6-14 OCR points published; native 256 K context. Same AWQ-Int4 footprint (~5 GB). Currently disabled by default on 12 GB; the swap doesn't unblock 12-GB users but does improve quality for the larger-VRAM tier.

### P3 — infrastructure

9. **Real-mode benchmark nightly CI** — `scripts/benchmark.py --real` measures cold start + first-token + embedding throughput. Needs a GPU runner: cloud (Lambda, RunPod) or a dedicated home rig. Workflow template already in `.github/workflows/`.
10. **Daemon process model** — `memex daemon start` currently runs a detached child + PID file. Production deployment would benefit from systemd unit (Linux) / launchd (macOS) templates so the OS handles restart-on-crash + log rotation.

### P4 — design decisions still owed

11. **Wikilink format** (per ADR-0003) — `[[doc_id]]` is committed but the section-anchor case (`[[doc_id#heading]]`) is unresolved. Punt until a real cross-doc citation that benefits from sub-document precision shows up.
12. **8 GB GPU tier** — ADR-0001 commits to "no first-class CPU fallback." Should there be a documented 8 GB tier with smaller model defaults (Qwen3-4B-AWQ + reranker-base + no VLM)? Decision can wait until a user actually has that rig.
13. **Trace retention** — `EventBus` has 30-day prune; Langfuse self-host hasn't been wired up. The plan is matching retention windows, but no concrete date.

### Filler — minor hardening (when bored, see `docs/audits/00-synthesis.md` nits)

- N1: LanceDB concurrent-search smoke test
- N2: `FTSStore` explicit `asyncio.Lock` for future `transaction()` use
- N3: `configure_client` should close prior `_client` on re-call
- N4: Verify `GraphStore.close()` is truly no-op-safe on the current ryugraph
- N5: `_pid_alive` returning True on EPERM (treated as alive-but-unkillable)
- N6: `_COMPILED_GRAPH` thread-safety on first compile
- N7: pypdfium2 `to_pil` lifetime vs `doc.close()` ordering
- N8: docling breaker `lambda` coroutine handling — verify

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

For the full per-commit log: `git log --oneline` (34 commits as of 2026-05-21).

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
