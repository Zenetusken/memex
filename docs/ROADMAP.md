# Memex Roadmap

**Last updated:** 2026-05-20 (end of session — live-verified on RTX 4070)

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

**File count:** 74 Python files in `src/memex/` + `tests/` + `scripts/`, all parse-clean. 7 ADRs. 8 audit reports under `docs/audits/`. 24 commits on `main` (public at `github.com/Zenetusken/memex`).

**Test suite:** 88/88 green on the reference rig (Linux + pyseccomp). 86/88 + 2 seccomp-skipped on environments without privileged seccomp.

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
| Single `memex ask` (cold path) | **~21 s** | retrieve → expand_graph → rerank → assess → answer → verify → compose, 6699 tokens |
| 5 concurrent `/ask` (web server) | **24.5 s wall** | 4/5 answered + 1/5 refused; 100% GPU utilisation; peak VRAM 11.7 GB |
| Ingest 109-page slide deck | **~96 s** parse + ~6 s index | OCR off; 163 chunks, 245 figures, 13 tables |
| Re-index unchanged doc | **89 ms** | Diff-based partial re-index (audit fix) |
| Enrich 5 small docs parallel | **3.7 s** | 49 entities, 60 MENTIONS, 1 CITES |

### OCR off vs on (settled — see `docs/audits/07-ocr-ab.md`)

For born-digital PDFs: **OCR off**. 10.8× faster, zero query-outcome changes on the test deck. OCR on is the right choice only for scanned content (`MEMEX_PARSE_DOCLING_OCR=1`).

---

## What's next — prioritised

### P0 — blocking quality measurement

1. **Eval corpus assembly** — `docs/eval-corpus-plan.md` calls for 125 docs across 7 categories. Bootstrap with the first 10-15 to land a real baseline that the next stack swap can measure against. **Multi-week curator effort**; no other roadmap item should ship without it eventually.

### P1 — high-leverage code work (next sessions)

2. **PyMuPDF4LLM pre-filter** — Docling's layout model takes ~1 s/page even with OCR off. PyMuPDF4LLM extracts native-text PDFs at ~20× speed; Docling can then be reserved for the hard cases (scans, complex tables). Cuts the easy ~60% of ingest time at zero licensing cost. Listed on the eval-gated watchlist but doesn't need eval gating — it's a parse-time speedup, the markdown is byte-identical for the clean-text path.
3. **MCP HTTP auth model** — `memex serve mcp --transport http` binds to localhost with no auth. Required before exposing to anything beyond the loopback: bearer token? mTLS? OAuth flow? Pick one + document it; the `serve mcp` CLI already has the bind-host knob.
4. **Annotation UI 409 conflict surface** — `/review` currently last-write-wins on stale-sha edits. Add a 409 + inline diff UI so concurrent edits don't silently clobber each other. (Audit fix landed the manifest-before-write race; this is the user-facing surfacing.)
5. **Vault concurrency hardening** — per-doc `asyncio.Lock` shipped in `vault/store.py`, but cross-process `fcntl.LOCK_EX` for users running `memex` in two terminals simultaneously is still TBD.

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
| 5.5 | MCP HTTP auth | ⌗ **Open** — P1 above |
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

For the full per-commit log: `git log --oneline` (24 commits as of 2026-05-20).

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
