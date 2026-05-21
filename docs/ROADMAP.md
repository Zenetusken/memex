# Memex Roadmap

**Last updated:** 2026-05-21 (phase wrap — P1 backlog + P3.2 daemon stack + P2.1 infra + P0 rigorous baseline + slide-deck Docling routing + **P2.4 refusal calibration** + **xgrammar empty-draft short-circuit** + **P0 corpus extension to 30 queries** + **retrieval-truncate-budget retune** all shipped; **P2.1 Qwen3-Reranker quality A/B hardware-blocked on 12 GB rig** post-retune (the +1 GB max-model-len bump consumed the headroom that Qwen3-Reranker's 2.1 GB live footprint needed); P3.3 chart-OCR is now the highest-priority next pickup, P4.2 smaller-orchestrator is the direct unblocker for P2.1)

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
| **P2.1** | **Qwen3-Reranker backend + quality A/B** | ✅ **Infra shipped + verdict shipped** (2026-05-21). Quality A/B unblocked by P4.2 Session 4 (Qwen3-4B-AWQ + gpu_fraction=0.50 frees the headroom). **3 head-to-head runs, same stack: cross_encoder bge-reranker-v2-m3 wins clearly**: median ANS=4 vs Qwen3-Reranker's 0. Diagnostic showed Qwen3-Reranker ranks generic-CUDA chunks above the literal-answer chunk; thematic-relevance ranking is worse than fact-extraction ranking for this slide-deck corpus. Default reranker stays `cross_encoder`; qwen3 backend remains opt-in via `MEMEX_MODELS__RERANKER_BACKEND=qwen3`. |
| **P3.2** | **Daemon process model templates (systemd + launchd)** | ✅ **Shipped + live-verified** (2026-05-21) — vLLM + web + MCP + watcher; the complete stack |
| **FU3.2.1** | **`Type=notify` for vLLM** — readiness gate, not just process-forked ordering | ✅ **Shipped + live-verified** (2026-05-21) |
| **FU3.2.2** | **`memex upgrade` CLI** — one-shot pull + sync + restart-installed-units | ✅ **Shipped + live-verified** (2026-05-21) |
| **FU3.2.3** | **`memex-vault-backup.timer`** — nightly encrypted restic snapshots | ✅ **Shipped + live-verified** (2026-05-21) |
| **P0** | **Eval corpus** — JSON query sets + `memex eval` rigorous baseline | ⚠️ **Extended baseline shipped** (2026-05-21): 1 category (slide-decks), 1 doc, **30 queries** (17 answerable + 5 empty-retrieval + 8 near-miss refusals). `mcp_answered_only = 0.89`, `refusal_rate_cf = 1.00`, 9 legitimate answers, zero hallucinations. n=17 is now statistically meaningful for P2.x A/B's. Multi-document + multi-category extension is the next P0 sub-goal. |
| **Parse — slide-deck → Docling** | Tier 0.5 classifier override routing slide-shaped PDFs to Docling | ✅ **Shipped + live-verified** (2026-05-21) — `_is_slide_deck` heuristic (aspect ≥ 1.3 AND chars-per-page in [50, 800)); 147 tests green; CUDA deck now routes correctly. |
| **P2.4 — Agent refusal calibration** | `prompts/answer/v2.md` with literal-presence rule | ✅ **Shipped + live-verified** (2026-05-21) — Q11 (FP128) + Q12 (FP4) hallucinations both eliminated; `refusal_rate_cf` 0.75 → 1.0; `mcp_ans` 0.33 → 0.67. **Q7 regression recovered** as a side effect of the retrieval-truncate-budget retune that shipped later the same day. |
| **Retrieval truncate-budget retune** | `truncate(700)→1800` in prompts; `MEMEX_RERANK_TOP_K 10→5`; `max_tokens 1024→640`; `max-model-len 4096→6144` | ✅ **Shipped + live-verified** (2026-05-21) — three legitimate REF→ANS flips (Q2, Q7-recovered, Q20). Diagnosed by direct instrumentation: agent was operating on ~32% of every chunk's content because the answer prompt clipped chunks at 700 chars while median chunk is 2172 chars. |

**File count:** 83 Python files in `src/memex/` + `tests/` + `scripts/`, all parse-clean. 7 ADRs. 8 audit reports under `docs/audits/`. 69 commits on `main` (public at `github.com/Zenetusken/memex`); the canonical count is `git rev-list --count main`.

**Test suite:** 147/147 green on the reference rig. Linux + pyseccomp; 5 tests skip on Windows.

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

**Phase wrap, 2026-05-21.** This session closes a long stretch of deployment + evaluation + parser-quality + agent-prompt work:

- **P1 backlog** shipped (PyMuPDF pre-filter, chunker tuning, MCP HTTP auth, annotation UI 409, cross-process vault lock).
- **P2.1 infra** shipped (Qwen3-Reranker backend behind a feature flag; honest verdict on the memory footprint).
- **P3.2 daemon stack** shipped end-to-end (vLLM + web + MCP + watcher + vault-backup timer) with all four FU3.2.* follow-ups (Type=notify gating, `memex upgrade` CLI, vault-backup timer, FastMCP startup warning fix).
- **P0 eval corpus** shipped its first rigorous baseline (15 queries against the GTC 2024 CUDA deck; `top_k ∈ {4,6,8}` sweep; `mcp_answered_only ≈ 0.5–0.6`; `refusal_rate_cf ≈ 0.75`).
- **Parser quality investigation** delivered an evidence-backed verdict — Docling beats PyMuPDF on slide decks (+50% legitimate answer rate) — and **shipped the routing override** as Tier 0.5 of the classifier.
- **P2.4 agent refusal calibration** shipped as `prompts/answer/v2.md` (literal-presence rule). Eliminated both near-miss hallucinations (Q11 FP128, Q12 FP4); `refusal_rate_cf 0.75 → 1.0`; `mcp_answered_only 0.33 → 0.67`. Cost: one regression on Q7 (5-bit/10-bit half — diagram grounding became too strict under the new rule). The follow-up `verify_grounding/v2` xgrammar crash was diagnosed and fixed: root cause was empty-draft renders to verify under unbounded `list[int]` schema (greedy decoding ran away into `[0, 1, 2, ...]+`); fix is an empty-draft short-circuit in the verify node. With the fix, `verify_grounding/v2` ships safely. Q7 regression is independent (verify-grounding can't confidently land claims on the precision-bit diagram even under v2) and expected to self-resolve under Docling re-ingest.

**Beyond this phase.** The Memex pipeline is now end-to-end shippable, observable, eval-instrumented, and refusal-calibrated. Remaining work falls into five tiers, ranked by pickability and impact. **The single most-impactful pending work item is P0 corpus extension** — every P2.x quality A/B verdict (reranker, orchestrator, VLM) is gated on it, and even one category bumped from 15 → 30+ queries unlocks statistically-meaningful comparisons.

### Tier 1 — pickable now (no corpus / infra dependency)

These items have everything they need to be picked up immediately and shipped in one session. P2.4 (the previous tier-1 pickup) is shipped; full ship narrative is in the status table above. Open items:

- ~~**xgrammar-interaction investigation**~~ — ✅ **Diagnosed + fixed 2026-05-21.** Root cause was empty-draft renders to verify under unbounded `list[int]` schema; greedy decoding ran away. Fix: empty-draft short-circuit in `agents/answering.py::verify`. Insight for P3.3: any structured output with unbounded array fields (Pix2Struct's tables, ChartQA's row sequences) needs an EMPTY-INPUT short-circuit OR a `max_length` constraint on the array; xgrammar will happily emit forever otherwise.
- **Filler N1–N8.** Audit-surfaced minor hardening — see the §Filler list at the end of this section. Each is small, atomic, and pickable in <1 hour. Useful when the user wants something contained while waiting on a bigger pickup.

### Tier 2 — multi-session foundational (unblocks downstream verdicts)

These need session-spanning investment but their completion unlocks several downstream items in tier 3.

- **P0 corpus extension** (curator-time, multi-session). The single most-impactful pending work item; gates P2.1, P2.2, P2.3, and the P1.6 chunker tuning verdict.
  1. Extend `slide-decks` to 30–50 queries across 3–5 documents.
  2. Bootstrap each of the other 6 categories from `docs/eval-corpus-plan.md`: modern-printed, scientific-papers, technical-docs, historical-scans, handwritten, forms.
  3. Wire CER/WER/structural-F1 from `src/memex/eval/scoring.py` into `runner.py`; needs per-doc hand-curated reference markdown (Phase 2 of the spec).
- **P3.3 — Chart-data extraction (chart-OCR over Docling figures).** Surfaced by the parser investigation: query 04 ("transistor density 2004–2022 per TSMC chart") stayed refused under both PyMuPDF and Docling, because the year-by-year numerics live inside chart imagery that Docling represents as `<!-- image -->` placeholders. A DocVQA-style chart-OCR pass over those figures (input: Docling-tagged figure crops; output: a structured GFM table or key:value list appended to the parent chunk's markdown) would extract them. Candidate models: ChartQA-style finetunes, DePlot, Pix2Struct. New parse-stage post-processor (sandbox-able alongside the existing Docling worker); new VRAM footprint to budget. Multi-session: model selection + load + integration + eval pass.

### Tier 3 — eval-gated stack swaps (unlocked by Tier 2 P0 extension)

These are ready to run once the corpus has the depth to discriminate between candidates.

- **P2.1 — Qwen3-Reranker-0.6B quality A/B.** ◐ Infra shipped (commits `714dd32`, `b485748`); the swap is a quality play, not a memory play (~2.1 GB live, ~equal to bge). Run the same 15-query (eventually larger) set with `MEMEX_MODELS__RERANKER_BACKEND=qwen3` vs the default `cross_encoder`; pick the winner per category.
- **P2.2 — Granite 4.1-8B-Instruct vs `Qwen/Qwen3-8B-AWQ`.** Apache-2.0 license + native OpenAI-tool-calling. Model-name change in `scripts/serve-vllm.sh`. Eval-gated.
- **P2.3 — Qwen3-VL-8B-Instruct vs `Qwen2.5-VL-7B`.** +6–14 OCR points published; native 256 K context. Same AWQ-Int4 footprint. Currently disabled by default on 12 GB; the swap helps the larger-VRAM tier. Eval-gated.
- **P1.6 chunker-size verdict.** Is `chunk_target_tokens=400` the right default, or should it be 500–600 with a longer model context? Needs corpus depth to be answerable.

### Tier 4 — cost / ops decision (no code blocker; resource blocker)

- **P3.1 — Real-mode benchmark nightly CI.** `scripts/benchmark.py --real` measures cold-start + first-token + embedding throughput. Needs a GPU runner: cloud (Lambda, RunPod, Modal) or a dedicated home rig. Workflow template already in `.github/workflows/`. Don't pick up without confirming the user has the resources allocated.

### Tier 5 — design decisions still owed (punt until a user needs it)

- **P4.1 — Wikilink section anchors** (per ADR-0003). `[[doc_id]]` is committed; `[[doc_id#heading]]` is unresolved. Punt until a real cross-doc citation needs it.
- **P4.2 — 8 GB GPU tier.** ADR-0001 says "no first-class CPU fallback." Should there be a documented 8 GB profile (Qwen3-4B-AWQ + reranker-base + no VLM)? Decision can wait. Note: a smaller orchestrator would also unblock P2.1's quality test on tighter rigs.
- **P4.3 — Trace retention.** `EventBus` has 30-day prune; Langfuse self-host wiring is open. Match retention windows when self-host lands.

### Recently shipped this phase (foundation cleanup)

- ✅ **P0 corpus extension to 30 queries** — n=17 answerable queries gives Tier-3 A/Bs statistical signal.
- ✅ **xgrammar empty-draft short-circuit** — diagnosed + fixed; verify node bypasses model call when draft is empty; insight saved for P3.3 (structured outputs with unbounded array fields need explicit empty-input handling).
- ✅ **Retrieval truncate-budget retune** — root cause was content-visibility; bumped truncate 700→1800 in prompts, top_k 10→5, max_tokens 1024→640, vLLM max-model-len 4096→6144. Three legitimate REF→ANS flips (Q2, Q7-recovered, Q20); zero new hallucinations.

### Next pickup — ranked by impact × feasibility

1. ~~**🪜 P2.1 Qwen3-Reranker quality A/B**~~ — ✅ **Resolved 2026-05-21**. Initially hardware-blocked on the 12 GB rig at production budget; unblocked by P4.2 Sessions 3-4 (Qwen3-4B-AWQ + gpu_fraction=0.50). **Verdict: cross_encoder (bge-reranker-v2-m3) wins clearly** — median ANS=4 vs qwen3's 0 across 3 head-to-head runs. Qwen3-Reranker promotes thematically-general chunks over the literal-answer chunk; for slide-decks-with-chart-text, cross_encoder's fact-extraction-relevance ranking is markedly better-suited. Default stays `cross_encoder`; qwen3 backend remains opt-in.
2. **🏗️ P3.3 chart-OCR pass over Docling figures** (Tier 2, multi-session). **Now the highest-priority pickup.** Addresses the Q4/Q16/Q21 residuals (chart-numerics buried in image format). New parse-stage post-processor + DocVQA-class model (Pix2Struct, DePlot, ChartQA). The xgrammar empty-input short-circuit pattern transfers here for safe structured outputs.
3. **🎯 P0 corpus extension — multi-doc / multi-category** (Tier 2, multi-session). Within-category variance (3-5 slide decks) AND breadth across the other 6 categories.
4. **🧰 Filler N1/N5** — Tier 1 atomic nits, <1 hour each. Useful when the user wants something contained.
5. **🪜 P2.2** (Granite vs Qwen3 orchestrator) and **P2.3** (Qwen3-VL vs Qwen2.5-VL) — Tier 3, single-session each. Same VRAM-ceiling concern as P2.1 — check footprints before queuing.

Items below (P3.1 benchmark CI / P4.x design decisions, **including P4.2 which directly unblocks P2.1**) stay queued; pickup needs an external trigger or a deliberate scope shift (e.g., 8 GB / smaller-orchestrator tier).

---

## P0 — eval corpus: extended baseline shipped 2026-05-21

**30-query extended baseline** (top_k=6, batch=4, against the live PyMuPDF vault):

| Metric | 15-query baseline | 30-query extended | Δ |
|---|---|---|---|
| `answered_count` | 3 | 9 | +6 (more answerable queries fired) |
| `refused_count` | 12 | 21 | +9 |
| `mcp_answered_only` | 0.67 | **0.89** | **+22 pp** |
| `refusal_rate_on_counterfactuals` | 1.00 | **1.00** | unchanged (perfect) |
| Hallucinations | 0 | 0 | unchanged |

**The 15 new queries** (qids 16–30): 10 answerable covering NVRTC compilation, kernel fusion, NVLink C2C, CUTLASS, mixed-precision LU, unified memory, FP16 tensor core MMA, performance-per-watt thesis + 3 near-miss refusals (FP4 mantissa, CUDA 11.0 compile time, NVLink latency) + 2 empty-retrieval refusals (H100 TDP, kernel-language survey).

**~~New finding to capture~~** — ✅ **Diagnosed + fixed 2026-05-21.** The "retrieval-vs-agent gap" hypothesis was wrong; the chunk WAS reaching the agent at rerank 0.98 top-1. Root cause: the answer/verify prompts use `{{ c.text | truncate(700) }}` in Jinja, and the median chunk is **2172 chars** — the agent has been operating on **the first ~32% of most chunks**. Query 16's NVRTC content sits at chars 1524-1700, well past the cliff.

**Fix**: a coordinated retune of the prompt/budget pair —
- Bump chunk truncate in all prompts: `answer/v{1,2}.md` and `verify_grounding/v{1,2}.md` go from `truncate(700)` → `truncate(1800)`; `assess_sufficiency/v1.md` from `truncate(500)` → `truncate(1200)`.
- Lower default agent `MEMEX_RERANK_TOP_K` from 10 → 5 to keep context-budget healthy when chunks are bigger.
- Lower default `complete_structured(max_tokens=...)` from 1024 → 640 (most outputs are <500 tokens; reserved budget was wasteful).
- Bump vLLM `--max-model-len` from 4096 → 6144 in `scripts/serve-vllm.sh` (costs ~1 GB KV-cache reservation under fp8_e5m2; comfortable on the 12 GB rig at gpu_memory_utilization=0.72). The 4096 ceiling was insufficient with the new prompt budgets at the lowest-feasible top_k.

**Result on the 30-query slide-decks corpus**:

| Metric | Before truncate fix | After fix | Δ |
|---|---|---|---|
| `answered_count` | 9 | **12** | +3 legitimate answers |
| `mcp_answered_only` | 0.89 | 0.75 | -14 pp (more queries answered with imperfect citations; not a precision drop on the same set) |
| `refusal_rate_cf` | 1.00 | 1.00 | unchanged (zero new hallucinations) |
| Hallucinations | 0 | 0 | unchanged |

**Three legitimate REF→ANS flips**:
- **Q2** (two power categories) — answer was past char 700 in chunk #3a6c6789e8.
- **Q7** (5-bit/10-bit half precision-bit diagram) — **the P2.4 Q7 regression is recovered as a bonus**; the precision-bit diagram is now visible to the agent.
- **Q20** (NVRTC '-minimal' flag) — the flag definition at char 1112+ is now visible.

**Residual refusals** (Q16, Q21 NVRTC chart numerics + Q4 transistor density + Q5 data-movement synthesis + Q25 performance-per-watt thesis) are now genuine answer-prompt / chart-OCR limits, not retrieval limits. P3.3 (chart-OCR over Docling figures) addresses Q4/Q16/Q21; Q5/Q25 are synthesis-required.

### Eval baseline + parser-investigation history (preserved for reference)

The eval harness now has its first concrete query set at **`tests/eval-data/slide-decks/queries.json`** — 15 queries (7 answerable + 3 empty-retrieval counterfactuals + 5 near-miss counterfactuals) against the GTC 2024 CUDA deck. The bootstrap (10 queries; published 2026-05-21 in commit `720fdfa`) was superseded the same day after the calibration was audited and the methodology hardened: broader chunk-set labelling, near-miss counterfactuals added, and `mean_citation_precision_answered_only` added to `EvalReport` so refused queries don't inflate the headline.

**Rigorous baseline sweep — `MEMEX_RERANK_TOP_K ∈ {4, 6, 8}`:**

| top_k | answered | refused | `mcp_all` | `mcp_ans` | `refusal_rate_cf` |
|---|---|---|---|---|---|
| 4 | 4 | 11 | 0.87 | **0.50** | 0.875 |
| 6 | 6 | 9 | 0.80 | **0.50** | 0.75 |
| 8 | 5 | 10 | 0.87 | **0.60** | 0.75 |

The bootstrap's headline `mean_citation_precision = 0.9` was the all-queries metric — refused queries score 1.0 (no false-positive citations) and inflated the number. The answered-only metric (now reported alongside) tells the honest story: **roughly half of citations on attempted answers are correct**. Refusal rate on counterfactuals dropped from a naïve `1.0` (3 empty-retrieval queries only) to `0.75–0.875` once near-miss queries entered the mix — the agent hallucinated on questions like "FP128 energy cost" and "FP4 tensor core cost," substituting nearby table values as if they answered the actual question.

**Findings carrying into P2.x:**

1. **Over-refusal is invariant to top_k.** Queries 02 ("two categories of power"), 04 ("transistor density 2004-2022"), 05 ("data movement power") refused at every `top_k ∈ {4, 6, 8}`. Since the same chunks are retrieved across top_k, the cause isn't context-budget — it's per-chunk quality or agent-prompt calibration. The chart-text-noise hypothesis is now the leading theory (an ablation run with `[chart-text]` blocks stripped from the chunk text would confirm; deferred to a future session).
2. **Near-miss hallucinations are real and unmeasured before.** Queries 11 (FP128) and 12 (FP4) got hallucinated answers at `top_k ∈ {6, 8}`; query 12 hallucinated at every top_k. The retrieval correctly routed to the precision-table chunks (high rerank), but the agent substituted nearby values from those chunks as if they answered the unrelated FP128/FP4 question. **The bootstrap's `refusal_rate_on_counterfactuals = 1.0` was a measurement artefact of only using easy empty-retrieval counterfactuals.**
3. **`max_model_len=4096` context pressure is real but narrower than first thought.** The bootstrap overflowed at default `top_k=10`; the rigorous sweep at `top_k ∈ {4, 6, 8}` runs cleanly. The operational floor for slide-deck content is probably around `top_k=8`. Per-category defaults still warrant consideration; bumping `max_model_len` to 8192 would unlock `top_k=10` again at ~2 GB more KV-cache VRAM.

**What's verified now that wasn't in the bootstrap:**

- The `relevant_chunk_ids` labelling rule is documented in `tests/eval-data/README.md` (top-10 search, include every chunk that contains the literal answer).
- `EvalReport.mean_citation_precision_answered_only` is the published honest metric.
- Counterfactual coverage includes near-miss queries with the `_counterfactual_mode` tag distinguishing the two refusal modes.

**Chart-text ablation — null result (2026-05-21):**

Hypothesis tested: stripping the 67 `[chart-text]` blocks from the canonical CUDA-deck markdown (40% word reduction; from 7818 → 4668 words) would reduce over-refusal on the previously-refused answerable queries (Q2, Q4, Q5). Method: copied the vault to `/tmp/memex-ablation-vault`, sed-stripped chart-text, re-indexed, re-ran the eval at the same `top_k=6`. Result: **zero queries flipped REF→ANS.**

Per-query comparison at `top_k=6`:

| qid | rigorous | ablation | reading |
|---|---|---|---|
| Q1 (FP16 cost) | ANS | ANS | precision-table answer preserved in both |
| Q2 (two power categories) | REF | REF | answer was *only* in chart-text → refusal is correct in ablation |
| Q3 (mantissa scaling) | ANS | ANS | answer preserved (in table prose, not chart-text) |
| Q4 (transistor density 2004–2022) | REF | REF | quantitative 2004→2022 data was in chart-text; only the qualitative claim survives |
| Q5 (data movement power) | REF | REF | "data movement" mentioned only in chart-text — refusal correct in ablation |
| Q6 (FP8 tensor core cost) | ANS | ANS | precision-table answer preserved |
| Q7 (5-bit/10-bit half) | ANS | REF | precision-bit diagram was in chart-text → flipped correctly |
| Q11 (FP128 near-miss) | ANS (hallucinated) | ANS (still hallucinated) | independent of chart-text |
| Q12 (FP4 near-miss) | ANS (hallucinated) | ANS (still hallucinated) | independent of chart-text |

The headline `refusal_rate_on_counterfactuals=0.75` was unchanged; `answered_count` dropped 6 → 5 because Q7 correctly flipped to refusal once its source diagram disappeared.

**What this reframes:**

1. **Over-refusal is NOT caused by `[chart-text]` noise.** Q2, Q4, Q5 stayed refused after the noise was removed. The hypothesis (chart-text interleaves with answer prose and confuses the agent) is falsified by this ablation.
2. **The chart-text blocks were the *location* of real numerical answers** for Q2 (data movement/computation), Q5 (data-movement-as-power-cost), Q7 (precision-bit diagrams), Q4 (year-by-year transistor data). The "over-refusal" in the rigorous baseline turns out to have been the agent *correctly* judging that the answers were in too-noisy chart-text format to extract reliably — partly-correct behaviour misread as miscalibration.
3. **The real failure mode is at the PARSER stage**, not the agent prompt: PyMuPDF chart extraction converts slide-chart imagery into chart-text blocks that contain the data values but in a format the agent considers unreliable. Three P2.x angles emerge:
   - **Parser swap** for slide decks: route slide decks to Docling (or a chart-OCR specialist) instead of PyMuPDF, accepting the wall-time hit for better structural fidelity.
   - **Post-parse cleanup**: a structure-aware chart-text→Markdown-table conversion step before chunking.
   - **Dual-storage**: keep chart-text in the index for retrieval (so chunks DO surface for related queries) but feed the agent only the surrounding prose for grounding.
4. **Hallucinations on Q11/Q12 (FP128, FP4) are independent of chart-text noise** — same answers in both runs. This is an agent-prompt issue: the precision-table grounding is solid, but the agent confabulates by interpolating absent rows.
5. **Citation precision is not directly comparable across vaults** (`mcp_ans=0.0` in ablation because chunk_ids change when the chunker re-runs on different content). The eval needs a content-anchor identifier (or per-vault baselines) for cross-vault comparisons.

**Outstanding P0 work (multi-session, curator-time):**

1. Extend `slide-decks` to 30–50 queries across 3–5 documents.
2. Bootstrap the other 6 categories (modern-printed, scientific-papers, technical-docs, historical-scans, handwritten, forms).
3. Wire CER/WER/structural-F1 from `src/memex/eval/scoring.py` into `runner.py`; needs per-doc hand-curated reference markdown.
4. ~~Run the chart-text ablation~~ — ✅ Done 2026-05-21 (null result; see above).
5. ~~investigate slide-deck parser quality~~ — ✅ **Done 2026-05-21** (see below).

P0 is now scaffolded with a baseline that can be trusted as a P2.x reference, and the parser investigation that the chart-text ablation surfaced has shipped its verdict.

### Parser investigation (2026-05-21): Docling beats PyMuPDF on slide decks

**Method.** Re-ingested the CUDA deck into a fresh vault (`/tmp/memex-docling-vault`) with `MEMEX_PARSE__PYMUPDF_ENABLED=false` forcing Docling. Ran the same 15-query eval at `top_k=6`. Three-way comparison vs the rigorous PyMuPDF baseline and the chart-text-stripped ablation.

**Structural differences in the markdown output:**

| Property | PyMuPDF | Docling |
|---|---|---|
| Word count | 7818 | 5797 (-26%) |
| `[chart-text]` blocks | 67 | 0 |
| GFM table rows | 68 | 98 (+44%) |
| Tables detected (manifest) | 0 | 13 |
| Figures detected | 0 | 245 |
| Q2 answer rendering | `1. Data movement 80x` in `[chart-text]` block | `## We spend power on two things\n1. Data movement\n2. Computation` (clean heading + list) |
| Q7 precision-bit diagram | chart-text dump | GFM table with sign / exponent / mantissa columns |
| Parse duration | 33 s | 93 s (~2.8× slower) |

**Eval result at top_k=6** (15 queries; 7 answerable + 8 counterfactual):

| Vault | answered | of which **legitimate** | of which **hallucinated** | refusal_rate_cf |
|---|---|---|---|---|
| PyMuPDF (rigorous) | 6 | 4 (Q1, Q3, Q6, Q7) | 2 (Q11, Q12) | 0.75 |
| PyMuPDF stripped (ablation) | 5 | 3 (Q1, Q3, Q6) | 2 (Q11, Q12) | 0.75 |
| **Docling** | **8** | **6 (Q1, Q2, Q3, Q5, Q6, Q7)** | 2 (Q11, Q12) | 0.75 |

**Two answerable queries flipped REF→ANS under Docling:**
- **Q2** ("two main categories where modern processors spend power"): Docling renders "1. Data movement / 2. Computation" as a clean numbered list under a proper H2 heading; the agent grounds on it. PyMuPDF buried it in a chart-text block with chart axis values (`80x`, `60x`).
- **Q5** ("data movement key concern for power efficiency"): same source content, same fix.

**Q4 stayed refused even under Docling** — the year-by-year transistor-density numbers were embedded in chart imagery and Docling represents charts as figures (`<!-- image -->` markers in the markdown). A chart-OCR pass (e.g., DocVQA-style model) would be needed to extract those values; out of scope for either parser. This is a chart-data-extraction ceiling that's independent of the PyMuPDF/Docling choice.

**Q11/Q12 hallucinations are independent of parser** — both runs hallucinated identically. Agent-prompt issue, separately diagnosable.

### Recommendation: angle 1 (parser swap, category-aware) — ✅ Shipped 2026-05-21

Routes slide-deck-shaped documents to Docling instead of PyMuPDF via a new Tier 0.5 in the classifier (`src/memex/parse/pipeline.py`). Non-slide-deck born-digital PDFs (whitepapers, books, technical docs) keep PyMuPDF — their parse profile is fine and PyMuPDF is 2.8× faster.

**The heuristic**: `_is_slide_deck(s)` fires when `avg_aspect_ratio >= 1.3` (slides are 4:3 ≈ 1.33 or 16:9 ≈ 1.78; documents are 0.77 portrait) **AND** `50 ≤ chars_per_page_avg < 800` (slides typically 200–700 chars per page; documents 2000+; below 50 falls through to Tier 1.C as rasterised). Two new `ParseSettings` knobs: `pymupdf_slide_deck_aspect_threshold` (default 1.3) and `pymupdf_slide_deck_max_chars_per_page` (default 800).

**Precedence**:
- Skips when `scan_hits` is detected (Tier 1.B's scan-producer routing is the right destination regardless of shape).
- When the doc is also mixed-content (chart imagery heavy enough to want OCR for figure-embedded labels), inherits `needs_ocr=True` and uses `doc_type="slide-deck-mixed"`.
- Otherwise returns `doc_type="slide-deck"` at confidence 0.10 — below the `pymupdf_min_confidence` default of 0.5 — so the existing fallthrough path routes to Docling.

**Live-verified on the CUDA deck**: aspect 1.778, 432 chars/page → routes to slide-deck at Tier 0.5 (confirmed by direct PyMuPDF probe + `_classify()` call). Three new unit tests in `tests/unit/test_pymupdf_classifier.py` + two integration tests updated (the existing `patch_pymupdf_born_digital` fixture was changed to portrait + text-dense signals so it still exercises Tier 1.A; slide-shaped PowerPoint is covered by the new Tier 0.5 tests). 147/147 tests green.

**Out of scope for this ship**: re-ingesting pre-existing slide-deck content. The classifier is forward-acting; the user's existing PyMuPDF-parsed CUDA deck stays as-is until manually re-ingested via `memex ingest`. A clean reindex of existing slide-deck content will be needed for the new routing to take effect on already-ingested docs.

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
- ~~**N9**~~ — ✅ **Fixed 2026-05-21**: lifted the `GraphNeighbor` import out of `TYPE_CHECKING` in `mcp/server.py` and removed the forward-ref quotes from `get_graph_neighbors`'s return annotation. MCP startup is now warning-free.

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
- **FU3.2.3 — Vault backup timer** (2026-05-21): `scripts/memex-vault-backup.sh` (restic-driven, idempotent, init-on-first-run, retention prune) + `memex-vault-backup.{service,timer,env}` user-systemd templates + standalone `docs/deploy/backup.md` (quickstart, ops table, cloud-target walkthrough for B2/S3/SSH, hot-restorable restore flow). Default schedule `OnCalendar=*-*-* 02:00:00` with `Persistent=true`, default retention `--keep-daily 7 --keep-weekly 4 --keep-monthly 6`. Whole-vault snapshot (including derived state, since restic deduplicates) minus the transient bits (`locks/`, `daemon/`, `events.sqlite`). Live-verified: backup against a tmp repo created a snapshot, timer scheduled correctly, the service ran clean via `systemctl --user start memex-vault-backup.service`, restore smoke test pulled `documents/` back into a tmp dir without touching the live vault.
- **FU3.2.2 — `memex upgrade` CLI** (2026-05-21): new top-level command bundles the four-step upgrade recipe (`git pull --ff-only` → `uv sync --extra models --extra parse --extra serve` → `systemctl --user restart memex-*.service`). Three flags: `--dry-run` (preview), `--no-restart` (Pattern B/C boxes), `--skip-sync` (git-pull + restart only). Detects installed units via `systemctl --user list-unit-files memex-*.service` so partial installs (e.g., only vLLM, no web/MCP) get partial restarts. Refuses dirty trees via `git pull --ff-only`'s own behaviour. ~150 lines of CLI orchestration (no new deps); the manual recipe in `docs/deploy/systemd.md` is now an `Operations` table row. README gains an "Update to a newer Memex" subsection.
- **FU3.2.1 — `Type=notify` readiness gate for vLLM** (2026-05-21): `scripts/serve-vllm.sh` gained a backgrounded sidecar that polls `/v1/models` and calls `systemd-notify --ready --status="…"` on the first 2xx; `memex-vllm.service` switched from `Type=simple` to `Type=notify` + `NotifyAccess=all`. `systemctl --user start memex-vllm` now blocks until vLLM is genuinely serving (~31 s measured), so `After=memex-vllm.service` on downstream units (web, MCP, watcher) is a real readiness gate. Live-verified: full-stack cold boot returned after 32 s with **zero Connection-refused logs** in any downstream unit's journal — and a watcher reaction triggered immediately afterward fired clean `extract_entities@v1` + `extract_citations@v1` model calls against vLLM. The sidecar no-ops when `$NOTIFY_SOCKET` is unset, so Pattern B (manual) and Pattern C (`memex daemon start`) are unaffected.
- **P3.2 — Daemon templates** (2026-05-21): three waves. (1) vLLM unit + plist + deploy guides. (2) `memex-web.service` + `memex-mcp.service` (+ matching plists/env) so the full stack boots together. (3) `memex-watch.service` (+ plist + env) for the vault file-watcher that re-enriches + re-indexes on canonical-markdown edits. Web + MCP + watcher all carry soft `Wants=memex-vllm.service` — they cluster but degrade gracefully if vLLM is down (search/get_document/list_documents work LLM-less; only `/ask` and enrich need vLLM). The wave-3 live verification was the most thorough: appended a line to the canonical CUDA-deck markdown → watcher logged `watcher.edit_confirmed` → 32 chunks went to enrich → vLLM was offline (cleanup state) so 32× `enrich.chunk_failed` → `enrich.done chunk_failures: 32` → watcher kept running → `index.done added: 1 deleted: 1 unchanged: 30 partial: true` (partial re-index isolated the changed chunk). Restoring the file fired another correct reaction. Two unit-template bugs caught during verification (StartLimitIntervalSec in `[Service]` instead of `[Unit]`, inline `#` comments on `ProtectHome=` line) — both fixed across all four units. No code change.

For the full per-commit log: `git log --oneline` (69 commits at session close, 2026-05-21; the up-to-date number is `git rev-list --count main`).

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
