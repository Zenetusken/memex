# Memex Roadmap

**Last updated:** 2026-05-24 (**frictionless retitle + full quality cleanup + parse-quality eval wiring + 3 synthetic eval-corpus fixtures + heading-level-recovery parser fix**) — **193 commits on `main`**, **393/393 tests green**, **pyright `--strict` 0 errors / 0 warnings**, **ruff check + format clean**, 100% public-surface docstring coverage. Wired the parse-fidelity eval (CER/WER/structural-F1) into the runner — `memex eval-parse <corpus_dir>` scores parsed vault markdown vs hand-curated `ground-truth.md` — and bootstrapped `eval-corpus/` with **3 synthetic (independence-respecting) fixtures across 3 categories** (modern-printed, forms, technical-docs), rendered via LibreOffice. These surfaced a real pymupdf4llm defect — **it collapsed every heading to `##`** — now **FIXED** (commit `bb66c82`, `parse/pymupdf_worker.py` re-derives heading levels from font size): full-corpus parse baseline **3/3 pass, mean CER 0.036 / WER 0.106 / structural-F1 0.566 → 0.974**. The fix changes only heading-level prefixes (chunk boundaries + body content byte-identical; HARD GATES are anchor-independent), so it's HARD-GATE-safe without a vault re-parse. (P0 sub-goal 3 plumbing done; representative cross-category numbers still need human-curated real PDFs.) This session: shipped `memex retitle` (metadata-only rename — frontmatter + FTS + vector + graph, no re-embed; CLI + webui inline `<h1>` edit) as the GPU-free replacement for a force-reindex that OOM'd; cleared **ruff 139 → 0** (genuine fixes for an exception-swallowing `finally`, a GC-able fire-and-forget task, sha1 intent, try/except→`contextlib.suppress`; documented justifications for the rest); completed the **pyright-strict typing pass `src/memex` 507 → 0 (whole repo 0/0)** via a new `stubs/` dir (langgraph, lancedb, pypdfium2, pyseccomp) + call-site casts for partial-`py.typed` libs, done by parallel agents partitioned by lib boundary; the typing pass surfaced + fixed a **latent runtime bug** (`CrossEncoder(automodel_args=)` → `model_kwargs` — would `TypeError` when the reranker loads, faked in tests so uncaught). Full eval suite re-baselined with the reranker now in **bf16** (ADR-0006 intent, restored by the bug fix): HARD GATES hold on all 5 corpora (refusal_cf=1.0, 0 hallucinations), ANS counts unchanged vs prior baselines (slide-decks 15/33, annual-report 7/10, chart-types 5/9, french-course 5/8, cr350-multidoc 11/15); minor citation-precision shifts (cr350 1.0→0.933, slide-decks 0.91→0.879) are equally-valid-sibling-chunk granularity, not quality loss. Prior context (2026-05-23): P3.3 v7 + P4.1 (read/write/webui) + **P0 multi-doc corpus 4→10 docs + multi-doc DISAMBIGUATION eval + answer-node chunk-id repair + course-ref wikilink resolver**. **Vault is 10 docs (7 related French CR350 lectures + CUDA deck + 10-K + Tableau guide)** — the multi-doc depth P0 has needed. **Multi-doc disambiguation eval** (`tests/eval-data/cr350-multidoc/`, 15 cross-lecture queries): **baseline 11/11 ANS, refusal_cf=1.0, 0 hallucinations, citation precision 1.0** — perfect document disambiguation across the 7 lectures, including two deliberate confusables. The eval surfaced + fixed the answer LLM corrupting cited `docid#hash` ids (`agents.answering._repair_claim_chunk_ids`; cp 0.62→1.0). **Wikilink finding RESOLVED**: the academic citation resolver emits 0 links on lecture slides, so `enrich/course_refs.py` now turns "Cours N"/"Semaine N" mentions into sibling-lecture `[[doc#section]]` wikilinks (by ordinal, course-code gated, cross-label aware) — enriching the CR350 syllabus went 0→6 wikilinks. **7 chart-content REF→ANS flips** measured across 3 corpora (chart-types-08, annual-report-09, annual-report-10, slide-decks-18, slide-decks-31, slide-decks-32, slide-decks-33) PLUS one sloppy-ANS→correct-ANS recovery (annual-report-03 Data Center revenue $193.737B); first time chart-OCR delivers measurable answering on chart-content questions. Slide-decks corpus: 11/30 → 15/33 ANS (+4 wins this arc), annual-report: 6/10 → 7/10 (Q03 recovered post-chunker-reflow), chart-types: 2/9 → 5/9 (+3 wins), HARD GATE refusal_cf=1.0 preserved across all 3 corpora. Foundation + audit + French support + Qwen3 prompt-engineering + **P3.3 chart-OCR investigation now fully closed** (backend chosen + dark-matter retrieval/recognition bug fixed). Eval suites stable on Qwen3-8B-AWQ: English (30q CUDA deck) ANS=11/30, French (8q CR350) ANS=5/5, both at `refusal_cf` 1.000, hallucinations 0. **MIRACL-fr retrieval benchmark**: nDCG@10 = 0.807. **Chart-content questions now answerable** for the first time after the v7 root-cause fix (commit a9e8326): chart-types Q08 "On Time 22 / Late 8" refused→ANS, annual-report Q09/Q10 CC timeline refused→ANS. **New operator surface**: `memex parse --force-docling <doc-id>` (commit a4b8493) bypasses the PyMuPDF classifier when chart-OCR validation is needed on born-digital text-heavy docs. **External-blocked items**: P0 multi-doc (more source material), P2.2 Granite (vLLM hybrid-arch hang), P2.3 VLM swap (needs scan corpus), P3.3-a autoawq compat, P3.1 benchmark CI (GPU runner), P4.1/P4.3/P4.4 (various external triggers).

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

**File count:** 90+ Python files in `src/memex/` + `tests/` + `scripts/`, all parse-clean + `pyright --strict` 0/0. Minimal `.pyi` stubs for untyped deps live under `stubs/` (langgraph, lancedb, pypdfium2, pyseccomp). 7 ADRs. 11 audit reports under `docs/audits/`. **181 commits on `main`** (public at `github.com/Zenetusken/memex`); the canonical count is `git rev-list --count main`.

**Test suite:** 378/378 green on the reference rig. Linux + pyseccomp; 5 tests skip on Windows. `pyright --strict` clean (0 errors / 0 warnings); `ruff check` + `ruff format` clean.

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

**Foundation arc complete (2026-05-22).** Phases 0–4 + P1.x backlog + P2.1/P2.4/P3.2/P4.2 verdicts + chart-OCR (v1–v6, opt-in) + Filler N1–N9 + defensive schema hardening + eval label refresh all shipped. Per-item ship summaries live in the §"Recently shipped this phase" subsection further down + the §"Status at a glance" table above.

**Beyond this phase.** The pipeline is end-to-end shippable, observable, eval-instrumented, and refusal-calibrated. Remaining work falls into five tiers, ranked by pickability and impact. **The single most-impactful pending work item is P0 multi-doc/multi-category extension** — every cross-corpus claim depends on it, and current verdicts are all sourced from one document (CUDA deck) in one category (slide-decks).

### Tier 1 — pickable now (no corpus / infra dependency)

All earlier Tier-1 items shipped: xgrammar empty-draft short-circuit, Filler N1–N9 audit-nit backlog, defensive schema hardening, eval label refresh.

- ~~**heading-level preservation in the PyMuPDF path**~~ ✅ **SHIPPED 2026-05-24** (commit `bb66c82`). pymupdf4llm 1.27.x correctly *detects* heading lines but its emitter collapsed **every** one to `## ` regardless of source font size (its `IdentifyHeaders` map is right; `to_markdown` ignores per-span sizes). Fix is a post-process in `parse/pymupdf_worker.py` (`_heading_size_to_level` builds a doc-wide size→level map — body = most-frequent span size, larger distinct sizes rank descending into levels 1..6 — and `_remap_heading_levels` rewrites each already-detected `#`-prefixed line's prefix from its real font size; skips fenced code, leaves unmatched/no-tier headings as emitted). **Never adds/removes/moves a heading or a chunk** — pinned by `tests/unit/test_heading_remap.py` (6). Lifted mean structural-F1 across the 3 eval-corpus fixtures **0.566 → 0.974** (forms 0.75→1.0, technical-docs 0.33→1.0, tidewater 0.615→0.923). **HARD-GATE safety proven without a vault re-parse**: the answering eval's HARD GATES (`refusal_cf`, hallucinations) are anchor-independent (runner reads only `should_refuse` + grounding, never `relevant_chunk_ids`); the only anchor-coupled metric is soft `citation_precision`, re-resolvable via `_anchor_phrase`. Isolated chunk-diff on a CR350 lecture (remap on vs off, same parse): **chunk boundaries + body content byte-identical** apart from the heading `#`-count — 25/50 chunk_ids shift purely from the prefix, embeddings differ by ~1 char per heading line ⇒ retrieval ranking essentially unchanged. **Known limitation**: on slide decks (erratic per-slide fonts) the size heuristic can mis-level a title slide (e.g. cours-1 title → H4) — still strictly better than flat all-`##` (recovers real hierarchy, loses no content), but messier than born-digital prose. **Vault migration done 2026-05-24** (full re-process parse→index→enrich): the migration surfaced that **9 of the 10 vault docs are Docling-parsed** — only the syllabus (`050df0d9`, born-digital Acrobat-Distiller PDF, Tier-1.A → PyMuPDF) is in the PyMuPDF path, so it's the **only** doc the fix reaches. Migrated it cleanly: title → H1 + 19 sections H2 (was flat all-`##`); index diff = 1 chunk changed / 21 unchanged (only the title chunk's id moved, referenced by no eval anchor); enrich restored its 6 course-ref wikilinks. Post-migration `cr350-multidoc` eval: **11/11 ANS, refusal_cf=1.0, 0 hallucinations**; the syllabus-anchored query `cr350-xref-08` answers at cp 1.0. The other 9 docs were left untouched (re-parsing them would not apply this PyMuPDF-only fix and could regress the chart-OCR docs — see the Docling item below).

- ~~**Docling heading-level collapse**~~ ✅ **SHIPPED + migrated 2026-05-24** (commit `5bd032c`). Root cause: docling_core's markdown serializer emits `num_hashes = 1 if TitleItem else item.level + 1`, and Docling's PDF reading-order model calls `add_heading()` **without a level** → every `SectionHeaderItem` defaults to `level=1` → all headings export as `## `. `SectionHeaderItem.level` is a writable field the serializer respects, so `docling_worker._recover_heading_levels` re-derives it **before export** from a font-size proxy (provenance `bbox.height` — Docling exposes no font size): ranks the distinct header heights **among themselves** (largest → level 1, bucketed to 0.5 pt, capped at 5 → `######`) and writes the level back in place. Ranking headers among themselves (not body-relative) means uniform slide-title heights stay flat — peers remain peers. Operates on structured items (no chart-block/fence defense needed; `[chart-extracted]` is stitched in later as text). 9 unit tests (`tests/unit/test_docling_heading_levels.py`, duck-typed → run without docling). **Empirically-gated migration** — Phase-A re-parse validation produced a clean split: **well-structured docs win, slide decks don't.** Migrated the 2 beneficiaries (`--force-docling`, chart blocks preserved): **Tableau guide** (textbook hierarchy — title→H2, chart-types→H3, tips→H4) and the **10-K** (real headings surfaced at H2–H5; the 491 mis-*detected* body-sentence "headings" — Docling's detection noise, a separate issue — sink to H6, so real sections now persist as `heading_path` ancestors instead of being overwritten by subsequent artifact-H2s). **Skipped** the CUDA deck + 6 CR350 lectures (slide decks: erratic per-element bbox heights → scrambled levels; the fix would add noise). Post-migration evals: annual-report **7/10 ANS, refusal_cf=1.0**; chart-types **5/9 ANS, refusal_cf=1.0**; P3.3-v7 chart-content answers preserved (09/10 + 08/09 still ANS, `[chart-extracted]` blocks intact: 10-K 2, Tableau 6). Anchors re-resolved (5/7 per corpus; the chart-content anchors are FTS-unresolvable by the P3.3-v3 strip, pre-existing). Known limitation: a multi-line heading's taller bbox can over-rank (pinned by a test).
- ~~**Docling heading MIS-DETECTION**~~ ✅ **SHIPPED + migrated 2026-05-24** (commit `87e71be`, follow-on to `5bd032c`). The level-fix above surfaced that Docling also mis-*detects* body sentences as `SectionHeaderItem`s on dense docs (the 10-K tagged "Data centers are becoming AI factories." and a 304-char paragraph as headings). Fixed at the **root** (per request — not by regexing markdown): `docling_worker._demote_misdetected_headers` reclassifies a prose-like header into a plain `TextItem` **before export** (SectionHeaderItem subclasses TextItem and the serializer dispatches headings by `isinstance`, so reassigning `item.__class__ = TextItem` makes it serialise as a paragraph natively — correcting the document model so chunking + `heading_path` see a paragraph). Signal is textual (bbox height is too noisy for detection on slide decks): demote if the text ends in terminal punctuation (≥4 words, guarding "Item 1.") or runs >15 words. The markdown post-process `_demote_prose_headings` is retained as a defence-in-depth **fallback**. Re-migrated the 10-K: **30 prose sentences reclassified to paragraphs** (H6 491→461, total headings 501→471, 0 prose-like headings remain, chart blocks preserved), eval still **7/10 ANS, refusal_cf=1.0**, chart-content 09/10 ANS, cp unchanged. 23 unit tests; full suite 416 green.

- ~~**Table-chunk enrich `chunk_failures`**~~ ✅ **SHIPPED + migrated 2026-05-24** (commits `c13c819` output-bound + `e2a795f` max_tokens; `fe1979c` chunker-cap/truncate hardening). The 10-K's ~56 `enrich.chunk_failed` events were **not** input-size (the first hypothesis): they're `"Model output did not match the requested schema"` — on a dense financial **table** the model emits an **unbounded** entity/citation list that runs past `max_tokens` and truncates the JSON mid-emission, so `model_validate_json` rejects it and the chunk is dropped. Real fix (the documented runaway-output pattern): bound `EntityList.entities` + `CitationList.citations` at `max_length=24` (xgrammar enforces maxItems → the list closes within budget), bound `CitationCandidate.surface_text` at 200, and raise the enrich per-call `max_tokens` to 3072. **56 → 4** chunk_failures (entity_count 2369 → 3080); the residual 4 are the absolute densest tables (content still retrievable via vectors/FTS — only graph entities missed). Complementary hardening shipped alongside (chunker `MAX_CHUNK_MULTIPLIER` cap that line-splits word-heavy oversized units; enrich prompts → v2 with `truncate(6000)`) — but those were a **no-op on these word-light/char-heavy tables**; output-bounding is what fixed it. Validation added 2 table-targeted counterfactuals to the annual-report eval (ar-11/12, both refuse correctly → refusal_cf 5/5=1.0). Full suite 429 green.
- **NEW — table ANSWERABILITY** (surfaced by the above validation). Table-*only* values (e.g. Graphics segment revenue, C&N segment OI growth — present only in a GFM table, not in prose) **false-refuse**: the dense table chunks rank poorly in dense+FTS retrieval and the answer prompt truncates them to 1800 chars (~9% of a 21K-char table). Data Center revenue answers (ar-03) only because it's *also* in prose. The enrich fix recovered table *entities*; making table *content* answerable is a separate retrieval-quality effort — char-aware chunk splitting + GFM table-header repetition on splits (the deferred work) + possibly table-row-aware indexing. Contained but multi-part; no external dependency.

### Tier 2 — multi-session foundational (unblocks downstream verdicts)

These need session-spanning investment but their completion unlocks several downstream items in tier 3.

- **P0 corpus extension** (curator-time, multi-session). The single most-impactful pending work item; would lift multi-doc generalisability. **Single-doc 30-query extension within slide-decks ✅ shipped 2026-05-21** (n=17 answerable). **Multi-doc vault ✅ expanded 2026-05-23**: ingested 6 user-provided CR350 French security-course PDFs (Cours 2/3/6, Semaine 4/5, syllabus) → **10-doc vault** (7 related French lectures + CUDA deck + 10-K + Tableau guide). **Multi-doc DISAMBIGUATION eval ✅ shipped 2026-05-23** (`tests/eval-data/cr350-multidoc/`, commits `4098add`+`477c166`): 15 cross-lecture queries (11 ANS + 4 REF) where each ANS answer lives in exactly one of the 7 lectures, plus two deliberate confusables (OSI layers in Cours 2 & 6; MAC addresses in Cours 3 & Semaine 4). **Baseline: 11/11 ANS, refusal_cf=1.0, 0 hallucinations, citation precision 1.0** — every query disambiguated the correct owning lecture. `scripts/extend_corpus.py resolve` made multi-doc aware (per-query `_doc_id`). The eval surfaced + fixed a real bug: the answer LLM occasionally corrupts the long `docid#hash` ids it cites (bare hash / single-char flip), so `agents.answering._repair_claim_chunk_ids` now snaps emitted ids back to real reranked chunk_ids (exact → suffix → bounded-fuzzy) before verify/compose — took citation precision 0.62 → 1.0. Remaining sub-goal: wire CER/WER/structural-F1. Sub-goals:
  1. ~~Extend to 3–5 documents~~ ✅ done (7 CR350 lectures); ~~hand-label cross-lecture French queries~~ ✅ done (`cr350-multidoc`, baseline clean).
  2. Bootstrap each of the other 6 categories from `docs/eval-corpus-plan.md`: modern-printed, scientific-papers, technical-docs, historical-scans, handwritten, forms. ◐ **Started 2026-05-24**: `eval-corpus/` has **3 synthetic fixtures across 3 categories** (all authored canonical-first → ground truth independent of the SUT; rendered to born-digital PDFs via **LibreOffice** through each `generate.py`, which carries the `LD_LIBRARY_PATH` fix for the broken `$ORIGIN` RUNPATH). Full-corpus `memex eval-parse` (post heading-level-recovery fix `bb66c82`): **3/3 pass, mean CER 0.036 / WER 0.106 / structural-F1 0.974** (was 0.566 before the fix). Per fixture: `modern-printed/tidewater-maintenance-log` (prose+lists+HTML-table; CER 0.051, F1 0.923) — the HTML table flattens under the writerweb filter; `forms/quarterly-uptime-report` (native `.fodt` bordered table; CER 0.024, F1 1.0) — parses as a proper **GFM table**, header row splits across lines; `technical-docs/widget-cli-reference` (deep H1–H4 + code; CER 0.030, F1 1.0) — **code blocks parse with high fidelity**. **Resolved follow-up**: the dropped-post-heading-paragraph gap was a PyMuPDF `Story` render artifact, not a real-PDF weakness (LibreOffice render parses them). **The heading-collapse finding these fixtures surfaced is now FIXED → see the Tier-1 heading-level item.** Still curator-gated: representative cross-category numbers need **human-curated real PDFs**; table/equation structural-F1 metrics remain to implement.
  3. ~~Wire CER/WER/structural-F1 from `src/memex/eval/scoring.py` into `runner.py`~~ ✅ **done 2026-05-24** (`run_parse_eval` + `memex eval-parse`; consumes the `<doc>/ground-truth.md` + `manifest.json` layout, predicted from the vault or a `predicted.md` override; headings F1 fence-/chart-block-aware). The plumbing is live and tested; what remains is **curator work** — hand-transcribed `ground-truth.md` per doc to run it against (+ table/equation structural-F1).
- **P4.1 wikilinks — course cross-reference resolver ✅ shipped 2026-05-23** (`enrich/course_refs.py`, commit `4a97c90`). Resolves the 2026-05-23 "0 wikilinks on lecture content" finding: the academic citation resolver emits nothing on lecture slides, so a pure-text resolver now turns "Cours N" / "Semaine N" mentions into `[[doc#section]]` wikilinks to sibling lecture docs (matched by ordinal number, gated to the same course code, cross-label aware so syllabus "Cours 4" → the "Semaine 4" doc). Validated live: enriching the CR350 syllabus went **0 → 6 wikilinks** to all six lectures, each with a P4.1 section anchor.
- **Frictionless document retitle ✅ shipped 2026-05-24** (`index.retitle_document`, commit `21e4ab7`). A title is pure metadata — `chunk.text` (the embedded input) and `chunk_id = hash(doc_id, text)` are both body-only, so a rename never invalidates a vector or chunk id. `retitle_document` rewrites the frontmatter (source of truth, only if changed) then fans the title out to FTS + vector + graph via **metadata-only** column updates — no GPU, no re-embed, idempotent; doubles as a "repair a stale denormalized title" tool. This is the GPU-free replacement for the force-reindex that OOM'd the 10-K when only its title needed fixing. Surfaces: CLI `memex retitle <doc_id> [TITLE] [--derive]` (`--derive` pulls from the manifest source filename) + webui inline `<h1>` rename (HTMX, mirrors the body edit/review flow). `parse._derive_title` promoted to public `parse.derive_title`. Tests pin no-re-embed + stable-chunk-ids.
- **Repo-wide quality cleanup ✅ 2026-05-24** (commits `21e4ab7`..`a9a1381`). **ruff 139 → 0**: genuine fixes (a `pause_vllm` `finally`-`return` that would have swallowed parse exceptions; a GC-able fire-and-forget client-close task; sha1 `usedforsecurity=False`; `try/except/pass`→`contextlib.suppress`) plus documented justifications (per-file-ignores / inline `# noqa … # reason`) for the false positives. **pyright `--strict` `src/memex` 507 → 0; whole repo 0 errors / 0 warnings**: every error was untyped-dependency cascade (amplified by pyright 1.1.409 vs the pinned floor), resolved at the boundary via the new `stubs/` dir + call-site casts — not silenced. Surfaced + fixed the latent `CrossEncoder(automodel_args=)` → `model_kwargs` bug. The type-discipline standard (stubs vs call-site casts; keep src strict-clean) is now in `GUIDELINES.md` + `src/memex/CLAUDE.md`.
- ~~**P3.3 — Chart-data extraction**~~ ✅ **Fully closed 2026-05-23 with v7 root-cause fix + Q05 verify-phantom-index fix**. The journey: 6 v1–v6 implementation iterations → 4-backend shootout chose Nemotron-Parse-v1.2 as the no-prose-regression winner (default-flipped on, [`audits/chart_ocr_shootout_2026-05-23.md`](audits/chart_ocr_shootout_2026-05-23.md)) → `--force-docling` flag (commit `a4b8493`) enabled chart-OCR validation on classifier-routed PyMuPDF docs → A/B revealed chart-content questions ALL still refused → trace identified retrieval was healthy (rank-1 score 0.80) but three layers downstream blocked the answer: raw LaTeX emission (unreadable to LLM), single-row label-number cell ambiguity, P3.3-v3/v5/v6 prompt-render strips hiding chart blocks. **v7 fix** (commit `a9e8326`) addresses all three: `_normalize_latex_tabulars` + `_split_label_number_cells` post-processors on Nemotron output, strips removed from assess/answer/verify, assess prompt updated. **First chart-content REF→ANS flips**: chart-types-08 "On Time vs Late" (impossible under both PyMuPDF and pre-v7 Docling, now ANS ✓), annual-report-09 "CC December 2024" via Docling+chart-OCR+v7 ✓. **Q05 verify-phantom fix** (commit `ccb70cb`): the verifier was hallucinating phantom claim-indices > n-1 when distracted by chunk content not in the draft (chart-types Q05 "bar charts + maps" had 1-claim draft + verifier emitted `ungrounded=[1]` referencing "Pareto chart" from chunk text). Fixed via `verify_grounding/v2` prompt tightening + defensive phantom-index filter in `verify()` (xgrammar bounded list length but not index values). chart-types ANS 2→4 with Q05 recovered. **Q09 table-literal answer-prompt fix** (commit `2e4b4ad`): chart-types Q09 "Which team members are assigned to GIF projects?" still false-refused with `claims:[]` after Q05 fix — the answer-prompt v3's Literal-presence + No-substitute rules were too strict for clearly structured tables (LLM interpreted "find rows where Project=GIF, report Assigned" as inferential rather than literal). Fixed via answer/v3 prompt extension mirroring verify_grounding/v2's table-row-literal rule (which was already permissive). chart-types ANS 4→5. Trade-off: annual-report Q03 "Data Center revenue" tightened from a sloppy ANS (citation_precision=0.0 — citing prose growth-rate, not the table $193,737 absolute) to a correct refusal. Net 0 across the two corpora; sloppy-ANS becoming correct-refusal is a CORRECTNESS improvement. Slide-decks unchanged (11/30, mcp 0.909). HARD GATES preserved throughout. Full session-by-session detail in `~/.claude/projects/.../memory/p33_tracker.md`. **v7 stack generalized to slide-decks (commit `63dcb80`)**: Added 3 chart-content queries to `tests/eval-data/slide-decks/queries.json` targeting content exclusively in chart-extracted blocks of the canonical CUDA deck. Result: 2/3 answer initially (Q32 "CameraGestureBase 2,070 ms" ✓; Q33 "NumPy, SciPy, CuPy..." ✓; Q31 "4 design principles" REF). **Q31 chunker chart-block H1 fix** (commit `c63b75f`): Q31 failure traced to the chunker splitting the chart-extracted block on `# Heading`-prefixed chart-figure labels — `_HEADING_RE` treated `# Minimal lead-time` etc. as document section boundaries, splitting the chart block such that only the last principle survived in the top-5 reranked chunk. Fixed via new `chart_extracted_spans` + `is_inside_any_span` helpers in `core/text.py`; `chunker._split_into_sections` and `_heading_path_at` now filter out heading matches inside chart blocks. **Three queries flipped REF→ANS** post-reindex: Q31, Q18 ("NVLink C2C bandwidth 900 GB/s" — serendipitous chunker-reflow recovery), AND annual-report-Q03 ("Data Center revenue $193.737B" — sloppy ANS that was citing prose growth-rate is now a correct ANS citing the actual table chunk). slide-decks: 11/30 → **15/33 ANS** (+4 wins this arc); annual-report: 6/10 → 7/10. 3 new chunker tests pin the contract. **Total chart-content REF→ANS flips across the 3 corpora**: **7** (chart-types-08, annual-report-09, annual-report-10, slide-decks-18, slide-decks-31, slide-decks-32, slide-decks-33) PLUS annual-report-03 sloppy→correct. HARD GATES preserved throughout. **Cleanup (commit `fc6a6a9`)**: stripped stale `_unresolved` markers from chart-content queries (anchors deliberately unresolvable via FTS by P3.3 v3 BM25-strip defense); fixed `extend_corpus.py` crash on string-shaped `_mix` (slide-decks legacy field). **All originally-queued P3.3 v7 follow-ups now closed**; the remaining open item is the broader P0 corpus extension (external-blocked).

### Tier 3 — eval-gated stack swaps (unlocked by Tier 2 P0 extension)

These are ready to run once the corpus has the depth to discriminate between candidates.

- **P2.1 — Qwen3-Reranker-0.6B quality A/B.** ◐ Infra shipped (commits `714dd32`, `b485748`); the swap is a quality play, not a memory play (~2.1 GB live, ~equal to bge). Run the same 15-query (eventually larger) set with `MEMEX_MODELS__RERANKER_BACKEND=qwen3` vs the default `cross_encoder`; pick the winner per category.
- **P2.2 — Granite 4.1-8B-FP8 vs `Qwen/Qwen3-8B-AWQ`.** ❌ **vLLM-blocked 2026-05-21**. Model downloaded (4 GB) but vLLM 0.21 hangs indefinitely at "Using FlashAttention version 2" in both cudagraph and `--enforce-eager` modes. Granite 4.1's hybrid Mamba2+attention architecture appears unsupported in vLLM 0.21 (deadlock during FA2 init for the hybrid backend). Resolution: vLLM 0.22+ with hybrid-arch support, OR GGUF variant via `--quantization gguf`, OR fall back to Granite 3.x.
- **P2.3 — Qwen3-VL-8B-Instruct vs `Qwen2.5-VL-7B`.** +6–14 OCR points published; native 256 K context. Same AWQ-Int4 footprint. Currently disabled by default on 12 GB; the swap helps the larger-VRAM tier. Eval-gated.
- **P1.6 chunker-size verdict.** Is `chunk_target_tokens=400` the right default, or should it be 500–600 with a longer model context? Needs corpus depth to be answerable.

### Tier 4 — cost / ops decision (no code blocker; resource blocker)

- **P3.1 — Real-mode benchmark nightly CI.** `scripts/benchmark.py --real` measures cold-start + first-token + embedding throughput. Needs a GPU runner: cloud (Lambda, RunPod, Modal) or a dedicated home rig. Workflow template already in `.github/workflows/`. Don't pick up without confirming the user has the resources allocated.

### Tier 5 — design decisions still owed (punt until a user needs it)

- ~~**P4.1 — Wikilink section anchors**~~ ✅ **Fully shipped end-to-end 2026-05-23** (ADR-0003 spec realised through to the browser). `[[doc_id]]` since v1. `[[doc_id#section]]` shipped in three layered commits: (1) **read-side primitives** `93db990` (`core/wikilinks.py`: `WikilinkTarget` + `parse_wikilink` + `extract_wikilinks` + `resolve_wikilink_section`, 19 unit tests); (2) **write-side wiring** `bee48fc` (enrichment emits `[[doc#section]]` opportunistically when citation context references a target heading via `DocSignature.headings`, populated from `core/text.py::extract_heading_texts`; chart-block-aware; longest-match wins; case-insensitive; ≥4-char heading floor; 12 unit tests); (3) **webui section-scroll rendering** `5dddb5e` (`webui/rendering.py` with `render_body_html` + `slugify_heading`: HTML-escapes the body for XSS safety, rewrites `[[doc#section]]` to `<a href="/documents/doc#slug">`, injects `<span id="slug">` before each heading so URL fragments scroll the heading into view; chart-block-aware heading skip; 18 unit tests). Tightened `_HEADING_RE` / `_MARKDOWN_HEADING_RE` from `\s` to `[ \t]` as a defensive side-fix (empty-text headings like `## ` no longer swallow subsequent newlines + match later content as heading text). **TOC** renders as a sticky sidebar in the markdown-only layout (commit `be8dee0`) and a collapsible drawer in the PDF-split layout (commit `001005b`), gated to 3–50 headings (the 10-K's 501 parse-artifact H2s correctly suppressed). **Browser-verified live (Claude-in-Chrome) + edge cases hardened** (commit `07dc789`): removed the visible `#` permalink glyph (it read as `## Heading #` ATX closing-hash in the raw-`<pre>` view — fidelity pollution); duplicate headings now GitHub-style deduped (`tips`/`tips-1`/`tips-2` — the Tableau guide has 8 `Tips:` headings that previously all collided on `id="tips"`); line-by-line rendering fixes HTML-escape offset-drift that misfired the chart-block filter when `<`/`&` preceded a block; `clean_heading_text` strips inline markdown so `## [Tips:](url)` shows "Tips:" in the TOC. `_walk_headings` is the single source of truth (chart-block-aware + dedup + clean) shared by anchor-span IDs and TOC hrefs. 39 webui-render unit tests + live browser confirmation (34 unique anchor IDs, TOC labels clean, click→pulse→scroll). **Remaining (out of P4.1 scope)**: MCP-side wikilink emission + agent-emits-wikilinks-in-answers — both need a use case.
- ~~**P4.2 — 8 GB GPU tier.**~~ ✅ **Shipped 2026-05-21**. Smaller-orchestrator tier (`Qwen/Qwen3-4B-AWQ` + `gpu_fraction=0.50`) eval-verified across 5 sessions (spec → baseline → Qwen3-4B benchmark → Granite 4.1-3B comparison → P2.1 A/B re-run). HARD GATES (refusal_cf=1.0, hallucinations=0) hold on both tiers. Side effect: also unblocked P2.1 (now resolved with cross_encoder winning). Full env-var matrix + per-tier eval numbers documented at [`docs/deploy/hardware-tiers.md`](deploy/hardware-tiers.md).
- **P4.3 — Trace retention.** `EventBus` has 30-day prune; Langfuse self-host wiring is open. Match retention windows when self-host lands.
- **P4.4 — Dynamic VRAM Manager.** Brainstorm 2026-05-21 (raised while planning P3.3): an active manager that loads/unloads GPU-resident models based on document type + stack config, replacing the static `gpu_memory_utilization` + the manual `pause-vLLM-during-parse` dance shipped in P3.3 Session 4. *Potential value*: removes per-feature "is VRAM tight?" decisions; smoother multi-model coexistence on the 12 GB tier; opportunistic upgrade on the 8 GB tier when VRAM is free. *Architectural overhead*: new abstraction owning every model's lifecycle; state machine across model combinations; coordination protocol between daemon processes; vLLM still needs full restart for downsize (it doesn't expose shrinkable reservation); multiplied test surface; failure-mode complexity during state transitions. *Initial verdict*: probably not worth shipping NOW — the 2-tier solution (P4.2) plus the P3.3 pause-vLLM strategy cover the current pain. Re-evaluate IF (a) a third mandatory parse-time model lands, OR (b) the 12 GB tier becomes the dominant user base, OR (c) a multi-GPU story emerges (the manager becomes a topology manager too). Punt until a forcing function arrives.

### Browser-OCR research turn 2026-05-21 — three candidates rejected

Three parallel research subagents evaluated browser-side OCR/VLM technologies as candidates for the chart-OCR problem. All three rejected; orthogonal to the actual problem.

- **Tesseract.js** (current v7.0.0, Apache 2.0): mature, ~15–25 MB first-load, pure text OCR. Same axis-label weakness as DePlot per ChartOCR / GenPlot / Scatteract literature; Docling already calls Tesseract server-side. Niche utility: client-side image-paste OCR UX only.
- **Surya OCR via Pyodide** (Surya 0.17.1 + Pyodide 0.29.4): Pyodide path fundamentally blocked — PyTorch on WASM has no working port and no announced roadmap (`pyodide/pyodide#1625` open since June 2021). License: Surya GPL-3.0 + model weights under custom Open RAIL-M.
- **InternVL3-9B via ONNX Runtime Web**: infeasible. No ONNX export of 8B+ InternVL3 exists; 9B q4 (~4.5 GB) exceeds ORT-Web's 4 GB WASM ceiling. Qwen2.5-VL-7B-AWQ already in our stack matches/exceeds on DocVQA (95.7 vs 92.7) and OCRBench (88.8 vs ~85) at 5 GB CUDA.

### Recently shipped this phase (compressed)

**Feature work (P0–P4 main-line):**
- ✅ **P0 30-query corpus + label refresh** — n=17 answerable; relabel 2026-05-22 restored `mcp_ans` from 0.0 → 1.0 after the Docling routing change orphaned the old PyMuPDF chunk_ids.
- ✅ **xgrammar empty-draft short-circuit** — verify node bypasses model call on empty draft.
- ✅ **Retrieval truncate-budget retune** — `truncate 700→1800`, `top_k 10→5`, `max_tokens 1024→640`, `max-model-len 4096→6144`. Three legitimate REF→ANS flips.
- ✅ **P2.1 Qwen3-Reranker quality A/B** — cross_encoder wins clearly at the 4B stack; default unchanged.
- ✅ **P4.2 smaller-orchestrator tier** — `Qwen3-4B-AWQ + gpu_fraction=0.50` ships as the 8 GB tier; full env-var matrix at `docs/deploy/hardware-tiers.md`.
- ✅ **P3.3 chart-OCR v1–v6** — pipeline shipped opt-in; see Tier-2 entry above for the closed mixed-verdict narrative + p33_tracker.md for session-by-session detail.
- ✅ **Filler N1–N9 audit-nit backlog** — closed; per-item one-liners in the §Filler section below.
- ✅ **Defensive schema hardening** — bounded 5 LLM-emit `str` fields + `DraftAnswer.claims` list cap so xgrammar enforces emission limits at the grammar level. Same pattern as the v6 `SufficiencyAssessment.reason` bound that fixed the counterfactual-query crash.

**Audit-and-housekeeping arc (2026-05-22):**
- ✅ **ROADMAP narrative consolidation** — collapsed the chart-OCR saga into one Tier-2 entry; removed scattered references; tightened headers.
- ✅ **CLAUDE.md ×3 drift audit** — synced project root + backend + webui to current code state (error subclasses, locking patterns, schema-bound convention, test counts).
- ✅ **GUIDELINES.md drift audit** — fixed stale model-stack table, config example (AWQ + chart-OCR + reranker_backend), structured-output idiom (`response_format` not deprecated `guided_json`), agent state diagram (added `expand_graph` + `compose` nodes).
- ✅ **ADR-0001/0006 outcome annotations** — every "candidate to evaluate" in ADR-0001 now has its shipped / blocked / eval-gated outcome.
- ✅ **Per-file test-organization audit** — 16 directly-tested + 14 indirectly-covered + 7 acceptable-gap modules + 1 actionable gap closed (`retrieve/fusion.py` got 10 unit tests pinning the RRF contract).
- ✅ **Pyright noise cleanup** — 33 systemic false positives suppressed via per-file pragmas (FastAPI/Typer handlers; `_FOO` module singletons). The remaining 307 errors are third-party stub gaps; 0 actual `Any` leaks from recent code.
- ✅ **Dead-code sweep** — removed `convert_page` (singular VLM); wired `get_pymupdf_breaker_state` + `list_prompts` into `memex doctor`.
- ✅ **Per-module docstring audit** — 78.2% → **100% public-surface coverage** (275/275). Pure documentation pass, no behaviour change.
- ✅ **Browser-OCR research turn** — three candidate technologies rejected (Tesseract.js, Surya-via-Pyodide, InternVL3-9B-via-ONNX-Web).

**P3.3-c chart-OCR shootout (2026-05-23):**
- ✅ **Three-agent landscape research** — investigated OneChart root cause + NVIDIA offerings + broader 2025-2026 landscape. Three agents converged on the architectural insight: Donut/VisionEncoderDecoder is the safe family. Captured in [`docs/audits/chart_ocr_landscape_2026-05-23.md`](audits/chart_ocr_landscape_2026-05-23.md).
- ✅ **Three-agent fine-tuning research** — official fine-tuning workflows for each candidate (UniChart, NeMo Retriever 2-stage, Nemotron-Parse-v1.2). Captured in [`docs/audits/chart_ocr_finetune_research_2026-05-23.md`](audits/chart_ocr_finetune_research_2026-05-23.md).
- ✅ **Path A — UniChart/chart-to-table** integration shipped + A/B eval (commit `8292169`). Donut-style VisionEncoderDecoder, 200M params, Apache 2.0, CUDA-stable. **Verdict: −1 ANS (10/30, same as DePlot)** on prose-heavy corpus.
- ✅ **Path C — NVIDIA Nemotron-Parse-v1.2** integration shipped + A/B eval (commit `68acac1`). 885M VisionEncoderDecoder; required ADR-0006 carve-out broadening (chart-OCR slot trust_remote_code now covers OneChart + Nemotron-Parse). **Verdict: ✅ ANS=11/30 (==baseline), refusal_cf=1.0, mcp_ans=0.955** — first chart-OCR backend that doesn't regress on prose-heavy corpora. Generates only 11 chart blocks (vs DePlot's 50, UniChart's 60); the conservative extraction is the architectural lever that finally aligns chart-OCR with prose-heavy use cases.
- ✅ **Final shootout audit**: [`docs/audits/chart_ocr_shootout_2026-05-23.md`](audits/chart_ocr_shootout_2026-05-23.md). Path B (NeMo Retriever 2-stage) deferred — single-model winner is sufficient.
- ✅ **Default flipped 2026-05-23** (commits `01b8c5e` + `3e8c22a` + `6957f78`): `ModelSettings.chart_ocr = "nvidia/NVIDIA-Nemotron-Parse-v1.2"` and `ParseSettings.disable_chart_ocr = False`. `[parse]` extras now include `albumentations`, `timm`, `open_clip_torch` (Nemotron-Parse processor deps). VRAM-fit estimator bumped 2.5 → 3.0 GB for the chart-OCR slot. Operators can revert via `MEMEX_PARSE__DISABLE_CHART_OCR=true` or switch backend via `MEMEX_MODELS__CHART_OCR=<alt-id>`.
- ✅ **Operational note**: 12 GB rig needs `MEMEX_VLLM_GPU_FRACTION=0.68` when chart-OCR is enabled (which is now the default).

**P3.3-c chart-OCR v7 — dark-matter root-cause fix (2026-05-23, late session):**

The 2026-05-23 default-flip shipped Nemotron-Parse-v1.2 as the chart-OCR backend with the published "no-regression" verdict on the chart-DOMINANT CUDA deck. Validating on chart-MIXED prose-heavy docs (NVIDIA 10-K annual report + Tableau visualization guide) via the new `--force-docling` flag revealed the wins on chart-content questions were **zero**: chart-extracted chunks reached rank 1 in the reranker with score 0.80 but the agent still refused with "no specific numbers." Three-fold root cause, all fixed in commit `a9e8326`:

1. **Nemotron-Parse emitted raw LaTeX** (`\begin{tabular}{cc} **On Time 22** & **Late 8**\\ \end{tabular}`) which the Qwen3-8B-AWQ assessor read as "no specific numbers." Fix: `_normalize_latex_tabulars` regex-based converter in `chart_ocr_backend.py` flattens LaTeX tabular → markdown table.
2. **Even after LaTeX→markdown**, single-row chart-summary cells like `**On Time 22**` were ambiguous (label vs. label+value). Fix: `_split_label_number_cells` heuristic splits them to key:value bullets (`- On Time: 22 / - Late: 8`).
3. **The P3.3-v3/v5/v6 defenses stripped `[chart-extracted]` blocks** from the chunks in the assess/answer/verify prompts — a correct defense against verbose-LaTeX-eating-the-truncate-budget but now the v7 compact markdown emission (~50-150 chars per block) lives comfortably alongside prose within `truncate(1200)/truncate(1800)`. Fix: removed all three strip-call-sites in `agents/answering.py`; chart blocks now flow through to the LLM. Belt-and-suspenders: `prompts/assess_sufficiency/v1.md` gained one paragraph explaining `[chart-extracted]` block semantics to the assessor.

**Validation A/B** (extended corpora: `tests/eval-data/annual-report/queries.json` 10q + `tests/eval-data/chart-types/queries.json` 9q):

| Corpus | Mode | ANS | mcp_ans | refusal_cf |
|---|---|---|---|---|
| chart-types (9q) | PyMuPDF baseline | 2/9 | 0.000 | 1.000 ✓ |
| chart-types | Docling+chart-OCR (pre-v7) | 2/9 | 0.000 | 1.000 ✓ |
| chart-types | **Docling+chart-OCR+v7** | **3/9** | **0.333** | **1.000 ✓** |
| annual-report (10q) | PyMuPDF baseline | 7/10 | 0.000 | 1.000 ✓ |
| annual-report | Docling+chart-OCR (pre-v7) | 6/10 (−1) | 0.333 | 1.000 ✓ |
| annual-report | **Docling+chart-OCR+v7** | **7/10** | 0.286 | **1.000 ✓** |

**First chart-content REF→ANS flips**: chart-types-08 "How many projects On Time vs Late" (was refused in **both** PyMuPDF and pre-v7 Docling modes, now answered ✓); annual-report-09 "What did the CC do in December 2024" via Docling+chart-OCR+v7 ✓.

**`--force-docling` operator override** (commit `a4b8493`): the PyMuPDF classifier sends most born-digital text-heavy PDFs to the fast path (Adobe InDesign / Acrobat-output / etc.), skipping chart-OCR entirely. `memex parse --force-docling <doc-id>` (or `MEMEX_PARSE__FORCE_DOCLING=true`) bypasses the classifier and routes directly to Docling. Cost: ~10× slower parse on text-heavy docs. Use cases: chart-OCR validation on mixed-content docs, chart-content question recovery on financial reports, dashboards, infographics.

**`scripts/extend_corpus.py` corpus-extension scaffolding** (commit `761bbcd`): 4-subcommand workflow (`init` / `inspect` / `resolve` / `ab`) that streamlines "drop a PDF, hand-label queries, A/B test before/after parse-routing change."

**Tests + new files:**
- 7 new unit tests in `tests/unit/test_chart_ocr_backend.py` covering the LaTeX→markdown converter (single-row label-number split, multi-row table, `\multicolumn` flatten, end-to-end mixed text, truncated tabular tolerance, empty inputs)
- 4 new unit tests in `tests/unit/test_force_docling.py` pinning the flag contract (default False, env-var True, kwarg short-circuit, settings-default path)
- New eval corpora: `tests/eval-data/annual-report/queries.json` (10q: 5 prose ANS + 3 REF + 2 chart-content) + `tests/eval-data/chart-types/queries.json` (9q: 5 prose ANS + 2 REF + 2 chart-content)

**Final P3.3 disposition (2026-05-23 fully closed):**
- `disable_chart_ocr=False` (default-on)
- Nemotron-Parse-v1.2 backend wired + LaTeX→markdown converter applied to its output
- assess/answer/verify see chart blocks (strips removed)
- `--force-docling` flag available for operator overrides
- HARD GATES (refusal_cf=1.0, hallucinations=0) preserved across all measurements
- Chart-content questions now actually answerable

**P3.3-b OneChart retry (2026-05-23):**
- ✅ **ADR-0006 amendment** — carved out `trust_remote_code=True` exception for the chart-OCR slot only, OneChart specifically. Documented mitigations (Apache 2.0, 0.3B params, opt-in via env var, seccomp-sandboxed parse-time process, `reliable_check` self-consistency gate).
- ✅ **Backend implementation** — new `_load_chart_ocr` dispatch branch for OneChart; new `_chart_ocr_transcribe_onechart` helper with `reliable_check` parsing + dict-to-markdown conversion + defensive TypeError / RuntimeError handling. 5 new unit tests pin the contract.
- ❌ **A/B/C eval verdict**: OneChart failed catastrophically on the CUDA deck — every single figure triggered a CUDA device-side assertion (position-embedding overflow on out-of-distribution chart imagery). Defensive handling kept HARD GATES intact (refusal_cf=1.0, mcp_ans=1.0) but the model produced zero usable extractions. Config A baseline 11 ANS = Config C OneChart 11 ANS (identical because no chart blocks were stitched). Config B DePlot 10 ANS (matches prior P3.3 v6 verdict). Full audit: [`docs/audits/onechart_2026-05-23.md`](audits/onechart_2026-05-23.md).
- ✅ **Decision**: revert to DePlot-only as the documented chart-OCR backend. OneChart stays in tree behind the ADR-0006 carve-out for future re-attempts on chart-heavy corpora or upstream revisions. Default stays `disable_chart_ocr=True`.

**French support hardening arc (2026-05-22 — 2026-05-23):**
- ✅ **Multilingual chunker + FTS5** — `_SENTENCE_RE` extended to Latin-1 + Œ + Ÿ; FTS5 schema flipped from `porter unicode61` to `unicode61 remove_diacritics 2`. End-to-end validated against a French course PDF (CR350, 45 pages, 51 chunks via Docling).
- ✅ **French eval corpus** — `tests/eval-data/french-course/queries.json` (5 ANS + 3 REF queries against the CR350 doc); HARD GATES pass at refusal_cf=1.0, mcp_ans=1.0.
- ✅ **Language-mirror answer prompt (`answer/v3`)** — adds an explicit "write summary + claims in the query's language" directive; positioning study found recency-bias matters (rule at top → refusal_cf dropped to 0.77; rule at end with explicit hedge → 1.0 preserved).
- ✅ **MIRACL-fr retrieval benchmark** — published 0.807 nDCG@10 with cross_encoder rerank (vs 0.755 dense-only); matches bge-reranker-v2-m3 published range. Validates bf16 dispatch + retrieval-pipeline composition.

**Qwen3 prompt-engineering follow-ups (all shipped 2026-05-22 — 2026-05-23):**
- ✅ **#1 Positive worked example** in `answer/v3` — required Path C schema tightening (max_length 600→300, max_items 20→8, source_chunk_id max_length=80) + `max_tokens` 640→1024 to avoid schema-overflow crash. Net English ANS +2.
- ✅ **#2 Sampling tuning + seed** — temperature 0.0→0.1, top_p=0.8, presence_penalty=1.0 (capped — >1.5 triggers AWQ language mixing), seed=42. English mcp_ans 0.955→1.0.
- ✅ **#3 System/user message split** — `render_messages()` + `<!-- ===USER=== -->` marker; answer node now sends a 2-message ChatML structure (system block ~800 tokens, user block dynamic). Enables vLLM prefix-cache reuse on the static system block.
- ✅ **#4 Schema-block trim** — removed redundant in-prompt JSON-schema spec (~80 tokens saved per call); xgrammar enforces the schema at the grammar level. Research had warned against this for <10B models; empirical A/B disproved the concern at our scale.
- ✅ **#5 SamplingSettings centralization** — sampling defaults moved from hardcoded `complete_structured` kwargs to `MemexSettings.inference.sampling`. Per-deployment override via `MEMEX_INFERENCE__SAMPLING__*` env vars.

Research notes: [`docs/audits/qwen3_prompt_engineering_2026-05-22.md`](audits/qwen3_prompt_engineering_2026-05-22.md). MIRACL-fr results: [`docs/audits/miracl_fr_2026-05-22.md`](audits/miracl_fr_2026-05-22.md).

### Next pickup — ranked by impact × feasibility

1. **🎯 P0 corpus extension — multi-doc / multi-category** (Tier 2, multi-session). The single biggest pending validation: all current quality verdicts come from one document per category. **Needs user-provided source material** — not autonomously executable. The post-v7 fix landscape opens a new sub-goal: chart-DOMINANT corpora (financial dashboards, infographic decks, dataviz tutorials) where the chart-OCR+v7 stack's chart-content answering capability would actually shine.
2. **🔧 P3.3 v7 follow-ups (not blocking)** — three open items captured in `p33_tracker.md`:
   - (a) chart-types-09 multi-row Gantt-assignee table still false-refuses. Possible angle: per-cell-pair prompt rendering or cross-product table reformat.
   - (b) Q05 prose ("bar charts + maps") false-refuses under chart-OCR+v7 but answers under PyMuPDF — likely chart-block-in-verifier-view perturbation. n=1; needs more reps to confirm.
   - (c) Validate v7 on additional chart-dominant corpora; expand chart-content query coverage in existing slide-deck eval.
3. **🪜 P2.2 Granite 4.1-8B vs Qwen3-8B** — vLLM-blocked; unblocks on vLLM 0.22+ hybrid-arch support OR Granite GGUF variant OR fallback to Granite 3.x.
4. **🪜 P2.3 Qwen3-VL vs Qwen2.5-VL VLM swap** — eval-gated; needs a scan-style document corpus to actually exercise the VLM path.
5. **🔬 P3.3-a Qwen2.5-VL chart-OCR retry** — external-blocked on autoawq/transformers compat for the `PytorchGELUTanh` rename.

Items below (P3.1 benchmark CI / P4.1 wikilink anchors / P4.3 trace retention / P4.4 Dynamic VRAM Manager) stay queued; pickup needs an external trigger (GPU runner, real cross-doc citation, Langfuse self-host, forcing function).

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

All N1-N9 ✅ shipped 2026-05-21:

- ~~**N1**~~ ✅ — LanceDB concurrent-search smoke test (3 integration tests; verified async surface tolerates burst-of-20 concurrent reads; multi-store independence + lancedb 0.30 deprecation cleanup as a bonus)
- ~~**N2**~~ ✅ — `FTSStore` explicit `asyncio.Lock` gating multi-statement writes (`upsert`, `delete_document`, `delete_chunks`); reads stay unlocked to preserve concurrent-`/ask` throughput. 3 integration tests verify burst-of-10 upserts + sequential-read consistency + interleaved upsert+delete. Documented in the lock docstring that concurrent-read-while-writing on the same sync sqlite3 connection is out of N2's scope.
- ~~**N3**~~ ✅ — `configure_client` httpx pool linger fix; best-effort `loop.create_task(old.close())` when an event loop is running, structured-log warning otherwise. 3 unit tests cover both code paths.
- ~~**N4**~~ ✅ — `GraphStore.close()` drops the `self._conn` reference so ryugraph's C++ destructor fires deterministically (flushing WAL, closing file handles). Idempotent; second call is a no-op.
- ~~**N5**~~ ✅ — `_pid_alive` EPERM-returns-True semantics verified + pinned with 6 unit tests (EPERM→True, ESRCH→False, success→True, pid<=0→False, unknown errno→False+warn, EPERM emits debug-level structured event for operator visibility into cross-uid mismatches).
- ~~**N6**~~ ✅ — `_COMPILED_GRAPH` thread-safety on first compile via `threading.Lock` + double-checked locking. Fast path (cache hit) skips the lock; slow path takes it and re-checks under the lock. 3 unit tests verify 10-thread concurrent-compile is single-shot, 100-call fast path triggers no rebuild, reset clears under lock.
- ~~**N7**~~ ✅ — pypdfium2 `to_pil` lifetime fix in both `vlm_backend.py` and `chart_ocr_backend.py`: explicit `.copy()` after `to_pil()` (and after lazy `.crop()` in chart-OCR) forces PIL to allocate its own buffer + memcpy BEFORE the parent doc closes. Removes a latent UAF that would corrupt or segfault on the first read of the returned image.
- ~~**N8**~~ ✅ — Docling breaker lambda coroutine pattern verified; the `breaker.run(lambda: docling_convert(...))` form correctly produces the awaited value (not the coroutine object). 5 unit tests pin the contract: lambda-returns-coroutine awaits cleanly, exceptions propagate, threshold-trip short-circuits without invoking the lambda, fresh coroutine per call, `is_failure` predicate sees actual exception types.
- ~~**N9**~~ ✅ — lifted the `GraphNeighbor` import out of `TYPE_CHECKING` in `mcp/server.py` and removed the forward-ref quotes from `get_graph_neighbors`'s return annotation. MCP startup is now warning-free.

The Filler list is closed.

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

For the full per-commit log: `git log --oneline` (141 commits at session close, 2026-05-23 extended; the up-to-date number is `git rev-list --count main`).

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
| `MEMEX_PARSE__FORCE_DOCLING` | `false` | When `true`, the PyMuPDF classifier is bypassed and Docling runs on every PDF. Per-call equivalent: `memex parse <doc-id> --force-docling` or `memex ingest <path> --force-docling`. Cost: ~10× slower than PyMuPDF on text-heavy docs (Docling is the heavyweight engine). Use to force chart-OCR onto born-digital text-heavy docs the classifier would otherwise route to PyMuPDF (Adobe InDesign / Acrobat output / etc.). |
| `MEMEX_MODELS__CHART_OCR` | `nvidia/NVIDIA-Nemotron-Parse-v1.2` | Chart-OCR backend HF id. Defaults to the P3.3-c shootout winner. Alternatives in tree: `khhuang/chart-to-table` (UniChart, smaller + faster but −1 ANS on prose-heavy corpora), `google/deplot` (legacy P3.3 v6 default; same −1 ANS), `kppkkp/OneChart` (CUDA-asserts on OOD imagery — keep for chart-heavy-only re-attempts). Requires `MEMEX_PARSE__DISABLE_CHART_OCR=false` to actually run. |
| `MEMEX_PARSE__DISABLE_CHART_OCR` | `false` | When `true`, the chart-OCR pass over Docling figures is skipped. Disable when (a) running an air-gapped install that didn't pull `albumentations`+`timm`+`open_clip_torch`, OR (b) running on a corpus where chart-OCR's ~30s pause-vLLM-during-parse overhead isn't justified by chart-content questions. Default is `false` since 2026-05-23 with v7 fix delivering chart-content answering. |
| `MEMEX_VLLM_GPU_FRACTION` | `0.72` | vLLM's `gpu_memory_utilization`. **Set to `0.68` when chart-OCR is enabled** to leave room for embedder + reranker + chart-stitched chunks on the 12 GB tier (otherwise KV-cache OOM during chart-OCR-enabled evals). The `serve-vllm.sh` script reads this; the `memex daemon start` path forwards it. |
