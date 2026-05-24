# Spec: Table-RAG — linearization (Phase 1) + structured store & SQL tool (Phase 2)

**Status:** Phase 1 in implementation (2026-05-24). Research basis: [[table-rag-research-2026-05-24]] (3-subagent verdict). **Two gated phases**, each spec-driven → subagent execution → independent validation → GPU acceptance.

## Problem & evidence
Table-only values false-refuse: value cells + column labels don't co-locate after chunk/truncate, number-dense table chunks embed/rank poorly, and the answer LLM can't map a value to its column. Evidence: markdown-KV beats markdown-table 60.7% vs 51.9%; header-in-every-row improves table retrieval (arXiv 2408.17008); TableRAG EMNLP'25 (SQL execution dominant on aggregation) on a BGE-family stack like ours.

## Scope note (honest)
These fixes target **well-formed tables** (a real GFM header row of column labels). The 10-K **segment table** is *degenerate at the source* — both PyMuPDF and Docling emit its column labels (Compute & Networking / Graphics / Total) as a detached heading + stray line, never as table structure — so the header-sanity gate will skip it and the SQL path inherits the same bad structure. Acceptance therefore targets a **well-formed** table; the segment table is a documented parse-degeneracy outlier (its only clean fix would be the fragile prose-label re-attach, deliberately gated off).

---

## Phase 1 — table linearization at parse (engine-agnostic, the foundation)

### `src/memex/parse/table_linearize.py` (NEW)
`linearize_gfm_tables(markdown: str) -> str`: scan for GFM tables (header pipe-row + delimiter row `_GFM_DELIM_RE` + data pipe-rows). For each table, append after it — **separated by a blank line (`\n\n`) so the block is a DISTINCT paragraph unit** (this is what makes the raw table and the `[table-rows]` block land in separate chunks, and keeps prose chunking byte-identical) — a derived block:
```
[table-rows]
<row-1 linearization>
<row-2 linearization>
[/table-rows]
```
Row linearization (markdown-KV, the evidence winner):
- If col-0 is a **row-label column** (col-0 cells non-numeric, header[0] empty/blank): each data row → `{cell0}: {header[1]}={cell1}, {header[2]}={cell2}, …`
- Else (no row-label column): each data row → `{header[0]}={cell0}, {header[1]}={cell1}, …`
- Strip footnote markers / collapse whitespace in cells; keep values verbatim (`$22.5 billion`).
- Prefix each block's rows with the nearest preceding heading as context: `[<section heading>] ` (contextual-retrieval lite) when available.
**Header-sanity gate:** skip a table entirely if its header row is all value-like (every header cell matches digit/`$`/`%` — promote `docling_tables._looks_like_value` → `core/text.py::looks_like_value`). Avoids emitting garbage from mis-structured headers (e.g. the segment table). Idempotent + deterministic (chunk_id stability).

Row linearization runs on **post-header-recovery** markdown (the Docling worker's `reattach_detached_table_headers` + `export_markdown_header_aware` already ran, so the GFM header is the corrected one). If re-attach DID fix a degenerate table, the header-sanity gate sees a valid header and linearizes it (correct); if not, the gate skips it. The gate re-checks the GFM header independently — it does not assume re-attach's outcome.

### THE 3-CHANNEL CONTRACT (the BLOCKER-1 fix — no double-indexing)
The `[table-rows]` block is the retrieval+answer payload; the raw GFM table is the human/source form. To avoid duplicating table content across the index (the chart-strip-class bug), each channel sees exactly ONE representation:

| Channel | Sees | Mechanism |
|---|---|---|
| **`.md` on disk** (source of truth, webui) | raw GFM table **+** `[table-rows]` block | `linearize_gfm_tables` appends the block after the table; both are real `.md` substrings (offsets valid) |
| **Chunks → dense embedding + answer node** | a large table → raw table and `[table-rows]` block in **SEPARATE chunks** (force-split → no within-chunk dilution); a small table → raw + `[table-rows]` may **co-locate in one chunk** (acceptable — a small table isn't truncation-diluted; the KV rows are additive co-located signal, and the values+labels already fit the answer budget) | normal chunking; the `[table-rows]` block is a normal chunkable unit (NOT budget-zeroed) so a large one forms its own chunk(s); rows are newline-separated so the `MAX_CHUNK_CHARS` force-split lands on row boundaries (rows stay whole). The `[table-rows]` block is NOT given the chart-block char-split exemption — it may char-split (unlike chart blocks). |
| **FTS / BM25 body** | `[table-rows]` only — the raw GFM table is **STRIPPED** | new `core/text.py::strip_superseded_gfm_tables(text)` removes a GFM table when immediately followed by a `[table-rows]` block; applied in `fts_store.upsert` alongside the existing chart strip. This is the inverse of the chart strip and is what prevents BM25 term-inflation (the KV rows carry the BM25 signal; the raw table doesn't double it). |

Net: no representation is indexed twice in the same channel. Dense has the table as two *separate* chunks (raw + KV); the KV chunk wins value queries, the raw-table chunk is harmless. BM25 sees only the KV rows. The `.md` keeps both for fidelity. Offsets stay valid (everything is a real `.md` substring — no transformed-text chunks). **This contract is the load-bearing decision and every row of the table above is pinned by a test.**

### `src/memex/core/text.py`
Add `looks_like_value(s)` (digit/`$`/`%` heuristic, promoted from `docling_tables`), `table_rows_spans(text)` (offsets of `[table-rows]…[/table-rows]`, **with the same orphan-opener/closer tolerance as `chart_extracted_spans`** so a force-split block's halves are still recognised), and `strip_superseded_gfm_tables(text)` (FTS helper above).

### Wiring — `src/memex/parse/pipeline.py`
Factor `_finalize_body(markdown) -> str` (applies `linearize_gfm_tables`) and call it on `conversion.markdown` in BOTH `_parse_with_docling` (after `_stitch_chart_extractions`) and `_parse_with_pymupdf`. **Bind the finalized result to a local and thread it to ALL consumers** — `body=`, `_bootstrap_ref(...)`, and the `markdown_bytes=` log/manifest lines — OR mutate `conversion.markdown` in place, so the content-hash + byte-count match what's written to disk. Engine-agnostic (both emit GFM the linearizer parses).

### `src/memex/index/chunker.py` + `src/memex/index/fts_store.py`
Chunker: `[table-rows]` is a normal chunkable unit (own chunk when sized, char-split on rows when oversized — explicitly NOT added to the chart-block char-split exemption). **Prose chunking byte-identical** (pinned). FTS: add `strip_superseded_gfm_tables` to the `upsert` strip pipeline (keep `[table-rows]`, drop the raw table it supersedes).

### Row-label-column detection (own it — GFM-text heuristic, not the `TableCell` one)
The linearizer parses GFM *text* (no `column_header` flags). Define: col-0 is a row-label column iff `header[0]` is empty/blank AND ≥1 data-row col-0 cell is non-value-like. Pin BOTH branches with tests: `| | A | B |` (empty corner → row-label layout) vs `| Metric | A | B |` (named first column → plain layout) — they produce different KV output. Pin the year-header cases too: `| Metric | 2024 | 2025 |` and `| | 2024 | 2025 |` must NOT be gate-skipped (only an *all*-value-like header is skipped; a non-value-like or empty col-0 keeps the table).

### Tests (`tests/unit/test_table_linearize.py` NEW + chunker/fts/embed regressions)
- GFM→KV rows (row-label + plain + named-first-column); section-context prefix; footnote/whitespace cleanup; idempotence/determinism (chunk_id stability).
- Header-sanity gate: value-like-header table skipped; year-header + empty-corner tables NOT skipped.
- **3-channel pins:** (a) FTS body of a table chunk = KV rows, raw GFM table stripped (`strip_superseded_gfm_tables`); (b) the raw table + `[table-rows]` land in SEPARATE chunks (not concatenated); (c) prose chunking byte-identical; (d) oversized `[table-rows]` char-splits on row boundaries with rows whole + `table_rows_spans` tolerates the orphaned halves; (e) `[table-rows]` does NOT get the chart-block char-exemption.
- **BM25-pollution regression** (mirror `test_chart_block_retrieval_isolation.py`): a linearized table row must not outrank a relevant prose chunk on a shared common term.
- Local gates: pytest all green, pyright 0/0, ruff.

### Phase-1 GPU acceptance
Re-parse the 10-K (default routing), confirm `[table-rows]` blocks appear under well-formed tables; pick a **well-formed** table deep-row query (e.g. an equity-award / director-comp / definitions row) and confirm it flips REF→ANS; re-index/enrich; re-resolve; eval HARD GATES hold (refusal_cf=1.0, 0 hallucinations, chart-content 09/10 ANS). Add flipped queries as ANS entries. (The segment table is a documented degenerate outlier — not the acceptance.)

---

## Phase 2 — structured table store + LangGraph text-to-SQL tool (the endgame)

**Under-specified here by design — Phase 2 gets its OWN detailed spec (`docs/specs/table-sql.md`) written + independently validated AFTER Phase 1 is gated green.** The spec review surfaced that Phase 2 is not buildable from a sketch because: (1) `agents/answering.py` is a fixed graph with **no tool-node pattern** — adding `query_tables` is a new node + routing + model-call schema + prompt, not "add a tool"; (2) the verify/compose contract is `chunk_id`→`Chunk`-based, but SQL result rows have **no `chunk_id`** — a distinct grounding path (verify the SQL output against the cited table's markdown-span text) must be designed; (3) the HARD-GATE risk is **confident-wrong SQL** (runs, returns a plausible but incorrect number) — the fallback must cover this, not just errors/empties. Direction (to detail in `table-sql.md`): regenerable per-vault `~/.memex/vault/.memex/tables.sqlite` (ADR-0003-consistent — rebuilt by `memex reindex`), a new answer-graph node gated to aggregation-shaped queries with a relevant table in retrieval, SQL-result grounding cited to the table's markdown span, and a confident-wrong containment story (cross-check the SQL number against a retrieved KV row before shipping; refuse on mismatch).

## Anti-scope
- No new `IndexSettings` field (module constants). Prose chunking byte-identical. Markdown vault stays source of truth (the SQLite store + the `[table-rows]` blocks are regenerable derived state).
- The 3-channel contract is fixed: no representation indexed twice per channel; FTS sees KV rows (raw table stripped); dense sees raw + KV as separate chunks.
- Don't attempt the fragile prose-label re-attach for degenerate tables (segment table stays a documented limitation; the re-attach already shipped is gated off for it).
- Phase 2 is not built from this spec — it gets `table-sql.md` first.
