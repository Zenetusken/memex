# Spec: Document Summarization

Status: shipped 2026-05-27. Decision record: [ADR-0008](../adr/0008-document-summarization.md).

The doc-type-aware, structured, **grounded** "summarize this document" path. A
standalone async orchestration (`agents/document_summarizer.py`) — NOT a
LangGraph; map-reduce is linear and grounding is reusable functions.

## Entry point

```python
async def summarize_document(
    doc_id: str, *,
    instruction: str | None = None,
    detail: SummaryDetail = "standard",       # "brief" | "standard" | "detailed"
    max_output_tokens: int = 2048,
    token_budget: int = 120_000,
) -> FinalResponse
```

Returns the same `FinalResponse` as the answering agent, plus the new optional
`sections: list[SectionSummary]`. A zero-grounded document refuses
(`answered=False`).

## Pipeline

1. **Load** — `FTSStore.chunks_for_document(doc_id)` returns the doc's stored
   chunks in reading order (`ORDER BY char_start`), reconstructing `heading_path`
   (split on `" > "`). This is the boundary-correct primitive: `agents/` may
   import `index/`, never `vault/` — so we read the *indexed* chunks, not the
   markdown body. No re-parse, no re-chunk.
2. **Group** — `_group_sections`: bucket chunks by deepest `heading_path` entry
   (fallback: document title), first-seen order → `[(section_title, [chunks])]`.
3. **Route** — `_classify_route` (v1): `short` if total chunk text ≤
   `_SHORT_DOC_CHARS` (14k) **or** ≤1 section; else `long`. Orthogonally,
   `is_tabular = len(_load_doc_tables(...)) >= _TABULAR_MIN_TABLES` (4) turns on the
   Key-figures pass (step 3b). A many-tiny-section doc (`_should_pack_sections`:
   ≥`_PACK_MIN_SECTIONS`, most under `_PACK_TINY_SECTION_CHARS`) is **deck-like** —
   the long route packs adjacent sections via `_pack_sections` (step 3c). (`scan`
   routes as `long` for now.)
3c. **Section-packing** (deck-like, long route) — `_pack_sections` greedily merges
   consecutive sections up to the fast-window budget so each MAP digests a
   substantive span (a slide deck's per-slide sections, or a bullet-heavy doc's
   subsections, instead of one thin digest each). Titled by the first section;
   reading order preserved; capped at `_MAX_SECTIONS`. A paper/standard with
   substantive sections is NOT packed (per-section digests preserved).
3b. **Key figures** (tabular, runs FIRST when `is_tabular`) — `_rank_tables` orders
   the doc's tables by `_table_salience` (PURE, no LLM: numeric density via
   `coerce_number` dominates, monetary/% + headline-section keyword break ties, a
   multiplicative width factor halves a framing-risky wide grid), so the headline
   data tables win the window budget. `_table_chunks` renders the top-ranked
   `StoredTable`s as synthetic bounded table-chunks (`chunk_id={doc_id}#tblN`,
   header-paired `col=cell` verbatim rows); `_key_figures_section` MAPs
   `summarize_tabular/v1` over them (bounded to the fast window) → a "Key figures"
   `SectionSummary`, GROUNDED against those same chunks (verbatim cell grounds;
   computed/fabricated drops — the Table-RAG row-verbatim boundary). It leads
   `section_summaries` so the figures head the summary; empty/ungrounded → skipped.
   Fail-open: no `tables.sqlite` → skipped.
4. **MAP** (`_map_section`) — render `summarize_section/v1` over the section's
   (bounded) chunks → `SectionSummary`. A failed call returns `(None, …)`; one bad
   section never sinks the summary.
5. **GROUND** (`_ground_points`) — reuse `verify_grounding/v2` **verbatim**: wrap
   the section's `key_points` (which are `CitedClaim`s) in a `DraftAnswer`, run
   the verify prompt with a `create_model(__base__=VerificationResult)` bounded to
   `n=len(key_points)`, keep only the confirmed-grounded indices. Conservative —
   missing index → dropped.
6. **REDUCE** (`_reduce`) — render `summarize_reduce/v1` over the grounded section
   digests → `DocAbstract.abstract`. The short **non-tabular** route skips this
   (the single section's digest IS the abstract); `long` OR `tabular` reduces so the
   abstract reflects both the prose digests and the Key-figures section.
7. **Compose** — `FinalResponse(summary=abstract, claims=grounded doc points
   [≤12], sections=…, used_chunks, wikilinks via core.wikilinks.format_wikilink,
   artifact_scope_doc_ids=[doc_id])`. Empty grounded set → refusal (sections still
   surfaced for transparency, key_points dropped).

## Schemas (`agents/answering.py`)

```
SectionSummary { section_title: str(max_length=200); digest: str(max_length=600);
                 key_points: list[CitedClaim](max_length=8) }   # CitedClaim: claim≤300, chunk_id≤80, confidence
DocAbstract    { abstract: str(max_length=800) }
FinalResponse  { … ; sections: list[SectionSummary] = [] }      # new field; [] on the answer path
```

All lists `maxItems`-bounded; all strings short + bounded. **xgrammar enforces
list bounds, NOT string `maxLength`** (the 2026-05-27 baseline) — hence bounded
lists of short strings, never one big free-form string.

## The mode-independence guarantee (the "baseline rule")

`summarize_document` never reads `co_residence_mode`. Quality is mode-independent
**by construction**: every LLM call is bounded to fit the SMALLEST supported
window (`fast`/6144), so the call's input is identical in `fast` and `full`.

| call | input bound | output reservation |
|---|---|---|
| MAP `summarize_section` | `_bound_section_chunks` → `_MAX_SECTION_INPUT_CHARS` (12k chars ≈ 3k tok) | `max_output_tokens` (2048) |
| GROUND `verify_grounding` | the SAME bounded section chunks | `_VERIFY_MAX_TOKENS` (768) |
| REDUCE `summarize_reduce` | digests capped to `_REDUCE_MAX_SECTIONS` (24) | `min(max_output_tokens, _REDUCE_MAX_TOKENS=1024)` |

`_bound_section_chunks` selects a section's chunks up to the char budget; the
per-group loop uses `_split_section_into_batches` (its generalization) so a section
LARGER than one window is split into consecutive window-sized batches — each
MAPped + GROUNDed into its own `SectionSummary` (`(part k)` suffix when >1) so no
content is truncated away (every chunk lands in one batch). Each batch is fed
identically to MAP and GROUND (mode-independent). A section that fits is one batch
(unchanged). The abstract REDUCEs whenever there's >1 summary (incl. a split
single-section doc); a lone summary uses its digest directly.

Validated live (GTE arXiv, `--token-budget 20000`): `fast`(6144) and `full`(24576)
both → 9 sections, 12 grounded claims, faithful abstract, ~identical latency, zero
window-overflow.

## The detail knob

`_DETAIL_GUIDANCE[detail]` maps to `{abstract, digest}` length phrases threaded
into the prompts. Short route maps with the *abstract* phrase (the single pass is
the whole-doc summary); long route maps with the tighter *digest* phrase and
REDUCE produces the abstract.

| detail | abstract | digest |
|---|---|---|
| brief | 1-2 sentences | 1 concise sentence |
| standard | 2-4 sentences | 1-3 sentences |
| detailed | a thorough paragraph of 5-8 sentences | 2-4 sentences |

## Surfaces

- **CLI** — `memex summarize <doc_id> [--detail b/s/d] [--token-budget N]
  [--instruction …] [--max-tokens N]`.
- **MCP** — `summarize(doc_id, instruction=None, detail="standard")`
  (`detail` validated → `SummaryDetail`, unknown falls back to `standard`).
- **webui** — a Summarize `<form>` on the document view (detail `<select>` +
  button) → `POST /documents/{id}/summarize` → renders `_summary.html` (the
  abstract in `.ans-answer`, grounded key-points as `.claim` cards with `.conf-*`
  chips + source ids, a collapsible `.summary-sections` per-section breakdown,
  Sources via `render_wikilink`, the audit footer). MemexError → 503 banner.

## HARD-gate safety

Grounding reuses `verify_grounding/v2` (the same machinery the answer path's
gate uses); only confirmed-grounded points are shipped; a zero-grounded result
refuses. The summarizer can only narrow what it emits — never assert an
ungrounded claim — so it preserves the no-hallucination invariant by
construction.

## Testing

- `tests/unit/test_doc_type.py` — `_group_sections` (heading grouping + doc-title
  fallback), `_classify_route` (short/long), `_should_pack_sections` (deck detection)
  + `_pack_sections` (merge-to-budget / title-by-first / cap), `_bound_section_chunks` (fast-window
  budget cap / always-keep-one / keep-all-when-small), `_render_table`
  (header-paired verbatim cells / row-cap / ragged-row fallback) + `_table_chunks`
  (id format / section / cap) + `_table_salience`/`_rank_tables` (numeric>text,
  fragment→0, keyword tiebreak, wide-grid multiplicative penalty, stable order).
- `tests/integration/test_document_summarizer.py` — faked `FTSStore` + (for the
  tabular route) faked `TableStore` + schema-dispatched `complete_structured`:
  long-route map-reduce, short-route single-pass (no REDUCE), grounding drops
  ungrounded → zero-grounded refuses, no-indexed-chunks refusal, the detail knob
  threading, token-budget early stop, **tabular** (Key-figures section leads +
  cites a synthetic table chunk, an unsupported figure is dropped, a sub-threshold
  doc skips the pass).

## Deferred

- `tabular` route **shipped** (key figures from `tables.sqlite`, cited not copied —
  step 3b) **with figure-salience ranking** (`_table_salience`). Residual: figure
  *framing* on a complex/mis-bounded table can mis-attribute a grounded value's
  metric/period (table-parse quality, not a hallucination). `deck` (section-packing,
  step 3c) shipped. `scan` (over VLM text) route remains.
- ~~Section sub-splitting~~ **SHIPPED** (`_split_section_into_batches`; a huge
  section spans window-sized batch-parts, no content dropped; validated live on a
  synthetic 18-chunk doc → 3 parts covering subsystems 1-36).
- ~~A grounded-summary eval suite~~ **SHIPPED** (`memex eval-summary` →
  `eval/runner.py::run_summary_eval`; cases at `tests/eval-data/summary/queries.json`;
  scorers `mention_recall` (soft) + `absent_assertion_violations` (the no-leak HARD
  gate); baseline 2026-05-27: 4 cases, 0 hallucinations, summarize_correct 4/4,
  mean_recall 1.0). Section sub-splitting + the `scan` route remain.
