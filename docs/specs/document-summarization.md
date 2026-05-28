# Spec: Document Summarization

Status: shipped 2026-05-27; `report` multi-paragraph mode + granularity tuning + the
cross-paragraph dedup gate added 2026-05-28. Decision records:
[ADR-0008](../adr/0008-document-summarization.md) (the grounded map-reduce),
[ADR-0009](../adr/0009-remove-free-form-synthesis-baseline.md) (output is a bounded
sentence-LIST, not a free-form string), [ADR-0010](../adr/0010-batched-reduce-report-mode.md)
(the `report` hierarchical reduce + coherence/granularity refinements).

The doc-type-aware, structured, **grounded** "summarize this document" path. A
standalone async orchestration (`agents/document_summarizer.py`) — NOT a
LangGraph; map-reduce is linear and grounding is reusable functions.

## Entry point

```python
async def summarize_document(
    doc_id: str, *,
    instruction: str | None = None,
    detail: SummaryDetail = "standard",       # "brief" | "standard" | "detailed" | "report"
    max_output_tokens: int = 2048,
    token_budget: int = 120_000,
    correlation_id: str | None = None,        # opt-in, threads logs/trace + drives the webui progress hook
    on_phase: Callable[[str], None] | None = None,  # observe-only phase sink (raises swallowed)
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
6. **REDUCE** (`_reduce`) — render `summarize_reduce/v2` (branched on `preceding`:
   OPENING keeps the doc thesis + metadata-suppression, CONTINUATIONS add only their
   sections' new specifics) over the grounded section digests → a `DocAbstract`
   (returned whole so the report route can dedup at sentence granularity; non-report
   callers read `.abstract`). The short **non-tabular** route skips REDUCE (the single
   section's digest IS the abstract); `long` OR `tabular` reduces so the abstract
   reflects both the prose digests and the Key-figures section. **`report` detail
   switches to the HIERARCHICAL `_reduce_report`** (see "Report mode" below): one
   bounded paragraph per planned section-group, stitched deterministically, with a
   cross-paragraph dedup gate — NO final whole-output pass.
7. **Compose** — `FinalResponse(summary=abstract, claims=grounded doc points
   [≤12], sections=…, used_chunks, wikilinks via core.wikilinks.format_wikilink,
   artifact_scope_doc_ids=[doc_id])`. Empty grounded set → refusal (sections still
   surfaced for transparency, key_points dropped).

## Schemas (`agents/answering.py`)

```
SectionSummary  { section_title: str(max_length=200); digest: str(max_length=600);
                  key_points: list[CitedClaim](max_length=8) }  # CitedClaim: claim≤300, chunk_id≤80, confidence
DocAbstract     { sentences: list[str(max_length=300)](max_length=8) }  # ADR-0009; .abstract joins them
ReportStructure { paragraph_starts: list[int](max_length=16) }  # report planner: section indices that START a paragraph
ReportConfidence{ overall; embedding; lexical; per_paragraph: list[float] }  # report; inform-only
FinalResponse   { … ; sections: list[SectionSummary] = [];
                  report_confidence: ReportConfidence | None = None }  # new; [] / None on the answer path
```

All lists `maxItems`-bounded; all strings short + bounded. **Correction (2026-05-28,
probe-verified): the pinned stack DOES enforce string `max_length` too** (the earlier
"enforces maxItems but NOT maxLength" note was stale — see ADR-0009). The win of a
bounded sentence-LIST over one free-form string is **natural-boundaries + count-bound +
grounding**: each sentence ends at a real boundary so the joined `.abstract` can never
force-close MID-WORD (the "policyEn" failure), `maxItems` bounds the COUNT, and a list
invites synthesis over regurgitation.

## The mode-independence guarantee (the "baseline rule")

`summarize_document` never reads `co_residence_mode`. Quality is mode-independent
**by construction**: every LLM call is bounded to fit the SMALLEST supported
window (`fast`/6144), so the call's input is identical in `fast` and `full`.

| call | input bound | output reservation |
|---|---|---|
| MAP `summarize_section` | `_bound_section_chunks` → `_MAX_SECTION_INPUT_CHARS` (10k RENDERED chars) | `max_output_tokens` (2048) |
| GROUND `verify_grounding` | the SAME bounded section chunks | `_VERIFY_MAX_TOKENS` (768) |
| REDUCE `summarize_reduce` | digests capped to `_REDUCE_MAX_SECTIONS` (24) | `min(max_output_tokens, _REDUCE_MAX_TOKENS=1024)` |

**The budget counts the RENDERED size, not just text** (`_chunk_budget_chars`, the
2026-05-28 overflow fix): a chunk costs `min(len(text),1800)` PLUS its per-chunk prompt
wrapper (`chunk_id` + repeated doc title + formatting, ~120-150 chars). A packed deck
holds 20-59 tiny slide-chunks per group, so a text-ONLY budget under-counted by
thousands of chars — a "9.5k-text" group rendered to ~18k chars / 6.6k tokens, the MAP
400'd, and the section was silently DROPPED. The 10k ceiling is the RENDERED budget that
clears the 6,144 window with the 2,048-token output reservation at the densest realistic
~2.7 chars/token (verified 0 overflows). `_bound_section_chunks` selects a section's
chunks up to that budget; the per-group loop uses `_split_section_into_batches` (its
generalization) so a section LARGER than one window is split into consecutive
window-sized batches — each
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
| report | one bounded paragraph PER section-group (hierarchical reduce, `_reduce_report`) | rich, like `detailed` |

## Report mode (`detail="report"`, ADR-0010)

A multi-paragraph "report" body via a HIERARCHICAL reduce (`_reduce_report`), the safe
inverse of the removed free-form synthesis (ADR-0009):

1. **Plan** (`_plan_report_structure`) — the model picks where paragraph breaks fall over
   the ordered grounded section digests (`ReportStructure.paragraph_starts` = section
   indices that START a paragraph; boundary-selection, so it can't drop/dup a section).
   Guardrails: in-range/dedup/force-0; a size cap (`_REPORT_MAX_SECTIONS_PER_PARAGRAPH`=6);
   **COALESCE** — merge runs up to `report_coalesce_target` sections (fixes the 8B's
   over-splitting); a paragraph **FLOOR** (split a lone ≥4-section group in two); and a
   fallback to mechanical `_REPORT_SECTIONS_PER_BATCH` (4) batching on a trivial/failed
   plan. All logged (`report.plan*`).
2. **Reduce each group** to ONE bounded `DocAbstract` paragraph (reusing `_reduce`), in
   SEQUENCE with ROLLING `preceding` context + a position-aware `scope_note` (OPEN /
   middle / CLOSE arc) so the paragraphs read as one narrative, not disjoint blocks.
3. **Dedup** (`_dedup_sentences`) — DETERMINISTIC cross-paragraph gate: the branched
   prompt + rolling context still don't stop the 8B re-covering a TOPIC when its sections
   overlap a prior paragraph's (topic-aware planning can't fix it on decks' coarse/noisy
   recovered headings). A sentence whose content tokens are >`_REPORT_DEDUP_THRESHOLD`
   (0.7) covered by a SINGLE earlier kept sentence (overlap-precision) is dropped; an
   emptied paragraph vanishes. LEXICAL → always-on + reproducible (the confidence embedder
   degrades under VRAM pressure, so it can't be the dedup signal). HARD-gate-safe: only
   REMOVES already-grounded prose. Logged `report.dedup`.
4. **Stitch** deterministically with blank lines (no final whole-output model pass — that
   would reintroduce the free-form trap), then **score** (`_score_report_confidence` →
   `ReportConfidence`, hybrid lexical+embedding per paragraph vs its source digests;
   inform-only, surfaced + logged `report.confidence`, NEVER a gate).

**Granularity knobs** (`AgentsSettings`, env-overridable, report-mode only): `report_pack_chars`
(deck-packing group size, default 4,000 — clamped to `_MAX_SECTION_INPUT_CHARS`) and
`report_coalesce_target` (paragraph fullness, default 2). The defaults are the TUNED
winner from a sweep over the **report-STRUCTURE validator** (`scripts/report_structure_audit.py`
+ corpus `tests/eval-data/report-structure/`, baseline `baseline.json`) which measures
paragraphs / confidence / distinctness (1 − pairwise content-Jaccard) / unique-openers /
must-not-assert leaks — the structure dimensions `eval-summary` doesn't. The knobs hit
DIFFERENT deck profiles: `pack` only bites when `_should_pack_sections` fires (tiny-section
decks); `coalesce` drives substantial-section decks.

The webui splits the `\n\n` body into one `<p>` per paragraph inside one `.ans-answer`.

## Surfaces

- **CLI** — `memex summarize <doc_id> [--detail b/s/d/report] [--token-budget N]
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
  fragment→0, keyword tiebreak, wide-grid multiplicative penalty, stable order) +
  **`_dedup_sentences`** (drops a near-restatement, keeps a sentence sharing only common
  terms, exact-dup dropped + ledger integrity).
- `tests/integration/test_document_summarizer.py` — faked `FTSStore` + (for the
  tabular route) faked `TableStore` + schema-dispatched `complete_structured`:
  long-route map-reduce, short-route single-pass (no REDUCE), grounding drops
  ungrounded → zero-grounded refuses, no-indexed-chunks refusal, the detail knob
  threading, token-budget early stop, **tabular** (Key-figures section leads +
  cites a synthetic table chunk, an unsupported figure is dropped, a sub-threshold
  doc skips the pass), **report** (multi-paragraph body + arc + rolling context +
  adaptive plan / coalesce / floor; a repeated sentence kept ONCE across paragraphs
  while distinct points survive; section sub-split).
- `scripts/report_structure_audit.py` — the report-STRUCTURE validator (granularity
  tuning, NOT a pytest gate): runs `summarize_document(detail="report")` over
  `tests/eval-data/report-structure/decks.json` → paragraphs / confidence / distinctness /
  openers / leaks; baseline `baseline.json` (the shipped H3 config). Sweep via
  `MEMEX_AGENTS__REPORT_PACK_CHARS` / `MEMEX_AGENTS__REPORT_COALESCE_TARGET`.

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
  gate); baseline 2026-05-28: **6 cases (incl. a `report`-mode drift case + a refuse
  case), 0 hallucinations, summarize_correct 6/6, mean_recall 1.0** — re-confirmed with
  the `report` defaults + dedup gate). The `report` route + granularity tuning + the
  cross-paragraph dedup gate shipped 2026-05-28 (ADR-0010). Only the `scan` route remains.
