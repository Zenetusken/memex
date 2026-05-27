# ADR-0008: Document Summarization (doc-type-aware, structured, grounded)

- **Status**: Accepted
- **Date**: 2026-05-27
- **Deciders**: Memex core team
- **Tags**: agents, summarization, grounding, models, architecture

## Context

"Summarize this document" is the staple LLM request Memex did not yet answer
well. ADR-0007 shipped `full` mode (a ~24,576-token orchestrator window) and a
free-form baseline (`agents/synthesize.py`) as the first consumer of it. That
baseline established two hard facts that drive this ADR:

1. **vLLM guided-JSON enforces list `maxItems` but NOT string `maxLength`.** A
   schema with one big free-form `summary: str(max_length=N)` does not stop the
   model at N — it runs to `max_tokens`, truncates mid-emission, and the JSON
   fails to parse. (`SufficiencyAssessment.reason` taught us the same lesson; see
   `src/memex/CLAUDE.md`.) So a reliable structured summary must be `maxItems`-
   bounded lists of SHORT bounded strings, never one unbounded string.
2. **An 8B model regurgitates on a large/dense document** — it copies tables and
   sentences instead of abstracting — overrunning the window. Whole-document
   single-shot summary is viable only for *small* prose; non-trivial docs need
   map-reduce, not one giant context.

A third force is the governing product rule the user set: **quality is never
compromised, and is independent of the co-residence mode.** A `fast`-mode and a
`full`-mode summary of the *same* document must be the same-quality grounded
summary; the mode may change speed/capacity, never faithfulness or coverage.

## Decision Drivers

- Extend Memex's no-hallucination **HARD gate** (refusal_cf=1.0, zero
  hallucinations) to summaries — every emitted point grounded or dropped; a
  zero-grounded result refuses.
- Reliability under the `maxItems`-not-`maxLength` constraint (fact 1).
- The baseline rule: identical quality in `fast` and `full` (mode-independence).
- Reuse, not reinvention: the existing grounding machinery (`verify`), the
  `FinalResponse` contract, the wikilink derivation, the module boundaries
  (`agents/ → index/`, never `agents/ → vault/`).

## Decisions

### 1. A standalone async orchestration, not a LangGraph

`agents/document_summarizer.py::summarize_document(doc_id, *, instruction=None,
detail="standard", max_output_tokens=2048, token_budget=120_000) -> FinalResponse`
mirrors `synthesize_documents` — map-reduce is linear and grounding is reusable
functions, so a graph would add ceremony without benefit. It returns the same
`FinalResponse` the answering agent does (a new optional `sections:
list[SectionSummary] = []` field carries the per-section breakdown;
HARD-gate-neutral, the answer path leaves it empty).

### 2. Doc-type-aware MAP → GROUND → REDUCE

1. **Load** the document's stored chunks in reading order via
   `FTSStore.chunks_for_document(doc_id)` — the boundary-correct primitive
   (`agents/` may import `index/`, not `vault/`; no re-parse, no re-chunk).
2. **Group** chunks into heading-sections (deepest `heading_path`, else doc
   title), first-seen order.
3. **Route** (`_classify_route`, v1): `short` (≤ ~14k chars total OR ≤1 section →
   one structured pass) vs `long` (map-reduce). Orthogonally, a doc with ≥
   `_TABULAR_MIN_TABLES` stored tables is **`tabular`** and additionally gets a
   grounded Key-figures pass (§7, shipped); a many-tiny-section doc is **deck-like**
   and packs sections (§8, shipped). `scan` routes as `long` for now — the router +
   schema accommodate it; specialised later.
4. **MAP** each section → a `SectionSummary` (digest + cited key-points), bounded
   structured output.
5. **GROUND** each section's key-points by **reusing `verify_grounding/v2`
   exactly** (the section's key-points ARE `CitedClaim`s; wrap in a `DraftAnswer`,
   run the same prompt, keep only confirmed indices). Conservative: a point
   survives only if explicitly grounded.
6. **REDUCE** the grounded section digests → a whole-doc abstract (short route:
   the single digest IS the abstract — no second call).
7. **Compose** → `FinalResponse`: `summary`=abstract, `claims`=grounded
   doc-level key-points, `sections`=per-section digests, `wikilinks`/`used_chunks`
   for Sources, `artifact_scope_doc_ids=[doc_id]`. **Zero grounded points →
   `answered=False` refusal.**

### 3. Bounded-list schemas (fact 1 made structural)

```
SectionSummary { section_title: str(max_length=200); digest: str(max_length=600);
                 key_points: list[CitedClaim](max_length=8) }
DocAbstract    { abstract: str(max_length=800) }
```

Every list is `maxItems`-bounded (xgrammar-enforced) and every string is short +
bounded, so no call can run away the way the free-form baseline did. The
per-section verification reuses the `create_model(__base__=VerificationResult)`
index-list bounding pattern from `answering.verify`.

### 4. Mode-independence BY CONSTRUCTION, not by luck

The strategy is a function of the **document** (size/type), never the mode — so
`summarize_document` never reads `co_residence_mode`. But the *window* differs by
mode (`fast`=6144, `full`=24576), so a call whose prompt+output exceeds the
smaller window would succeed in `full` and fail in `fast` → mode-dependent
quality. We close that hole structurally: **every LLM call is bounded to fit the
SMALLEST supported window (fast/6144).**

- Each MAP/GROUND call sees at most `_MAX_SECTION_INPUT_CHARS` (12k ≈ 3k tokens)
  of chunk text, selected ONCE per section and fed to both calls
  (`_bound_section_chunks`) — identical input in `fast` and `full`.
- The GROUND (`_VERIFY_MAX_TOKENS=768`) and REDUCE (`_REDUCE_MAX_TOKENS=1024`,
  digests capped to `_REDUCE_MAX_SECTIONS=24`) reservations keep prompt+output
  inside 6144.

A section larger than the budget is **sub-split** across multiple window-sized
MAP calls (`_split_section_into_batches`, shipped 2026-05-27 — see §9) so none of
its content is dropped, while every call stays mode-independent. Validated live:
the same doc in `fast` and `full` produced the same 9 sections / 12 grounded
claims / faithful abstract, no window-overflow errors.

### 5. A tunable detail knob; a token budget

`detail ∈ {brief, standard, detailed}` threads natural-language length guidance
into the MAP/REDUCE prompts (`_DETAIL_GUIDANCE`) — the *requested* verbosity
within the hard schema bounds (the user's fine-tuning variable; the "2-4
sentences" default is `standard`). `token_budget` caps the whole map-reduce's
total token traffic; a typical document completes within the default (120k,
ultimately bounded by `_MAX_SECTIONS=40`), a pathologically long one stops early
(a v1 limit, tunable per call / via `memex summarize --token-budget`).

### 6. Surfaces

CLI `memex summarize <doc> [--detail --token-budget --instruction]`; MCP
`summarize(doc_id, instruction?, detail?)`; webui a **Summarize** action on the
document view (`POST /documents/{id}/summarize` → `_summary.html`: the abstract,
grounded key-points with confidence chips + source ids, a collapsible
per-section breakdown, Sources). Quality is identical across all three and across
modes (§4).

### 7. The tabular route — a grounded "Key figures" pass (shipped 2026-05-27)

The first doc-type specialisation. A data-heavy document's headline numbers
(revenues, totals, rates, named maxima) live in tables that the prose map-reduce
either never reaches (deep in reading order, starved by the token budget) or
describes rather than quotes — so a generic-`long` 10-K summary surfaced the CEO
letter and **zero financial figures** (validated live). The `tabular` route adds a
dedicated pass that runs **first** (the figures are the point; the prose budget
must not starve them):

- **Detect**: `_load_doc_tables` reads the doc's `tables.sqlite` (`TableStore`,
  fail-open to `[]` — a missing/corrupt store just skips the route, fully
  backward-compatible); `is_tabular = len(tables) >= _TABULAR_MIN_TABLES`.
- **Render** each `StoredTable` as a synthetic, bounded **table-chunk**
  (`_table_chunks`, `chunk_id={doc_id}#tblN`): header-paired `col=cell` rows
  (verbatim cells, the Phase-1 linearization shape), row- and char-capped.
- **MAP** `prompts/summarize_tabular/v1` over the table-chunks (bounded to the
  fast window like every call) → a "Key figures" `SectionSummary` whose key-points
  quote figures VERBATIM (the prompt forbids computing/summing).
- **GROUND** those figures against the same table-chunks via `verify_grounding/v2`
  — a verbatim cell value grounds; a computed/fabricated one drops. This is the
  Table-RAG **row-verbatim fabrication boundary** (ADR / `agents/table_sql.py`)
  reused: no new grounding path, HARD-gate-safe by construction.

The Key-figures section leads `sections` + the doc-level `claims`; its synthetic
table-chunks join `used_chunks` so Sources/wikilinks resolve.

**Figure-salience (shipped 2026-05-27).** Tables are **ranked** before the window
budget is applied (`_rank_tables` / `_table_salience`, PURE + deterministic — no
LLM), so the headline data tables win on a many-table doc rather than whatever
comes first in reading order. The score: numeric density (via `coerce_number`)
dominates, a monetary/percent signal + a headline-section keyword break ties, and a
**multiplicative width factor halves a wide grid** — a performance graph or a
multi-year/scenario projection is framing-risky (the MAP can attach a value to the
wrong column/period), while a narrow `metric: value` table yields unambiguous
figures. Validated live on the 10-K: salience surfaced the income-statement /
operating figures (revenue, operating income, other income, deferred revenue,
stock-based comp) the document-order v1 missed entirely.

**Residual (not a hallucination).** Grounding guarantees each value is a verbatim
cell; it does NOT guarantee the metric *label* the MAP attaches is the right one
for a complex/mis-bounded table — so a figure can be mis-framed (wrong period or
metric) while still being a real cell. The width factor + the prompt's
period/column rule cut the worst of this, but the deeper fix is table-PARSING
quality (Docling's 10-K bounding), a separate concern.

### 8. The deck route — section-packing (shipped 2026-05-27)

A slide deck makes one tiny heading-section per slide, so the per-section MAP
produces thin digests that just echo the slide title (validated: the CUDA deck's
"introduces new CUDA features", "efficiency is a key consideration"). The deck
route is realized as **section-packing**: when `_should_pack_sections` finds many
sections of which most are slide-sized (`≥ _PACK_MIN_SECTIONS` and `≥
_PACK_TINY_FRACTION` under `_PACK_TINY_SECTION_CHARS`), the long route replaces
per-heading grouping with `_pack_sections` — greedily merging adjacent sections up
to the fast-window budget so each MAP call digests a substantive span (the VLM
transcribes diagrams inline, so the figure text rides along → figure-aware for
free, no separate marker). It's **doc-type-agnostic**: it fires on slide decks AND
bullet-heavy technical docs (both have thin sections) while a paper/standard with
substantive sections keeps its own per-section digests (calibrated on real docs —
CUDA deck 89% tiny → pack, GUIDELINES 77% → pack, GTE paper 31% / NIST 21% → no).
Validated live: the CUDA deck's shallow per-slide digests became 6 substantive
thematic groups. HARD-gate-neutral (grouping only — grounding/refusal unchanged).

### 9. Section sub-splitting (shipped 2026-05-27)

The flip side of packing: a section LARGER than one window. The per-group loop used
to call `_bound_section_chunks`, which kept only the first window-worth of chunks
and **truncated the rest — silently dropping that content** from the summary.
`_split_section_into_batches` instead splits the section into consecutive batches
that each fit the fast window (every chunk lands in exactly one batch — no content
dropped), and each batch is MAPped + GROUNDed into its own `SectionSummary`
(a multi-batch section is suffixed `(part k)`). A section that already fits is one
batch (unchanged). It applies to BOTH routes: the long route's per-section groups
AND the short route's single whole-doc group (which also truncated before). The
abstract now REDUCEs whenever there's >1 summary (incl. a split single-section doc);
a lone summary still uses its digest directly. Bounded by `token_budget` +
`_MAX_SECTIONS`. **Validated live** on a synthetic 18-chunk single-section doc → 3
batch-parts; the late content (subsystems 25-36, exclusive to batch 3, chunk 17)
was summarized in "part 3" where the pre-split route would have dropped everything
past batch 1. Mode-independent (each batch fits fast) + HARD-gate-neutral (grouping
only). Pinned by `test_split_section_into_batches_*` + the no-content-dropped
integration test (a late-batch chunk is cited).

## Consequences

### Positive

- "Summarize this doc" now produces a structured, **grounded** summary that
  refuses rather than fabricate — the HARD gate, extended to summaries.
- Reliable under guided-JSON (no runaway truncation) by construction.
- Mode-independent quality by construction (§4) — the baseline rule holds, proven
  live in both modes.
- Reuses the verify machinery + `FinalResponse` + wikilink derivation — no new
  grounding path, no new agent coupling (`agents/ → index/` only).

### Negative / Trade-offs

- A section/doc with more total content than the budget allows still stops early
  (`token_budget` / `_MAX_SECTIONS`) — but within those bounds, sub-splitting (§9)
  now covers a large section in full rather than truncating it.
- A long document is heavier than a single RAG answer (sequential per-section
  MAP+GROUND); the token budget bounds it but a very long doc is partial.
- `scan` routes as generic `long` until specialised; the `tabular` (§7, with
  figure-salience) and `deck`/section-packing (§8) routes are shipped. The tabular
  route's figure *framing* on a complex/mis-bounded table can mis-attribute a
  (grounded) value's metric/period — bounded by table-parse quality, not a
  hallucination.

### Neutral

- `full` mode is no longer *required* for a good summary (the bounding makes
  `fast` equally faithful); `full`'s larger window simply allows the eventual
  short-route single-shot on a bigger small-doc.

## Expanding

- **Doc-type routes**: `tabular` (key figures from `tables.sqlite`, cited not
  copied, **with figure-salience ranking**) **shipped** (§7); `deck` (section-packing
  for tiny-sectioned docs) **shipped** (§8); `scan` → over VLM text remains. Each
  slots into `_classify_route` + grouping/MAP — the GROUND/REDUCE/compose spine is
  unchanged. The tabular route's next deepening is **figure-framing** robustness on
  complex/mis-bounded tables (better table parsing + label-attribution).
- **Section sub-splitting** (no content dropped on a huge section) **shipped** (§9,
  `_split_section_into_batches`).
- This is the structured-summary **capability tier** ADR-0007 §"Expanding
  horizontally" anticipated: it layers on the same resource postures rather than
  adding a resource bundle.

## Revisit When

- A doc-type route is specialised (fold its MAP contract in here).
- (Section sub-splitting landed 2026-05-27 — §9.)
- The grounded-summary eval suite (`memex eval-summary`, shipped 2026-05-27 —
  `eval/runner.py::run_summary_eval`, `tests/eval-data/summary/queries.json`,
  HARD gates `hallucination_count==0` ∧ `summarize_correct_count==case_count`)
  grows or its baseline shifts.

## References

- ADR-0007 (co-residence modes; `full` window; the anticipated structured-summary
  capability tier), ADR-0001 (vLLM), ADR-0003 (markdown vault / derived state),
  ADR-0004 (observability; the correlation_id threads the map-reduce).
- `docs/specs/document-summarization.md` (the implementation spec).
- `src/memex/agents/document_summarizer.py`, `prompts/summarize_section/v1.md`,
  `prompts/summarize_reduce/v1.md`, `webui/templates/_summary.html`.
