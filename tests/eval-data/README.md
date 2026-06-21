# Memex eval query sets

This directory holds the **versioned query sets** that `memex eval` consumes to score the retrieval/answering pipeline. One subdirectory per category from [`docs/eval-corpus-plan.md`](../../docs/eval-corpus-plan.md); each subdirectory holds a `queries.json` plus optional supporting notes.

```
tests/eval-data/
├── README.md          ← this file
├── slide-decks/       ← CUDA deck (33 q; +chart-content blocks)
├── annual-report/     ← NVIDIA FY26 10-K (16 q; +table-SQL)
├── chart-types/       ← "which chart" dataviz guide (chart-content)
├── french-course/     ← CR350 French security course (8 q)
├── cr350-multidoc/    ← 7-lecture cross-doc disambiguation (15 q)
├── nist-zero-trust/      ← NIST SP 800-207 security standard (18 q; 2026-05-25)
├── scientific-gte/       ← GTE paper arXiv 2308.03281 (18 q; 2026-05-25)
├── technical-guidelines/ ← Memex docs/GUIDELINES.md, rendered (18 q; 2026-05-25)
├── forms-w9/             ← IRS Form W-9 (18 q; 2026-05-25)
├── cr350-diagrams/       ← VLM-transcribed network/security DIAGRAMS, CR350 Cours 6 (17 q; 2026-05-25; first corpus to exercise the VLM escalation path)
├── ccna-multidoc/        ← CCNA SRWE+ENSA .pptx deck library (8 q; 2026-05-26; Office→PDF + VLM diagrams)
├── summary/              ← grounded document-summary eval (memex eval-summary; 2026-05-27)
├── handwritten/          ← HANDWRITTEN C++ note, cs-notes-1 (10 q; 2026-05-27; first corpus to exercise the scan→VLM parse route)
├── linux-fundamentals/   ← Linux Essentials course, 16 real born-digital PDFs (18 q; 2026-06-07; NEW DOMAIN — the `modern-printed` real-doc category; refusal_cf=1.0 N=3)
└── legal-statutes/       ← FOIA §552 + Privacy Act §552a, 2 cross-referencing US statutes (29 q; 2026-06-20; legal/regulatory — cross-reference + defined-term stress; N=2 byte-stable; refusal_cf=1.0. Surfaced 4 answerable false refusals; an empirics-first probe (`scripts/parse_fragmentation_probe.py`) REFUTED the parse-fragmentation guess → 3 are gate over-refusals on visible top-ranked values (ADR-0022 class) + 1 truncation-horizon (foia-08, value past the ~1800-char budget); see queries.json `_baseline_2026_06_20`)
```

Each subdirectory holds a `queries.json` (+ optional notes). Source PDFs are
**not** committed — only the query sets are (eval material stays local).

Run outputs land under `tests/eval-results/` (gitignored — these are timestamped + regenerable and shouldn't be versioned).

## Schema

Each `queries.json` is consumed by `src/memex/eval/runner.py::run_eval`. Shape:

```json
{
  "queries": [
    {
      "qid": "slide-decks-01",
      "question": "What is the energy cost of FP16 matrix multiplication relative to FP32?",
      "relevant_chunk_ids": ["2f96ae1c-...#3a6c6789e8"],
      "should_refuse": false
    },
    {
      "qid": "slide-decks-08",
      "question": "In what year did NVIDIA acquire Mellanox?",
      "relevant_chunk_ids": [],
      "should_refuse": true
    }
  ]
}
```

| Field | Type | Notes |
|---|---|---|
| `qid` | string | Unique identifier; prefix with the category (e.g. `slide-decks-01`) so qids stay globally unique when categories combine. |
| `question` | string | The natural-language question. The agent only sees this. |
| `relevant_chunk_ids` | string[] | Chunk IDs (`{doc_id}#{hash}`) that contain the answer. Used for the citation-precision metric. Empty list for `should_refuse: true` queries. |
| `should_refuse` | bool | `true` when no chunk in the vault answers the question — exercises the refusal path. |

Fields with a leading underscore (`_description`, `_expected_answer`, `_note`, etc.) are tolerated as documentation and ignored by the loader. Use them to capture the human-readable answer + any caveats.

## Labelling rules (the rigorous version)

These rules were arrived at after the first rigorous sweep exposed weaknesses in the bootstrap:

### `relevant_chunk_ids` for answerable queries

> Run `memex search "<question>" --k 10`. **Every chunk in the top-10 whose text contains the literal answer (or a clear paraphrase) goes into `relevant_chunk_ids`.** Single-chunk labels appear when only one chunk has the answer; multi-chunk labels appear when the chunker's overlap creates near-duplicate chunks containing the same answer prose.

The earlier mistake was labelling a single chunk per query "for cleanliness," which artificially penalised correct answers that cited an equivalent overlap-chunk. The rule above eliminates that bias.

### Counterfactual mix

A category's `should_refuse: true` queries should test **both refusal modes**:

1. **Empty-retrieval counterfactuals** — questions whose chunks come back nothing relevant (e.g., "what year was Mellanox acquired" against a CUDA architecture deck). The easy refusal case.
2. **Near-miss counterfactuals** — questions phrased so retrieval pulls *related but non-answering* chunks (e.g., "FP128 energy cost" pulls the FP precision table which lists FP64/32/16/8 but not FP128). The hard refusal case — exercises whether the agent recognises that *being grounded in the chunk-neighbourhood* doesn't mean *the specific value is present*.

Without near-miss counterfactuals, a category's `refusal_rate_on_counterfactuals = 1.0` is misleadingly easy. The bootstrap had 3 empty-retrieval counterfactuals and scored 1.0; the rigorous version added 5 near-miss counterfactuals and the rate dropped to 0.75-0.875 — surfacing real hallucination behaviour.

Tag each counterfactual with an underscore field: `"_counterfactual_mode": "empty-retrieval"` or `"near-miss"`.

### Headline metric interpretation

`mean_citation_precision` counts refused queries as 1.0 (no citations → no false positives), which inflates the number when most queries refuse. Use **`mean_citation_precision_answered_only`** (added 2026-05-21) as the honest signal of citation quality. The all-queries number stays as a tie-breaker / regression indicator but is not the primary metric.

## How to add a query

```sh
# 1. Make sure the doc is ingested.
uv run memex ingest path/to/source.pdf
uv run memex list documents    # confirm the new doc_id

# 2. Author the question. Aim for factual, single-paragraph-scoped questions
#    for answerable ones; for refusals, pick plausible-sounding things the
#    doc explicitly doesn't cover.

# 3. Find the chunk_id(s) that hold the answer.
uv run memex search "your question here" --k 5 | grep '^{"chunk_id"' \
  | head -3 | jq -r '.chunk_id + "  rerank=" + (.rerank_score | tostring | .[0:6])'

# 4. Append the entry to the right category's queries.json with the
#    chunk_id(s) you found. For should_refuse=true, use an empty array.
```

## How to run an eval

```sh
# Quick run (random ~20% sample)
uv run memex eval tests/eval-data/slide-decks/queries.json --quick > /tmp/eval.json

# Full run
uv run memex eval tests/eval-data/slide-decks/queries.json > /tmp/eval.json

# Headline metrics
jq '{
  query_count,
  answered_count,
  refused_count,
  mean_citation_precision,
  refusal_rate_on_counterfactuals
}' /tmp/eval.json
```

The `EvalReport` schema lives at `src/memex/eval/runner.py::EvalReport`. `mean_citation_precision` is `len(cited ∩ relevant) / len(cited)` averaged over answered queries; `refusal_rate_on_counterfactuals` is the fraction of `should_refuse: true` queries the agent correctly refused.

## What's NOT here yet

- **Parsing evals** (CER / WER / structural F1) — the metrics are implemented in `src/memex/eval/scoring.py` but not wired into `runner.py`. The spec calls these out as Phase 2; they need hand-curated ground-truth markdown per document, which is a deeper labour expense.
- **Remaining parse-plan categories** — `historical-scans` (the printed-OCR path none of the current corpora touch). **`modern-printed` (REAL docs) was CLOSED 2026-06-07 by `linux-fundamentals`** — 16 real born-digital Linux Essentials PDFs (a NEW domain, ingested from a local USB key), replacing the synthetic-fixture-only state. Added 2026-05-25: `nist-zero-trust` (security standard), `scientific-gte` (scientific paper), `technical-guidelines` (technical docs — code/deep-headings), `forms-w9` (IRS Form W-9 — government form). **Added 2026-05-27: `handwritten` (cs-notes-1) — the first corpus to exercise the scan→VLM parse route, closing the handwritten slice of the P2.3 scan gap.** Each remaining one needs a doc in the vault + a `queries.json` here; the repeatable playbook is in [`scripts/extend_corpus.py`](../../scripts/extend_corpus.py) (ingest → init → author anchors → resolve → eval). `historical-scans` still exercises the printed-OCR path none of the current corpora touch.
- **Counterfactual diversity** — three refusal queries against a single-document corpus is the minimum. A larger corpus would let counterfactuals exercise *retrieval-distractor* refusals (questions that pull near-miss chunks the agent has to recognise as off-topic), not just *empty-retrieval* refusals.

See `docs/eval-corpus-plan.md` for the full multi-category vision and CER/F1 thresholds.
