# Memex Eval Corpus — Assembly Plan

### How we measure whether Memex actually works

---

## Why this document exists

Every quality threshold in the developer guidelines — "≥ 98% character accuracy on modern print," "citation precision ≥ 95%," "regressions > 15% fail the build" — is meaningless until we have a corpus to measure against. This document is the plan for assembling that corpus, defining ground truth, and operationalizing the scoring.

The eval corpus is the regression net under the entire system. Without it, every model swap, prompt edit, and pipeline change is a guess. With it, every change has a measurable delta.

The corpus is **not** the same as production data. Production is whatever the user puts in their vault. The corpus is a curated, versioned, copyright-cleared set of documents designed to exercise the system's known failure modes.

---

## Goals

1. **Detect regressions** in parsing accuracy, retrieval quality, and answer faithfulness across model and prompt changes.
2. **Calibrate quality thresholds** so the guidelines' numerical bars (95%, 98%, etc.) reflect achievable real-world performance.
3. **Surface failure modes** that pure unit testing misses — bad equation parsing, citation hallucinations, retrieval blind spots in long-tail languages.
4. **Provide a shareable baseline** so contributors can reproduce quality measurements without access to private documents.
5. **Stay legally clean** — every document in the corpus must be redistributable under terms compatible with the project's license.

---

## Document categories

Seven categories, each exercising a distinct part of the pipeline.

| # | Category | Target size (v1) | Hardest part | Threshold (CER / structural F1) |
|---|---|---|---|---|
| 1 | Modern printed text | 30 | Long-tail Unicode, ligatures | ≥ 98% / ≥ 95% |
| 2 | Scientific papers | 25 | Equations, multi-column, tables | ≥ 95% / ≥ 90% |
| 3 | Slide decks | 15 | Irregular layout, text-in-image | ≥ 90% / ≥ 85% |
| 4 | Technical documentation | 20 | Code blocks, deep headings | ≥ 98% / ≥ 95% |
| 5 | Historical / degraded scans | 15 | Low contrast, broken glyphs | ≥ 85% / N/A |
| 6 | Handwritten | 10 | Cursive, mixed scripts | ≥ 75% / N/A |
| 7 | Forms and structured documents | 10 | Field detection, checkbox state | ≥ 95% / ≥ 95% |

Total v1 corpus: **~125 documents**. Big enough for meaningful statistics, small enough to ground-truth by hand in a few weeks.

Multilingual coverage: at least 30% of the corpus must be in non-English languages, with explicit coverage of (a) right-to-left scripts (Arabic, Hebrew), (b) CJK, (c) at least three European languages with diacritics (French, German, Polish), and (d) one Indic script (Devanagari).

---

## Sources (copyright-cleared)

Every document in the corpus has a redistributable license. The acceptable list:

- **Public domain** — works by US federal government, works pre-1929, works explicitly dedicated to PD (CC0)
- **Creative Commons BY / BY-SA** — most permissive academic and open-content licenses
- **Open-access journals with explicit redistribution rights** — DOAJ-listed journals, arXiv (most CS papers), PMC Open Access subset
- **Project-owned content** — documents we authored ourselves for the corpus

The acceptable-but-attribution-required list (allowed with proper `CITATION.md` per document):

- **CC BY-NC** — non-commercial use only; acceptable because the corpus is for an open-source project
- **Research-use licenses** — e.g., IAM Handwriting Database; allowed with proper attribution and use restriction documented

### Per-category source recommendations

**Modern printed text**
- OpenStax textbooks (CC BY) — chapters across history, science, business
- Project Gutenberg modern editions (post-1929 but PD-dedicated) — fiction, non-fiction
- US federal documents — reports, manuals, technical bulletins
- Wikipedia article PDFs — broad topic coverage, multilingual

**Scientific papers**
- arXiv preprints under CC BY — physics, math, CS, statistics
- PMC Open Access subset — biomedical
- DOAJ-listed open-access journals — broad disciplinary coverage
- Aim for variety in typesetting: LaTeX-default, journal-formatted, Word-converted

**Slide decks**
- Open conference talks (LinuxFoundation, Mozilla, academic conferences with CC licenses)
- Lecture slides from MIT OpenCourseWare and similar (most are CC BY-NC)
- Government technical briefings (US PD)

**Technical documentation**
- Open-source project docs with permissive licenses (PostgreSQL, Python stdlib, Linux kernel — many CC BY or compatible)
- W3C and IETF specs (royalty-free with attribution)
- Our own documentation (project-owned)

**Historical / degraded scans**
- Library of Congress *Chronicling America* — US newspapers 1777–1963, public domain
- Internet Archive Books — public domain works with scan artifacts
- HathiTrust public domain subset
- Europeana — multilingual European cultural materials

**Handwritten**
- IAM Handwriting Database (research-use, with attribution) — English, multiple writers
- George Washington Papers (LOC, public domain) — historical English
- CVL Database (research-use) — multilingual handwriting samples
- Synthetic: text rendered in handwriting fonts — lower realism but copyright-clean and ground-truth is trivial

**Forms and structured documents**
- IRS forms (US federal, public domain) — tax forms with varied complexity
- Other government forms (US, EU, Canada federal — many PD or open license)
- Synthetic forms generated from templates with controlled field placement

---

## Ground truth format

Each corpus document lives in its own directory.

```
eval-corpus/
├── corpus.toml                          # version, manifest, summary stats
├── modern-printed/
│   ├── openstax-bio-ch3/
│   │   ├── source.pdf                   # the input
│   │   ├── ground-truth.md              # the expected output
│   │   ├── manifest.json                # metadata, license, scoring config
│   │   └── CITATION.md                  # attribution, license text
│   └── ...
├── scientific/
├── slides/
├── technical-docs/
├── historical/
├── handwritten/
└── forms/
```

### `manifest.json` schema

```json
{
  "doc_id": "openstax-bio-ch3",
  "category": "modern-printed",
  "language": "en",
  "page_count": 18,
  "source_url": "https://openstax.org/...",
  "license": "CC-BY-4.0",
  "license_url": "https://creativecommons.org/licenses/by/4.0/",
  "attribution": "OpenStax, Biology 2e, Chapter 3",
  "expected_features": {
    "headings": {"h1": 1, "h2": 4, "h3": 12},
    "tables": 3,
    "equations": 0,
    "figures": 7,
    "code_blocks": 0
  },
  "scoring": {
    "cer_threshold": 0.02,
    "wer_threshold": 0.05,
    "structural_f1_threshold": 0.95
  },
  "notes": "Includes complex multi-column figure captions on pages 4 and 12."
}
```

### Ground truth Markdown conventions

The ground-truth `.md` file is what Memex *should* produce when given `source.pdf`. It uses the same spec committed to in ADR-0003 (CommonMark + GFM tables + YAML frontmatter + wikilinks + LaTeX math).

Conventions:

- **Frontmatter is complete.** Title, authors, date, source — whatever the original document provides.
- **Headings are hierarchical and match the source.**
- **Tables are GFM tables.** Where the source has a complex multi-row-header table, the ground truth represents it as best the GFM table extension allows, with a note in the manifest if information is lost.
- **Equations are LaTeX, normalized.** `\frac{a}{b}` not `\dfrac{a}{b}`. Whitespace inside `$...$` is collapsed to single spaces in comparison.
- **Figures are referenced by relative path.** `![caption](figures/fig-3.png)`. The actual image file may or may not be in the corpus (we don't ship every figure).
- **Page boundaries are not preserved.** Output is logical structure, not pagination.

### Curation process

Ground truth is **hand-curated** for v1. Realistic time budget per category, with one curator:

| Category | Time per document | Total for v1 size |
|---|---|---|
| Modern printed | 20–30 min | 10–15 hours |
| Scientific | 45–60 min | 19–25 hours |
| Slides | 20–30 min | 5–8 hours |
| Technical docs | 30–45 min | 10–15 hours |
| Historical | 60–90 min | 15–23 hours |
| Handwritten | 30–60 min | 5–10 hours |
| Forms | 20–30 min | 3–5 hours |

**Total v1 corpus curation: ~70–100 person-hours.** Realistic over 3–4 weeks alongside other work.

Acceleration strategies that are safe:

- For categories where source PDFs were generated from LaTeX/Word/Markdown, use the source as the starting point and convert.
- For OpenStax-style content, the publisher often releases source XML — convert programmatically, then spot-check.
- For arXiv papers, the LaTeX source is usually available; render to Markdown via pandoc and spot-check.

Acceleration strategies that are **not** safe:

- Using Memex itself to bootstrap ground truth. This creates a circular dependency: the system passes its own evals because it's grading its own homework. Ground truth must be independent of the system under test.
- Using a frontier LLM to bootstrap ground truth. Even if the LLM is accurate, it imports its own biases and failure modes into the "truth."

---

## Scoring rubric

### Parsing accuracy (per document)

| Metric | What it measures | Tool |
|---|---|---|
| **CER** | Character Error Rate (Levenshtein at character level, normalized) | `jiwer` |
| **WER** | Word Error Rate (Levenshtein at word level) | `jiwer` |
| **Structural F1 — headings** | Precision/recall of (level, text) heading tuples | in-house |
| **Structural F1 — tables** | Precision/recall of cell content given table identity | in-house |
| **Structural F1 — equations** | Normalized LaTeX equality after whitespace and trivial-form normalization | in-house |
| **Metadata accuracy** | Did frontmatter capture title, authors, date correctly? | in-house |

CER and WER use standard normalization: lowercase, strip leading/trailing whitespace, collapse internal whitespace to single space, unicode-normalize to NFC. Punctuation is **not** stripped — punctuation accuracy matters for documents.

### Retrieval quality (per query set)

| Metric | What it measures | Tool |
|---|---|---|
| **nDCG@10** | Ranked relevance of top-10 results | `ragas` or in-house |
| **Recall@10** | Did the relevant chunk appear at all? | in-house |
| **MRR** | Mean reciprocal rank of first relevant result | in-house |

Per-category query sets: 30–50 queries each, with chunk-level relevance judgments (binary for v1, graded later).

### End-to-end answer quality (per question set)

| Metric | What it measures | Tool |
|---|---|---|
| **Citation precision** | Of the chunks the agent cites, what fraction actually support the cited claim? | LLM-as-judge with local judge model + periodic human spot-check |
| **Citation recall** | Of the claims that could be cited, how many are? | LLM-as-judge |
| **Answer faithfulness** | Are all claims in the answer supported by the cited chunks? | `ragas` faithfulness |
| **Refusal rate on counterfactuals** | When asked something not in the corpus, does the agent refuse? | exact-match on refusal indicator |
| **Hallucination rate** | Fraction of answers containing any claim not supported by any chunk | derived from faithfulness |

Question set composition (v1 target: 50 questions):

- 30 in-corpus answerable questions (single-document and multi-document)
- 10 in-corpus unanswerable questions (corpus has the topic but not the specific answer)
- 10 fully out-of-corpus questions (should always refuse)

The unanswerable and out-of-corpus questions are the most important — they test refusal, which is what makes the system trustworthy.

**Chart-content sub-class (added 2026-05-23 with the P3.3 v7 fix arc).** Some answerable queries target content that exists ONLY in chart-extracted markdown blocks emitted by the chart-OCR backend (e.g. a Gantt chart's `On Time 22 / Late 8` status, an architecture figure's 4 design principles). These have an `_answer_type: "chart_content"` annotation and an empty `relevant_chunk_ids` array (the chart-extracted-block chunks aren't FTS-discoverable — they're stripped from BM25 per the P3.3 v3 defense, and live only in dense embeddings). The interesting metric for them is `answered_count` + `refusal_correct` per query, not citation_precision against canonical labels. The current corpora ship 7 chart-content queries across 3 docs (chart-types-08, annual-report-09/10, slide-decks-31/32/33, plus slide-decks-18 which the v7 chunker reflow promoted from REF to ANS via prose+chart-block-in-same-chunk).

### Per-PR delta reporting

`memex eval` produces a JSON report. The CI diff against the last successful main-branch run highlights:

- Any metric regression > 5% (warning)
- Any metric regression > 15% (failure)
- Any new pass/fail on the binary refusal metrics
- Per-document and per-query drilldowns for the worst regressions

The eval report is also a Markdown artifact attached to the PR, so reviewers can see quality impact alongside code changes.

---

## Tooling

### `memex eval` command

```
memex eval [--category CAT] [--quick] [--corpus VERSION] [--baseline REF]
```

- `--category` runs only one category's documents and queries
- `--quick` runs a sampled subset (10% of documents, 20% of queries) for fast iteration
- `--corpus VERSION` pins to a specific corpus version
- `--baseline REF` diffs against a previous eval run (by git ref or run ID)

Output:

- Console summary (rich table)
- `tests/eval-results/{run-id}/report.json` — full structured results
- `tests/eval-results/{run-id}/report.md` — human-readable summary with regressions highlighted

### Dependencies

- `jiwer` for CER/WER
- `ragas` for retrieval and faithfulness metrics (use selectively; we don't want a heavy dep for things we already have)
- A **local judge model** for LLM-as-judge — same Qwen3-8B that runs the orchestrator, with a dedicated judge prompt
- In-house structural F1 implementation (small, no external dep)

Using the same local model as both the system-under-test and the judge has obvious risks (the judge will be biased toward its own outputs). Mitigations:

- **Different prompts** for judge and answerer
- **Periodic human spot-checks** on a random 5% sample
- **Cross-model audit**: every quarter, run the judge with a different model (e.g., a Llama variant) and compare — divergence is a signal to investigate

---

## Versioning

The corpus is itself versioned semantically.

- **Major bump (v1 → v2)**: documents removed, scoring rubric changed, ground truth re-annotated. Forces re-baselining of all metrics; old eval runs are not directly comparable.
- **Minor bump (v1.0 → v1.1)**: documents added, ground truth corrected. Existing thresholds remain valid; new documents may have provisional thresholds for one release cycle.
- **Patch bump (v1.0.0 → v1.0.1)**: typo fixes in ground truth, metadata corrections.

`corpus.toml` declares the current version and lists every document with a content hash. The hash is checked at eval time; a mismatch indicates the corpus has drifted and refuses to run.

---

## Bootstrap plan: the first 30 days

Realistic phasing for going from zero to a working eval suite.

**Week 1 — Foundations**

- Set up `eval-corpus/` directory structure and `corpus.toml`
- Implement `memex eval` skeleton (no metrics yet, just runs through documents)
- Curate 5 documents in `modern-printed` as the test bed for the curation process
- Decide on (and document) all spec ambiguities (heading numbering, table column alignment, equation normalization)

**Week 2 — Parsing evals**

- ✅ **CER, WER, structural-F1 (headings) implemented + wired** (2026-05-24): `eval/scoring.py::score_parse_quality` + `eval/runner.py::run_parse_eval`, exposed as `memex eval-parse <corpus_dir>`. Consumes the `<doc>/ground-truth.md` + `manifest.json` layout above; predicted markdown comes from the vault by doc_id (or a `predicted.md` override). Heading extraction is fence- and chart-block-aware. **Still needs hand-curated `ground-truth.md` docs to run against** — the wiring is done; the curator work isn't. Structural-F1 for tables + equations remains to implement.
- Curate 10 more `modern-printed` and 5 `scientific` documents
- First end-to-end eval run; calibrate thresholds against actual measured performance
- Adjust developer-guidelines thresholds if they prove unrealistic

**Week 3 — Retrieval evals**

- Implement query set runner and nDCG/Recall/MRR
- Create initial query sets for `modern-printed` and `scientific` (~30 queries each)
- Add `historical` and `handwritten` documents (5 each) to stress the parsing pipeline
- First retrieval eval run

**Week 4 — Answer evals**

- Implement answer-quality eval with local judge
- Create 30-question end-to-end question set
- Set up CI integration: eval runs on PRs touching prompts, models, or pipeline code
- Document the regression triage process

**End of month 1**: a working eval suite that scores parsing, retrieval, and answering on a ~50-document corpus, integrated into CI, with calibrated thresholds.

**Months 2–3**: fill out the remaining categories to the v1 target of ~125 documents. Expand query and question sets to full v1 sizes.

---

## Open questions

1. **Cross-category questions.** Do we have multi-hop questions that require evidence from documents in different categories (e.g., a scientific paper plus its corresponding slide deck)? These exercise the graph layer and are genuinely useful, but are more expensive to curate. Defer to v1.1?

2. **Adversarial examples.** Do we include intentionally tricky documents — adversarial OCR, prompt-injection attempts in document content, contradictory information across documents? These are not in v1 but matter for production-readiness. Worth a separate corpus track in v2.

3. **Human evaluator process.** When the LLM-as-judge disagrees with the human spot-check, who wins? Almost certainly the human, but we need a clean update mechanism for the corpus when this happens.

These don't block v1 but should be resolved before v2.
