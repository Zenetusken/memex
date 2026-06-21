# Spec — Table-aware parse routing (born-digital → Docling when table-dense)

Status: PROPOSED — empirics-first gate PASSED (census-stats, PR #11). Spec only; the BUILD +
the full-eval gate are resource-deferred (the dev rig's VMs left no host-RAM headroom).

## Problem

The parse classifier (`src/memex/parse/pipeline.py`, the `classify_*` → dispatch path) routes a
born-digital PDF to **PyMuPDF** (`engine=pymupdf`). For TABLE-DENSE born-digital docs this is wrong:
PyMuPDF's table extraction **collapses dense tables into run-on mega-cells** (whole columns of
numbers in one cell), so the `[table-rows]`/text-to-SQL path receives garbage and table-cell queries
false-refuse.

**Confirmed empirically** (census-stats corpus, PR #11): 2 dense US Census P60 reports.
- PyMuPDF route: 11 "tables" in the store, all mangled (run-on mega-cells).
- `--force-docling` (Docling/TableFormer): 43 tables, well-formed GFM (value-per-cell).
- The eval on the Docling tables: cell-retrieval **8/10**, aggregate/text-to-SQL **4/5**, prose **4/4**,
  `refusal_cf=1.0`, 0 hallucinations — table-RAG WORKS, where the mangled PyMuPDF tables would have
  refused nearly all of these.

This is the deep-research's "structure-preserving serialization" direction made concrete
([[form-table-research-2026-06-15]]).

## Why not always Docling / why density-gated

Docling is **SLOW** — ~12.6 min/doc (TableFormer per page) vs PyMuPDF's ~8 s. Routing *every*
born-digital doc to Docling would 90×-cost the common (table-light) case for no benefit. So the lever
must **gate on table-density**: only docs where PyMuPDF would mangle a meaningful amount of table
content earn the Docling cost.

## Design

1. **A cheap table-density signal in the classifier.** During the existing born-digital classification
   pass, run PyMuPDF's `page.find_tables()` on a SAMPLE of pages (e.g. first/middle/last N) and compute
   a density metric: `tables_per_page` and/or `table_area_fraction`. `find_tables` only needs to
   *detect* tables (count + bbox) — reliable even though PyMuPDF's *extraction* of dense ones is poor.
   This is the chicken-and-egg break: detect cheaply with PyMuPDF, parse properly with Docling.
2. **Route on a threshold.** If the density metric ≥ `settings.parse.docling_table_density_threshold`
   (new `ParseSettings` field, tuned so genuinely-dense docs route, table-light docs don't), set
   `engine=docling` for that doc; else keep `engine=pymupdf` (unchanged default). Chart-OCR follows
   the existing config (the census corpus ran `disable_chart_ocr=true`; routing is orthogonal).
3. **Kill-switch:** `MEMEX_PARSE__DOCLING_TABLE_DENSITY_THRESHOLD` set very high (or a
   `MEMEX_PARSE__TABLE_AWARE_ROUTING=false` bool) fully reverts to PyMuPDF routing.
4. **Migration:** existing table-dense docs in a vault keep their PyMuPDF parse until re-ingested; the
   routing only affects NEW ingests (+ explicit `memex reindex --force` / re-ingest). No silent
   vault-wide re-render.

## HARD-gate invariant — do NOT overclaim (the truncation-lever lesson)

**This is a MUTATIVE parse change, NOT HARD-gate-neutral "by construction."** Docling produces
*different chunk text* than PyMuPDF for the same doc → re-ingest re-chunks → embeddings change → ANN
shift → retrieval/grounding can move on OTHER queries for that doc, and (suite-wide ANN) a borderline
CF in an unrelated corpus can tip (the tg-15 lesson). This is the SAME class as the index-time form
linearizer (forms Increment C, measured-negative) and the query-aware truncation lever (measured-
negative, reverted 2026-06-21) — a mutative re-render that fixed its target but regressed elsewhere.

So the win is **proven on the table-dense docs** (census-stats), but the BROADER impact is **unproven
and must be measured**: re-ingesting table-dense docs across the vault could churn other corpora. The
correction the commit message over-claimed ("HARD-gate-neutral by construction") is WRONG for the
ROUTING change (only the *isolated* "Docling tables are better" claim is construction-true; the
re-render's suite impact is not).

## Files

- `src/memex/parse/pipeline.py` — add the `find_tables` density probe to the born-digital classifier
  branch + the threshold route to `engine=docling`.
- `src/memex/core/config.py` — `ParseSettings.docling_table_density_threshold` (+ optional
  `table_aware_routing` bool kill-switch).
- `tests/unit/test_parse_routing.py` — density-metric computed correctly; threshold routes
  dense→docling / light→pymupdf; kill-switch reverts; degenerate (0-page / no-table) safe.

## Validation (the eval gate — RESOURCE-DEFERRED, the load-bearing step)

- A density-classification probe on a labelled set (known table-dense vs table-light docs) → the
  threshold separates them without false-routing table-light docs to the slow path.
- **Re-ingest the vault's table-dense docs** under the new routing, then the **FULL `rebaseline.py`
  suite, N≥2** (the mutative-re-render regression check): the table-RAG corpora improve (census-stats
  ANS↑), and `refusal_cf==1.0` + 0 hallucinations + no answered→refused flips on EVERY other corpus.
  Acceptance = strictly ≥ current ANS with 0 HARD-gate regressions. If it regresses like the truncation
  lever did → it's a measured-negative, document + revert. **The win is not assumed; the eval decides.**

## Anti-scope (deferred / out)

- **Vault-wide auto re-ingest** of existing docs — out (silent re-render churn). Routing affects new
  ingests + explicit re-ingest only.
- **Tuning the density threshold against one corpus** — whack-a-mole; tune on a labelled density set,
  validate on the full suite.
- The Docling **chart-OCR** path — orthogonal (config-gated); not part of this routing change.
- A faster table parser than Docling — out (Docling is the validated structure-preserving option today).
