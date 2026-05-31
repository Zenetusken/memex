# Audit 10 — Raw-markdown output quality (precursor to the rich document view)

**Date:** 2026-05-30
**Scope:** the parsed `vault/documents/{doc_id}.md` — the canonical "raw" output that is the source of
truth and the substrate for retrieval/embeddings. **Goal:** make the raw `.md` a perfectly-structured
source-of-truth (clean headers / sections / blocks) and harden extraction + retrieval, as the precursor
to a **rich document view** (original PDF / clean raw markdown / rich rendering, side-by-side or toggled).
**Method:** a structured multi-agent audit over a 7-doc representative set spanning every parse path
(`.claude/workflows/raw-md-audit.js`), cross-checked against vault-wide artifact scans over all 47 docs.

## Governing mandates (from the user)

1. **Fix at the source.** Prefer fixing the parse worker / pipeline emission over post-processing. A
   post-processing/render-time strip is a fallback only, never the recommended fix.
2. **Structured output.** Findings are produced and tracked as structured data (the workflow schema).
3. **Keep the `<!-- image -->` marker.** It is load-bearing — the rich view inserts the actual image at
   the marker programmatically, and it gives figure-loss (W9) a visible anchor. Do **not** drop image
   markers; *enrich* them (carry the figure type + an asset/page reference) instead.

## Scorecard (structural quality of the raw `.md`, 0–10)

| Score | Doc | Engine | Headline issue |
|------:|-----|--------|----------------|
| **8.5** | cs-notes-1 | scan-VLM (whole-doc) | clean baseline — zero derived pollution, clean nested headings. **Proof the target is reachable.** |
| 5.5 | NIST SP 800-207 | pymupdf | flat outline (82×H2 / 0×H3); **all 12 figures silently lost**; running header repeated ×56; corrupt acronyms table (every row keyed `API=`) |
| 5.5 | gte-2308.03281 | pymupdf | **all display equations dropped** (dangling lead-ins); two-column order scrambled (5.1→5.3→5.2); flat `##` |
| 4.5 | ENSA-3 deck | docling + per-page VLM | clean content buried under pollution; VLM narrates a decorative meme + a medallion as course content; leaked ` ```markdown ` fence |
| 3.5 | 10-K (annual report) | docling | 445×H6 vs 9×H1; `[table-rows]` ≈ half the file; layout graphic → garbled, duplicated table; VLM fences trap the only correct headings as code |
| 3.5 | IRS W-9 | pymupdf + chart_ocr | **the form is structurally destroyed** — whole page wrapped in `[chart-text]`, two-column order scrambled to run-on prose |
| 2.5 | CUDA deck | docling (VLM disabled) | lowest S/N — ~245 bare classifier labels + ~120 empty placeholders + uniform-H6 titles + 10× build-animation duplication |

## Vault-wide breadth (47 docs — the true scope; per-doc sample counts undercount)

| Artifact | Vault-wide | Root stage |
|---|---|---|
| `[table-rows]` derived blocks | **362 occ / 42 docs** | `table_linearize` via `_finalize_body` |
| `<!-- image -->` placeholders | **1053 occ / 39 docs** | `docling_worker` |
| classifier-label lines (`Logo`×649, `Icon`×206, `Bar chart`×37, `Photograph`×34, `Line chart`×20) | **~946 lines** | `docling_worker` (PictureClassifier) |
| ` ```markdown ` fence-wrapper leak | **81 wrappers / 30 docs** | `vlm_backend` per-page path |
| `[chart-extracted]` blocks | 20 occ / 7 docs | `chart_ocr` via `_stitch_chart_extractions` |
| mass-H6 (docling) / flat-H2 (pymupdf) | pervasive | heading-level recovery |

## The keystone architectural finding — the `.md` conflates source-of-truth with retrieval substrate

`parse/pipeline.py::_finalize_body` (`linearize_gfm_tables`) and `_stitch_chart_extractions` write derived,
retrieval-oriented blocks **into the canonical `.md`** at all three parse paths, right before write. This
single design choice causes three problems at once:

1. **"raw isn't clean."** In table-heavy docs the derived blocks are ~half the file.
2. **Embedding noise (verified).** The dense path embeds `c.text` *including* these blocks
   (`index/pipeline.py:197` chunks `doc.body`; `:132` embeds `document_input(chunk_title(c), c.text)`).
   So tables are encoded twice (GFM + `[table-rows]`), and `<!-- image -->` / `Logo` tokens dilute the
   300M mean-pooled EmbeddingGemma vector (which the contextual-retrieval negative result proved is
   noise-sensitive). Meanwhile BM25 (`_strip_for_fts`) sees a *different* stripped variant — a latent
   **dense/BM25 asymmetry**.
3. **Fence-injection.** Derived blocks get stitched *inside* leaked ` ```markdown ` fences.

**Fix (source):** make the canonical `.md` content-only; re-derive the linearization at **index time**.
`linearize_gfm_tables` is pure, and the chunker already strips/handles `[chart-extracted]`, so the index
layer can both clean and re-derive. This is **retrieval-neutral by construction**: because the current
`.md` *is* `linearize_gfm_tables(clean_body)`, linearizing the clean body at index time reproduces the
exact same chunk input → identical chunk_ids → no re-embed, while the vault file becomes clean. It also
lets us choose **per-arm** what dense vs. BM25 each consume (the future fix for the embedding noise).

## Recorded design decisions

- **D1 — Two representations, split at the write boundary.** Canonical `.md` = clean content only
  (structured headings, GFM tables, prose, enriched image markers). Retrieval substrate (`[table-rows]`,
  chart-OCR text, per-arm noise stripping) is derived at **index time**, never written to the vault `.md`.
- **D2 — Keep + enrich the `<!-- image -->` marker.** The marker stays in the clean `.md` (rich-view image
  insertion + W9 figure-gap visibility). Enrich it to carry the figure type and an asset/page reference
  (e.g. `<!-- image: kind=line-chart page=12 -->`) so (a) the classifier label stops leaking as prose
  (W3) and (b) the rich view can resolve the marker to the real rendered page/figure.
- **D3 — Make loss visible.** Figures/equations dropped by born-digital parsing (W9) must leave an
  explicit placeholder + truthful manifest counters, so a quality gate can catch silent loss.

## Weakness map (17 weaknesses, prioritized; all source-fixable)

| ID | Sev | Title | Root stage | Source fix (summary) |
|----|-----|-------|------------|----------------------|
| **W1** | crit | `[table-rows]`/`[chart-extracted]` derived blocks baked into the `.md` | architectural (`_finalize_body`/`_stitch_chart_extractions`) | clean `.md`; re-derive at index time (see keystone). **← step 1** |
| **W2** | crit | heading hierarchy flattened to one level / collapsed to H6 | docling/pymupdf size→level remap | one shared normalizer; use the dotted section-number as authoritative depth; monotonic nesting; masthead→H1 |
| **W3** | high | PictureClassifier labels emitted as bare prose (~946 lines) | docling_worker | never emit the class string as body text; fold it into the (kept) image marker as `kind=` metadata (D2) |
| **W4** | med | bare `<!-- image -->` placeholders with no payload | docling_worker | **keep** the marker (D2) but attach caption/kind/asset ref; drop only true duplicates |
| **W5** | crit | per-page VLM ` ```markdown ` fence-wrap traps headings/tables as code (81/30) | vlm_backend | apply the per-page fence unwrap to **every** VLM-escalated page, page-boundary-anchored, before linearization |
| **W6** | high | VLM narrates decorative imagery / visual styling / restatement | vlm_backend prompt | tighten prompt: transcribe instructional text only; skip decorative imagery; semantic diagram content, no styling; once each; no editorial notes |
| **W7** | med | HTML entities leaked un-decoded (`&amp;` `&gt;`) | docling_worker / finalize | html-unescape before emitting markdown |
| **W8** | med | running headers/footers/page-numbers not stripped (NIST ×56) | pymupdf/docling | repeating top/bottom y-band detection; drop before body + table assembly |
| **W9** | crit | born-digital figures & equations **silently dropped**, manifest says 0 | pymupdf_worker | detect figure/formula regions → explicit placeholder (D3) or escalate to VLM/OCR-LaTeX; populate manifest counts from emitted artifacts |
| **W10** | high | two-column reading order scrambled; forms destroyed | pymupdf_worker | (column, y) block ordering by bbox x-midpoint; block-level paragraph join; key:value adjacency for forms |
| **W11** | high | layout graphics / infographics / bit-rulers mis-detected as tables | docling_worker | tighten table-vs-figure: mostly-empty / <2 data cols / identical headers / header-in-cells → figure or bullets |
| **W12** | high | linearizer corrupts data (empty-corner / headerless → wrong keys; furniture as keys) | table_linearize / pymupdf | positional keys for headerless tables; row-label column for empty corner; exclude caption/furniture from header rows |
| **W13** | high | animated slide-build duplication; figure/seam re-emission | docling_worker / vlm seam | near-duplicate collapse for animation build-up; de-dup VLM/Docling page seams |
| **W14** | med | code-fence misuse (pull-quotes fenced; code line-breaks collapsed) | docling/vlm | pull-quotes → blockquote; preserve code line breaks from text-layer geometry |
| **W15** | med | real titles emitted as plain paragraphs; no doc H1; filename-slug titles | docling/pymupdf | promote title frames + masthead→H1; set frontmatter title from masthead not slug |
| **W16** | low | block hygiene (redundant heading-bold, mixed bullets, inline footnotes, trailing ws, empty frontmatter) | all | shared scrubber: strip heading-bold, normalize bullets→`-`, rstrip, footnote defs, drop empty frontmatter |
| **W17** | low | glyph-spacing / OCR space-join / ref-ID digit-drop | vlm/docling/chart_ocr | prefer text-layer over OCR where present; residual OCR drift accepted |

## Recommended sequence (all source-level)

1. **Architectural split — `[table-rows]` off the `.md`, re-derive at index** (W1). Retrieval-neutral;
   unblocks the clean raw view for all docs at once; de-noises a future embedding pass; removes the
   inside-fence injection mode. **← in progress.** (`[chart-extracted]` follows as step 1b — it needs the
   chart-OCR output persisted for index-time re-attachment, since it is not re-derivable from clean text.)
2. **Shared finalize scrubber** (W3, W4-enrich, W7, W8, W16) — one pass across all 3 parse paths.
3. **Heading-hierarchy normalizer** (W2, W15) — section-number depth + monotonic nesting → a real TOC.
4. **VLM fence unwrap (W5) → VLM prompt tightening (W6)** — measure VLM-corpus HARD gates multi-run.
5. **Table detection + linearizer logic** (W11, W12).
6. **Dedup (W13) → reading-order/code-fence (W10, W14) → content-loss escalation + manifest truth (W9).**

## Validation discipline

- Parse-output changes require a **vault re-process** (back up first — the vault is not git-tracked) and
  are **retrieval-gated** with `gold_chunk_recall` + the answer-eval HARD gates (refusal_cf / zero-halluc),
  measured multi-run for borderline counterfactuals (eval non-determinism rule).
- Step 1 is retrieval-neutral *by construction* (identical chunk input) — validated by unit tests that the
  `.md` carries no `[table-rows]` while the indexed chunk text does, plus the full suite.

## Status

- ✅ Audit complete (this doc). Reusable workflow: `.claude/workflows/raw-md-audit.js`.
- ✅ **Step 1 (W1) — `[table-rows]` relocated off the canonical `.md`** (2026-05-30). Moved
  `linearize_gfm_tables` → `core/table_linearize.py` (also fixes a latent `index→parse`
  import); `parse/pipeline.py::_finalize_body` now emits a content-only body; `index/pipeline.py`
  re-derives the linearization before chunking. **Retrieval-neutral by construction** (new chunker
  input `= linearize(clean) =` the old `.md`) + idempotent (so a not-yet-re-parsed `.md` re-derives
  to identical bytes). **Validation:** 1061 unit/integration tests (+2 regression) + ruff/pyright
  clean; 47/47 vault docs idempotent (re-index can't double-linearize); a live NIST re-parse proved
  `.md` clean (−7,567 chars) with `linearize(clean) == old .md` byte-identical; webui raw view
  confirmed 0 `[table-rows]` / 9 tables preserved; vault restored pristine.
  - **`[chart-extracted]` (step 1b, deferred):** not re-derivable from clean text (it is OCR'd from
    images), so it needs the chart-OCR output persisted (manifest sidecar) for index-time
    re-attachment, or to be rendered as clean content. 20 occ / 7 docs.
  - **Rollout:** the vault re-process (re-parse all → clean `.md`) is batched after the remaining
    parse-stage fixes (steps 2–9) to avoid re-parsing 47 docs repeatedly. The code is live + idempotent
    on existing docs in the meantime.
- ✅ **Step 2 (W3/W4/W7/W16-rstrip) — Docling image-marker enrichment + content-only scrub** (2026-05-30).
  Source-fixed in `docling_worker` (`enrich_image_markers` + `clean_docling_markdown`, run on the doc-level
  AND per-page markdown): the PictureClassifier label is folded INTO the kept `<!-- image -->` marker as
  `kind=` metadata (decision D2 — marker stays for rich-view insertion + W9 gap visibility) and the bare
  label paragraph dropped (~946 noise lines); over-escaped HTML entities decoded (W7); lines rstripped
  (W16). `_IMAGE_PLACEHOLDER_RE` made tolerant of `<!-- image: kind=… -->` so the chart-OCR stitch +
  figure-count alignment are unchanged. **Validation:** 1066 tests (+5) + ruff/pyright clean; deterministic
  on real output (CUDA 244→10, ENSA-3 67→0, 10-K 75→0 bare labels + 109→0 entities); a **live re-parse of a
  Docling deck** (srwe-16) produced 14 enriched `kind=` markers / 0 bare labels / 0 `[table-rows]`, confirmed
  in the webui raw view; vault restored pristine.
- ✅ **Step 2b (W8) — repeating page-furniture strip in `pymupdf_worker`** (2026-05-30).
  `strip_repeating_page_furniture` drops running headers/footers + page numbers that recur in the
  page-boundary BAND (first/last `_FURNITURE_BAND`=3 non-blank lines) on ≥ max(3, 0.5×n_pages) pages, OR
  are bare page numbers. POSITION-AWARE (only the band occurrence removed → legit mid-page text kept),
  structural lines (heading/table/list/code) never furniture, page/markdown records kept in lockstep.
  **Validation:** 1073 tests (+7) + ruff/pyright clean; an **adversarial cross-doc audit** of EVERY removed
  line across all 5 born-digital docs found **0 real false-positives** (NIST 162 stripped = the two-line
  header ×56 + page numbers; gte/guidelines 0; the one "suspicious" hit was pymupdf4llm's own
  `==> picture omitted <==` placeholder — correctly removed noise). A **live NIST re-parse** took the ×56
  running header + page numbers → 0, confirmed in the webui raw view; vault restored pristine.
  - **Deferred:** docling-deck running-header furniture (decks rarely have them) + a multi-line-aware band
    for docling, if a deck case appears.
- ✅ **Step 2c — roman page numbers + TOC dot-leaders** (2026-05-30, follow-up to user-reported residual
  artifacts). `pymupdf_worker`: a STRICT roman-numeral pattern (`(?=[ivx])` + tens/units to `$`) adds
  standalone front-matter roman page numbers (`iv`/`v`/`vi`/`vii`/`x`) to the furniture strip while
  rejecting the all-[ivxlcdm] English false-friends (`mix`/`did`/`lid`/`civil`/`mild`). `_finalize_body`
  (the now-active engine-agnostic finalize scrubber) collapses dot-leader pagination artifacts
  (`Introduction ......... 1` → `Introduction`), incl. inside GFM table cells, fence-aware. **Validation:**
  1075 tests (+3) + ruff/pyright clean; a live NIST re-parse took dot-leaders 106→0, the TOC table to a
  clean `|**1**|**Introduction**|`, roman page numbers → 0; an adversarial collapse audit across all 47
  docs removed only pure pagination spans (0 content-loss); webui-confirmed; vault restored pristine.
- ✅ **Step 3 (W2/W15) — engine-agnostic heading-hierarchy normalizer** (2026-05-30). `parse/pipeline.py::normalize_heading_levels`, run in `_finalize_body` on BOTH worker outputs: each heading's level is overridden by its **section number** when it has one (`N`→H2, `N.N`→H3, `N.N.N`→H4; 1–2 digits/group so a 4-digit YEAR is never a section number; `Item N`/`Section N`/`Part N`/`Appendix X`→H2), else the engine level is kept; a doc with **no H1 whose first heading is an unnumbered masthead** gets it promoted to H1; a final **monotonic-nesting clamp** forbids descending >1 level at a step (`H2→H5`→`H2→H3`). Fence-aware; only the `#`-count changes, heading TEXT is preserved verbatim. **Validation:** 1083 tests (+8) + ruff/pyright clean; **live NIST re-parse** (PyMuPDF) took the flat **82×H2 → {H2:36, H3:30, H4:16}** (a real `1`/`1.1`/`2`/`2.1` tree), matching the deterministic preview exactly; gte masthead→H1 + tree; an **adversarial cross-doc audit across all 47 docs** found 0 regressions (heading COUNT preserved — levels only — idempotent, no H1 inflation, no tree collapse; the monotonic guard correctly pulls orphan-deep H5/H6 up); the webui `extract_toc` + a doc-view screenshot confirm the nested TOC renders end-to-end. Vault restored pristine.
- ✅ **Step 3b (W2) — Docling bbox-height level cap** (2026-05-30). `docling_worker._recover_heading_levels` now caps the height-rank at **level 4** (`#####`/H5 deepest), so a dense doc's near-continuous height range can't bottom out at mass-H6; the cap is 4 (not 3) because H1 is reserved for the document TITLE (`TitleItem`), leaving FOUR header layers (H2–H5) below it (user spec). **Validation:** unit (`test_caps_at_level_4`) + a **live force-docling 10-K re-parse**: 445 H6 → **0 H6** (capped to H5). **HONEST RESIDUAL (roadmapped to its own session):** the 10-K's 415 H5 are *real* section titles, not mis-detected labels — docling's bbox-height ranking buries the document's DOMINANT section tier (the most-populated, smallest-height bucket) beneath 4 rarer larger headings, so the cap bounds the DEPTH but can't un-flatten them into a true tree. Step 3's section-number signal (the reliable fix) doesn't apply — these headings are unnumbered — and a frequency/mode-anchored docling re-tiering trades the burying failure mode for an over-flatten one, so it needs a focused session (heuristic + multi-doc force-docling re-parse A/B). Recorded in `docs/ROADMAP.md` ("◐ In progress — Raw-`.md` output hardening" → the own-session deferral). NB the 10-K's DEFAULT route is PyMuPDF (clean tree); force-docling is only for chart-OCR. The residual `415 real headings at one level` is partly INHERENT (the 10-K genuinely has ~415 peer sections) — the open question is only WHICH level, not the count.
- ✅ **Step 4a (W5) — per-page VLM ` ```markdown ` fence unwrap** (2026-05-30, commit `97e0e4d`). The VLM often wraps its whole-page answer in a ` ```markdown `/` ```md ` fence, trapping the page's headings/tables/prose as a code block (30 vault docs affected; 11 in ensa-module-10). The scan path already stripped it (`_assemble_scan_pages`); the per-page VLM-ESCALATION re-stitch (`_route_and_escalate`) did NOT — the gap. Fix: apply the existing `_strip_markdown_fence_wrapper` at the per-page escalation store point (mirrors the scan path; done at the page object so `char_count` counts the unwrapped length). **CACHE-NEUTRAL** (the cache still stores the raw draft → existing docs replay + unwrap on read → a plain re-parse retires the wrappers, no cache bust). HARD-gate-neutral (parse-side stitch). **Validation:** an adversarial sweep across all 30 affected docs found 0 false-unwrap risk (only genuine line-0 whole-page wrappers fire; nested code stays balanced; a lost close is a conservative no-op); +3 tests (1086) incl. the escalation-path unwrap + char_count; ruff/pyright clean.
- ⏸ **Step 4b (W6) — VLM prompt decorative-narration suppression: IMPLEMENTED + VALIDATED → DEFERRED to its own session** (2026-05-30). A calm-register prompt rewrite (no forceful absolutes — respecting the 2026-05-25 revert) to skip decorative-image / visual-styling / editorial narration (the `*Image Description: a man in a suit ... OBEY ... orange background*` class; ~15-25 decorative blocks vault-wide), led by a "Transcribe every heading, title, label, and line" preservation guard to avoid over-suppression. **Empirically validated** on a `ccd09479-cr350-network-diagrams` re-parse (fresh VLM, `prompt_sha8` cache-miss) + N=3 answer-eval: **the over-suppression risk did NOT materialise** (single-tier connectivity `Internet -> Router -> Firewall -> Private Network`, EAPOL, every instructional token PRESERVED or richer) and **the HARD gates HELD** (refusal_cf=1.0, 0 hallucinations, deterministic ×3). **BUT it failed the ship bar (`ANS ≥ baseline`) with a CONSISTENT −1 ANS** (10 vs the confirmed 11/6 baseline; diag-02 flipped answered→refused). **Root cause (diagnosed): NOT content loss** — the answer is present; the preservation guard makes the VLM transcribe MORE multi-tier flow detail, which crowds the single-tier answer in retrieval → a benign borderline false-refuse. Suppress-decoration vs preserve-content vs don't-induce-retrieval-crowding-verbosity is a genuine tension. **DEFERRED**: the prompt was reverted byte-identical to the validated baseline (`prompt_sha8` unchanged → VLM cache untouched); the attempt + finding are recorded in the `vlm_backend._PROMPT` NB comment + ROADMAP (audit-10 W6). The own-session refinement: a less-verbosity-inducing preservation guard, validated ANS-neutral across ALL VLM corpora (cr350-diagrams + the 8 ccna-multidoc decks) before any vault-wide re-parse. The deterministic-filter alternative was assessed NON-viable (the `*Image Description:*` marker is used for BOTH decorative noise AND instructional diagrams). Vault restored pristine.
- ✅ **Step 5 (W11/W12) — table-vs-figure demotion + linearizer key correctness** (2026-05-30, commit `517182a`). **W11:** `is_layout_table` (the shared predicate, keyed off DATA-row width — a block with no 2-D relation) + `demote_layout_tables` (engine-agnostic, in `_finalize_body`) re-render a layout graphic / infographic / single-column list mis-detected as a GFM table (the 10-K "RISK OVERSIGHT AT NVIDIA" infographic with bullets-in-cells, a metric list, a references list) as faithful markdown bullets — content raw (bold-balanced), deterministic + idempotent. The index-time linearizer + `table_store` reuse `is_layout_table` to SKIP the same blocks (no nonsense `[table-rows]` KV / `tables.sqlite` row). **W12:** `header_has_lost_columns` (≥2 empty header cells = a flattened merged-cell header whose real labels fell into data row 0) ORed into the header-sanity skip gate (linearizer + store) — fabrication-safe SKIP-over-mis-key; `_trim_furniture_columns` drops a trailing header-only OCR furniture column. **Designed via a 7-agent workflow then independently REVIEWED — which caught + fixed 2 correctness defects the agents left:** (1) `is_layout_table` flagged on `len(header) < 2` alone → would flatten a real 2-col table whose HEADER under-split to 1 cell (fixed to data-row width + a guard test); (2) `header_has_lost_columns` skipped ANY non-leading empty → over-skipped a real table that lost ONE interior label (gte hyperparameters; the linearizer already keys an empty column POSITIONALLY = fabrication-safe), refined to ≥2 empties (skips 16→14). **Validation:** +table tests (1117 total) + ruff/pyright clean; a deterministic 47-doc audit (5 demotions / 14 skips / 0 false-positives / byte-idempotent); the 10-K infographic demotes to clean bullets; a live 10-K reindex + **annual-report eval HOLDS the HARD gates** (refusal_cf=1.0, 0 hallucinations, 0 incorrect verdicts; the income-statement golds W11/W12 never touch are answered, all 6 counterfactuals refused). Vault `.md` pristine (the index-time skips applied via reindex; the W11 `.md` demotion lands at the batched re-parse rollout).
  - **Infra finding (mapped, NOT W11/W12): the annual-report eval at `top_k=5` / the 6144 `fast`-mode window hits a CONTEXT OVERFLOW** on a single income-statement query (4345 input + 1800 output = 6145, 1 token over) and the answer node **CRASHES** (`ModelCallError` 400) rather than degrading gracefully. Orthogonal to step 5 (the query's content is untouched by W11/W12; validated at `top_k=4` which fits). Recorded in `docs/ROADMAP.md` as a robustness item: the answer node should catch a context-length 400 and refuse/truncate, not propagate. The 6144-window×large-10-K-chunk tension is a known co-residence limit (`full` mode's 24,576 window fits but bumps `top_k` to 18).
- ✅ **Step 6a (W13) — consecutive exact-duplicate block collapse** (2026-05-30, commit `6615534`). Animation slide-build re-emission + figure/page-seam double-transcription produce CONSECUTIVE duplicate blocks. `collapse_consecutive_duplicates` (engine-agnostic, in `_finalize_body`) drops a block whose whitespace-normalized text is EXACTLY equal to the immediately-preceding KEPT block. **RAW EQUALITY + strict adjacency (window 1) is the ONLY false-positive-free setting** (a ratio/token-overlap threshold collapses PARALLEL DATA sharing a template — different IPs, footnote numbers, precision-table rows — proven unsafe at every value). Excluded blocks (`<!-- image -->` marker, bare PictureClassifier labels, box-drawing connectors) are kept but skipped for adjacency; headings compare level-insensitively (keep the shallower). **Designed via a 5-agent (design-only) workflow then implemented + reviewed by hand — the conservative raw-equality subset was chosen over the synthesis's riskier Jaccard-0.90 section-collapse** (the empirical FP-sweep + critic showed it reintroduces parallel-data content loss). **Validation:** +12 unit tests (1129 total) + ruff/pyright clean; a deterministic 47-doc audit — **13 collapses across 4 docs** (CUDA deck 8 slide-title re-emissions, cr350-cours-3 3, cr350-plan 1 bullet, srwe-13 1 glyph), **EVERY dropped block verified byte-equal to its survivor** (no unique content removed), **0 false-positives** (the 10-K with all the legit scattered repeats collapses 0; the guidelines flowchart box-art connectors spared), byte-idempotent. Content-preserving by construction ⇒ HARD-gate-neutral (answer-eval at the batched rollout, like W5).
  - **Deferred (FP-risky follow-up, mapped in ROADMAP): the near-duplicate SECTION collapse** (animation supersets/reorders via token-overlap) — catches more animation frames but the FP sweep proved every ratio threshold reintroduces parallel-data content loss; needs a heading-equality-gated + code-guarded design with its own validation.
- ✅ **Step 6b (W14) — un-fence mid-document markdown-tagged code blocks** (2026-05-30, commit `f141271`). The parsers (and the per-page VLM, beyond W5's whole-page wrapper) wrap a markdown REGION (headings + prose) in a ` ```markdown `/` ```md ` fence MID-document, trapping its headings/prose as inert code (the chunker + fence-aware finalize passes treat a fenced region as code → headings invisible to grounding; 28 docs). `unfence_markdown_blocks` (engine-agnostic, wired FIRST in `_finalize_body` so the recovered headings/tables are visible to the downstream demote/dedup/heading passes) drops the fences + splices the inner lines VERBATIM. **A markdown/md language tag is by construction a prose signal (code-ratio 0 across all such vault blocks) ⇒ false-positive-free**; ABSTAINS on a block with no bare close OR a NESTED fence (un-fencing would splice an unbalanced inner ` ```text `). **Validation:** +9 unit tests (1138 total) + ruff/pyright clean; a deterministic 47-doc audit — **52 clean markdown blocks un-fenced across 28 docs**, the nested-fence (LLDP CLI) blocks correctly ABSTAINED, **LANGUAGE-TAGGED CODE fences IDENTICAL before/after (0 real code touched)**, fence-parity preserved (the 4 pre-existing odd-fence docs were already odd), byte-idempotent. Content-preserving ⇒ HARD-gate-neutral (answer-eval at the batched rollout, like W5/W13).
  - **Deferred (mapped in ROADMAP):** the BARE-tagged heading-block un-fence (disentangling a heading mis-fenced WITH real CLI/code — the FP analysis rated it **8/10 real-code-corruption risk**; needs a positive-prose-fingerprint + mixed-block-abstain design) + the pull-quote→blockquote sub-rule.
- ⏭ **W10 (two-column reading-order) — DEFERRED / data-gated** (2026-05-30). pymupdf4llm 1.27.x's built-in column detection already reads the vault's two-column papers correctly: the gte academic paper's prose flows ("We present GTE… In line with recent advancements…"); the only reproduced scramble (gte §5.1→5.3→5.2 on page 9) is COSMETIC — each subsection stays internally coherent under its heading, with **zero answer-eval impact** (the gte HARD gates hold). A custom `(column,y)` bbox reorder in `pymupdf_worker` is high-blast-radius (overrides pymupdf4llm on the many correctly-read docs) for **no measured benefit and no failing gold to validate against**. The severe form-destruction case (IRS W-9) is a chart-OCR-wrap (W5/W9 family), NOT column ordering. Unblocks on a curated born-digital two-column doc whose scramble causes a *measurable answer regression*.
- ✅ **Step 6c (W9, crit) — born-digital figure visibility placeholder + honest manifest figure_count** (2026-05-30, commit `0141c8b`). Figures were SILENTLY DROPPED on the pymupdf path with the manifest reporting 0 (the gte paper references "Figure/Table N" 34× but had 0 markers + figure_count=0; NIST 37×, 0). **Root cause (probed): pymupdf4llm ALREADY emits a ` **==> picture [WxH] intentionally omitted <==** ` marker per detected-but-unrendered figure (gte 12, NIST 14, zero file I/O) — the worker DELETED them.** The fix is CONVERT-instead-of-delete: each marker → a blank-line-padded `<!-- image -->` visibility placeholder (the Docling D2/D3 convention; furniture-exempt already via the `<!--` structural prefix); `figure_count` is taken from the written body's placeholder count (the field already threads worker→`PyMuPDFConversion`→`ParseStage`; only the hardcoded 0 changed). No `embed_images`/temp-PNG/seccomp work needed (the markers are already there). No per-page cap needed — figures are modest on the pymupdf path (figure-dense decks route to Docling). **Validation:** +4 unit tests (1142 total) + ruff/pyright clean; a **live re-parse of gte + NIST**: manifest figure_count **0→12 / 0→14**, the `.md` gains exactly that many `<!-- image -->` placeholders (**figure_count == body placeholder count**), prose otherwise unchanged. Content-additive ⇒ HARD-gate-neutral.
  - **Deferred (documented, ROADMAP):** `table_count` (an honest count must be taken AFTER `_finalize_body`'s W11 layout-table demotion — a pipeline-level concern, not the worker's) + `equation_count`/equation handling (OCR-LaTeX is a heavy separate model; equation refs near-absent on these docs — gte 1, NIST 0). VLM/OCR transcription of the dropped figures (vs the bare visibility placeholder) is the heavier escalation, also deferred.
- ✅ **W9 follow-up — honest manifest `table_count`** (2026-05-30, commit `7247d1f`). The pymupdf path hardcoded `table_count=0` (the companion to the figure_count lie); now counts the GFM tables in the WRITTEN `final_body` AFTER the W11 layout-table demotion (so a demoted infographic isn't counted; the pymupdf path runs no chart-OCR so there are no `[chart-extracted]` tables to conflate). Live: gte `table_count` 0→13, figure_count 12, both matching the `.md`.
- ⏭ **Step 1b (`[chart-extracted]` off the `.md`) — DEFERRED with a design (own session, complex).** The 20 `[chart-extracted]` blocks (7 docs) pollute the canonical `.md`, but unlike `[table-rows]` they are NOT re-derivable from clean text (OCR'd from chart images). **Option B (strip the wrapper, keep the data as content) is UNSAFE** — the `[chart-extracted]` markers are LOAD-BEARING downstream: `index/chunker.py` uses `chart_extracted_spans` to treat an inner `# H1` chart-label as inert (else it mis-splits the section), and `index/fts_store.py` uses `strip_chart_extracted_for_index` to drop the block from the BM25 body while keeping `chunks_meta.full_text` UNSTRIPPED so the figure survives into the answer context. **Option A (the safe one): persist the chart-OCR output in a manifest sidecar at parse + re-attach `[chart-extracted]` at the `<!-- image -->` positions at INDEX time** (mirroring W1's `[table-rows]` re-derivation, but sidecar-backed since it's not re-derivable) — so the canonical `.md` is clean while the chunker/FTS still see the markers in the indexed body. Substantial (new sidecar + index-time re-attach + chunker/FTS coordination) + must be chart-types-eval-gated → a focused session.
- ⏭ **W17 (glyph-spacing / OCR space-join / ref-ID digit-drop) — ACCEPTED, no fix.** 0 systemic glyph-spacing runs across the 47-doc vault (the lone NIST cover-page "C O M P U T E R   S E C U R I T Y" is stylized cover art, not an OCR artifact); the audit's own fix summary is "prefer text-layer over OCR where present; **residual OCR drift accepted**", and the born-digital docs this surfaces on already use the text layer. No actionable fix.
- 🏁 **Remaining: the BATCHED VAULT RE-PROCESS — the audit's payoff.** Every parse-stage fix above (steps 1–6c + table_count) is live in CODE but applies only on RE-PARSE; the vault's 47 `.md` files are still the pre-audit output. The culmination is a one-shot re-parse of all 47 → clean `.md`, **retrieval-gated against the full eval suite** (all answer-eval corpora hold the HARD gates + the parse-eval structural-F1), with the eval baselines re-recorded. GPU-heavy (many docling+VLM docs) + a deliberate operation (back up the baseline-pinned eval docs first; the vault is not git-tracked) — best run as its own focused session. The deferred FP-risky/data-gated levers (W6 prompt, step-3b docling leveling, W10 reading-order, W13 section-collapse, W14 bare-tagged un-fence, step 1b, the answer-node overflow robustness) are all mapped in `docs/ROADMAP.md` + the agent memory.
