# False-refusal autopsy — the 18 answerable refusals, root-caused (2026-06-09)

**Goal (user-directed): drive answerable false refusals deterministically toward 0.** Stage 1
classified the 18 by pipeline death point (`scripts/fr_classifier`-style probe); stage 2 captured,
per query, the draft + per-claim verify verdicts + relevance verdict + cited-chunk text via the
committed observe-only harness **`scripts/fr_autopsy.py`** (artifact: `docs/audits/data-15-fr-autopsy.json`).

## Stage-1 map (death points)

**13 GATE-killed with evidence at rerank #1–3** — NOT a retrieval problem (the audits-12-era "P2.5
embedder swap" framing was wrong for this population); 3 RERANK-CUTOFF (sd-17 #15, sd-25 #15,
ftc-big-runmain **#6** — one past the top-5 window); 1 RETRIEVAL-MISS (ar-14); 1 chart-content (sd-31).

## Stage-2 mechanisms (each from captured evidence, not inference)

- **M1 — correct draft, WRONG CITATION TARGET (4): chart-types-01, nist-zero-trust-10, slide-decks-03,
  slide-decks-21.** The draft states the RIGHT answer but cites the topically-nearest window chunk
  instead of the one carrying verbatim support (nist-10 "20–30 records" cited to a PE/PA prose chunk;
  sd-21 "1.6x" cited to the flag-description chunk; sd-03 "square of mantissa" cited to the energy
  table; chart-01 byline-without-"wrote"). Verify correctly fails the cited PAIR → refuse.
  **Lever: a verify-time CITATION-RETARGET pass** — before declaring a claim ungrounded, re-test it
  against the OTHER window chunks (same unchanged gate, retargeted); promote only on positive support.
  HARD-gate-safe by construction. ~4 flips, the highest-yield single fix.
- **M2 — DENIAL-FRAMED drafts that contain the answer (2–3): scientific-gte-09/-11, chart-types-04.**
  The draft literally states the answer while framing it as a denial (gte-09: "The chunks do not state
  which specific GPUs… only that training was conducted on up to 8 NVIDIA A100 GPUs"). Verify grounds 0
  (the headline claim is the denial). Detectable signature (denial-prefix + substantive content);
  candidate fix at the answer node or deterministic detect-and-regenerate.
- **M3 — relevance gate applying WORLD KNOWLEDGE (1 confirmed: handwritten-06; finishes off M2
  victims).** The gate rejected the note's own 3-stage compilation answer because it "doesn't
  correspond to the standard C++ compilation process" — the assess_relevance@v2 world-knowledge-ban
  disease recurring on CONTENT (v2 banned it for dates). **Lever: widen the ban → v3.**
- **M4 — VAULT CONTENT-LOSS (1 confirmed: slide-decks-31; sd-04/16 suspected).** The W6 re-parse
  (2026-06-06) dropped the nvmath-python figure's "4 design principles" (0/8 manifest chart_extractions
  carry it) — the chart-types-08/09 class (audit-14 close-out) by a different event. Mechanical
  restore + a re-parse content-regression guard. sd-04 (TSMC chart) / sd-16 (NVRTC 11.8 row) need the
  same vault-state check before being blamed on the model.
- **M5 — RERANK CUTOFF (3): sd-17, sd-25, ftc-big-runmain (#6).** No draft possible; the window/cutoff
  lever (runmain mirrors the usage-gap #6 pattern exactly).
- **M6 — TABLE AGGREGATE (annual-report-14).** The question needs a SUM the doc never states; the
  Table-RAG SQL path didn't fire (the documented ar-14 under-split). Route via the SQL oracle only.
- **M7 — SYNTHESIS (cr350-img-01).** The documented principled residual (gate-relaxation is a recorded
  NO-GO; any fix must be evidence-side).

## Stage-3 hardening backlog (ordered by yield × safety)

1. M1 citation-retarget (4 flips, HARD-gate-safe by construction, deterministic).
2. M3 relevance v3 world-knowledge-ban widening (1+ flips; prompt-versioned, kill-switched, multi-run).
3. M4 vault restores + a re-parse content-regression guard (1–3 flips; mechanical).
4. M2 denial-framing (2–3 flips; needs careful design — answer-node changes are HARD-gate-adjacent).
5. M5 window lever (3 flips; budget-bound — measure context fit first).
6. M6 ar-14 SQL routing; M7 img-01 stays the principled residual unless evidence-side options appear.

Projected ceiling: **18 → ~2–3** with M1–M5 landed, HARD gate untouched throughout (every lever
kill-switched, validated N≥2 full-suite per the ADR-0022 discipline).

## M1 + M1b SHIPPED (2026-06-10, branch `fix/m1-citation-retarget`)

**+2 deterministic flips (chart-types-01, nist-zero-trust-10), 0 regressions** — autopsy ×3 + the full
ship ladder (14 corpora × N=2: `refusal_cf`=1.0 everywhere, 0 errors, ANS at-or-above baseline
[technical-guidelines −1 = the re-scoped tg-13 CF now correctly refusing — tg is at a PERFECT
12/12+6/6]; eval-summary 6/6, 0 leaks; codex-rs `answer_text_correct` 32/38 stable both runs).

The build (4 commits, each validation-caught issue fixed at the root):
1. **M1 retarget** — promote-only re-test of still-ungrounded claims vs window chunks via the SAME
   gate; `source_chunk_id` rewritten on promotion. Kill-switch `MEMEX_AGENTS__CITATION_RETARGET_ENABLED`.
2. **M1b-i render dedup** — the verifier's copy strips the `[table-rows]` duplicate when the GFM table
   is present (the doubled noise drowned a verbatim-present trailing sentence, 3/3; verify-NODE-only ⇒
   summarizer unaffected). New `core/text.strip_table_rows_blocks`.
3. **M1b-ii cited-first isolation probe** — the batch render (5 chunks per claim) drowns support the
   1×1 view grounds (nist-10); the batch verdict is NOT an isolation verdict.
4. **Eligibility filter (HARD-gate-critical, caught by the suite):** only LLM-rejected claims (empty
   reason) are retargetable — a backstop-demoted claim must NEVER be re-probed (the 1×1 LLM call is the
   rubber-stamp the deterministic backstops exist to overrule; the retarget had re-promoted a fabricated
   table SUM). + the final-pass BUDGET PROJECTION (the router refuses on post-merge tokens, so
   budget-exhaustion refusals — "after 0 attempts", the dominant path — never reached the retarget).

**Recorded residuals (not chased):** sd-03 = 4B verifier TABLE-FIXATION (rejects a character-identical
sentence trailing a table even at 424 chars 1×1, reason citing only the table — a verify-prompt-v3
candidate, its own increment); sd-21 = `'-minimal'` vs `"--minimal"` claim/chunk spelling mismatch
(arguably correct strictness; possibly a corpus-wording fix). Inventory: 18 → **16**.

## CORRECTIONS from the M4-M6 forensics (2026-06-10) — the M4/M6 narratives above were WRONG

A 4-agent read-only forensics pass (vault backups, VLM/chart-OCR caches, manifests, live indexes,
rendered slides) falsified two of the stage-2 classifications. Recording so the wrong fixes are never
re-walked:

- **M4 DISSOLVES — nothing was lost.** (a) **sd-31**: the "4 design principles" (the slide labels them
  "Key Features") are PRESENT in the canonical md AND indexed (chunk `2f96ae1c-…#59c55cabec`); the W6
  re-parse changed the VLM prompt (`c911a80` → new prompt_sha8 → cache-key miss → fresh draws) but BOTH
  draws carry the identical bullets. The 06-01 VLM unification moved the content from a `[chart-extracted]`
  block into VLM page text — the audit's "0/8 manifest chart_extractions carry it" was a true observation,
  wrong inference. The actual failure is RERANK (4 same-doc "Architecture Diagram" distractor chunks
  out-rank the Key-Features chunk; not in top-10) + a STALE gold (`relevant_chunk_ids` empty with an
  FTS-unresolvable note that no longer applies — re-anchor to `#59c55cabec`). (b) **sd-04**: the TSMC
  chart's numeric series was NEVER parsed in ANY era (all 6 backups empty of it; Nemotron returned
  length-0 for all 5 crops in a fresh draw — and empties are cached BY DESIGN, so the chart-guide
  cache-replay precedent does NOT apply). Gold expects never-parsed data → re-label (qualitative half,
  already in-window at ranks 1/2/4) or an evidence-side chart-OCR-empty→VLM-escalation build (its own
  eval-gated increment). (c) **sd-16**: the gold chunk is rerank RANK-1 and contains the verbatim
  "**6x speedup since CUDA 11.8**"; the chart's x-axis literally labels its bars 1-6 (the "mis-OCR"
  hypothesis falsified by rendering the slide) — the failure is the answer LLM anchoring on the table
  ordinals and denial-framing → **M2-class**.
- **M6 REFRAMED — the SQL path WORKED.** For ar-14 the Table-RAG path fired end-to-end: the recompute-
  gated `#sql0001` chunk ("SUM of Fees Earned or Paid in Cash ($) = 956250 over 12 rows") was IN the
  answer window in both autopsy snapshots; the 4B then drafted the empty-claims denial ("…but do not
  provide a total"). The failure class is answer-node denial of the pre-verified synthetic aggregate —
  adjacent to M2, NOT retrieval/SQL routing. (The audit's stage-1 "RETRIEVAL-MISS" label was technically
  true of the raw table chunk but irrelevant — the synthetic chunk is the designed compensation.)
- **M5 refinements:** the live fast/auto window is **8192** (not 6144 — the 8B-era figure); the answer
  node's overflow degrade-loop drops the LOWEST-RANKED real chunk first, so pushing top_k to ≥15 at 8192
  deterministically evicts the rank-15 gold it was meant to admit (self-defeating) — sd-17/sd-25 are
  RERANKER-DEFICIENCY residuals (tiny 163-405-char gold chunks buried under long same-doc prose +
  cross-corpus 10-K lexical noise). The one measured hypothesis: `MEMEX_RERANK_TOP_K=8` for
  ftc-big-runmain (gold at #6; budget-verified ~7.5k worst-case at 8192).

**The shipped levers for the corrected map (branch `fix/m3-relevance-v3`):** M3 = assess_relevance v3
+ the deterministic world-knowledge override (handwritten-06 flips 3/3; tripwires hold 3/3). M2 = the
denial-reframe retry (ONE bounded re-draft through the existing v5 feedback slot when a zero-claims
draft denial-frames a present answer; the retry faces the full gate). M6 = the aggregate framing now
NAMES its source table (the ar-15 precedent). Stack validation: /tmp/stack_ladder.
