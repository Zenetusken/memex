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
