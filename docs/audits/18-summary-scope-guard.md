# Audit 18 — The summary-scope guard: semantic-seam research + the Step-1 kill-test

**Date:** 2026-06-12 · **Status:** kill-test COMPLETE across ALL SEVEN arms (in-house
+ the user-authorized doc-trained checkers, §8) — no off-the-shelf scorer prices the
content-class binding with a usable threshold; the provenance class has a deterministic
in-house lever. Verdict + the production-exposure note in §9.

Predecessor: `docs/audits/17-reranker-ab.md` (the summary-scope HOLE: the draft summary
re-attributes a TRUE grounded claim to the question's subject; verify gates claims only;
three lexical guards failed → "the fix must be semantic"). This audit records the
research synthesis for the semantic seam and the measured kill-test of every cheap arm.

## 1. The research synthesis (4-agent workflow, 2026-06-12)

Full output: `data-17-scope-calibration/` + the conversation record. Headlines:

- **Structural rule:** the check must be EVIDENCE-side (does the cited evidence entail
  the summary AS WRITTEN?), never QUESTION-side (RAGAS-style answer-relevancy passes
  the breaches by construction — the summary is question-matched BY the re-attribution).
- Entity re-attribution is a NAMED error class (FactCC "entity swap"; FRANK/AggreFact);
  small doc-trained grounding checkers are trained on it: **HHEM-2.1-Open** (110M,
  Apache-2.0, CPU), **MiniCheck** (355–770M, MIT), **LettuceDetect** (210/610M, MIT,
  EuroBERT multilingual = FR coverage), AlignScore. Vanilla MNLI degrades on chunked/
  noisy premises (the SummaC granularity finding) — prefer doc-trained checkers.
- 8 candidate designs ranked; the recommendation was a two-stage guard (restored v3
  lexical span TRIGGER + resident bge span-as-query CONFIRMATION + 4B band judge),
  advisory-layer only, contingent on a pre-registered kill-test. RARR-style
  repair-not-refuse deferred (conflicts with counterfactual gold semantics).
- Step 0 executed: the volatile /tmp evidence (final_ladder, prose_sweep, mxbai_stage2,
  breach traces) preserved under `data-17-scope-calibration/raw/`.

## 2. The kill-test setup

`scripts/scope_guard_span_probe.py` (committed; subcommands `capture`/`score`/`nli`/
`judge`/`hhem`). Breach side: the 2 frozen traces (ar-12, tg-13; cited chunks resolved
from search.sqlite — ar-12 → the 10-K BUSINESS OVERVIEW chunk `…b#6688b1c5c3`, tg-13 →
the codex-rs `tui/src/log_layer.rs` `const _DEFAULT_MAX_LEN` chunk `…r#a270b0ee35`).
FP side: 12 of the 20 recorded v3 mini-sweep false-positive qids re-captured LIVE under
the shipped bge default (observe-only recorders, the fr_autopsy pattern): **12/12
answered, 12/12 trigger-fired** (22 spans) — the trigger's near-total false-fire rate on
this class re-confirmed, incl. the sharpest pair: tg-01 extracts the EXACT breach span
`developer guidelines` with legit cited evidence. FR spans confirmed noisy (`la cyber`,
`reconnaissance de` — the English `_SUBJECT_STOP` gap).

Probe script independently reviewed before the run (no real defects; CrossEncoder
`.predict` = sigmoid probs, monotone, so ordering verdicts are activation-safe).

## 3. Results per arm (per-span max-over-cited-chunks unless noted)

### 3a. bge-reranker-v2-m3, span-as-query (`{chunk_title} — {text}`) — the design's stage 2

| side | spans | scores |
|---|---|---|
| BREACH ar-12 | `graphics segment` / `nvidia graphics` / `margin nvidia` / `segment fiscal` | **0.725 / 0.845 / 0.961 / 0.707** |
| BREACH tg-13 | `developer guidelines` | 0.000 |
| FP (22 spans) | — | 0.000 … 0.986 (9 spans ≤ 0.015) |

**Margin = −0.961; 21/22 FP at-or-below the top breach span. INVERSION, not sliver —
monotone rescoring cannot resurrect it. DEAD.** (Bare-text form identical: −0.956.)

### 3b. EmbeddingGemma cosine / delta — overlap 17/22 and 19/22. DEAD.

### 3c. Vanilla NLI (DeBERTa-v3-large-mnli-fever-anli-ling-wanli), summary-vs-cited-premise

| side | whole-summary P(entail) |
|---|---|
| BREACH ar-12 / tg-13 | **0.260 / 0.027** — both correctly LOW |
| FP supported-prose (7) | 0.666–0.997 — correctly HIGH |
| FP false-lows (4–5) | ar-03 **0.143** (premise = GFM table), linux-01 **0.033** (markdown bullets), xref-02 **0.003** (FRENCH, near-verbatim supported), nist-05 **0.055** (provenance framing), ar-05 min-sent 0.114 |

**Direction CORRECT on both breaches; the instrument fails on non-prose/FR premises —
exactly the documented vanilla-MNLI weakness doc-trained checkers exist to fix.** The
false-lows are format artifacts, not hard negatives (verified by reading the tuples).
Mitigation cell for the next probe: feed the scorer the `linearize_gfm_tables`
verify-render instead of raw chunk text.

### 3d. The 4B evidence-only judge ("does the evidence mention <span>?", bool-only, N=2)

Misses `graphics segment` (mentioned=true 2/2 — the pre-registered kill) AND false-fires
on **18/22** FP spans (e.g. `sp 800-207` judged not-mentioned on an SP 800-207 chunk —
extractively true, provenance qualifiers aren't in chunk bodies). Stable N=2. **DEAD in
both directions.**

### 3e. Doc-identity (deterministic, separator-normalized substring; report-only)

`sp 800-207` matches its NIST doc 2/2 (clears the FP); tg-13's `developer guidelines`
vs `tui/src/log_layer.rs` correctly FIRES under every variant; tg-01's doc is named just
"guidelines" → needs any-token matching; junk cross-boundary spans (`800-207 policy`)
need cleaner span extraction (the artifact-scope positional-qualifier machinery).
**The mechanism is real for the provenance class; the engineering is span quality +
match normalization.**

## 4. The diagnosis revision (the probe's key discovery)

The ar-12 cited chunk **contains the Graphics segment** — 12× "graphics", a reportable-
segments table with segment REVENUE, and "(2) Segment operating income differs from
consolidated…". The consolidated "Gross margin was 71.1%" sentence sits beside it. The
v3 bigram only fired because "Graphics segment" never appears ADJACENTLY — luck, not
design (one adjacent occurrence anywhere and the trigger silently misses the breach).

Consequence: **subject-PRESENCE is the structurally wrong test for this breach class.**
bge (0.73–0.96) and the 4B judge (mentioned=true) answered presence CORRECTLY — the
fabrication is the predicate-subject BINDING ("gross margin OF the Graphics segment"),
which only proposition-level entailment over the summary can test. The breach class
splits:

1. **Provenance-class (tg-13):** the summary asserts a false SOURCE ("according to the
   developer guidelines" cited to a Rust file). Deterministic doc-identity check —
   in-house, cheap, no model.
2. **Content-class (ar-12):** the summary fabricates a predicate-subject binding while
   the subject is genuinely present in the evidence. Needs a doc-trained entailment
   checker over the summary verbatim (HHEM/MiniCheck/LettuceDetect); vanilla NLI proves
   the direction, fails on format.

## 5. Verdict

- The recommended two-stage design (lexical trigger + bge confirmation + 4B band judge)
  is **DEAD at L0 — do not re-walk.** All three in-house semantic arms measured
  non-separating; two of them for the structural reason in §4.
- The lexical bigram trigger is additionally **luck-dependent on the content class**
  (adjacency-sensitive) — any successor design must not rely on it for breach recall.
- The surviving directions: **(a)** a deterministic doc-identity backstop for the
  provenance class (buildable now, mini-sweep-gated, the numeric-backstop precedent);
  **(b)** a doc-trained entailment checker scoring the summary against the cited
  evidence for the content class — the research's #2 design, now promoted to primary.

## 6. The blocking decision (user)

Probing the doc-trained checkers requires running external model code, which the
session's permission policy correctly flagged: HHEM-2.1-Open loads via
`trust_remote_code=True`; MiniCheck/LettuceDetect ship as pip packages. The probe's
`hhem` subcommand is written and ready; the frozen tuples make each checker scoreable
in minutes on CPU. Options: authorize one/all checkers; or proceed with the
doc-identity lever only and accept the documented content-class residual.

## 8. The doc-trained checker round (user-authorized, same frozen 14 cases)

All three checkers from the research's #2 design were scored. Ops notes: the
`minicheck`/`lettucedetect` pip overlays resolved broken torch combos, so both run as
FAITHFUL REIMPLEMENTATIONS transcribed verbatim from the cached package sources (input
templates, label ids, span grouping — see the probe's `minicheck`/`lettuce`
subcommands); HHEM ran via `trust_remote_code` as published. The
`linearize_gfm_tables` mitigation cell returned NO-SIGNAL — the linearizer no-ops on
these chunk bodies (raw == linearized premises), so the table-premise question stays
open at the linearizer level, not answered.

### 8a. HHEM-2.1-Open (whole-summary entailment) — DEAD

ar-12 scores **0.879 = confidently supported** (its own pre-registered kill: an
overlap-biased scorer passing the ~2-token-delta breach). tg-13 0.27 but 5/12 FPs score
lower. Margin −0.744.

### 8b. MiniCheck-Flan-T5-Large — the only arm that catches BOTH breaches; FP-swamped

Breaches **0.121 / 0.149** — correct direction on both, including ar-12 (unique among
all seven arms). But **7/12 FPs score below both breaches** (0.019–0.089), concentrated
in TABLE/list-derived answers (e.g. ar-03's "$193.737 billion" derives from a table
cell "193,737" — a unit transform no entailment scorer survives). Margin −0.130. No
usable threshold; a table-exemption rule would exempt ar-12 itself (its chunk contains
the segment table).

### 8c. LettuceDetect (ModernBERT, RAGTruth) — cleanest FP profile; binding-blind

With the REAL question (its trained shape): **9/12 FPs flag NOTHING** — including every
table/bullet/FR case that broke NLI/MiniCheck. The 3 false-fires are all one systematic
cause: the Memex answer style's citation tail ("…as stated in the 2026 Annual Review
(Form 10-K)"), absent from RAGTruth answers. Both breaches flag (0.59 / 0.96) — BUT the
**tail-strip sub-test kills the ar-12 catch: stripped of the citation tail, ar-12 →
0.000, no spans.** Its flagged spans were the tail fragments, never "Graphics segment";
the content-class catch was style noise. (The no-question variant is unusable —
12/12 FP overlap — confirming the question-echo training dependence.) tg-13's flag
survives stripping (0.957), though on the supported spans (`log layer`, `log_layer.rs`)
rather than the false provenance — usable signal, wrong explanation.

## 9. Final verdict (all seven arms measured)

| arm | ar-12 content breach | tg-13 provenance breach | FP false-fires |
|---|---|---|---|
| bge span-as-query | **MISS** (0.73–0.96) | catch (0.00) | 9/22 spans ≈0 |
| EmbeddingGemma cos/delta | overlap | overlap | 17–19/22 |
| 4B evidence judge | **MISS** (mentioned=true 2/2) | catch | 18/22 |
| vanilla NLI | catch (0.26, thin) | catch (0.03) | 4–5/12 (format) |
| HHEM-2.1-Open | **MISS** (0.88) | weak (0.27) | 5/12 below it |
| MiniCheck-FT5-L | catch (0.12) | catch (0.15) | **7/12 below both** |
| LettuceDetect+q | **MISS** (tail-strip→0.0) | catch (0.96) | 3/12 (all citation-tail) |

- **Content class (binding fabrication): no off-the-shelf scorer separates it on our
  distribution.** The only catchers (MiniCheck, NLI) drown in table/unit-transform
  false-lows. The remaining instrument is the banked in-domain fine-tune recipe
  (TinyLettuce/FactCC entity-swap synthetic training, research design F) — now with a
  frozen calibration set to gate it. Until then: a DOCUMENTED RESIDUAL.
- **Provenance class: the deterministic doc-identity check stands** (catches tg-13
  under every variant; clears the `sp 800-207` FPs with separator-normalized matching;
  needs any-token matching + provenance-shaped span extraction). This is the buildable
  increment.
- **Production exposure note:** under the SHIPPED bge default the live breach is
  tg-13 — the provenance class (`raw/prose_sweep`: answered=True under bge; ar-12
  correctly refuses under bge and breaches only under the mxbai env). So the
  doc-identity backstop closes the production hole; the content class is currently
  mxbai-gated, which also keeps mxbai blocked.
- LESSON (ops): pip-overlay envs (`uv run --with`) routinely resolve broken torch
  combos against this project env — transcribe small checker recipes from the cached
  package source instead (the MiniCheck/LettuceDetect pattern; verbatim templates +
  label ids, measurement-faithful).
- LESSON (instrument): a span detector's verdict is only as good as WHERE it fires —
  always autopsy the spans (the tail-strip sub-test), never trust case-level
  confidences alone.

## 10. Artifacts

`data-17-scope-calibration/`: `scope_probe_fp.json` (12 live FP tuples: summaries,
claims, cited chunks, spans), `scope_probe_scores.json` (bge+embedder),
`scope_probe_nli.json`, `scope_probe_judge.json`, `scope_probe_hhem.json`,
`scope_probe_minicheck.json`, `scope_probe_lettuce.json`, `scope_probe_capture.log`,
`raw/` (the preserved /tmp evidence incl. both breach traces). Probe:
`scripts/scope_guard_span_probe.py` (subcommands: capture / score / nli / judge /
hhem / minicheck / lettuce).
