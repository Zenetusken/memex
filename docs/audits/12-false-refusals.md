# Audit 12 — False refusals (answerable queries wrongly refused)

**Date:** 2026-06-08 · **Trigger:** a Phase-2 confirmatory eval surfaced a false refusal
(linux-fundamentals-11); the user directed a thorough investigation + fixes, with the framing
**"a system that over-refuses when the answer is clear as day is as unreliable as one that
hallucinates."** · **Harness:** `scripts/false_refusal_audit.py`.

A **false refusal** = an ANSWERABLE eval query (`should_refuse=false`) the agent REFUSES. It does
not breach the HARD gate (it's a wrong refusal, not a hallucination) but it is a first-class
reliability failure — the answer is sitting in the retrieved chunks and the system says it can't
find it. The HARD gate (`refusal_cf` = counterfactuals correctly refusing) is **non-negotiable**
and was held at **1.0** through every shipped fix (validated multi-run, N=3).

## Method

`scripts/false_refusal_audit.py` runs each query through the REAL `answer_query` and records the
trifecta: (1) **outcome** (answered? refusal_reason?), (2) the **refusing NODE** (regex-attributed
from `refusal_reason` against the verified templates in `agents/answering.py`), and (3)
**gold-in-pool** from `FinalResponse.used_chunks` (the agent's ACTUAL reranked pool), refined to a
**gold-CONTENT check** (`_anchor_phrase` token presence in any pooled chunk) since a gold
chunk-id-in-pool does not mean the answer is *visible* (it can sit past a truncation budget, only
in a heading, or in a different pooled chunk). Borderline outcomes are measured **multi-run (N=3)**
per the eval-non-determinism rule.

## Baseline (v2 prompts, N=1, 13 answer corpora, 215 queries)

- **0 hallucinations** (HARD gate intact).
- **30 false refusals / 145 answerable** (~21%). By node: sufficiency **19**, relevance 6,
  compose-zero 3, verify-zero 2.
- **The dominant finding:** for **~20 of the 30**, the gold CONTENT is in the pool the gates saw
  — the system refused with the answer present — and it was **every gate** being over-conservative
  (sufficiency, relevance, the answer node), not one. Only ~7-8 are genuine retrieval misses.

## The architecture principle behind the fixes

The agent has **one** real safety gate: `verify` (grounding) + its numeric/name-only backstops.
Everything around it — `assess` (sufficiency, a PRE-filter) and `assess_relevance` (a post-verify
TOPIC check) — is advisory and should be biased toward letting the answer through, because a
counterfactual that slips a loose advisory gate still gets refused at `verify`. The fixes loosen
the advisory gates and surgically rebalance the answer node, leaving `verify` untouched.

### assess (sufficiency) — `v3` then `v4`
- **v3 (truncation fix):** v2 rendered chunks at `truncate(1200)` while answer/verify use
  `truncate(1800)`, so a fact at char 1200-1800 of a long chunk was invisible to the pre-filter
  that refuses first (proven: nist-10 "20 to 30" @1599, technical-01 `MemexError` @1432). v3 =
  v2 + `truncate(1800)`.
- **v4 (light pre-filter):** the gate was still *judging* sufficiency too strictly (refusing on
  perceived conflicts / "not specific enough" when the answer was present). v4 reframes its ROLE:
  strongly default to `sufficient: true`; refuse ONLY on OFF-TOPIC/empty retrieval; defer claim
  support to verify. HARD-gate-safe by construction; +5 recoveries, 0 regressions.

### answer node — `v4` REJECTED, `v5` shipped (the load-bearing lesson)
- **v4 (REJECTED, never shipped):** a broad "default to ANSWER" rebalance recovered +10 false
  refusals but **DETERMINISTICALLY breached the HARD gate** — french-course-07 ("how to configure
  a Cisco PIX firewall?", a deck with only GENERIC firewall content) fabricated *"the course says
  to configure a Cisco PIX firewall by … (any-any-deny)"* 3/3. A **named-entity substitution**:
  verify grounded the generic policy claim; nothing caught that it was attributed to a product the
  deck never names. **Lesson: loosening the answer node trades false-refusal recovery for
  substitution risk; the broad cue is unsafe.**
- **v5 (shipped, the SUBJECT-PRESENCE reframe):** lead with a subject test — is the specific
  subject the query names (product/model/vendor/dataset/value-type) actually present in the
  chunks? If only a GENERIC category or a DIFFERENT/neighbouring item is present → empty (the
  Cisco-PIX / FP4 / FP128 / Hopper-Blackwell traps). If the subject IS present, the answer counts
  as stated in ANY form — heading, table/figure cell, brief clause, different wording → draft (the
  heading/form/figure recoveries). **Validated: refusal_cf=1.0 across 70 counterfactuals × N=3
  (french-course-07 back to 0/3); recovers linux-11, forms-w9-07, technical-07, etc.**

### assess_relevance — `v2` (a WORLD-KNOWLEDGE-INJECTION grounding violation)
v1 refused grounded answers about NVIDIA's *fiscal 2026* figures with reasons like **"fiscal 2026
does not exist yet / has not yet occurred / making the claim factually impossible."** The 4B
applied its training-cutoff belief that 2026 is the future to OVERRIDE a grounded answer from the
user's own document — a fundamental RAG breach (the corpus is authoritative, not the model's world
knowledge) that would break ANY future-dated corpus. It refused annual-report-01/02/03/05/13 and
caused the answer/v5 annual-report-02 regression. **v2 = v1 + an explicit rule: the cited sources
are AUTHORITATIVE; do NOT apply outside/world knowledge about dates, what "exists yet," or what is
"factually possible"; judge ONLY topic match.** HARD-gate-safe by construction (relevance only acts
on already-grounded answers); annual-report figure questions recover 3/3 incl. the regression.

## Result (final bundle: assess-v4 + answer-v5 + relevance-v2)

- **refusal_cf = 1.0** — 0 breaches across 70 counterfactuals × N=3. (The rejected answer-v4 is
  the recorded breach lesson; nothing shipped breaches.)
- **False refusals: 30 → 18 (−12, −40%)** · **Correct answers: 115 → 127 (+12)** · **0
  regressions on this N=1 pass** (a regression is another false refusal, not a hallucination; the
  answer-v5 annual-report-02 regression was fixed by relevance-v2). The 12
  recovered: linux-11, forms-w9-07, technical-01, technical-07, cr350-img-02, scientific-gte-04,
  scientific-gte-05, chart-types-03, and the fiscal-2026 world-knowledge class
  annual-report-01/03/05/13. Remaining 18 by node: relevance 7, compose-zero 4, sufficiency 4,
  verify-zero 3 — the deferred retrieval-miss / unreadable-figure / genuine-partial-topic classes.
- 1859 unit/integration tests pass; the three prompt versions are pinned
  (`test_assess_sufficiency_prompt.py`, `test_answer_prompt.py`, `test_assess_relevance_prompt.py`)
  + the harness's node-attribution; kill-switches per gate
  (`MEMEX_PROMPTS__PIN__{ASSESS_SUFFICIENCY,ANSWER,ASSESS_RELEVANCE}`).

## Deferred (recorded negatives)
- **Retrieval-miss (~7-8):** gold content genuinely absent from the reranked pool → the
  retrieval/embedder lever (Phase-2.5), not a gate.
- **Answer-node hedging on unreadable figure data** (slide-decks-04/16: "data unavailable"):
  the answer node legitimately can't read the chart figure → a parse/figure-extraction concern,
  not a prompt fix.
- **chart-types-05:** a genuine partial/topic case (asks "why bar charts work *with maps*", answer
  is general about maps) — relevance correctly conservative.

## Notes on what "grounded" now includes
- **Heading-named answers are a DELIBERATE widening** (not a pure bug fix): answer/v5 treats a
  section title that names the answer ("Hard Links and Symbolic Links" → "the two kinds are hard
  and symbolic links") as literal presence, so verify now accepts that small inference. Defensible
  for a heading that unambiguously names the answer; recorded as a choice, not an accident.
- **The attribution-substitution class rests on the answer/v5 prompt** (verify is structurally
  blind to it — it checks claim-vs-chunk, not entity-match — and relevance was loosened). Guarded
  by 3 standing counterfactual tripwires (named entity ABSENT, generic category PRESENT):
  french-course-07 (Cisco PIX), nist-zero-trust-19 (Palo Alto firewall), slide-decks-34 (AMD
  MI300X) — all must refuse; a prompt edit or model swap that regresses the class trips them.

## Lessons
1. **Over-refusal is a first-class reliability failure**, weighed against hallucination — but the
   HARD gate (`refusal_cf=1.0`) is the hard constraint, so loosen the ADVISORY gates (which feed
   verify) freely; rebalance the answer node SURGICALLY (subject-presence, not "default to answer").
2. **A gate must never inject the model's world knowledge** to override grounded content (the
   fiscal-2026 case) — the corpus is authoritative.
3. **Validate multi-run.** The single hallucination that disqualified answer-v4 (french-course-07)
   was deterministic 3/3; the N=3 sweeps are what made the HARD-gate claims real, not N=1.
4. `gold-in-pool` ≠ answer-visible — check the gold CONTENT, the truncation budget, and headings.
