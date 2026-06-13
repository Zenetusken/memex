# ADR-0022: Advisory-Gate Rebalance — Treating Over-Refusal as a First-Class Failure

- **Status**: Accepted (shipped 2026-06-08, `0d76ace`)
- **Date**: 2026-06-08
- **Deciders**: Memex core team
- **Tags**: agent, grounding, prompts, reliability, process

## Context

The no-hallucination HARD gate (`refusal_cf=1.0`) is the project's bright line, and every gate change
to date has been defended for *tightening* it. But the dual failure had gone unmeasured: a **false
refusal** — an answerable question the vault *can* answer, refused (`answered=false` AND
`refusal_correct=false`). The user's framing is the premise of this ADR: *"a system that overly
refuses when the answer is clear as day is as unreliable as one that hallucinates."* A false refusal
is HARD-gate-*safe* (a wrong refusal, never a fabrication) but it directly hurts utility
(answered-count / recall), and at scale it makes the system feel broken.

A dedicated diagnostic (`scripts/false_refusal_audit.py` — per answerable query: `answer_query` →
answered? + the refusing *node* (regex over the `refusal_reason` templates) + gold-in-pool from the
agent's own `FinalResponse.used_chunks`) measured the baseline across 13 answer corpora: **0
hallucinations (HARD gate intact)** and **30 false refusals / 145 answerable** — split into
retrieval-miss (gold never reranked) and **gate over-refusal** (gold in pool, refused anyway).

**The enabling architectural insight:** in the `/ask` graph, `verify` (grounding + the
numeric/name-only backstops) is the **sole HARD gate** — no ungrounded claim leaves it. `assess`
(sufficiency, a *pre*-filter) and `assess_relevance` (a *post*-verify topic check) are **advisory**:
they can only *withhold* a draft that `verify` would have gated anyway. So an advisory gate can be
loosened **without any path to an answered counterfactual** — `refusal_cf` cannot regress by
construction. This is what makes the over-refusal fix safe to pursue at all.

## Decision Drivers

- **`refusal_cf=1.0` is non-negotiable** — every fix must hold the HARD gate, multi-run.
- **Loosen only what is advisory** — the safety argument is structural (verify still gates every
  claim), not empirical hope.
- **Rebalance the answer node *surgically*** — the one node that *can* breach (it drafts claims) must
  not be loosened with a blunt "default to answer."
- **Remove the bad bar only** — the `assess_sufficiency@v2` lesson: when fixing an over-strict gate,
  do not also re-tighten "to stay safe" (that nets negative).
- **A gate must never inject the model's world knowledge** to override grounded corpus content.

## Considered Options

1. **Loosen the advisory gates + surgically rebalance the answer node** (chosen).
2. **A broad "default to ANSWER" rebalance of the answer node** — *built then REJECTED*: it
   DETERMINISTICALLY breached the HARD gate (see below).
3. **Do nothing** — leave the 30 false refusals; rejected (the user directive: over-refusal is a
   first-class failure).
4. **Chase individual borderline flips by re-tuning a gate** — rejected as whack-a-mole; borderline
   `refusal_cf` flips run-to-run under greedy AWQ (the standing measure-multi-run rule).

## Decision

**Treat over-refusal as a first-class reliability failure and fix it across the advisory gates, with
the answer node rebalanced surgically.** Three versioned, kill-switchable prompt changes
(`MEMEX_PROMPTS__PIN__{ASSESS_SUFFICIENCY,ANSWER,ASSESS_RELEVANCE}`); the mechanism prose lives in
`src/memex/CLAUDE.md` and the audit, this ADR records the principle.

- **`assess_sufficiency` → v3 then v4** — v3 aligned the gate's per-chunk view with the answer/verify
  nodes (`truncate(1200)→1800`; a fact in the 1200–1800 dead-zone was invisible to the pre-filter but
  visible downstream). v4 reframes the *role*: a **light pre-filter** that strongly defaults to
  `sufficient: true` and refuses only OFF-TOPIC/empty retrieval, explicitly deferring "is the answer
  fully supported" to the grounding step.

- **`answer` → v5, the SUBJECT-PRESENCE reframe** — *not* "default to answer." Lead with a subject
  test: is the specific subject the query names actually present in the chunks? If only a generic
  category or a neighbouring item is present → empty (the named-entity/numeric substitution traps); if
  the subject *is* present, the answer counts as stated in any form (heading / table / figure / brief
  clause / different wording) → draft (the heading/form/figure recoveries).

- **`assess_relevance` → v2, the WORLD-KNOWLEDGE ban** — the cited sources are AUTHORITATIVE; the gate
  must **not** apply outside/world knowledge about dates or "which years exist yet." This fixed a
  genuine grounding violation: v1 refused grounded fiscal-2026 figures reasoning "fiscal 2026 does not
  exist yet" — the model's training-cutoff belief overriding the grounded corpus.

## Consequences

### Positive

- **False refusals 30 → 18 (−40%); correct answers 115 → 127 (+12); 0 regressions;
  `refusal_cf=1.0` across 70 counterfactuals × N=3.** Utility up, the HARD gate untouched.
- The **world-knowledge-injection** finding is the deeper catch: a gate must never let the model's
  training-cutoff beliefs override grounded content — a general rule now encoded in the prompt.
- The advisory-vs-HARD-gate distinction is now explicit, so future over-refusal work has a safe lever.

### Negative / Trade-offs

- **A broad "default to ANSWER" answer-node rebalance (the rejected v4) DETERMINISTICALLY breaches the
  HARD gate** — it fabricated "the course says to configure a Cisco PIX firewall by …" from *generic*
  firewall content, 3/3. A named-entity substitution is grounded at the claim level and missed by
  relevance, so the answer node cannot be loosened bluntly. v5's subject-presence framing is the
  surgical alternative; v4 was deleted. (Pinned by `tests/unit/test_answer_prompt.py`.)
- Three more prompt versions to carry; each has a per-gate kill-switch and a content-pinning test.

### Neutral

- The fixes are prompt-only on advisory gates + the answer node; the `verify` HARD gate and the
  numeric/name-only backstops are untouched, so `eval-summary` is byte-stable.

## Revisit When

- A new corpus surfaces a deterministic gate over-refusal with the gold in-pool — loosen the bad bar
  only, validate multi-run.
- The residual 18 false refusals are addressed: ~retrieval-miss (an embedder/recall lever, Phase-2.5),
  unreadable-figure-data, and genuine partial-topic / synthesis-class (co-located evidence needing a
  one-step inference — the reason-over-evidence direction, deferred; cf. ADR-0016 and the reverted
  contextual-retrieval experiment).
- Any future answer-node loosening is proposed — re-confirm against the named-entity/numeric
  substitution tripwires before shipping.

### Update 2026-06-13 — the summary-scope hole (reranker A/B arc, audits 17/18/19)

The sharper-reranker A/B (`mxbai-rerank-base-v2`) surfaced a structural gap in this same advisory layer:
the draft **summary** re-attributes a TRUE grounded claim to the question's subject, while `verify`
gates the *claims* and nothing gated the *framing*. It splits two ways:

- **Provenance class** (e.g. tg-13, a claim mis-cited "according to the developer guidelines") — **FIXED +
  SHIPPED default-ON**: a deterministic doc-identity check at the TOP of `assess_relevance`
  (`core/text.extract_provenance_source` + `provenance_tokens_match` + `FTSStore.document_identities()` →
  `_provenance_scope_violation`; kill-switch `MEMEX_AGENTS__PROVENANCE_SCOPE_ENABLED`). It is an advisory
  addition with no path to an answered counterfactual (`refusal_cf` can't regress); full-ladder PASS
  (cf=1.0×28). `verify` and the backstops are untouched. See `docs/audits/18-summary-scope-guard.md`.
- **Content class** (ar-12 binding fabrication: a true *consolidated* figure FALSELY bound to a named
  part) — **DOCUMENTED RESIDUAL, do-not-re-walk**: a gate-detection fine-tune was a K2 structural
  negative (question-echo entanglement) and a generation-time `answer@v6` "whole-is-not-part" prompt
  clause was reverted (the THIRD instance of the answer-prompt-blast-radius lesson above — it regressed an
  unrelated linux-18 counterfactual under the shipped bge env). The breach is mxbai-reachable only, so
  production exposure is UNCHANGED and `mxbai` stays blocked; the only remaining instrument is far-future
  complete-evidence-entailment / masked-subject machinery. See `docs/audits/19-binding-checker-design.md`
  §9 (K2) + §10 (the v6 revert).

## References

- `docs/audits/{17,18,19}` — the reranker A/B, the shipped provenance-scope backstop, and the
  content-class binding-checker K2 negative + `answer@v6` revert (the 2026-06-13 update above).

- `docs/audits/12-false-refusals.md` — the full taxonomy, per-class counts, root causes, and the
  three attribution-substitution tripwires (nist-zero-trust-19, slide-decks-34, french-course-07).
- Mechanism + lessons: `src/memex/CLAUDE.md` (the assess / answer-prompt / relevance-gate bullets).
- Prior art: the `assess_sufficiency@v2` citation-floor over-refusal fix (2026-06-06) and the
  measure-multi-run / change-one-bar lessons it established.
- The standing rules: the **measure-don't-assert** discipline and the borderline-gate multi-run rule
  (CLAUDE.md → "Measure, don't assert").
- Commit: `0d76ace` (merge); `e16b183` (the fix bundle).
