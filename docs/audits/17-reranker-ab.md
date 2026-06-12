# Reranker A/B — stages 0-2 + the layer attribution (2026-06-11)

**The arc:** bge-reranker-v2-m3 implicated by 4 converging case files (codex usage-class,
tiny-gold burial, the k=8-conditional pair, the k=8 capacity NO-GO). Measure-first, staged.

## Stage 0 — candidates (web-researched, adversarially card-verified)
Advanced: **mixedbread-ai/mxbai-rerank-base-v2** (494M, CrossEncoder drop-in, Apache, FR-tagged,
verified code/BEIR wins), Qwen3-Reranker-0.6B (wired in-repo; carries its 2026-05-21 loss),
zerank-1-small (1.7B, Apache, English-only docs). NO-GOs recorded: jina-v3 (CC-BY-NC + custom arch),
Qwen3-Reranker-4B (no co-residence path — the "AWQ" community quant is compressed-tensors → the
in-process decompress-to-dense trap), zerank-1 flagship (CC-BY-NC).

## Stage 1 — frozen-pool rank-delta probe (`scripts/reranker_ab_probe.py`, 31 queries × 3 models)
**mxbai SWEEPS: 10/10 case-file golds in top-5** (sd-31 #43→#3, sd-17 #15→#5, sd-25 #15→#4,
runmain/isknownsafe/unifieddiff #6→#2/#2/#4) **with 0/12 control escapes and the demotion-lever
regression pair HELD** — at ~incumbent CPU speed. qwen3-0.6B 7/10 (fails the headline caller class,
sd-25 #15→#24 — the in-house prior confirmed), 2.4× slower. zerank: host-RAM-killed, decision-moot.
(Probe lesson: 4 co-resident fp32 CPU models OOM-kill the host → one-resident-at-a-time.)

## Stage 2 — answer-path evals (mxbai, N=2, all deterministic)
- **slide-decks 16 ANS (+3: sd-04, sd-05, sd-25), cf=1.0 ×2.**
- **The 18-FR autopsy: 11/18 answer (vs 8)** — gains incl. **cr350-img-01, the synthesis-class
  "principled residual"**, + runmain AT k=5. Regressions: chart-types-01/-04 (the M-stack rescues
  undone), codex atc 30/37 raw.
- **codex decomposed by the layer probes (below): the raw atc UNDERSTATES mxbai** — 2 of the 4
  "new wrongs" were annotation artifacts (slots widened, `21071ca`); unifieddiff genuinely fixed.

## THE LAYER ATTRIBUTION (instrumented probes, N=2 deterministic each)
The unifying finding: **the 4B's draft/verify behavior is brittle to window COMPOSITION — not rank,
not k, not the reranker's correctness.** Three named mechanisms:
1. **Answer-model DEFINITION-ANCHORING** (isknownsafe, issafetocall): DEF visible at #1 → the model
   describes it even with the caller at #2. **mxbai + `usage_intent_demotion_enabled` FIXES both**
   (leg B drafts the caller; and mxbai's ranks invert the lever's old risk — its former victims
   convhistory/applyhunks sit #5/#3 PRE-demotion).
2. **Verify BATCH-COMPOSITION flips** (ct-01): the IDENTICAL claim + gold@#1 grounds under bge's
   window company, rejected under mxbai's. The M1 retarget fired and failed — its candidate set is
   the TOP-5 window and the reshuffle removed the sibling the bge-era rescue grounded via.
   **Candidate fix: widen the retarget probe set to top-10** (promote-only, final-pass-only).
3. **Answer-model drafting flips on adjacent content** (ct-04; the sd-16 family): gold@#1 yet the
   draft flips to a denial purely from different neighbors; the M2 retry fires and re-denies.
   The honest model-capability residual.
Annotation layer: newstdio (snake-vs-camel token), ftc-def-assess (verb inflection) — answers
verified correct, slots calibrated. Unattributed pending the ladder: rollout, appnew.

## Stage 3 (pending): the full 14-corpus ladder at the candidate config
**`models.reranker=mixedbread-ai/mxbai-rerank-base-v2` + `usage_intent_demotion_enabled=true`**
(+ the retarget top-10 widening as a small pre-validated addition). Gates: cf=1.0 ×28, ANS ≥
baseline per corpus (the FR corpora are mxbai's real exam — its one below-incumbent benchmark),
codex atc ≥ baseline, eval-summary 6/6. Ship = config-only default flip with bge as the
kill-switch; the k=8 NO-GO discipline applies (probe-clean ≠ ladder-clean).

## The summary-scope guard v1 — CATASTROPHIC over-refusal, REVERTED (2026-06-12)

The guard concept validated on the breach pair (ar-12/tg-13 refuse 2/2; ar-14/hw-06 hold) but the
full ladder measured **net −107 ANS**: the bigram trigger (query∩summary bigram absent from claims)
fires on INNOCENT shared phrasing ("compilation process", "zero trust"), and the deterministic
relevance refuse converted every false-positive into a refusal. All 3 commits reverted; 1984 green.
**LESSONS:** (1) a deterministic REFUSE amplifies its trigger's false-positive rate corpus-wide — a
4-case validation cannot price a base-rate; ANY new deterministic gate needs a ~30-query mini-sweep
BEFORE a ladder; (2) my ladder grader gated cf/errors but not ANS — graders must encode EVERY gate.
**The tightening candidate (unbuilt):** require the unsupported subject-bigram to be absent from the
CITED CHUNKS too (ar-12/tg-13 still fire — the subject is genuinely absent from the evidence; legit
paraphrase answers don't — the chunk usually carries the subject). Its own increment, mini-sweep-gated.
**Standing state:** bge remains the shipped default (no breaches); the mxbai config CANNOT ship while
the scope hole is open. The hole itself is PRE-EXISTING and reachable under bge (less often) — a
first-class finding independent of the reranker verdict.
