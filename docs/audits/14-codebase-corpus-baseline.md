# Codebase corpus — full ingest + grounded-answering baseline (2026-06-09)

**Verdict: the codebase corpus is an askable domain — retrieval + the HARD gate validated; answer
correctness strong on definitions, with a measured wrong-chunk gap on usage-class answers.** Phases 4–5
of the code-as-documents arc (ADR-0021): the full `codex-rs` (99 `.rs`) lives in the MAIN vault alongside
the prose corpora, the HARD gate holds on code (`refusal_cf`=1.0, N=3), retrieval recall@50 = 1.0, and
adding code did NOT perturb the prose gates. Retrieval was solved in Phase 3 (Lever A, `docs/audits/13`);
this is the first end-to-end test of the full `/ask` on code — and a spot-check found the answer stage
still prefers a title-matching DEFINITION over the usage gold on ~1–2 of 8 usage queries (the next
frontier; see §2). No hallucinations.

## Setup

- **Corpus:** the full `codex-rs` (99 `.rs`, dedup-clean) ingested into the MAIN vault → ~1235 code chunks
  among 7356 total (prose intact; the vault was backed up first). The find-the-code query set grew to
  **47 queries** (`tests/eval-data/codex-rs-find-the-code/`): 16 definition + 17 usage/reference + **6
  big-function truncation probes** (gold chunk > 1800 chars) + 8 counterfactuals.
- Device-pinned (`co_residence_mode=manual`, embedder/reranker CPU), 4B orchestrator, Lever A default-ON.

## (1) Retrieval recall@k — `scripts/code_bm25_arm_probe.py` on the MAIN vault (39 ANS)

| arm | @5 | @10 | @20 | @50 |
|---|---|---|---|---|
| dense | 0.692 | 0.769 | 0.821 | 0.923 |
| bm25-phrase (LIVE prose default) | 0.026 | — | — | 0.026 |
| bm25-term-whole | 0.692 | 0.872 | 0.897 | 0.949 |
| **union (ceiling)** | 0.872 | 0.974 | 0.974 | **1.000** |
| **rrf@50 (PRODUCTION fusion)** | 0.667 | 0.872 | 0.974 | **1.000** |

Per-regime @50: **definition** dense 1.00 (Phase-2 title lever saturates even against 7356-chunk prose
competition — BM25 redundant); **usage/reference** dense 0.870 → union **1.000**, Δ=+0.130, **3/3
dense-misses recovered only by bm25-term**. **Production recall@50 = 1.000** — every code gold lands in
the rerank pool. Lever A's contribution is the usage/reference regime, as Phase 3 found.

## (2) Grounded code answering — `memex eval` (47 queries; gate N=3; answer-correctness spot-checked)

**38/39 ANS *answered* (97%) · `refusal_cf` = 1.0 · mean citation_precision 0.771 (answered-only 0.717) ·
0 errors** (the answered_count is a single full run; the gate is N=3 and the borderline ANS — runmain,
mcpstdio — were N=2 spot-confirmed). NB `memex eval` scores answered/refused + `cp`, NOT answer *text*
correctness, so the headline below is qualified by a manual spot-check. **(CLOSED 2026-06-09: `memex
eval` now grades answer TEXT via the deterministic `answer_must_mention` → `answer_mention_recall`
metric — whole-token boundary match, any-of slots; the 39 ANS queries are annotated. First baseline
(N=2 BYTE-STABLE): 32/38 text-correct, mean recall 0.855; the 6 deterministic flags = the 3 usage-class
definition-distraction answers below + 3 honest def-class partials (modelclient misses the streaming
purpose; nl-safe never names the decision function; ansiescapeline misses the spans transform). The
metric reproduces this section's manual ground truth exactly — incl. PASSING the 2 valid-sibling answers
GOLD_CITED false-negatived. See `_answer_text_baseline_2026_06_09` in queries.json.)**

### Answer-correctness spot-check (N=8, usage-weighted — the advisor's required check)

`cp`=0.717 (answered-only) is NOT all the sibling artifact — a manual spot-check of 8 answered queries
(weighted to the usage class + `cp<1.0`) against `_expected_answer` found **6/8 correct, 2 usage-class
wrong/partial**:
- `ftc-use-isknownsafe` — **WRONG**: "which function CALLS `is_known_safe_command`?" was answered by
  *describing `is_known_safe_command` itself* (the definition), not its caller `assess_command_safety`.
- `ftc-use-unifieddiff` — **PARTIAL**: named the *test* callers, missed the gold production caller
  `maybe_parse_apply_patch_verified`.
- Correct (6): `seekseq`/`newstdio` (cross-crate)/`ansiescape` (cross-crate)/`applyhunks`/`mcpstdio`
  (correct DESPITE the >1800 truncation) / the `applypatchaction` definition.

**The finding:** Lever A solved *retrieval* of the usage site (it's in the pool), but at the ANSWER
stage the reranker + answer LLM still often prefer the **title-matching DEFINITION** over the usage gold
— so "which function calls X" is sometimes answered "X is defined here and does Y" (wrong) on the usage
class (~1–2 of 8). This is the **next frontier — an answer-stage lever, beyond Lever A's retrieval
lever** — recorded, not fixed here. Definitions + the easy usages answer correctly. So the honest
headline is: **retrieval recall@50 = 1.0 and `refusal_cf` = 1.0 are validated; answer correctness is
strong on definitions but has a real wrong-chunk gap on usage/reference answers.**

- **The HARD gate holds on code.** All 8 counterfactuals refused on EVERY run (N=3) — including the ones
  with 37–50 lexical retrieval hits (`computetax`/`jwt`/`renderpdf`/`oauth`). Lexical hits do not breach
  the gate: no real gold → no grounded claim → refuse, with sensible reasons ("None of the retrieved
  chunks mention a Quantum…").
- **One deterministic false-refusal: `ftc-big-runmain`** (`exec/lib.rs::run_main`, 4465 chars) — refused
  N=2, reason "addresses a related topic but no direct answer" (the `assess_relevance` gate). It is **NOT
  a truncation casualty**: the TWO BIGGER probes (`mcpstdio` 4647, `resolveargs` 4912) answer fine.
  run_main is an orchestration ENTRY POINT the relevance gate judges non-responsive — the same borderline
  class as the prose corpora's documented residuals. Recorded, not fixed.
- `citation_precision` 0.77 is PARTLY the single-gold sibling artifact (a valid sibling cited, answer
  correct) but — per the spot-check above — PARTLY real wrong-chunk usage answers. Do not attribute the
  whole `cp` gap to the artifact.

## (3) The truncation risk — MEASURED, largely BENIGN

~6% of code chunks exceed the answer/verify `truncate(1800)` budget (73/1235, max 5960). The 6
big-function probes test it: **5/6 answer correctly despite the clip**, including the two LARGEST. The
lone failure (runmain) is content/relevance, not size. So `truncate(1800)` does not broadly break
grounded code answering — a measure-first NEGATIVE: no code-specific truncate budget / symbol
sub-chunking fix is warranted now (would be its own measured increment if a real truncation class appears).

## (4) Shared-vault prose-gate re-verify — WITH code present

The advisor's required check: the detector-triggering prose queries re-run with codex-rs in the vault
(`model_provider_info.rs` carries openai/deepseek URLs, live against `scientific-gte-16` "OpenAI
dimension"). N=2: **all 3 prose counterfactuals refused both runs (`refusal_cf`=1.0)**, all 5 prose ANS
byte-stable vs the pre-ingest baseline. Adding code did NOT make a prose counterfactual answerable; the
Lever A × code interaction is safe.

## The usage-class answer gap — root-caused + a measured DOUBLE-EDGED lever (2026-06-09)

The spot-check residual above was investigated end-to-end (the user: "get to the bottom of the gap and
remedy this"). **Root cause (probe-validated, MAIN vault, device-pinned):** two stacked mechanisms.
(1) **Rerank cutoff** — the cross-encoder (bge-reranker-v2-m3) is a TOPICAL scorer, so for "which function
calls X" it ranks the chunks containing X most prominently — X's own **definition** (#1, ~0.95) and its
**test** chunks (#2–#5) — above the real **caller**, which mentions X once at the call site; across the 17
usage queries the gold caller reranks to #6 (out of the top-5) in exactly the 2 worst cases. (2)
**Answer-LLM distraction** — even with the caller forced to #1, the answer LLM still describes X's
definition whenever a definition chunk of X is anywhere in its visible top-k window (answer/v5's
subject-presence is satisfied by the definition); remove the definition + tests from the window and it
answers the caller correctly.

**The remedy (built, then measured at scale):** a **usage-intent rerank-demotion** lever
(`index/code_query.detect_usage_intent` + `reorder_for_usage_intent`, wired into the `rerank` node) that,
for a "which/where … calls/uses/… X" query, demotes X's definition + test chunks below the top-k cut so
the answer node grounds on the caller. Pure reorder ⇒ HARD-gate-safe by construction; the detector fires
on 17/17 usage queries, 0/16 definition + 0/6 big-function (validated on all 47). The 2-query probe
(isknownsafe, unifieddiff) confirmed the fix.

**But the full 17-query, N=2, text-verified measurement found it DOUBLE-EDGED — net +3 / −2:**

| flag-OFF → flag-ON | queries | verdict |
|---|---|---|
| genuinely wrong → correct | isknownsafe, unifieddiff, issafetocall | **+3 real fixes** (LLM was describing the definition) |
| correct → **wrong answer** | convhistory | **−1 regression** (demoting the `ConversationHistory` definition removed the disambiguating context → the LLM picked the similarly-named distractor `ChatWidget`/`ConversationHistoryWidget`) |
| correct → **false refusal** | applyhunks | **−1 regression** (demoting the def + 13 test chunks over-stripped context; gold stayed at #2 but the LLM refused) |
| already-correct sibling (no change) | getplatform, rollout | GOLD_CITED false-negatives, not failures |
| correct → correct | the other 10 | unchanged |

The root finding: **demoting X's definition is double-edged — a distractor when the LLM anchors on it
(def@top, caller buried), but NECESSARY CONTEXT when a similarly-named sibling needs disambiguating or the
subject needs anchoring.** No clean rerank rule separates the two at inference time (a "demote-def-only-
if-#1" gate spares convhistory but applyhunks's def IS at #1; not demoting tests re-breaks isknownsafe).
Introducing a NEW false-refusal + wrong-answer to fix 3 distraction cases is a bad trade by the project's
"over-refusal is a first-class failure" standard.

**Disposition (user-chosen): SHIP DEFAULT-OFF, keep the infra.** `usage_intent_demotion_enabled` defaults
**False**; the validated, kill-switched code + unit/integration tests stay as opt-in infra
(`MEMEX_AGENTS__USAGE_INTENT_DEMOTION_ENABLED=true`) for a future reranker/embedder revisit (a sharper
reranker that ranks the caller above the definition would moot the whole lever). The default product is
byte-identical to the pre-lever baseline. ADR-0021 (usage-intent-demotion amendment).

## Residuals + next

- **The usage-class answer gap is root-caused but UNFIXED by default** (above) — the obvious rerank remedy
  is double-edged; recorded as a measured partial. A future sharper reranker is the clean revisit path.
- `ftc-big-runmain` — a borderline relevance false-refusal on an orchestration entry point (1/39).
  Recorded; the same class as the prose synthesis/relevance residuals, deferred.
- Enrich (the MENTIONS graph over code) was NOT run — the find-the-code eval doesn't use it; code-entity
  NER quality is a separate question.
- Citation rendering for code in the web UI (symbol-named wikilinks in a `<pre>` code-view) is unverified.

## Shared-vault close-out — the FULL prose-gate sweep (2026-06-09, post-merge `f0f47bb`)

The audit-14 prose-gate re-verify above was a SPOT-CHECK (3 detector-triggering counterfactuals, N=2).
Per "an eval-gated-path change needs the REAL eval," the full 13-corpus prose sweep was run on the live
shared vault (device-pinned, codex-rs present, Lever A ON, usage-intent lever OFF):

| corpus | refusal_cf | answered | err | vs in-file baseline |
|---|---|---|---|---|
| annual-report | 1.0 | 9 | 0 | +2 |
| ccna-multidoc | 1.0 | 8 | 0 | 0 |
| chart-types | 1.0 | 3→**4** | 0 | see triage (restored) |
| cr350-diagrams | 1.0 | 11 | 0 | 0 |
| cr350-multidoc | 1.0 | 13 | 0 | +2 |
| forms-w9 | 1.0 | 12 | 0 | +1 |
| french-course | 1.0 | 5 | 0 | 0 |
| handwritten | 1.0 | 5 | 0 | +1 |
| linux-fundamentals | 1.0 | 13 | 0 | 0 |
| nist-zero-trust | 1.0 | 11 | 0 | +3 |
| scientific-gte | 1.0 | 12 | 0 | +2 |
| slide-decks | 1.0 | 13 | 0 | −2 (pre-existing) |
| technical-guidelines | 0.833→**1.0** | 13 | 0 | +2 (CF re-scoped) |

The positive deltas are ADR-0022's recoveries landing after the (stale, 05-24/05-25-era) in-file
baselines. **Verdict: the codex-rs ingest did NOT displace prose answers.** Three flags, triaged to root
cause:

1. **technical-guidelines-13 — the ONE real code-ingest effect (deterministic 3/3), a CORPUS-DESIGN
   finding, not a system defect.** The counterfactual "what exact maximum line length do the coding
   standards enforce?" stopped being counterfactual: codex-rs `tui/src/log_layer.rs` carries
   `const _DEFAULT_MAX_LEN: usize = 120` ("Maximum characters forwarded to the TUI"), dense retrieval
   surfaced it (Lever A did NOT fire — no code identifier in the query), and the answer was a GROUNDED,
   honestly-hedged "120 for TUI log messages … the guidelines document does not specify one" — **no
   fabrication; `verify` held.** The CF's "no answer exists in the vault" premise broke. **The
   shared-vault CF principle:** a counterfactual must be counterfactual against the WHOLE vault, not just
   its fixture. FIX: re-scoped the question to name its fixture ("According to the developer guidelines,
   …" — its siblings' phrasing); refuses deterministically N=3. tg refusal_cf restored to 1.0.
2. **chart-types −2 (08/09) — a pre-existing VAULT-STATE regression, newly root-caused (NOT the
   ingest).** The chart-guide doc was silently re-parsed 2026-06-01 via PyMuPDF (manifest-proven:
   `engine=pymupdf` all pages, `chart_extractions` MISSING), losing ALL chart-OCR content — the
   chart-content queries (gold ONLY in `[chart-extracted]` blocks, dense-only by design) had been
   refusing since, mis-binned as generic retrieval-miss residuals in the audit-12 era. FIX: backed up
   (`~/.memex/chart-guide-backup-20260609`) + `memex parse --force-docling` (the chart-OCR cache replayed
   the original extractions byte-reproducibly; 6 chart blocks restored; 21→34 chunks) + reindex +
   re-resolve golds 01-05. **08/09 RECOVERED** (refusal_cf=1.0, 0 err). One new stable (N=3) conservative
   false-refusal appeared: chart-types-01 — the Docling byline chunk lists the authors WITHOUT the verb
   "wrote", so verify judges the authorship claim unsupported (the borderline-verify class; HARD-gate-safe;
   recorded, not chased). Net chart-types: 3→4 ANS with the chart-content class restored.
   **LESSON: check the manifest `engine` before debugging chart-content refusals as retrieval — and a
   re-parse that switches parser route is a baseline-invalidating event.**
3. **slide-decks −2 — pre-existing (NOT the ingest).** Today's 8 refused ANS match the documented
   post-ADR-0022 residual set (04/16 unreadable-figure + the borderline classes); ADR-0022 recovered no
   slide-decks queries, and the in-file baseline (05-24) predates that era.

Raw sweep reports: `/tmp/prose_sweep/*.json` (transient; the table above is the durable record).

## Reproducibility

```sh
scripts/memex-vault-backup.sh   # (or rsync) back up ~/.memex/vault first
MEMEX_MODELS__CO_RESIDENCE_MODE=manual MEMEX_MODELS__EMBEDDER_DEVICE=cpu MEMEX_MODELS__RERANKER_DEVICE=cpu \
  uv run python scripts/ingest_codebase.py ~/project/codex/codex-rs --suffix .rs   # 99 docs into MAIN vault
… uv run python scripts/extend_corpus.py resolve tests/eval-data/codex-rs-find-the-code/queries.json --max-chunks 1
… uv run python scripts/code_bm25_arm_probe.py tests/eval-data/codex-rs-find-the-code/queries.json   # recall@k
… uv run memex eval tests/eval-data/codex-rs-find-the-code/queries.json                              # grounded answering
```

Content-addressed doc_ids ⇒ a re-ingest regenerates the same gold ids. codex-rs is now a permanent
main-vault corpus; the query set + baselines ship (source stays local).
