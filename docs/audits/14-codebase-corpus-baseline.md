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
correctness, so the headline below is qualified by a manual spot-check.

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

## Residuals + next

- **The usage-class answer-stage gap (the headline next-frontier).** Lever A puts the usage gold in the
  pool, but the reranker + answer LLM still often prefer the title-matching DEFINITION, so "which function
  calls X" is sometimes answered "X is defined here and does Y" (wrong) — ~1–2 of 8 usage queries
  spot-checked. The fix is an ANSWER-stage lever (a usage-vs-definition disambiguation at rerank/answer,
  or query-intent detection for "calls/uses X" → prefer non-title-match chunks), NOT a retrieval one.
  Measure it properly (an answer-correctness eval, not just `answered`+`cp`) before building.
- `ftc-big-runmain` — a borderline relevance false-refusal on an orchestration entry point (1/39).
  Recorded; the same class as the prose synthesis/relevance residuals, deferred.
- Enrich (the MENTIONS graph over code) was NOT run — the find-the-code eval doesn't use it; code-entity
  NER quality is a separate question.
- Citation rendering for code in the web UI (symbol-named wikilinks in a `<pre>` code-view) is unverified.

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
