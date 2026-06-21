# Spec — Query-aware (value-centric) prompt-chunk truncation

Status: **MEASURED-NEGATIVE — BUILT then REVERTED 2026-06-21. DO NOT re-attempt as designed.**

The lever was built end-to-end (the `relevance_window` helper + answer/v6, verify_grounding/v3,
assess_sufficiency/v5 + query threaded into both verify renders + grounding.py `query=""`; ruff +
pyright + pytest 2063 green; the helper verified to recover foia-08's `$250`), then **the REAL
full-suite eval FAILED it**:

- **Fixed +2:** legal-statutes foia-08 (the target truncation case) + pa-12.
- **DETERMINISTIC HARD-gate regressions (N=2 byte-stable):** `forms` refusal_cf 1.0→**0.833**
  (counterfactual `f1040-10` answered) + `slide-decks` 1.0→**0.923** (`slide-decks-08` answered)
  — **2 new hallucinations.** Plus answered→refused flips: nist-zero-trust −2, chart-types −1,
  companion-augment −1, technical-guidelines −1.

This **FALSIFIES the "HARD-gate invariant" below** ("refusal_cf cannot regress by construction").
The reasoning was wrong: a *near-miss* counterfactual chunk DOES contain related content, and
centering the window on the query terms surfaces a misleading-but-non-answering span the gate then
grounds. Re-centering is double-edged — it also drops head context other queries needed (the flips).
The independent review hedged the answered→refused direction as "empirical"; the CF direction is too.

**Conclusion:** value-centric truncation joins the global budget-raise (forms Increment B) as a
MEASURED-NEGATIVE — **both** truncation levers regress, so the truncation-horizon false-refusal class
(foia-08) is now DOCUMENTED-HARD with no safe truncation lever found. The eval-gated-path law in
action: a green build + a clean review + a verified helper, caught only by the real eval.

--- original spec preserved below for the record ---

## Problem

The grounded-answer gates render each candidate chunk into their prompt as a **head
truncation** — `{{ c.text | truncate(1800) }}` — keeping the first ~1800 chars and dropping
the rest. When a chunk exceeds the budget **and the answer value sits past char ~1800**, the
gate is blind to it → drafts a denial / cannot ground → a **false refusal on an answerable
query**.

**Confirmed empirically** (`scripts/parse_fragmentation_probe.py` + a truncated-vs-full causal
test, 2026-06-20): legal-statutes **foia-08** — the answer "$250" sits at char offset **1854**
in a 2260-char chunk reranked to **#1**; the gate refuses. Temporarily raising the budget so
"$250" is visible flips it to a correct grounded answer ("…the fee will exceed **$250**").

**Scope of the problem — honest framing.** 826 / 7889 live chunks (10.5%) exceed 1800 chars,
so truncation *fires* often — but 10.5% of *chunks* ≠ 10.5% of *queries* affected (most long
chunks aren't reranked to the top, or have the answer in the head). The **confirmed upside is
n=1** (foia-08); the change re-slices **all** long chunks at every gate render. So the
justification is **not** "10.5% de-risks it" and **not** "safe by construction" — **the
multi-run eval gate does the load-bearing de-risking.** Matches the deep-research
form/table-legibility finding ([[form-table-research-2026-06-15]]) as a real recurring class.

## Why not the naive fix

Globally raising the budget (1800 → N) is **MEASURED-NEGATIVE**: the forms "Increment B"
gate-render truncate-lift regressed the suite (couldn't isolate the win from dense-table
noise; [[realworld-corpora-2026-06-15]]) and costs tokens on every query. The fix must be
**value-centric, not bigger**.

## Design — relevance-centered window (replaces head truncation)

For a chunk whose text **exceeds the budget**, render a budget-sized window **centered on the
chunk's most query-relevant span**, instead of the head:

1. Split the chunk text into sentences / lines.
2. Score each span by **lexical overlap with the query** — reuse `core.text.atomise` +
   `STOPWORDS` (already imported in `answering.py:68-69`) for the query content-term set, so
   tokenization matches the rest of the gates (no new tokenizer).
3. Take a `budget`-char window centered on the highest-scoring span, clamped to text bounds,
   snapped to sentence boundaries; ellipsis on any cut side.
4. Chunks **≤ budget+leeway (~1805)** returned unchanged — the common case has **zero** behavior
   change. (NB Jinja `truncate` has `leeway=5`; the helper's passthrough threshold must match,
   and on a genuinely-truncated chunk the helper's sentence-snap/ellipsis will differ byte-for-byte
   from Jinja's word-snap — so "byte-identical" holds only for ≤~1805-char chunks.)

**Mechanism.** A pure helper `relevance_window(text, query, budget) -> str`
(`src/memex/agents/_windowing.py`). Exposed to templates as a Jinja filter
`relevance_window(query, 1800)` in **new prompt versions** — `answer/v6.md`,
`verify_grounding/v3.md`, `assess_sufficiency/v5.md` — loader-selected (highest version;
`loader.py:91,121` confirmed) and kill-switchable via `MEMEX_PROMPTS__PIN__{ANSWER,
VERIFY_GROUNDING,ASSESS_SUFFICIENCY}` (`loader.py:94-96` confirmed). NB the loader is `@cache`'d
(`loader.py:99`) → a pin flip needs a fresh process (fine for the eval harness + daemon restart).

**REQUIRED wiring fix (the spec-review catch).** `answer` and `assess_sufficiency` already pass
`query` into their render (`answering.py:1722`, `1523`). **`verify_grounding` does NOT** — it
renders `draft`+`chunk_by_id` only, at **two** sites: the main verify (`answering.py:2040`) AND
the citation-retarget probe (`answering.py:2295`, live by default, `AnswerState.citation_retarget=True`).
The build MUST thread `query=state.query` into **both** verify render calls; otherwise
`StrictUndefined` (`loader.py:51`) makes a `query`-referencing v3 raise `UndefinedError` → an
unhandled 500 on every `/ask` verify (the failure class `verify_overflow_fix` eliminated). The
`state.query` is in scope at both sites.

**Cross-gate determinism invariant (required).** `answer` and `verify` must window the SAME
chunk with the SAME (query, budget) — else `answer` drafts from a span `verify` can't see →
spurious ungrounded → false refusal. The helper is pure/deterministic and MUST be called with
identical args at all render sites (answer `1722`, verify `2040`, retarget probe `2295`).

## HARD-gate invariant (corrected per review — do NOT overclaim)

- **By construction (sound):** the window re-slices bytes ALREADY in the chunk, so it cannot
  surface a value absent from the chunk → a counterfactual chunk has nothing to window toward →
  **`refusal_cf` cannot regress.** This holds because it is single-chunk, existing-bytes-only
  (unlike the earlier cross-doc "by construction" miss). The numeric/name-only `verify` backstops
  are unchanged.
- **NOT by construction — the empirical risk the eval must catch:** the **currently-answered →
  newly-refused** direction. Lexical centering targets the highest query-overlap span, which is
  not necessarily the answer/support span; a value the head currently shows could be windowed
  OUT → a NEW false refusal. This risk lives in **`answer` + `verify`** (bidirectional).
- **Out of scope / unaffected:** `assess_sufficiency` is an advisory pre-filter that can only be
  pushed toward `sufficient=true` (safe-directional). `assess_relevance` renders **no chunk
  text** (`answering.py:2553-2557` — only `query`/`summary`/`grounded_claims`), so windowing
  **cannot** shift it — it is NOT in the affected set.

## Files

- `src/memex/agents/_windowing.py` (new) — `relevance_window` (pure, sync, typed); reuses `atomise`/`STOPWORDS`.
- `src/memex/prompts/{answer/v6, verify_grounding/v3, assess_sufficiency/v5}.md` — swap the
  filter; otherwise byte-identical to the active versions.
- `src/memex/agents/answering.py` — thread `query=state.query` into BOTH `verify_grounding`
  renders (`2040`, `2295`); register the `relevance_window` Jinja filter.
- `tests/unit/test_windowing.py` — centering, sentence-snap, ≤budget+leeway passthrough,
  empty/degenerate, the foia-08 value-recovery shape, cross-gate-identical-args.

## Validation (the eval gate — the real de-risking)

- **foia-08 flips** refuse → answer ("$250"); re-measure the legal-statutes false-refusal set.
- **FULL suite, N≥2 multi-run**: `refusal_cf == 1.0` on **every** corpus (CF direction), **0** new
  hallucinations, and **no currently-answered query flips to refused** — watch the long-chunk
  corpora especially (forms, legal-statutes, annual-report) AND the **chart corpora**
  (slide-decks, annual-report, chart-types): a compact `[chart-extracted]` block in a long PROSE
  chunk can be windowed OUT if it has low query overlap → a regression against the P3.3 chart
  wins (`src/memex/CLAUDE.md:43`). Acceptance = strictly ≥ current ANS with **0 HARD-gate regressions**.
- Re-run after the change per the eval-gated-path law.

## Anti-scope (deferred)

- The **gate over-refusals on VISIBLE values** (foia-05/11, pa-12) — ADR-0022 documented-hard class; NOT this lever.
- **Re-chunking / index-time re-render** — mutative churn (the W6 / forms-linearizer trap); out of scope.
- **Embedding/reranker-based** span scoring (costly) — lexical first; revisit only if it underperforms.
- **Synthetic chunks** (resolve_form_field / Table-RAG) are SAFE/unaffected: `_SYNTHETIC_TEXT_MAX=900 < 1800`
  (`answering.py:1122`) → passthrough, no double-truncation, no fight with their deliberate evidence-first packing.
