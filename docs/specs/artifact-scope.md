# Spec — Deterministic artifact→document re-scope (#256)

**Status:** Shipped 2026-05-26. Closes the `cr350-diag-12` cross-doc scope conflation that the relevance gate left open (roadmapped "multi-doc scope disambiguation").

## Problem

When a query NAMES a specific artifact that lives in one document — cr350-diag-12
*"Quelle est la plage d'adresses VLAN configurée dans **le diagramme de coupe-feu**?"* —
the answer must come from THAT document. The firewall diagram (`ccd09479` / the
firewall lecture `499c900d-cours-6`) has **no** VLAN address range, so the correct
behaviour is to refuse. Instead the agent answered from a *different* document —
semaine-4's VLM-transcribed VLAN content ("Fa0/21 → VLAN 10/20/30"). The answer was
grounded and responsive, just **wrong-source**.

A prior fix — an LLM `assess_relevance` "source-scope" prompt clause — was built,
validated **ineffective**, and reverted: Qwen3-8B gave 0 non-responsive verdicts (the
answer genuinely IS responsive; it's the *source* that's wrong). The relevance gate
catches grounded-but-non-responsive answers, not grounded-responsive-but-wrong-source
ones. So the mechanism had to be **deterministic** (regex + lookups, no LLM), per the
determinism mandate.

The user chose **retrieval-time re-scope** (not a post-answer refuse-gate): when a
query names an artifact, resolve it to its document(s) and scope retrieval there — the
named artifact acts as an automatic doc-selection (the Notebook-LM model applied
early). The normal pipeline then answers from the right source or refuses naturally.
This is forward-compatible with the planned "select which docs the question uses"
feature and preserves open/multi-file synthesis untouched (queries that name no
artifact take the unchanged full-corpus path).

## Why re-scope makes the bug case deterministic (the key property)

The three stages are each deterministic for a fixed index. For diag-12 the re-scope
**removes semaine-4's VLAN chunk from the candidate pool before the LLM sees it**, so
the LLM has no VLAN evidence and *must* refuse — regardless of its (non-deterministic,
BF16/AWQ greedy) sampling. This converts a borderline ANS↔REF flip into a guaranteed
refuse. The only residual non-determinism is the downstream LLM on *should-answer*
artifact queries (where the scoped doc DOES contain the answer) — answering is the
correct outcome there, and the multi-run acceptance gate guards a borderline
false-refuse.

## Design — three deterministic stages (`agents/artifact_scope.py`, pure)

1. **Detect** — `detect_artifact_reference(query) -> ArtifactReference | None`.
   Regex EN + FR. Returns the artifact noun + its QUALIFIER (the content tokens BOUND
   to the artifact noun), or `None`.
   - **EN:** `the/this/that <≤3 qualifier words> <arttype>` — qualifier *precedes* the
     noun; the data term sits elsewhere.
   - **FR:** `<arttype> <connector> <tail>` — qualifier *follows* via a connector
     (`de`/`du`/`des`/`de la` + space, OR elided `d'`/`de l'` binding directly). A
     bare `le diagramme <verbe>` (no connector) does NOT capture a qualifier. The data
     term *precedes* the noun ("…VLAN configurée dans le diagramme de coupe-feu"), so
     it's never in the tail. **This positional rule is what excludes the data term**:
     diag-12 → `{coupe, feu}` (NOT `vlan`); the near-twin img-01 ("diagramme de
     configuration VLAN du commutateur") → `{configuration, vlan, commutateur}`
     (INCLUDES vlan, because there it follows the connector).
   - Qualifier tokens are diacritic-/hyphen-/apostrophe-folded into atoms
     (`coupe-feu`→`{coupe,feu}`), stop-worded, capped at 4 surface words.
   - Returns `None` for: no artifact noun; ≥2 *distinct* artifact references
     (ambiguous); a bare numbered ref ("Figure 3", no qualifier); an empty qualifier
     ("the diagram"). Artifact vocab deliberately omits FP-risky common nouns
     (`table`/`tableau`, `plan`, `map`, `graph`).

2. **Resolve** — `resolve_scope(ref, search) -> ResolvedArtifactScope`, pure given an
   injected per-token BM25 `search` (FTS phrase-quotes its input, so one search PER
   qualifier atom = token-OR). Docs ranked by **hit-count** (sign-agnostic; the
   flipped-BM25 score is only a tie-breaker), with a **`doc_id`-ascending** final
   tie-break (byte-stable run-to-run). Confidence branches:
   - **dominant:** `top.hits ≥ 2 × second.hits` (or a single doc) → `(top,)`.
   - **small-set:** ≤2 docs in a hit-band, each with a title hit or `hits ≥ 3`.
   - **sibling-by-title:** when a resolved doc's TITLE carries the qualifier, admit
     other title-matching docs (the firewall diagram + its lecture), capped at 2.
   - **single-token specificity gate** (see below).
   - Everything else (`none-weak`, `none-spread`, `none-no-qualifier`,
     `none-singletoken-ambiguous`) → empty scope = **no re-scope** (full-corpus path).

3. **Re-scope node** — `agents/answering.py::resolve_artifact_scope`, wired
   `retrieve → resolve_artifact_scope → expand_graph`. On a confident resolution it
   **REPLACES** the candidate pool with `hybrid_search_in_docs(query, scope_docs, k=50)`
   (REPLACE, not top-up — topping up would re-admit the wrong-source chunk). It is
   **fail-open**: no artifact, no confident resolution, the kill-switch, or any
   resolver/store error all leave `candidates` untouched. `expand_graph`
   short-circuits while a scope is active (it would otherwise re-admit other
   documents' chunks and defeat the determinism guarantee). Settings flag
   `agents.artifact_scope_enabled` (default ON; env
   `MEMEX_AGENTS__ARTIFACT_SCOPE_ENABLED=false` fully reverts).

## The single-token specificity gate (the bulletproofing finding)

During real-index validation, `slide-decks-04` ("…per the **TSMC chart**?", an ANS
query whose answer is in the CUDA deck) mis-resolved: the qualifier `{tsmc}`
dominantly matched the **annual report** (NVIDIA's 10-K names TSMC in prose), because
the CUDA deck's TSMC mention is locked inside a **FTS-stripped `[chart-extracted]`
block** (the P3.3 v3 defense). Re-scoping there would have **false-refused** an
answerable query. This is the `risk #6` blind spot — and crucially it did NOT
"degrade to no-scope" as the plan assumed; it resolved *confidently to the wrong doc*.

**The diag-12 fix and the slide-04 safety are in fundamental tension** — both have
qualifier-doc ≠ data-doc, but diag-12 (REF) needs the wrong doc *removed* while
slide-04 (ANS) needs the right doc *retained*. No retrieval signal distinguishes them
(diag-12's home `cours-6` is NOT in its full-retrieval top-50; slide-04's wrong doc
`annual-report` IS in its top-50). The clean separator is **qualifier arity**: every
query that NEEDS the fix is **multi-token** (`coupe`+`feu`, `configuration`+`vlan`+
`commutateur`), while the risky cross-doc entities are **single tokens**.

**Gate:** a single-token qualifier only scopes when distinctive — the resolved doc
carries it in the TITLE, or it's the SOLE doc with any hit on the token. Otherwise →
`none-singletoken-ambiguous` (no scope; the query answers via full corpus as before).
Multi-token qualifiers are exempt (a conjunction is a strong artifact-name signal).
Measured against the real index this BLOCKS `{tsmc}` (2 docs, no title → slide-04
safe) while ALLOWING `{gantt}` (sole doc → chart-08/09) and `{attaque}` (semaine-5's
title carries it → img-03), and never touches the multi-token fixes.

## HARD-gate invariant

The node can only NARROW retrieval for artifact-named queries; it never adds a chunk,
never fabricates, never relaxes a gate. Worst case = a conservative false-refuse on a
mis-resolved ANS query (HARD-gate-safe: never a hallucination, never a
wrongly-answered counterfactual). The existing assess/verify/relevance gates still
decide answer-vs-refuse on the narrowed pool.

## Observability (ADR-0004)

The re-scope is silent by default — a user just gets the right answer or a refusal.
To make the narrowing auditable, the resolved scope is surfaced on
`FinalResponse.artifact_scope_doc_ids` (populated in `compose` + the `refuse` node
from `state.artifact_scope_doc_ids`; `[]` on the full-corpus path). It's most useful
on a REFUSAL — it explains WHY the pool was narrowed (e.g. "scoped to the firewall
doc, which has no VLAN range" → refuse). MCP `ask` + the CLI auto-serialize the field
(it's part of the `FinalResponse` model); the webui answer panel renders a quiet
`.ans-scope` note ("Scoped to the document(s) you named: …"). HARD-gate-neutral — a
derived field, never alters answered/claims/refusal. GPU-validated end-to-end:
diag-12 surfaces the firewall lecture, img-01 surfaces the VLAN deck, a no-artifact
query surfaces `[]`.

## Files

| File | Change |
|---|---|
| `agents/artifact_scope.py` (NEW) | `detect_artifact_reference` + `resolve_scope` + the two dataclasses; pure, imports only `re` + `core/types` |
| `agents/answering.py` | `resolve_artifact_scope` node + `_resolve_artifact_scope_via_corpus` (lazy `FTSStore.open`, fail-open); `AnswerState.artifact_scope_doc_ids`; `FinalResponse.artifact_scope_doc_ids` (surfaced in `compose`/`refuse`); the `expand_graph` scope-active skip; graph wiring |
| `core/config.py` | `AgentsSettings(artifact_scope_enabled=True)` + `MemexSettings.agents` |
| `webui/templates/_answer.html` + `static/style.css` | `.ans-scope` note surfacing the resolved scope |
| `tests/unit/test_artifact_scope.py` (NEW) | detection + resolution tables, edge cases, the diag-12-vs-img-01 discriminator, single-token gate, N=50 determinism |
| `tests/integration/test_answering_with_fakes.py` | re-scope scenarios: wrong-source refuses, near-twin answers, no-artifact no-op, fail-open, kill-switch, scope surfaced, expand_graph skip |
| `tests/integration/test_webui.py` | the `.ans-scope` note renders on a scoped refusal + is absent otherwise |

## Validation

- **Determinism** (unit): `detect` + `resolve` run N=50× on the same inputs → byte-identical.
- **Real-index resolution probe** (no LLM): all 14 detection-triggering queries across
  6 corpora resolve to the correct doc, no-scope, or the intended firewall docs.
- **Multi-run GPU acceptance** (≥5 runs, `MEMEX_RERANK_BATCH_SIZE=1`): see the
  `_baseline_2026_05_26_artifact_scope` notes in each affected corpus.

## Anti-scope (deferred)

- NOT a full Notebook-LM doc-picker UI (this is the *automatic* artifact→doc selection;
  the manual picker reuses `resolve_scope` + `hybrid_search_in_docs` later).
- NOT an LLM resolver or synonym model (determinism mandate).
- The FTS chart-strip blind spot is handled defensively (single-token gate), not
  solved — an EN cross-doc single-token artifact whose content is chart-stripped takes
  the full-corpus path rather than scoping. A structural "doc has a chart/diagram near
  the qualifier" signal would let single-token EN chart references scope; deferred.
