# BM25-for-code: arm-separation probe (2026-06-09)

**Verdict: GO** — the prose finding "BM25 recall ⊆ dense" (`09-fts-bm25-arm-separation.md`) **INVERTS
for code**, but COMPLEMENTARILY: BM25 patches a specific gap (usage/reference queries), it does NOT
beat dense in general. Phase-3 measure-first (ADR-0021, spec `docs/specs/code-chunking.md`). The
build is gated on this + a user check-in (it touches the shared `FTSStore.search` primitive).

## The question

Code retrieval is identifier search. Phase 2 makes each Rust symbol the chunk's `## <symbol>` heading
→ its EmbeddingGemma embed-title, a strong DENSE lever. The live FTS wraps the whole query as one
literal phrase (`fts_store.py:search`), so BM25 ≈ 0 on a natural-language code question. Does a
TERM-query BM25 arm recover exact-identifier chunks the dense arm misses?

## The corpus + probe

A dedicated throwaway vault holds a slice of `codex-rs/core` (29 `.rs` files → **357 Rust symbol
chunks**, `code-rust-v1`). 20 ANS queries (`tests/eval-data/codex-rs-find-the-code/queries.json`),
deliberately split into two regimes, + 4 counterfactuals. `scripts/code_bm25_arm_probe.py` opens the
stores directly and computes recall@k of each arm's PRE-rerank candidate pool (the k=50 pool is what
feeds the reranker — `answering.py:642`). Deterministic (CPU dense + LanceDB flat + raw FTS):
**byte-identical across two runs.**

## The decisive result — read PER REGIME (the blended number is mix-dependent)

The headline is the per-regime split, NOT a blended mean (the blend depends on the
definition:usage ratio, which this corpus sets 50/50 by design):

| regime | n | dense@50 | bm25-phrase@50 (LIVE) | **bm25-term@50** | union@50 |
|---|---|---|---|---|---|
| **definition** (`where is X defined`, `what does fn Y do`) | 10 | **1.00** | — | 1.00 | 1.00 |
| **usage/reference** (`which fn calls X`, `where is X used`) | 10 | **0.70** | — | **1.00** | 1.00 |
| blended (artifact of the 50/50 mix) | 20 | 0.85 | **0.05** | 1.00 | 1.00 |

recall@k (mean over 20 ANS): dense `0.65/0.70/0.80/0.85` (@5/10/20/50); bm25-phrase `0.05` flat (the
LIVE production FTS contributes ~nothing for code); **bm25-term-whole `0.80/0.95/1.00/1.00`**;
term-sub `0.70/0.95/1.00/1.00`; term-both `0.75/0.95/1.00/1.00`; **union `0.85/1.00/1.00/1.00`**;
rrf(dense,term) `0.60/0.85/0.90/1.00`.

**3 usage golds are recovered ONLY by bm25-term, never by dense** (`ftc-use-embeddedinstr`,
`ftc-use-getplatform`, `ftc-use-consumetrunc`) — all `usage_reference`.

## Why (mechanistic, not a fluke)

Phase 2's symbol-as-embed-title lever **saturates definition queries** (dense 1.00 — the query names
the symbol that IS the chunk's title) and **structurally defeats dense on usage/reference queries**:
the gold there is a chunk titled by a DIFFERENT symbol (the caller), with the queried identifier in
the BODY. The 300M mean-pooled EmbeddingGemma can't surface that body identifier against the
title-dominated chunk vector — consistent with the [[contextual-retrieval-negative-2026-05-25]]
finding. A lexical body-match (the quoted-whole-identifier FTS term) recovers it directly. So the gap
ORIGINATES in the (correct, deliberate) Phase-2 title choice; **BM25-term is one lever for it, not
provably the only one** (a stronger/asymmetric code embedder is another, deferred).

## Disposition — GO, complementary, regime-scoped

BM25-for-code adds recall **on the usage/reference class** (0.70 → 1.00, 3/3 dense-misses recovered);
it is **redundant on definitions** (dense already 1.00). Frame it as a COMPLEMENT that patches the
usage/reference gap, **not** "BM25 beats dense." `rrf@50 = 1.00 ≥ dense 0.85` confirms the production
RRF fusion can realize the headroom from the k=50 pool that feeds the reranker.

**Honest caveats (do not overclaim):**
- **rrf@5 (0.60) is BELOW dense@5 (0.65).** Naively RRF-fusing the term arm slightly degrades the
  HEAD of the ranking (a definition-win pushed out of top-5). The GO rests on the **k=50 pool recall**
  (what the reranker sees), which is fine — but fusion *weighting* is a build-phase tuning question;
  do NOT assume "fusion strictly improves retrieval."
- **term-WHOLE is the recommended build variant**, not term-both. It weakly dominates (`0.80` vs
  `0.70/0.75` @5; ties after) AND is clean on counterfactuals: `validate_jwt_token` as a whole quoted
  phrase → 0 hits, but `_`-sub-splitting → 2 spurious hits via `token`/`validate` (and
  `compute_tax_rate` → 1 via `compute`/`rate`). Sub-token splitting buys **zero** recall and adds
  noise + rerank-pool dilution. Build default = whole-only.
- **Single-gold scoping:** each usage query's gold is its *primary* caller; some identifiers have
  multiple call sites (e.g. `get_platform_sandbox` is also called by `assess_command_safety`). The
  recovered-gold finding holds; the recall numbers are over the single primary gold.
- **Confirmed at full scale (99 files, 1235 chunks — 3.5× the `core/` slice).** The regime split is
  IDENTICAL: definition dense@50 = **1.00** (the Phase-2 title lever holds against 3.5× more competing
  chunks), usage/reference dense **0.70 → bm25-term 1.00**, the **same 3** dense-misses recovered
  only-by-term. Dense's HEAD recall dipped slightly with more competition (@5: 0.65 → 0.60) while
  term-whole held (@5: 0.80 → 0.75). The counterfactual **sub-token noise scaled up** (jwt 2 → 11
  hits, computetax 1 → 7) — which REINFORCES the term-WHOLE recommendation (whole-identifier matching
  stays clean; the noise is entirely from `_`-splitting). All 20 content-addressed gold ids were
  present in the full vault unchanged (a re-ingest regenerates the same ids).

## The conditional build (NEXT, gated on a user go-ahead — NOT done here)

A SCOPED, env-gated code term-query path in `FTSStore.search` (or a sibling), building the
OR'd-quoted-WHOLE-identifier MATCH for the code path ONLY — **never** changing the prose phrase-wrap.
Because `FTSStore.search` is shared (hybrid + `artifact_scope`), it MUST ship with the **prose
HARD-gate corpora re-run** (`refusal_cf=1.0`, byte-stable answered counts) + the prose arm-separation
re-confirmed (`union@50 == dense@50` still holds for prose). Lever B (add `_` as an FTS tokenchar so
snake_case indexes whole — reindex-forcing) is deferred: the quoted-WHOLE term already matches
snake_case contiguously, so it is not needed for this result.

## Reproducibility

```sh
# Ingest the slice into a THROWAWAY vault (the prose vault is never touched):
MEMEX_VAULT_PATH=/tmp/memex-code-probe-vault MEMEX_MODELS__CO_RESIDENCE_MODE=manual \
  MEMEX_MODELS__EMBEDDER_DEVICE=cpu MEMEX_MODELS__RERANKER_DEVICE=cpu \
  uv run python scripts/ingest_codebase.py ~/project/codex/codex-rs/core --suffix .rs
# Resolve gold ids (idempotent) + run the probe:
… uv run python scripts/extend_corpus.py resolve tests/eval-data/codex-rs-find-the-code/queries.json --max-chunks 1
… uv run python scripts/code_bm25_arm_probe.py tests/eval-data/codex-rs-find-the-code/queries.json
```

The query corpus is REPRODUCIBLE: `doc_id = sha256(bytes)[:8] + slug(stem)`, so a re-ingest of the
same files regenerates the SAME gold `relevant_chunk_ids` (content-addressed chunk_ids). The corpus
ships (query set + ids; source stays local) and is the Phase 4-5 find-the-code baseline regardless of
the build decision.

## Lever A — SHIPPED (2026-06-09, default-ON, prose-validated)

The build is the term-WHOLE recommendation, scoped to code so the prose phrase-wrap is untouched:
- `index/code_query.py` — `query_has_code_identifier` (the per-query gate: a `_` or camelCase token),
  the validated `build_code_term_match` WHOLE builder, `code_term_query_enabled` (fail-open).
- `FTSStore.search`/`search_in_docs` gain `term_query: bool = False` (branch the MATCH; empty-term →
  phrase-wrap fallback; default byte-identical — every non-/ask caller unaffected).
- `retrieve/hybrid.py` threads `term = code_term_query_enabled() and query_has_code_identifier(query)`
  into its 2 /ask calls ONLY. Config `AgentsSettings.code_term_query_enabled = True` (kill-switch
  `MEMEX_AGENTS__CODE_TERM_QUERY_ENABLED=false`). Query-side only ⇒ NO reindex.

**Validation (the build's measure-don't-assert):**
- **Code benefit (production path):** `hybrid_search` flag-ON recovers all 3 only-by-term usage golds
  (ranks 3 / 3 / 7); flag-OFF misses all 3; the definition control is rank 1 either way.
- **Prose safety:** the detector changes behaviour on EXACTLY the 8 prose queries it fires on (across
  cr350-diagrams / scientific-gte / technical-guidelines / slide-decks — **3 of them COUNTERFACTUALS**).
  Ran those 8 through the production `/ask` path flag ON vs OFF, **N=2: byte-identical** — the 3
  counterfactuals refused both runs (`refusal_cf` = 1.0 held), the 5 ANS answered both. Every
  non-triggering prose query is byte-identical by construction (detector → phrase-wrap). So the default
  ships **ON**. 1452 unit + 27 feature tests pass; pyright 0/0; ruff clean.

The validation is the **triggering-query A/B** (the complete set of behaviour-changing queries), not a
full multi-corpus eval — non-triggering queries can't change, so a full sweep would only re-measure
baseline LLM flakiness. ADR-0021 (Amendment) records the decision.

**Scope of the prose validation (honest):** it ran on the **prose-only main vault** (the current
deployed state), so default-ON is genuinely safe *today*. But the feature's target is a vault that
CONTAINS code, where a detector-triggering prose query retrieves over a pool that now includes code
chunks — untested here. Concrete interaction: `scientific-gte-16` ("OpenAI text-embedding-3-large
dimension", a counterfactual) triggers the detector, and codex-rs `model_provider_info.rs` carries
openai/deepseek base URLs, so once code is in the vault term-WHOLE on "openai" can pull that code chunk
into the counterfactual's pool. It almost certainly still refuses (no dimension there → `verify`
rejects), but the **Phase 4-5 shared-vault re-verify MUST explicitly re-run the detector-triggering
prose counterfactuals** after adding the code domain.

**Known limitation:** the detector misses a usage query naming a BARE all-lowercase identifier ("which
function calls `backoff`?" → no `_` / camelCase → stays on the phrase-wrap). Acceptable for v1 — the
targeted usage/reference queries mostly name snake_case or PascalCase identifiers, which do fire.
