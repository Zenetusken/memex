# Spec: Symbol-Aware Code Chunking

**Status:** Phase 1 + Phase 2 shipped (2026-06-07, `268e39e` + `b9c3a58`); Phases 3–5 planned.
**Decision record:** [ADR-0021](../adr/0021-codebase-corpus-code-as-documents.md).
**Backend cheat-sheet:** `src/memex/CLAUDE.md` (the code-ingest + symbol-chunking bullets).

This spec is the build-level design for ingesting a **source-code repository as documents** with
**symbol-precise retrieval** — the mechanism behind ADR-0021. It assumes the grounded contract is
unchanged: answer from the vault or refuse, `refusal_cf=1.0`.

## Goal

A code question ("where is `X` defined", "what does `fn Y` do") must rank the **defining symbol**
chunk, not a size-budgeted fragment. The lever is the chunk's title: the deepest `heading_path`
becomes the EmbeddingGemma document-prompt title (`index/embed_prompts.py`), so if each chunk's
heading *is* its symbol, the symbol name is in the embedding input. Everything below exists to make
the chunker split on symbol boundaries while keeping the canonical `.md` the verbatim source.

## Phase 1 — verbatim ingest (`268e39e`)

The pipeline previously mangled code: a non-(md/media/office/image/pdf) file fell through to Docling,
which renders aligned code as a markdown pipe-table. Phase 1 routes code around that.

- **`core/source_types.py`** — `CODE_SUFFIXES` (the known code extensions), `LANGUAGE_FOR_SUFFIX` +
  `language_for_suffix(suffix)` (suffix → display language, bare-ext fallback), and
  `code_language_for_doc(asset_dir)` (globs `source.*` → the language, or `None`) — the shared
  index+enrich gate.
- **`ingest/validation.py`** — a `code` `DetectedKind`: a suffix ∈ `CODE_SUFFIXES` **and**
  `_looks_like_text(head)`. It is deliberately **absent from `_EXTENSION_FOR_KIND`**, so the original
  suffix is preserved on copy (the image/audio precedent — a `.rs` mapped to `.txt` would lose the
  language before parse).
- **`parse/pipeline.py`** — `parse_document` routes a `CODE_SUFFIXES` source to the existing
  **`_passthrough_markdown`** (verbatim UTF-8; no chart-OCR / no VLM) **before** the Docling fallback.
  Rust `#[attr]` / `#!` have no space after `#`, so they don't trip the chunker's `# `-heading regex.
- **Title = repo-relative path** (`exec/src/lib.rs`), not the basename — `lib.rs`/`mod.rs`/`main.rs`
  collide across crates. `scripts/ingest_codebase.py` walks a repo with repo-relative titles and skips
  `.git`/`target`/`node_modules`.

HARD-gate-neutral: code is a verbatim-stored prose-shaped document; it adds no grounding surface.

## Code-view rendering (`6e477b3`)

A source-code doc renders **as code**, not as markdown. `webui/app.py::_is_code_source(source)`
(suffix ∈ `CODE_SUFFIXES`) gates all three body-render sites (detail route + the inline-edit
cancel/save partials) to: render the verbatim body in a `<pre>` under a `source · <language>` label,
and **suppress** the markdown heading-anchor + `[[wikilink]]` transforms (a Python/shell `# comment`
must not become an H1; a literal `[[x]]` must not become a link). Zero template edits — a code doc is
already the `has_source=True`/`has_preview=False` pane-solo branch, and the `_document_body.html`
`rendered_body or document.body` fallback renders the raw (Jinja-autoescaped) body when
`rendered_body=None`. Presentation-only, HARD-gate-neutral. No syntax highlighting in v1 (deliberate;
`<pre>` word-wrap kept to preserve the zero-template-edit elegance).

## Phase 2 — symbol-aware chunking (`b9c3a58`)

The mechanism is **transient `## <symbol>` heading injection** into the chunker's copy only — the
blessed `reattach_chart_extractions` / `linearize_gfm_tables` transform pattern. No `chunk_document`
surgery (injection can't drop bytes; the embed-title lever is mechanism-independent).

### The Rust symbol splitter — `index/rust_symbols.py`

`split_rust_symbols(body) -> list[Symbol]` is pure and dependency-free (NO tree-sitter — the air-gap
constraint):

- a **masking lexer** blanks string literals, comments, char-literals, and raw-strings so a brace or
  `fn` *inside* them can't corrupt the brace-depth scan;
- a **line-start regex grammar** for `fn` / `struct` / `enum` / `union` / `trait` / `type` / `mod` /
  `const` / `static` / `macro_rules` / `impl`, plus methods nested inside `impl` / `trait` / `mod tests`
  (fully-qualified as `Type::method`);
- **preamble capture** backs each symbol's start up over its `#[..]` attributes (multi-line,
  bracket-balanced) and `///` / `//!` doc-comments;
- **bounded failure** — any error returns `[]`, so the doc falls back to the unchanged prose path and
  never drops content.

`inject_symbol_headings(body, symbols)` prepends `## <symbol>` / `### <Type::method>` ATX lines at the
symbol starts, so the existing section-splitter produces one chunk per symbol with
`heading_path = [impl Foo, Foo::bar]`.

### The shared seam — `index/pipeline.py::build_chunking_body`

```python
def build_chunking_body(doc, chart_extractions, *, code_language) -> tuple[str, str]:
    body = linearize_gfm_tables(reattach_chart_extractions(doc.body, chart_extractions))
    if code_language == "rust":
        return inject_symbol_headings(body, split_rust_symbols(body)), "code-rust-v1"
    return body, "v0"
```

Called by **both** `index_document` and `enrich.pipeline.enrich_document` (the #394 MENTIONS-resolution
parity — both chunk byte-identical bytes). `code_language` is the caller's one-shot
`code_language_for_doc(doc.ref.asset_dir)` result (passed in so index + enrich glob once, not twice).
The returned `chunking_recipe_version` is keyed on the transform **actually applied**: `"code-rust-v1"`
iff the Rust grammar ran, else `"v0"` (prose **and** non-Rust code, which both fall through to the
unchanged prose path). It is recorded on the manifest `IndexStage` (default `"v0"`) and OR'd into the
`index_document` force-detection — mirroring `embedding_recipe_version` — so a future grammar/label
change force-rechunks exactly the docs that used that grammar. (Page attribution is off for code.)

### Decisions (user-confirmed)

- **Rust-only v1** — a Python splitter misfires on `# comment` lines and significant-whitespace blocks
  (a separate increment).
- **Method-level granularity**; **one symbol per chunk** (no packing in v1).

### Honest costs

- **Chunk *text* is not byte-verbatim** — the chunker's `_PARAGRAPH_RE.split` + `.strip()` normalizes
  blank runs / post-blank indentation. Cosmetic for Rust; a second reason Python is out of v1. The
  canonical `.md` stays verbatim.
- **No in-source citation anchors** (`#fn-foo` jump-to-symbol) — deferred to v2.

### HARD-gate neutrality — by construction

The code path is gated (prose chunking byte-identical — verified 16/16 chunk_ids on a real ingest),
and off-topic code chunks cannot make a counterfactual answerable regardless of LLM sampling. The
linux-fundamentals eval (code doc present, `refusal_cf=1.0`) is a confirmatory backstop, not the
proof. Tests: `test_rust_symbols.py` (grammar + edge cases: brace-in-string / char-lit / raw-string /
lifetime / multiline-sig / multiline-attr / unbalanced→graceful), `test_source_types.py`,
`test_code_chunking.py` (rust → symbol chunks + `heading_path`; prose byte-identical; non-rust →
prose + `v0`), `test_partial_reindex.py` (recipe record + force-on-mismatch), `test_enrich_and_graph.py`
(code MENTIONS parity).

## Phases 3–5 — planned

- **Phase 3 — BM25 for code, MEASURE-FIRST → MEASURED 2026-06-09, verdict GO (complementary).**
  Record: [`docs/audits/13-code-bm25-arm-separation.md`]. The prose "BM25 recall ⊆ dense" finding
  (`docs/audits/09`) **inverts for code, but only on the usage/reference regime.** On a 357-chunk
  `codex-rs/core` slice + a 20-query find-the-code corpus (`tests/eval-data/codex-rs-find-the-code/`,
  the Phase 4-5 seed) measured via the read-only `scripts/code_bm25_arm_probe.py` (deterministic, N=2
  byte-identical): **definition queries** dense recall@50 = 1.00 (Phase-2's symbol-as-embed-title lever
  saturates them — BM25 redundant); **usage/reference queries** ("which fn calls X", "where is X used",
  gold titled by a DIFFERENT symbol) dense = 0.70 → **bm25-term = 1.00, 3/3 dense-misses recovered**;
  the LIVE bm25-phrase = 0.05 (≈ nothing). So BM25-for-code is a COMPLEMENT that patches the
  usage/reference gap, NOT "BM25 beats dense." The recommended builder is **term-WHOLE** (OR'd quoted
  WHOLE identifiers; `_`-sub-splitting buys 0 recall + adds counterfactual noise). **The BUILD is gated
  on a user go-ahead** (it edits the SHARED `FTSStore.search`): a scoped, env-gated code-path term-query
  for the code path ONLY (never the prose phrase-wrap), shipped with the prose HARD-gate corpora re-run
  + the prose arm-separation re-confirmed. Lever B (FTS `tokenchars=_`, reindex-forcing) is NOT needed —
  the quoted-WHOLE term already matches snake_case contiguously.
- **Phase 3 BUILD — Lever A SHIPPED 2026-06-09 (default-ON, prose-validated).** `index/code_query.py`
  (`query_has_code_identifier` = the per-query gate: a query with a `_` or camelCase token; the validated
  `build_code_term_match` WHOLE builder; `code_term_query_enabled` fail-open) + a `term_query: bool=False`
  param on `FTSStore.search`/`search_in_docs` (branch the MATCH; empty-term → phrase-wrap fallback; default
  byte-identical) threaded ONLY through `retrieve/hybrid.py`'s 2 /ask calls as
  `term = code_term_query_enabled() and query_has_code_identifier(query)`. Config flag
  `AgentsSettings.code_term_query_enabled: bool = True` (kill-switch `MEMEX_AGENTS__CODE_TERM_QUERY_ENABLED=
  false`). **Query-side only → NO reindex.** Validation: (a) production `hybrid_search` flag-ON recovers
  3/3 usage golds (flag-OFF misses all 3; definitions rank-1 either way); (b) the 8 detector-triggering
  prose queries — incl. 3 COUNTERFACTUALS — are **byte-stable flag ON vs OFF, N=2** (`refusal_cf`=1.0 held;
  answered counts identical), and every non-triggering prose query is byte-identical by construction. The
  recommended **term-WHOLE** variant ships (sub-token splitting rejected — 0 recall gain, counterfactual
  noise). See `docs/audits/13` + ADR-0021 (Amendment).
- **Phases 4–5 — corpus + baseline — DONE 2026-06-09 (`docs/audits/14`).** The full `codex-rs` (99 `.rs`)
  is now a permanent MAIN-vault corpus (~1235 chunks among 7356; vault backed up first), with a 47-query
  find-the-code set (16 definition + 17 usage/reference + 6 big-function truncation probes + 8
  counterfactuals). **Baseline: retrieval `rrf@50` (production) = 1.000 / dense 0.923 (usage-regime Δ via
  Lever A +0.130, 3/3 dense-misses recovered); grounded answering 38/39 ANS answered, `refusal_cf`=1.0
  (N=3 — the HARD gate holds on code even with 37–50 lexical counterfactual hits); cp 0.77.** An
  answer-correctness spot-check (N=8, usage-weighted) found 6/8 correct but **2 usage-class answers
  wrong/partial** — the answer stage prefers the title-matching DEFINITION over the usage gold (`memex
  eval` scores answered/cp, not answer text). So retrieval + the HARD gate are validated; answer
  correctness has a usage-class gap — now root-caused; the rerank remedy is DOUBLE-EDGED → shipped
  DEFAULT-OFF (see the usage-class-answer-gap bullet below + audit-14). The
  shared-vault prose gates HELD with code present (`refusal_cf`=1.0 on the 3 detector-triggering prose
  counterfactuals — the Lever A × `model_provider_info.rs`-openai interaction is safe). **Truncation
  measured-benign:** 5/6 big-function probes (incl. the 2 largest, 4912/4647 chars) answer despite
  `truncate(1800)`; the lone false-refusal (`run_main`, an entry-point fn) is borderline relevance, not
  truncation — recorded, no fix. The metric was the standalone `code_bm25_arm_probe.py` (recall@k) +
  `memex eval` (grounded answering); the gold_chunk_recall stays a committed probe, not a new eval mode.
- **Entry condition before the full ingest** (a doc-identity hazard): `doc_id = sha256(bytes)[:8] +
  slug(basename-stem)`, so two byte-identical same-stem files (e.g. two trivial `mod.rs`) merge to one
  doc and the second `retitle` silently overwrites the first's path. Pre-check
  `#distinct(sha256, stem) == #files`; if short, fold the repo-relative path into the hashed identity
  or accept+log the merges deliberately.
- **The usage-class answer gap — root-caused, a rerank lever measured DOUBLE-EDGED, shipped DEFAULT-OFF
  (2026-06-09, `docs/audits/14`, ADR-0021 amendment).** The audit-14 spot-check residual ("which function
  calls X" answered by describing X's DEFINITION) was investigated. Root cause: the cross-encoder ranks
  X's definition + tests above the caller, AND the answer LLM describes the definition whenever a
  definition chunk of X is in its visible top-k window. The remedy — a **usage-intent rerank-demotion**
  lever (`index/code_query.detect_usage_intent` + `reorder_for_usage_intent` → the `rerank` node; demote
  X's def + tests below the top-k cut for a "which/where … calls/uses X" query; HARD-gate-safe pure
  reorder; detector validated 17/17 usage, 0/16 def, 0/6 big-fn) — was measured on all 17 usage queries
  (N=2, text-verified): **net +3 / −2** — fixes 3 definition-distraction cases (isknownsafe, unifieddiff,
  issafetocall) but REGRESSES 2 previously-correct ones (convhistory → WRONG answer, the demoted definition
  was disambiguating a similarly-named sibling; applyhunks → FALSE REFUSAL, over-demotion). Demoting the
  definition is double-edged (distractor vs necessary context); no clean rerank rule. **Shipped DEFAULT-OFF**
  (`usage_intent_demotion_enabled=False`, kill-switch `MEMEX_AGENTS__USAGE_INTENT_DEMOTION_ENABLED=true`):
  validated opt-in infra + tests kept; the default product is byte-identical to the pre-lever baseline. The
  clean revisit path is a sharper reranker, not a cleverer demotion heuristic.

## References

- [ADR-0021](../adr/0021-codebase-corpus-code-as-documents.md) — the decision.
- `docs/audits/09-fts-bm25-arm-separation.md` — the prose BM25 finding Phase 3 re-measures.
- The transient-transform precedents: `reattach_chart_extractions` (the chart-sidecar, #362) and
  `linearize_gfm_tables` (`[table-rows]`) — the pattern symbol injection rides.
