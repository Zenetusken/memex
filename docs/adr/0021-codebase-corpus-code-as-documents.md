# ADR-0021: Codebase Corpus — Source Code as Documents

- **Status**: Accepted (Phase 1 + code-view + Phase 2 shipped 2026-06-07; Phases 3–5 planned)
- **Date**: 2026-06-07
- **Deciders**: Memex core team
- **Tags**: ingest, parse, index, retrieval, eval, doc-type, architecture

## Context

Memex ingests prose-shaped documents (PDF / Office / scan / image / audio / video). A **source-code
repository** is a different shape, and a desirable one to make queryable: "where is `X` defined",
"what does `fn Y` do", "which module handles `Z`" are document-intelligence questions, not agentic
coding. The goal is the **same grounded contract** as every other corpus (answer from the vault or
refuse; `refusal_cf=1.0`), with **razor-sharp retrieval precision** — a code question should land on
the exact symbol, not a fragment.

**The existing pipeline mangles code.** Anything not md/media/office/image/pdf fell through to the
Docling fallback, whose layout model renders aligned code as a markdown **pipe-table** (a `.rs` file
stored as `| mod cli | |`, then size-chunked so functions fragment and `heading_path` is empty).
Verified empirically this session: code ingested end-to-end but as unusable pipe-table soup. So code
needs a real adaptation, not a config tweak — which makes it ADR-worthy (a new doc-type).

The constraints are the project's standing ones: **local-first / air-gap** (no heavy new dependency
— so no tree-sitter), **Markdown-as-source-of-truth** (the canonical `.md` stays the verbatim
source; ADR-0003), and **HARD-gate neutrality** (a new doc-type must not be able to make a
counterfactual answerable). The reference target is `codex-rs` (~99 `.rs` files — coherent,
single-language, clean `fn`/`struct`/`impl` symbol boundaries).

## Decision Drivers

- **Retrieval precision** — the whole point. A code query must rank the defining symbol, which means
  splitting on **symbol boundaries**, not a size budget.
- **Verbatim fidelity** — what-you-read = what-was-ingested. The stored `.md` must be the literal
  source file, not a re-serialized approximation (ADR-0003).
- **HARD-gate neutrality by construction** — the code path must be *gated* so prose chunking is
  byte-identical, and off-topic code chunks must be unable to make a counterfactual answerable.
- **Local-first / no heavy dep** — a lightweight per-language symbol splitter over tree-sitter.
- **Reuse the blessed seams** — the transient-transform pattern (`reattach_chart_extractions` /
  `linearize_gfm_tables`) and the recipe-version migration (`embedding_recipe_version`) already
  exist; ride them rather than adding chunker surgery.

## Considered Options

1. **Code as documents, verbatim ingest + transient symbol chunking** (chosen)
2. **Leave code on the Docling fallback** — rejected; produces pipe-table soup, no symbol structure.
3. **Tree-sitter symbol extraction** — rejected for v1; a heavy native dependency against the
   air-gap principle. A lightweight per-language splitter gets Rust's clean grammar at no dep cost;
   tree-sitter is *banked* if/when many languages with irregular grammars are needed.
4. **Persist `## <symbol>` headings into the canonical `.md`** — rejected; it would break the
   content-only invariant (the `.md` would no longer be the verbatim source).

## Decision

**Treat source code as a first-class document type, in phases.** The mechanism detail lives in the
spec [`docs/specs/code-chunking.md`](../specs/code-chunking.md) and the backend cheat-sheet
(`src/memex/CLAUDE.md`); this ADR records the decision and its shape.

- **Phase 1 — verbatim ingest (shipped, `268e39e`).** A new `code` `DetectedKind`
  (`ingest/validation.py`: a known suffix ∈ `core/source_types.py::CODE_SUFFIXES` **and**
  `_looks_like_text`) that is **absent from `_EXTENSION_FOR_KIND`** so the original suffix is
  preserved (the image/audio precedent — losing `.rs`→`.txt` would lose the language). `parse_document`
  routes a `CODE_SUFFIXES` source to the existing **`_passthrough_markdown`** (verbatim UTF-8, no
  chart-OCR/VLM) **before** the Docling fallback. The doc **title is the repo-relative path**
  (`exec/src/lib.rs`), not the basename — `lib.rs`/`mod.rs`/`main.rs` collide across crates.
  `scripts/ingest_codebase.py` walks a repo (repo-relative titles, skips `.git`/`target`/
  `node_modules`).

- **Code-view rendering (shipped, `6e477b3`).** A source-code doc renders **as code**: the webui
  detects it (`webui/app.py::_is_code_source`, suffix ∈ `CODE_SUFFIXES`) and renders the verbatim
  body in a `<pre>` under a `source · <language>` label, **suppressing the markdown heading-anchor +
  `[[wikilink]]` transforms** that misfire on code (a Python/shell `# comment` is not an H1; a literal
  `[[x]]` is not a link). Presentation-only, zero template edits, HARD-gate-neutral.

- **Phase 2 — symbol-aware chunking (shipped, `b9c3a58`).** A pure Rust symbol splitter
  (`index/rust_symbols.py::split_rust_symbols`) finds `fn`/`struct`/`enum`/`trait`/`impl`/… spans
  via a masking lexer + brace-depth scan, and `inject_symbol_headings` prepends transient
  `## <symbol>` ATX lines **into the copy the chunker sees only** — never the canonical `.md`. This is
  the established `reattach_chart_extractions`/`linearize_gfm_tables` transient-transform pattern,
  shared via `index/pipeline.py::build_chunking_body` (index + enrich parity) and migrated via
  `chunking_recipe_version` (`"code-rust-v1"` iff Rust injection ran, else `"v0"`). The deepest
  `heading_path` (now the symbol) flows into the EmbeddingGemma doc-prompt title — that is the
  precision lever (live: "how is `FunctionCallOutputPayload` serialized" ranks the exact method #1).

- **Phases 3–5 — planned.** Phase 3: **BM25-for-code, measure-first** — re-run the dense-vs-union
  arm-separation probe on *code* queries (the prose "BM25 ⊆ dense" finding, ADR/audit-09, likely
  *inverts* for identifier search where EmbeddingGemma is off-distribution); only if BM25 adds recall
  does a scoped code-path branch ship (with the prose corpora's HARD gates re-run, since `FTSStore.search`
  is shared). Phases 4–5: full `codex-rs` ingest + a find-the-code query corpus + a `gold_chunk_recall@k`
  baseline (device-pinned, multi-run; re-verify the shared-vault gates still hold). **Entry condition
  before the full ingest:** `#distinct(sha256, basename-stem) == #files` (two byte-identical same-stem
  files merge to one doc and the second `retitle` silently overwrites the first's path).

## Consequences

### Positive

- A codebase becomes a queryable, grounded corpus with symbol-precise retrieval — the same air-gap,
  same refusal contract as every other domain.
- The canonical `.md` is the **verbatim source file** (content-only invariant intact, ADR-0003): the
  symbol headings exist only in the transient chunking body.
- HARD-gate neutrality is **true by construction**, not measured: the code path is gated (prose
  chunking byte-identical, verified 16/16 chunk_ids), and off-topic code chunks cannot make a
  counterfactual answerable regardless of LLM sampling. The linux-fundamentals eval (code doc present,
  `refusal_cf=1.0`) is a confirmatory backstop, not the proof.

### Negative / Trade-offs

- **Chunk *text* is not byte-verbatim** (only the canonical `.md` is): the chunker's
  `_PARAGRAPH_RE.split` + `.strip()` normalizes blank runs / post-blank indent — cosmetic for Rust,
  but a second reason Python is out of v1 (significant whitespace).
- **Rust-only v1.** A Python symbol splitter misfires on `# comment` lines and significant-whitespace
  blocks — a separate increment, not a v1 corner-cut.
- **No in-source citation anchors yet** (`#fn-foo` jump-to-symbol) — deferred to a v2.
- We bet that the lightweight per-language splitter scales to the languages we care about before
  tree-sitter's cost is justified.

### Neutral

- `chunking_recipe_version` makes a future grammar/granularity change a clean force-rechunk (only new
  code docs churn; additive to existing prose docs).

## Revisit When

- A non-Rust codebase corpus is wanted at scale — promote the Python splitter (its own increment) or
  reconsider tree-sitter if many irregular-grammar languages pile up.
- Phase 3 measurement shows BM25 recovers identifier chunks the dense arm misses — ship the scoped
  code FTS branch.
- Citation UX needs jump-to-symbol — build the in-source anchor (v2).
- The full-ingest entry-condition pre-check fails — fold the repo-relative path into the hashed doc
  identity, or accept+log the merges deliberately.

## References

- Spec: [`docs/specs/code-chunking.md`](../specs/code-chunking.md) — the build-level mechanism.
- ADR-0003 (Markdown vault as source of truth) — the content-only invariant the transient transform
  preserves.
- `docs/audits/09-fts-bm25-arm-separation.md` — the prose BM25 finding Phase 3 re-measures for code.
- Commits: `268e39e` (Phase 1), `6e477b3` (code-view), `b9c3a58` (Phase 2).
- Backend cheat-sheet: `src/memex/CLAUDE.md` (the code-ingest + symbol-chunking bullets).
