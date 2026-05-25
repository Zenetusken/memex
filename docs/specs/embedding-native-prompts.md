# Spec: EmbeddingGemma native `task:`/`title:` prompts

**Status:** ✅ SHIPPED + GPU-accepted 2026-05-25 (1 spec-validation round + 1 impl-validation round, both SHIP). Clean same-vault A/B (bare vs native, all 5 corpora): HARD GATES perfect on both (`refusal_cf=1.0`, 0 hallucinations); native a strict marginal win (slide-decks 15→16 ANS / one fewer false-refuse; others flat; citation_precision flat-modulo-sibling-granularity). Default ON (`MEMEX_EMBED_NATIVE_PROMPTS=1`); `=0` self-heals to bare via the recipe bump.

## Problem
Memex embeds BOTH queries and document chunks **bare** — `retrieve/hybrid.py::_embed_query` and `index/pipeline.py::_embed_chunks` call `embedder.encode([...])` with no prompt. EmbeddingGemma is a **prompt-trained asymmetric retriever**; its model card prescribes:
- queries → `task: search result | query: {q}`
- documents → `title: {title|"none"} | text: {chunk}`

Embedding bare is **off the model's trained distribution** and leaves recall on the table. This is the in-distribution, schema-preserving lever that BOTH the contextual-retrieval negative result (`memory/contextual_retrieval_negative_2026_05_25`) and the Gemma-4 embedder research (`memory/gemma4_embedder_research_2026_05_25`) independently pointed to as the right move — and it's a prerequisite gate for the deferred P2.5 embedder swap. Bare vs prompted embeddings differ materially (measured cosine ≈ 0.69), so this is a real change, eval-gated.

## The invariant (HARD-gate safety — same as always)
The prompt wraps ONLY the **embedding input** (query string at query time; a `title: … | text: …` wrapper at index time). `chunk.text`, `chunk_id` (`sha1(chunk.text)`), and the stored/retrieved chunk text (`vector_store._ChunkRow.text`, FTS body) are **unchanged** → the prompt never reaches assess/answer/verify; it only changes which chunks retrieve. Refusal/grounding logic unaffected.

## Design

### 1. Prompt helpers — `src/memex/index/embed_prompts.py` (NEW; pure-sync, `core/`-only deps)
- `EMBED_QUERY_PROMPT_NAME = "query"` — EmbeddingGemma's built-in prompt (`task: search result | query: `), applied via sentence-transformers `prompt_name=` (the model ships this prompt in its ST config; `registry._load_embedder` preserves it — no loader change). `search result` is the retrieval task (correct for RAG).
- `document_input(title: str, text: str) -> str` → `f"title: {title} | text: {text}"` — the document side is built MANUALLY (not `prompt_name="document"`, whose built-in is hardcoded `title: none | text: ` and would discard the heading). Matches EmbeddingGemma's trained doc format with a real title.
- `chunk_title(chunk: Chunk) -> str` → the deepest `heading_path[-1]` if non-empty, else `document_title`, else `"none"` (most-specific available locating signal; short — no homogenization risk, unlike the failed LLM context). Pure function, unit-tested.

### 2. Query side — `src/memex/retrieve/hybrid.py::_embed_query`
`embedder.encode([query], prompt_name=EMBED_QUERY_PROMPT_NAME, normalize_embeddings=True, convert_to_numpy=True)` when native prompts are enabled, else bare (current). `_embed_query` is the SINGLE query-embed entry point (both `hybrid_search` + `hybrid_search_in_docs` route through it — grep-confirmed), so one change covers all query paths symmetrically.

### 3. Document side — `src/memex/index/pipeline.py::_embed_chunks`
`embedder.encode([document_input(chunk_title(c), c.text) for c in chunks], normalize_embeddings=True, convert_to_numpy=True, batch_size=…)` when enabled, else `[c.text for c in chunks]` (current). The `Chunk` objects flowing to the stores keep their original `text`.

### 4. Toggle + re-embed migration (`core/config.py`/module + `core/manifest.py` + `pipeline.py`)
Query- and doc-side MUST change together (a bi-encoder needs query+doc in the matching space) → a doc-side change requires a full re-embed, triggered automatically:
- `MEMEX_EMBED_NATIVE_PROMPTS` env (default `"1"` = ON; `"0"` = bare, the revert/A-B path). Read per-call in `_embed_chunks`/`_embed_query` (mirrors `MEMEX_INDEX_EMBED_BATCH`).
- `IndexStage.embedding_recipe_version: str = "v0"` (back-compat default; `write_manifest`'s `exclude_none=True` keeps a non-None string). NB the live `~/.memex/vault/.memex/manifests/*.json` already physically contain `"embedding_recipe_version": "v0"` (written by the now-reverted contextual work's bare baseline; the reverted model silently ignores the extra field today). Re-adding the field with default `"v0"` matches that on-disk value → no spurious force on the current bare vault; enabling native prompts (`v0`→`v1-gemma-prompts`) cleanly forces the re-embed.
- `pipeline._embed_recipe_version() -> str` pure helper: `"v1-gemma-prompts"` when enabled, else `"v0"`.
- Extend the force-detection (the `prior.index.embedding_model != settings.models.embedder` block, ~pipeline.py:163) to OR-in `prior.index.embedding_recipe_version != _embed_recipe_version()` → `force=True`, log `index.recipe_changed`. Manifest write records the recipe. → enabling/flipping native prompts auto-forces a full re-embed on the next `memex index`/`reindex` (no `--force` flag); toggling off self-heals back to bare. (This re-introduces the minimal recipe-version mechanism — justified for a default embedding change that must trigger re-embed; unlike the discarded contextual-retrieval, this feature is in-distribution + expected to work.)

NB: NO vLLM dependency (unlike contextual-retrieval — this is pure prompt strings, no generation). The re-embed is embed-only / GPU-light.

## Tests
- `tests/unit/test_embed_prompts.py` (NEW): `document_input` format (`title: X | text: Y`); `chunk_title` (deepest heading → doc_title fallback → "none"; empty heading_path); `_embed_recipe_version` on/off; `EMBED_QUERY_PROMPT_NAME` constant.
- Integration (`tests/integration/test_partial_reindex.py` + a query-embed test, fake embedder capturing `encode` args): (a) with prompts ON, the captured doc-embed input is `title: … | text: …` and the query-embed call passes `prompt_name="query"`; with OFF, both are bare; (b) **chunk_id byte-identical** ON vs OFF (prompt never feeds `_stable_chunk_id`); (c) retrieved `Chunk.text` is unprefixed (leak guard); (d) recipe-version bump (OFF→ON) forces re-embed (mirror `test_embedder_change_triggers_implicit_force`). The fake embedder must accept `prompt_name=`/`**_kw: object`.
- Gates: `uv run pytest tests/ -q`, `uv run pyright` (0/0), `ruff check` + `format`.

## Migration + eval-gating (measure-first; HARD GATES immutable)
1. Land code (default ON). Gates green. Commit.
2. **Re-embed (embed-only, GPU-light, vLLM not required):** `uv run memex reindex` — the recipe mismatch (`v0`→`v1-gemma-prompts`) auto-forces a full re-embed with the new prompts. (`MEMEX_INDEX_EMBED_BATCH=4` if VRAM tight co-resident with vLLM, though vLLM isn't needed for embed.)
3. Anchors: chunk_ids unchanged (prompt is input-only) → committed anchors stay valid; **do NOT re-resolve** (the FTS body is untouched; re-resolving would be a no-op — and the contextual-retrieval lesson was that anchor-drift via FTS pollution is harmful, N/A here since FTS body is unchanged).
4. Re-run all 5 corpora with `MEMEX_RERANK_BATCH_SIZE=1`: annual-report, slide-decks, chart-types, french-course, cr350-multidoc.
5. **Accept:** `refusal_cf=1.0` + 0 hallucinations MUST hold (anchor-independent). Retrieval ANS counts + `mean_citation_precision_answered_only` should improve or at least not regress (esp. cr350-multidoc disambiguation + french-course, where the asymmetry fix should help most). If it regresses or breaks a gate → toggle `MEMEX_EMBED_NATIVE_PROMPTS=0` → recipe bump → clean re-embed to bare. Self-healing, revertible.

## Risks / edge cases
- **Query↔doc coupling**: changing only one side mismatches the shared space → the recipe couples them (both flip together). Confirmed.
- **Title-slot choice** (deepest heading vs doc title): eval validates; short either way (low homogenization risk — the failed LLM context was ~50 tokens; a heading is ~2-6 words). If cr350 multi-doc disambiguation regresses, revisit (doc_title may disambiguate better than a heading); the toggle makes the A/B cheap.
- **Query task type**: `search result` (retrieval) is correct for RAG; not `question answering`/`fact checking`. Documented.
- **prompt_name availability**: depends on EmbeddingGemma's ST config shipping the `query` prompt (grounding confirmed `model.prompts` includes it). If a future embedder lacks it, fall back to a manual `task: search result | query: ` string — note in the helper.
- **HARD-gate safety**: prompt is index/query-input only, never in the answer/verify prompt. The retrieved chunk text is unchanged. (This is the key DIFFERENCE from the failed contextual-retrieval: the prompt is SHORT + in-distribution + symmetric, not a long off-distribution LLM sentence on one side only.)

## Acceptance
5-corpora HARD GATES hold (`refusal_cf=1.0`, 0 hallucinations); retrieval ANS/citation_precision improve or hold (target: cr350 + french improve from the asymmetry fix); `memex ask` shows no prompt leak in the answer/cited text; the recipe-version migration auto-re-embeds on upgrade + self-heals on toggle-off.

## Anti-scope
- No embedder swap (that's the gated P2.5, downstream of this).
- No Matryoshka `truncate_dim` change (768 stays).
- No new task types beyond `search result` (retrieval) in v1.
- Not a no-op revert of the contextual-retrieval machinery — this is a distinct, minimal, in-distribution change (short native prompts, symmetric, no LLM, no FTS context column).
