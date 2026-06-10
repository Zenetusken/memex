# Audit 16 — Parent-Child (Multi-Vector) Retrieval: Theory Validation

**Date:** 2026-06-10
**Subject:** Theory A from `docs/RAG-COMPARISON-AND-UPGRADES.md` — "index small (150-word) child chunks for retrieval, expand to large (800-word) parent chunks at generation time."
**Verdict:** **NO-GO as written.** A reshaped, HARD-gate-safe variant (additive neighbor-window augmentation) is plausible and gets a measure-first probe (`scripts/parent_context_probe.py`) before any build decision.

---

## 1. The theory under audit

The parent-document / multi-vector retriever is standard RAG lore (LangChain `ParentDocumentRetriever`, Kotaemon's multi-vector indexing): embed small chunks for precise vector matching, but hand the LLM the larger enclosing window so it answers with full context. The claimed fit for Memex: cheap retrieval/rerank over small chunks, wide context into Qwen3.5-4B's 8,192-token window, no vector-store bloat.

## 2. Findings against the codebase

### F1 — The storage premise is already satisfied (no schema gap)

Chunks are persisted in **both** stores today: LanceDB rows (`src/memex/index/vector_store.py::_ChunkRow`) carry the full text, and SQLite keeps the **unstripped** text in `chunks_meta.full_text` (`src/memex/index/fts_store.py::upsert`). Reading-order access already exists: `FTSStore.chunks_for_document` returns a document's chunks ordered by `char_start`, and `chunks_by_ids` fetches arbitrary ids. The theory's "add a parent store to SQLite" step is solving a problem the codebase doesn't have — parent context is reconstructible for free from `Chunk.char_start`/`char_end` against the canonical `.md`, or by walking the ordered chunk list.

### F2 — 150-word children is a full-vault migration with no demonstrated payoff

`chunk_id = sha1(chunk_text)[:10]` (`src/memex/index/chunker.py::_stable_chunk_id`). Changing the chunk target from the current 400 words / 60 overlap (`IndexSettings.chunk_target_tokens`) to 150-word children would:

- churn **every chunk_id in the vault** → full re-embed of ~7,356 chunks;
- bump `chunking_recipe_version` and invalidate the enrich MENTIONS chunk-id attestations (the #394 parity contract);
- invalidate every eval corpus's `relevant_chunk_ids` gold labels (15 corpora would need relabeling);
- shrink the per-chunk evidence the reranker scores — the bge cross-encoder's documented strength here is *fact-extraction* ranking (the P2.1 A/B), which benefits from the answer and its surrounding sentences co-occurring in one scored unit.

That cost demands a measured recall gain, and the measurements say there isn't one (F4).

### F3 — The 800-word parent would be truncated away, or would evict its siblings

The answer prompt truncates each chunk to **1,800 chars** (`agents/answering.py`; `MAX_CHUNK_CHARS` in the chunker is matched to it precisely so table chunks stay fully answer-visible). The production window is `top_k=5 × 1800` chars inside `max_model_len=8192` (`fast`/`manual`; `full` trades the reranker's GPU slot for ~18 chunks at 24K). An ~800-word parent is ~4,800+ chars:

- under the existing truncate it ships at **37%** — the "wide context" claim is voided;
- raising the truncate to fit parents forces `top_k`→2 in the 8K window — **less** evidence diversity per answer, the opposite of the goal;
- the overflow degrade path (vLLM 400 → drop chunks) starts firing routinely on table-dense docs.

The window budget is a zero-sum resource the theory's framing ignores.

### F4 — Retrieval recall is measured-saturated; the only candidate benefit is answer-stage

- `audits/09`: union@50 == dense@50 on every prose corpus — BM25 recall ⊆ dense, no lexical headroom.
- NER leverage audit (2026-05-29): entity-as-query-signal probe — **58/58 gold docs already in dense@50**, "CONCLUSIVE no headroom."
- `audits/14`: codex-rs production `rrf@50` recall = **1.000** against 7,356 competing chunks.

So smaller/denser children cannot improve what's already at ceiling. The plausible benefit is **answer-stage context width** — consistent with the audit-14 usage-class answer gap and the ADR-0022 false-refusal arc. But two precedents demand caution:

- truncation was measured **BENIGN** on the big-function probes (5/6 answered despite `truncate(1800)`, audit-14) — "more context helps" is an assumption that has already failed once here;
- the contextual-retrieval LLM-prefix (Anthropic-style chunk contextualization) was built and **REVERTED** as a measured negative (2026-05-25) — standard-RAG-lore features do not transplant on faith.

### F5 — Text-swap is HARD-gate-invasive; the codebase already owns the safe pattern

The no-hallucination gate's contract: every claim's `source_chunk_id` names a chunk, and `verify` grounds the claim **against that chunk's text**. The canonical parent-document retriever *swaps* the retrieved child's text for the parent's at generation time — which means either (a) claims cite child ids while the model saw parent text (the gate verifies against text the model didn't ground on), or (b) claims cite synthetic parent ids that don't exist in the index (breaking citation navigation, `repair_claim_chunk_ids`, and the eval gold sets). Both are unacceptable mutations of the gate's substrate.

The codebase already demonstrates the safe alternative: `agents/answering.py::augment_companion` (ADR-0018) appends **real indexed chunks** (with their own ids and single sources) after rerank — additive, per-chunk-pure, bounded, HARD-gate-safe *by construction*. A parent-context feature must take this shape.

## 3. The reshaped variant (the only buildable form)

**Neighbor-window augmentation:** after `rerank`, for the top-N winners, append their reading-order adjacent chunks (±1 by `char_start`, same document, via `chunks_for_document`), de-duplicated, capped (~4 added chunks), behind a default-OFF flag. Properties:

- zero index/schema changes, zero migration, zero chunk-id churn;
- every appended chunk is a real chunk → `verify` grounds per-chunk exactly as today;
- window cost: +4 × 1800 chars ≈ +1,650 tokens — fits the 8K window at `top_k=5` with the existing scaffold (same budget class as `augment_companion`'s cap);
- the chunker's 60-word overlap already gives partial adjacency coverage — which *lowers* the expected headroom and is precisely why this must be measured first.

## 4. Stage-1 probe (shipped with this audit)

`scripts/parent_context_probe.py` — read-only, production untouched. Per answerable eval query: production `hybrid_search(k=50)` → `cross_encoder_rerank(top_k=5)`, then the bounded neighbor expansion, then exclusive classification:

| Bucket | Meaning |
|---|---|
| `gold_in_winners` | production window already carries gold — no headroom |
| **`HEADROOM`** | gold NOT in winners but IS adjacent to a winner — the only REF→ANS convertible class |
| `gold_below_cut` | gold in pool, below the rerank cut, not adjacent — a rerank-depth issue, not a parent issue |
| `gold_missed` | retrieval miss |

Secondary: `adjacency_extension` (gold already in winners AND further gold chunks adjacent — answer-completeness signal, not REF→ANS).

Run (CPU retrieval models, main vault):

```sh
MEMEX_MODELS__CO_RESIDENCE_MODE=manual \
MEMEX_MODELS__EMBEDDER_DEVICE=cpu MEMEX_MODELS__RERANKER_DEVICE=cpu \
uv run python scripts/parent_context_probe.py \
    tests/eval-data/slide-decks/queries.json \
    tests/eval-data/annual-report/queries.json \
    tests/eval-data/linux-fundamentals/queries.json \
    tests/eval-data/codex-rs-find-the-code/queries.json
```

## 5. GO/NO-GO criteria

- **GO** (build Stage 2): `HEADROOM` > **10%** of answerable queries on any corpus. Then: implement the variant as a default-OFF post-rerank node (the `augment_companion` template), and run the live `memex eval` A/B with gates: ANS must not regress on any corpus, **`refusal_cf` must hold 1.0** (additive evidence must be confirmed harmless to the counterfactual gate, same as the companion validation), and the slide-decks + annual-report N≥3 sweeps must be stable.
- **NO-GO**: record the measured table below and close Theory A; `gold_below_cut` mass, if large, instead motivates a separate rerank-depth experiment (raising `top_k` in `full` mode already does this).

## 6. Results

*(to be filled after the probe run)*

| corpus | answerable | in_winners | HEADROOM | below_cut | missed | headroom% |
|---|---|---|---|---|---|---|
| | | | | | | |

## 7. Cross-references

- `docs/RAG-COMPARISON-AND-UPGRADES.md` §3 Theory A — amended with this verdict.
- `agents/answering.py::augment_companion` — the additive-augmentation template (ADR-0018).
- `audits/09`, `audits/14`, the 2026-05-25 contextual-retrieval revert, ADR-0022 — the prior measurements this audit leans on.
