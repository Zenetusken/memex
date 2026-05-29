# FTS BM25-on-NL questions: arm-separation probe (2026-05-29)

Settles whether the FTS5 phrase-wrap "bug" is worth fixing, against the real
eval corpora, not a hypothesis. **Verdict: the bug is REAL but PROVABLY BENIGN —
a dead lever.** Rig: RTX 4070, 12 GB; embedder + reranker on CPU.

## The bug

`FTSStore.search` / `search_in_docs` wrap the whole query as one double-quoted
FTS5 **literal phrase**. A natural-language question never matches as a
contiguous phrase, so BM25 returns nothing and contributes 0 to the RRF fusion —
**hybrid retrieval is effectively dense-only on every real question** (measured:
58/58 ANS questions across 5 corpora returned `bm25=0`). Present since the first
FTS commit.

## The decisive probe (arm separation)

A term-query fix was implemented, then validated with a deterministic
`gold_chunk_recall@k` probe (`eval/scoring.py`) over each corpus's
`relevant_chunk_ids` — measuring three retrieval arms separately at k=50:
dense-only, bm25-only (with the fix active), and their union (the ceiling any
fusion could reach).

| corpus | dense@50 | bm25@50 | union@50 | n |
|---|---|---|---|---|
| nist-zero-trust | 1.00 | 1.00 | 1.00 | 12 |
| technical-guidelines | 1.00 | 1.00 | 1.00 | 12 |
| scientific-gte | 0.58 | 0.58 | 0.58 | 12 |
| cr350-multidoc | 0.71 | 0.64 | 0.71 | 14 |
| ccna-multidoc | 0.44 | 0.44 | 0.44 | 8 |
| **OVERALL** | **0.767** | **0.750** | **0.767** | 58 |

**`union@50 == dense@50 == 0.767` on every corpus.** The bm25 arm reaches ZERO
gold chunk the dense arm misses — its recall is a strict **subset** of dense
(cr350: bm25 0.64 < dense 0.71; identical elsewhere). There is no lexical
headroom; a correctly-working lexical arm recovers nothing new. (`answering.py`
feeds all 50 hybrid candidates to the reranker, so recall@50 *is* the reranker's
input pool — and it's byte-flat with or without the fix.)

## Why

EmbeddingGemma + its native `task:`/`title:` prompts (ADR-era 2026-05-25)
dominates the lexical arm on this corpus. The real recall gaps (scientific 0.58,
ccna 0.44) are chunks **neither** arm gets into the top-50 — a chunking /
query-mismatch problem, not a BM25 problem; no smarter FTS builder could close
them (the union ceiling already equals dense).

## Disposition

- **Reverted** the term-query fix (the kill-switch flag + the `_build_fts_match`
  builder + its tests are gone; `config.py` is zero-diff vs HEAD).
- **Kept** the behaviour-neutral useful parts: `core/text.py` `STOPWORDS` +
  `atomise` promotion (+ the `artifact_scope` de-dup that rides on it); the
  `gold_chunk_recall` scorer in `eval/scoring.py` + `tests/unit/test_retrieval_scoring.py`.
- `index/fts_store.py` is back to the literal phrase-wrap, now carrying a
  "validated benign, do not re-fix" docstring on `search()`.

## Recommendation

**Don't re-fix the phrase wrap without re-measuring.** The only revival path is a
**future embedder swap (roadmapped P2.5)** with worse dense recall — re-run this
probe then. The same cheap-probe discipline that retired this (and the Cisco
orchestrator pre-check) is documented in `docs/GUIDELINES.md`.

## Reproducibility

```sh
# Arm-separation recall@50, dense vs bm25 vs union, per corpus.
# eval/scoring.py::gold_chunk_recall over each corpus's relevant_chunk_ids;
# run hybrid_search per ANS query and pass the ranked chunk_ids.
MEMEX_MODELS__EMBEDDER_DEVICE=cpu MEMEX_MODELS__RERANKER_DEVICE=cpu \
  uv run python - <<'PY'   # see the session harness; gold_chunk_recall is the scorer
PY
```

The disposable probe scripts lived under `/tmp` during the session; the
conclusion above (and the `fts_store.py::search` docstring) is what survives.
