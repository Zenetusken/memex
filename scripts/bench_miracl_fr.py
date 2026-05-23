"""MIRACL-fr reranking benchmark — measures Memex's retrieval stack
(EmbeddingGemma 300M + bge-reranker-v2-m3) on the MTEB-standard
MIRACL-FR-Reranking benchmark.

Phase 3 of the French support hardening sequence. Provides external,
published-benchmark numbers on the multilingual retrieval stack so
we can compare against bge-reranker-v2-m3's official scores.

Dataset: OrdalieTech/MIRACL-FR-Reranking-benchmark (343 test queries,
each with multiple positive + negative French Wikipedia passages —
the standard MTEB French reranking corpus).

Two configurations are evaluated:
  - DENSE-only: rank candidates by EmbeddingGemma 300M dot-product
  - DENSE+RERANK: re-rank top candidates with bge-reranker-v2-m3

Metrics: nDCG@10, MRR@10, MAP. Numbers go into
docs/audits/miracl_fr_2026-05-22.md.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from memex.cli.bootstrap import bootstrap
from memex.core.config import get_settings
from memex.models.registry import get_registry


def _dcg(rels: list[float], k: int) -> float:
    """Discounted cumulative gain at k."""
    rels = rels[:k]
    return sum(r / np.log2(i + 2) for i, r in enumerate(rels))


def _ndcg_at_k(predicted_order: list[int], gold_positives: set[int], k: int) -> float:
    """nDCG@k. `predicted_order` is a list of candidate indices ranked
    descending. `gold_positives` is the set of indices considered relevant."""
    actual = [1.0 if c in gold_positives else 0.0 for c in predicted_order[:k]]
    ideal = sorted(actual, reverse=True)
    if sum(ideal) == 0:
        return 0.0
    return _dcg(actual, k) / _dcg(ideal, k)


def _mrr_at_k(predicted_order: list[int], gold_positives: set[int], k: int) -> float:
    """Mean Reciprocal Rank — reciprocal of the rank of the FIRST
    relevant document, or 0 if none in top-k."""
    for rank, idx in enumerate(predicted_order[:k], start=1):
        if idx in gold_positives:
            return 1.0 / rank
    return 0.0


def _ap(predicted_order: list[int], gold_positives: set[int]) -> float:
    """Average Precision — used for MAP."""
    if not gold_positives:
        return 0.0
    relevant_hits = 0
    precision_sum = 0.0
    for rank, idx in enumerate(predicted_order, start=1):
        if idx in gold_positives:
            relevant_hits += 1
            precision_sum += relevant_hits / rank
    return precision_sum / len(gold_positives) if relevant_hits else 0.0


async def main() -> None:
    bootstrap()

    print("Loading MIRACL-FR-Reranking-benchmark (343 queries)...")
    ds = load_dataset("OrdalieTech/MIRACL-FR-Reranking-benchmark")["test"]
    print(f"  loaded {len(ds)} queries")

    registry = get_registry()

    # Stage 1 — embedder-only ranking
    print("\n=== Stage 1: dense-only (EmbeddingGemma 300M) ===")
    dense_ndcg10: list[float] = []
    dense_mrr10: list[float] = []
    dense_map: list[float] = []

    t0 = time.time()
    async with registry.use("embedder") as embedder:
        for i, row in enumerate(ds):
            query: str = row["query"]
            positives: list[str] = row["positive"]
            negatives: list[str] = row["negative"]
            candidates = positives + negatives
            # Gold positives are the first len(positives) indices.
            gold_positive_idx = set(range(len(positives)))

            # Embed query + candidates. EmbeddingGemma uses task-specific
            # prefixes; the registry's helper handles this. We call the
            # SentenceTransformer directly here for batching.
            q_emb = embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
            c_emb = embedder.encode(candidates, convert_to_tensor=True, normalize_embeddings=True)
            # Cosine similarity via dot product on normalized vectors.
            scores = torch.matmul(q_emb, c_emb.T).float().cpu().numpy().flatten()

            # Predicted order: indices sorted by descending score
            order = list(np.argsort(-scores))
            dense_ndcg10.append(_ndcg_at_k(order, gold_positive_idx, 10))
            dense_mrr10.append(_mrr_at_k(order, gold_positive_idx, 10))
            dense_map.append(_ap(order, gold_positive_idx))

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(ds)} processed ({time.time() - t0:.1f}s)")

    dense_results = {
        "ndcg_at_10": float(np.mean(dense_ndcg10)),
        "mrr_at_10": float(np.mean(dense_mrr10)),
        "map": float(np.mean(dense_map)),
        "queries": len(dense_ndcg10),
        "elapsed_s": time.time() - t0,
    }
    print(f"  done in {dense_results['elapsed_s']:.1f}s")
    print(f"  dense nDCG@10: {dense_results['ndcg_at_10']:.4f}")
    print(f"  dense MRR@10:  {dense_results['mrr_at_10']:.4f}")
    print(f"  dense MAP:     {dense_results['map']:.4f}")

    # Stage 2 — dense + cross-encoder rerank
    print("\n=== Stage 2: dense top-10 + cross_encoder rerank ===")
    rerank_ndcg10: list[float] = []
    rerank_mrr10: list[float] = []
    rerank_map: list[float] = []

    t0 = time.time()
    async with registry.use("embedder") as embedder, registry.use("reranker") as reranker:
        for i, row in enumerate(ds):
            query: str = row["query"]
            positives: list[str] = row["positive"]
            negatives: list[str] = row["negative"]
            candidates = positives + negatives
            gold_positive_idx = set(range(len(positives)))

            # Dense retrieval → top-10 (or all if fewer)
            q_emb = embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
            c_emb = embedder.encode(candidates, convert_to_tensor=True, normalize_embeddings=True)
            dense_scores = torch.matmul(q_emb, c_emb.T).float().cpu().numpy().flatten()
            top10 = np.argsort(-dense_scores)[:10].tolist()

            # Rerank: bge-reranker scores (query, passage) pairs
            pairs = [(query, candidates[c]) for c in top10]
            rerank_scores = reranker.predict(pairs)
            # Order the top10 indices by rerank score descending
            order_within_top10 = list(np.argsort(-np.asarray(rerank_scores)))
            reranked_order = [top10[i] for i in order_within_top10]

            # Use reranked order + dense tail for the full ranking
            tail = [c for c in np.argsort(-dense_scores) if c not in reranked_order]
            final_order = reranked_order + list(tail)

            rerank_ndcg10.append(_ndcg_at_k(final_order, gold_positive_idx, 10))
            rerank_mrr10.append(_mrr_at_k(final_order, gold_positive_idx, 10))
            rerank_map.append(_ap(final_order, gold_positive_idx))

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(ds)} processed ({time.time() - t0:.1f}s)")

    rerank_results = {
        "ndcg_at_10": float(np.mean(rerank_ndcg10)),
        "mrr_at_10": float(np.mean(rerank_mrr10)),
        "map": float(np.mean(rerank_map)),
        "queries": len(rerank_ndcg10),
        "elapsed_s": time.time() - t0,
    }
    print(f"  done in {rerank_results['elapsed_s']:.1f}s")
    print(f"  rerank nDCG@10: {rerank_results['ndcg_at_10']:.4f}")
    print(f"  rerank MRR@10:  {rerank_results['mrr_at_10']:.4f}")
    print(f"  rerank MAP:     {rerank_results['map']:.4f}")

    # Emit JSON for downstream report generation
    out = {
        "benchmark": "MIRACL-FR-Reranking-benchmark (OrdalieTech, MTEB)",
        "stack": {
            "embedder": get_settings().models.embedder,
            "reranker": get_settings().models.reranker,
        },
        "n_queries": len(ds),
        "dense_only": dense_results,
        "dense_plus_rerank": rerank_results,
    }

    out_path = Path("docs/audits/miracl_fr_2026-05-22.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n  → wrote {out_path}")
    print("\nSummary:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
