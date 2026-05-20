"""Hybrid retrieval — BM25 (SQLite FTS5) + dense (LanceDB) + RRF fusion, plus
a cross-encoder reranker (bge-reranker-v2-m3). See GUIDELINES.md Part IV
"Hybrid retrieval".
"""

from memex.retrieve.fusion import reciprocal_rank_fusion
from memex.retrieve.hybrid import hybrid_search, hybrid_search_in_docs
from memex.retrieve.rerank import cross_encoder_rerank

__all__ = [
    "cross_encoder_rerank",
    "hybrid_search",
    "hybrid_search_in_docs",
    "reciprocal_rank_fusion",
]
