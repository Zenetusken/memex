"""Hybrid retrieval — BM25 (SQLite FTS5) + dense (LanceDB) + RRF fusion,
plus a backend-dispatched reranker (`rerank`, defaulting to bge cross-
encoder; Qwen3-Reranker via the `qwen3` backend flag). See
GUIDELINES.md Part IV "Hybrid retrieval" and `retrieve/rerank.py` for
the backend dispatch.
"""

from memex.retrieve.entity import EntityOverview, entity_overview
from memex.retrieve.fusion import reciprocal_rank_fusion
from memex.retrieve.hybrid import hybrid_search, hybrid_search_in_docs
from memex.retrieve.rerank import cross_encoder_rerank

__all__ = [
    "EntityOverview",
    "cross_encoder_rerank",
    "entity_overview",
    "hybrid_search",
    "hybrid_search_in_docs",
    "reciprocal_rank_fusion",
]
