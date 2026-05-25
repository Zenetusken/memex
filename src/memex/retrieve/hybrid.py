"""Hybrid search: BM25 (SQLite FTS5) + dense (LanceDB) fused with RRF."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from memex.core.config import get_settings
from memex.core.types import Chunk
from memex.index.fts_store import FTSStore
from memex.index.vector_store import VectorStore
from memex.models.registry import get_registry
from memex.retrieve.fusion import reciprocal_rank_fusion

logger = structlog.get_logger(__name__)


async def _embed_query(query: str) -> list[float]:
    """Embed `query` with the registry's embedder."""
    registry = get_registry()
    async with registry.use("embedder") as embedder:

        def _encode() -> Any:
            return embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]

        embedding = await asyncio.to_thread(_encode)
    return [float(x) for x in embedding]


async def hybrid_search(query: str, k: int = 50) -> list[Chunk]:
    """Run BM25 + dense in parallel; fuse with reciprocal rank fusion."""
    settings = get_settings()
    log = logger.bind(query_len=len(query), k=k)
    log.info("hybrid.start")

    vstore = await VectorStore.open(settings.vault_path)
    fstore = await FTSStore.open(settings.vault_path)
    try:
        embedding = await _embed_query(query)
        dense_task = asyncio.create_task(vstore.search(embedding, k=k))
        bm25_task = asyncio.create_task(fstore.search(query, k=k))
        dense, bm25 = await asyncio.gather(dense_task, bm25_task)
    finally:
        await fstore.close()
        await vstore.close()

    fused = reciprocal_rank_fusion([dense, bm25])[:k]
    log.info("hybrid.done", dense=len(dense), bm25=len(bm25), fused=len(fused))
    return fused


async def hybrid_search_in_docs(
    query: str,
    doc_ids: list[str],
    *,
    k: int,
) -> list[Chunk]:
    """Same RRF pipeline as `hybrid_search`, restricted to a doc-id whitelist.

    Used by the answering agent's `expand_graph` node to pull
    additional chunks from documents the graph says are related to
    the initial retrieval. Returns the top-`k` fused chunks across
    the whitelisted documents.
    """
    if not doc_ids or k <= 0:
        return []
    settings = get_settings()
    log = logger.bind(query_len=len(query), doc_ids=len(doc_ids), k=k)
    log.info("hybrid_in_docs.start")

    vstore = await VectorStore.open(settings.vault_path)
    fstore = await FTSStore.open(settings.vault_path)
    try:
        embedding = await _embed_query(query)
        dense_task = asyncio.create_task(vstore.search_in_docs(embedding, doc_ids=doc_ids, k=k))
        bm25_task = asyncio.create_task(fstore.search_in_docs(query, doc_ids=doc_ids, k=k))
        dense, bm25 = await asyncio.gather(dense_task, bm25_task)
    finally:
        await fstore.close()
        await vstore.close()

    fused = reciprocal_rank_fusion([dense, bm25])[:k]
    log.info(
        "hybrid_in_docs.done",
        dense=len(dense),
        bm25=len(bm25),
        fused=len(fused),
    )
    return fused
