"""Hybrid search: BM25 (SQLite FTS5) + dense (LanceDB) fused with RRF."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any

import structlog

from memex.core.config import get_settings
from memex.core.types import Chunk
from memex.index.code_query import code_term_query_enabled, query_has_code_identifier
from memex.index.embed_prompts import (
    EMBED_QUERY_PROMPT_NAME,
    EMBED_QUERY_PROMPT_TEXT,
    native_prompts_enabled,
)
from memex.index.fts_store import FTSStore
from memex.index.vector_store import VectorStore
from memex.models.registry import get_registry
from memex.retrieve.fusion import reciprocal_rank_fusion

logger = structlog.get_logger(__name__)


async def _embed_query(query: str) -> list[float]:
    """Embed `query` with the registry's embedder."""
    registry = get_registry()
    use_prompt = native_prompts_enabled()
    async with registry.use("embedder") as embedder:
        # EmbeddingGemma's built-in `query` prompt (`task: search result |
        # query: `) when enabled (default ON); else bare (the A/B / revert
        # path). This is the SINGLE query-embed entry point — both
        # hybrid_search and hybrid_search_in_docs route through it.
        #
        # `prompt_name=` raises ValueError/KeyError if the embedder's ST config
        # lacks the registered `query` prompt (a future embedder swap). In that
        # case fall back to MANUALLY prepending the trained query prompt text
        # (`EMBED_QUERY_PROMPT_TEXT`) so the query side still embeds in the
        # model's query distribution — query+doc must share one space.
        def _encode() -> Any:
            if not use_prompt:
                return embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
            try:
                return embedder.encode(
                    [query],
                    prompt_name=EMBED_QUERY_PROMPT_NAME,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )[0]
            except (ValueError, KeyError):
                logger.info("embed_query.prompt_name_fallback", prompt=EMBED_QUERY_PROMPT_NAME)
                return embedder.encode(
                    [EMBED_QUERY_PROMPT_TEXT + query],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )[0]

        embedding = await asyncio.to_thread(_encode)
    return [float(x) for x in embedding]


async def hybrid_search(query: str, k: int = 50) -> list[Chunk]:
    """Run BM25 + dense in parallel; fuse with reciprocal rank fusion."""
    settings = get_settings()
    log = logger.bind(query_len=len(query), k=k)
    log.info("hybrid.start")

    # Each store is registered for close the instant it opens, so a SECOND open
    # failing (e.g. FTSStore.open after VectorStore.open succeeded) can't leak the
    # first; callbacks run LIFO → fstore then vstore (the original order).
    async with AsyncExitStack() as stack:
        vstore = await VectorStore.open(settings.vault_path)
        stack.push_async_callback(vstore.close)
        fstore = await FTSStore.open(settings.vault_path)
        stack.push_async_callback(fstore.close)
        embedding = await _embed_query(query)
        # Code-only term path (Phase-3 Lever A): the BM25 arm builds an OR'd-quoted-WHOLE-
        # identifier MATCH ONLY when the flag is on AND the query NAMES a code identifier;
        # a natural-language prose question keeps the unchanged phrase-wrap. Fail-open
        # (config-read error → False) lives in `code_term_query_enabled`.
        term = code_term_query_enabled() and query_has_code_identifier(query)
        dense_task = asyncio.create_task(vstore.search(embedding, k=k))
        bm25_task = asyncio.create_task(fstore.search(query, k=k, term_query=term))
        dense, bm25 = await asyncio.gather(dense_task, bm25_task)

    fused = reciprocal_rank_fusion([dense, bm25])[:k]
    log.info("hybrid.done", dense=len(dense), bm25=len(bm25), fused=len(fused), term=term)
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

    async with AsyncExitStack() as stack:
        vstore = await VectorStore.open(settings.vault_path)
        stack.push_async_callback(vstore.close)
        fstore = await FTSStore.open(settings.vault_path)
        stack.push_async_callback(fstore.close)
        embedding = await _embed_query(query)
        term = code_term_query_enabled() and query_has_code_identifier(query)
        dense_task = asyncio.create_task(vstore.search_in_docs(embedding, doc_ids=doc_ids, k=k))
        bm25_task = asyncio.create_task(
            fstore.search_in_docs(query, doc_ids=doc_ids, k=k, term_query=term)
        )
        dense, bm25 = await asyncio.gather(dense_task, bm25_task)

    fused = reciprocal_rank_fusion([dense, bm25])[:k]
    log.info(
        "hybrid_in_docs.done",
        dense=len(dense),
        bm25=len(bm25),
        fused=len(fused),
        term=term,
    )
    return fused
