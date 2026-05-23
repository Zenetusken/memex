"""LanceDB-backed vector store for chunk embeddings.

Schema lives here as a `LanceModel`; the embedding dimension is pinned
to EmbeddingGemma's native 768 (IMPLEMENTATION-PLAN.md §5.2 — a
one-way door, do not change without a corpus rebuild).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from lancedb.pydantic import LanceModel, Vector

from memex.core.types import Chunk

if TYPE_CHECKING:
    from lancedb import AsyncConnection

logger = structlog.get_logger(__name__)

EMBEDDING_DIM = 768
_TABLE = "chunks"


def _sql_quote(value: str) -> str:
    """Single-quote a value for inclusion in LanceDB `where` expressions.

    LanceDB's `where` accepts SQL-ish predicates but its async API does
    not expose parameter binding for the `where` clause as of v0.x.
    doc_ids and chunk_ids are content-derived and shouldn't contain
    quotes, but external sync, future schema changes, or sideloaded
    filenames could surface a quote — escape defensively so a stray `'`
    becomes `''` rather than terminating the literal early.
    """
    return "'" + value.replace("'", "''") + "'"


class _ChunkRow(LanceModel):
    """LanceDB on-disk schema for the chunks table."""

    chunk_id: str
    document_id: str
    document_title: str
    text: str
    page: int = -1
    char_start: int = 0
    char_end: int = 0
    heading_path: str = ""  # joined " > " — LanceDB doesn't love list[str] in v0.x
    embedding: Vector(EMBEDDING_DIM)  # type: ignore[valid-type]


def _row_from_chunk(chunk: Chunk, embedding: list[float]) -> _ChunkRow:
    return _ChunkRow(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document_title=chunk.document_title,
        text=chunk.text,
        page=chunk.page if chunk.page is not None else -1,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        heading_path=" > ".join(chunk.heading_path),
        embedding=embedding,
    )


def _chunk_from_row(row: _ChunkRow, *, score: float = 0.0) -> Chunk:
    return Chunk(
        chunk_id=row.chunk_id,
        document_id=row.document_id,
        document_title=row.document_title,
        text=row.text,
        page=row.page if row.page >= 0 else None,
        char_start=row.char_start,
        char_end=row.char_end,
        heading_path=row.heading_path.split(" > ") if row.heading_path else [],
        score=score,
    )


class VectorStore:
    """Async wrapper over a LanceDB connection.

    Use `await VectorStore.open(vault_path)` to construct.
    """

    def __init__(self, db: AsyncConnection):
        self._db = db

    @classmethod
    async def open(cls, vault_path: Path) -> VectorStore:
        """Open (or create) the LanceDB instance under
        `{vault_path}/.memex/embeddings.lance` and return a ready-to-use
        store. The `chunks` table is created on first open via the
        `_ChunkRow` schema."""
        import lancedb

        path = vault_path / ".memex" / "embeddings.lance"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        db = await lancedb.connect_async(str(path))
        # Idempotent create. LanceDB 0.30 deprecated `table_names()`
        # in favour of `list_tables()` — but the return shape differs:
        # the older method returned `list[str]`; the newer returns a
        # `ListTablesResponse` (a pydantic-like model with `tables`
        # field). We pull the strings out of `.tables` so the
        # membership test works against table names. Verified by
        # `lancedb.connect_async(...).list_tables()` returning
        # `ListTablesResponse(tables=['chunks'], page_token=None)`.
        response = await db.list_tables()
        names = (
            response.tables if hasattr(response, "tables") else list(response)
        )
        if _TABLE not in names:
            await db.create_table(_TABLE, schema=_ChunkRow)
        return cls(db)

    async def upsert(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        """Insert chunks + their parallel dense vectors. Duplicate
        `chunk_id` rows are deleted first, then the new rows are
        added (LanceDB's supported idempotency pattern). Raises
        `ValueError` if `chunks` and `embeddings` differ in length."""
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must be the same length"
            )
        # Dedupe by chunk_id — same rationale as FTSStore.upsert (see
        # that docstring). Pair the *first* occurrence with its
        # embedding so the post-rerank citation lookup still resolves.
        seen: set[str] = set()
        deduped_chunks: list[Chunk] = []
        deduped_embeddings: list[list[float]] = []
        for c, e in zip(chunks, embeddings, strict=True):
            if c.chunk_id in seen:
                continue
            seen.add(c.chunk_id)
            deduped_chunks.append(c)
            deduped_embeddings.append(e)
        duplicates = len(chunks) - len(deduped_chunks)

        rows = [
            _row_from_chunk(c, e)
            for c, e in zip(deduped_chunks, deduped_embeddings, strict=True)
        ]
        table = await self._db.open_table(_TABLE)
        # LanceDB upsert: delete by chunk_id then add. Deleting by primary key
        # is the supported idempotency pattern.
        ids = ",".join(_sql_quote(r.chunk_id) for r in rows)
        await table.delete(f"chunk_id IN ({ids})")
        await table.add(rows)
        logger.info("vector.upsert", count=len(rows), deduped=duplicates)

    async def delete_document(self, doc_id: str) -> int:
        """Drop every chunk belonging to `doc_id`. Returns the count of
        rows removed (LanceDB's delete doesn't surface a count, so we
        count first then delete)."""
        table = await self._db.open_table(_TABLE)
        # No row count returned by LanceDB delete; we do a count before.
        where = f"document_id = {_sql_quote(doc_id)}"
        before = await table.count_rows(where)
        await table.delete(where)
        logger.info("vector.delete_document", doc_id=doc_id, deleted=before)
        return before

    async def delete_chunks(self, chunk_ids: list[str]) -> int:
        """Selective delete by chunk_id. Pairs with `FTSStore.delete_chunks`
        in the incremental re-indexing path; together they remove the
        rows for chunks that disappeared between re-indexings without
        touching the rows that stayed.
        """
        if not chunk_ids:
            return 0
        table = await self._db.open_table(_TABLE)
        ids_sql = ",".join(_sql_quote(cid) for cid in chunk_ids)
        where = f"chunk_id IN ({ids_sql})"
        before = await table.count_rows(where)
        await table.delete(where)
        logger.info("vector.delete_chunks", count=len(chunk_ids), deleted=before)
        return before

    async def search(
        self, query_embedding: list[float], *, k: int
    ) -> list[Chunk]:
        """Dense L2 search; returns top `k` chunks ordered by ascending
        distance. The returned `Chunk.score` is a synthetic
        rank-descending float so the downstream RRF fusion (which only
        needs ordering) treats higher = better."""
        table = await self._db.open_table(_TABLE)
        # LanceDB 0.30+ split the surface: `table.search(...)` is async
        # and returns an `AsyncVectorQuery`; the builders (`.limit`,
        # `.where`) are sync and chain; the terminal `.to_pydantic`
        # is async again. `_distance` is LanceDB's distance column;
        # score = rank-descending so larger is better.
        query = await table.search(query_embedding)
        results = await query.limit(k).to_pydantic(_ChunkRow)
        # to_pydantic drops `_distance`; we lose the per-row score but the
        # downstream RRF only needs rank order. Return rank as a descending score.
        return [
            _chunk_from_row(r, score=float(k - i) / max(k, 1))
            for i, r in enumerate(results)
        ]

    async def search_in_docs(
        self,
        query_embedding: list[float],
        *,
        doc_ids: list[str],
        k: int,
    ) -> list[Chunk]:
        """Dense search restricted to a whitelist of document IDs.

        Uses LanceDB's `where` clause to filter at the storage layer
        before computing distances on the survivors. Used by the
        agent's `expand_graph` node.
        """
        if not doc_ids or k <= 0:
            return []
        table = await self._db.open_table(_TABLE)
        # Escape via `_sql_quote` defensively even though doc_ids are
        # content-derived hex+slug. Cheap insurance against a future
        # change that would surface quotes (external sync, NFS mount,
        # filename with apostrophe, etc.).
        ids_sql = ",".join(_sql_quote(d) for d in doc_ids)
        where = f"document_id IN ({ids_sql})"
        query = await table.search(query_embedding)
        results = await query.where(where).limit(k).to_pydantic(_ChunkRow)
        return [
            _chunk_from_row(r, score=float(k - i) / max(k, 1))
            for i, r in enumerate(results)
        ]

    async def close(self) -> None:
        """No-op today — LanceDB's `AsyncConnection` releases via GC.
        Kept as an async surface so future driver versions can wire
        an explicit close in without breaking callers."""
        # AsyncConnection has no explicit close in current LanceDB; left as
        # a hook for future driver versions / tests.
        await asyncio.sleep(0)
