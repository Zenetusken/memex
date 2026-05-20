"""SQLite FTS5-backed full-text search for chunks.

Stdlib `sqlite3` ships with FTS5 enabled on every modern build. BM25
is the default rank. We keep the FTS table and a parallel metadata
table so we can return rich Chunk objects, not just text + rowid.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import structlog

from memex.core.types import Chunk

logger = structlog.get_logger(__name__)


_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    document_id UNINDEXED,
    document_title UNINDEXED,
    text,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS chunks_meta (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    document_title TEXT NOT NULL,
    page INTEGER,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    heading_path TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_meta_doc ON chunks_meta(document_id);
"""


class FTSStore:
    """Sync SQLite under the hood; async wrapper via `asyncio.to_thread`."""

    def __init__(self, db: sqlite3.Connection):
        self._db = db

    @classmethod
    async def open(cls, vault_path: Path) -> FTSStore:
        path = vault_path / ".memex" / "search.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        def _connect() -> sqlite3.Connection:
            db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
            db.executescript(_SCHEMA)
            return db

        db = await asyncio.to_thread(_connect)
        return cls(db)

    async def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        def _write() -> None:
            ids = [c.chunk_id for c in chunks]
            placeholders = ",".join("?" for _ in ids)
            self._db.execute(
                f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})",
                ids,
            )
            self._db.execute(
                f"DELETE FROM chunks_meta WHERE chunk_id IN ({placeholders})",
                ids,
            )
            self._db.executemany(
                "INSERT INTO chunks_fts (chunk_id, document_id, document_title, text) "
                "VALUES (?, ?, ?, ?)",
                [
                    (c.chunk_id, c.document_id, c.document_title, c.text)
                    for c in chunks
                ],
            )
            self._db.executemany(
                "INSERT INTO chunks_meta (chunk_id, document_id, document_title, "
                "page, char_start, char_end, heading_path) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        c.chunk_id,
                        c.document_id,
                        c.document_title,
                        c.page,
                        c.char_start,
                        c.char_end,
                        " > ".join(c.heading_path),
                    )
                    for c in chunks
                ],
            )

        await asyncio.to_thread(_write)
        logger.info("fts.upsert", count=len(chunks))

    async def delete_document(self, doc_id: str) -> int:
        def _delete() -> int:
            cur = self._db.execute(
                "SELECT count(*) FROM chunks_meta WHERE document_id = ?",
                (doc_id,),
            )
            count = cur.fetchone()[0]
            self._db.execute(
                "DELETE FROM chunks_fts WHERE document_id = ?", (doc_id,)
            )
            self._db.execute(
                "DELETE FROM chunks_meta WHERE document_id = ?", (doc_id,)
            )
            return count

        deleted = await asyncio.to_thread(_delete)
        logger.info("fts.delete_document", doc_id=doc_id, deleted=deleted)
        return deleted

    async def delete_chunks(self, chunk_ids: list[str]) -> int:
        """Selective delete by chunk_id. Used by incremental re-indexing
        to remove the chunks that disappeared from a document without
        touching the chunks that stayed.
        """
        if not chunk_ids:
            return 0
        placeholders = ",".join("?" for _ in chunk_ids)

        def _delete() -> int:
            self._db.execute(
                f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
            cur = self._db.execute(
                f"DELETE FROM chunks_meta WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
            return cur.rowcount

        deleted = await asyncio.to_thread(_delete)
        logger.info("fts.delete_chunks", count=len(chunk_ids), deleted=deleted)
        return deleted

    async def chunk_ids_for_document(self, doc_id: str) -> set[str]:
        """Cheap meta-only lookup: what chunk_ids does this doc have today?

        Used by `index.pipeline.index_document` to compute the
        re-indexing diff without re-reading any chunk bodies.
        """
        def _read() -> set[str]:
            rows = self._db.execute(
                "SELECT chunk_id FROM chunks_meta WHERE document_id = ?",
                (doc_id,),
            ).fetchall()
            return {r[0] for r in rows}

        return await asyncio.to_thread(_read)

    async def search(self, query: str, *, k: int) -> list[Chunk]:
        # FTS5 MATCH treats some punctuation as operators; quote the whole
        # query to keep it literal. Users can use FTS5 syntax via raw_query.
        cleaned = _normalize_fts_query(query)
        if not cleaned:
            return []
        match = '"' + cleaned.replace('"', '""') + '"'

        def _read() -> list[Chunk]:
            rows = self._db.execute(
                """
                SELECT
                  f.chunk_id, f.document_id, f.document_title, f.text,
                  m.page, m.char_start, m.char_end, m.heading_path,
                  bm25(chunks_fts) AS score
                FROM chunks_fts f
                JOIN chunks_meta m ON m.chunk_id = f.chunk_id
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
                """,
                (match, k),
            ).fetchall()
            return [
                Chunk(
                    chunk_id=r[0],
                    document_id=r[1],
                    document_title=r[2],
                    text=r[3],
                    page=r[4],
                    char_start=r[5],
                    char_end=r[6],
                    heading_path=r[7].split(" > ") if r[7] else [],
                    # BM25 is "lower is better"; flip + offset so larger is better.
                    score=-float(r[8]),
                )
                for r in rows
            ]

        return await asyncio.to_thread(_read)

    async def search_in_docs(
        self,
        query: str,
        *,
        doc_ids: list[str],
        k: int,
    ) -> list[Chunk]:
        """BM25 search restricted to a whitelist of document IDs.

        Used by the agent's `expand_graph` node — given a set of
        documents the graph says are related to the original query
        results, fetch the BM25-best chunks from those documents
        specifically.
        """
        if not doc_ids or k <= 0:
            return []
        cleaned = _normalize_fts_query(query)
        if not cleaned:
            return []
        match = '"' + cleaned.replace('"', '""') + '"'
        placeholders = ",".join("?" for _ in doc_ids)

        def _read() -> list[Chunk]:
            rows = self._db.execute(
                f"""
                SELECT
                  f.chunk_id, f.document_id, f.document_title, f.text,
                  m.page, m.char_start, m.char_end, m.heading_path,
                  bm25(chunks_fts) AS score
                FROM chunks_fts f
                JOIN chunks_meta m ON m.chunk_id = f.chunk_id
                WHERE chunks_fts MATCH ?
                  AND f.document_id IN ({placeholders})
                ORDER BY bm25(chunks_fts)
                LIMIT ?
                """,
                (match, *doc_ids, k),
            ).fetchall()
            return [
                Chunk(
                    chunk_id=r[0],
                    document_id=r[1],
                    document_title=r[2],
                    text=r[3],
                    page=r[4],
                    char_start=r[5],
                    char_end=r[6],
                    heading_path=r[7].split(" > ") if r[7] else [],
                    score=-float(r[8]),
                )
                for r in rows
            ]

        return await asyncio.to_thread(_read)

    async def close(self) -> None:
        await asyncio.to_thread(self._db.close)


def _normalize_fts_query(query: str) -> str:
    """Strip control characters (including NUL) and collapse whitespace.

    An empty / whitespace-only / NUL-only query produces `""` for the
    MATCH expression, which FTS5 rejects with "malformed MATCH
    expression". Sqlite's text binding can also truncate NUL-bearing
    strings. Returning empty here lets the caller short-circuit.
    """
    if not query:
        return ""
    # Drop ASCII control chars and Unicode line/paragraph separators.
    stripped = "".join(
        ch for ch in query if ord(ch) >= 0x20 and ch not in ("\x7f",)
    )
    return stripped.strip()
