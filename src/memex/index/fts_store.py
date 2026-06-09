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

from memex.core.sqlite_tuning import apply_sqlite_pragmas
from memex.core.text import (
    strip_chart_extracted_for_index,
    strip_superseded_gfm_tables,
)
from memex.core.types import Chunk
from memex.index.code_query import build_code_term_match

logger = structlog.get_logger(__name__)


def _strip_for_fts(text: str) -> str:
    """Compose the index-layer BM25-body strips applied at `upsert`.

    Two inverse strips, both keeping exactly ONE representation per channel in
    the FTS body (no double-counting of value tokens):
      - `strip_chart_extracted_for_index`: drop the DERIVED `[chart-extracted]`
        block, keep the source prose;
      - `strip_superseded_gfm_tables`: drop the SOURCE GFM table that a
        `[table-rows]` block supersedes, keep the derived KV rows.
    The stored chunk `.text` (LanceDB + chunks_meta + vault `.md`) is untouched
    — only the FTS5 `text` column gets the stripped body.
    """
    return strip_superseded_gfm_tables(strip_chart_extracted_for_index(text))


_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    document_id UNINDEXED,
    document_title UNINDEXED,
    text,
    -- Multilingual word segmentation via unicode61. `remove_diacritics 2`
    -- maps accented letters to their base form during BOTH indexing
    -- and query (FTS5's default mode 1 only strips a small set;
    -- mode 2 is the modern "strip all diacritics" behaviour). This
    -- makes `café` and `cafe` interchangeable — a common need when
    -- French users type queries without accents. Trade-off: drops
    -- the porter stemmer so English `running` / `runs` / `ran` no
    -- longer stem-match each other; the dense embedder picks up that
    -- slack on semantic match anyway, and porter was actively wrong
    -- on non-English content. Migration: existing vaults need
    -- `memex reindex --force` to drop the old chunks_fts table —
    -- FTS5's IF NOT EXISTS silently ignores tokenize changes on an
    -- already-created table.
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS chunks_meta (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    document_title TEXT NOT NULL,
    page INTEGER,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    heading_path TEXT NOT NULL,
    -- Audio time anchor (ADR-0017): the chunk's transcript time range, two REALs with a
    -- -1.0 sentinel for "none" (the `page` convention). -1 on doc/PDF + legacy rows.
    time_start REAL DEFAULT -1.0,
    time_end REAL DEFAULT -1.0,
    -- The UNSTRIPPED chunk body. The FTS5 `text` column is stripped for BM25
    -- (chart-extracted blocks + superseded GFM tables removed); the non-search
    -- read primitives (`chunks_for_document`, `chunks_by_ids`) reconstruct the
    -- chunk from here so summaries / entity passages keep chart-OCR figures.
    -- NULL on legacy rows (pre-migration) → callers COALESCE to the stripped FTS text.
    full_text TEXT
);

CREATE INDEX IF NOT EXISTS chunks_meta_doc ON chunks_meta(document_id);
"""


class FTSStore:
    """Sync SQLite under the hood; async wrapper via `asyncio.to_thread`."""

    def __init__(self, db: sqlite3.Connection):
        self._db = db
        # N2 (audit 2026-05-20): gate multi-statement writes so a
        # future `transaction()` method gets atomic BEGIN/COMMIT
        # semantics. The connection runs in autocommit mode
        # (`isolation_level=None`), so SQLite's kernel mutex
        # serializes individual statements — but `upsert()` issues
        # four consecutive statements (2 DELETE + 2 INSERT) and
        # `delete_document/_chunks` each issue two. Without a lock,
        # two concurrent coroutines could interleave their statements
        # and leave the FTS table and chunks_meta table out of sync.
        # Reads stay unlocked: SQLite handles concurrent SELECT
        # natively and serializing them would block the daemon's
        # parallel `/ask` workload.
        self._lock = asyncio.Lock()

    @classmethod
    async def open(cls, vault_path: Path) -> FTSStore:
        """Open (or create) the FTS5 database under
        `{vault_path}/.memex/search.sqlite` and return a ready-to-use
        store. The schema is created on first open via `_SCHEMA`.
        """
        path = vault_path / ".memex" / "search.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        def _connect() -> sqlite3.Connection:
            db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
            apply_sqlite_pragmas(db)  # WAL + cache + mmap (ADR-0003 derived state)
            db.executescript(_SCHEMA)
            # Migrate a chunks_meta created before `full_text` existed: CREATE IF NOT
            # EXISTS won't add the column to a pre-existing table, so ALTER it. Guarded
            # (a fresh table already has the column from _SCHEMA → "duplicate column"
            # which we swallow; any other error re-raises). Legacy rows stay NULL until
            # `reindex --force` repopulates them — the read primitives COALESCE meanwhile.
            try:
                db.execute("ALTER TABLE chunks_meta ADD COLUMN full_text TEXT")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
            for col in ("time_start", "time_end"):  # ADR-0017 — same guarded ALTER as full_text
                try:
                    db.execute(f"ALTER TABLE chunks_meta ADD COLUMN {col} REAL DEFAULT -1.0")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
            return db

        db = await asyncio.to_thread(_connect)
        return cls(db)

    async def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or replace chunks. Atomic across the multi-statement
        write via `self._lock` (Filler N2). Idempotent on the same
        `chunk_id` — re-upserting overwrites the prior row.
        """
        if not chunks:
            return

        # Dedupe by chunk_id. The chunker emits chunk_ids as
        # `sha1(text)[:10]`, which collides when two slides have
        # identical content (e.g. a deck where many pages are just
        # `<!-- image -->` placeholders). chunks_meta has chunk_id as
        # PRIMARY KEY, so an executemany INSERT with duplicates raises
        # `IntegrityError: UNIQUE constraint failed`. First occurrence
        # wins — they're identical anyway.
        seen: set[str] = set()
        deduped: list[Chunk] = []
        for c in chunks:
            if c.chunk_id in seen:
                continue
            seen.add(c.chunk_id)
            deduped.append(c)
        duplicates = len(chunks) - len(deduped)

        def _write() -> None:
            ids = [c.chunk_id for c in deduped]
            placeholders = ",".join("?" for _ in ids)
            self._db.execute(
                f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})",
                ids,
            )
            self._db.execute(
                f"DELETE FROM chunks_meta WHERE chunk_id IN ({placeholders})",
                ids,
            )
            # P3.3 v3 follow-up: strip `[chart-extracted]` blocks
            # from the BM25 body so dense numerical tokens in
            # extracted chart tables don't inflate term frequency for
            # unrelated queries. The chunk's `.text` field (and the
            # row in chunks_meta) keeps the full content — only the
            # FTS5 `text` column gets the stripped version. When the
            # chunk is retrieved, the agent reads the FULL text and
            # sees the chart-extracted block in its context.
            self._db.executemany(
                "INSERT INTO chunks_fts (chunk_id, document_id, document_title, text) "
                "VALUES (?, ?, ?, ?)",
                [
                    (
                        c.chunk_id,
                        c.document_id,
                        c.document_title,
                        _strip_for_fts(c.text),
                    )
                    for c in deduped
                ],
            )
            self._db.executemany(
                "INSERT INTO chunks_meta (chunk_id, document_id, document_title, "
                "page, char_start, char_end, heading_path, time_start, time_end, full_text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        c.chunk_id,
                        c.document_id,
                        c.document_title,
                        c.page,
                        c.char_start,
                        c.char_end,
                        " > ".join(c.heading_path),
                        c.time_range[0] if c.time_range is not None else -1.0,
                        c.time_range[1] if c.time_range is not None else -1.0,
                        c.text,  # UNSTRIPPED — non-search reads reconstruct from here
                    )
                    for c in deduped
                ],
            )

        async with self._lock:
            await asyncio.to_thread(_write)
        logger.info("fts.upsert", count=len(deduped), deduped=duplicates)

    async def update_document_title(self, doc_id: str, title: str) -> int:
        """Rewrite `document_title` for every chunk of `doc_id` — a
        metadata-only update, no re-chunk or re-embed.

        `search` reads the title from `chunks_fts` (not `chunks_meta`),
        so both copies are updated to stay consistent. Returns the
        number of metadata rows touched. Used by the retitle path
        (`index.pipeline.retitle_document`), which keeps the frontmatter
        title (source of truth) and the denormalized copies in sync
        without the cost — or the delete-before-write risk — of a full
        force-reindex.
        """

        def _update() -> int:
            self._db.execute(
                "UPDATE chunks_fts SET document_title = ? WHERE document_id = ?",
                (title, doc_id),
            )
            cur = self._db.execute(
                "UPDATE chunks_meta SET document_title = ? WHERE document_id = ?",
                (title, doc_id),
            )
            return cur.rowcount

        async with self._lock:
            updated = await asyncio.to_thread(_update)
        logger.info("fts.update_title", doc_id=doc_id, rows=updated)
        return updated

    async def delete_document(self, doc_id: str) -> int:
        """Drop every chunk belonging to `doc_id` from both the FTS and
        metadata tables. Returns the count of rows removed.
        """

        def _delete() -> int:
            cur = self._db.execute(
                "SELECT count(*) FROM chunks_meta WHERE document_id = ?",
                (doc_id,),
            )
            count = cur.fetchone()[0]
            self._db.execute("DELETE FROM chunks_fts WHERE document_id = ?", (doc_id,))
            self._db.execute("DELETE FROM chunks_meta WHERE document_id = ?", (doc_id,))
            return count

        async with self._lock:
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

        async with self._lock:
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

    async def chunks_for_document(self, doc_id: str) -> list[Chunk]:
        """Every chunk of a document, in READING ORDER (char_start ascending).

        The whole-document primitive for summarization (ADR-0008): unlike
        `search`, there is NO query — it returns the FULL ordered set so a
        map-reduce summarizer can walk the document section by section
        (grouping by `heading_path`). Joins `chunks_fts` with `chunks_meta`
        (offsets + heading_path), like `search` but unfiltered. Text comes from
        `chunks_meta.full_text` (UNSTRIPPED) so a chart-extracted figure survives into
        the summary — NOT the BM25-stripped FTS `text` (legacy NULL → COALESCE fallback).
        """

        def _read() -> list[Chunk]:
            # Drive the lookup off chunks_meta (which has the `chunks_meta_doc`
            # index on document_id) and ORDER BY its char_start — NOT off the FTS
            # `document_id` column, which is UNINDEXED, so filtering there forces a
            # full virtual-table scan + a temp-B-tree sort. Same rows, same order;
            # an index seek instead of an O(all-chunks-in-vault) scan (the
            # summarizer's per-doc load path). Text comes from chunks_meta.full_text
            # (UNSTRIPPED) so chart-OCR figures survive; COALESCE to the FTS text for
            # legacy rows written before the full_text column existed.
            rows = self._db.execute(
                """
                SELECT
                  m.chunk_id, m.document_id, m.document_title, COALESCE(m.full_text, f.text),
                  m.page, m.char_start, m.char_end, m.heading_path, m.time_start, m.time_end
                FROM chunks_meta m
                JOIN chunks_fts f ON f.chunk_id = m.chunk_id
                WHERE m.document_id = ?
                ORDER BY m.char_start
                """,
                (doc_id,),
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
                    time_range=(r[8], r[9]) if r[8] >= 0 else None,
                )
                for r in rows
            ]

        return await asyncio.to_thread(_read)

    async def chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        """Fetch specific chunks by id, in the INPUT order (missing ids skipped). The
        targeted by-id primitive — e.g. the entity-attested passages (the MENTIONS edge's
        representative chunk_id, where the NER actually found the entity) without a
        full per-document scan or a text search."""
        if not chunk_ids:
            return []

        def _read() -> list[Chunk]:
            placeholders = ",".join("?" for _ in chunk_ids)
            # Text from chunks_meta.full_text (UNSTRIPPED) so an entity's attested
            # passage keeps its chart-OCR figure; COALESCE to the FTS text for legacy rows.
            rows = self._db.execute(
                f"""
                SELECT
                  m.chunk_id, m.document_id, m.document_title, COALESCE(m.full_text, f.text),
                  m.page, m.char_start, m.char_end, m.heading_path, m.time_start, m.time_end
                FROM chunks_meta m
                JOIN chunks_fts f ON f.chunk_id = m.chunk_id
                WHERE m.chunk_id IN ({placeholders})
                """,
                tuple(chunk_ids),
            ).fetchall()
            by_id = {
                r[0]: Chunk(
                    chunk_id=r[0],
                    document_id=r[1],
                    document_title=r[2],
                    text=r[3],
                    page=r[4],
                    char_start=r[5],
                    char_end=r[6],
                    heading_path=r[7].split(" > ") if r[7] else [],
                    time_range=(r[8], r[9]) if r[8] >= 0 else None,
                )
                for r in rows
            }
            return [by_id[c] for c in chunk_ids if c in by_id]  # input order; skip missing

        return await asyncio.to_thread(_read)

    async def search(self, query: str, *, k: int, term_query: bool = False) -> list[Chunk]:
        """BM25 search over chunks_fts; returns top `k` joined with chunks_meta.

        Default (`term_query=False`): the whole query is wrapped as one FTS5 literal phrase,
        which matches nothing for a natural-language question — so BM25 contributes 0 to RRF and
        hybrid retrieval is effectively dense-only. This is intentional and BENIGN for PROSE:
        a 2026-05-29 arm-separation probe found BM25 recall is a strict SUBSET of the dense
        embedder's on every PROSE eval corpus (union@50 ceiling == dense@50). Don't "fix" the
        phrase wrap without re-measuring (e.g. against a future embedder swap).

        `term_query=True` (the code-only path, Phase-3 Lever A, ADR-0021 / `docs/audits/13`):
        build an OR'd-quoted-WHOLE-identifier MATCH instead (`index/code_query`) — for CODE the
        prose finding INVERTS (a body identifier in a chunk titled by a different symbol is what
        dense misses). Only `retrieve/hybrid.py` passes it, and only when the query NAMES a code
        identifier; every other caller keeps the phrase-wrap.
        """
        match = _fts_match_expr(query, term_query=term_query)
        if match is None:
            return []

        def _read() -> list[Chunk]:
            rows = self._db.execute(
                """
                SELECT
                  f.chunk_id, f.document_id, f.document_title, f.text,
                  m.page, m.char_start, m.char_end, m.heading_path,
                  bm25(chunks_fts) AS score, m.time_start, m.time_end
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
                    time_range=(r[9], r[10]) if r[9] >= 0 else None,
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
        term_query: bool = False,
    ) -> list[Chunk]:
        """BM25 search restricted to a whitelist of document IDs.

        Used by the agent's `expand_graph` node — given a set of
        documents the graph says are related to the original query
        results, fetch the BM25-best chunks from those documents
        specifically. `term_query` is the code-only path (see `search`).
        """
        if not doc_ids or k <= 0:
            return []
        match = _fts_match_expr(query, term_query=term_query)
        if match is None:
            return []
        placeholders = ",".join("?" for _ in doc_ids)

        def _read() -> list[Chunk]:
            rows = self._db.execute(
                f"""
                SELECT
                  f.chunk_id, f.document_id, f.document_title, f.text,
                  m.page, m.char_start, m.char_end, m.heading_path,
                  bm25(chunks_fts) AS score, m.time_start, m.time_end
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
                    time_range=(r[9], r[10]) if r[9] >= 0 else None,
                    score=-float(r[8]),
                )
                for r in rows
            ]

        return await asyncio.to_thread(_read)

    async def close(self) -> None:
        """Close the underlying SQLite connection. Safe to call multiple
        times — the connection is released; subsequent operations on
        this store will fail at the SQLite layer."""
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
    stripped = "".join(ch for ch in query if ord(ch) >= 0x20 and ch not in ("\x7f",))
    return stripped.strip()


def _fts_match_expr(query: str, *, term_query: bool) -> str | None:
    """Build the FTS5 MATCH expression for `query`; `None` if it normalises to nothing.

    Default (`term_query=False`) is the literal phrase-wrap — byte-identical to the historical
    behaviour, the only path any non-/ask caller ever uses. `term_query=True` is the code-only
    path (Phase-3 Lever A, `index/code_query.build_code_term_match`): an OR'd-quoted-WHOLE-
    identifier MATCH that GRACEFULLY FALLS BACK to the phrase-wrap when no identifier survives,
    so a degenerate query can never break.
    """
    cleaned = _normalize_fts_query(query)
    if not cleaned:
        return None
    if term_query:
        term = build_code_term_match(query)
        if term:
            return term
    # Double-quoted, so FTS5 operator chars (" * : ( ^ AND/OR/NEAR) stay literal.
    return '"' + cleaned.replace('"', '""') + '"'
