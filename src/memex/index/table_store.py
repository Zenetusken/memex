"""SQLite-backed structured table store (Table-RAG Phase 2).

Persists every WELL-FORMED GFM table extracted from a document body as a
`StoredTable` (header + verbatim row cells + source span), so the answering
agent's text-to-SQL helper (`agents/table_sql.py`) can select/compute over a
document's tables at query time. Per-vault, regenerable derived state
(ADR-0003) — dropped + rebuilt by `memex reindex --force`.

Mirrors `index/fts_store.py` exactly: stdlib `sqlite3` opened in autocommit
mode (`isolation_level=None`, `check_same_thread=False`) via `asyncio.to_thread`,
schema-on-open via `executescript`, an `asyncio.Lock` gating multi-statement
writes, per-doc `upsert_document`/`delete_document`. Path alongside
`search.sqlite` under `{vault}/.memex/tables.sqlite`.

Only tables that pass the SAME header-sanity gate as Phase-1 linearization
(`header_all_value_like` OR `header_has_prose_cell` OR `header_has_lost_columns`
→ skip) enter the store, so the mis-bounded segment table AND the flattened
merged-cell comp table (audit-10 W12) are absent and no wrong SQL can run over
them (coherent with Phase 1).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

import structlog

from memex.core.sqlite_tuning import apply_sqlite_pragmas
from memex.core.table_linearize import (
    GFM_TABLE_RE,
    header_all_value_like,
    header_has_lost_columns,
    header_has_prose_cell,
    is_layout_table,
    nearest_heading_text,
    parse_gfm_table,
)
from memex.core.types import StoredTable

logger = structlog.get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS doc_tables (
    doc_id TEXT, table_id TEXT, section TEXT,
    header_json TEXT,         -- list[str] column names
    rows_json TEXT,           -- list[list[str]] cell values (original text, verbatim)
    char_start INT, char_end INT,   -- source span in the .md (citation back to source)
    n_rows INT, n_cols INT,
    PRIMARY KEY (doc_id, table_id));
CREATE INDEX IF NOT EXISTS doc_tables_doc ON doc_tables(doc_id);
"""


def extract_tables(doc_id: str, body: str) -> list[StoredTable]:
    """Extract every well-formed GFM table from *body* as a `StoredTable`.

    Scans `body` with the Phase-1 `_GFM_TABLE_RE` (so `match.start()/end()`
    give `char_start`/`char_end`), parses each block via `parse_gfm_table`, and
    applies the SAME header-sanity gate as the linearizer (skip an all-value-like
    OR prose-like header). The `table_id` is `sha1(doc_id + char_start +
    first-data-row-text)` truncated to 10 hex chars — content-and-position-derived
    so it's unique per table (two tables sharing a first row don't collide) and a
    re-extraction of an unchanged body keeps each id. Pure-sync (string transforms only).
    """
    tables: list[StoredTable] = []
    for m in GFM_TABLE_RE.finditer(body):
        parsed = parse_gfm_table(m.group(0))
        if parsed is None:
            continue
        header, rows = parsed
        if (
            header_all_value_like(header)
            or header_has_prose_cell(header)
            or header_has_lost_columns(header)
        ):
            # A flattened merged-cell header (audit-10 W12) has interior columns
            # that lost their labels to a Docling flatten — a SQL table built
            # over it would carry empty/duplicate column names AND mis-attribute
            # a cell to the wrong column, so skip it (coherent with the Phase-1
            # linearizer skip; the raw GFM stays in the `.md`).
            continue
        # A layout graphic / infographic / single-column list mis-detected as a
        # GFM table (audit-10 W11) has no 2-D relation to query — skip it so no
        # nonsense table enters tables.sqlite (coherent with the Phase-1
        # linearizer skip + the cleaned `.md` the finalize re-renders to bullets).
        if is_layout_table(header, rows):
            continue
        # An empty header would build `CREATE TABLE "t" ()` downstream
        # (sqlite OperationalError). Not reachable via the GFM parser's
        # well-formed path today, but skip defensively so a malformed table
        # never enters the store.
        if not header:
            continue
        # Position-qualify the id with char_start: two tables that share a
        # first row (common in financial filings — repeated "$ in millions"
        # header rows) must not collide on the (doc_id, table_id) primary
        # key. char_start is unique per table in the body and stable across
        # re-extraction of an unchanged body, so the id stays content-and-
        # position-derived (the store is regenerable derived state anyway).
        #
        # The parts are joined with a NUL (`\x00`) separator that cannot
        # appear in markdown source text, so distinct (start, first_row)
        # pairs can't alias each other (`start=105 + "2024"` vs
        # `start=1052 + "024"` both concatenated to "1052024" without a
        # separator → same hash → UNIQUE-constraint collision mid-batch).
        first_row_text = " ".join(rows[0])
        table_id = hashlib.sha1(  # noqa: S324  # non-crypto content id, mirrors chunk_id
            f"{doc_id}\x00{m.start()}\x00{first_row_text}".encode()
        ).hexdigest()[:10]
        tables.append(
            StoredTable(
                doc_id=doc_id,
                table_id=table_id,
                section=nearest_heading_text(body, m.start()),
                header=header,
                rows=rows,
                char_start=m.start(),
                char_end=m.end(),
            )
        )
    return tables


class TableStore:
    """Sync SQLite under the hood; async wrapper via `asyncio.to_thread`."""

    def __init__(self, db: sqlite3.Connection):
        self._db = db
        # Gate multi-statement writes (DELETE + INSERTs in upsert_document)
        # so two concurrent coroutines can't interleave their statements.
        # Mirrors FTSStore._lock. Reads stay unlocked — SQLite serializes
        # individual SELECTs natively.
        self._lock = asyncio.Lock()

    @classmethod
    async def open(cls, vault_path: Path) -> TableStore:
        """Open (or create) the table database under
        `{vault_path}/.memex/tables.sqlite` and return a ready store. The
        schema is created on first open via `_SCHEMA`.
        """
        path = vault_path / ".memex" / "tables.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        def _connect() -> sqlite3.Connection:
            db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
            apply_sqlite_pragmas(db)  # WAL + cache + mmap (ADR-0003 derived state)
            db.executescript(_SCHEMA)
            return db

        db = await asyncio.to_thread(_connect)
        return cls(db)

    async def upsert_document(self, doc_id: str, tables: list[StoredTable]) -> None:
        """Replace this document's stored tables with *tables*.

        Delete-then-insert under the lock (mirrors FTSStore.upsert) so a
        re-index leaves exactly the current table set — idempotent on the same
        body, and a removed/changed table doesn't linger.
        """

        def _write() -> None:
            self._db.execute("DELETE FROM doc_tables WHERE doc_id = ?", (doc_id,))
            if not tables:
                return
            self._db.executemany(
                "INSERT INTO doc_tables (doc_id, table_id, section, header_json, "
                "rows_json, char_start, char_end, n_rows, n_cols) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        t.doc_id,
                        t.table_id,
                        t.section,
                        json.dumps(t.header),
                        json.dumps(t.rows),
                        t.char_start,
                        t.char_end,
                        len(t.rows),
                        len(t.header),
                    )
                    for t in tables
                ],
            )

        async with self._lock:
            await asyncio.to_thread(_write)
        logger.info("table_store.upsert", doc_id=doc_id, count=len(tables))

    async def delete_document(self, doc_id: str) -> int:
        """Drop every stored table belonging to *doc_id*. Returns rows removed."""

        def _delete() -> int:
            cur = self._db.execute("SELECT count(*) FROM doc_tables WHERE doc_id = ?", (doc_id,))
            count = cur.fetchone()[0]
            self._db.execute("DELETE FROM doc_tables WHERE doc_id = ?", (doc_id,))
            return count

        async with self._lock:
            deleted = await asyncio.to_thread(_delete)
        logger.info("table_store.delete_document", doc_id=doc_id, deleted=deleted)
        return deleted

    async def tables_for_document(self, doc_id: str) -> list[StoredTable]:
        """Return every `StoredTable` stored for *doc_id* (empty when none)."""

        def _read() -> list[StoredTable]:
            rows = self._db.execute(
                "SELECT doc_id, table_id, section, header_json, rows_json, "
                "char_start, char_end FROM doc_tables WHERE doc_id = ?",
                (doc_id,),
            ).fetchall()
            out: list[StoredTable] = []
            for r in rows:
                header: list[str] = json.loads(r[3])
                table_rows: list[list[str]] = json.loads(r[4])
                # Defensive: a header-less row could never build a valid SQL
                # table (`CREATE TABLE "t" ()`), so it can't be queried —
                # skip it rather than hand the SQL helper a degenerate table.
                if not header:
                    continue
                out.append(
                    StoredTable(
                        doc_id=r[0],
                        table_id=r[1],
                        section=r[2],
                        header=header,
                        rows=table_rows,
                        char_start=r[5],
                        char_end=r[6],
                    )
                )
            return out

        return await asyncio.to_thread(_read)

    async def close(self) -> None:
        """Close the underlying SQLite connection. Safe to call repeatedly."""
        await asyncio.to_thread(self._db.close)
