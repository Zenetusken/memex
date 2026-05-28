"""Unit tests for the shared SQLite PRAGMA helper + the graph schema-parse cache.

The DB-performance pass (2026-05-28): every derived-state SQLite store applies the
same tuned PRAGMAs from one helper, and the graph store parses its Cypher DDL ONCE at
import instead of per-open.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from memex.core.sqlite_tuning import apply_sqlite_pragmas


def test_apply_pragmas_sets_wal_and_normal_sync(tmp_path: Path) -> None:
    # WAL can't be set on :memory: (it reports "memory"), so use a real file.
    db = sqlite3.connect(tmp_path / "t.sqlite", isolation_level=None, check_same_thread=False)
    try:
        apply_sqlite_pragmas(db)
        assert db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert db.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
        # negative cache_size = KiB; we asked for 64 MiB.
        assert db.execute("PRAGMA cache_size").fetchone()[0] == -65536
        assert db.execute("PRAGMA mmap_size").fetchone()[0] == 268435456
        assert db.execute("PRAGMA temp_store").fetchone()[0] == 2  # MEMORY
    finally:
        db.close()


def test_wal_persists_across_reconnect(tmp_path: Path) -> None:
    """journal_mode=WAL is stored in the DB file, so a fresh connection inherits it
    even without re-applying — the property that lets per-call opens stay cheap."""
    p = tmp_path / "t.sqlite"
    db = sqlite3.connect(p, isolation_level=None)
    apply_sqlite_pragmas(db)
    db.close()
    db2 = sqlite3.connect(p, isolation_level=None)
    try:
        assert db2.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        db2.close()


def test_graph_schema_statements_parsed_once_and_clean() -> None:
    """The Cypher DDL is pre-parsed at import into a non-empty list of statements,
    each terminated with exactly one ';' and none a stray comment line."""
    from memex.index.graph_store import _SCHEMA_STATEMENTS

    assert _SCHEMA_STATEMENTS, "schema must parse to at least one statement"
    for stmt in _SCHEMA_STATEMENTS:
        assert stmt.endswith(";")
        assert not stmt.endswith(";;")  # the double-semicolon ryugraph rejects
        assert not stmt.lstrip().startswith("//")  # comments stripped
    # The Document node table must be present (the first DDL the rest depends on).
    assert any("Document" in s for s in _SCHEMA_STATEMENTS)
