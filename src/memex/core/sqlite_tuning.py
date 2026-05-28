"""Shared performance PRAGMAs for Memex's derived-state SQLite stores.

Every SQLite store in Memex (`index/fts_store`, `index/table_store`,
`parse/vlm_cache`, `parse/chart_ocr_cache`, `core/bus`) holds REGENERABLE derived
state under `vault/.memex/` (ADR-0003) — never the source of truth, which is the
vault Markdown. That makes the durability/throughput trade-off easy: favour speed +
read concurrency, accept that an OS-level crash could lose the last commit (a lost
cache write just re-runs the model next parse; a lost index write is fixed by
`reindex`). This helper applies the standard tuned PRAGMAs so every store gets the
same posture from one place.

`core/` has no Memex deps, so `index/` and `parse/` may both import this.
"""

from __future__ import annotations

import sqlite3


def apply_sqlite_pragmas(conn: sqlite3.Connection) -> None:
    """Apply Memex's standard derived-state SQLite PRAGMAs to `conn`.

    - **journal_mode=WAL** — readers don't block the writer (the daemon serves
      `/ask` while a background reindex writes; the old rollback journal blocked
      them). WAL is PERSISTENT (stored in the DB file), so it survives reconnects.
    - **synchronous=NORMAL** — durable across an app crash; only an OS/power crash
      can lose the last commit. Correct for regenerable derived state (vs FULL's
      fsync-per-commit tax on the bulk write path).
    - **cache_size=-65536** — a 64 MiB page cache (negative = KiB), so a hot index
      stays resident instead of re-faulting from disk (the default 2 MiB can't hold
      even a small index). Per-connection.
    - **mmap_size=256 MiB** — memory-map the DB so reads skip the syscall layer.
      Per-connection.
    - **temp_store=MEMORY** — sort/group temp B-trees (e.g. an `ORDER BY` spill)
      stay in RAM rather than hitting a disk temp file. Per-connection.
    """
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-65536")
    conn.execute("PRAGMA mmap_size=268435456")
    conn.execute("PRAGMA temp_store=MEMORY")
