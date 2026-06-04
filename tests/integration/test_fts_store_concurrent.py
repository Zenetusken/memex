"""Filler N2 — FTSStore concurrent-write smoke test.

The audit memo (2026-05-20) flagged that `FTSStore` lacked an explicit
`asyncio.Lock` to gate its multi-statement writes. The connection runs
in SQLite autocommit mode, so each statement is its own implicit
transaction — but `upsert()` issues four consecutive statements
(2 DELETE + 2 INSERT) and `delete_document` / `delete_chunks` each
issue two. SQLite's kernel mutex serializes individual statements but
does NOT group them into an atomic unit.

The lock was added in this commit. These tests verify:
- 10 concurrent `upsert()` calls produce a consistent final state
  (FTS table and chunks_meta table have matching row counts; nothing
  is dropped, nothing is duplicated)
- `search()` reads are NOT blocked by writes (concurrent search +
  upsert via asyncio.gather must complete without deadlock)

The tests use the real FTSStore (sync sqlite3 under `asyncio.to_thread`,
real SQLite on disk in tmp_path), not a fake — the point is to
exercise the actual concurrency path the daemon uses in production.
Pattern mirrors `tests/integration/test_vector_store_concurrent.py`
shipped for Filler N1.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from pathlib import Path

import pytest

from memex.core.types import Chunk
from memex.index.fts_store import FTSStore


def _chunk(chunk_id: str, document_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title=f"Doc {document_id}",
        text=text,
        char_start=0,
        char_end=len(text),
    )


@pytest.mark.asyncio
async def test_concurrent_upsert_burst_is_consistent(tmp_path: Path) -> None:
    """10 concurrent `upsert()` calls with disjoint chunk_id sets must
    produce a final state where the FTS and metadata tables agree on
    every row. Without the lock, the interleaved DELETE/INSERT
    statements could leave the two tables out of sync.
    """
    vault_path = tmp_path / "vault"
    vault_path.mkdir(parents=True)

    store = await FTSStore.open(vault_path)

    # 10 batches of 5 chunks each, all disjoint chunk_ids. Total 50
    # chunks should end up in both tables after the burst.
    burst_size = 10
    chunks_per_batch = 5
    batches: list[list[Chunk]] = []
    for batch_idx in range(burst_size):
        batch = [
            _chunk(
                f"chunk-{batch_idx:02d}-{i:02d}",
                f"doc-{batch_idx:02d}",
                f"batch {batch_idx} chunk {i} text body",
            )
            for i in range(chunks_per_batch)
        ]
        batches.append(batch)

    # Fire all upserts concurrently. asyncio.gather raises if any
    # individual task raises, so an exception here would indicate
    # a race condition tripping an IntegrityError or similar.
    await asyncio.gather(*(store.upsert(b) for b in batches))

    # Verify both tables have the expected row count AND match each
    # other. The atomicity invariant: |chunks_fts| == |chunks_meta|.
    def _counts() -> tuple[int, int]:
        fts_count = store._db.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
        meta_count = store._db.execute("SELECT count(*) FROM chunks_meta").fetchone()[0]
        return fts_count, meta_count

    fts_count, meta_count = await asyncio.to_thread(_counts)
    expected_total = burst_size * chunks_per_batch
    assert fts_count == expected_total, (
        f"chunks_fts has {fts_count} rows; expected {expected_total} "
        "— concurrent upserts may have dropped writes"
    )
    assert meta_count == expected_total, (
        f"chunks_meta has {meta_count} rows; expected {expected_total}"
    )
    assert fts_count == meta_count, (
        f"FTS/metadata table mismatch after concurrent upserts: "
        f"fts={fts_count}, meta={meta_count} — atomicity invariant broken"
    )

    # Post-burst sanity: a search returns chunks from the inserted set.
    results = await store.search("batch", k=10)
    assert len(results) == 10
    for r in results:
        assert r.chunk_id.startswith("chunk-")

    await store.close()


@pytest.mark.asyncio
async def test_sequential_reads_after_concurrent_writes_are_consistent(
    tmp_path: Path,
) -> None:
    """After a burst of concurrent writes, subsequent sequential
    `search()` reads must return self-consistent results — every row
    returned has a matching entry in BOTH the FTS and metadata tables
    (the JOIN in `search()` would fail to return rows otherwise).

    This is the read-side complement of the write-burst test: the lock
    must leave the post-burst state in a queryable shape.

    Note: This test does NOT verify concurrent reads against writes.
    With sync `sqlite3` + `check_same_thread=False`, interleaving reads
    and writes on the same connection from different threads is not
    safe at the C API level (sqlite3 can raise "bad parameter or other
    API misuse" non-deterministically). N2's lock guards multi-statement
    writes among themselves; concurrent-read-while-writing safety would
    require a connection pool or WAL mode + per-thread connections —
    out of N2's scope.
    """
    vault_path = tmp_path / "vault"
    vault_path.mkdir(parents=True)

    store = await FTSStore.open(vault_path)

    # Concurrent write burst.
    write_batches = [
        [_chunk(f"w{i}-{j}", f"doc-w{i}", f"writeburst {i}-{j} payload") for j in range(3)]
        for i in range(5)
    ]
    await asyncio.gather(*(store.upsert(b) for b in write_batches))

    # Sequential reads — the FTS JOIN returns only rows present in
    # BOTH tables, so consistency mismatch would show up as missing
    # rows here.
    results = await store.search("writeburst", k=20)
    assert len(results) == 15, (
        f"expected 15 chunks visible to search after concurrent burst, "
        f"got {len(results)} — FTS/metadata mismatch on the JOIN?"
    )

    await store.close()


@pytest.mark.asyncio
async def test_concurrent_upsert_and_delete_interleaved(tmp_path: Path) -> None:
    """upsert + delete on the same FTSStore must serialize cleanly.
    Without the lock, an interleaved DELETE-from-upsert and DELETE-from-
    delete_document could touch the same rows in inconsistent orders
    and leave dangling FTS-without-meta or meta-without-FTS rows.
    """
    vault_path = tmp_path / "vault"
    vault_path.mkdir(parents=True)

    store = await FTSStore.open(vault_path)

    # Seed five documents, each with five chunks.
    for doc_idx in range(5):
        chunks = [
            _chunk(f"d{doc_idx}-c{i}", f"doc-{doc_idx}", f"doc {doc_idx} chunk {i}")
            for i in range(5)
        ]
        await store.upsert(chunks)

    # Concurrent: re-upsert doc-0 + doc-1 + doc-2, delete doc-3,
    # delete_chunks for two chunks of doc-4. All on the same store.
    re_upserts = [
        [
            _chunk(
                f"d{doc_idx}-c{i}",
                f"doc-{doc_idx}",
                f"updated doc {doc_idx} chunk {i}",
            )
            for i in range(5)
        ]
        for doc_idx in (0, 1, 2)
    ]
    tasks: list[Awaitable[object]] = [store.upsert(b) for b in re_upserts]
    tasks.append(store.delete_document("doc-3"))
    tasks.append(store.delete_chunks(["d4-c0", "d4-c1"]))

    await asyncio.gather(*tasks, return_exceptions=False)

    # Expected end state:
    # - doc-0/1/2: 5 chunks each, updated text -> 15 total
    # - doc-3: removed entirely -> 0
    # - doc-4: 5 - 2 deleted = 3 chunks
    # Total: 15 + 3 = 18
    def _counts() -> tuple[int, int]:
        fts_count = store._db.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
        meta_count = store._db.execute("SELECT count(*) FROM chunks_meta").fetchone()[0]
        return fts_count, meta_count

    fts_count, meta_count = await asyncio.to_thread(_counts)
    assert fts_count == 18, f"fts_count={fts_count}, expected 18"
    assert meta_count == 18, f"meta_count={meta_count}, expected 18"
    assert fts_count == meta_count

    # Verify the re-upsert took effect — search for "updated" should
    # find chunks from doc-0/1/2.
    updated = await store.search("updated", k=20)
    assert len(updated) == 15
    for r in updated:
        assert r.document_id in {"doc-0", "doc-1", "doc-2"}

    await store.close()


@pytest.mark.asyncio
async def test_chunks_by_ids_returns_input_order_skips_missing(tmp_path: Path) -> None:
    """The targeted by-id fetch (entity-attested passages): returns the requested chunks in
    INPUT order, and silently skips ids not present (a stale/un-indexed attested chunk_id)."""
    vault_path = tmp_path / "vault"
    vault_path.mkdir(parents=True)
    store = await FTSStore.open(vault_path)
    await store.upsert(
        [
            _chunk("d1#a", "d1", "alpha body"),
            _chunk("d1#b", "d1", "bravo body"),
            _chunk("d2#c", "d2", "charlie body"),
        ]
    )
    # Requested out of insertion order + one missing id → input order, missing dropped.
    out = await store.chunks_by_ids(["d2#c", "d1#a", "ghost#x"])
    assert [c.chunk_id for c in out] == ["d2#c", "d1#a"]
    assert out[0].text == "charlie body"
    assert await store.chunks_by_ids([]) == []


@pytest.mark.asyncio
async def test_time_range_round_trips_across_all_read_paths(tmp_path: Path) -> None:
    """ADR-0017: a chunk's audio `time_range` persists through the FTS store and is
    reconstructed by EVERY read path (search / search_in_docs / chunks_for_document /
    chunks_by_ids). A non-audio chunk (no time_range) round-trips back to None — so the
    -1.0 sentinel is never mistaken for a real 0.0s anchor."""
    vault_path = tmp_path / "vault"
    vault_path.mkdir(parents=True)
    store = await FTSStore.open(vault_path)

    timed = Chunk(
        chunk_id="aud#0",
        document_id="aud",
        document_title="Lecture",
        text="the router forwards packets across the link",
        char_start=0,
        char_end=43,
        time_range=(62.0, 66.0),
    )
    untimed = _chunk("pdf#0", "pdf", "the router forwards packets across the link")  # no time_range
    await store.upsert([timed, untimed])

    # 1) search — the BM25 arm reconstructs time_range.
    hits = {c.chunk_id: c for c in await store.search("router", k=10)}
    assert hits["aud#0"].time_range == (62.0, 66.0)
    assert hits["pdf#0"].time_range is None  # -1.0 sentinel → None, not (-1.0, -1.0)

    # 2) search_in_docs — scoped BM25 reconstructs it too.
    scoped = await store.search_in_docs("router", doc_ids=["aud"], k=10)
    assert scoped and scoped[0].time_range == (62.0, 66.0)

    # 3) chunks_for_document — the summarizer's full-doc read.
    by_doc = {c.chunk_id: c for c in await store.chunks_for_document("aud")}
    assert by_doc["aud#0"].time_range == (62.0, 66.0)

    # 4) chunks_by_ids — the entity-attested by-id read.
    by_id = await store.chunks_by_ids(["aud#0", "pdf#0"])
    assert by_id[0].time_range == (62.0, 66.0)
    assert by_id[1].time_range is None

    await store.close()
