"""Unit tests for the structured table store (Table-RAG Phase 2).

Covers `index/table_store.py`:
  - `extract_tables`: well-formed GFM tables → StoredTable; same header-sanity
    gate as Phase 1 (mis-bounded / all-value-like skipped); char spans; section;
    content-derived table_id.
  - `TableStore`: open / upsert_document / tables_for_document / delete_document
    round-trip; regenerable (re-upsert replaces); empty-body no-op.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memex.index.table_store import TableStore, extract_tables

# ======================================================================
# extract_tables
# ======================================================================


def test_extract_well_formed_table() -> None:
    body = (
        "## Segments\n\n| Segment | FY24 |\n|---|---|\n| Datacenter | 47525 |\n| Gaming | 10447 |\n"
    )
    tables = extract_tables("doc-1", body)
    assert len(tables) == 1
    t = tables[0]
    assert t.doc_id == "doc-1"
    assert t.header == ["Segment", "FY24"]
    assert t.rows == [["Datacenter", "47525"], ["Gaming", "10447"]]
    assert t.section == "Segments"
    # char span points at the raw GFM table.
    assert body[t.char_start : t.char_end].startswith("| Segment | FY24 |")
    assert "Gaming" in body[t.char_start : t.char_end]
    # table_id is a 10-hex-char content id.
    assert len(t.table_id) == 10
    assert all(c in "0123456789abcdef" for c in t.table_id)


def test_extract_skips_all_value_like_header() -> None:
    """Same gate as Phase 1: an all-value-like header is mis-structured → skip."""
    body = "| 2024 | 2025 | $ |\n|---|---|---|\n| 10 | 20 | 30 |\n"
    assert extract_tables("doc-1", body) == []


def test_extract_skips_misbounded_prose_header() -> None:
    """The 10-K mis-bounding case: a prose-like header cell → skip (no wrong SQL)."""
    body = (
        "## Reportable Segments\n\n"
        "| Fiscal 2026 Reportable Segments | Our two reportable segments are "
        "'Compute & Networking' and 'Graphics': | Compute & Networking | Graphics Total |\n"
        "|---|---|---|---|\n"
        "| Revenue | $193.5B | $22.5B | $215.9B |\n"
    )
    assert extract_tables("doc-1", body) == []


def test_extract_multiple_tables() -> None:
    body = "# A\n\n| X | Y |\n|---|---|\n| 1 | 2 |\n\n# B\n\n| P | Q |\n|---|---|\n| 3 | 4 |\n"
    tables = extract_tables("doc-1", body)
    assert len(tables) == 2
    assert {t.section for t in tables} == {"A", "B"}
    # Distinct ids (different first-data-row text).
    assert tables[0].table_id != tables[1].table_id


def test_extract_no_tables() -> None:
    assert extract_tables("doc-1", "# Title\n\nJust prose.\n") == []


def test_table_id_is_stable_and_doc_scoped() -> None:
    body = "| Segment | FY24 |\n|---|---|\n| Datacenter | 47525 |\n"
    id_a = extract_tables("doc-1", body)[0].table_id
    id_a_again = extract_tables("doc-1", body)[0].table_id
    id_b = extract_tables("doc-2", body)[0].table_id
    assert id_a == id_a_again  # deterministic
    assert id_a != id_b  # doc-scoped (doc_id in the hash)


# ======================================================================
# TableStore round-trip
# ======================================================================


@pytest.mark.asyncio
async def test_store_round_trip(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    body = (
        "## Segments\n\n| Segment | FY24 |\n|---|---|\n| Datacenter | 47525 |\n| Gaming | 10447 |\n"
    )
    tables = extract_tables("doc-1", body)

    store = await TableStore.open(vault)
    try:
        await store.upsert_document("doc-1", tables)
        out = await store.tables_for_document("doc-1")
        assert len(out) == 1
        assert out[0].header == ["Segment", "FY24"]
        assert out[0].rows == [["Datacenter", "47525"], ["Gaming", "10447"]]
        assert out[0].section == "Segments"
        assert out[0].table_id == tables[0].table_id
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_store_regenerable_replace(tmp_path: Path) -> None:
    """A second upsert REPLACES the doc's tables (delete-then-insert)."""
    vault = tmp_path / "vault"
    vault.mkdir()
    store = await TableStore.open(vault)
    try:
        t1 = extract_tables("doc-1", "| A | B |\n|---|---|\n| 1 | 2 |\n")
        await store.upsert_document("doc-1", t1)
        assert len(await store.tables_for_document("doc-1")) == 1
        # Re-index with a different table set.
        t2 = extract_tables(
            "doc-1",
            "| A | B |\n|---|---|\n| 9 | 9 |\n\n| C | D |\n|---|---|\n| 5 | 6 |\n",
        )
        await store.upsert_document("doc-1", t2)
        out = await store.tables_for_document("doc-1")
        assert len(out) == 2  # replaced, not appended
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_store_delete_document(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    store = await TableStore.open(vault)
    try:
        await store.upsert_document(
            "doc-1", extract_tables("doc-1", "| A | B |\n|---|---|\n| 1 | 2 |\n")
        )
        deleted = await store.delete_document("doc-1")
        assert deleted == 1
        assert await store.tables_for_document("doc-1") == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_store_empty_upsert_is_noop(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    store = await TableStore.open(vault)
    try:
        await store.upsert_document("doc-1", [])
        assert await store.tables_for_document("doc-1") == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_store_isolation_between_docs(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    store = await TableStore.open(vault)
    try:
        await store.upsert_document(
            "doc-1", extract_tables("doc-1", "| A | B |\n|---|---|\n| 1 | 2 |\n")
        )
        await store.upsert_document(
            "doc-2", extract_tables("doc-2", "| C | D |\n|---|---|\n| 3 | 4 |\n")
        )
        await store.delete_document("doc-1")
        assert await store.tables_for_document("doc-1") == []
        assert len(await store.tables_for_document("doc-2")) == 1
    finally:
        await store.close()


# ======================================================================
# table_id uniqueness — GPU-acceptance regression 2026-05-24
# ======================================================================


def test_two_tables_sharing_first_row_get_distinct_table_ids() -> None:
    """Two tables whose first DATA row is identical must still get distinct
    table_ids — else the (doc_id, table_id) PRIMARY KEY upsert raises
    IntegrityError. The id is position-qualified (char_start) to guarantee it.
    Regression for the 10-K's repeated header-row tables.
    """
    body = (
        "## A\n\n"
        "| Metric | 2026 |\n| --- | --- |\n| Same Row | 1 |\n| X | 2 |\n\n"
        "## B\n\n"
        "| Metric | 2026 |\n| --- | --- |\n| Same Row | 1 |\n| Y | 9 |\n"
    )
    tables = extract_tables("doc-dup", body)
    assert len(tables) == 2
    ids = [t.table_id for t in tables]
    assert ids[0] != ids[1], "tables sharing a first row must not collide"
    assert len({*ids}) == 2


@pytest.mark.asyncio
async def test_upsert_tables_sharing_first_row_no_integrity_error(tmp_path: Path) -> None:
    """End-to-end: extracting + upserting two same-first-row tables must not
    raise sqlite IntegrityError (the bug GPU acceptance surfaced)."""
    body = (
        "## A\n\n| M | 2026 |\n| --- | --- |\n| R | 1 |\n\n"
        "## B\n\n| M | 2026 |\n| --- | --- |\n| R | 1 |\n"
    )
    store = await TableStore.open(tmp_path)
    try:
        await store.upsert_document("doc-dup", extract_tables("doc-dup", body))
        got = await store.tables_for_document("doc-dup")
        assert len(got) == 2
    finally:
        await store.close()
