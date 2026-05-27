"""Unit tests for `core/scope_sets.py` — the saved-scope-set store.

Pure filesystem I/O against a `tmp_path` vault; no models, no settings. Covers
the CRUD round-trip plus the hardening edge cases: name/doc validation,
case+whitespace-insensitive lookup, upsert semantics, dedup/strip, a corrupt
store, and the fail-open resolve the answer path depends on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memex.core.errors import ScopeSetError, VaultIntegrityError
from memex.core.scope_sets import (
    delete_scope_set,
    get_scope_set,
    list_scope_sets,
    normalize_set_name,
    read_scope_sets,
    resolve_scope_set_doc_ids,
    save_scope_set,
    scope_sets_path,
)

# ----- empty / missing store -----


async def test_missing_store_is_empty(tmp_path: Path) -> None:
    assert (await read_scope_sets(tmp_path)).sets == []
    assert await list_scope_sets(tmp_path) == []
    assert await get_scope_set(tmp_path, "anything") is None
    assert await resolve_scope_set_doc_ids(tmp_path, "anything") == []


# ----- create + round-trip -----


async def test_save_then_read_round_trip(tmp_path: Path) -> None:
    rec = await save_scope_set(tmp_path, "CR350 Networking", ["aaaa1111-a", "bbbb2222-b"])
    assert rec.name == "CR350 Networking"
    assert rec.doc_ids == ["aaaa1111-a", "bbbb2222-b"]

    # Lands at the documented path and reloads byte-faithfully.
    assert scope_sets_path(tmp_path) == tmp_path / ".memex" / "scope_sets.json"
    again = await get_scope_set(tmp_path, "CR350 Networking")
    assert again is not None
    assert again.doc_ids == ["aaaa1111-a", "bbbb2222-b"]
    assert again.created_at == rec.created_at
    assert await resolve_scope_set_doc_ids(tmp_path, "CR350 Networking") == [
        "aaaa1111-a",
        "bbbb2222-b",
    ]


async def test_resolve_returns_a_copy(tmp_path: Path) -> None:
    await save_scope_set(tmp_path, "s", ["d1"])
    resolved = await resolve_scope_set_doc_ids(tmp_path, "s")
    resolved.append("mutation")
    # Mutating the returned list must not corrupt the stored set.
    assert await resolve_scope_set_doc_ids(tmp_path, "s") == ["d1"]


# ----- name normalization + lookup -----


def test_normalize_set_name() -> None:
    assert normalize_set_name("  CR350   Networking ") == "cr350 networking"
    assert normalize_set_name("cr350 networking") == "cr350 networking"
    # A space is meaningful; case is not.
    assert normalize_set_name("CR350") != normalize_set_name("CR 350")
    assert normalize_set_name("CR350") == normalize_set_name("cr350")
    assert normalize_set_name("   ") == ""


async def test_lookup_is_case_and_whitespace_insensitive(tmp_path: Path) -> None:
    await save_scope_set(tmp_path, "CR350 Networking", ["d1"])
    for query in ("cr350 networking", "  CR350   Networking  ", "CR350 NETWORKING"):
        found = await get_scope_set(tmp_path, query)
        assert found is not None, query
        assert found.name == "CR350 Networking"  # display case preserved


async def test_display_name_is_trimmed_and_collapsed_case_kept(tmp_path: Path) -> None:
    rec = await save_scope_set(tmp_path, "  My   Set  ", ["d1"])
    assert rec.name == "My Set"


# ----- upsert semantics -----


async def test_upsert_preserves_created_at_bumps_updated_at(tmp_path: Path) -> None:
    first = await save_scope_set(tmp_path, "set", ["d1"])
    # A case/whitespace variant targets the SAME set (normalized key).
    second = await save_scope_set(tmp_path, "  SET ", ["d2", "d3"])
    assert second.created_at == first.created_at  # same set, original birth time
    assert second.updated_at >= first.updated_at
    assert second.doc_ids == ["d2", "d3"]  # docs replaced, not merged
    # Still exactly one set on disk.
    assert len(await list_scope_sets(tmp_path)) == 1


# ----- doc-id hygiene -----


async def test_doc_ids_deduped_and_stripped_order_preserved(tmp_path: Path) -> None:
    rec = await save_scope_set(tmp_path, "s", [" d2 ", "d1", "d2", "  ", "d1", "d3"])
    assert rec.doc_ids == ["d2", "d1", "d3"]


# ----- validation -----


@pytest.mark.parametrize("bad_name", ["", "   ", "\t\n"])
async def test_empty_name_rejected(tmp_path: Path, bad_name: str) -> None:
    with pytest.raises(ScopeSetError):
        await save_scope_set(tmp_path, bad_name, ["d1"])


async def test_over_long_name_rejected(tmp_path: Path) -> None:
    with pytest.raises(ScopeSetError):
        await save_scope_set(tmp_path, "x" * 101, ["d1"])
    # Exactly at the cap is fine.
    rec = await save_scope_set(tmp_path, "x" * 100, ["d1"])
    assert len(rec.name) == 100


@pytest.mark.parametrize("bad_docs", [[], ["", "   "], [" "]])
async def test_no_documents_rejected(tmp_path: Path, bad_docs: list[str]) -> None:
    with pytest.raises(ScopeSetError):
        await save_scope_set(tmp_path, "s", bad_docs)


# ----- delete -----


async def test_delete_removes_only_the_named_set(tmp_path: Path) -> None:
    await save_scope_set(tmp_path, "keep", ["d1"])
    await save_scope_set(tmp_path, "drop", ["d2"])
    assert await delete_scope_set(tmp_path, "DROP") is True  # case-insensitive
    remaining = await list_scope_sets(tmp_path)
    assert [s.name for s in remaining] == ["keep"]
    # Deleting again is a no-op.
    assert await delete_scope_set(tmp_path, "drop") is False
    assert await delete_scope_set(tmp_path, "never-existed") is False


# ----- ordering determinism -----


async def test_list_is_sorted_by_normalized_name(tmp_path: Path) -> None:
    for name in ("Zebra", "alpha", "Mango"):
        await save_scope_set(tmp_path, name, ["d1"])
    names = [s.name for s in await list_scope_sets(tmp_path)]
    assert names == ["alpha", "Mango", "Zebra"]
    # Stable across repeated reads (no insertion-order leakage).
    for _ in range(5):
        assert [s.name for s in await list_scope_sets(tmp_path)] == names


# ----- corrupt store -----


async def test_corrupt_store_raises_at_management_surface(tmp_path: Path) -> None:
    path = scope_sets_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(VaultIntegrityError):
        await read_scope_sets(tmp_path)
    with pytest.raises(VaultIntegrityError):
        await get_scope_set(tmp_path, "s")


async def test_corrupt_store_resolve_fails_open(tmp_path: Path) -> None:
    """The answer path must never raise on a corrupt store — it degrades to the
    full-corpus search (empty scope)."""
    path = scope_sets_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"sets": [{"name": "x"}]}', encoding="utf-8")  # schema-invalid
    assert await resolve_scope_set_doc_ids(tmp_path, "x") == []


async def test_on_disk_shape_is_human_readable_json(tmp_path: Path) -> None:
    await save_scope_set(tmp_path, "s", ["d1", "d2"])
    raw = json.loads(scope_sets_path(tmp_path).read_text(encoding="utf-8"))
    assert raw["sets"][0]["name"] == "s"
    assert raw["sets"][0]["doc_ids"] == ["d1", "d2"]
    assert "created_at" in raw["sets"][0] and "updated_at" in raw["sets"][0]
