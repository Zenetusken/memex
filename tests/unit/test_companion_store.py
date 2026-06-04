"""Unit tests for the companion-alignment sidecar store (B2, ADR-0018) — atomic round-trip, upsert-
replaces, by-doc lookup, delete, and the fail-open vs loud-read contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from memex.core.companion_store import (
    alignments_for_doc,
    companion_alignments_path,
    delete_alignment,
    get_alignment,
    read_alignments,
    read_alignments_open,
    upsert_alignment,
)
from memex.core.errors import VaultIntegrityError
from memex.core.types import AlignmentBlock, CompanionAlignment


def _align(transcript: str, deck: str) -> CompanionAlignment:
    return CompanionAlignment(
        transcript_doc=transcript,
        deck_doc=deck,
        embedding_recipe_version="v1-gemma-prompts",
        blocks=[
            AlignmentBlock(transcript_chunk_id=f"{transcript}#a", time_range=(0.0, 30.0),
                           deck_chunk_id=f"{deck}#p1", deck_page=1, score=0.61),
            AlignmentBlock(transcript_chunk_id=f"{transcript}#b", time_range=(30.0, 60.0),
                           deck_chunk_id=None, deck_page=None, score=0.18),
        ],
        null_count=1,
    )


@pytest.mark.asyncio
async def test_round_trip_and_get(tmp_path: Path) -> None:
    (tmp_path / ".memex").mkdir(parents=True)
    await upsert_alignment(tmp_path, _align("lec1", "deck1"))
    coll = await read_alignments(tmp_path)
    assert len(coll.pairs) == 1
    a = await get_alignment(tmp_path, "lec1", "deck1")
    assert a is not None and a.null_count == 1 and a.blocks[0].deck_page == 1
    assert a.blocks[0].time_range == (0.0, 30.0)  # cached anchor survives the JSON round-trip
    assert await get_alignment(tmp_path, "lec1", "other") is None  # wrong deck → no match


@pytest.mark.asyncio
async def test_upsert_replaces_and_for_doc(tmp_path: Path) -> None:
    (tmp_path / ".memex").mkdir(parents=True)
    await upsert_alignment(tmp_path, _align("lec1", "deck1"))
    await upsert_alignment(tmp_path, _align("lec1", "deck1"))  # REPLACES, not appends
    await upsert_alignment(tmp_path, _align("lec2", "deck2"))
    assert len((await read_alignments(tmp_path)).pairs) == 2
    # alignments_for_doc matches EITHER the transcript or the deck side.
    assert len(await alignments_for_doc(tmp_path, "deck1")) == 1
    assert len(await alignments_for_doc(tmp_path, "lec2")) == 1
    assert await alignments_for_doc(tmp_path, "nope") == []


@pytest.mark.asyncio
async def test_delete(tmp_path: Path) -> None:
    (tmp_path / ".memex").mkdir(parents=True)
    await upsert_alignment(tmp_path, _align("lec1", "deck1"))
    assert await delete_alignment(tmp_path, "lec1", "deck1") is True
    assert await delete_alignment(tmp_path, "lec1", "deck1") is False  # already gone
    assert (await read_alignments(tmp_path)).pairs == []


@pytest.mark.asyncio
async def test_missing_and_corrupt_fail_open(tmp_path: Path) -> None:
    (tmp_path / ".memex").mkdir(parents=True)
    # Missing file → empty (both readers).
    assert (await read_alignments(tmp_path)).pairs == []
    assert await read_alignments_open(tmp_path) == []
    # Corrupt file → the management read raises LOUD; the /ask-path read fails OPEN to [].
    companion_alignments_path(tmp_path).write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(VaultIntegrityError):
        await read_alignments(tmp_path)
    assert await read_alignments_open(tmp_path) == []
