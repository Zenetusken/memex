"""Tests for the companion-merge retrieval augmentation node (B4, ADR-0018 §7): the pure
counterpart-id helper + the `augment_companion` /ask node (default-OFF no-op; enabled pulls the
aligned counterpart from a real tmp store; bounded; HARD-gate-safe additive-only)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from memex.agents.answering import (
    AnswerState,
    _companion_counterpart_ids,
    augment_companion,
)
from memex.core.companion_store import upsert_alignment
from memex.core.config import MemexSettings, set_settings
from memex.core.types import AlignmentBlock, Chunk, CompanionAlignment
from memex.index.fts_store import FTSStore


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemexSettings]:
    (tmp_path / "documents").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    s = MemexSettings()  # type: ignore[call-arg]
    set_settings(s)
    yield s
    set_settings(None)


def _align() -> CompanionAlignment:
    return CompanionAlignment(
        transcript_doc="lec", deck_doc="deck", null_count=1,
        blocks=[
            AlignmentBlock(transcript_chunk_id="lec#a", deck_chunk_id="deck#p1", deck_page=1, score=0.6),
            AlignmentBlock(transcript_chunk_id="lec#b", deck_chunk_id=None, deck_page=None, score=0.2),
        ],
    )


def test_counterpart_ids_both_directions() -> None:
    a = _align()
    tchunk = Chunk(chunk_id="lec#a", document_id="lec", document_title="L", text="x")
    assert _companion_counterpart_ids(tchunk, [a]) == ["deck#p1"]  # transcript → its aligned slide chunk
    dchunk = Chunk(chunk_id="deck#p1", document_id="deck", document_title="D", text="y", page=1)
    assert _companion_counterpart_ids(dchunk, [a]) == ["lec#a"]  # deck page 1 → aligned commentary chunk
    other = Chunk(chunk_id="z#1", document_id="z", document_title="Z", text="z")
    assert _companion_counterpart_ids(other, [a]) == []  # in neither doc → nothing


async def _seed_chunks(vault: Path) -> tuple[Chunk, Chunk]:
    tc = Chunk(chunk_id="lec#a", document_id="lec", document_title="Lecture",
               text="the teacher explains firewall ACL rules", char_start=0, char_end=39,
               time_range=(60.0, 90.0))
    dc = Chunk(chunk_id="deck#p1", document_id="deck", document_title="Deck",
               text="Firewalls and ACLs", char_start=0, char_end=18, page=1)
    fstore = await FTSStore.open(vault)
    await fstore.upsert([tc, dc])
    await fstore.close()
    return tc, dc


@pytest.mark.asyncio
async def test_augment_disabled_is_noop(settings: MemexSettings, tmp_path: Path) -> None:
    tc, _ = await _seed_chunks(tmp_path)
    await upsert_alignment(tmp_path, _align())
    state = AnswerState(query="firewall", reranked=[tc], companion_augment_enabled=False)
    update = await augment_companion(state)
    assert "reranked" not in update  # default-OFF → never touches the candidate set
    assert update["nodes_traversed"] == 1


@pytest.mark.asyncio
async def test_augment_pulls_aligned_counterpart(settings: MemexSettings, tmp_path: Path) -> None:
    tc, _ = await _seed_chunks(tmp_path)
    await upsert_alignment(tmp_path, _align())
    state = AnswerState(
        query="firewall", reranked=[tc], companion_augment_enabled=True, companion_augment_max=3
    )
    update = await augment_companion(state)
    ids = [c.chunk_id for c in update["reranked"]]
    assert ids == ["lec#a", "deck#p1"]  # the aligned slide chunk APPENDED (additive, original first)
    # the appended counterpart is the REAL indexed chunk (own id + source) — per-chunk-pure.
    deck_chunk = next(c for c in update["reranked"] if c.chunk_id == "deck#p1")
    assert deck_chunk.document_id == "deck" and deck_chunk.text == "Firewalls and ACLs"


@pytest.mark.asyncio
async def test_augment_no_alignment_is_noop(settings: MemexSettings, tmp_path: Path) -> None:
    tc, _ = await _seed_chunks(tmp_path)  # chunks indexed but NO alignment stored
    state = AnswerState(query="firewall", reranked=[tc], companion_augment_enabled=True)
    update = await augment_companion(state)
    assert "reranked" not in update  # no pair aligned → additive no-op


@pytest.mark.asyncio
async def test_augment_bounded_by_max(settings: MemexSettings, tmp_path: Path) -> None:
    # A deck chunk (slide 1) aligned to by 5 transcript chunks → augment caps at companion_augment_max.
    fstore = await FTSStore.open(tmp_path)
    dc = Chunk(chunk_id="deck#p1", document_id="deck", document_title="Deck", text="slide one",
               char_start=0, char_end=9, page=1)
    tx = [Chunk(chunk_id=f"lec#{i}", document_id="lec", document_title="Lecture", text=f"t{i}",
                char_start=i, char_end=i + 1, time_range=(float(i), float(i + 1))) for i in range(5)]
    await fstore.upsert([dc, *tx])
    await fstore.close()
    await upsert_alignment(tmp_path, CompanionAlignment(
        transcript_doc="lec", deck_doc="deck", null_count=0,
        blocks=[AlignmentBlock(transcript_chunk_id=f"lec#{i}", deck_chunk_id="deck#p1", deck_page=1,
                               score=0.6) for i in range(5)],
    ))
    state = AnswerState(query="x", reranked=[dc], companion_augment_enabled=True, companion_augment_max=2)
    update = await augment_companion(state)
    added = [c.chunk_id for c in update["reranked"] if c.chunk_id != "deck#p1"]
    assert len(added) == 2  # capped at companion_augment_max
