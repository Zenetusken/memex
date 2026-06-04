"""Integration test for the companion-merge compute orchestrator (B2, ADR-0018): `compute_alignment`
reads both docs' indexed chunks from a real tmp FTSStore, embeds via a FAKED `_embed` seam (no GPU),
and aligns — exercising the full read→embed→align path + the validation errors."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from memex.core.config import MemexSettings, set_settings
from memex.core.types import Chunk
from memex.index import companion
from memex.index.companion import CompanionMergeError, compute_alignment
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


async def _seed(vault: Path) -> None:
    fstore = await FTSStore.open(vault)
    transcript = [
        Chunk(chunk_id="lec#a", document_id="lec", document_title="Lecture",
              text="the firewall filters traffic with ACL rules", char_start=0, char_end=43,
              time_range=(0.0, 30.0)),
        Chunk(chunk_id="lec#b", document_id="lec", document_title="Lecture",
              text="VLANs segment the broadcast domain", char_start=45, char_end=79,
              time_range=(30.0, 60.0)),
    ]
    deck = [
        Chunk(chunk_id="deck#p1", document_id="deck", document_title="Deck",
              text="Firewalls and ACLs", char_start=0, char_end=18, page=1),
        Chunk(chunk_id="deck#p2", document_id="deck", document_title="Deck",
              text="VLAN segmentation", char_start=20, char_end=37, page=2),
    ]
    await fstore.upsert(transcript + deck)
    await fstore.close()


def _fake_embed_by_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    # firewall/ACL → [1,0]; vlan → [0,1] — deterministic, keyword-keyed (ignores prompt_name; the deck
    # side's `document_input` wrapper still contains the keyword).
    async def fake_embed(inputs: list[str], *, prompt_name: str | None) -> list[list[float]]:
        out: list[list[float]] = []
        for s in inputs:
            low = s.lower()
            out.append([1.0, 0.0] if ("firewall" in low or "acl" in low) else [0.0, 1.0])
        return out

    monkeypatch.setattr(companion, "_embed", fake_embed)


@pytest.mark.asyncio
async def test_compute_alignment_matches_pages(
    settings: MemexSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _seed(tmp_path)
    _fake_embed_by_keyword(monkeypatch)
    a = await compute_alignment("lec", "deck")
    assert a.transcript_doc == "lec" and a.deck_doc == "deck"
    by_id = {b.transcript_chunk_id: b for b in a.blocks}
    assert by_id["lec#a"].deck_page == 1 and by_id["lec#a"].deck_chunk_id == "deck#p1"  # firewall → slide 1
    assert by_id["lec#b"].deck_page == 2  # vlan → slide 2
    assert a.null_count == 0
    assert by_id["lec#a"].time_range == (0.0, 30.0)  # the audio anchor is carried onto the block
    assert a.embedding_recipe_version == "v1-gemma-prompts"


@pytest.mark.asyncio
async def test_compute_alignment_null_below_floor(
    settings: MemexSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _seed(tmp_path)

    async def orthogonal(inputs: list[str], *, prompt_name: str | None) -> list[list[float]]:
        # transcript (query-side, prompt_name="query") ⊥ deck (doc-side, prompt_name=None) → cosine 0
        # < floor → all NULL. (Keying on prompt_name is how we make the two sides orthogonal.)
        vec = [0.0, 0.0, 1.0] if prompt_name == "query" else [1.0, 0.0, 0.0]
        return [list(vec) for _ in inputs]

    monkeypatch.setattr(companion, "_embed", orthogonal)
    a = await compute_alignment("lec", "deck")
    assert a.null_count == 2 and all(b.deck_chunk_id is None for b in a.blocks)


@pytest.mark.asyncio
async def test_compute_alignment_validations(
    settings: MemexSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_embed_by_keyword(monkeypatch)
    # Neither doc indexed → a clear error (not a crash).
    with pytest.raises(CompanionMergeError, match="transcript has no indexed"):
        await compute_alignment("missing", "alsomissing")

    # A deck with chunks but NO page attribution → the explicit re-parse guidance.
    fstore = await FTSStore.open(tmp_path)
    await fstore.upsert([
        Chunk(chunk_id="lec#a", document_id="lec", document_title="L", text="x", char_start=0, char_end=1),
        Chunk(chunk_id="deck#1", document_id="deck", document_title="D", text="y", char_start=0, char_end=1),
    ])  # deck#1 has page=None (default)
    await fstore.close()
    with pytest.raises(CompanionMergeError, match="no per-page attribution"):
        await compute_alignment("lec", "deck")
