"""K2 integration test for the keyframe-OCR alignment (companion-merge, ADR-0018 §13).

`compute_alignment(keyframe_texts=...)` reads both docs from a real tmp FTSStore, embeds via a FAKED
`_embed` seam (no GPU), and proves the keyframe signal is PRIMARY: a chunk whose video-frame OCR text
matches a DIFFERENT slide than its transcript text aligns to the FRAME's slide; a chunk whose frame
text matches no slide (below the floor — a demo / off-slide moment) FALLS BACK to its transcript-text
slide. The transcript-only path (`keyframe_texts=None`) stays byte-identical to B2.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from memex.core.config import MemexSettings, set_settings
from memex.core.types import Chunk
from memex.index import companion
from memex.index.companion import compute_alignment
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
        Chunk(chunk_id="deck#p3", document_id="deck", document_title="Deck",
              text="DNS overview and resolution", char_start=39, char_end=66, page=3),
    ]
    await fstore.upsert(transcript + deck)
    await fstore.close()


def _fake_embed_by_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    # 3D keyword space: firewall/acl → [1,0,0]; vlan → [0,1,0]; dns → [0,0,1]; unmatched → [0,0,0]
    # (cosine 0 with every slide → below the keyframe floor → fallback). Applied to ALL sides
    # (transcript query-side, deck doc-side, keyframe doc-side) so they share the space.
    def _vec(s: str) -> list[float]:
        low = s.lower()
        if "firewall" in low or "acl" in low:
            return [1.0, 0.0, 0.0]
        if "vlan" in low:
            return [0.0, 1.0, 0.0]
        if "dns" in low:
            return [0.0, 0.0, 1.0]
        return [0.0, 0.0, 0.0]

    async def fake_embed(inputs: list[str], *, prompt_name: str | None) -> list[list[float]]:
        return [_vec(s) for s in inputs]

    monkeypatch.setattr(companion, "_embed", fake_embed)


@pytest.mark.asyncio
async def test_keyframe_overrides_transcript_and_falls_back(
    settings: MemexSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _seed(tmp_path)
    _fake_embed_by_keyword(monkeypatch)

    # lec#a: transcript text → slide 1 (firewall), but its FRAME shows the DNS slide → keyframe wins.
    # lec#b: frame OCR matches NO slide (a live demo) → below floor → transcript-text fallback (vlan).
    keyframe_texts = {
        "lec#a": "DNS resolution diagram with a query and response",
        "lec#b": "a terminal window running tcpdump, no slide visible",
    }
    a = await compute_alignment("lec", "deck", keyframe_texts=keyframe_texts)
    by_id = {b.transcript_chunk_id: b for b in a.blocks}

    # KEYFRAME-PRIMARY: lec#a aligns to the DNS slide (3), NOT the transcript-text slide (1).
    assert by_id["lec#a"].deck_page == 3 and by_id["lec#a"].deck_chunk_id == "deck#p3"
    assert by_id["lec#a"].score == pytest.approx(1.0)
    # FALLBACK: lec#b's frame matched nothing → transcript-text slide 2 (vlan).
    assert by_id["lec#b"].deck_page == 2 and by_id["lec#b"].deck_chunk_id == "deck#p2"
    assert a.null_count == 0
    # Provenance: the recipe records the video-frame signal produced this alignment.
    assert a.embedding_recipe_version == "v1-gemma-prompts+keyframe"


@pytest.mark.asyncio
async def test_keyframe_none_is_transcript_only(
    settings: MemexSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # keyframe_texts=None → byte-identical to the B2 transcript-only path (slide 1 / slide 2, no suffix).
    await _seed(tmp_path)
    _fake_embed_by_keyword(monkeypatch)
    a = await compute_alignment("lec", "deck", keyframe_texts=None)
    by_id = {b.transcript_chunk_id: b for b in a.blocks}
    assert by_id["lec#a"].deck_page == 1 and by_id["lec#b"].deck_page == 2
    assert a.embedding_recipe_version == "v1-gemma-prompts"  # no +keyframe


@pytest.mark.asyncio
async def test_keyframe_all_below_floor_is_transcript_only(
    settings: MemexSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every frame matches nothing (all demos) → empty keyframe_signal → transcript-only result + NO
    # +keyframe suffix (the alignment was effectively transcript-text).
    await _seed(tmp_path)
    _fake_embed_by_keyword(monkeypatch)
    a = await compute_alignment(
        "lec", "deck", keyframe_texts={"lec#a": "demo only", "lec#b": "desktop only"}
    )
    by_id = {b.transcript_chunk_id: b for b in a.blocks}
    assert by_id["lec#a"].deck_page == 1 and by_id["lec#b"].deck_page == 2  # transcript-text
    assert a.embedding_recipe_version == "v1-gemma-prompts"  # no signal survived → no suffix
