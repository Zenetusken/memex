"""Audio parse route (ADR-0017; spec §3/§6). `transcribe_audio` is faked, so this pins
`parse_document → _parse_audio`: an audio source routes to ASR, writes the timestamped `##
[mm:ss]` body, records `segments` (NOT `pages`) with exact char-spans in the manifest, and an
empty transcript refuses (HARD-gate-safe).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from memex.core.config import MemexSettings, set_settings
from memex.core.errors import ParseConfidenceTooLow
from memex.core.manifest import read_manifest
from memex.parse import pipeline
from memex.parse.asr_backend import ASRSegment
from memex.vault.store import read_document


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemexSettings]:
    (tmp_path / "documents").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    s = MemexSettings()  # type: ignore[call-arg]
    s.models.asr = "tiny"  # configured (transcribe_audio is faked anyway)
    set_settings(s)
    yield s
    set_settings(None)


@asynccontextmanager
async def _no_pause() -> AsyncIterator[None]:
    yield


def _seed_audio(vault: Path, doc_id: str) -> None:
    d = vault / "documents" / doc_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "source.mp3").write_bytes(b"\xff\xfb\x90\x00fake audio bytes")


@pytest.mark.asyncio
async def test_parse_document_routes_audio_to_asr(
    settings: MemexSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc_id = "aud1"
    _seed_audio(tmp_path, doc_id)

    async def fake_transcribe(
        *, source: Path, cache: object = None, refresh: bool = False
    ) -> list[ASRSegment]:
        return [
            ASRSegment(text="the router forwards packets", start_s=0.0, end_s=3.0, language="en"),
            ASRSegment(text="VLANs segment the network", start_s=62.0, end_s=66.0, language="en"),
        ]

    monkeypatch.setattr(pipeline, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(pipeline, "pause_vllm_for_gpu", _no_pause)

    result = await pipeline.parse_document(doc_id)
    assert result.engine == "asr"
    assert result.pages == []

    doc = await read_document(tmp_path, doc_id)
    assert "## [00:00]" in doc.body
    assert "## [01:02]" in doc.body
    assert "the router forwards packets" in doc.body

    m = await read_manifest(tmp_path, doc_id)
    assert m is not None
    assert m.parse is not None
    assert m.parse.pages == []  # audio has no pages — the record lives in `segments`
    assert len(m.parse.segments) == 2
    assert m.parse.segments[1].start_s == 62.0
    assert m.parse.segments[1].language == "en"
    # the char-span addresses the segment's block exactly in the written body
    s0 = m.parse.segments[0]
    assert doc.body[s0.char_start : s0.char_end].startswith("## [00:00]")
    assert "the router forwards packets" in doc.body[s0.char_start : s0.char_end]


@pytest.mark.asyncio
async def test_parse_document_routes_video_to_asr(
    settings: MemexSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An audio-bearing VIDEO container (.mp4, the ADR-0017 "class video" extension) routes to the
    SAME ASR path as audio — `transcribe_audio` decodes its audio track. Pinned with a faked
    transcribe + a real ZOOM-style ftyp head, so no GPU/codec dep."""
    doc_id = "vid1"
    d = tmp_path / "documents" / doc_id
    d.mkdir(parents=True, exist_ok=True)
    # The real CR350 ftyp head (major brand isom; compatible isom/iso2/avc1/mp41).
    (d / "source.mp4").write_bytes(
        b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41" + b"\x00" * 64
    )

    async def fake_transcribe(
        *, source: Path, cache: object = None, refresh: bool = False
    ) -> list[ASRSegment]:
        return [ASRSegment(text="le routeur achemine les paquets", start_s=0.0, end_s=4.0, language="fr")]

    monkeypatch.setattr(pipeline, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(pipeline, "pause_vllm_for_gpu", _no_pause)

    result = await pipeline.parse_document(doc_id)
    assert result.engine == "asr"
    assert result.pages == []
    doc = await read_document(tmp_path, doc_id)
    assert "## [00:00]" in doc.body
    assert "le routeur achemine les paquets" in doc.body


@pytest.mark.asyncio
async def test_empty_transcript_refuses(
    settings: MemexSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc_id = "aud2"
    _seed_audio(tmp_path, doc_id)

    async def fake_empty(
        *, source: Path, cache: object = None, refresh: bool = False
    ) -> list[ASRSegment]:
        return []

    monkeypatch.setattr(pipeline, "transcribe_audio", fake_empty)
    monkeypatch.setattr(pipeline, "pause_vllm_for_gpu", _no_pause)
    with pytest.raises(ParseConfidenceTooLow):  # all-fail → recoverable → refuse
        await pipeline.parse_document(doc_id)
