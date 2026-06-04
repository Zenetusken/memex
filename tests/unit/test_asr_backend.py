"""Unit tests for the ASR backend (Increment 3, ADR-0017).

Fakes `_run_faster_whisper` (the GPU/dep seam) so the whole pipeline — backend dispatch, the
per-file cache, and the deterministic per-segment normalization — is testable with no model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from memex.core.config import MemexSettings, set_settings
from memex.parse import asr_backend
from memex.parse.asr_backend import ASRSegment, ASRUnavailable, transcribe_audio
from memex.parse.asr_cache import ASRTranscriptionCache

_Backend = Literal["faster_whisper", "vllm", "transformers"]


def _configure(
    *,
    asr: str | None = "tiny",
    backend: _Backend = "faster_whisper",
    normalize: bool = True,
    coalesce: float = 0.0,  # default OFF in tests so the short fixtures aren't merged
) -> None:
    s = MemexSettings()
    s.models.asr = asr
    s.models.asr_backend = backend
    s.parse.asr_normalize = normalize
    s.parse.asr_coalesce_seconds = coalesce
    set_settings(s)


def _audio_file(tmp_path: Path, name: str = "lecture.mp3") -> Path:
    p = tmp_path / name
    p.write_bytes(
        b"\xff\xfb\x90\x00" + b"\x00" * 64
    )  # plausible bytes (only hashed, never decoded here)
    return p


async def test_no_model_configured_raises(tmp_path: Path) -> None:
    _configure(asr=None)
    with pytest.raises(ASRUnavailable):
        await transcribe_audio(source=_audio_file(tmp_path))


async def test_unimplemented_backend_raises(tmp_path: Path) -> None:
    _configure(backend="vllm")
    with pytest.raises(ASRUnavailable):
        await transcribe_audio(source=_audio_file(tmp_path))


async def test_faster_whisper_happy_path_is_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(normalize=True)

    def fake_run(source: Path, **_kw: object) -> list[ASRSegment]:
        return [
            ASRSegment(text="um so the router forwards it", start_s=0.0, end_s=2.0, language="en"),
            ASRSegment(text="the the packet", start_s=2.0, end_s=3.5, language="en"),
        ]

    monkeypatch.setattr(asr_backend, "_run_faster_whisper", fake_run)
    segs = await transcribe_audio(source=_audio_file(tmp_path))
    # "um" stripped (non-lexical); the content-word repetition "the the" preserved.
    assert [s.text for s in segs] == ["so the router forwards it", "the the packet"]
    assert segs[0].start_s == 0.0 and segs[0].language == "en"


async def test_normalization_drops_all_filler_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(normalize=True)

    def fake_run(source: Path, **_kw: object) -> list[ASRSegment]:
        return [
            ASRSegment(text="um uh euh", start_s=0.0, end_s=1.0),
            ASRSegment(text="real content here", start_s=1.0, end_s=2.0),
        ]

    monkeypatch.setattr(asr_backend, "_run_faster_whisper", fake_run)
    segs = await transcribe_audio(source=_audio_file(tmp_path))
    assert [s.text for s in segs] == ["real content here"]  # the all-filler segment is dropped


async def test_normalize_off_returns_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(normalize=False)
    monkeypatch.setattr(
        asr_backend,
        "_run_faster_whisper",
        lambda source, **_kw: [ASRSegment(text="um so", start_s=0.0, end_s=1.0)],
    )
    segs = await transcribe_audio(source=_audio_file(tmp_path))
    assert segs[0].text == "um so"  # raw — the filler is kept


async def test_cache_miss_then_hit_skips_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure()
    calls = 0

    def fake_run(source: Path, **_kw: object) -> list[ASRSegment]:
        nonlocal calls
        calls += 1
        return [ASRSegment(text="hello world", start_s=0.0, end_s=1.0, language="en")]

    monkeypatch.setattr(asr_backend, "_run_faster_whisper", fake_run)
    cache = await ASRTranscriptionCache.open(tmp_path)
    src = _audio_file(tmp_path)
    first = await transcribe_audio(source=src, cache=cache)
    second = await transcribe_audio(source=src, cache=cache)
    assert calls == 1  # the 2nd call replayed the cache (no model run)
    assert [s.text for s in first] == [s.text for s in second]
    await cache.close()


async def test_refresh_busts_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure()
    calls = 0

    def fake_run(source: Path, **_kw: object) -> list[ASRSegment]:
        nonlocal calls
        calls += 1
        return [ASRSegment(text="x", start_s=0.0, end_s=1.0)]

    monkeypatch.setattr(asr_backend, "_run_faster_whisper", fake_run)
    cache = await ASRTranscriptionCache.open(tmp_path)
    src = _audio_file(tmp_path)
    await transcribe_audio(source=src, cache=cache)
    await transcribe_audio(source=src, cache=cache, refresh=True)
    assert calls == 2  # refresh re-ran the model
    await cache.close()


# ----- segment coalescing (ADR-0017; the large-v3-turbo phrase-granularity fix) -----


def test_coalesce_merges_phrase_segments_into_target_blocks() -> None:
    # 5 phrase-level segments at ~1s each → with a 3s target, blocks flush once their span ≥ 3s.
    segs = [
        ASRSegment(text=f"phrase{i}", start_s=float(i), end_s=float(i + 1), language="fr")
        for i in range(5)
    ]
    out = asr_backend._coalesce_segments(segs, target_seconds=3.0)
    # block 1 = phrases 0,1,2 (span 0→3 ≥ 3) ; block 2 = phrases 3,4 (tail flush).
    assert [s.text for s in out] == ["phrase0 phrase1 phrase2", "phrase3 phrase4"]
    assert out[0].start_s == 0.0 and out[0].end_s == 3.0  # first start, last end — faithful span
    assert out[1].start_s == 3.0 and out[1].end_s == 5.0
    assert out[0].language == "fr"
    # FAITHFUL: every original word survives, in order, exactly once.
    assert " ".join(s.text for s in out) == " ".join(s.text for s in segs)


def test_coalesce_disabled_and_single_segment_are_noops() -> None:
    segs = [
        ASRSegment(text="a", start_s=0.0, end_s=1.0),
        ASRSegment(text="b", start_s=1.0, end_s=2.0),
    ]
    assert asr_backend._coalesce_segments(segs, target_seconds=0.0) is segs  # disabled
    one = [ASRSegment(text="only", start_s=0.0, end_s=1.0)]
    assert asr_backend._coalesce_segments(one, target_seconds=30.0) is one  # ≤1 segment


async def test_transcribe_coalesces_after_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # End-to-end through transcribe_audio: normalization strips "um", THEN coalescing merges the
    # surviving short segments into one ~30s block (the whole span is 4s < 30s → one block).
    _configure(normalize=True, coalesce=30.0)

    def fake_run(source: Path, **_kw: object) -> list[ASRSegment]:
        return [
            ASRSegment(text="um the router", start_s=0.0, end_s=2.0, language="fr"),
            ASRSegment(text="forwards packets", start_s=2.0, end_s=4.0, language="fr"),
        ]

    monkeypatch.setattr(asr_backend, "_run_faster_whisper", fake_run)
    segs = await transcribe_audio(source=_audio_file(tmp_path))
    assert [s.text for s in segs] == ["the router forwards packets"]  # normalized + merged
    assert segs[0].start_s == 0.0 and segs[0].end_s == 4.0
