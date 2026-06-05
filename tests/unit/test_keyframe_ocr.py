"""K1 unit tests for the keyframe-OCR module (companion-merge accuracy lever, ADR-0018 §13).

Faked-testable: the PyAV decode seam (`_decode_frame`) and the VLM OCR seam (`_ocr_images`)
are monkeypatched, so the orchestration — cache hit/miss, the content-addressed key, the
near-empty + decode-failure omission, the cache round-trip — runs with NO GPU and NO video.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memex.core.config import MemexSettings, set_settings
from memex.parse import keyframe_ocr as K
from memex.parse.keyframe_ocr import (
    KeyframeDecodeError,
    KeyframeOCRError,
    _cache_key,
    ocr_frames_for_chunks,
)
from memex.parse.keyframe_ocr_cache import KeyframeOCRCache


def _fake_video(tmp_path: Path) -> Path:
    """A real on-disk file so `.is_file()` + `_sha256_file` work (the decode is faked)."""
    p = tmp_path / "source.mp4"
    p.write_bytes(b"not-a-real-video-but-hashable")
    return p


def _set_settings(tmp_path: Path) -> None:
    set_settings(MemexSettings(vault_path=tmp_path))


def test_cache_key_is_video_timestamp_model_prompt() -> None:
    # timestamp → integer milliseconds; the key is independent of frame-PNG bytes.
    assert _cache_key("vsha", 12.5, "qwen3-vl", "ab12cd34") == "vsha:12500:m=qwen3-vl:p=ab12cd34"
    assert _cache_key("vsha", 0.0, "m", "p") == "vsha:0:m=m:p=p"


@pytest.mark.asyncio
async def test_cache_round_trip_and_delete(tmp_path: Path) -> None:
    (tmp_path / ".memex").mkdir(parents=True)
    cache = await KeyframeOCRCache.open(tmp_path)
    try:
        assert await cache.get("k1") is None
        await cache.put(
            "k1", video_sha256="v", timestamp_ms=1000, vlm_model="m", prompt_sha8="p", markdown="hello slide"
        )
        assert await cache.get("k1") == "hello slide"
        # delete_by_video drops every row for the source video.
        await cache.put(
            "k2", video_sha256="v", timestamp_ms=2000, vlm_model="m", prompt_sha8="p", markdown="another"
        )
        assert await cache.delete_by_video("v") == 2
        assert await cache.get("k1") is None
    finally:
        await cache.close()


@pytest.mark.asyncio
async def test_miss_then_hit_skips_second_ocr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_settings(tmp_path)
    (tmp_path / ".memex").mkdir(parents=True)
    video = _fake_video(tmp_path)
    cache = await KeyframeOCRCache.open(tmp_path)
    ocr_calls: list[int] = []

    def _fake_decode(_video: Path, time_s: float) -> object:
        return time_s  # the "image" carries the timestamp so the OCR fake can label it

    async def _fake_ocr(images: list[object]) -> list[str]:
        ocr_calls.append(len(images))
        return [f"slide content at t={img} (long enough to pass the floor)" for img in images]

    monkeypatch.setattr(K, "_decode_frame", _fake_decode)
    monkeypatch.setattr(K, "_ocr_images", _fake_ocr)

    frames = [("lec#a", 15.0), ("lec#b", 45.0)]
    try:
        first = await ocr_frames_for_chunks(video, frames, cache=cache)
        assert set(first) == {"lec#a", "lec#b"}
        assert "t=15.0" in first["lec#a"] and "t=45.0" in first["lec#b"]
        assert ocr_calls == [2]  # both were misses → one OCR batch of 2

        # Second run: everything served from cache → NO further OCR.
        second = await ocr_frames_for_chunks(video, frames, cache=cache)
        assert second == first
        assert ocr_calls == [2]  # unchanged — no new OCR call
    finally:
        await cache.close()


@pytest.mark.asyncio
async def test_decode_failure_omits_that_chunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_settings(tmp_path)
    video = _fake_video(tmp_path)

    def _fake_decode(_video: Path, time_s: float) -> object:
        if time_s == 45.0:
            raise KeyframeDecodeError("boom", context={"time_s": time_s})
        return time_s

    async def _fake_ocr(images: list[object]) -> list[str]:
        return [f"slide content t={img} padded to twenty plus chars" for img in images]

    monkeypatch.setattr(K, "_decode_frame", _fake_decode)
    monkeypatch.setattr(K, "_ocr_images", _fake_ocr)

    out = await ocr_frames_for_chunks(video, [("lec#a", 15.0), ("lec#b", 45.0)], cache=None)
    assert set(out) == {"lec#a"}  # the failed decode (#b) is omitted → transcript fallback


@pytest.mark.asyncio
async def test_near_empty_ocr_is_omitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_settings(tmp_path)
    video = _fake_video(tmp_path)

    monkeypatch.setattr(K, "_decode_frame", lambda _v, t: t)

    async def _fake_ocr(images: list[object]) -> list[str]:
        # First frame returns a real slide; second is a no-slide / punted frame (< _MIN_OCR_CHARS).
        return ["a real slide with plenty of transcribed text", "x"]

    monkeypatch.setattr(K, "_ocr_images", _fake_ocr)

    out = await ocr_frames_for_chunks(video, [("lec#a", 15.0), ("lec#b", 45.0)], cache=None)
    assert set(out) == {"lec#a"}  # the near-empty OCR (#b) is dropped


@pytest.mark.asyncio
async def test_empty_frames_and_missing_video(tmp_path: Path) -> None:
    _set_settings(tmp_path)
    # No frames → no work, no error.
    assert await ocr_frames_for_chunks(tmp_path / "source.mp4", [], cache=None) == {}
    # A non-existent video with real frames → loud KeyframeOCRError (the CLI gate precludes this).
    with pytest.raises(KeyframeOCRError):
        await ocr_frames_for_chunks(tmp_path / "missing.mp4", [("lec#a", 1.0)], cache=None)
