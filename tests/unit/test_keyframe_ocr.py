"""K1 unit tests for the keyframe-OCR module (companion-merge accuracy lever, ADR-0018 §13).

Faked-testable: the PyAV decode seam (`_decode_frame`) and the VLM OCR seam (`_ocr_images`)
are monkeypatched, so the orchestration — cache hit/miss, the content-addressed key, the
near-empty + decode-failure omission, the cache round-trip — runs with NO GPU and NO video.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from memex.core.config import MemexSettings, set_settings
from memex.parse import keyframe_ocr as K
from memex.parse.keyframe_ocr import (
    KeyframeDecodeError,
    KeyframeOCRError,
    _avg_hash,
    _cache_key,
    _dedup_plan,
    _hamming_distance,
    ocr_frames_for_chunks,
)
from memex.parse.keyframe_ocr_cache import KeyframeOCRCache


def _fake_video(tmp_path: Path) -> Path:
    """A real on-disk file so `.is_file()` + `_sha256_file` work (the decode is faked)."""
    p = tmp_path / "source.mp4"
    p.write_bytes(b"not-a-real-video-but-hashable")
    return p


def _set_settings(tmp_path: Path, *, dedup: bool = False) -> None:
    # Dedup OFF by default — the legacy orchestration tests fake `_decode_frame` to return floats,
    # which the perceptual hash can't process; the dedup tests below opt in with real images.
    s = MemexSettings(vault_path=tmp_path)
    s.parse.keyframe_dedup_enabled = dedup
    set_settings(s)


def _half_image(*, vertical: bool) -> Image.Image:
    """A 32×32 image split half-black / half-white — a structured average-hash (not a flat fill,
    which hashes to 0). `vertical` flips the split so the hash differs clearly."""
    im = Image.new("RGB", (32, 32), (0, 0, 0))
    for x in range(32):
        for y in range(32):
            on = (x >= 16) if vertical else (y >= 16)
            if on:
                im.putpixel((x, y), (255, 255, 255))
    return im


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
            "k1",
            video_sha256="v",
            timestamp_ms=1000,
            vlm_model="m",
            prompt_sha8="p",
            markdown="hello slide",
        )
        assert await cache.get("k1") == "hello slide"
        # delete_by_video drops every row for the source video.
        await cache.put(
            "k2",
            video_sha256="v",
            timestamp_ms=2000,
            vlm_model="m",
            prompt_sha8="p",
            markdown="another",
        )
        assert await cache.delete_by_video("v") == 2
        assert await cache.get("k1") is None
    finally:
        await cache.close()


@pytest.mark.asyncio
async def test_miss_then_hit_skips_second_ocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
async def test_decode_failure_omits_that_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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


# ----- Arc 1: perceptual-hash dedup (ADR-0018 §13) -----


def test_avg_hash_and_hamming_pure() -> None:
    a = _half_image(vertical=True)
    b = a.copy()  # identical content → identical hash
    c = _half_image(vertical=False)  # a different split → many flipped bits
    assert _hamming_distance(_avg_hash(a), _avg_hash(b)) == 0
    assert _hamming_distance(_avg_hash(a), _avg_hash(c)) > 8  # a real slide change flips many bits
    assert _hamming_distance(_avg_hash(a), _avg_hash(a)) == 0


def test_dedup_plan_reuses_previous_kept_frame() -> None:
    a = _half_image(vertical=True)
    plan = _dedup_plan([a, a.copy(), _half_image(vertical=False), a.copy()], hamming=4)
    # [kept, reuse-0, kept(new slide), kept(differs from frame 2)] — sliding window vs LAST KEPT.
    assert plan[0] == 0
    assert plan[1] == 0  # frame 1 is ~identical to the kept frame 0 → reuse its OCR
    assert plan[2] == 2  # a different slide → kept, OCR'd fresh


@pytest.mark.asyncio
async def test_dedup_skips_ocr_for_identical_consecutive_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_settings(tmp_path, dedup=True)
    video = _fake_video(tmp_path)
    imgs = {
        10.0: _half_image(vertical=True),
        20.0: _half_image(vertical=True),
        30.0: _half_image(vertical=False),
    }
    monkeypatch.setattr(K, "_decode_frame", lambda _v, t: imgs[t])
    ocr_batches: list[int] = []

    async def _fake_ocr(images: list[Image.Image]) -> list[str]:
        ocr_batches.append(len(images))
        return [
            f"slide number {i} transcribed with plenty of characters" for i in range(len(images))
        ]

    monkeypatch.setattr(K, "_ocr_images", _fake_ocr)

    out = await ocr_frames_for_chunks(video, [("a", 10.0), ("b", 20.0), ("c", 30.0)], cache=None)
    assert ocr_batches == [2]  # only the 2 UNIQUE frames were OCR'd (b reused a)
    assert out["a"] == out["b"]  # b reused a's OCR text
    assert out["a"] != out["c"]  # c is a different slide


@pytest.mark.asyncio
async def test_dedup_disabled_ocrs_every_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_settings(tmp_path, dedup=False)
    video = _fake_video(tmp_path)
    imgs = {10.0: _half_image(vertical=True), 20.0: _half_image(vertical=True)}
    monkeypatch.setattr(K, "_decode_frame", lambda _v, t: imgs[t])
    ocr_batches: list[int] = []

    async def _fake_ocr(images: list[Image.Image]) -> list[str]:
        ocr_batches.append(len(images))
        return [
            f"slide content {i} padded well past the twenty char floor" for i in range(len(images))
        ]

    monkeypatch.setattr(K, "_ocr_images", _fake_ocr)

    await ocr_frames_for_chunks(video, [("a", 10.0), ("b", 20.0)], cache=None)
    assert ocr_batches == [2]  # dedup OFF → both identical frames OCR'd separately
