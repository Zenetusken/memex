"""Unit tests for the ASR per-chunk transcription cache (Increment 3, ADR-0017).

Mirrors `test_vlm_cache.py`: the store is sync sqlite under an async wrapper. Also pins the
pure key helpers — `cfg_sha8` (order-independent decoding-param hash) and `build_asr_cache_key`
(the input-derived per-VAD-chunk key) — which keep a decoding-param change a clean miss.
"""

from __future__ import annotations

from pathlib import Path

from memex.parse.asr_cache import ASRTranscriptionCache, build_asr_cache_key, cfg_sha8


def test_cfg_sha8_is_deterministic_and_order_independent() -> None:
    a = cfg_sha8({"backend": "faster_whisper", "beam_size": 1, "language": "fr"})
    b = cfg_sha8({"language": "fr", "beam_size": 1, "backend": "faster_whisper"})
    assert a == b  # key order does not matter
    assert len(a) == 8
    # A decoding-param change → a different hash → a clean cache miss.
    assert a != cfg_sha8({"backend": "faster_whisper", "beam_size": 5, "language": "fr"})


def test_build_cache_key_format() -> None:
    key = build_asr_cache_key(
        audio_sha256="abc123", chunk_index=4, model="large-v3", cfg="deadbeef"
    )
    assert key == "abc123:4:large-v3:deadbeef"


async def test_put_get_roundtrip(tmp_path: Path) -> None:
    cache = await ASRTranscriptionCache.open(tmp_path)
    key = build_asr_cache_key(audio_sha256="aaa", chunk_index=0, model="m", cfg="cfg00000")
    await cache.put(
        key,
        audio_sha256="aaa",
        chunk_index=0,
        asr_model="m",
        cfg_sha8="cfg00000",
        segments_json='[{"text": "bonjour", "start": 0.0, "end": 1.5}]',
    )
    assert await cache.get(key) == '[{"text": "bonjour", "start": 0.0, "end": 1.5}]'
    await cache.close()


async def test_miss_returns_none(tmp_path: Path) -> None:
    cache = await ASRTranscriptionCache.open(tmp_path)
    assert await cache.get("no:such:key:00000000") is None
    await cache.close()


async def test_insert_or_ignore_first_writer_wins(tmp_path: Path) -> None:
    cache = await ASRTranscriptionCache.open(tmp_path)
    key = build_asr_cache_key(audio_sha256="aaa", chunk_index=1, model="m", cfg="cfg00000")
    common = {"audio_sha256": "aaa", "chunk_index": 1, "asr_model": "m", "cfg_sha8": "cfg00000"}
    await cache.put(key, segments_json="first", **common)
    await cache.put(key, segments_json="second", **common)
    assert await cache.get(key) == "first"
    await cache.close()


async def test_delete_by_audio_scopes_to_one_file(tmp_path: Path) -> None:
    cache = await ASRTranscriptionCache.open(tmp_path)
    k1 = build_asr_cache_key(audio_sha256="aaa", chunk_index=0, model="m", cfg="c")
    k2 = build_asr_cache_key(audio_sha256="bbb", chunk_index=0, model="m", cfg="c")
    await cache.put(
        k1, audio_sha256="aaa", chunk_index=0, asr_model="m", cfg_sha8="c", segments_json="A"
    )
    await cache.put(
        k2, audio_sha256="bbb", chunk_index=0, asr_model="m", cfg_sha8="c", segments_json="B"
    )
    deleted = await cache.delete_by_audio("aaa")
    assert deleted == 1
    assert await cache.get(k1) is None
    assert await cache.get(k2) == "B"  # the other file is untouched
    await cache.close()


async def test_persists_on_disk_across_reopen(tmp_path: Path) -> None:
    cache = await ASRTranscriptionCache.open(tmp_path)
    key = build_asr_cache_key(audio_sha256="aaa", chunk_index=2, model="m", cfg="c")
    await cache.put(
        key, audio_sha256="aaa", chunk_index=2, asr_model="m", cfg_sha8="c", segments_json="kept"
    )
    await cache.close()

    reopened = await ASRTranscriptionCache.open(tmp_path)
    assert await reopened.get(key) == "kept"
    await reopened.close()
