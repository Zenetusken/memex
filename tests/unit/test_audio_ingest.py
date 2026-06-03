"""Unit tests for audio ingest acceptance (Increment 2, ADR-0017).

Validates that `validate_file` accepts the common audio container formats by their
magic bytes (offset-0 magics via `_MAGIC`, the non-offset-0 WAV/`ftyp` containers via the
dedicated branch), rejects non-audio binaries, and does not regress PDF/text detection.
"""

from __future__ import annotations

import struct
from pathlib import Path

from memex.ingest.validation import ValidationResult, validate_file


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def _validate(p: Path) -> ValidationResult:
    return validate_file(p, max_bytes=10_000, allow_macros=False)


def test_mp3_id3_tag_accepted(tmp_path: Path) -> None:
    r = _validate(_write(tmp_path, "lecture.mp3", b"ID3\x04\x00\x00" + b"\x00" * 200))
    assert r.accepted
    assert r.kind == "audio"
    assert r.mime == "audio/mpeg"


def test_mp3_frame_sync_untagged_accepted(tmp_path: Path) -> None:
    r = _validate(_write(tmp_path, "raw.mp3", b"\xff\xfb\x90\x00" + b"\x00" * 200))
    assert r.accepted
    assert r.kind == "audio"


def test_wav_riff_wave_accepted(tmp_path: Path) -> None:
    # RIFF + 4-byte little-endian size + WAVE — the `WAVE` tag is at byte offset 8.
    data = b"RIFF" + struct.pack("<I", 36) + b"WAVEfmt " + b"\x00" * 200
    r = _validate(_write(tmp_path, "clip.wav", data))
    assert r.accepted
    assert r.kind == "audio"
    assert r.mime == "audio/wav"


def test_m4a_ftyp_box_accepted(tmp_path: Path) -> None:
    # 4-byte big-endian box size + `ftyp` brand — `ftyp` is at byte offset 4.
    data = struct.pack(">I", 24) + b"ftypM4A " + b"\x00" * 200
    r = _validate(_write(tmp_path, "voice.m4a", data))
    assert r.accepted
    assert r.kind == "audio"
    assert r.mime == "audio/mp4"


def test_flac_accepted(tmp_path: Path) -> None:
    r = _validate(_write(tmp_path, "song.flac", b"fLaC\x00\x00" + b"\x00" * 200))
    assert r.accepted
    assert r.kind == "audio"
    assert r.mime == "audio/flac"


def test_ogg_accepted(tmp_path: Path) -> None:
    r = _validate(_write(tmp_path, "clip.ogg", b"OggS\x00\x02" + b"\x00" * 200))
    assert r.accepted
    assert r.kind == "audio"
    assert r.mime == "audio/ogg"


def test_non_audio_binary_rejected(tmp_path: Path) -> None:
    # A real binary that is neither a known doc nor audio nor text (PNG magic) → unknown.
    r = _validate(_write(tmp_path, "image.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 200))
    assert not r.accepted
    assert r.kind == "unknown"


def test_pdf_detection_not_regressed(tmp_path: Path) -> None:
    r = _validate(_write(tmp_path, "doc.pdf", b"%PDF-1.7\n" + b"x" * 200))
    assert r.accepted
    assert r.kind == "pdf"


def test_plain_text_detection_not_regressed(tmp_path: Path) -> None:
    r = _validate(_write(tmp_path, "notes.txt", b"just some plain text, no magic here\n"))
    assert r.accepted
    assert r.kind == "text"


def test_aac_adts_accepted(tmp_path: Path) -> None:
    r = _validate(_write(tmp_path, "clip.aac", b"\xff\xf1\x50\x80" + b"\x00" * 200))
    assert r.accepted
    assert r.kind == "audio"
    assert r.mime == "audio/aac"


def test_m4a_via_compatible_brand_accepted(tmp_path: Path) -> None:
    # Major brand mp42 (generic), but an M4A audio brand in the compatible list (offset 16).
    data = (
        struct.pack(">I", 32)
        + b"ftyp"
        + b"mp42"
        + b"\x00\x00\x00\x00"
        + b"M4A "
        + b"isom"
        + b"\x00" * 200
    )
    r = _validate(_write(tmp_path, "voice.m4a", data))
    assert r.accepted
    assert r.kind == "audio"


def test_heic_image_rejected(tmp_path: Path) -> None:
    # An ISO-BMFF `ftyp` box with an IMAGE brand (heic) must NOT pass as audio (the
    # validation-loosening the brand check closes).
    data = struct.pack(">I", 24) + b"ftypheic" + b"\x00" * 200
    r = _validate(_write(tmp_path, "photo.heic", data))
    assert not r.accepted
    assert r.kind == "unknown"


def test_avif_image_rejected(tmp_path: Path) -> None:
    data = struct.pack(">I", 24) + b"ftypavif" + b"\x00" * 200
    r = _validate(_write(tmp_path, "photo.avif", data))
    assert not r.accepted
    assert r.kind == "unknown"


def test_mp4_video_rejected(tmp_path: Path) -> None:
    # MP4/MOV video (no audio brand) is deliberately rejected at ingest in v1 — the "class
    # video" case is a Phase-2 audio-extraction extension (ADR-0017).
    data = struct.pack(">I", 24) + b"ftypisom" + b"\x00\x00\x00\x00" + b"mp41" + b"\x00" * 200
    r = _validate(_write(tmp_path, "lecture.mp4", data))
    assert not r.accepted
    assert r.kind == "unknown"


def test_id3_prefixed_text_is_not_audio(tmp_path: Path) -> None:
    # A prose/markdown/CSV file whose first bytes are "ID3" must stay TEXT, not be misdetected
    # as audio — the ASCII magics are gated on the head NOT looking like text (validator
    # AUDIO-BC-001). A real MP3 is binary (NUL / invalid UTF-8) and still detects.
    r = _validate(_write(tmp_path, "notes.txt", b"ID3 tags are metadata containers in MP3 files.\n"))
    assert r.accepted
    assert r.kind == "text"


def test_oggs_prefixed_text_is_not_audio(tmp_path: Path) -> None:
    r = _validate(_write(tmp_path, "notes.txt", b"OggS is the Ogg container page magic.\n"))
    assert r.accepted
    assert r.kind == "text"


def test_mp3_crc_protected_sync_accepted(tmp_path: Path) -> None:
    # CRC-protected MP3 frame sync (0xFFFA) — covered by the frame-sync bitmask (AUDIO-BC-002).
    r = _validate(_write(tmp_path, "crc.mp3", b"\xff\xfa\x90\x00" + b"\x00" * 200))
    assert r.accepted
    assert r.kind == "audio"
