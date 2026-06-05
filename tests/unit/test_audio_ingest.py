"""Unit tests for audio + video ingest acceptance (Increment 2 + the "class video" extension, ADR-0017).

Validates `_detect_audio`: the binary MPEG/AAC frame-sync (a precise 2nd-byte set), the
text-gated ASCII magics (ID3/fLaC/OggS — the last requiring an audio codec, since Ogg also
wraps Theora video), and the WAV/`ftyp` containers (M4A-brand-only) — all kept out of the
generic `_MAGIC` loop. Plus `_detect_video`: audio-bearing VIDEO containers (MP4/MOV/MKV/WebM via
a curated `ftyp` VIDEO brand set or the EBML magic) are ACCEPTED as kind `video` (transcribed
audio-only); HEIC/AVIF IMAGE containers and BOM-text stay rejected, and PDF/text is intact.
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


def test_empty_file_rejected(tmp_path: Path) -> None:
    # A 0-byte upload otherwise slips through as empty "text" → a junk 0-chunk document.
    r = _validate(_write(tmp_path, "zero.pdf", b""))
    assert not r.accepted
    assert r.rejection_reason == "file is empty"


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


def test_ogg_opus_accepted(tmp_path: Path) -> None:
    # Ogg requires an AUDIO codec signature (Opus/Vorbis/FLAC/Speex) in the first page.
    data = b"OggS" + b"\x00" * 23 + b"OpusHead" + b"\x00" * 200
    r = _validate(_write(tmp_path, "clip.ogg", data))
    assert r.accepted
    assert r.kind == "audio"
    assert r.mime == "audio/ogg"


def test_non_audio_binary_rejected(tmp_path: Path) -> None:
    # A real binary that is neither a known doc nor audio nor video nor image nor text → unknown.
    # (PNG/JPEG/etc. are now accepted as `image`, ADR-0020 — use a no-magic binary here.)
    r = _validate(_write(tmp_path, "blob.bin", b"\xde\xad\xbe\xef" + b"\x00" * 200))
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


def test_mp4_video_accepted_as_video(tmp_path: Path) -> None:
    # MP4 video (the real ZOOM `ftypisom…mp41` brand, ADR-0017 "class video") is now ACCEPTED as
    # kind `video` — the route transcribes its AUDIO track. This is the exact head of the user's
    # CR350 recordings (major brand isom, compatible isom/iso2/avc1/mp41).
    data = struct.pack(">I", 32) + b"ftypisom" + b"\x00\x00\x02\x00" + b"isomiso2avc1mp41" + b"\x00" * 200
    r = _validate(_write(tmp_path, "lecture.mp4", data))
    assert r.accepted
    assert r.kind == "video"
    assert r.mime == "video/mp4"


def test_mov_quicktime_accepted_as_video(tmp_path: Path) -> None:
    # QuickTime `.mov` — ftyp major brand `qt  ` (two trailing spaces).
    data = struct.pack(">I", 24) + b"ftypqt  " + b"\x00" * 200
    r = _validate(_write(tmp_path, "screen.mov", data))
    assert r.accepted
    assert r.kind == "video"


def test_webm_ebml_accepted_as_video(tmp_path: Path) -> None:
    # WebM — EBML magic (0x1A45DFA3) + a `webm` DocType in the head → video/webm.
    data = b"\x1aE\xdf\xa3" + b"\x01\x00\x00\x00" + b"\x42\x82\x84webm" + b"\x00" * 200
    r = _validate(_write(tmp_path, "clip.webm", data))
    assert r.accepted
    assert r.kind == "video"
    assert r.mime == "video/webm"


def test_mkv_ebml_accepted_as_video(tmp_path: Path) -> None:
    # Matroska — EBML magic with a `matroska` DocType (no `webm` marker) → video/x-matroska.
    data = b"\x1aE\xdf\xa3" + b"\x01\x00\x00\x00" + b"\x42\x82\x88matroska" + b"\x00" * 200
    r = _validate(_write(tmp_path, "clip.mkv", data))
    assert r.accepted
    assert r.kind == "video"
    assert r.mime == "video/x-matroska"


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
    # CRC-protected MP3 frame sync (0xFFFA) — in the precise sync-byte set (AUDIO-BC-002).
    r = _validate(_write(tmp_path, "crc.mp3", b"\xff\xfa\x90\x00" + b"\x00" * 200))
    assert r.accepted
    assert r.kind == "audio"


def test_flac_prefixed_text_is_not_audio(tmp_path: Path) -> None:
    r = _validate(_write(tmp_path, "notes.txt", b"fLaC is the Free Lossless Audio Codec magic.\n"))
    assert r.accepted
    assert r.kind == "text"


def test_ogg_theora_video_rejected(tmp_path: Path) -> None:
    # Ogg-Theora VIDEO carries no audio codec signature → rejected (shared-container tightening).
    data = b"OggS" + b"\x00" * 23 + b"\x80theora" + b"\x00" * 200
    r = _validate(_write(tmp_path, "clip.ogv", data))
    assert not r.accepted
    assert r.kind == "unknown"


def test_utf16le_bom_text_not_audio(tmp_path: Path) -> None:
    # A UTF-16-LE BOM (0xFF 0xFE) must NOT be misdetected as an MP3 frame (validator AUDIO-BC-003).
    data = b"\xff\xfe" + "hello world".encode("utf-16-le")
    r = _validate(_write(tmp_path, "utf16.txt", data))
    assert r.kind != "audio"
