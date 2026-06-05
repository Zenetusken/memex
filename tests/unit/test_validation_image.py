"""Unit tests for standalone image-file ingest acceptance (ADR-0020).

Validates `_detect_image`: the unambiguous binary magics (PNG/JPEG/TIFF; WebP via `RIFF`+`WEBP`,
distinct from WAV's `RIFF`+`WAVE`) match directly, while the ASCII-startable magics (BMP `BM`,
GIF `GIF8…`) are gated on the head NOT looking like text. HEIC/AVIF `ftyp`-box images stay
rejected. The detected `kind == "image"` flows to the suffix-preserving ingest copy.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from memex.ingest.validation import ValidationResult, _detect_image, validate_file


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def _validate(p: Path) -> ValidationResult:
    return validate_file(p, max_bytes=10_000, allow_macros=False)


# Real-ish magic heads (+ binary padding so `_looks_like_text` is False where it matters).
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 64
_WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 64
_BMP = b"BM\x36\x00\x00\x00" + b"\x00" * 64
_TIFF_LE = b"II*\x00\x08\x00\x00\x00" + b"\x00" * 64
_TIFF_BE = b"MM\x00*\x00\x00\x00\x08" + b"\x00" * 64
_GIF89 = b"GIF89a\x10\x00\x10\x00" + b"\x00" * 64
_GIF87 = b"GIF87a\x10\x00\x10\x00" + b"\x00" * 64


@pytest.mark.parametrize(
    ("head", "mime"),
    [
        (_PNG, "image/png"),
        (_JPEG, "image/jpeg"),
        (_WEBP, "image/webp"),
        (_BMP, "image/bmp"),
        (_TIFF_LE, "image/tiff"),
        (_TIFF_BE, "image/tiff"),
        (_GIF89, "image/gif"),
        (_GIF87, "image/gif"),
    ],
)
def test_detect_image_matches_each_magic(head: bytes, mime: str) -> None:
    assert _detect_image(head) == ("image", mime, False)


@pytest.mark.parametrize(
    ("name", "head", "mime"),
    [
        ("shot.png", _PNG, "image/png"),
        ("scan.jpg", _JPEG, "image/jpeg"),
        ("diagram.jpeg", _JPEG, "image/jpeg"),
        ("export.webp", _WEBP, "image/webp"),
        ("legacy.bmp", _BMP, "image/bmp"),
        ("page.tif", _TIFF_LE, "image/tiff"),
        ("page.tiff", _TIFF_BE, "image/tiff"),
        ("anim.gif", _GIF89, "image/gif"),
    ],
)
def test_validate_file_accepts_images(tmp_path: Path, name: str, head: bytes, mime: str) -> None:
    r = _validate(_write(tmp_path, name, head))
    assert r.accepted
    assert r.kind == "image"
    assert r.mime == mime
    assert not r.has_macros


def test_heic_ftyp_image_still_rejected(tmp_path: Path) -> None:
    # ADR-0020 accepts raster images but NOT the HEIC/AVIF `ftyp` containers (decode dep, deferred);
    # their brands stay excluded from `_VIDEO_FTYP_BRANDS` and unmatched by `_detect_image`.
    data = struct.pack(">I", 24) + b"ftypheic" + b"\x00" * 200
    r = _validate(_write(tmp_path, "photo.heic", data))
    assert not r.accepted
    assert r.kind == "unknown"


def test_wav_riff_still_audio_not_image(tmp_path: Path) -> None:
    # WebP and WAV share the `RIFF`@0 magic; `_detect_audio` claims `RIFF`+`WAVE` first, so a WAV
    # must stay `audio` (the image branch only matches `RIFF`+`WEBP`@8).
    data = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 64
    r = _validate(_write(tmp_path, "clip.wav", data))
    assert r.accepted
    assert r.kind == "audio"


def test_bmp_gif_ascii_magic_text_gated(tmp_path: Path) -> None:
    # The ASCII-startable magics (BMP `BM`, GIF `GIF8`) are gated on NOT looking like text, so prose
    # that happens to start with those letters stays `text`, never a misdetected image.
    assert _detect_image(b"BMW is a car, not a bitmap.\n") is None
    assert _detect_image(b"GIF89a is a format you can describe in plain prose.\n") is None
    r = _validate(_write(tmp_path, "notes.txt", b"BMW and GIF87a are words here.\n"))
    assert r.accepted
    assert r.kind == "text"


def test_truncated_head_does_not_crash() -> None:
    # Short/empty heads must safely yield no match (safe-slice semantics), never raise.
    assert _detect_image(b"") is None
    assert _detect_image(b"\xff") is None
    assert _detect_image(b"RI") is None  # partial RIFF, no offset-8 brand


def test_image_kind_preserves_source_suffix() -> None:
    # `image` is deliberately ABSENT from `_EXTENSION_FOR_KIND`, so the ingest copy falls back to the
    # ORIGINAL suffix — `.png` vs `.jpg` vs `.webp` are preserved (the audio precedent), which the
    # suffix-based parse route keys on.
    from memex.ingest.pipeline import _EXTENSION_FOR_KIND

    assert "image" not in _EXTENSION_FOR_KIND
    assert _EXTENSION_FOR_KIND.get("image", ".jpg") == ".jpg"
    assert _EXTENSION_FOR_KIND.get("image", ".webp") == ".webp"
