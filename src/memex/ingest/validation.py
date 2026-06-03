"""File-format and content validation — see GUIDELINES.md Part VI security.

Magic-number checks are non-optional. We never trust the filename
extension. Office documents are inspected for macros and rejected
unless `IngestSettings.allow_macros=True`. PDFs are verified for the
`%PDF` header. Markdown and plain text are accepted but length-checked.

Audio sources (MP3/WAV/M4A/FLAC/Ogg/Opus/AAC, ADR-0017) are accepted by
magic via `_detect_audio`. The ASCII magics (ID3/fLaC/OggS) are gated on
the head NOT looking like text (they collide with prose ABOUT those
formats); the binary MPEG/AAC frame sync is matched by a bitmask; and the
ISO-BMFF `ftyp` box — SHARED by M4A audio, MP4/MOV video, and HEIC/AVIF
images — requires the `M4A ` brand, so video and image containers stay
rejected (audio-only ingest; the "class video" case is a Phase-2
audio-extraction extension).

Validation is intentionally tight in the formats it recognises; new
formats arrive with an ADR explaining what they look like and what
risks they bring.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

DetectedKind = Literal[
    "pdf",
    "docx",
    "pptx",
    "xlsx",
    "html",
    "markdown",
    "text",
    "audio",
    "unknown",
]


class ValidationResult(BaseModel):
    """Output of `validate_file` — the gate that decides whether an
    ingest request gets accepted. Carries the detected kind/mime
    (used by downstream parse routing), the size in bytes, and any
    rejection diagnostics."""

    accepted: bool
    kind: DetectedKind
    mime: str
    size_bytes: int
    rejection_reason: str | None = None
    has_macros: bool = False


_MAGIC: list[tuple[bytes, DetectedKind, str]] = [
    (b"%PDF-", "pdf", "application/pdf"),
    (b"PK\x03\x04", "docx", "application/zip"),  # also pptx/xlsx; refined below
    (b"<!doctype html", "html", "text/html"),
    (b"<!DOCTYPE html", "html", "text/html"),
    (b"<html", "html", "text/html"),
    # NB: audio (ADR-0017) is NOT a `_MAGIC` row — it is handled by `_detect_audio` below,
    # which the plain offset-0 prefix rows can't do: the ASCII magics ID3/fLaC/OggS collide
    # with prose ABOUT those formats, and the MPEG/AAC frame sync + the `ftyp` container need
    # structural checks (a binary-file gate + a brand check) a prefix row can't express.
]


def _looks_like_text(head: bytes) -> bool:
    """Heuristic — if the first 4 KiB decode as UTF-8 with no NULs."""
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _refine_office(path: Path) -> tuple[DetectedKind, str, bool]:
    """Office documents are ZIPs. Look at the entries to distinguish
    docx/xlsx/pptx and detect macros (presence of `vbaProject.bin`).
    """
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile:
        return "unknown", "application/octet-stream", False

    # `n` is already lowercased via `.lower()`; the substring must also be
    # lowercase or the check is a no-op (pre-existing case-sensitivity bug).
    has_macros = any("vbaproject.bin" in n.lower() for n in names)
    if any(n.startswith("word/") for n in names):
        return (
            "docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            has_macros,
        )
    if any(n.startswith("ppt/") for n in names):
        return (
            "pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            has_macros,
        )
    if any(n.startswith("xl/") for n in names):
        return (
            "xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            has_macros,
        )
    return "unknown", "application/zip", has_macros


# ISO-BMFF (`ftyp`) AUDIO brand — `M4A ` only (the single ftyp-audio brand whose suffix
# `.m4a` the route handles in v1). The SAME box wraps MP4/MOV video, HEIC/AVIF images, and
# audiobook `.m4b` / DRM `.m4p` (whose suffixes are not routable yet), so the ftyp branch
# accepts ONLY this brand; everything else stays rejected (audio-only ingest — the "class
# video" / audiobook cases are later extensions, ADR-0017).
_M4A_AUDIO_BRANDS: frozenset[bytes] = frozenset({b"M4A "})


def _is_m4a_audio(head: bytes) -> bool:
    """True iff an ISO-BMFF `ftyp` head declares the `M4A ` audio brand. The box is
    `size(4) 'ftyp'(4) major_brand(4) minor_version(4) compatible_brands(4·N)`, so the brand can
    be the MAJOR brand (offset 8) OR any COMPATIBLE brand (offset 16, 20, …). A short/truncated
    head simply yields no match (safe slice semantics)."""
    if head[8:12] in _M4A_AUDIO_BRANDS:
        return True
    return any(head[off : off + 4] in _M4A_AUDIO_BRANDS for off in range(16, min(len(head), 40), 4))


# MPEG-audio / AAC-ADTS frame-sync 2nd bytes (`0xFF` is the 1st). A PRECISE set, NOT the broad
# `0xFF + 0xE0`-mask: that mask also passed a UTF-16-LE BOM (`0xFF 0xFE`, syntactically MPEG-1
# Layer I — extinct) and reserved syncs (`0xFFE0/1`), misdetecting BOM text as audio. These cover
# MP3 Layer III MPEG-1/2 (±CRC) and AAC-ADTS MPEG-4/2 (±CRC) — the realistic untagged-audio cases.
_AAC_SYNC_BYTES: frozenset[int] = frozenset({0xF1, 0xF9, 0xF0, 0xF8})  # AAC-ADTS (layer bits 00)
_MP3_SYNC_BYTES: frozenset[int] = frozenset({0xFB, 0xFA, 0xF3, 0xF2})  # MP3 Layer III ±CRC
# Ogg is a SHARED container (audio: Vorbis/Opus/FLAC/Speex; video: Theora), so an `OggS` head is
# accepted ONLY when an AUDIO codec signature is in its first page — Theora video stays rejected.
_OGG_AUDIO_CODECS: tuple[bytes, ...] = (b"OpusHead", b"vorbis", b"FLAC", b"Speex")


def _detect_audio(head: bytes) -> tuple[DetectedKind, str, bool] | None:
    """All audio-format detection (ADR-0017), kept OUT of the generic offset-0 `_MAGIC` loop
    because audio needs structural guards a plain prefix row can't express:

    - the MPEG/AAC frame sync is BINARY (`0xFF` + a known sync 2nd byte) — `0xFF` never starts a
      UTF-8 file and the PRECISE 2nd-byte set excludes the UTF-16-LE BOM (`0xFFFE`) + reserved
      syncs, so no text/image collision; AAC vs MP3 is split by the 2nd byte for the MIME;
    - the ASCII container magics `ID3`/`fLaC`/`OggS` COLLIDE with prose ABOUT those formats (a
      note "ID3 tags …"), so they require the head to NOT look like text (a real audio file is
      binary) — the same rigour the `ftyp` branch has; `OggS` additionally requires an AUDIO
      codec (Ogg also wraps Theora video);
    - WAV (`RIFF`@0 + `WAVE`@8) and the `ftyp` box (`ftyp`@4) are not offset-0; `ftyp` is shared
      with MP4/MOV video + HEIC/AVIF images, so it needs the `M4A ` audio brand (`_is_m4a_audio`).

    Returns `(kind, mime, has_macros=False)`, or None if not audio."""
    if len(head) >= 2 and head[0] == 0xFF:
        if head[1] in _AAC_SYNC_BYTES:
            return "audio", "audio/aac", False
        if head[1] in _MP3_SYNC_BYTES:
            return "audio", "audio/mpeg", False
    if not _looks_like_text(head):  # the real files are binary; prose ABOUT them stays text
        if head.startswith(b"ID3"):
            return "audio", "audio/mpeg", False
        if head.startswith(b"fLaC"):
            return "audio", "audio/flac", False
        if head.startswith(b"OggS") and any(c in head[:64] for c in _OGG_AUDIO_CODECS):
            return "audio", "audio/ogg", False  # audio Ogg only; Theora video stays rejected
        if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
            return "audio", "audio/wav", False
        if head[4:8] == b"ftyp" and _is_m4a_audio(head):
            return "audio", "audio/mp4", False
    return None


def _detect(path: Path) -> tuple[DetectedKind, str, bool]:
    """Return (kind, mime, has_macros). Reads at most 4 KiB."""
    with open(path, "rb") as f:
        head = f.read(4096)

    for prefix, kind, mime in _MAGIC:
        if head.lower().startswith(prefix.lower()) if kind == "html" else head.startswith(prefix):
            if kind == "docx":  # ZIP-shaped — refine
                return _refine_office(path)
            return kind, mime, False

    # All audio formats (ADR-0017), with structural guards (kept out of the `_MAGIC` loop).
    # BEFORE the text fallback (a binary audio container would otherwise miss every branch).
    audio = _detect_audio(head)
    if audio is not None:
        return audio

    if path.suffix.lower() in {".md", ".markdown"} and _looks_like_text(head):
        return "markdown", "text/markdown", False
    if _looks_like_text(head):
        return "text", "text/plain", False

    return "unknown", "application/octet-stream", False


def validate_file(
    path: Path,
    *,
    max_bytes: int,
    allow_macros: bool,
) -> ValidationResult:
    """Inspect `path` and decide whether to accept it.

    Never raises for rejections — returns a `ValidationResult` with
    `accepted=False` and a `rejection_reason`. Callers turn that into
    an `IngestResult` and a `document.rejected` event.
    """
    size = path.stat().st_size
    if size > max_bytes:
        return ValidationResult(
            accepted=False,
            kind="unknown",
            mime="application/octet-stream",
            size_bytes=size,
            rejection_reason=(
                f"file is {size} bytes; max is {max_bytes} "
                "(raise MEMEX_INGEST__MAX_BYTES to override)"
            ),
        )

    kind, mime, has_macros = _detect(path)
    if kind == "unknown":
        return ValidationResult(
            accepted=False,
            kind=kind,
            mime=mime,
            size_bytes=size,
            rejection_reason="content does not match any supported format",
        )
    if has_macros and not allow_macros:
        return ValidationResult(
            accepted=False,
            kind=kind,
            mime=mime,
            size_bytes=size,
            has_macros=True,
            rejection_reason=(
                "document contains macros (vbaProject.bin); set "
                "ingest.allow_macros=true to accept anyway"
            ),
        )

    return ValidationResult(
        accepted=True,
        kind=kind,
        mime=mime,
        size_bytes=size,
        has_macros=has_macros,
    )
