"""File-format and content validation — see GUIDELINES.md Part VI security.

Magic-number checks are non-optional. We never trust the filename
extension. Office documents are inspected for macros and rejected
unless `IngestSettings.allow_macros=True`. PDFs are verified for the
`%PDF` header. Markdown and plain text are accepted but length-checked.

Audio sources (MP3/WAV/M4A/FLAC/Ogg/Opus/AAC, ADR-0017) are accepted by
magic via `_detect_audio`. The ASCII magics (ID3/fLaC/OggS) are gated on
the head NOT looking like text (they collide with prose ABOUT those
formats); the binary MPEG/AAC frame sync is matched by a precise sync set;
and the ISO-BMFF `ftyp` box — SHARED by M4A audio, MP4/MOV video, and
HEIC/AVIF images — requires the `M4A ` brand for the audio path.

Audio-bearing VIDEO containers (MP4/M4V/MOV/WebM/MKV — the "class video"
case, ADR-0017) are accepted via `_detect_video` (a curated `ftyp` VIDEO
brand set, or the Matroska/WebM EBML magic): the parse route transcribes
their AUDIO track (the visual track is ignored in v1; the slide content
comes from the companion PDF via the Phase-2 merge).

Standalone image files (PNG/JPEG/WebP/BMP/TIFF/GIF, ADR-0020) are accepted
by magic via `_detect_image` and routed to the scan→VLM transcription path
(an image is a one-page scan). The unambiguous binary magics (PNG/JPEG/
TIFF; WebP via `RIFF`+`WEBP`, distinct from WAV's `RIFF`+`WAVE`) match
directly; the ASCII-startable ones (BMP `BM`, GIF `GIF8…`) are gated on the
head NOT looking like text. HEIC/AVIF `ftyp`-box image containers stay
rejected (their brands stay excluded from `_VIDEO_FTYP_BRANDS`) — they need
a separate decode dependency, deferred.

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
    "video",
    "image",
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

# ISO-BMFF (`ftyp`) VIDEO brands (ADR-0017 — the "class video" extension): the route transcribes
# the AUDIO track of a video container (faster-whisper/PyAV decodes it). A CURATED set of MP4/MOV
# brands — NOT "any ftyp" — so HEIC/AVIF IMAGE brands (`heic`/`heix`/`mif1`/`avif`/`avis`) + the
# `M4A ` audio brand (handled above) stay out; the real ZOOM/screen-recording `.mp4` uses `isom`.
_VIDEO_FTYP_BRANDS: frozenset[bytes] = frozenset(
    {b"isom", b"iso2", b"iso4", b"iso5", b"iso6", b"mp41", b"mp42", b"avc1", b"M4V ", b"M4VH",
     b"M4VP", b"qt  ", b"dash", b"hev1", b"hvc1"}
)
# Matroska/WebM EBML magic (offset 0); the DocType string in the head splits webm vs mkv mime.
_EBML_MAGIC: bytes = b"\x1aE\xdf\xa3"


def _is_video_ftyp(head: bytes) -> bool:
    """True iff an ISO-BMFF `ftyp` head declares a known VIDEO brand (major @8 OR any compatible
    brand @16,20,…). Short head → no match (safe slicing). Mirrors `_is_m4a_audio`'s brand scan."""
    if head[8:12] in _VIDEO_FTYP_BRANDS:
        return True
    return any(
        head[off : off + 4] in _VIDEO_FTYP_BRANDS for off in range(16, min(len(head), 40), 4)
    )


def _detect_video(head: bytes) -> tuple[DetectedKind, str, bool] | None:
    """Audio-bearing VIDEO containers (ADR-0017 "class video"): the route extracts + transcribes
    the AUDIO track (the visual track is ignored in v1 — the slide content comes from the companion
    PDF via the Phase-2 merge). Detected by the ISO-BMFF `ftyp` box with a curated VIDEO brand
    (`.mp4`/`.m4v`/`.mov`) or the Matroska/WebM EBML magic (`.webm`/`.mkv`); a container with no
    audio track transcribes to nothing → recoverable refuse (HARD-gate-safe). Gated on the head
    being BINARY (real containers are). Returns `(kind, mime, has_macros=False)`, or None."""
    if _looks_like_text(head):  # prose ABOUT a video format stays text
        return None
    if head[4:8] == b"ftyp" and _is_video_ftyp(head):
        return "video", "video/mp4", False
    if head.startswith(_EBML_MAGIC):
        mime = "video/webm" if b"webm" in head[:64] else "video/x-matroska"
        return "video", mime, False
    return None


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


def _detect_image(head: bytes) -> tuple[DetectedKind, str, bool] | None:
    """Standalone image files (ADR-0020) — accepted by magic, routed to the scan→VLM
    transcription path (an image is a one-page scan). The unambiguous binary magics match
    directly: PNG (`\\x89PNG\\r\\n\\x1a\\n`), JPEG (`\\xff\\xd8\\xff`), TIFF (`II*\\x00` LE /
    `MM\\x00*` BE), and WebP (`RIFF`@0 + `WEBP`@8 — distinct from WAV's `RIFF`+`WAVE`, which
    `_detect_audio` claims first). The ASCII-STARTABLE magics — BMP (`BM`) and GIF
    (`GIF87a`/`GIF89a`) — are gated on the head NOT looking like text (a real image is binary),
    mirroring the audio ASCII-magic rigour, so prose like "BMW" / "GIF is a format" stays text.
    HEIC/AVIF (`ftyp`-box images) are NOT matched here → stay rejected. Note JPEG's `\\xff\\xd8`
    is not an MP3/AAC frame sync (`\\xd8` ∉ the sync sets), so `_detect_audio` returns None for it
    first. Returns `(kind, mime, has_macros=False)`, or None."""
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image", "image/png", False
    if head.startswith(b"\xff\xd8\xff"):
        return "image", "image/jpeg", False
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image", "image/webp", False
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "image", "image/tiff", False
    if not _looks_like_text(head):  # the ASCII-startable magics need the binary guard
        if head[:2] == b"BM":
            return "image", "image/bmp", False
        if head[:6] in (b"GIF87a", b"GIF89a"):
            return "image", "image/gif", False
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

    # Audio-bearing VIDEO containers (ADR-0017 "class video") — AFTER audio (M4A is more specific
    # than a generic video ftyp brand) and before the text fallback.
    video = _detect_video(head)
    if video is not None:
        return video

    # Standalone image files (ADR-0020) — after audio/video (WAV's RIFF + M4A's ftyp are claimed
    # first), before the text fallback (a binary image would otherwise miss every branch).
    image = _detect_image(head)
    if image is not None:
        return image

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
    if size == 0:
        # An empty file otherwise slips through as "text" (`_looks_like_text(b"")` is True — b""
        # decodes cleanly with no NUL), producing a junk 0-chunk document with an empty body.
        return ValidationResult(
            accepted=False,
            kind="unknown",
            mime="application/octet-stream",
            size_bytes=0,
            rejection_reason="file is empty",
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
