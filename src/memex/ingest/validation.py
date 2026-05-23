"""File-format and content validation — see GUIDELINES.md Part VI security.

Magic-number checks are non-optional. We never trust the filename
extension. Office documents are inspected for macros and rejected
unless `IngestSettings.allow_macros=True`. PDFs are verified for the
`%PDF` header. Markdown and plain text are accepted but length-checked.

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


def _detect(path: Path) -> tuple[DetectedKind, str, bool]:
    """Return (kind, mime, has_macros). Reads at most 4 KiB."""
    with open(path, "rb") as f:
        head = f.read(4096)

    for prefix, kind, mime in _MAGIC:
        if head.lower().startswith(prefix.lower()) if kind == "html" else head.startswith(prefix):
            if kind == "docx":  # ZIP-shaped — refine
                return _refine_office(path)
            return kind, mime, False

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
