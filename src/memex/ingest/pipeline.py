"""Ingest stage — validate, copy, queue. See IMPLEMENTATION-PLAN §1.2.

The on-ramp. Validates a file's magic + size + MIME, derives a stable
`doc_id`, copies the original into `vault/documents/{doc_id}/source.{ext}`,
writes the initial manifest entry, and emits `document.queued`.

This stage does not look inside the document beyond the validation
header read. Parsing is a separate stage (`memex.parse`) that picks up
queued documents.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import structlog
import ulid
from pydantic import BaseModel, Field

from memex.core.config import get_settings
from memex.core.manifest import IngestStage, now_utc, update_manifest
from memex.ingest.validation import ValidationResult, validate_file
from memex.vault.store import (
    DocumentRef,
    Frontmatter,
    assign_doc_id,
    create_document,
)

logger = structlog.get_logger(__name__)

_EXTENSION_FOR_KIND: dict[str, str] = {
    "pdf": ".pdf",
    "docx": ".docx",
    "pptx": ".pptx",
    "xlsx": ".xlsx",
    "html": ".html",
    "markdown": ".md",
    "text": ".txt",
}


class IngestRequest(BaseModel):
    """Caller-supplied ingest input — a path to the source file plus
    the ULID correlation_id that threads through the whole pipeline
    (logs + Langfuse traces)."""

    source_path: Path
    correlation_id: str = Field(default_factory=lambda: str(ulid.ULID()))


class IngestResult(BaseModel):
    """Outcome of one ingest call — whether the file was accepted,
    its assigned `doc_id` (set on acceptance), and any rejection
    diagnostics (kind / mime detection, rejection_reason)."""

    correlation_id: str
    source_path: str
    accepted: bool
    doc_id: str | None = None
    rejection_reason: str | None = None
    detected_kind: str | None = None
    detected_mime: str | None = None
    size_bytes: int | None = None
    is_markdown: bool = False


async def _copy_source(
    vault_path: Path,
    doc_id: str,
    source_path: Path,
    detected_kind: str,
) -> Path:
    """Copy the original into vault/documents/{doc_id}/source.{ext}."""
    ext = _EXTENSION_FOR_KIND.get(detected_kind, source_path.suffix or ".bin")
    target_dir = vault_path / "documents" / doc_id
    target_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = target_dir / f"source{ext}"
    # shutil.copy2 preserves mtime; reasonable for archival.
    shutil.copy2(source_path, target)
    return target


def _hash_file(path: Path) -> str:
    """Streaming sha256. Files up to `IngestSettings.max_bytes` (256 MiB
    by default; user-configurable) won't materialise into RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


async def ingest_file(req: IngestRequest) -> IngestResult:
    """Validate, copy, write manifest, queue. Single-file path."""
    settings = get_settings()
    log = logger.bind(
        correlation_id=req.correlation_id,
        source=str(req.source_path),
    )
    log.info("ingest.start")

    if not req.source_path.exists():
        return IngestResult(
            correlation_id=req.correlation_id,
            source_path=str(req.source_path),
            accepted=False,
            rejection_reason="source file does not exist",
        )

    validation: ValidationResult = validate_file(
        req.source_path,
        max_bytes=settings.ingest.max_bytes,
        allow_macros=settings.ingest.allow_macros,
    )

    if not validation.accepted:
        log.info("ingest.rejected", reason=validation.rejection_reason)
        return IngestResult(
            correlation_id=req.correlation_id,
            source_path=str(req.source_path),
            accepted=False,
            rejection_reason=validation.rejection_reason,
            detected_kind=validation.kind,
            detected_mime=validation.mime,
            size_bytes=validation.size_bytes,
        )

    content_sha = _hash_file(req.source_path)
    doc_id = assign_doc_id(content_sha, req.source_path.stem)

    # For markdown sources we still copy the original to source.md, but
    # the canonical {doc_id}.md is the parsed/normalized version that the
    # parse stage will produce. To avoid a chicken-and-egg with `memex
    # ask` when no parse step has run, the parse stage's passthrough
    # branch reads source.md and writes {doc_id}.md identically.
    await _copy_source(settings.vault_path, doc_id, req.source_path, validation.kind)

    ingest_stage = IngestStage(
        correlation_id=req.correlation_id,
        ingested_at=now_utc(),
        source_path=str(req.source_path),
        source_size_bytes=validation.size_bytes,
        detected_mime=validation.mime,
    )
    await update_manifest(
        settings.vault_path,
        doc_id,
        content_sha256=content_sha,
        ingest=ingest_stage,
        correlation_id=req.correlation_id,
    )

    # `document.queued` would land on the event bus here once the bus is
    # mounted by the CLI. Phase 1 ships without the live bus on the CLI
    # path; the integration tests run with a tmp-bus fixture.

    log.info("ingest.accepted", doc_id=doc_id, kind=validation.kind)
    return IngestResult(
        correlation_id=req.correlation_id,
        source_path=str(req.source_path),
        accepted=True,
        doc_id=doc_id,
        detected_kind=validation.kind,
        detected_mime=validation.mime,
        size_bytes=validation.size_bytes,
        is_markdown=(validation.kind == "markdown"),
    )


async def ingest_directory(
    path: Path,
    *,
    recursive: bool = True,
) -> AsyncIterator[IngestResult]:
    """Walk `path` and yield one IngestResult per file."""
    iterator = path.rglob("*") if recursive else path.glob("*")
    for child in sorted(iterator):
        if not child.is_file():
            continue
        # Skip hidden files and previously-ingested vault content.
        if any(part.startswith(".") for part in child.parts):
            continue
        yield await ingest_file(IngestRequest(source_path=child))


async def ingest_markdown_passthrough(
    body: str, *, source_stem: str, correlation_id: str | None = None
) -> DocumentRef:
    """Phase-0 convenience: a markdown string straight into the vault,
    skipping validation. Used by the `--skip-parse` CLI shortcut and by
    tests that don't exercise the validator path.
    """
    settings = get_settings()
    cid = correlation_id or str(ulid.ULID())
    ref = await create_document(
        settings.vault_path,
        body=body,
        source_stem=source_stem,
        frontmatter_fields=Frontmatter(title=source_stem),
    )
    await update_manifest(
        settings.vault_path,
        ref.doc_id,
        content_sha256=ref.content_sha256,
        ingest=IngestStage(
            correlation_id=cid,
            ingested_at=datetime.now(UTC),
            source_path=f"<inline:{source_stem}>",
            source_size_bytes=len(body.encode("utf-8")),
            detected_mime="text/markdown",
        ),
        correlation_id=cid,
    )
    return ref
