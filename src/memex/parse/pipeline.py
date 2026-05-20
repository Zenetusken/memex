"""Parse stage entry point.

`parse_document(doc_id)` reads `vault/documents/{doc_id}/source.{ext}`,
runs Docling (or passes markdown through unchanged), writes the
canonical `vault/documents/{doc_id}.md`, and updates the manifest with
per-page provenance.

The Docling parser circuit breaker (Part VI) trips after
`parse.docling_crash_threshold` failures, after which parse refuses
new work and emits `system.degraded`. Failures count crashes and
timeouts, not validation rejections (those happen at ingest).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Final

import structlog
import ulid
from pydantic import BaseModel

from memex.core.breakers import CircuitBreaker, CircuitBreakerOpen
from memex.core.config import get_settings
from memex.core.errors import ParseConfidenceTooLow, VaultIntegrityError
from memex.core.manifest import (
    PageDecision,
    ParseStage,
    now_utc,
    update_manifest,
)
from memex.parse.docling_backend import (
    DoclingConversion,
    DoclingCrashed,
    DoclingPageOutput,
    DoclingTimeout,
    DoclingUnavailable,
    SandboxLoadFailed,
)
from memex.parse.docling_backend import (
    convert as docling_convert,
)
from memex.parse.vlm_backend import (
    VLMUnavailable,
)
from memex.parse.vlm_backend import (
    convert_pages as vlm_convert_pages,
)
from memex.vault.store import (
    DocumentRef,
    Frontmatter,
    VaultDocument,
    hash_bytes,
    make_ref,
    read_document,
    write_document,
)

logger = structlog.get_logger(__name__)

_PARSER_VERSION: Final[str] = "memex.parse@v1"


class ParseResult(BaseModel):
    doc_id: str
    correlation_id: str
    engine: str
    pages: list[PageDecision]
    markdown_bytes: int


_DOCLING_BREAKER: CircuitBreaker[DoclingConversion] | None = None


def _docling_breaker() -> CircuitBreaker[DoclingConversion]:
    """Lazily construct so the threshold comes from settings."""
    global _DOCLING_BREAKER
    if _DOCLING_BREAKER is None:
        settings = get_settings()
        _DOCLING_BREAKER = CircuitBreaker[DoclingConversion](
            name="docling",
            threshold=settings.parse.docling_crash_threshold,
            reset_after_s=300.0,
        )
    return _DOCLING_BREAKER


def reset_docling_breaker() -> None:
    """For `memex doctor` and tests."""
    global _DOCLING_BREAKER
    _DOCLING_BREAKER = None


def get_docling_breaker_state() -> tuple[str, int]:
    """Public accessor for the Docling breaker's (state, failures).

    Exposed for `memex doctor`; `_docling_breaker` itself is module-private
    because constructing/accessing it has side effects (it lazily reads
    settings).
    """
    b = _docling_breaker()
    return (b.state, b.failures)


def _source_file(vault_path: Path, doc_id: str) -> Path:
    """Find the single `source.*` file under the document's asset dir."""
    asset_dir = vault_path / "documents" / doc_id
    if not asset_dir.is_dir():
        raise VaultIntegrityError(
            f"asset directory missing for doc_id={doc_id!r}",
            context={"path": str(asset_dir), "doc_id": doc_id},
        )
    candidates = sorted(asset_dir.glob("source.*"))
    if not candidates:
        # Markdown-passthrough documents may not have a copied source —
        # the canonical {doc_id}.md is itself the source.
        return vault_path / "documents" / f"{doc_id}.md"
    return candidates[0]


async def _passthrough_markdown(vault_path: Path, doc_id: str, source: Path) -> ParseResult:
    """No real parse needed — the source is already markdown.

    Read the source, write it as the canonical `{doc_id}.md` (preserving
    any frontmatter), record a single passthrough PageDecision in the
    manifest.
    """
    log = logger.bind(doc_id=doc_id, engine="passthrough")
    log.info("parse.passthrough.start")

    body = source.read_text(encoding="utf-8")
    canonical = await read_document(vault_path, doc_id) if (
        vault_path / "documents" / f"{doc_id}.md"
    ).exists() else None
    fm = canonical.frontmatter if canonical else Frontmatter(title=doc_id)
    doc = VaultDocument(
        ref=canonical.ref if canonical else _bootstrap_ref(vault_path, doc_id, body),
        frontmatter=fm,
        body=_strip_frontmatter(body),
        mtime_ns=0,
    )
    ref = await write_document(vault_path, doc)

    correlation_id = str(ulid.ULID())
    page = PageDecision(
        page=1, engine="passthrough", confidence=1.0, rationale="markdown source"
    )
    parse_stage = ParseStage(
        correlation_id=correlation_id,
        parsed_at=now_utc(),
        parser_version=_PARSER_VERSION,
        pages=[page],
        duration_ms=0,
    )
    await update_manifest(
        vault_path,
        doc_id,
        content_sha256=ref.content_sha256,
        parse=parse_stage,
        correlation_id=correlation_id,
    )
    log.info("parse.passthrough.done")
    return ParseResult(
        doc_id=doc_id,
        correlation_id=correlation_id,
        engine="passthrough",
        pages=[page],
        markdown_bytes=len(doc.body.encode("utf-8")),
    )


def _strip_frontmatter(text: str) -> str:
    """If the file starts with YAML frontmatter, drop it; we re-add via
    `write_document`.
    """
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end < 0:
        return text
    after = text[end + 4:]
    return after.lstrip("\n")


def _bootstrap_ref(vault_path: Path, doc_id: str, body: str) -> DocumentRef:
    return make_ref(
        vault_path,
        doc_id,
        content_sha256=hash_bytes(body.encode("utf-8")),
    )


async def _parse_with_docling(
    vault_path: Path, doc_id: str, source: Path
) -> ParseResult:
    settings = get_settings()
    correlation_id = str(ulid.ULID())
    log = logger.bind(doc_id=doc_id, correlation_id=correlation_id, engine="docling")
    log.info("parse.docling.start", source=str(source))

    start = time.monotonic()
    breaker = _docling_breaker()
    try:
        conversion = await breaker.run(
            lambda: docling_convert(
                source,
                timeout_s=settings.parse.docling_timeout_s,
                sandbox_network=settings.parse.docling_sandbox_network,
            ),
            is_failure=_is_docling_failure,
        )
    except CircuitBreakerOpen as e:
        log.warning("parse.docling.circuit_open")
        await _record_crash(vault_path, doc_id, correlation_id, e, start)
        raise
    except (DoclingTimeout, DoclingUnavailable, DoclingCrashed, SandboxLoadFailed) as e:
        await _record_crash(vault_path, doc_id, correlation_id, e, start)
        raise

    # Decide per-page engine routing AND escalate low-confidence pages to
    # the VLM when enabled. Pages the VLM successfully transcribes have
    # their Docling output replaced in `conversion.markdown`.
    pages, conversion = await _route_and_escalate(
        conversion,
        source=source,
        threshold=settings.parse.vlm_confidence_threshold,
        disable_vlm=settings.parse.disable_vlm,
        log=log,
    )
    duration_ms = int((time.monotonic() - start) * 1000)

    confident = [p for p in pages if p.confidence >= settings.parse.vlm_confidence_threshold]
    if not confident and pages:
        raise ParseConfidenceTooLow(
            "No pages above the confidence threshold (VLM escalation "
            "either disabled or also failed).",
            context={
                "doc_id": doc_id,
                "threshold": settings.parse.vlm_confidence_threshold,
                "pages": [p.model_dump() for p in pages],
            },
            recoverable=True,
        )

    # Write the canonical markdown via the vault. Preserve title from the
    # ingest stage if available; otherwise default to the doc_id.
    existing = await read_document(vault_path, doc_id) if (
        vault_path / "documents" / f"{doc_id}.md"
    ).exists() else None
    fm = existing.frontmatter if existing else Frontmatter(title=doc_id)
    doc = VaultDocument(
        ref=_bootstrap_ref(vault_path, doc_id, conversion.markdown),
        frontmatter=fm,
        body=conversion.markdown,
        mtime_ns=0,
    )
    ref = await write_document(vault_path, doc)

    parse_stage = ParseStage(
        correlation_id=correlation_id,
        parsed_at=now_utc(),
        parser_version=_PARSER_VERSION,
        docling_version=conversion.docling_version,
        pages=pages,
        figure_count=conversion.figure_count,
        table_count=conversion.table_count,
        equation_count=conversion.equation_count,
        duration_ms=duration_ms,
    )
    await update_manifest(
        vault_path,
        doc_id,
        content_sha256=ref.content_sha256,
        parse=parse_stage,
        correlation_id=correlation_id,
    )

    log.info(
        "parse.docling.done",
        pages=len(pages),
        figures=conversion.figure_count,
        tables=conversion.table_count,
        duration_ms=duration_ms,
    )
    return ParseResult(
        doc_id=doc_id,
        correlation_id=correlation_id,
        engine="docling",
        pages=pages,
        markdown_bytes=len(conversion.markdown.encode("utf-8")),
    )


async def _route_and_escalate(
    conversion: DoclingConversion,
    *,
    source: Path,
    threshold: float,
    disable_vlm: bool,
    log: structlog.stdlib.BoundLogger,
) -> tuple[list[PageDecision], DoclingConversion]:
    """For each Docling page, decide engine routing. When `disable_vlm`
    is False AND the source is a PDF, batch every below-threshold page
    through a single VLM acquisition (`vlm_convert_pages`) so we pay
    the lock + future-eviction cost once per document, not once per
    page. Successfully escalated pages replace Docling's per-page
    markdown, and the document-level `conversion.markdown` is
    re-stitched so the canonical write picks up the corrections.
    """
    decisions: list[PageDecision] = []
    escalated_pages: dict[int, DoclingPageOutput] = {}

    # First pass: classify pages, collect the ones to escalate.
    to_escalate: list[int] = []
    page_index = {p.page: p for p in conversion.pages}
    for p in conversion.pages:
        below = p.confidence < threshold
        if not below or disable_vlm or source.suffix.lower() != ".pdf":
            decisions.append(
                PageDecision(
                    page=p.page,
                    engine="docling",
                    confidence=p.confidence,
                    rationale=(
                        "low-confidence; VLM disabled — Docling output kept"
                        if below and disable_vlm
                        else "high-confidence docling output"
                    ),
                )
            )
            continue
        to_escalate.append(p.page)

    # Second pass: batch VLM call. One context acquisition for the lot.
    if to_escalate:
        results = await vlm_convert_pages(
            source_pdf=source,
            page_numbers=to_escalate,
        )
        for page_no in to_escalate:
            result = results.get(page_no)
            original = page_index[page_no]
            if isinstance(result, DoclingPageOutput):
                escalated_pages[page_no] = result
                decisions.append(
                    PageDecision(
                        page=page_no,
                        engine="vlm",
                        confidence=result.confidence,
                        rationale="escalated from low-confidence Docling output",
                    )
                )
                log.info("parse.vlm.escalated", page=page_no)
            else:
                err = result or VLMUnavailable("VLM call returned no result")
                decisions.append(
                    PageDecision(
                        page=page_no,
                        engine="docling",
                        confidence=original.confidence,
                        rationale=f"VLM escalation failed: {err}",
                    )
                )
                log.warning("parse.vlm.failed", page=page_no, error=str(err))

    if escalated_pages:
        # Re-stitch the document-level markdown so VLM output makes it
        # into the canonical {doc_id}.md. We keep the original page order
        # and substitute per page.
        stitched_pages: list[DoclingPageOutput] = []
        for p in conversion.pages:
            stitched_pages.append(escalated_pages.get(p.page, p))
        conversion = conversion.model_copy(
            update={
                "pages": stitched_pages,
                "markdown": "\n\n".join(
                    sp.markdown for sp in stitched_pages if sp.markdown
                ),
            }
        )

    return decisions, conversion


def _is_docling_failure(exc: BaseException) -> bool:
    """Filter for the breaker — only count infra-style failures.

    Docling timeouts, unavailability, and crashes (subprocess exit
    non-zero) count toward tripping the breaker. `SandboxLoadFailed`
    and `ParseConfidenceTooLow` are not in the first tuple, so they
    return False here without needing an explicit exclusion — both are
    caller-level expected outcomes the breaker should not punish.
    """
    return isinstance(
        exc, (DoclingTimeout, DoclingUnavailable, DoclingCrashed)
    )


async def _record_crash(
    vault_path: Path,
    doc_id: str,
    correlation_id: str,
    exc: BaseException,
    start: float,
) -> None:
    """Record a parse-stage crash in the manifest.

    Manifest may not yet exist (user ran `parse` against a sideloaded
    markdown). `update_manifest` requires `content_sha256` only for
    initial creation, so we pass it iff the manifest doesn't exist.
    The source file is the authoritative bytes at this point.
    """
    from memex.core.manifest import read_manifest

    parse_stage = ParseStage(
        correlation_id=correlation_id,
        parsed_at=now_utc(),
        parser_version=_PARSER_VERSION,
        crashed=True,
        crash_message=str(exc),
        duration_ms=int((time.monotonic() - start) * 1000),
    )
    existing = await read_manifest(vault_path, doc_id)
    if existing is None:
        source = _source_file(vault_path, doc_id)
        content_sha = hash_bytes(source.read_bytes()) if source.exists() else "0" * 64
        await update_manifest(
            vault_path,
            doc_id,
            content_sha256=content_sha,
            parse=parse_stage,
            correlation_id=correlation_id,
        )
    else:
        await update_manifest(
            vault_path,
            doc_id,
            parse=parse_stage,
            correlation_id=correlation_id,
        )


async def parse_document(doc_id: str) -> ParseResult:
    """Parse the document with `doc_id`'s source into canonical markdown."""
    settings = get_settings()
    source = _source_file(settings.vault_path, doc_id)

    if source.suffix.lower() in {".md", ".markdown"}:
        return await _passthrough_markdown(settings.vault_path, doc_id, source)

    return await _parse_with_docling(settings.vault_path, doc_id, source)
