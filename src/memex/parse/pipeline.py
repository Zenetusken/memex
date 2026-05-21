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
from dataclasses import dataclass
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
from memex.parse.pymupdf_backend import (
    PdfSignals,
    PyMuPDFConversion,
    PyMuPDFCrashed,
    PyMuPDFTimeout,
    PyMuPDFUnavailable,
)
from memex.parse.pymupdf_backend import (
    convert as pymupdf_convert,
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
_PYMUPDF_BREAKER: CircuitBreaker[PyMuPDFConversion] | None = None


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


def _pymupdf_breaker() -> CircuitBreaker[PyMuPDFConversion]:
    global _PYMUPDF_BREAKER
    if _PYMUPDF_BREAKER is None:
        settings = get_settings()
        _PYMUPDF_BREAKER = CircuitBreaker[PyMuPDFConversion](
            name="pymupdf",
            threshold=settings.parse.pymupdf_crash_threshold,
            reset_after_s=300.0,
        )
    return _PYMUPDF_BREAKER


def reset_pymupdf_breaker() -> None:
    """For `memex doctor` and tests."""
    global _PYMUPDF_BREAKER
    _PYMUPDF_BREAKER = None


def get_pymupdf_breaker_state() -> tuple[str, int]:
    b = _pymupdf_breaker()
    return (b.state, b.failures)


def _is_pymupdf_failure(exc: BaseException) -> bool:
    return isinstance(exc, (PyMuPDFTimeout, PyMuPDFCrashed))


# ----- Tiered routing classifier -----
#
# Producer-metadata signals are the highest-fidelity routing data
# a PDF carries: a file whose /Creator says "Microsoft PowerPoint" is
# overwhelmingly likely to have a usable native text layer; a file
# whose /Creator says "ABBYY FineReader" is the output of an OCR
# engine and routing it to a *second* OCR engine doubles the work.
# Tier 1 exploits this directly. Subsequent tiers look at structural
# cues, text distribution, text quality, then aspect-ratio-aware
# density as the final fallback.
BORN_DIGITAL_PRODUCERS: Final[frozenset[str]] = frozenset(
    {
        "powerpoint",
        "keynote",
        "word",
        "pages",
        "openoffice",
        "libreoffice",
        "google docs",
        "google slides",
        "latex",
        "tex",
        "xetex",
        "lualatex",
        "pandoc",
        "wkhtmltopdf",
        "chromium",
        "weasyprint",
        "prince",
        "acrobat distiller",
        "indesign",
        "quark",
    }
)
SCAN_PRODUCERS: Final[frozenset[str]] = frozenset(
    {
        "scanner",
        "omnipage",
        "abbyy",
        "readiris",
        "tesseract",
        "kofax",
        "capture",
        "scansoft",
        "iris",
    }
)


@dataclass(frozen=True)
class _Classification:
    doc_type: str
    confidence: float
    attribution: dict[str, object]
    needs_ocr: bool = False


def _is_mixed_content(s: PdfSignals) -> bool:
    """True iff the document has both native text AND substantial image
    area covering enough pages — the cases where image-embedded text
    (chart labels, screenshots, diagram annotations) needs OCR to be
    captured.
    """
    settings = get_settings()
    return (
        s.chars_per_page_avg >= 50.0
        and s.image_area_fraction
        >= settings.parse.pymupdf_mixed_content_image_area_threshold
        and s.image_heavy_page_fraction
        >= settings.parse.pymupdf_mixed_content_min_image_heavy_pages
    )


def _classify(signals: PdfSignals) -> _Classification:
    """Route a PDF to PyMuPDF / Docling / Docling-with-OCR.

    Tier 1 — producer metadata (the gold signal).
    Tier 2 — structural cues (tagged PDF, fonts, image-heavy).
    Tier 3 — text quality + distribution (mojibake, near-empty).
    Tier 4 — aspect-ratio-aware density with markdown-structure bonus.

    Returns the doc-type label, a confidence in [0, 1], the attribution
    dict for logging, and a `needs_ocr` flag the dispatcher uses when
    routing the fall-through to Docling.
    """
    s = signals
    producer_text = f"{s.creator or ''} {s.producer or ''}".lower().strip()

    born_digital_hits = sorted(
        p for p in BORN_DIGITAL_PRODUCERS if p in producer_text
    )
    scan_hits = sorted(p for p in SCAN_PRODUCERS if p in producer_text)

    # Mixed-content check fires at every "would-use-PyMuPDF" point.
    # When the doc has substantial image area + native text, force
    # Docling-with-OCR so chart labels, screenshots, and diagram
    # annotations make it into the final markdown.
    mixed = _is_mixed_content(s)

    # Tier 1.A — known born-digital producer + text. Highest-signal case.
    if born_digital_hits and s.chars_per_page_avg >= 50.0:
        if mixed:
            return _Classification(
                doc_type="mixed-content",
                confidence=0.20,
                attribution={
                    "tier": "1.A-mixed",
                    "producer_match": born_digital_hits,
                    "image_area_fraction": s.image_area_fraction,
                    "image_heavy_page_fraction": s.image_heavy_page_fraction,
                },
                needs_ocr=True,
            )
        return _Classification(
            doc_type="born-digital",
            confidence=0.98,
            attribution={
                "tier": "1.A",
                "producer_match": born_digital_hits,
                "chars_per_page_avg": s.chars_per_page_avg,
            },
        )

    # Tier 1.B — known scan/OCR producer. Always Docling-with-OCR.
    if scan_hits:
        return _Classification(
            doc_type="scan",
            confidence=0.0,
            attribution={"tier": "1.B", "scan_producer": scan_hits},
            needs_ocr=True,
        )

    # Tier 1.C — born-digital producer but no text. Almost certainly a
    # PowerPoint rasterised to images on export. Force OCR.
    if born_digital_hits and s.chars_per_page_avg < 50.0:
        return _Classification(
            doc_type="born-digital-but-rasterised",
            confidence=0.10,
            attribution={
                "tier": "1.C",
                "producer_match": born_digital_hits,
                "chars_per_page_avg": s.chars_per_page_avg,
            },
            needs_ocr=True,
        )

    # Tier 2 — Structural cues for unknown / generic producers.
    if s.is_tagged and s.chars_per_page_avg >= 80.0:
        if mixed:
            return _Classification(
                doc_type="mixed-content",
                confidence=0.20,
                attribution={
                    "tier": "2-mixed",
                    "tagged": True,
                    "image_area_fraction": s.image_area_fraction,
                },
                needs_ocr=True,
            )
        return _Classification(
            doc_type="tagged-pdf",
            confidence=0.90,
            attribution={"tier": "2", "tagged": True},
        )

    if s.embedded_font_count >= 3 and s.chars_per_page_avg >= 80.0:
        if mixed:
            return _Classification(
                doc_type="mixed-content",
                confidence=0.20,
                attribution={
                    "tier": "2-mixed",
                    "fonts": s.embedded_font_count,
                    "image_area_fraction": s.image_area_fraction,
                },
                needs_ocr=True,
            )
        return _Classification(
            doc_type="fonted-born-digital",
            confidence=0.85,
            attribution={"tier": "2", "fonts": s.embedded_font_count},
        )

    if s.image_heavy_page_fraction > 0.5 and s.chars_per_page_avg < 100.0:
        # Image-heavy with little text. Force OCR — likely a scanned-style
        # doc even if the producer didn't self-identify.
        return _Classification(
            doc_type="image-heavy",
            confidence=0.15,
            attribution={
                "tier": "2",
                "image_pages": s.image_heavy_page_fraction,
                "chars_per_page_avg": s.chars_per_page_avg,
            },
            needs_ocr=True,
        )

    # Tier 3 — text quality + distribution signals.
    if s.replacement_char_fraction > 0.05:
        # >5% U+FFFD → broken encoding → don't trust this extraction.
        # OCR won't help (the underlying glyph mapping is wrong), so
        # let Docling re-extract with its own machinery.
        return _Classification(
            doc_type="mojibake",
            confidence=0.10,
            attribution={
                "tier": "3",
                "replacement_char_fraction": s.replacement_char_fraction,
            },
        )

    if s.chars_per_page_avg < 10.0 and s.chars_per_page_p90 < 30.0:
        # Almost no native text anywhere across the whole distribution.
        # Scan-like — force OCR.
        return _Classification(
            doc_type="scan-like",
            confidence=0.0,
            attribution={
                "tier": "3",
                "chars_per_page_avg": s.chars_per_page_avg,
                "chars_per_page_p90": s.chars_per_page_p90,
            },
            needs_ocr=True,
        )

    if s.empty_page_fraction > 0.5:
        # Mostly-empty PDF. OCR may or may not help; force it on so
        # the rare populated pages get their image-text captured.
        return _Classification(
            doc_type="mostly-empty",
            confidence=0.15,
            attribution={
                "tier": "3",
                "empty_page_fraction": s.empty_page_fraction,
            },
            needs_ocr=True,
        )

    # Tier 4 — aspect-ratio-aware density fallback. Markdown structure
    # gives a confidence boost: clean extraction of headings/tables/
    # lists is strong evidence PyMuPDF got the content right.
    aspect = s.avg_aspect_ratio
    avg = s.chars_per_page_avg

    structure_bonus = 0.0
    if s.has_headings:
        structure_bonus += 0.10
    if s.has_tables:
        structure_bonus += 0.05
    if s.has_lists:
        structure_bonus += 0.05
    if s.has_code_blocks:
        structure_bonus += 0.05

    if aspect >= 1.3:
        base = min(0.85, avg / 200.0)
        conf = min(1.0, base + structure_bonus)
        attribution: dict[str, object] = {
            "tier": "4",
            "doc_shape": "slide",
            "aspect": aspect,
            "chars_per_page_avg": avg,
            "structure_bonus": structure_bonus,
        }
        if mixed and conf >= 0.5:
            return _Classification(
                doc_type="mixed-content",
                confidence=0.20,
                attribution={**attribution, "tier": "4-mixed"},
                needs_ocr=True,
            )
        return _Classification(
            doc_type="slide", confidence=conf, attribution=attribution
        )

    if aspect < 1.0:
        base = min(0.85, avg / 800.0)
        conf = min(1.0, base + structure_bonus)
        attribution = {
            "tier": "4",
            "doc_shape": "paper",
            "aspect": aspect,
            "chars_per_page_avg": avg,
            "structure_bonus": structure_bonus,
        }
        if mixed and conf >= 0.5:
            return _Classification(
                doc_type="mixed-content",
                confidence=0.20,
                attribution={**attribution, "tier": "4-mixed"},
                needs_ocr=True,
            )
        return _Classification(
            doc_type="paper", confidence=conf, attribution=attribution
        )

    base = min(0.85, avg / 400.0)
    conf = min(1.0, base + structure_bonus)
    attribution = {
        "tier": "4",
        "doc_shape": "unknown",
        "aspect": aspect,
        "chars_per_page_avg": avg,
        "structure_bonus": structure_bonus,
    }
    if mixed and conf >= 0.5:
        return _Classification(
            doc_type="mixed-content",
            confidence=0.20,
            attribution={**attribution, "tier": "4-mixed"},
            needs_ocr=True,
        )
    return _Classification(
        doc_type="unknown", confidence=conf, attribution=attribution
    )


@dataclass(frozen=True)
class _PreFilterDecision:
    """Outcome of the PyMuPDF pre-filter step.

    `result` is the parse result if PyMuPDF won the routing decision;
    otherwise None and the caller must run Docling. `force_ocr_on_fallthrough`
    is the classifier's hint about whether the fall-through Docling
    call should have OCR forced on (mixed-content, scan-like,
    image-heavy, etc.).
    """

    result: ParseResult | None
    force_ocr_on_fallthrough: bool = False


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
    vault_path: Path,
    doc_id: str,
    source: Path,
    *,
    force_ocr: bool | None = None,
) -> ParseResult:
    settings = get_settings()
    correlation_id = str(ulid.ULID())
    log = logger.bind(
        doc_id=doc_id,
        correlation_id=correlation_id,
        engine="docling",
        force_ocr=force_ocr,
    )
    log.info("parse.docling.start", source=str(source))

    start = time.monotonic()
    breaker = _docling_breaker()
    try:
        conversion = await breaker.run(
            lambda: docling_convert(
                source,
                timeout_s=settings.parse.docling_timeout_s,
                sandbox_network=settings.parse.docling_sandbox_network,
                force_ocr=force_ocr,
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


async def _parse_with_pymupdf(
    vault_path: Path, doc_id: str, source: Path
) -> _PreFilterDecision:
    """Try the PyMuPDF4LLM pre-filter on a PDF.

    Returns a `_PreFilterDecision`:
      - On clean PyMuPDF win: result=ParseResult, force_ocr_on_fallthrough=False
      - On fall-through with classifier hint: result=None, force_ocr=<classifier>
      - On unavailable/timeout/crash: result=None, force_ocr=False (let Docling run default)

    Never raises — the pre-filter is a *try*. Only Docling produces
    manifest crash records; PyMuPDF failures are logged and the caller
    proceeds to Docling.
    """
    settings = get_settings()
    correlation_id = str(ulid.ULID())
    log = logger.bind(
        doc_id=doc_id, correlation_id=correlation_id, engine="pymupdf"
    )

    if not settings.parse.pymupdf_enabled:
        log.info("parse.pymupdf.disabled")
        return _PreFilterDecision(result=None)

    breaker = _pymupdf_breaker()
    log.info("parse.pymupdf.start", source=str(source))

    start = time.monotonic()
    try:
        conversion = await breaker.run(
            lambda: pymupdf_convert(
                source,
                timeout_s=settings.parse.pymupdf_timeout_s,
                sandbox_network=settings.parse.pymupdf_sandbox_network,
            ),
            is_failure=_is_pymupdf_failure,
        )
    except CircuitBreakerOpen:
        log.info("parse.pymupdf.circuit_open")
        return _PreFilterDecision(result=None)
    except PyMuPDFUnavailable as e:
        log.info("parse.pymupdf.unavailable", error=str(e))
        return _PreFilterDecision(result=None)
    except (PyMuPDFTimeout, PyMuPDFCrashed) as e:
        log.warning("parse.pymupdf.failed", error=str(e), error_type=type(e).__name__)
        return _PreFilterDecision(result=None)
    except SandboxLoadFailed as e:
        log.warning("parse.pymupdf.sandbox_failed", error=str(e))
        return _PreFilterDecision(result=None)

    classification = _classify(conversion.signals)
    log.info(
        "parse.pymupdf.classified",
        doc_type=classification.doc_type,
        confidence=classification.confidence,
        needs_ocr=classification.needs_ocr,
        attribution=classification.attribution,
        creator=conversion.signals.creator,
        producer=conversion.signals.producer,
        chars_per_page_avg=conversion.signals.chars_per_page_avg,
        image_area_fraction=conversion.signals.image_area_fraction,
    )

    if classification.confidence < settings.parse.pymupdf_min_confidence:
        log.info(
            "parse.pymupdf.low_confidence_falling_through",
            doc_type=classification.doc_type,
            confidence=classification.confidence,
            threshold=settings.parse.pymupdf_min_confidence,
            force_ocr=classification.needs_ocr,
        )
        return _PreFilterDecision(
            result=None, force_ocr_on_fallthrough=classification.needs_ocr
        )

    # PyMuPDF wins. Write the canonical markdown, record the manifest,
    # return the ParseResult.
    duration_ms = int((time.monotonic() - start) * 1000)

    existing = (
        await read_document(vault_path, doc_id)
        if (vault_path / "documents" / f"{doc_id}.md").exists()
        else None
    )
    fm = existing.frontmatter if existing else Frontmatter(title=doc_id)
    doc = VaultDocument(
        ref=_bootstrap_ref(vault_path, doc_id, conversion.markdown),
        frontmatter=fm,
        body=conversion.markdown,
        mtime_ns=0,
    )
    ref = await write_document(vault_path, doc)

    pages: list[PageDecision] = [
        PageDecision(
            page=p.page,
            engine="pymupdf",
            confidence=classification.confidence,
            rationale=f"pymupdf:{classification.doc_type}",
        )
        for p in conversion.pages
    ]

    parse_stage = ParseStage(
        correlation_id=correlation_id,
        parsed_at=now_utc(),
        parser_version=_PARSER_VERSION,
        pymupdf_version=conversion.pymupdf_version,
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
        "parse.pymupdf.done",
        pages=len(pages),
        markdown_bytes=len(conversion.markdown.encode("utf-8")),
        duration_ms=duration_ms,
    )
    return _PreFilterDecision(
        result=ParseResult(
            doc_id=doc_id,
            correlation_id=correlation_id,
            engine="pymupdf",
            pages=pages,
            markdown_bytes=len(conversion.markdown.encode("utf-8")),
        )
    )


async def _parse_pdf(
    vault_path: Path, doc_id: str, source: Path
) -> ParseResult:
    """Route a PDF through PyMuPDF pre-filter → Docling fallback.

    The pre-filter inspects the doc, runs the tiered classifier, and
    either wins outright (high-confidence born-digital) or falls
    through with a hint about whether Docling should force OCR on
    (mixed-content, scan-like, image-heavy).
    """
    decision = await _parse_with_pymupdf(vault_path, doc_id, source)
    if decision.result is not None:
        return decision.result
    return await _parse_with_docling(
        vault_path, doc_id, source, force_ocr=decision.force_ocr_on_fallthrough or None
    )


async def parse_document(doc_id: str) -> ParseResult:
    """Parse the document with `doc_id`'s source into canonical markdown."""
    settings = get_settings()
    source = _source_file(settings.vault_path, doc_id)

    if source.suffix.lower() in {".md", ".markdown"}:
        return await _passthrough_markdown(settings.vault_path, doc_id, source)

    if source.suffix.lower() == ".pdf":
        return await _parse_pdf(settings.vault_path, doc_id, source)

    return await _parse_with_docling(settings.vault_path, doc_id, source)
