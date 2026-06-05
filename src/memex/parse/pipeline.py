# pyright: reportConstantRedefinition=false
# `_DOCLING_BREAKER` and `_PYMUPDF_BREAKER` are uppercase module-level
# singletons intentionally rebound by their lazy-init helpers
# (`_docling_breaker()`, `_pymupdf_breaker()`) and the test-facing
# reset utilities (`reset_docling_breaker`, `reset_pymupdf_breaker`).

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

import asyncio
import re
import shutil
import tempfile
import time
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import structlog
import ulid
from pydantic import BaseModel

from memex.core.breakers import CircuitBreaker, CircuitBreakerOpen
from memex.core.config import get_settings
from memex.core.errors import ConfigurationError, ParseConfidenceTooLow, VaultIntegrityError
from memex.core.manifest import (
    ChartExtraction,
    PageDecision,
    ParseStage,
    TranscriptSegment,
    now_utc,
    read_manifest,
    update_manifest,
)
from memex.core.table_linearize import (
    GFM_TABLE_RE,
    is_layout_table,
    parse_gfm_table,
    table_cell_lines,
)
from memex.core.text import IMAGE_PLACEHOLDER_RE
from memex.parse.asr_backend import ASRSegment, transcribe_audio
from memex.parse.asr_cache import ASRTranscriptionCache
from memex.parse.chart_ocr_backend import (
    ChartOCROutput,
    ChartOCRUnavailable,
    chart_ocr_extract,
)
from memex.parse.chart_ocr_cache import ChartOCRCache
from memex.parse.docling_backend import (
    DoclingConversion,
    DoclingCrashed,
    DoclingPageOutput,
    DoclingTimeout,
    DoclingUnavailable,
    FigureMetadata,
    SandboxLoadFailed,
)
from memex.parse.docling_backend import (
    convert as docling_convert,
)
from memex.parse.image_convert import IMAGE_SUFFIXES, convert_image_to_pdf
from memex.parse.office_convert import OFFICE_SUFFIXES, convert_to_pdf
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
from memex.parse.vlm_cache import VLMTranscriptionCache
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
    """Return value of `parse_document` — which engine handled the
    document, the per-page routing record, and how much markdown was
    written. Serialized into the manifest's `ParseStage`."""

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
    """Current state + failure count of the PyMuPDF circuit breaker.
    Surfaced by the `memex doctor` report (symmetric to
    `get_docling_breaker_state`)."""
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
        and s.image_area_fraction >= settings.parse.pymupdf_mixed_content_image_area_threshold
        and s.image_heavy_page_fraction
        >= settings.parse.pymupdf_mixed_content_min_image_heavy_pages
    )


def _is_slide_deck(s: PdfSignals) -> bool:
    """True iff the document looks like a slide deck — landscape
    aspect ratio plus either moderate-to-low text-density per page
    OR substantial image area (chart-heavy escape valve).

    The chars-per-page gate catches typical text-thin slide decks.
    The image-area gate (P3.3 Session 2) catches chart-heavy slide
    decks where per-page char count is inflated past 800 by PyMuPDF's
    `[chart-text]` extraction of axis labels — the CUDA deck pattern
    that P3.3 chart-OCR targets. Without the second gate those decks
    stay on PyMuPDF and the agent grounds on noisy unstructured
    chart-text; with the gate they route to Docling where the
    chart-OCR backend can extract structured tables.

    The lower 50 chars/page floor stays in place so rasterised
    image-only PDFs fall to Tier 1.C instead.
    """
    settings = get_settings().parse
    aspect_ok = s.avg_aspect_ratio >= settings.pymupdf_slide_deck_aspect_threshold
    if not aspect_ok or s.chars_per_page_avg < 50.0:
        return False
    text_thin = s.chars_per_page_avg < float(settings.pymupdf_slide_deck_max_chars_per_page)
    chart_heavy = (
        s.image_area_fraction >= settings.pymupdf_slide_deck_chart_heavy_image_area_threshold
    )
    return text_thin or chart_heavy


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

    born_digital_hits = sorted(p for p in BORN_DIGITAL_PRODUCERS if p in producer_text)
    scan_hits = sorted(p for p in SCAN_PRODUCERS if p in producer_text)

    # Mixed-content check fires at every "would-use-PyMuPDF" point.
    # When the doc has substantial image area + native text, force
    # Docling-with-OCR so chart labels, screenshots, and diagram
    # annotations make it into the final markdown.
    mixed = _is_mixed_content(s)

    # Tier 0.5 — slide-deck override. Preempts Tier 1.A because
    # PowerPoint-produced decks otherwise win confidence 0.98 →
    # PyMuPDF, which loses chart structure to [chart-text] noise.
    # When both signals fire (landscape AND moderate-low chars-per-
    # page), return a low-confidence slide-deck classification so the
    # existing fallthrough routes to Docling.
    #
    # Skipped when scan_hits is set (Tier 1.B's scan-producer routing
    # to Docling-with-OCR is the right destination regardless of
    # shape). When the doc is *also* mixed-content (chart imagery
    # heavy enough to want OCR for figure-embedded labels), we
    # inherit `needs_ocr=True` so the mixed-content quality bar still
    # holds.
    if _is_slide_deck(s) and not scan_hits:
        return _Classification(
            doc_type="slide-deck-mixed" if mixed else "slide-deck",
            confidence=0.10,  # below pymupdf_min_confidence default 0.5
            attribution={
                "tier": "0.5-slide-deck" + ("-mixed" if mixed else ""),
                "avg_aspect_ratio": s.avg_aspect_ratio,
                "chars_per_page_avg": s.chars_per_page_avg,
                "producer_match": born_digital_hits,
            },
            needs_ocr=mixed,
        )

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
        return _Classification(doc_type="slide", confidence=conf, attribution=attribution)

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
        return _Classification(doc_type="paper", confidence=conf, attribution=attribution)

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
    return _Classification(doc_type="unknown", confidence=conf, attribution=attribution)


# A doc with fewer extractable chars/page than this is "image-only" — a scan, not a
# text doc with some images. Mirrors the image-heavy classifier threshold (line ~413).
_SCAN_MAX_CHARS_PER_PAGE = 100.0


@dataclass(frozen=True)
class _PreFilterDecision:
    """Outcome of the PyMuPDF pre-filter step.

    `result` is the parse result if PyMuPDF won the routing decision;
    otherwise None and the caller must run Docling. `force_ocr_on_fallthrough`
    is the classifier's hint about whether the fall-through Docling
    call should have OCR forced on (mixed-content, scan-like,
    image-heavy, etc.). `is_scan` flags a predominantly-image doc
    (`doc_type` in {"scan","image-heavy"}) — when the VLM is enabled the
    caller routes it to the scan→VLM route instead of Docling-OCR (which
    can't read handwriting); see `docs/specs/scan-vlm-parse.md`.
    """

    result: ParseResult | None
    force_ocr_on_fallthrough: bool = False
    is_scan: bool = False


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
    canonical = (
        await read_document(vault_path, doc_id)
        if (vault_path / "documents" / f"{doc_id}.md").exists()
        else None
    )
    fm = (
        canonical.frontmatter
        if canonical
        else Frontmatter(title=await derive_title(vault_path, doc_id))
    )
    doc = VaultDocument(
        ref=canonical.ref if canonical else _bootstrap_ref(vault_path, doc_id, body),
        frontmatter=fm,
        body=_strip_frontmatter(body),
        mtime_ns=0,
    )
    ref = await write_document(vault_path, doc)

    correlation_id = str(ulid.ULID())
    page = PageDecision(
        page=1,
        engine="passthrough",
        confidence=1.0,
        rationale="markdown source",
        char_count=len(doc.body),
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
    after = text[end + 4 :]
    return after.lstrip("\n")


def _bootstrap_ref(vault_path: Path, doc_id: str, body: str) -> DocumentRef:
    return make_ref(
        vault_path,
        doc_id,
        content_sha256=hash_bytes(body.encode("utf-8")),
    )


async def derive_title(vault_path: Path, doc_id: str) -> str:
    """Derive a human-readable frontmatter title for a freshly-parsed
    doc from its original source filename, recorded in the manifest's
    ingest stage.

    `PDF CR350 - Cours 2.pdf` ingested → manifest `ingest.source_path`
    is the original path → stem `"CR350 - Cours 2"` becomes the title.
    Falls back to `doc_id` when there's no manifest or ingest stage
    (e.g. a doc created out-of-band, or a test fixture). The ingest
    stage always runs before parse in the normal pipeline, so the
    source_path is available by the time any parse function calls this.

    A meaningful title (not the doc-id slug) is what lets the citation
    resolver match cross-document references — `enrich.citations`
    scores candidates against other docs' titles/author-year/tokens,
    and a slug like `5795b16a-pdf-cr350-cours-1` matches nothing.
    """
    manifest = await read_manifest(vault_path, doc_id)
    if manifest and manifest.ingest and manifest.ingest.source_path:
        stem = Path(manifest.ingest.source_path).stem.strip()
        if stem and not stem.startswith("<inline:"):
            return stem
    return doc_id


# `IMAGE_PLACEHOLDER_RE` now lives in `core/text.py` (one home, shared with the index-time
# re-attach `reattach_chart_extractions` so both find byte-identical placeholders).


async def _vllm_reachable(base_url: str, timeout_s: float = 2.0) -> bool:
    """True iff vLLM's `/v1/models` returns a 200 within `timeout_s`.

    Uses `httpx` if available (the agent layer already depends on it);
    falls back to a stdlib `urllib` call on import error so the parse
    stage stays importable in test environments without httpx.
    """
    url = f"{base_url.rstrip('/')}/models"
    try:
        import httpx  # type: ignore[import-not-found]
    except ImportError:
        # Stdlib fallback.
        from urllib.error import URLError
        from urllib.request import Request, urlopen

        def _check() -> bool:
            try:
                # `url` is the operator-configured vLLM base_url (localhost
                # health probe), not user input — fixed http(s) scheme.
                with urlopen(Request(url), timeout=timeout_s) as resp:  # noqa: S310
                    return 200 <= resp.status < 300
            except (URLError, TimeoutError):
                return False

        return await asyncio.to_thread(_check)

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(url)
            return 200 <= response.status_code < 300
    except Exception:
        return False


async def _vllm_pkill() -> None:
    """Send SIGTERM to any running `vllm serve` process.

    Works whether vLLM was started via systemd or via the bare
    `serve-vllm.sh` script (pkill matches the process name regardless).
    A clean SIGTERM gives vLLM time to release CUDA contexts so the
    subsequent chart-OCR load doesn't compete for half-released memory.
    """
    proc = await asyncio.create_subprocess_exec(
        "pkill",
        "-TERM",
        "-f",
        "vllm serve",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()  # pkill exits non-zero if no match; we don't care


async def _vllm_restart(scripts_dir: Path) -> None:
    """Bring vLLM back up after the chart-OCR pass.

    Tries `systemctl --user start memex-vllm` first (the canonical
    daemon-stack path). If systemctl isn't available OR the unit isn't
    installed, falls back to spawning `scripts/serve-vllm.sh` as a
    detached background process with `nohup`-style redirection.

    Either way, this returns once the START command is issued — caller
    is responsible for waiting on `/v1/models` to come back.
    """
    log = logger.bind(component="chart_ocr.vllm_restart")
    import shutil

    if shutil.which("systemctl") is not None:
        proc = await asyncio.create_subprocess_exec(
            "systemctl",
            "--user",
            "start",
            "memex-vllm",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            log.info("vllm.restart.via_systemctl")
            return
        # systemctl exists but the unit isn't installed → fall through.
        log.info(
            "vllm.restart.systemctl_failed_fallback_to_script",
            stderr=(stderr or b"").decode("utf-8", errors="replace")[:200],
        )

    script = scripts_dir / "serve-vllm.sh"
    if not script.exists():
        raise ConfigurationError(
            f"vLLM restart fallback failed: {script} not found. "
            "Set MEMEX_SCRIPTS_DIR or restart manually.",
            context={"script": str(script)},
        )
    proc = await asyncio.create_subprocess_exec(
        "nohup",
        str(script),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        stdin=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    log.info("vllm.restart.via_script", pid=proc.pid)


# Bounded restart-retry budget for `pause_vllm_for_gpu` (the dynamic-VRAM-manager reliability fix).
_RESTART_ATTEMPTS = 2  # re-pkill + GPU-settle on the 2nd attempt (the script fallback isn't idempotent)
_RESTART_SETTLE_S = 5  # wait for VRAM release before re-launching (the contention failure is an EXIT)
_RESTART_POLL_S = 120  # cold-start budget per attempt (model materialisation + CUDA-graph compile)


@asynccontextmanager
async def pause_vllm_for_gpu() -> AsyncGenerator[None]:
    """Pause-and-restart vLLM around a GPU-heavy parse OR index pass.

    Callers: the chart-OCR pass (P3.3), the VLM escalation path
    (low-confidence/scanned/diagram pages → Qwen2.5-VL), and the CLI
    `ingest` chain — which holds it across parse + index so the embedder
    isn't starved by a co-resident vLLM. On the 12 GB reference rig
    vLLM's ~8.5 GB resident footprint plus the embedder/reranker/parse
    model doesn't fit. Stopping vLLM frees the budget; restarting after
    keeps the user-facing inference daemon alive in the long run. A
    GPU-model caller MUST `registry.unload(...)` its model inside the
    context (before the `finally` restart) so the VRAM is actually free
    when vLLM comes back. Nests safely: an inner use is a no-op when an
    outer one already paused vLLM (it sees vLLM unreachable, yields
    without pausing, and skips the restart).

    The restart is in `finally` so a parse crash doesn't strand the
    user without inference. The restart failure (rare) logs at ERROR
    but doesn't propagate — the parse itself succeeded; vLLM
    unavailability is a follow-on issue the user can address.

    No-op when vLLM is not reachable at the start of the context
    (e.g., the user is running parse with vLLM intentionally off).
    """
    log = logger.bind(component="parse.pause_vllm")
    settings = get_settings()
    base_url = settings.inference.base_url

    was_running = await _vllm_reachable(base_url)
    if not was_running:
        log.info("vllm.not_running.skip_pause")
        yield
        return

    log.info("vllm.pause.start")
    await _vllm_pkill()
    # Wait for the port to actually be free. ~15s is a generous budget;
    # vLLM typically exits in 2-5s on SIGTERM.
    for _ in range(15):
        if not await _vllm_reachable(base_url, timeout_s=1.0):
            break
        await asyncio.sleep(1.0)
    log.info("vllm.paused")

    try:
        yield
    finally:
        # BOUNDED RESTART RETRY (the load-bearing reliability fix): the single 120s attempt used to log
        # one `vllm.restart.timeout` and leave the 4B DOWN — which broke the next `/ask` (hit live this
        # session after a keyframe run). Now N attempts; from the 2nd on, re-`_vllm_pkill` (the
        # `serve-vllm.sh` fallback is NOT idempotent — a naive re-issue double-spawns a 2nd vLLM racing
        # the port) + a GPU-settle wait (the REAL fix — a GPU-contention failure makes vLLM EXIT, so
        # re-issuing into a still-occupied card just re-fails; waiting for VRAM release is what lets the
        # next launch fit). NB: log only — NEVER `raise`/`return` out of this `finally` (B012) — that
        # would suppress a parse-body exception.
        scripts_dir = _detect_scripts_dir()
        restarted = False
        for attempt in range(1, _RESTART_ATTEMPTS + 1):
            log.info("vllm.restart.start", attempt=attempt)
            try:
                if attempt > 1:
                    await _vllm_pkill()  # clear any half-started/stale vLLM before re-issuing
                    await asyncio.sleep(_RESTART_SETTLE_S)  # let the GPU release before re-launch
                await _vllm_restart(scripts_dir)
                for _ in range(_RESTART_POLL_S):
                    if await _vllm_reachable(base_url, timeout_s=1.0):
                        log.info("vllm.restarted", attempt=attempt)
                        restarted = True
                        break
                    await asyncio.sleep(1.0)
            except Exception as e:
                log.error("vllm.restart.failed", attempt=attempt, error=str(e))
            if restarted:
                break
        if not restarted:
            log.error(
                "vllm.restart.exhausted",
                attempts=_RESTART_ATTEMPTS,
                base_url=base_url,
                fix="the next /ask self-heals, or run `memex daemon restart`",
            )


def _detect_scripts_dir() -> Path:
    """Best-effort: find the repo's `scripts/` directory for the vLLM
    restart fallback path. Honours `MEMEX_SCRIPTS_DIR` env var first.
    """
    import os

    env = os.environ.get("MEMEX_SCRIPTS_DIR")
    if env:
        return Path(env)
    # Walk up from this file looking for a `.git` sibling (the project
    # root).
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        scripts = parent / "scripts"
        if scripts.is_dir() and (parent / ".git").exists():
            return scripts
    # Fallback — cwd/scripts. May not exist; caller raises if so.
    return Path.cwd() / "scripts"


def _figures_for_chart_ocr(
    figures: list[FigureMetadata], decisions: list[PageDecision]
) -> list[FigureMetadata]:
    """Figures eligible for the chart-OCR stitch: those NOT on a VLM-escalated
    page.

    A VLM-escalated page's markdown is replaced WHOLESALE by the VLM
    transcription, which carries no `<!-- image -->` placeholder (the prompt
    transcribes figures as text + `_strip_image_links` removes any). So a figure
    on an escalated page has no placeholder left to stitch into — extracting it
    anyway makes the extraction count exceed the surviving-placeholder count and
    trips `stitch_count_mismatch`, which aborts the WHOLE stitch and silently
    drops EVERY chart block (observed on the chart guide: 18 figures vs 12
    surviving placeholders → all chart content lost, chart-types 5→3 ANS). The
    VLM already transcribed those pages, so skip their figures; the rest realign
    1:1 with the surviving placeholders in document order. This is also what
    makes the chart-exclusion escalation arm coherent — charts route to
    chart-OCR, diagrams to the VLM, and the two passes no longer collide at
    stitch time."""
    escalated = {d.page for d in decisions if d.engine == "vlm"}
    return [f for f in figures if f.page_no not in escalated]


def _stitch_chart_extractions(  # pyright: ignore[reportUnusedFunction]  # retained as the byte-equality REFERENCE for the golden test (reattach must reproduce its output); not on the vault-write path since audit-10
    conversion: DoclingConversion,
    extractions: list[ChartOCROutput | Exception],
) -> DoclingConversion:
    """Replace each `<!-- image -->` placeholder with the placeholder
    plus a `[chart-extracted]...[/chart-extracted]` block carrying the
    DePlot output for that figure.

    Iterates placeholders from last to first so insertions don't shift
    the offsets of unprocessed positions. Skips extractions that are
    exceptions or empty strings — the placeholder stays unchanged
    (the agent still sees `<!-- image -->` but no extracted data).

    On a count mismatch between placeholders and extractions, logs a
    warning and returns the conversion unchanged. The mismatch could
    happen if Docling emitted placeholders for non-figure sources
    (rare; some equation renderers), in which case alignment isn't
    reliable and we prefer no-stitch over wrong-stitch.
    """
    log = logger.bind(component="chart_ocr.stitch")
    placeholders = list(IMAGE_PLACEHOLDER_RE.finditer(conversion.markdown))
    if len(placeholders) != len(extractions):
        log.warning(
            "stitch_count_mismatch",
            placeholders=len(placeholders),
            extractions=len(extractions),
        )
        return conversion

    new_markdown = conversion.markdown
    for placeholder, extraction in reversed(list(zip(placeholders, extractions, strict=True))):
        if isinstance(extraction, Exception):
            continue
        text = extraction.markdown.strip()
        if not text:
            continue
        start, end = placeholder.span()
        new_markdown = (
            new_markdown[:start]
            + placeholder.group(0)
            + "\n\n[chart-extracted]\n"
            + text
            + "\n[/chart-extracted]"
            + new_markdown[end:]
        )

    return conversion.model_copy(update={"markdown": new_markdown})


def build_chart_extractions(
    conversion: DoclingConversion,
    extractions: list[ChartOCROutput | Exception],
) -> list[ChartExtraction]:
    """The sidecar producer (audit-10 follow-on): the alignment/skip logic of
    `_stitch_chart_extractions`, but RECORDING each surviving chart block as a
    `ChartExtraction(placeholder_index, markdown)` instead of stitching it into the body.

    The canonical `.md` then stays content-only (the `<!-- image -->` placeholders are kept,
    NOT replaced by blocks); the blocks are persisted on `ParseStage.chart_extractions` and
    re-attached at index time (`core/text.reattach_chart_extractions`), which reproduces the
    historical stitched body byte-for-byte. Same skips as the old stitch (count-mismatch abort →
    `[]`; exception/empty extractions emit no entry), and `placeholder_index` is the forward
    `enumerate` ordinal of the placeholder — identical to the slot the old stitch inserted into,
    so the re-attach is byte-equal."""
    placeholders = list(IMAGE_PLACEHOLDER_RE.finditer(conversion.markdown))
    if len(placeholders) != len(extractions):
        logger.bind(component="chart_ocr.stitch").warning(
            "stitch_count_mismatch",
            placeholders=len(placeholders),
            extractions=len(extractions),
        )
        return []
    out: list[ChartExtraction] = []
    for i, extraction in enumerate(extractions):
        if isinstance(extraction, Exception):
            continue
        text = extraction.markdown.strip()
        if not text:
            continue
        out.append(ChartExtraction(placeholder_index=i, markdown=text))
    return out


# A dot-leader run (≥4 dots) + the page number that trails it — the TOC / List-of-Figures /
# List-of-Tables pagination artifact (`Introduction ......... 1`), including inside GFM table
# cells. The leading `[ \t]*` eats the space before the dots so `Introduction ... 1` → `Introduction`.
_TOC_LEADER_RE: Final[re.Pattern[str]] = re.compile(r"[ \t]*\.{4,}[ \t]*\d*")
_FENCE_LINE_RE: Final[re.Pattern[str]] = re.compile(r"^\s*(?:```|~~~)")


# ----- Markdown-fence un-wrap (audit-10 step 6b, W14) -----
#
# The parsers (and the per-page VLM, beyond W5's whole-page wrapper) sometimes wrap a markdown
# REGION — headings + prose — in a ```markdown / ```md fenced code block MID-document, trapping its
# headings/prose as inert code (the chunker + the fence-aware finalize passes treat a fenced region
# as code, so the headings are invisible and grounding degrades — the W5 failure mode, here mid-doc).
# A ```markdown / ```md language tag is by construction a "this is markdown" signal — measured prose,
# code-ratio 0 across all 73 such vault blocks — so un-fencing them is FALSE-POSITIVE-FREE. We
# ABSTAIN on any such block that contains a NESTED fence line (a real ```text/```python the VLM
# embedded): un-fencing it would splice an UNBALANCED inner fence into the body. The riskier
# BARE-tagged heading-block un-fence — which must disentangle a heading mis-fenced WITH real CLI/code
# — is DEFERRED (an FP analysis rated it 8/10 real-code-corruption risk; tracked in docs/ROADMAP.md),
# as is the pull-quote→blockquote sub-rule. Engine-agnostic, deterministic, idempotent.
_FENCE_MD_OPEN_RE: Final[re.Pattern[str]] = re.compile(r"^[ \t]*(?:```|~~~)(?:markdown|md)[ \t]*$", re.IGNORECASE)
_FENCE_BARE_CLOSE_RE: Final[re.Pattern[str]] = re.compile(r"^[ \t]*(?:```|~~~)[ \t]*$")


def unfence_markdown_blocks(markdown: str) -> str:
    """Un-fence a mid-document ```markdown / ```md block whose content is plain markdown (W14).

    Drops the opening ```markdown / closing ``` lines and splices the inner lines back VERBATIM at
    the same position, so the trapped headings + prose become top-level (and visible to the
    downstream chunker + fence-aware finalize passes). ABSTAINS — leaves the block fenced — when it
    has no matching bare close, OR contains a NESTED fence line (un-fencing would expose an
    unbalanced inner ```text/```python the VLM embedded). Only an explicitly markdown/md-LABELLED
    fence is touched; a real-code fence (any other / no language tag) is never considered. Pure-sync;
    fast-path no-op when no markdown fence is present; idempotent (the un-fenced content has no outer
    fence left to match)."""
    if "`" not in markdown and "~" not in markdown:
        return markdown
    lines = markdown.split("\n")
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        if _FENCE_MD_OPEN_RE.match(lines[i]):
            j = i + 1
            nested = False
            while j < n and not _FENCE_BARE_CLOSE_RE.match(lines[j]):
                if _FENCE_LINE_RE.match(lines[j]):  # an inner ```/~~~ line → nested code, abstain
                    nested = True
                j += 1
            if j < n and not nested:  # clean markdown block with a bare close → un-fence
                out.extend(lines[i + 1 : j])
                i = j + 1
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _collapse_toc_leaders(markdown: str) -> str:
    """Strip dot-leader + trailing-page-number pagination artifacts (audit-10 step 2c). Skips
    fenced code (a literal `....` there is content). Pure-sync; no-op when no leader is present."""
    if "...." not in markdown:  # fast path — the overwhelming majority of docs
        return markdown
    out: list[str] = []
    in_fence = False
    for line in markdown.split("\n"):
        if _FENCE_LINE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
        else:
            out.append(line if in_fence else _TOC_LEADER_RE.sub("", line))
    return "\n".join(out)


# ----- Layout-table demotion (audit-10 step 5, W11) -----
#
# Both engines recover a TABLE from bbox/line geometry, which mis-fires on layout
# graphics: a single-column list (a metric list, a references list, a symptom
# list), an INFOGRAPHIC whose bullets each land in their own 1-cell row, or a
# bullet list mis-grouped under multi-column headers all serialize as GFM tables.
# Left in the `.md` they read as garbled "tables" (raw view) and — worse — the
# index-time linearizer would emit NONSENSE KV (`References=S62162 …`,
# `RISK OVERSIGHT AT NVIDIA=- Business model …`) that pollutes retrieval. The
# `is_layout_table` predicate (shared with the linearizer + table-store SKIP
# paths) decides "not a 2-D data table"; here, at the SOURCE, we re-render the
# flagged block as plain markdown bullets so the content survives but the
# spurious table structure is gone. Engine-agnostic (runs in `_finalize_body` on
# both worker outputs), deterministic, idempotent (the bullet output has no
# pipe-rows, so a re-run finds no GFM table to touch). Only LAYOUT blocks are
# rewritten — a real data table (incl. one Docling under-filled) is left verbatim.
# A leading markdown bullet marker on a cell (`- Business model …`) — stripped so
# the re-render doesn't emit a doubled `- - …`. Only a list-bullet marker, never a
# leading `*emphasis*` (the `*` must be followed by whitespace to be a marker).
_BULLET_PREFIX_RE: Final[re.Pattern[str]] = re.compile(r"^(?:[-+]|\*(?=\s)|•)\s+")


def _layout_table_to_bullets(block: str) -> str:
    """Render a flagged layout block as deterministic markdown bullets.

    Emits one `- {cell}` line per NON-EMPTY cell of *block*, in document order
    (header cells first, then each data-row cell) via `table_cell_lines` — which
    keeps the cell text RAW (whitespace-collapsed only), so inline markdown like
    `**Board of Directors**` stays bold-balanced (the KV-side footnote strip would
    break it). A cell that is ALREADY a markdown bullet (`- Business model …`) has
    its existing marker normalized so there's no `- - …` doubling. The block
    always renders to >=1 bullet (a layout table reaching here had >=1 non-empty
    cell). The trailing newline is the caller's concern.
    """
    return "\n".join(f"- {_BULLET_PREFIX_RE.sub('', cell)}" for cell in table_cell_lines(block))


def demote_layout_tables(markdown: str) -> str:
    """Re-render layout-graphic GFM tables (audit-10 W11) as markdown bullets.

    For each GFM table block, parse it (`parse_gfm_table` — the shared core) and,
    if `is_layout_table` flags it as a layout graphic / infographic / list rather
    than a 2-D data table, replace the raw block with `_layout_table_to_bullets`.
    A block that doesn't parse, or that is a real data table (incl. an under-
    filled one), is left BYTE-IDENTICAL. Engine-agnostic, deterministic, and
    idempotent (the bullet output carries no pipe-rows). Pure-sync; no-op when the
    body has no GFM table.

    Fence-unaware by design, mirroring the index-time `linearize_gfm_tables` that
    re-derives KV from this same body — so the SET of blocks treated as tables is
    identical across the parse-finalize and the index stages, and a block this
    pass demotes is exactly the block the linearizer would otherwise mis-KV.
    """
    out: list[str] = []
    cursor = 0
    for m in GFM_TABLE_RE.finditer(markdown):
        parsed = parse_gfm_table(m.group(0))
        if parsed is None or not is_layout_table(*parsed):
            continue  # leave real tables (and non-parsing blocks) verbatim
        out.append(markdown[cursor : m.start()])
        out.append(_layout_table_to_bullets(m.group(0)))
        cursor = m.end()
        # `GFM_TABLE_RE` consumes the trailing newline of the final data row when
        # one is present (EOF-no-newline is the only exception); the bullet
        # rendering above drops it, so re-add a single newline to keep the bullet
        # block separated from whatever follows — except at EOF, where the table
        # ended without one.
        if markdown[m.end() - 1 : m.end()] == "\n":
            out.append("\n")
    out.append(markdown[cursor:])
    return "".join(out)


# ----- Consecutive duplicate collapse (audit-10 step 6a, W13) -----
#
# A slide deck with N animation steps is transcribed as N near-identical slides, and a figure at
# a page seam can be transcribed twice — both produce CONSECUTIVE duplicate blocks (a slide title
# re-emitted, a bullet / diagram-transcription repeated back-to-back). RAW EQUALITY + STRICT
# ADJACENCY (window 1) is the ONLY false-positive-free setting: a ratio / token-overlap threshold
# collapses PARALLEL DATA that shares a template (different IPs, footnote numbers, "5 GHz -1" vs
# "-2", precision-table rows) — verified unsafe at every value tested. So we drop only a block
# whose normalized text is EXACTLY equal to the immediately-preceding KEPT block. Scattered
# legitimate repeats (the 10-K's "vote required" ×3 per proposal, the ×21 running header) are NOT
# consecutive → untouched (the 10-K collapses 0). Excluded blocks — the load-bearing
# `<!-- image -->` marker (D2 / audit #3), bare PictureClassifier labels (W3 residue), and
# box-drawing diagram connectors — are KEPT but skipped for adjacency, so two identical slide
# titles separated only by image+Logo noise are still adjacent. Headings compare
# level-insensitively (keep the SHALLOWER). Engine-agnostic, deterministic, idempotent.
# (The riskier near-duplicate SECTION collapse — animation supersets/reorders via token-overlap —
# is DEFERRED: the FP sweep showed every ratio threshold reintroduces parallel-data content loss;
# tracked in docs/ROADMAP.md as a W13 follow-up.)
_DEDUP_FURNITURE_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:Logo|Line chart|Bar chart|Pie chart|Icon|Music|Screenshot from computer"
    r"|Engineering drawing|Flow ?chart|Picture|Photograph|Table|Image)$",
    re.IGNORECASE,
)
_DEDUP_BOXART_CHARS: Final[frozenset[str]] = frozenset("│─┌┐└┘├┤┬┴┼╴╵╶╷▼▲◄►←→↑↓")
_DEDUP_WS_RE: Final[re.Pattern[str]] = re.compile(r"\s+")


def _dedup_is_boxart(text: str) -> bool:
    """A box-drawing / arrow diagram block (a connector run) — a DIAGRAM unit, never a dup target
    (the guidelines flowchart's identical `│   │` connector rows each link a different node pair)."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    box = sum(1 for c in chars if c in _DEDUP_BOXART_CHARS)
    return box >= 2 and box / len(chars) >= 0.3


def _dedup_is_excluded(block_lines: list[str]) -> bool:
    """A block kept verbatim but NOT counted toward adjacency and never collapsed: a block whose
    every non-blank line is the image placeholder marker or a bare PictureClassifier label, or a
    box-art connector block."""
    if _dedup_is_boxart("\n".join(block_lines)):
        return True
    nonblank = [ln.strip() for ln in block_lines if ln.strip()]
    return bool(nonblank) and all(
        IMAGE_PLACEHOLDER_RE.fullmatch(ln) or _DEDUP_FURNITURE_RE.match(ln) for ln in nonblank
    )


def _dedup_key(block_lines: list[str]) -> tuple[str, int]:
    """The raw-equality comparison key + heading level (0 = not a heading) for a block.

    A single-line ATX heading is keyed LEVEL-INSENSITIVELY (strip the `#`-run, drop backslash
    escapes like `single\\_program.cu`, whitespace-collapse, casefold) so a same-title-different-
    level animation seam collapses; its level is returned so the caller keeps the shallower. Any
    other block is keyed by a plain whitespace-collapse of its full text (no casefold — exact for
    prose / code), so only a byte-faithful re-emission matches."""
    if len(block_lines) == 1:
        m = _HEADING_RE.match(block_lines[0])
        if m:
            key = _DEDUP_WS_RE.sub(" ", m.group(2).replace("\\", "")).strip().casefold()
            return key, len(m.group(1))
    return _DEDUP_WS_RE.sub(" ", "\n".join(block_lines)).strip(), 0


def _dedup_segment(lines: list[str]) -> list[tuple[int, int]]:
    """Segment the body into block line-ranges `[start, end)` (blank lines are separators, not in
    any block). A fenced ```/~~~ region is ONE block (inner blanks inert); a heading line is its
    own block; everything else is a maximal run of non-blank, non-heading, non-fence lines."""
    blocks: list[tuple[int, int]] = []
    i, n = 0, len(lines)
    while i < n:
        if not lines[i].strip():
            i += 1
            continue
        if _FENCE_LINE_RE.match(lines[i]):
            start = i
            i += 1
            while i < n and not _FENCE_LINE_RE.match(lines[i]):
                i += 1
            if i < n:
                i += 1  # include the closing fence line
            blocks.append((start, i))
        elif _HEADING_RE.match(lines[i]):
            blocks.append((i, i + 1))
            i += 1
        else:
            start = i
            while (
                i < n
                and lines[i].strip()
                and not _HEADING_RE.match(lines[i])
                and not _FENCE_LINE_RE.match(lines[i])
            ):
                i += 1
            blocks.append((start, i))
    return blocks


def collapse_consecutive_duplicates(markdown: str) -> str:
    """Drop a block that is an EXACT (whitespace-normalized) re-emission of the immediately
    preceding KEPT block — animation slide-build + figure/page-seam double-transcription (W13).

    Excluded blocks (image markers, PictureClassifier labels, box-art) are kept but don't count
    toward adjacency. A run of >2 identical blocks collapses to one. Two adjacent same-text
    headings at different levels keep the shallower. Pure, deterministic, idempotent (the kept
    survivor is not itself a dup, so a re-run drops nothing). No-op when there is nothing to
    collapse."""
    lines = markdown.split("\n")
    blocks = _dedup_segment(lines)
    drop: set[int] = set()
    prev: tuple[int, str, int] | None = None  # (block_index, key, heading_level)
    for bi, (s, e) in enumerate(blocks):
        block_lines = lines[s:e]
        if _dedup_is_excluded(block_lines):
            continue  # kept verbatim; does not advance adjacency
        key, level = _dedup_key(block_lines)
        if prev is not None and key == prev[1]:
            # A consecutive exact duplicate → drop it; keep `prev`. If both are headings and this
            # one is SHALLOWER, rewrite the survivor's heading line to the shallower level first.
            if level and prev[2] and level < prev[2]:
                lines[blocks[prev[0]][0]] = block_lines[0]
                prev = (prev[0], key, level)
            drop.add(bi)
        else:
            prev = (bi, key, level)
    if not drop:
        return markdown
    keep = [True] * len(lines)
    for bi in drop:
        s, e = blocks[bi]
        for k in range(s, e):
            keep[k] = False
        j = s - 1  # also remove ONE immediately-preceding blank separator
        if j >= 0 and not lines[j].strip():
            keep[j] = False
    return "\n".join(ln for k, ln in enumerate(lines) if keep[k])


# ----- Heading-hierarchy normalizer (audit-10 step 3, W2/W15) -----
#
# Both engines recover a heading's level from FONT SIZE (PyMuPDF span size, Docling bbox
# height), which is noisy: born-digital standards whose subsections share the body's heading
# font collapse to a flat wall of H2 (NIST: 82×H2), and a dense doc spreads near-continuous
# heights into 5 tiers that bottom out at H6. When a heading carries a SECTION NUMBER, that
# number is an AUTHORITATIVE, engine-independent depth signal — `1` → a top section, `1.1` →
# one level deeper, `1.1.1` → deeper still — so we override the font-derived level with the
# number's depth. Unnumbered headings keep their engine level, a masthead title with no H1 is
# promoted, and a final monotonic-nesting clamp forbids a level skipping more than one deeper
# than its predecessor (an `H2 → H5` jump becomes `H2 → H3`). Engine-agnostic, so it fixes
# BOTH worker outputs from one place; it ONLY rewrites the `#`-count, never heading text.
_HEADING_RE: Final[re.Pattern[str]] = re.compile(r"^(#{1,6})[ \t]+(.*)$")
_HEADING_BOLD_WRAP_RE: Final[re.Pattern[str]] = re.compile(r"^\*\*(.*?)\*\*\s*$")
# A leading section number: 1–2 digits per dot-group (so a 4-digit YEAR like "2023" is NOT
# mistaken for a section), an optional trailing `.`/`)`, then whitespace. "1 ", "1.2 ",
# "1.2.3. ", "4) " all match; "2023 Results" does not.
_SECTION_NUM_RE: Final[re.Pattern[str]] = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)[.)]?\s")
_ITEM_HEADING_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:Item|Section|Part)\s+\d+[A-Za-z]?[.:)]", re.IGNORECASE
)
_APPENDIX_HEADING_RE: Final[re.Pattern[str]] = re.compile(r"^Appendix\s+[A-Z0-9]", re.IGNORECASE)


def _heading_inner_text(text: str) -> str:
    """The heading's text with a single surrounding bold wrap removed, for number detection
    (both workers emit `## **1.1 History**`). Only unwraps a wrap spanning the WHOLE line."""
    m = _HEADING_BOLD_WRAP_RE.match(text.strip())
    return m.group(1).strip() if m else text.strip()


def _authoritative_heading_level(inner: str) -> int | None:
    """The level a heading's own text DICTATES, independent of font size — or None to keep the
    engine level. A section number sets depth = dot-groups + 1 (`1`→H2, `1.1`→H3, `1.1.1`→H4);
    `Item N`/`Section N`/`Part N`/`Appendix X` labels anchor at H2. Capped at H6."""
    m = _SECTION_NUM_RE.match(inner)
    if m:
        return min(1 + len(m.group(1).split(".")), 6)
    if _ITEM_HEADING_RE.match(inner) or _APPENDIX_HEADING_RE.match(inner):
        return 2
    return None


def normalize_heading_levels(markdown: str) -> str:
    """Re-derive heading `#`-levels from section-number depth + a monotonic-nesting guard.

    Engine-agnostic (runs in `_finalize_body` on both PyMuPDF and Docling output). For each
    `#`-heading: use the level its SECTION NUMBER dictates if it has one, else keep the
    engine-recovered level. Then, if the doc has no H1 and its first heading is an unnumbered
    masthead, promote that first heading to H1. Finally clamp the sequence so no heading nests
    more than one level below its predecessor. Fence-aware (a `# ` inside ```code``` is inert).
    Only the hash count changes — heading TEXT (incl. any bold wrap) is preserved verbatim.
    """
    lines = markdown.split("\n")
    # Pass 1 — locate headings (fence-aware), recording their line index, current level, and
    # the level their own text dictates.
    parsed: list[tuple[int, int, int | None]] = []  # (line_idx, current_level, auth_level)
    in_fence = False
    for i, line in enumerate(lines):
        if _FENCE_LINE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING_RE.match(line)
        if not m:
            continue
        current = len(m.group(1))
        auth = _authoritative_heading_level(_heading_inner_text(m.group(2)))
        parsed.append((i, current, auth))

    if not parsed:
        return markdown

    targets = [auth if auth is not None else current for _, current, auth in parsed]
    # Masthead → H1: a doc whose headings never reach H1 and whose FIRST heading is an
    # unnumbered title (no authoritative level) gets that title promoted, so the tree has a root.
    if not any(current == 1 for _, current, _ in parsed) and parsed[0][2] is None:
        targets[0] = 1

    # Monotonic-nesting clamp: forbid descending more than one level at a step (the first
    # heading is never clamped — a tree may legitimately start at H2).
    prev = 0
    for k, lvl in enumerate(targets):
        if prev and lvl > prev + 1:
            lvl = prev + 1
        lvl = max(1, min(lvl, 6))
        targets[k] = lvl
        prev = lvl

    for (line_idx, _current, _auth), lvl in zip(parsed, targets, strict=True):
        text = _HEADING_RE.match(lines[line_idx]).group(2)  # type: ignore[union-attr]  # matched in pass 1
        lines[line_idx] = "#" * lvl + " " + text
    return "\n".join(lines)


def _finalize_body(markdown: str) -> str:
    """Engine-agnostic post-parse finalize of the body that is WRITTEN TO THE VAULT.

    **The canonical `.md` is content-only (audit-10, 2026-05-30).** The `[table-rows]`
    KV linearization (Table-RAG Phase 1) is a retrieval aid, NOT document content, so it
    is no longer written into the source-of-truth `.md` — it would pollute the raw view
    and the embedding input (a table is otherwise encoded twice). It is re-derived at
    INDEX time instead (`index/pipeline.py` runs `linearize_gfm_tables` on the body before
    chunking), which is retrieval-NEUTRAL by construction: linearizing the clean body
    reproduces the exact pre-split input the chunker used to see, so chunk_ids are stable
    and no re-embed is needed. See `docs/audits/10-raw-md-output-audit.md` (W1).

    The engine-agnostic content scrubbers live here (audit-10 step 2+): collapse TOC dot-leader
    pagination artifacts, re-render layout-graphic GFM tables (infographics / single-column
    lists Docling/PyMuPDF mis-detect as tables, W11) as bullets, then normalize the heading
    hierarchy (section-number depth + masthead promotion + monotonic-nesting guard). The result
    is the bytes written to disk, so EVERY consumer of the parsed body (the vault `body=`, the
    `_bootstrap_ref` content hash, and the `markdown_bytes` manifest/log count) is threaded
    from this one value. The layout-table demotion runs BEFORE the heading normalizer (it only
    rewrites GFM blocks, never touches `#`-headings) and at the SOURCE here it also clears the
    block from the index-time `linearize_gfm_tables` / `extract_tables` scan — so no nonsense
    `[table-rows]` KV nor `tables.sqlite` row is derived for it.
    """
    return normalize_heading_levels(
        collapse_consecutive_duplicates(
            demote_layout_tables(unfence_markdown_blocks(_collapse_toc_leaders(markdown)))
        )
    )


async def _parse_with_docling(
    vault_path: Path,
    doc_id: str,
    source: Path,
    *,
    force_ocr: bool | None = None,
    refresh_vlm: bool = False,
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
    # VLM-escalation transcriptions are cached per (pdf-bytes, page, model,
    # prompt) so a re-parse reuses them — the VLM is non-deterministic (see
    # vlm_cache.py). Regenerable derived state (ADR-0003), dropped by
    # `reindex --force`. Only opened when the VLM path is enabled.
    vlm_cache = (
        await VLMTranscriptionCache.open(vault_path) if not settings.parse.disable_vlm else None
    )
    try:
        pages, conversion = await _route_and_escalate(
            conversion,
            source=source,
            threshold=settings.parse.vlm_confidence_threshold,
            image_area_threshold=settings.parse.vlm_image_area_threshold,
            disable_vlm=settings.parse.disable_vlm,
            log=log,
            cache=vlm_cache,
            refresh_vlm=refresh_vlm,
            diagram_classes=settings.parse.vlm_diagram_classes,
            diagram_min_confidence=settings.parse.vlm_diagram_min_confidence,
        )
    finally:
        if vlm_cache is not None:
            await vlm_cache.close()

    # P3.3 chart-OCR pass over Docling figures (opt-in via
    # `MEMEX_PARSE__DISABLE_CHART_OCR=false`). vLLM is paused for the
    # duration so DePlot's ~2.3 GB live fits alongside embedder +
    # reranker on the 12 GB reference rig; restarted via the `finally`
    # block of the pause context manager. Skips entirely when the
    # feature is disabled OR Docling reported no figures.
    chart_ocr_count = 0
    chart_blocks: list[ChartExtraction] = []
    # Skip figures on VLM-escalated pages — their `<!-- image -->` placeholders
    # are gone (replaced by VLM prose), so chart-OCR'ing them would abort the
    # whole stitch on a count mismatch. See `_figures_for_chart_ocr`.
    chart_figures = _figures_for_chart_ocr(conversion.figures, pages)
    if not settings.parse.disable_chart_ocr and chart_figures:
        log.info(
            "chart_ocr.start",
            figure_count=len(chart_figures),
            skipped_on_escalated_pages=len(conversion.figures) - len(chart_figures),
        )
        # Cache chart-OCR per (pdf, figure, model) so a re-parse replays the
        # extraction byte-identically (chart-OCR is non-deterministic, like the
        # VLM). Regenerable derived state (ADR-0003), dropped by reindex --force.
        chart_cache = await ChartOCRCache.open(vault_path)
        try:
            async with pause_vllm_for_gpu():
                extractions = await chart_ocr_extract(
                    source_pdf=source,
                    figures=chart_figures,
                    cache=chart_cache,
                    extraction_samples=settings.parse.chart_ocr_extraction_samples,
                )
                # P3.3 v2 Session 2: unload the chart-OCR model BEFORE
                # vLLM restarts (the pause context's `finally` block).
                # The VLM-backed chart-OCR is ~5-6 GB live; vLLM is
                # ~7 GB; embedder + reranker ~2.5 GB. All four
                # resident exceeds the 12 GB rig's budget. The unload
                # call is idempotent — safe for the DePlot path too,
                # where it just frees the modest ~2.3 GB earlier than
                # GC would.
                from memex.models.registry import get_registry

                try:
                    await get_registry().unload("chart_ocr")
                except Exception as ex:
                    log.warning("chart_ocr.unload_failed", error=str(ex))
            # audit-10 follow-on: do NOT stitch the chart blocks into the body that gets
            # written to the canonical `.md` — keep `conversion` content-only (placeholders
            # intact) and record the blocks on the manifest sidecar (threaded onto ParseStage
            # below). They're re-attached at index time (`reattach_chart_extractions`),
            # reproducing the historical stitched body byte-for-byte → chunk_ids stable.
            chart_blocks = build_chart_extractions(conversion, extractions)
            chart_ocr_count = len(chart_blocks)
            log.info(
                "chart_ocr.done",
                processed=len(extractions),
                stitched=chart_ocr_count,
            )
        except ChartOCRUnavailable as e:
            # Missing transformers / pypdfium2 / torch — log and skip.
            # The parse still ships; just without chart-OCR enrichment.
            log.warning("chart_ocr.unavailable", error=str(e))
        finally:
            await chart_cache.close()

    # Table-RAG Phase 1: linearize GFM tables AFTER chart-OCR stitching (so the
    # `[chart-extracted]` blocks are already in place and the GFM tables seen
    # are the post-header-recovery ones). The finalized body is what gets
    # written; thread it to body=/_bootstrap_ref/markdown_bytes below.
    final_body = _finalize_body(conversion.markdown)

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
    existing = (
        await read_document(vault_path, doc_id)
        if (vault_path / "documents" / f"{doc_id}.md").exists()
        else None
    )
    fm = (
        existing.frontmatter
        if existing
        else Frontmatter(title=await derive_title(vault_path, doc_id))
    )
    doc = VaultDocument(
        ref=_bootstrap_ref(vault_path, doc_id, final_body),
        frontmatter=fm,
        body=final_body,
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
        chart_extractions=chart_blocks,
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
        markdown_bytes=len(final_body.encode("utf-8")),
    )


def _strip_markdown_fence_wrapper(md: str) -> str:
    """PURE: unwrap a VLM transcription that emitted its WHOLE page inside a single
    ```markdown / ```md code fence. The VLM, asked to transcribe a full page, often
    answers ```markdown\\n<the page>\\n``` — and left in place that fence makes the
    chunker + grounding treat the note's prose as a code block, which degrades
    answerability (validated on the handwritten scan corpus: 'is C++ compiled?' /
    'compilation stages?' false-refused until the wrapper was removed). Only an
    explicitly markdown-LABELLED outer fence is unwrapped, and only when its matching
    close is the LAST fenced line — a bare ``` wrapper is left alone (it may be a real
    code block), and any nested fences (e.g. an ASCII diagram) stay balanced inside."""
    lines = md.strip().splitlines()
    if len(lines) < 2 or lines[0].strip() not in ("```markdown", "```md"):
        return md
    close_idx = next((i for i in range(len(lines) - 1, 0, -1) if lines[i].strip() == "```"), None)
    if close_idx is None:
        return md
    inner = "\n".join(lines[1:close_idx]).strip()
    return inner or md


def _assemble_scan_pages(
    results: Mapping[int, DoclingPageOutput | Exception],
    page_count: int,
    *,
    engine: Literal["scan", "image"] = "scan",
) -> tuple[list[PageDecision], list[str]]:
    """PURE: turn the VLM per-page transcription results into ordered `PageDecision`s
    (engine=`engine` — "scan" for a scanned PDF, "image" for a standalone image, ADR-0020)
    + the non-empty markdown parts to stitch, in reading order. A failed/missing page →
    confidence 0 with the error in `rationale` (recorded, not silently dropped) and no
    markdown contribution. Each page's markdown has a whole-page ```markdown fence wrapper
    stripped (`_strip_markdown_fence_wrapper`)."""
    pages: list[PageDecision] = []
    parts: list[str] = []
    for page_no in range(1, page_count + 1):
        res = results.get(page_no)
        if isinstance(res, DoclingPageOutput):
            md = _strip_markdown_fence_wrapper(res.markdown)
            page_md = md if md.strip() else ""
            pages.append(
                PageDecision(
                    page=page_no,
                    engine=engine,
                    confidence=1.0,
                    rationale=f"VLM {engine}",
                    char_count=len(page_md),
                )
            )
            if page_md:
                parts.append(page_md)
        else:
            err = res if isinstance(res, Exception) else None
            pages.append(
                PageDecision(
                    page=page_no,
                    engine=engine,
                    confidence=0.0,
                    rationale=f"VLM {engine} failed: {err}" if err else f"VLM {engine}: no output",
                    char_count=0,
                )
            )
    return pages, parts


async def _parse_scan_with_vlm(
    vault_path: Path,
    doc_id: str,
    source: Path,
    *,
    engine: Literal["scan", "image"] = "scan",
    refresh_vlm: bool = False,
) -> ParseResult:
    """Parse an image-only PDF by transcribing EVERY page with the VLM (`convert_pages`),
    bypassing Docling-OCR — which is printed-text-only and crashed on image-only PDFs. The
    document-level analogue of the per-page VLM escalation; reuses the same serving + cache +
    prompt. Two entry points: a scanned/handwritten PDF (`engine="scan"`, gated by the caller —
    `_parse_pdf` only routes here when `not disable_vlm`) AND a standalone image wrapped into a
    1-page PDF (`engine="image"`, ADR-0020 — routed UNCONDITIONALLY, the audio-route precedent: an
    image has no non-VLM extraction path). `engine` tags the `PageDecision`/`ParseResult` so the
    manifest records which route ran. See `docs/specs/scan-vlm-parse.md` + `docs/specs/image-ingestion.md`."""
    import pypdfium2  # lazy — a [parse] dep, same style as the worker imports

    correlation_id = str(ulid.ULID())
    log = logger.bind(doc_id=doc_id, correlation_id=correlation_id, engine=engine)
    log.info("parse.scan.start", source=str(source))
    start = time.monotonic()

    pdf = pypdfium2.PdfDocument(str(source))
    try:
        page_count = len(pdf)
    finally:
        pdf.close()

    # Transcribe ALL pages in one VLM acquisition, orchestrator paused (nestable — the
    # CLI ingest/index already holds the pause, so this is a no-op there).
    vlm_cache = await VLMTranscriptionCache.open(vault_path)
    try:
        async with pause_vllm_for_gpu():
            results = await vlm_convert_pages(
                source_pdf=source,
                page_numbers=list(range(1, page_count + 1)),  # convert_pages is 1-based
                cache=vlm_cache,
                refresh_vlm=refresh_vlm,
            )
    finally:
        await vlm_cache.close()

    pages, parts = _assemble_scan_pages(results, page_count, engine=engine)

    if not parts:
        # Nothing transcribed off the whole scan — fail recoverably rather than write an
        # empty doc (HARD-gate-safe: the agent never fabricates from an unreadable scan).
        raise ParseConfidenceTooLow(
            "VLM transcribed no content from the scanned document.",
            context={"doc_id": doc_id, "pages": page_count},
            recoverable=True,
        )

    final_body = _finalize_body("\n\n".join(parts))
    duration_ms = int((time.monotonic() - start) * 1000)

    existing = (
        await read_document(vault_path, doc_id)
        if (vault_path / "documents" / f"{doc_id}.md").exists()
        else None
    )
    fm = (
        existing.frontmatter
        if existing
        else Frontmatter(title=await derive_title(vault_path, doc_id))
    )
    doc = VaultDocument(
        ref=_bootstrap_ref(vault_path, doc_id, final_body),
        frontmatter=fm,
        body=final_body,
        mtime_ns=0,
    )
    ref = await write_document(vault_path, doc)

    parse_stage = ParseStage(
        correlation_id=correlation_id,
        parsed_at=now_utc(),
        parser_version=_PARSER_VERSION,
        pages=pages,
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
        "parse.scan.done",
        pages=page_count,
        transcribed=len(parts),
        markdown_bytes=len(final_body.encode("utf-8")),
        duration_ms=duration_ms,
    )
    return ParseResult(
        doc_id=doc_id,
        correlation_id=correlation_id,
        engine=engine,
        pages=pages,
        markdown_bytes=len(final_body.encode("utf-8")),
    )


AUDIO_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac"}
)
# Audio-bearing VIDEO containers (ADR-0017 "class video"): routed to the SAME `_parse_audio`,
# which transcribes the AUDIO track (faster-whisper/PyAV decodes the container's audio stream;
# the visual track is ignored in v1 — the slide content comes from the companion PDF via the
# Phase-2 merge). A video with no audio track → empty transcript → recoverable refuse.
VIDEO_SUFFIXES: Final[frozenset[str]] = frozenset({".mp4", ".m4v", ".mov", ".webm", ".mkv"})
# The set the parse dispatch + the ingest acceptance key on — both audio + audio-bearing video.
MEDIA_SUFFIXES: Final[frozenset[str]] = AUDIO_SUFFIXES | VIDEO_SUFFIXES


def video_source_path(vault_path: Path, doc_id: str) -> Path | None:
    """The doc's VIDEO source file (`source.mp4`/`.mov`/…) if it has one, else `None`.

    The `link-slides --use-video` gate (ADR-0018 §13): keyframe-OCR can only decode frames from a
    VIDEO source. An audio-only doc (`.mp3` — its slides have no on-screen frames), a non-media doc,
    or a doc with no copied source returns `None` → the caller falls back to the transcript-text
    alignment. A missing asset dir is treated as "no video" here; the real "doc not indexed" error
    surfaces from `compute_alignment`."""
    try:
        source = _source_file(vault_path, doc_id)
    except VaultIntegrityError:
        return None
    if source.is_file() and source.suffix.lower() in VIDEO_SUFFIXES:
        return source
    return None


def _format_timestamp(seconds: float) -> str:
    """`mm:ss` (or `hh:mm:ss` past an hour) for a transcript segment heading."""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _assemble_transcript(asr_segments: list[ASRSegment]) -> tuple[str, list[TranscriptSegment]]:
    """PURE: turn ASR segments into the transcript body (`## [mm:ss]` blocks joined by blank
    lines) + the manifest `TranscriptSegment`s. Each segment's char-span addresses its whole
    BLOCK (the `## [mm:ss]` header + the text) in the assembled body — so a chunk that starts at
    the header (where the chunker splits sections) attributes to the segment's time range. The
    body is written VERBATIM (the route skips `_finalize_body`), so the spans are exact — they
    drive `Chunk.time_range`. The companion-deck link is document-level, not a field here (§8)."""
    blocks: list[str] = []
    segments: list[TranscriptSegment] = []
    cursor = 0
    for i, a in enumerate(asr_segments):
        block = f"## [{_format_timestamp(a.start_s)}]\n{a.text}"
        segments.append(
            TranscriptSegment(
                index=i,
                char_start=cursor,  # the block start (the `## [mm:ss]` header)
                char_end=cursor + len(block),
                start_s=a.start_s,
                end_s=a.end_s,
                language=a.language,
                confidence=a.confidence,
                rationale=a.rationale,
            )
        )
        blocks.append(block)
        cursor += len(block) + 2  # the "\n\n" delimiter between blocks
    return "\n\n".join(blocks), segments


async def _parse_audio(
    vault_path: Path, doc_id: str, source: Path, *, refresh_asr: bool = False
) -> ParseResult:
    """Parse an audio source by transcribing it to timestamped Markdown (ADR-0017). The
    document-level analogue of the scan→VLM route: `transcribe_audio` (under the parse-time GPU
    pause) → assemble `## [mm:ss]` blocks → the existing chunk/embed/answer pipeline. All-fail →
    0 segments → recoverable error (HARD-gate-safe; never fabricates from silence).

    Unlike the PDF routes it does NOT run `_finalize_body`: the transcript is already normalized
    (`normalize_transcript_text`) + structured, has none of the PDF artifacts `_finalize_body`
    targets, and its duplicate-collapse would corrupt distinct timestamped segments + drift the
    char-spans. The body is written verbatim, so the segments' char-spans stay exact."""
    correlation_id = str(ulid.ULID())
    log = logger.bind(doc_id=doc_id, correlation_id=correlation_id, engine="asr")
    log.info("parse.audio.start", source=str(source))
    start = time.monotonic()

    asr_cache = await ASRTranscriptionCache.open(vault_path)
    try:
        async with pause_vllm_for_gpu():  # nestable — the CLI's outer pause makes this a no-op
            asr_segments = await transcribe_audio(
                source=source, cache=asr_cache, refresh=refresh_asr
            )
    finally:
        await asr_cache.close()

    if not asr_segments:
        raise ParseConfidenceTooLow(
            "ASR produced no transcript from the audio source.",
            context={"doc_id": doc_id},
            recoverable=True,
        )

    final_body, segments = _assemble_transcript(asr_segments)
    duration_ms = int((time.monotonic() - start) * 1000)

    existing = (
        await read_document(vault_path, doc_id)
        if (vault_path / "documents" / f"{doc_id}.md").exists()
        else None
    )
    fm = (
        existing.frontmatter
        if existing
        else Frontmatter(title=await derive_title(vault_path, doc_id))
    )
    doc = VaultDocument(
        ref=_bootstrap_ref(vault_path, doc_id, final_body),
        frontmatter=fm,
        body=final_body,
        mtime_ns=0,
    )
    ref = await write_document(vault_path, doc)

    parse_stage = ParseStage(
        correlation_id=correlation_id,
        parsed_at=now_utc(),
        parser_version=_PARSER_VERSION,
        pages=[],  # audio has no pages; the time/char record lives in `segments`
        segments=segments,
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
        "parse.audio.done",
        segments=len(segments),
        markdown_bytes=len(final_body.encode("utf-8")),
        duration_ms=duration_ms,
    )
    return ParseResult(
        doc_id=doc_id,
        correlation_id=correlation_id,
        engine="asr",
        pages=[],
        markdown_bytes=len(final_body.encode("utf-8")),
    )


# Minimum figure area (PDF user-space sq. points) for the diagram
# classification arm — mirrors chart_ocr_backend's pre-filter so a tiny
# mis-classified badge/watermark (a 50×50 "flow_chart") doesn't trigger a
# VLM page transcription. ~140×140 pt ≈ the smallest real diagram observed.
_MIN_DIAGRAM_AREA_SQPT = 20000.0


def _diagram_pages(
    figures: list[FigureMetadata],
    *,
    classes: frozenset[str],
    min_confidence: float,
    min_area: float = _MIN_DIAGRAM_AREA_SQPT,
) -> set[int]:
    """Page numbers carrying a sized, confidently-classified diagram figure.

    These route to the VLM even when the page is not image-area-dominant:
    Docling's chart-OCR pass excludes diagram classes (no extractable
    rows+cols) and a small diagram on a text-heavy page falls under
    `vlm_image_area_threshold`, so without this arm a flow chart /
    engineering drawing / screenshot is transcribed by neither pass. The
    confidence + area gates mirror the chart-OCR pre-filter so a low-trust
    or badge-sized classification doesn't trigger a needless transcription.
    Pure + synchronous — unit-tested directly with `FigureMetadata` fakes.
    """
    pages: set[int] = set()
    if not classes:
        return pages
    for fig in figures:
        cls = fig.classification
        if cls is None or cls not in classes:
            continue
        if fig.classification_confidence < min_confidence:
            continue
        x0, y_bot, x1, y_top = fig.bbox
        if (x1 - x0) * (y_top - y_bot) < min_area:
            continue
        pages.add(fig.page_no)
    return pages


async def _route_and_escalate(
    conversion: DoclingConversion,
    *,
    source: Path,
    threshold: float,
    image_area_threshold: float,
    disable_vlm: bool,
    log: structlog.stdlib.BoundLogger,
    cache: VLMTranscriptionCache | None = None,
    refresh_vlm: bool = False,
    diagram_classes: tuple[str, ...] = (),
    diagram_min_confidence: float = 0.5,
) -> tuple[list[PageDecision], DoclingConversion]:
    """For each Docling page, decide engine routing. When `disable_vlm`
    is False AND the source is a PDF, batch every VLM-eligible page
    through a single VLM acquisition (`vlm_convert_pages`) so we pay the
    lock + future-eviction cost once per document, not once per page. A
    page is VLM-eligible when ANY of three arms fire: (1) its Docling
    confidence is below `threshold`; (2) its figures cover at least
    `image_area_threshold` of the page (catches diagram/screenshot pages
    Docling reads "confidently" while losing the figure content — the
    dominant arm, since per-page confidence is effectively always 1.0);
    or (3) it carries a figure the PictureClassifier labels as one of
    `diagram_classes` above `diagram_min_confidence` (catches a small
    flow chart / engineering drawing on a text-heavy page, which the
    image-area arm misses and the chart-OCR pass excludes by design).
    Successfully escalated pages replace Docling's per-page markdown, and
    the document-level `conversion.markdown` is re-stitched so the
    canonical write picks up the corrections.
    """
    decisions: list[PageDecision] = []
    escalated_pages: dict[int, DoclingPageOutput] = {}

    # First pass: classify pages, collect the ones to escalate.
    diagram_arm_pages = _diagram_pages(
        conversion.figures,
        classes=frozenset(diagram_classes),
        min_confidence=diagram_min_confidence,
    )
    to_escalate: list[int] = []
    page_index = {p.page: p for p in conversion.pages}
    for p in conversion.pages:
        below_conf = p.confidence < threshold
        image_dominant = p.image_fraction >= image_area_threshold
        has_diagram = p.page in diagram_arm_pages
        wants_vlm = below_conf or image_dominant or has_diagram
        if not wants_vlm or disable_vlm or source.suffix.lower() != ".pdf":
            if wants_vlm and disable_vlm:
                rationale = "VLM-eligible but VLM disabled — Docling output kept"
            elif wants_vlm:
                rationale = "VLM-eligible but source is not a PDF — Docling output kept"
            else:
                rationale = "kept Docling output (confident, not image-dominant)"
            decisions.append(
                PageDecision(
                    page=p.page,
                    engine="docling",
                    confidence=p.confidence,
                    rationale=rationale,
                )
            )
            continue
        to_escalate.append(p.page)

    # Second pass: batch VLM call. One acquisition for the lot. Either the
    # VLM loads in-process (~5-6 GB AWQ, legacy Qwen2.5-VL via
    # vlm_serving="transformers") OR a short-lived VLM vLLM is started inside
    # `vlm_convert_pages` (Qwen3-VL, ~7.4 GB — see
    # vlm_backend._serve_vlm_vllm). Either way, pause the orchestrator vLLM
    # around it (same dance as chart-OCR) so the GPU budget is free; the
    # in-process `unload("vlm")` below is an idempotent no-op for the vLLM
    # path. The pause itself is a no-op when vLLM isn't running.
    if to_escalate:
        async with pause_vllm_for_gpu():
            results = await vlm_convert_pages(
                source_pdf=source,
                page_numbers=to_escalate,
                cache=cache,
                refresh_vlm=refresh_vlm,
            )
            try:
                from memex.models.registry import get_registry

                await get_registry().unload("vlm")
            except Exception as ex:
                log.warning("parse.vlm.unload_failed", error=str(ex))
        for page_no in to_escalate:
            result = results.get(page_no)
            original = page_index[page_no]
            if isinstance(result, DoclingPageOutput):
                # W5 (audit-10 step 4): the VLM, asked to transcribe a page, often wraps its
                # WHOLE answer in a ```markdown / ```md fence — which traps the page's headings,
                # tables and prose as a code block (degrades chunking + grounding). Unwrap it at
                # the SOURCE store point — the SAME `_strip_markdown_fence_wrapper` the scan path
                # already applies (`_assemble_scan_pages`), here mirrored for the escalation path
                # (the gap that was W5). Doing it here (not at the join below) keeps the page
                # OBJECT clean too, so the `char_count` recompute below counts the unwrapped
                # length, not the ~13-char wrapper. Cache-NEUTRAL: the VLM cache still stores the
                # raw (wrapped) draft, so existing cached pages replay and get unwrapped on read —
                # a plain re-parse retires the trapped wrappers with no cache bust. Idempotent +
                # a no-op on a non-wrapped page, so it can never drop real content.
                result = result.model_copy(
                    update={"markdown": _strip_markdown_fence_wrapper(result.markdown)}
                )
                escalated_pages[page_no] = result
                via_diagram = page_no in diagram_arm_pages and not (
                    original.confidence < threshold
                    or original.image_fraction >= image_area_threshold
                )
                decisions.append(
                    PageDecision(
                        page=page_no,
                        engine="vlm",
                        confidence=result.confidence,
                        rationale=(
                            "escalated to VLM (diagram-class figure)"
                            if via_diagram
                            else "escalated to VLM (low-confidence or image-dominant page)"
                        ),
                    )
                )
                log.info(
                    "parse.vlm.escalated",
                    page=page_no,
                    docling_confidence=original.confidence,
                    image_fraction=round(original.image_fraction, 3),
                    diagram_arm=page_no in diagram_arm_pages,
                )
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
                "markdown": "\n\n".join(sp.markdown for sp in stitched_pages if sp.markdown),
            }
        )

    # Record per-page markdown length on the decisions — feeds the chunker's
    # chunk→page attribution for the webui's click-source→jump-to-PDF-page UX
    # (chunker reads `ParseStage.pages[].char_count`, computes cumulative
    # intervals, binary-searches each chunk's char_start). Computed AFTER the
    # escalation stitch above so escalated pages report the VLM-transcribed
    # length, not the original Docling output. Post-pipeline transforms
    # (`_stitch_chart_extractions`, `_finalize_body`'s table linearization)
    # may insert into / after a page's content; the page mapping is
    # navigation-grade, not citation-grade — small drift on a chunk near a
    # boundary is acceptable. Pre-existing manifests carry the default 0
    # → chunker sees no usable intervals → falls back to section-only nav.
    char_counts: dict[int, int] = {p.page: len(p.markdown) for p in conversion.pages}
    decisions = [d.model_copy(update={"char_count": char_counts.get(d.page, 0)}) for d in decisions]

    return decisions, conversion


def _is_docling_failure(exc: BaseException) -> bool:
    """Filter for the breaker — only count infra-style failures.

    Docling timeouts, unavailability, and crashes (subprocess exit
    non-zero) count toward tripping the breaker. `SandboxLoadFailed`
    and `ParseConfidenceTooLow` are not in the first tuple, so they
    return False here without needing an explicit exclusion — both are
    caller-level expected outcomes the breaker should not punish.
    """
    return isinstance(exc, (DoclingTimeout, DoclingUnavailable, DoclingCrashed))


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


async def _parse_with_pymupdf(vault_path: Path, doc_id: str, source: Path) -> _PreFilterDecision:
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
    log = logger.bind(doc_id=doc_id, correlation_id=correlation_id, engine="pymupdf")

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
        # A scan/handwriting candidate = the classifier wants OCR AND the doc is
        # image-only (≈no extractable text). That's the robust signal across every
        # scan-dominant doc_type (scan / scan-like / image-heavy / rasterised /
        # mostly-empty) WITHOUT enumerating labels, and it excludes mixed-content
        # (which has real text → Docling + per-page VLM escalation handles it better).
        is_scan = (
            classification.needs_ocr
            and conversion.signals.chars_per_page_avg < _SCAN_MAX_CHARS_PER_PAGE
        )
        return _PreFilterDecision(
            result=None,
            force_ocr_on_fallthrough=classification.needs_ocr,
            is_scan=is_scan,
        )

    # PyMuPDF wins. Write the canonical markdown, record the manifest,
    # return the ParseResult.
    duration_ms = int((time.monotonic() - start) * 1000)

    # Table-RAG Phase 1: linearize GFM tables on the PyMuPDF markdown too
    # (engine-agnostic). Thread the finalized body to all consumers below.
    final_body = _finalize_body(conversion.markdown)

    existing = (
        await read_document(vault_path, doc_id)
        if (vault_path / "documents" / f"{doc_id}.md").exists()
        else None
    )
    fm = (
        existing.frontmatter
        if existing
        else Frontmatter(title=await derive_title(vault_path, doc_id))
    )
    doc = VaultDocument(
        ref=_bootstrap_ref(vault_path, doc_id, final_body),
        frontmatter=fm,
        body=final_body,
        mtime_ns=0,
    )
    ref = await write_document(vault_path, doc)

    pages: list[PageDecision] = [
        PageDecision(
            page=p.page,
            engine="pymupdf",
            confidence=classification.confidence,
            rationale=f"pymupdf:{classification.doc_type}",
            char_count=p.char_count,
        )
        for p in conversion.pages
    ]

    parse_stage = ParseStage(
        correlation_id=correlation_id,
        parsed_at=now_utc(),
        parser_version=_PARSER_VERSION,
        pymupdf_version=conversion.pymupdf_version,
        pages=pages,
        # figure_count is the worker's <!-- image --> placeholder count (W9); it equals the
        # final_body count because `_finalize_body` never removes a placeholder (`<!-- image -->`
        # is dedup-excluded + untouched by the other scrubbers). table_count was hardcoded 0 on
        # this path (a lie — the doc has GFM tables); count them HONESTLY in the WRITTEN body,
        # AFTER `_finalize_body`'s W11 layout-table demotion, so a demoted infographic is not
        # counted (the pymupdf path runs no chart-OCR, so final_body has no [chart-extracted]
        # tables to conflate). equation_count stays 0 — born-digital equations need OCR-LaTeX
        # (deferred, ROADMAP); near-absent on the born-digital docs this path handles.
        figure_count=conversion.figure_count,
        table_count=len(GFM_TABLE_RE.findall(final_body)),
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
        markdown_bytes=len(final_body.encode("utf-8")),
        duration_ms=duration_ms,
    )
    return _PreFilterDecision(
        result=ParseResult(
            doc_id=doc_id,
            correlation_id=correlation_id,
            engine="pymupdf",
            pages=pages,
            markdown_bytes=len(final_body.encode("utf-8")),
        )
    )


async def _parse_pdf(
    vault_path: Path,
    doc_id: str,
    source: Path,
    *,
    force_docling: bool = False,
    refresh_vlm: bool = False,
) -> ParseResult:
    """Route a PDF through PyMuPDF pre-filter → Docling fallback.

    The pre-filter inspects the doc, runs the tiered classifier, and
    either wins outright (high-confidence born-digital) or falls
    through with a hint about whether Docling should force OCR on
    (mixed-content, scan-like, image-heavy).

    When `force_docling=True`, the PyMuPDF pre-filter is bypassed
    entirely and the source goes straight to Docling. Use to enable
    chart-OCR on docs the classifier would otherwise route to
    PyMuPDF (chart-OCR only fires on the Docling path).
    """
    if force_docling:
        logger.bind(doc_id=doc_id, engine="docling").info("parse.force_docling")
        return await _parse_with_docling(vault_path, doc_id, source, refresh_vlm=refresh_vlm)
    decision = await _parse_with_pymupdf(vault_path, doc_id, source)
    if decision.result is not None:
        return decision.result
    # A predominantly-image doc (scan / handwriting) routes to the VLM, which CAN read
    # handwriting + diagrams — NOT to Docling-OCR (printed-text only; crashes on image-only
    # PDFs). VLM-gated: with the VLM off, fall through to Docling-OCR (unchanged). See
    # docs/specs/scan-vlm-parse.md.
    disable_vlm = get_settings().parse.disable_vlm
    if decision.is_scan and not disable_vlm and source.suffix.lower() == ".pdf":
        return await _parse_scan_with_vlm(vault_path, doc_id, source, refresh_vlm=refresh_vlm)
    return await _parse_with_docling(
        vault_path,
        doc_id,
        source,
        force_ocr=decision.force_ocr_on_fallthrough or None,
        refresh_vlm=refresh_vlm,
    )


async def _ensure_converted_pdf(vault_path: Path, doc_id: str, source: Path) -> Path:
    """Return a cached PDF rendering of an Office/ODF source, converting on the
    first parse via headless LibreOffice (`office_convert.convert_to_pdf`).

    Cached at `documents/{doc_id}/converted.pdf` — deliberately NOT a `source.*`
    name, so `_source_file` still resolves the ORIGINAL Office file as the
    provenance source — and reused on re-parse so the PDF bytes (hence the
    content-addressed VLM / chart-OCR cache keys) stay byte-stable across runs
    (LibreOffice stamps a fresh CreationDate per conversion, so re-converting
    every parse would churn those caches).
    """
    converted = source.parent / "converted.pdf"
    if converted.is_file():
        return converted
    with tempfile.TemporaryDirectory(prefix="memex-office-") as tmp:
        produced = await convert_to_pdf(source, Path(tmp))
        shutil.move(str(produced), str(converted))
    logger.bind(doc_id=doc_id).info("office.converted_cached", path=str(converted))
    return converted


async def _ensure_converted_image_pdf(vault_path: Path, doc_id: str, source: Path) -> Path:
    """Return a cached single-page PDF wrapping a standalone image (ADR-0020).

    Mirrors `_ensure_converted_pdf` (the Office precedent): cached at
    `documents/{doc_id}/converted.pdf` — NOT a `source.*` name, so `_source_file`
    still resolves the ORIGINAL image (`.png`/`.jpg`/…) as provenance — and reused
    on re-parse so the PDF bytes (hence the content-addressed VLM cache key) stay
    byte-stable across runs. `PIL.Image.save(..., "PDF")` stamps a fresh `gmtime()`
    CreationDate into the PDF Info trailer, so re-converting every parse would churn
    the VLM cache — exactly the LibreOffice CreationDate problem caching solves.
    """
    converted = source.parent / "converted.pdf"
    if converted.is_file():
        return converted
    with tempfile.TemporaryDirectory(prefix="memex-image-") as tmp:
        produced = await convert_image_to_pdf(source, Path(tmp))
        shutil.move(str(produced), str(converted))
    logger.bind(doc_id=doc_id).info("image.converted_cached", path=str(converted))
    return converted


async def parse_document(
    doc_id: str,
    *,
    force_docling: bool | None = None,
    refresh_vlm: bool = False,
    refresh_asr: bool = False,
) -> ParseResult:
    """Parse the document with `doc_id`'s source into canonical markdown.

    `force_docling` overrides the classifier and routes the source
    straight to Docling. `None` (default) means "use the
    `ParseSettings.force_docling` value." Explicit `True`/`False`
    overrides the setting for this call. `refresh_vlm` busts this
    document's cached VLM transcriptions first, forcing a fresh draw
    (the VLM is non-deterministic; transcriptions are cached by default).

    Office/ODF sources (pptx/docx/xlsx/…) are converted to a cached PDF first
    and run through the full PDF pipeline, so their figures + diagrams flow
    through the VLM / chart-OCR passes like any PDF. Audio sources (mp3/wav/…,
    ADR-0017) route to the ASR transcription path (`refresh_asr` busts the
    cached transcription, mirroring `refresh_vlm`). Standalone images
    (png/jpg/webp/…, ADR-0020) are wrapped into a cached 1-page PDF and run
    through the scan→VLM route — VLM-mandatory (an image has no non-VLM
    extraction path, the audio-route precedent), so `disable_vlm` is bypassed.
    """
    settings = get_settings()
    effective_force = force_docling if force_docling is not None else settings.parse.force_docling
    source = _source_file(settings.vault_path, doc_id)

    if source.suffix.lower() in {".md", ".markdown"}:
        return await _passthrough_markdown(settings.vault_path, doc_id, source)

    # Audio sources AND audio-bearing video containers (ADR-0017) → the ASR route: transcribe
    # to timestamped Markdown (the video's audio track), then hand to the existing
    # chunk/embed/answer pipeline. A parse-stage perception model, off the grounded path;
    # `refresh_asr` busts this doc's cached transcription first.
    if source.suffix.lower() in MEDIA_SUFFIXES:
        return await _parse_audio(settings.vault_path, doc_id, source, refresh_asr=refresh_asr)

    # Office/ODF sources can't be rasterised by pypdfium2 (the VLM + chart-OCR
    # figure renderers are PDF-only), so convert to a cached PDF and run the
    # full PDF pipeline on it. force_docling=True so a slide-deck export reaches
    # the Docling VLM/chart-OCR diagram pass rather than the classifier routing
    # it to PyMuPDF (which has no figure-transcription stage).
    if source.suffix.lower() in OFFICE_SUFFIXES:
        source = await _ensure_converted_pdf(settings.vault_path, doc_id, source)
        return await _parse_pdf(
            settings.vault_path, doc_id, source, force_docling=True, refresh_vlm=refresh_vlm
        )

    # Standalone images (ADR-0020): wrap into a cached 1-page PDF and run the scan→VLM route.
    # An image is a one-page scan — pypdfium2 (the VLM page rasteriser) is PDF-only. Routing
    # DIRECT to `_parse_scan_with_vlm` (not `_parse_pdf`) makes the VLM mandatory by construction:
    # the route never consults `disable_vlm` (the audio-route precedent — an image has no non-VLM
    # extraction path). `engine="image"` tags the manifest so the route is auditable.
    if source.suffix.lower() in IMAGE_SUFFIXES:
        source = await _ensure_converted_image_pdf(settings.vault_path, doc_id, source)
        return await _parse_scan_with_vlm(
            settings.vault_path, doc_id, source, engine="image", refresh_vlm=refresh_vlm
        )

    if source.suffix.lower() == ".pdf":
        return await _parse_pdf(
            settings.vault_path,
            doc_id,
            source,
            force_docling=effective_force,
            refresh_vlm=refresh_vlm,
        )

    return await _parse_with_docling(settings.vault_path, doc_id, source, refresh_vlm=refresh_vlm)
