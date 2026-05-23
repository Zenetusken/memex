# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Subprocess entry-point for PyMuPDF4LLM — see GUIDELINES.md Part VI.

`parse.pymupdf_backend.convert` spawns this module via
`python -m memex.parse.pymupdf_worker <source_path>`. Running PyMuPDF
in a separate process contains its crashes: a segfault here kills only
this child, not the parent agent.

pymupdf + pymupdf4llm have incomplete type stubs. Rather than annotate
every untyped return value, this file suppresses the unknown-type
reports at the file level — consistent with the convention for
worker modules that interface with native C extensions.

**Network egress is blocked** before pymupdf is imported — see
`memex.parse.sandbox`. PyMuPDF has no business making network calls
but the sandbox is symmetric with Docling's for the same defence-in-
depth reason.

The worker reads the source path from argv, opens the PDF, collects
the rich signal set the routing classifier needs, runs
`pymupdf4llm.to_markdown`, and writes the resulting payload to stdout
as JSON.

Exit codes (mirror docling_worker):
  0 success
  2 usage error
  3 source not found
  4 pymupdf unavailable
  5 unhandled exception during extraction
  6 sandbox requested but couldn't be applied
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any

# CRITICAL: the worker's stdout IS the protocol channel — the parent
# parses it as JSON. Anything else writing to stdout (structlog
# default renderer, the sandbox.applied event, chatty third-party
# libs) corrupts the payload. Redirect logging into stderr before
# anything imports.
logging.basicConfig(stream=sys.stderr, level=logging.WARNING, force=True)
try:
    import structlog

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.KeyValueRenderer(
                key_order=["timestamp", "level", "event"]
            ),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
except ImportError:
    pass


_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_LIST_RE = re.compile(r"^(\s*[-*+]\s|\s*\d+\.\s)", re.MULTILINE)
_TABLE_RE = re.compile(r"^\|.*\|\s*$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"^```", re.MULTILINE)
_WORD_LIKE_RE = re.compile(r"\b[A-Za-z]{2,15}\b")
# pymupdf4llm emits `<br>`-joined runs inside "picture text" blocks
# (chart axis labels, screenshot annotations, etc). For the chunker
# downstream, these need to be paragraph breaks — otherwise a single
# chart's data ends up as one massive paragraph that bypasses the
# token target and produces chunks that exceed the reranker's
# attention window.
_BR_RUN_RE = re.compile(r"(?:\s*<br\s*/?>\s*){1,}")
# pymupdf4llm's "picture omitted" / "picture text boundary" markers
# are extraction-engine metadata, not document content. They bloat
# chunks with content that never aids retrieval (~23% of the markdown
# bytes on the canonical CUDA deck). Strip them in the worker; the
# content *between* the picture-text boundary markers is real text
# that PyMuPDF lifted from inside images via the PDF's text operators
# — that stays.
_PICTURE_OMITTED_RE = re.compile(
    r"^\s*\*\*==>\s*picture\s*\[[^\]]*\]\s*intentionally omitted\s*<==\*\*\s*$",
    re.MULTILINE,
)
_PICTURE_TEXT_BOUNDARY_RE = re.compile(
    r"^\s*\*\*-{3,}\s*(?:Start|End)\s+of\s+picture\s+text\s*-{3,}\*\*\s*$",
    re.MULTILINE,
)
# Strip runs of 3+ consecutive blank lines that the marker-strips leave behind.
_BLANK_LINE_RUN_RE = re.compile(r"\n{3,}")


def _normalise_breaks(text: str) -> str:
    """Convert `<br>` runs to paragraph breaks.

    pymupdf4llm renders text extracted from inside images as a single
    `<br>`-joined run (chart labels, screenshot text, diagram
    annotations). The downstream chunker uses double-newline as the
    paragraph boundary; a `<br>`-joined block of 50+ items becomes one
    paragraph that bypasses the chunker's size cap and produces chunks
    too large for the reranker's attention window. Turning each `<br>`
    run into `\\n\\n` keeps the data points addressable while letting
    the chunker bundle them into reasonable-sized windows.
    """
    if "<br" not in text:
        return text
    return _BR_RUN_RE.sub("\n\n", text)


def _strip_pymupdf_markers(text: str) -> str:
    """Tidy pymupdf4llm's structural metadata so chunks stay clean.

    Two patterns:
      - `**==> picture [W x H] intentionally omitted <==**` lines —
        pure extraction metadata describing a region PyMuPDF didn't
        render. No retrieval signal; deleted outright.
      - `**----- Start/End of picture text -----**` boundary markers —
        replaced with compact `[chart-text]` / `[/chart-text]` tags.
        The verbose form bloats chunks (~5 KB on the canonical CUDA
        deck), but the boundary signal itself is load-bearing: it
        tells the agent that the bare data points between belong to
        a chart, not body prose. Stripping the boundaries entirely
        made the assessor reject chart-grounded chunks as
        "fragmented" in the P1.6 verification run.

    Then collapse runs of 3+ blank lines that the substitutions leave behind.
    """
    if "**==>" not in text and "Start of picture text" not in text:
        return text
    text = _PICTURE_OMITTED_RE.sub("", text)
    text = _PICTURE_TEXT_BOUNDARY_RE.sub(_compact_picture_text_marker, text)
    text = _BLANK_LINE_RUN_RE.sub("\n\n", text)
    return text


def _compact_picture_text_marker(match: re.Match[str]) -> str:
    """Replace the verbose boundary marker with a compact tag.

    The original full-width banner is ~38 chars; the compact tag is
    14 chars (`[chart-text]` or `[/chart-text]`). Preserves the
    semantic signal while shedding ~3 KB on a marker-heavy deck.
    """
    return (
        "[chart-text]" if "Start" in match.group(0) else "[/chart-text]"
    )


def _clean_pymupdf_markdown(text: str) -> str:
    """Both passes in order: `<br>` normalisation, then marker stripping."""
    return _strip_pymupdf_markers(_normalise_breaks(text))


def _safe_meta(doc: Any, key: str) -> str | None:
    md = getattr(doc, "metadata", None) or {}
    val = md.get(key)
    if val in (None, "", "none"):
        return None
    return str(val)


def _detect_tagged(doc: Any) -> bool:
    """Best-effort detection of a tagged PDF (/MarkInfo /Marked true).

    Try several APIs because pymupdf's exact surface varies by
    version. Wrap each in try/except — tagged-PDF detection is a
    nice-to-have signal, not load-bearing.
    """
    for attr in ("is_pdf_tagged", "is_tagged", "has_marked"):
        v = getattr(doc, attr, None)
        if callable(v):
            try:
                return bool(v())
            except Exception:
                pass
        elif isinstance(v, bool):
            return v

    try:
        cat = doc.pdf_catalog()
        mark_info = doc.xref_get_key(cat, "MarkInfo")
        if isinstance(mark_info, tuple) and len(mark_info) >= 2 and mark_info[0] == "xref":
            xref_str = mark_info[1].split()[0]
            xref = int(xref_str)
            marked = doc.xref_get_key(xref, "Marked")
            if isinstance(marked, tuple) and len(marked) >= 2:
                return str(marked[1]).lower() == "true"
    except Exception:
        pass

    return False


def _detect_layers(doc: Any) -> bool:
    """Best-effort /OCProperties presence check."""
    try:
        cat = doc.pdf_catalog()
        oc = doc.xref_get_key(cat, "OCProperties")
        if isinstance(oc, tuple) and len(oc) >= 2:
            return oc[0] != "null"
    except Exception:
        pass
    return False


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * pct
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def _collect_signals(doc: Any, joined_markdown: str) -> dict[str, Any]:
    """Walk the document once, build the PdfSignals dict.

    All loops are O(pages) with cheap per-page work — no rendering,
    no OCR, no model calls. Sub-second on a 100-page deck.
    """
    page_count: int = int(getattr(doc, "page_count", 0) or 0)
    char_counts: list[int] = []
    image_counts: list[int] = []
    aspect_ratios: list[float] = []
    image_areas: list[float] = []
    image_heavy_pages = 0
    embedded_fonts: set[str] = set()

    raw_text_parts: list[str] = []
    for i in range(page_count):
        page: Any = doc.load_page(i)
        rect: Any = page.rect
        page_area = max(1.0, float(rect.width) * float(rect.height))
        aspect = float(rect.width) / max(1.0, float(rect.height))
        aspect_ratios.append(aspect)

        # Native text
        try:
            text_any: Any = page.get_text()
            text = text_any if isinstance(text_any, str) else ""
        except Exception:
            text = ""
        raw_text_parts.append(text)
        char_counts.append(len(text))

        # Images — count and area
        try:
            infos_any: Any = page.get_image_info()
            infos: list[Any] = list(infos_any) if infos_any else []
        except Exception:
            infos = []
        image_count = len(infos)
        image_counts.append(image_count)
        if image_count > 3:
            image_heavy_pages += 1

        page_image_area = 0.0
        for info in infos:
            bbox: Any = info.get("bbox") if isinstance(info, dict) else None
            if bbox and len(bbox) == 4:
                w = max(0.0, float(bbox[2]) - float(bbox[0]))
                h = max(0.0, float(bbox[3]) - float(bbox[1]))
                page_image_area += w * h
        # Clamp per-page image area to page area — multiple images can
        # overlap and the sum can exceed the page rect; we cap to 1.0
        # ratio so the average stays meaningful.
        image_areas.append(min(1.0, page_image_area / page_area))

        # Fonts
        try:
            fonts_any: Any = page.get_fonts()
            fonts: list[Any] = list(fonts_any) if fonts_any else []
        except Exception:
            fonts = []
        for f in fonts:
            # Tuples vary by pymupdf version: (xref, ext, type, basefont, name, encoding, ...)
            if isinstance(f, tuple) and len(f) >= 4:
                embedded_fonts.add(str(f[3]))

    raw_text = "".join(raw_text_parts)
    total_chars = len(raw_text)

    chars_avg = float(statistics.mean(char_counts)) if char_counts else 0.0
    chars_median = float(statistics.median(char_counts)) if char_counts else 0.0
    chars_p10 = _percentile([float(c) for c in char_counts], 0.10)
    chars_p90 = _percentile([float(c) for c in char_counts], 0.90)
    empty_pages = sum(1 for c in char_counts if c < 5)

    aspect_avg = float(statistics.mean(aspect_ratios)) if aspect_ratios else 1.0
    aspect_stdev = (
        float(statistics.pstdev(aspect_ratios)) if len(aspect_ratios) > 1 else 0.0
    )

    image_area_avg = float(statistics.mean(image_areas)) if image_areas else 0.0

    # Text quality
    replacement_chars = raw_text.count("�")
    word_like_tokens = len(_WORD_LIKE_RE.findall(raw_text))
    rough_token_count = max(1, len(raw_text.split()))
    whitespace_chars = sum(1 for ch in raw_text if ch.isspace())

    # Markdown structure
    has_headings = bool(_HEADING_RE.search(joined_markdown))
    has_lists = bool(_LIST_RE.search(joined_markdown))
    has_tables = bool(_TABLE_RE.search(joined_markdown))
    has_code_blocks = bool(_CODE_FENCE_RE.search(joined_markdown))

    pdf_format = _safe_meta(doc, "format")

    return {
        "creator": _safe_meta(doc, "creator"),
        "producer": _safe_meta(doc, "producer"),
        "pdf_version": pdf_format,
        "is_encrypted": bool(getattr(doc, "is_encrypted", False)),
        "is_tagged": _detect_tagged(doc),
        "has_layers": _detect_layers(doc),
        "page_count": page_count,
        "avg_aspect_ratio": aspect_avg,
        "aspect_ratio_consistent": aspect_stdev < 0.1,
        "embedded_font_count": len(embedded_fonts),
        "image_count_total": sum(image_counts),
        "image_heavy_page_fraction": (
            image_heavy_pages / page_count if page_count > 0 else 0.0
        ),
        "image_area_fraction": image_area_avg,
        "total_chars": total_chars,
        "chars_per_page_avg": chars_avg,
        "chars_per_page_median": chars_median,
        "chars_per_page_p10": chars_p10,
        "chars_per_page_p90": chars_p90,
        "empty_page_fraction": empty_pages / page_count if page_count > 0 else 0.0,
        "replacement_char_fraction": (
            replacement_chars / total_chars if total_chars > 0 else 0.0
        ),
        "word_like_token_fraction": word_like_tokens / rough_token_count,
        "unique_char_variety": len(set(raw_text)),
        "whitespace_fraction": (
            whitespace_chars / total_chars if total_chars > 0 else 0.0
        ),
        "has_headings": has_headings,
        "has_tables": has_tables,
        "has_lists": has_lists,
        "has_code_blocks": has_code_blocks,
    }


def _coerce_finite(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return x


def _convert_to_payload(source: Path) -> dict[str, Any]:
    """Open the PDF, collect signals, run pymupdf4llm.to_markdown.

    Returns a JSON-serialisable dict matching the `PyMuPDFConversion`
    schema in `pymupdf_backend.py`. pymupdf + pymupdf4llm ship no
    type stubs so we treat their handles as Any — same convention as
    the docling worker.
    """
    try:
        import pymupdf  # type: ignore[import-untyped]
        import pymupdf4llm  # type: ignore[import-untyped]
    except ImportError as e:
        raise SystemExit(
            json.dumps({"error": "pymupdf_unavailable", "detail": str(e)})
        ) from e

    pymupdf_version: str | None = (
        getattr(pymupdf, "__version__", None)
        or getattr(pymupdf4llm, "__version__", None)
    )

    doc: Any = pymupdf.open(str(source))
    try:
        # page_chunks=True returns list[dict] with per-page records.
        # force_text=True ensures we still get text for pages where
        # pymupdf can't classify structure perfectly. write_images +
        # embed_images = False keeps the output pure text (we have a
        # separate VLM path for figure description if ever needed).
        chunks_raw: Any = pymupdf4llm.to_markdown(
            doc,
            page_chunks=True,
            show_progress=False,
            write_images=False,
            embed_images=False,
            force_text=True,
        )

        if isinstance(chunks_raw, str):
            # Older API path — single string, no per-page split. Wrap
            # into a single-chunk list so downstream is uniform.
            chunks: list[dict[str, Any]] = [
                {"text": _clean_pymupdf_markdown(chunks_raw), "metadata": {"page": 0}}
            ]
        else:
            chunks = []
            for c in chunks_raw:
                if not isinstance(c, dict):
                    continue
                text_val = c.get("text")
                if isinstance(text_val, str):
                    c = {**c, "text": _clean_pymupdf_markdown(text_val)}
                chunks.append(c)

        pages: list[dict[str, Any]] = []
        markdown_parts: list[str] = []
        for i, chunk in enumerate(chunks):
            page_md = str(chunk.get("text") or "")
            meta_raw = chunk.get("metadata") or {}
            meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
            page_no_raw = meta.get("page", i)
            try:
                page_no = int(page_no_raw) + (0 if "page" in meta else 1)
            except (TypeError, ValueError):
                page_no = i + 1
            page_no = max(1, page_no)

            # Per-page rect / image-count for the page record. Use the
            # actual document page (chunk indices align with doc pages
            # when page_chunks=True).
            doc_page_count: int = int(getattr(doc, "page_count", 0) or 0)
            page_idx = min(i, max(0, doc_page_count - 1))
            try:
                page_obj: Any = doc.load_page(page_idx)
                rect: Any = page_obj.rect
                aspect = float(rect.width) / max(1.0, float(rect.height))
                try:
                    images: Any = page_obj.get_images()
                    image_count = len(images)
                except Exception:
                    image_count = 0
            except Exception:
                aspect = 1.0
                image_count = 0

            pages.append(
                {
                    "page": page_no,
                    "markdown": page_md,
                    "char_count": len(page_md),
                    "image_count": image_count,
                    "aspect_ratio": _coerce_finite(aspect),
                }
            )
            markdown_parts.append(page_md)

        if not pages:
            # Defensive — pymupdf4llm should always emit at least one
            # chunk for a non-empty PDF. Fall back to all-doc text.
            try:
                doc_page_count = int(getattr(doc, "page_count", 0) or 0)
                fallback_parts: list[str] = []
                for i in range(doc_page_count):
                    page_obj_fb: Any = doc.load_page(i)
                    text_any: Any = page_obj_fb.get_text()
                    fallback_parts.append(text_any if isinstance(text_any, str) else "")
                fallback = "\n\n".join(fallback_parts)
            except Exception:
                fallback = ""
            pages.append(
                {
                    "page": 1,
                    "markdown": fallback,
                    "char_count": len(fallback),
                    "image_count": 0,
                    "aspect_ratio": 1.0,
                }
            )
            markdown_parts.append(fallback)

        joined = "\n\n".join(p for p in markdown_parts if p)
        signals = _collect_signals(doc, joined)

        return {
            "markdown": joined,
            "pages": pages,
            "pymupdf_version": pymupdf_version,
            "signals": signals,
            "figure_count": 0,
            "table_count": 0,
            "equation_count": 0,
        }
    finally:
        try:
            doc.close()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    """Subprocess entry point — invoked by
    `parse/pymupdf_backend.py::convert` via `python -m memex.parse.
    pymupdf_worker <source_path>`. Returns the conversion as JSON on
    stdout; non-zero exit code signals a crash. Runs under the
    seccomp sandbox installed before pymupdf import."""
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print(
            "usage: python -m memex.parse.pymupdf_worker <source_path>",
            file=sys.stderr,
        )
        return 2

    source = Path(args[0])
    if not source.exists():
        print(f"source file does not exist: {source}", file=sys.stderr)
        return 3

    if os.environ.get("MEMEX_PARSE_SANDBOX_NETWORK", "1") == "1":
        from memex.parse.sandbox import enable_network_block

        status, reason = enable_network_block()
        print(f"sandbox: {status} ({reason})", file=sys.stderr)
        if status == "failed":
            return 6

    try:
        payload = _convert_to_payload(source)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            return 4
        return e.code or 4
    except Exception:
        traceback.print_exc()
        return 5

    try:
        json.dump(payload, sys.stdout, allow_nan=False)
    except ValueError as e:
        print(f"json.dump failed: {e}", file=sys.stderr)
        return 5
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
