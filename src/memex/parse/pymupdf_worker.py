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

import contextlib
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
            structlog.processors.KeyValueRenderer(key_order=["timestamp", "level", "event"]),
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
# audit-10 W9: the honest content-loss visibility placeholder a picture-omitted marker is CONVERTED
# to (was deleted). Blank-line padded on BOTH sides so it is a STANDALONE paragraph regardless of
# pymupdf4llm's spacing — a figure-dense page can emit adjacent markers, and the bare form would glue
# two placeholders onto one line (`<!-- image --><!-- image -->`) or onto a following prose line,
# which breaks the per-line `<!-- image -->` recognition downstream (the W13 dedup's `_dedup_is_excluded`,
# the chunker's image-block handling). The `_BLANK_LINE_RUN_RE` collapse below then makes each its own
# paragraph. The bare `<!-- image -->` matches the Docling D2/D3 convention + `_IMAGE_PLACEHOLDER_RE`.
_IMAGE_PLACEHOLDER_SUB = "\n\n<!-- image -->\n\n"


def _looks_like_table_row(line: str) -> bool:
    """A GFM table row carries ≥2 pipe characters bordering its cells.

    Used only to decide `<br>` handling: a `<br>` inside a table row is an
    in-cell line wrap, not a paragraph boundary."""
    return line.count("|") >= 2


def _normalise_breaks(text: str) -> str:
    """Convert `<br>` runs to paragraph breaks — except inside GFM table rows.

    pymupdf4llm renders text extracted from inside images as a single
    `<br>`-joined run (chart labels, screenshot text, diagram
    annotations). The downstream chunker uses double-newline as the
    paragraph boundary; a `<br>`-joined block of 50+ items becomes one
    paragraph that bypasses the chunker's size cap and produces chunks
    too large for the reranker's attention window. Turning each `<br>`
    run into `\\n\\n` keeps the data points addressable while letting
    the chunker bundle them into reasonable-sized windows.

    **Table exception.** pymupdf4llm also emits `<br>` for in-cell line
    wraps *inside* GFM table rows (e.g. `|Gateway|99.95%|1|Brief restart
    during a<br>deploy.|`, and even a wrapped header `|<br>Service|<br>…`).
    A blind `<br>`→`\\n\\n` there shatters the row across lines and destroys
    the pipe table (breaking the chunker, table linearization, and text-to-
    SQL). Within a table row a `<br>` run collapses to a single space so the
    row stays on one line and cell content matches the source; everywhere
    else the paragraph-break behaviour is unchanged.
    """
    if "<br" not in text:
        return text
    out: list[str] = []
    for line in text.split("\n"):
        if "<br" in line and _looks_like_table_row(line):
            out.append(_BR_RUN_RE.sub(" ", line))
        else:
            out.append(line)
    # Remaining (non-table) <br> runs → paragraph breaks, as before.
    return _BR_RUN_RE.sub("\n\n", "\n".join(out))


def _strip_pymupdf_markers(text: str) -> str:
    """Tidy pymupdf4llm's structural metadata so chunks stay clean.

    Two patterns:
      - `**==> picture [W x H] intentionally omitted <==**` lines —
        pymupdf4llm's marker for a FIGURE it detected but did not render
        (a born-digital vector drawing / image). It used to be deleted
        outright, which SILENTLY DROPPED the figure AND left the manifest
        reporting 0 figures (audit-10 W9, crit: a paper that says "see
        Figure 2" with no figure). It is now CONVERTED to a `<!-- image -->`
        visibility placeholder (the Docling D2/D3 convention) so the gap is
        honest in the raw view + countable for the manifest; the figures are
        modest on the pymupdf path (gte 12, NIST 14 — figure-dense decks
        route to Docling, never here), so no per-page cap is needed.
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
    text = _PICTURE_OMITTED_RE.sub(_IMAGE_PLACEHOLDER_SUB, text)
    text = _PICTURE_TEXT_BOUNDARY_RE.sub(_compact_picture_text_marker, text)
    text = _BLANK_LINE_RUN_RE.sub("\n\n", text)
    return text


def _compact_picture_text_marker(match: re.Match[str]) -> str:
    """Replace the verbose boundary marker with a compact tag.

    The original full-width banner is ~38 chars; the compact tag is
    14 chars (`[chart-text]` or `[/chart-text]`). Preserves the
    semantic signal while shedding ~3 KB on a marker-heavy deck.
    """
    return "[chart-text]" if "Start" in match.group(0) else "[/chart-text]"


def _clean_pymupdf_markdown(text: str) -> str:
    """Both passes in order: `<br>` normalisation, then marker stripping."""
    return _strip_pymupdf_markers(_normalise_breaks(text))


# ----- Heading-level recovery -----
#
# pymupdf4llm (1.27.x) correctly DETECTS heading lines but collapses every
# one to `## ` regardless of source font size — its `IdentifyHeaders` map
# is right (e.g. {24:'# ', 18:'## ', 14:'### '}) but the markdown emitter
# ignores per-span sizes. Result: a paper's title, sections, and
# subsections all become H2, and the vault loses document hierarchy.
# Confirmed across the eval-corpus fixtures (2026-05-24). We fix the LEVEL
# here without touching pymupdf4llm's detection of WHICH lines are
# headings: re-derive each heading's level from its real font size.

_HEADING_LINE_RE = re.compile(r"^(#{1,6})[ \t]+\S")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_HEADING_EMPHASIS_RE = re.compile(r"[*`]")


def _heading_size_to_level(doc: Any) -> dict[int, int]:
    """Map rounded font size → heading level (1..6) for the whole doc.

    Body size is the most frequent span size; every distinct size larger
    than body is a heading tier, ranked descending (largest → level 1).
    Mirrors `pymupdf4llm.IdentifyHeaders` intent but stays inclusive of
    near-body heading tiers (e.g. an H4 only 2pt above body)."""
    freq: dict[int, int] = {}
    for page in doc:
        for blk in page.get_text("dict").get("blocks", []):
            for line in blk.get("lines", []):
                for sp in line.get("spans", []):
                    if str(sp.get("text", "")).strip():
                        size = round(float(sp.get("size", 0.0)))
                        freq[size] = freq.get(size, 0) + 1
    if not freq:
        return {}
    body = max(freq, key=lambda s: freq[s])
    heading_sizes = sorted((s for s in freq if s > body), reverse=True)
    return {size: min(i + 1, 6) for i, size in enumerate(heading_sizes)}


def _norm_heading(text: str) -> str:
    """Normalise heading text for matching markdown ↔ PDF spans:
    drop emphasis/code markers, collapse whitespace, lowercase."""
    return " ".join(_HEADING_EMPHASIS_RE.sub("", text).split()).lower()


# pymupdf4llm sometimes over-flags a body sentence as a heading (e.g. a short
# standalone line just before a table). `_remap_heading_levels` only demotes
# one whose font size is also body-size-or-smaller, so this prose check is a
# conservative second guard. Ported from the Docling worker's
# `_looks_like_prose_heading` (same thresholds).
_TERMINAL_PUNCT = (".", "!", "?")
_MAX_HEADING_WORDS = 15  # longer than this reads as prose regardless of punctuation
_MIN_PROSE_WORDS = 4  # a terminal-punct line needs ≥ this many words to be prose


def _looks_like_prose_heading(heading_text: str) -> bool:
    """True if a heading line's text reads as a body sentence, not a title."""
    t = _HEADING_EMPHASIS_RE.sub("", heading_text).strip()
    if not t:
        return False
    words = t.split()
    if len(words) > _MAX_HEADING_WORDS:
        return True
    return len(words) >= _MIN_PROSE_WORDS and t[-1] in _TERMINAL_PUNCT


def _remap_heading_levels(page_md: str, page: Any, size_to_level: dict[int, int]) -> str:
    """Rewrite the level of each heading line in `page_md` from the real
    font size of the matching span on `page`, and DEMOTE misdetected
    headings to paragraphs.

    Only lines pymupdf4llm already marked as headings are touched; fenced
    code blocks are skipped. A heading line is:
      - re-levelled when its span font size is a heading tier;
      - DEMOTED to a paragraph (prefix stripped) when its span font size is
        body-size-or-smaller (not a heading tier) AND its text reads as prose
        — pymupdf4llm over-flagged a body sentence (ported from the Docling
        worker's `_demote_misdetected_headers`, here gated on real font size);
      - left exactly as emitted when it matches no span (font unknown →
        conservative)."""
    if not size_to_level:
        return page_md

    text_size: dict[str, int] = {}
    for blk in page.get_text("dict").get("blocks", []):
        for line in blk.get("lines", []):
            spans = [s for s in line.get("spans", []) if str(s.get("text", "")).strip()]
            if not spans:
                continue
            key = _norm_heading("".join(str(s.get("text", "")) for s in spans))
            if key:
                size = round(max(float(s.get("size", 0.0)) for s in spans))
                text_size.setdefault(key, size)

    out: list[str] = []
    in_fence = False
    for line in page_md.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        m = _HEADING_LINE_RE.match(line)
        if m is not None and not in_fence:
            text = line[len(m.group(1)) :]
            key = _norm_heading(text)
            matched_size = text_size.get(key)
            level = size_to_level.get(matched_size) if matched_size is not None else None
            if level is not None:
                out.append("#" * level + text)
                continue
            # Demote: matched a real span whose font is body-size-or-smaller
            # (not a heading tier) AND reads as prose → strip the prefix so the
            # chunker + heading_path see a paragraph. No span match → keep it
            # (can't confirm the font, so stay conservative).
            if matched_size is not None and _looks_like_prose_heading(text):
                out.append(text.lstrip())
                continue
        out.append(line)
    return "\n".join(out)


# A bare page-number / "Page N (of M)" line in the boundary band is furniture even though
# each page's number differs (so it can't be caught by repeat-frequency).
_PAGE_NUMBER_RE = re.compile(r"^(?:\d{1,4}|[Pp]age\s+\d{1,4}(?:\s+of\s+\d{1,4})?)$")
# Lowercase roman page numbers 1–39 (the front-matter `iv`/`vi`/`vii`/`x` case). A STRICT roman
# grammar (`(?=[ivx])` forces a non-empty roman start; tens-then-units must consume to `$`) so it
# rejects the all-[ivxlcdm] English false-friends — `mix`/`did`/`lid`/`civil`/`mild` all fail the
# tens/units body. (Roman ≥40 in front matter is vanishingly rare; not matched, deliberately.)
_ROMAN_PAGENO_RE = re.compile(r"^(?=[ivx])(?:x{0,3})(?:ix|iv|v?i{0,3})$", re.IGNORECASE)
# A markdown-structural boundary line is NEVER furniture (heading / table / code / list / quote).
_FURNITURE_STRUCTURAL_PREFIXES = ("#", "|", "```", "- ", "* ", "> ", "[", "<!--")
_MAX_FURNITURE_LEN = 90


def _is_furniture_candidate(text: str) -> bool:
    s = text.strip()
    if not s or len(s) > _MAX_FURNITURE_LEN:
        return False
    return not s.startswith(_FURNITURE_STRUCTURAL_PREFIXES)


# A running header/footer can be MULTIPLE lines (e.g. NIST stacks "NIST SP 800-207" over
# "ZERO TRUST ARCHITECTURE"). Look at a BAND of the first/last N non-blank lines, not just the
# single boundary line. N=3 is robust for up-to-3-line headers; the high recurrence threshold +
# structural-line exclusion keep it from catching real body content (which is unique per page).
_FURNITURE_BAND = 3


def _page_band_indices(md: str) -> list[int]:
    """Sorted indices of the first N and last N NON-BLANK lines of `md` (the header/footer band)."""
    lines = md.split("\n")
    nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
    if not nonblank:
        return []
    return sorted(set(nonblank[:_FURNITURE_BAND] + nonblank[-_FURNITURE_BAND:]))


def strip_repeating_page_furniture(
    pages: list[str], *, min_repeat: int = 3, min_fraction: float = 0.5
) -> tuple[list[str], int]:
    """Strip running headers/footers + page numbers that recur in the PAGE-BOUNDARY BAND across
    many pages (audit-10 W8 — e.g. NIST's two-line header repeated ×56 interleaved mid-prose).

    `pages` is the per-page markdown list. A band line (in the first/last `_FURNITURE_BAND`
    non-blank lines of a page) is furniture when it (a) recurs in the band on >= max(`min_repeat`,
    `min_fraction` × n_pages) pages, OR (b) is a bare page number. Stripping is POSITION-AWARE —
    only the band occurrence is removed, so a legitimate mid-page occurrence of the same text is
    untouched — and markdown-structural lines (headings/tables/lists/code) are never furniture.
    Pure-sync; returns `(cleaned_pages, lines_stripped)`. A short doc (< `min_repeat` pages) is
    returned unchanged."""
    if len(pages) < min_repeat:
        return pages, 0
    counts: dict[str, int] = {}
    for md in pages:
        lines = md.split("\n")
        seen: set[str] = set()
        for idx in _page_band_indices(md):
            txt = lines[idx].strip()
            if _is_furniture_candidate(txt) and txt not in seen:  # dedup within a page
                counts[txt] = counts.get(txt, 0) + 1
                seen.add(txt)
    threshold = max(min_repeat, int(min_fraction * len(pages)))
    repeating = {t for t, c in counts.items() if c >= threshold}

    stripped = 0
    out: list[str] = []
    for md in pages:
        lines = md.split("\n")
        drop: set[int] = set()
        for idx in _page_band_indices(md):
            txt = lines[idx].strip()
            is_pageno = _is_furniture_candidate(txt) and bool(
                _PAGE_NUMBER_RE.match(txt) or _ROMAN_PAGENO_RE.match(txt)
            )
            if txt in repeating or is_pageno:
                drop.add(idx)
        if drop:
            stripped += len(drop)
            out.append("\n".join(ln for i, ln in enumerate(lines) if i not in drop).strip("\n"))
        else:
            out.append(md)
    return out, stripped


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
            with contextlib.suppress(Exception):
                return bool(v())
        elif isinstance(v, bool):
            return v

    with contextlib.suppress(Exception):
        cat = doc.pdf_catalog()
        mark_info = doc.xref_get_key(cat, "MarkInfo")
        if isinstance(mark_info, tuple) and len(mark_info) >= 2 and mark_info[0] == "xref":
            xref_str = mark_info[1].split()[0]
            xref = int(xref_str)
            marked = doc.xref_get_key(xref, "Marked")
            if isinstance(marked, tuple) and len(marked) >= 2:
                return str(marked[1]).lower() == "true"

    return False


def _detect_layers(doc: Any) -> bool:
    """Best-effort /OCProperties presence check."""
    with contextlib.suppress(Exception):
        cat = doc.pdf_catalog()
        oc = doc.xref_get_key(cat, "OCProperties")
        if isinstance(oc, tuple) and len(oc) >= 2:
            return oc[0] != "null"
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
    aspect_stdev = float(statistics.pstdev(aspect_ratios)) if len(aspect_ratios) > 1 else 0.0

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
        "image_heavy_page_fraction": (image_heavy_pages / page_count if page_count > 0 else 0.0),
        "image_area_fraction": image_area_avg,
        "total_chars": total_chars,
        "chars_per_page_avg": chars_avg,
        "chars_per_page_median": chars_median,
        "chars_per_page_p10": chars_p10,
        "chars_per_page_p90": chars_p90,
        "empty_page_fraction": empty_pages / page_count if page_count > 0 else 0.0,
        "replacement_char_fraction": (replacement_chars / total_chars if total_chars > 0 else 0.0),
        "word_like_token_fraction": word_like_tokens / rough_token_count,
        "unique_char_variety": len(set(raw_text)),
        "whitespace_fraction": (whitespace_chars / total_chars if total_chars > 0 else 0.0),
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
        raise SystemExit(json.dumps({"error": "pymupdf_unavailable", "detail": str(e)})) from e

    pymupdf_version: str | None = getattr(pymupdf, "__version__", None) or getattr(
        pymupdf4llm, "__version__", None
    )

    doc: Any = pymupdf.open(str(source))
    try:
        # Doc-wide font-size → heading-level map, to repair pymupdf4llm's
        # level collapse (all headings → `## `). Computed once; applied
        # per page below. See `_heading_size_to_level`.
        size_to_level = _heading_size_to_level(doc)

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
                # Repair pymupdf4llm's heading-level collapse using this
                # page's real span sizes (only touches detected heading
                # lines; no-op when no heading tiers exist).
                page_md = _remap_heading_levels(page_md, page_obj, size_to_level)
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

            # Record + join in LOCKSTEP: skip an empty page from BOTH (the join's
            # `if p` filter already drops empty parts, so an empty page contributes
            # neither text NOR a "\n\n" delimiter to `joined`). Keeping a char_count=0
            # record while the join skipped its delimiter drifts the chunker's
            # page→offset mapping +2 chars per empty page. This keeps `joined`
            # byte-identical (no content-hash change / forced re-parse). The PyMuPDF
            # path does no per-page VLM escalation, so an empty page is genuinely
            # blank — safe to drop (unlike the Docling path, where an empty-markdown
            # page may be a diagram awaiting escalation).
            if page_md:
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

        # audit-10 W8: drop running headers/footers + page numbers that recur at the page
        # boundary (e.g. a header repeated on most pages, interleaved mid-prose once stitched).
        # `pages` and `markdown_parts` are 1:1 (appended together); keep them in lockstep, and
        # drop any page that became empty after the strip (so page→offset mapping stays right,
        # same discipline as the empty-page handling above).
        cleaned_parts, furniture_n = strip_repeating_page_furniture(markdown_parts)
        if furniture_n:
            kept_pages: list[dict[str, Any]] = []
            kept_parts: list[str] = []
            for rec, part in zip(pages, cleaned_parts, strict=True):
                if part:
                    rec["markdown"] = part
                    rec["char_count"] = len(part)
                    kept_pages.append(rec)
                    kept_parts.append(part)
            pages, markdown_parts = kept_pages, kept_parts
            print(f"pymupdf: stripped {furniture_n} repeating page-furniture lines", file=sys.stderr)

        joined = "\n\n".join(p for p in markdown_parts if p)
        signals = _collect_signals(doc, joined)

        return {
            "markdown": joined,
            "pages": pages,
            "pymupdf_version": pymupdf_version,
            "signals": signals,
            # audit-10 W9: the honest figure count = the `<!-- image -->` placeholders in the WRITTEN
            # body (each is one converted picture-omitted marker; `<!--` is furniture-exempt so the
            # furniture strip never drops one → body count == converted count). Was hardcoded 0 (a
            # lie). table_count + equation_count stay 0 — DEFERRED (a documented W9 follow-up in
            # ROADMAP): an honest table_count must be counted AFTER `_finalize_body`'s layout-table
            # demotion (a pipeline-level concern, not the worker's), and born-digital equations need
            # OCR-LaTeX (a heavy separate model); near-absent on these docs (gte 1 ref, NIST 0).
            "figure_count": joined.count("<!-- image -->"),
            "table_count": 0,
            "equation_count": 0,
        }
    finally:
        with contextlib.suppress(Exception):
            doc.close()


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
