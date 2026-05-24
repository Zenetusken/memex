"""Subprocess entry-point for Docling — see GUIDELINES.md Part VI.

`parse.docling_backend.convert` spawns this module via
`python -m memex.parse.docling_worker <source_path>`. Running Docling
in a separate process contains its crashes: a segfault here kills only
this child, not the parent agent.

**Network egress is blocked** before docling is imported — see
`memex.parse.sandbox`. Any attempt by docling or its transitive
imports to phone home (model download, telemetry, DNS) fails at the
kernel level with `EPERM`. The parent toggles the block via
`MEMEX_PARSE_SANDBOX_NETWORK` (1 = block, 0 = skip).

The worker reads the source path from argv, runs Docling, and writes
the resulting `DoclingConversion` payload to stdout as JSON. Errors go
to stderr and are surfaced via the exit code.

Exit codes:
  0 success
  2 usage error
  3 source not found
  4 docling unavailable
  5 unhandled exception during conversion
  6 sandbox requested but couldn't be applied (security policy violation)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

# CRITICAL: the worker's stdout IS the protocol channel — the parent
# parses it as JSON via `json.loads`. Anything written to stdout by
# log handlers (structlog's default ConsoleRenderer; the seccomp
# sandbox's `sandbox.applied` event; any chatty third-party lib)
# corrupts the JSON. Redirect *all* stderr-bound logging into actual
# stderr by configuring structlog + the stdlib logging root before
# any module that uses them is imported.
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


# ----- Heading-level recovery -----
#
# Docling's PDF reading-order model classifies headings as
# `SectionHeaderItem` but never assigns a level, so every one defaults to
# `level=1` and `export_to_markdown` emits them all as `## `
# (`num_hashes = item.level + 1` in docling_core's markdown serializer).
# The document loses hierarchy: a paper's sections + subsections, and a
# 200-page 10-K's 500 headings, all flatten to H2. `SectionHeaderItem.level`
# is a writable field the serializer respects, so we re-derive it from a
# font-size proxy BEFORE export. Docling exposes no font size, but each
# item's provenance carries a bbox whose height tracks size. This is the
# Docling-path analogue of `pymupdf_worker._heading_size_to_level`.


def _recover_heading_levels(doc: Any) -> int:
    """Re-derive Docling section-header levels from provenance bbox height.

    Ranks the distinct section-header heights **among themselves** (largest
    → level 1) and writes the level back on each item in place, so the
    markdown serializer emits the recovered hierarchy. Ranking headers
    against each other (not against body text) means a slide deck whose
    titles are all one size stays flat — peers remain peers — rather than
    mis-promoting a single title slide.

    Returns the number of headers re-levelled. `TitleItem` (no `.level`,
    always `#`) and headers without provenance are left untouched.
    """
    texts: list[Any] = getattr(doc, "texts", None) or []
    headers: list[tuple[Any, float]] = []
    for item in texts:
        raw_label = getattr(item, "label", None)
        label = getattr(raw_label, "value", raw_label)
        if label != "section_header" or not hasattr(item, "level"):
            continue
        heights: list[float] = []
        prov_items: Any = getattr(item, "prov", None) or []
        for p in prov_items:
            bbox = getattr(p, "bbox", None)
            h = getattr(bbox, "height", None)
            if h is not None:
                heights.append(float(h))
        if not heights:
            continue
        # Bucket to nearest 0.5 pt: absorbs float jitter (11.98 vs 12.03)
        # without merging genuinely distinct heading tiers. `max` over the
        # provs resists a short wrapped-line continuation fragment.
        headers.append((item, round(max(heights) * 2) / 2))

    if not headers:
        return 0

    # Distinct buckets, largest first → levels 1..5. The serializer emits
    # `level + 1` hashes, so cap at 5 → `######`, markdown's deepest.
    tiers = sorted({bucket for _, bucket in headers}, reverse=True)
    level_of = {bucket: min(i + 1, 5) for i, bucket in enumerate(tiers)}
    for item, bucket in headers:
        item.level = level_of[bucket]
    return len(headers)


# ----- Prose-heading demotion (mis-DETECTION fix) -----
#
# Beyond mis-LEVELLING, Docling's reading-order model also mis-DETECTS body
# sentences as `SectionHeaderItem`s on dense docs (e.g. NVIDIA's annual review
# tagged "Data centers are becoming AI factories." and a 304-char paragraph as
# headings). `_recover_heading_levels` can only rank what Docling labelled; it
# can't un-label a sentence. A level-fix can't repair a detection error.
#
# We fix this at the ROOT — on the structured `DoclingDocument` before export —
# rather than by regexing the markdown. `SectionHeaderItem` is a subclass of
# `TextItem`, and the serializer dispatches headings by `isinstance` (not by
# the `label` field), so reassigning a mis-detected item's `__class__` to
# `TextItem` makes it serialize as a plain paragraph natively (verified). That
# keeps the document model itself correct, so export, chunking, and
# `heading_path` all see a paragraph. `_demote_prose_headings` (the markdown
# post-process below) is retained only as a defence-in-depth FALLBACK for any
# heading that slips past the structured pass.
#
# The discriminating signal is textual (most small headings ARE legitimate
# sub-headings; bbox height is too noisy on slide decks — see
# `_recover_heading_levels`): a real heading is short and title-like; a
# mis-detected one ends in sentence punctuation or runs long. Measured on the
# 10-K: real headings ≤12 words and never end with `. ! ?`; the prose
# mis-detections end with terminal punctuation or run well past any heading.
# `bbox` height/width is available on each item for richer signal if a future
# corpus needs it, but the text shape suffices today.

_HEADING_LINE_RE = re.compile(r"^(#{1,6})[ \t]+\S")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_EMPHASIS_RE = re.compile(r"[*`]")
_TERMINAL_PUNCT = (".", "!", "?")
# A heading longer than this many words is prose regardless of punctuation
# (real headings here top out at ~12 words).
_MAX_HEADING_WORDS = 15
# A terminal-punctuation line needs at least this many words to count as prose
# — guards short labels like "Item 1." / "Note 5." that legitimately end in `.`.
_MIN_PROSE_WORDS = 4


def _looks_like_prose_heading(heading_text: str) -> bool:
    """True if a heading line's text reads as a body sentence, not a title."""
    t = _EMPHASIS_RE.sub("", heading_text).strip()
    if not t:
        return False
    words = t.split()
    if len(words) > _MAX_HEADING_WORDS:
        return True
    return len(words) >= _MIN_PROSE_WORDS and t[-1] in _TERMINAL_PUNCT


def _demote_misdetected_headers(doc: Any, *, text_item_cls: type) -> int:
    """Reclassify SectionHeaderItems whose text reads as prose into plain text.

    The root-level mis-detection fix: a body sentence Docling tagged as a
    `SectionHeaderItem` is reassigned to `text_item_cls` (the real `TextItem`),
    so `export_to_markdown` — which routes headings by `isinstance` — emits it
    as a paragraph. The document model is corrected in place, so downstream
    chunking + `heading_path` see a paragraph too. Returns the count demoted.

    Duck-typed (`label`/`text`/`level`) + `text_item_cls` injected, so it's
    unit-testable without docling. The `__class__` swap is wrapped defensively:
    if a future pydantic/docling version rejects it, the item is left as a
    heading and the markdown-level `_demote_prose_headings` fallback catches it.
    """
    texts: list[Any] = getattr(doc, "texts", None) or []
    demoted = 0
    for item in texts:
        raw_label = getattr(item, "label", None)
        label = getattr(raw_label, "value", raw_label)
        if label != "section_header" or not hasattr(item, "level"):
            continue
        if not _looks_like_prose_heading(getattr(item, "text", "") or ""):
            continue
        try:
            item.__class__ = text_item_cls
            demoted += 1
        except (TypeError, AttributeError):
            # __class__ reassignment refused — leave it; the markdown
            # fallback will demote it post-export.
            continue
    return demoted


def _demote_prose_headings(markdown: str) -> tuple[str, int]:
    """FALLBACK: strip the heading prefix from lines that read as body prose.

    Docling mis-detects body sentences as headings; this turns those lines
    back into paragraphs so they stop creating spurious section breaks in
    the chunker and stop polluting `heading_path`. Only the `#` prefix is
    removed — the text is preserved verbatim. Fenced code is skipped; lines
    pymupdf-style detected as real headings are left untouched. Returns
    `(markdown, demoted_count)`.
    """
    out: list[str] = []
    in_fence = False
    demoted = 0
    for line in markdown.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        m = _HEADING_LINE_RE.match(line)
        if m is not None and not in_fence:
            text = line[len(m.group(1)) :].lstrip()
            if _looks_like_prose_heading(text):
                out.append(text)
                demoted += 1
                continue
        out.append(line)
    return "\n".join(out), demoted


def _convert_to_payload(source: Path) -> dict[str, Any]:
    """Run Docling, return a JSON-serialisable dict matching the
    `DoclingConversion` pydantic schema in `docling_backend.py`.

    OCR is disabled by default: most PDFs have a native text layer
    (slide decks, papers exported from LaTeX/Word, etc.), so running
    RapidOCR per page just adds 3-5s/page of redundant work. Set
    `MEMEX_PARSE_DOCLING_OCR=1` to flip it on for scanned-image PDFs.
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.types.doc.document import TextItem
    except ImportError as e:
        raise SystemExit(json.dumps({"error": "docling_unavailable", "detail": str(e)})) from e

    try:
        import docling

        version = getattr(docling, "__version__", None)
    except Exception:
        version = None

    do_ocr = os.environ.get("MEMEX_PARSE_DOCLING_OCR", "0") == "1"
    # P3.3 v2: enable Docling's built-in picture classifier so we can
    # pre-filter chart-OCR candidates by `class_name` (bar_chart,
    # line_chart, pie_chart, scatter_plot, box_plot) and skip the rest
    # (logo, flow_chart, photograph, icon, etc.). On the canonical
    # CUDA deck this drops 245 picture objects → ~26 actual chart
    # candidates, an 89% reduction that prevents DePlot's
    # OOD-hallucination cascade on non-chart content. Opt-out via env
    # var if a future Docling-version bug requires it.
    do_classify = os.environ.get("MEMEX_PARSE_DOCLING_PICTURE_CLASSIFICATION", "1") == "1"
    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.do_ocr = do_ocr
    pipeline_opts.do_picture_classification = do_classify
    print(
        f"docling: do_ocr={do_ocr}, do_picture_classification={do_classify}",
        file=sys.stderr,
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
    )
    result = converter.convert(source)
    doc = result.document
    # Repair Docling's heading-level collapse (all `## `) from provenance
    # bbox heights, before export so the serializer emits real hierarchy.
    relevelled = _recover_heading_levels(doc)
    if relevelled:
        print(f"docling: re-levelled {relevelled} section headers", file=sys.stderr)
    # Root fix: reclassify body sentences Docling mis-detected as headings into
    # plain text items BEFORE export, so they serialise as paragraphs natively.
    reclassified = _demote_misdetected_headers(doc, text_item_cls=TextItem)
    if reclassified:
        print(f"docling: reclassified {reclassified} prose headings as text", file=sys.stderr)
    markdown = doc.export_to_markdown()
    # Fallback: catch any prose heading that slipped past the structured pass
    # (e.g. a refused __class__ swap) at the markdown level.
    markdown, demoted = _demote_prose_headings(markdown)
    if demoted:
        print(f"docling: demoted {demoted} prose headings (fallback)", file=sys.stderr)

    pages: list[dict[str, Any]] = []
    # Docling's `DoclingDocument` ships `py.typed`, but its dynamic surface
    # (`.pages`, `.pictures`, provenance entries) resolves to Unknown under
    # strict, and we probe it defensively across docling versions anyway.
    # Annotate the boundary as `Any` so the `getattr` probing type-checks.
    page_list: list[Any] = getattr(doc, "pages", None) or []
    for p in page_list:
        page_no = getattr(p, "page_no", None) or getattr(p, "number", None) or len(pages) + 1
        page_md = ""
        export = getattr(p, "export_to_markdown", None)
        if callable(export):
            try:
                raw_page_md: Any = export()
                page_md, _ = _demote_prose_headings(raw_page_md)
            except Exception:
                page_md = ""
        confidence = float(getattr(p, "confidence", 1.0))
        # Docling sometimes reports NaN for empty pages. The parent's
        # `json.loads` rejects literal NaN, so coerce to 0.0 here — the
        # downstream VLM-escalation decision is "below threshold" either
        # way, but the JSON has to be RFC-strict for the parent to parse.
        if math.isnan(confidence) or math.isinf(confidence):
            confidence = 0.0
        pages.append({"page": page_no, "markdown": page_md, "confidence": confidence})
    if not pages:
        pages.append({"page": 1, "markdown": markdown, "confidence": 1.0})

    # P3.3 Session 2: serialise per-figure metadata for the chart-OCR
    # backend. The image BYTES are NOT included — the chart-OCR pass
    # re-renders from the source PDF via pypdfium2 (see
    # `parse/chart_ocr_backend.py` once shipped). We tolerate any
    # Docling internal-API shift by `getattr`-defaulting each field;
    # missing metadata just yields an empty `figures` list and the
    # chart-OCR pass skips the doc.
    figures: list[dict[str, Any]] = []
    pictures: list[Any] = getattr(doc, "pictures", None) or []
    for pic in pictures:
        # Docling's picture object exposes its position through
        # `.prov[0].bbox` — a `BoundingBox` with `.l / .t / .r / .b`
        # attributes in `CoordOrigin.BOTTOMLEFT` coords (verified on
        # docling 2.x). The picture itself doesn't carry a direct
        # `.bbox` or `.page_no`; both live on the provenance entry.
        # We defensively probe but the common path is `.prov[0]`.
        bbox_obj: Any = None
        page_no: Any = None
        if hasattr(pic, "bbox") and not isinstance(pic.bbox, (list, tuple)):
            bbox_obj = pic.bbox
        prov_list: list[Any] = getattr(pic, "prov", None) or []
        if bbox_obj is None and prov_list:
            bbox_obj = getattr(prov_list[0], "bbox", None)
        if hasattr(pic, "page_no") and pic.page_no is not None:
            page_no = pic.page_no
        elif prov_list:
            page_no = getattr(prov_list[0], "page_no", None)
        if bbox_obj is None or page_no is None:
            continue
        try:
            # Try the `.l/.t/.r/.b` schema first (docling's canonical
            # BoundingBox); fall back to `.x0/.y0/.x1/.y1` for older
            # versions or alternate types.
            x0_attr = getattr(bbox_obj, "l", None)
            x0 = float(x0_attr if x0_attr is not None else getattr(bbox_obj, "x0", 0.0))
            y_bot_attr = getattr(bbox_obj, "b", None)
            y_bot = float(y_bot_attr if y_bot_attr is not None else getattr(bbox_obj, "y0", 0.0))
            x1_attr = getattr(bbox_obj, "r", None)
            x1 = float(x1_attr if x1_attr is not None else getattr(bbox_obj, "x1", 0.0))
            y_top_attr = getattr(bbox_obj, "t", None)
            y_top = float(y_top_attr if y_top_attr is not None else getattr(bbox_obj, "y1", 0.0))
        except (TypeError, ValueError):
            continue
        # FigureMetadata.bbox is documented as (x0, y0_bottom, x1,
        # y1_top) in bottom-left coords — matches what the chart-OCR
        # backend's renderer expects.
        caption_obj = getattr(pic, "caption_text", None)
        if callable(caption_obj):
            try:
                caption_obj = caption_obj()
            except Exception:
                caption_obj = None
        caption = str(caption_obj) if caption_obj else None

        # P3.3 v2: Docling's PictureClassifier emits a ranked list of
        # `PictureClassificationPrediction(class_name, confidence)`
        # under `pic.meta.classification.predictions`. Capture the top
        # prediction; the chart-OCR backend uses this to skip non-chart
        # pictures (logos, diagrams, photos) before invoking the model.
        classification = None
        classification_confidence = 0.0
        meta = getattr(pic, "meta", None)
        if meta is not None:
            cls_field = getattr(meta, "classification", None)
            if cls_field is not None:
                preds: list[Any] = getattr(cls_field, "predictions", None) or []
                if preds:
                    top: Any = preds[0]
                    classification = str(getattr(top, "class_name", "") or "") or None
                    try:
                        classification_confidence = float(getattr(top, "confidence", 0.0))
                    except (TypeError, ValueError):
                        classification_confidence = 0.0

        figures.append(
            {
                "page_no": int(page_no),
                "bbox": [x0, y_bot, x1, y_top],
                "caption": caption,
                "classification": classification,
                "classification_confidence": classification_confidence,
            }
        )

    return {
        "markdown": markdown,
        "pages": pages,
        "docling_version": version,
        "figure_count": len(getattr(doc, "pictures", []) or []),
        "table_count": len(getattr(doc, "tables", []) or []),
        "equation_count": len(getattr(doc, "equations", []) or []),
        "figures": figures,
    }


def main(argv: list[str] | None = None) -> int:
    """Subprocess entry point — invoked by
    `parse/docling_backend.py::convert` via `python -m memex.parse.
    docling_worker <source_path>`. Returns the conversion as JSON on
    stdout; non-zero exit code signals a crash (segfault, abort, or
    a recoverable error written to stderr). Runs under the seccomp
    sandbox installed before Docling import."""
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print(
            "usage: python -m memex.parse.docling_worker <source_path>",
            file=sys.stderr,
        )
        return 2

    source = Path(args[0])
    if not source.exists():
        print(f"source file does not exist: {source}", file=sys.stderr)
        return 3

    # Install the seccomp network-egress block BEFORE importing docling.
    # If the user disabled the sandbox via env var, skip; otherwise
    # treat a sandbox-load failure as a security-policy violation and
    # refuse to run the parse.
    if os.environ.get("MEMEX_PARSE_SANDBOX_NETWORK", "1") == "1":
        from memex.parse.sandbox import enable_network_block

        status, reason = enable_network_block()
        print(f"sandbox: {status} ({reason})", file=sys.stderr)
        if status == "failed":
            return 6

    try:
        payload = _convert_to_payload(source)
    except SystemExit as e:
        # Already wrote a JSON error to stdout via SystemExit().
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            return 4
        return e.code or 4
    except Exception:
        # Print the traceback to stderr; the parent surfaces it via
        # DoclingCrashed.
        traceback.print_exc()
        return 5

    # `allow_nan=False` makes json.dump raise on NaN/Inf rather than
    # emit the non-RFC literal `NaN` (which the parent's json.loads
    # would reject with a misleading "unparseable output"). The
    # confidence-coercion above is the primary guard; this is the safety
    # net for any other field that ever sneaks a float through.
    try:
        json.dump(payload, sys.stdout, allow_nan=False)
    except ValueError as e:
        print(f"json.dump failed: {e}", file=sys.stderr)
        return 5
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
