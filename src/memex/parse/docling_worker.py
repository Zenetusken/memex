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
            structlog.processors.KeyValueRenderer(
                key_order=["timestamp", "level", "event"]
            ),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
except ImportError:
    pass


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
    except ImportError as e:
        raise SystemExit(
            json.dumps({"error": "docling_unavailable", "detail": str(e)})
        ) from e

    try:
        import docling

        version = getattr(docling, "__version__", None)
    except Exception:
        version = None

    do_ocr = os.environ.get("MEMEX_PARSE_DOCLING_OCR", "0") == "1"
    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.do_ocr = do_ocr
    print(f"docling: do_ocr={do_ocr}", file=sys.stderr)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
        }
    )
    result = converter.convert(source)
    doc = result.document
    markdown = doc.export_to_markdown()

    pages: list[dict[str, Any]] = []
    page_list = getattr(doc, "pages", None) or []
    for p in page_list:
        page_no = (
            getattr(p, "page_no", None)
            or getattr(p, "number", None)
            or len(pages) + 1
        )
        page_md = ""
        export = getattr(p, "export_to_markdown", None)
        if callable(export):
            try:
                page_md = export()
            except Exception:
                page_md = ""
        confidence = float(getattr(p, "confidence", 1.0))
        # Docling sometimes reports NaN for empty pages. The parent's
        # `json.loads` rejects literal NaN, so coerce to 0.0 here — the
        # downstream VLM-escalation decision is "below threshold" either
        # way, but the JSON has to be RFC-strict for the parent to parse.
        if math.isnan(confidence) or math.isinf(confidence):
            confidence = 0.0
        pages.append(
            {"page": page_no, "markdown": page_md, "confidence": confidence}
        )
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
    for pic in getattr(doc, "pictures", None) or []:
        # Docling's picture object exposes its position through
        # `.prov[0].bbox` — a `BoundingBox` with `.l / .t / .r / .b`
        # attributes in `CoordOrigin.BOTTOMLEFT` coords (verified on
        # docling 2.x). The picture itself doesn't carry a direct
        # `.bbox` or `.page_no`; both live on the provenance entry.
        # We defensively probe but the common path is `.prov[0]`.
        bbox_obj = None
        page_no = None
        if hasattr(pic, "bbox") and not isinstance(pic.bbox, (list, tuple)):
            bbox_obj = pic.bbox
        prov_list = getattr(pic, "prov", None) or []
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
            x0 = float(
                x0_attr
                if x0_attr is not None
                else getattr(bbox_obj, "x0", 0.0)
            )
            y_bot_attr = getattr(bbox_obj, "b", None)
            y_bot = float(
                y_bot_attr
                if y_bot_attr is not None
                else getattr(bbox_obj, "y0", 0.0)
            )
            x1_attr = getattr(bbox_obj, "r", None)
            x1 = float(
                x1_attr
                if x1_attr is not None
                else getattr(bbox_obj, "x1", 0.0)
            )
            y_top_attr = getattr(bbox_obj, "t", None)
            y_top = float(
                y_top_attr
                if y_top_attr is not None
                else getattr(bbox_obj, "y1", 0.0)
            )
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
        figures.append(
            {
                "page_no": int(page_no),
                "bbox": [x0, y_bot, x1, y_top],
                "caption": caption,
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
