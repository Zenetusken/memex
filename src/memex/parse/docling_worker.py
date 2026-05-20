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
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _convert_to_payload(source: Path) -> dict[str, Any]:
    """Run Docling, return a JSON-serialisable dict matching the
    `DoclingConversion` pydantic schema in `docling_backend.py`."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as e:
        raise SystemExit(
            json.dumps({"error": "docling_unavailable", "detail": str(e)})
        ) from e

    try:
        import docling

        version = getattr(docling, "__version__", None)
    except Exception:
        version = None

    converter = DocumentConverter()
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

    return {
        "markdown": markdown,
        "pages": pages,
        "docling_version": version,
        "figure_count": len(getattr(doc, "pictures", []) or []),
        "table_count": len(getattr(doc, "tables", []) or []),
        "equation_count": len(getattr(doc, "equations", []) or []),
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
