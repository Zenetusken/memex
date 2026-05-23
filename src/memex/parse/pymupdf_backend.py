"""PyMuPDF4LLM pre-filter — fast native-text-layer PDF extraction.

For PDFs with a usable native text layer, PyMuPDF4LLM extracts
markdown by walking the PDF's text objects directly — ~10-20× faster
than running Docling's layout model end-to-end. The worker
additionally collects a rich signal set (PDF producer metadata,
char distribution per page, image area fraction, mojibake ratio,
markdown structure flags) so the pipeline's tiered classifier can
route the document accurately:

  - Born-digital PowerPoint/LaTeX/Word PDFs   → use PyMuPDF
  - Scanner-produced PDFs                     → fall through to Docling
  - Mixed-content (text + charts/screenshots) → fall through to Docling-with-OCR
  - Mojibake / broken-encoding extractions    → fall through to Docling

Runs out-of-process via `python -m memex.parse.pymupdf_worker` for the
same crash-containment reasons as Docling. The worker also installs
the seccomp-bpf network-egress block (see `memex.parse.sandbox`)
before importing pymupdf.

The pre-filter never blocks a parse — any failure (unavailable,
timeout, crash) is logged and Docling runs as if the pre-filter had
never been attempted.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

from memex.core.errors import MemexError
from memex.parse.docling_backend import SandboxLoadFailed

logger = structlog.get_logger(__name__)


class PdfSignals(BaseModel):
    """Rich signal set collected by the worker for the routing classifier.

    Every field is a cheap derivation from one PyMuPDF document open
    plus the extracted markdown. The classifier in `pipeline._classify`
    combines them via a tiered policy — producer metadata first,
    structural cues next, text distribution + quality after that, then
    aspect-ratio-aware density as the fallback.
    """

    # --- PDF producer metadata (Tier 1) ---
    creator: str | None = None
    producer: str | None = None
    pdf_version: str | None = None
    is_encrypted: bool = False
    is_tagged: bool = False
    has_layers: bool = False

    # --- Structural cues (Tier 2) ---
    page_count: int = 0
    avg_aspect_ratio: float = 1.0
    aspect_ratio_consistent: bool = True
    embedded_font_count: int = 0
    image_count_total: int = 0
    image_heavy_page_fraction: float = 0.0
    image_area_fraction: float = 0.0

    # --- Text density distribution (Tier 3) ---
    total_chars: int = 0
    chars_per_page_avg: float = 0.0
    chars_per_page_median: float = 0.0
    chars_per_page_p10: float = 0.0
    chars_per_page_p90: float = 0.0
    empty_page_fraction: float = 0.0

    # --- Text quality (Tier 3) ---
    replacement_char_fraction: float = 0.0
    word_like_token_fraction: float = 0.0
    unique_char_variety: int = 0
    whitespace_fraction: float = 0.0

    # --- Markdown structure (Tier 4) ---
    has_headings: bool = False
    has_tables: bool = False
    has_lists: bool = False
    has_code_blocks: bool = False


class PyMuPDFPageOutput(BaseModel):
    """Per-page PyMuPDF output — Markdown plus the signals
    (`char_count`, `image_count`, `aspect_ratio`) the parse-stage
    classifier uses to decide whether to escalate to Docling."""

    page: int
    markdown: str
    char_count: int
    image_count: int = 0
    aspect_ratio: float = 1.0


class PyMuPDFConversion(BaseModel):
    """The full output the PyMuPDF worker returns via IPC — stitched
    Markdown, per-page outputs, the document-level `PdfSignals` used
    by the routing classifier, and the counts."""

    markdown: str
    pages: list[PyMuPDFPageOutput]
    pymupdf_version: str | None = None
    signals: PdfSignals = Field(default_factory=PdfSignals)
    figure_count: int = 0
    table_count: int = 0
    equation_count: int = 0


class PyMuPDFUnavailable(MemexError):
    """PyMuPDF4LLM is not installed. Install with `uv sync --extra parse`."""


class PyMuPDFTimeout(MemexError):
    """PyMuPDF worker exceeded its per-document timeout."""


class PyMuPDFCrashed(MemexError):
    """The PyMuPDF subprocess exited non-zero."""


_WORKER_MODULE = "memex.parse.pymupdf_worker"


async def convert(
    source: Path,
    *,
    timeout_s: int,
    sandbox_network: bool = True,
) -> PyMuPDFConversion:
    """Convert `source` to markdown via the PyMuPDF4LLM worker subprocess.

    Mirrors `docling_backend.convert` for crash-containment + sandbox
    semantics. The child process opens the PDF, collects the rich
    signal set, extracts markdown via `pymupdf4llm.to_markdown`, and
    JSON-encodes the `PyMuPDFConversion` payload to stdout. Non-zero
    exit signals failure (segfault, unavailable, sandbox refusal).

    With `sandbox_network=True` (default), the child installs a
    seccomp network-egress block before importing pymupdf — the
    library has no business making network calls, and the sandbox
    enforces that.

    Wraps the call in `asyncio.wait_for(timeout_s)`. On timeout (or
    cancellation), the child is SIGTERM'd → 2 s grace → SIGKILL'd
    so pipes drain and the OS reaps the process.
    """
    log = logger.bind(
        source=str(source),
        timeout_s=timeout_s,
        sandbox_network=sandbox_network,
    )
    log.info("pymupdf.start")

    env = os.environ.copy()
    env["MEMEX_PARSE_SANDBOX_NETWORK"] = "1" if sandbox_network else "0"

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        _WORKER_MODULE,
        str(source),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    stdout: bytes
    stderr: bytes
    try:
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
        except TimeoutError as e:
            raise PyMuPDFTimeout(
                f"PyMuPDF worker exceeded {timeout_s}s on {source}",
                context={"source": str(source), "timeout_s": timeout_s},
                recoverable=True,
            ) from e
    except BaseException:
        if proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await proc.wait()
                except BaseException:
                    pass
        raise

    if proc.returncode != 0:
        stderr_text = (stderr or b"").decode("utf-8", errors="replace")
        # Worker exit codes match docling_worker:
        #   2 — usage error
        #   3 — source not found
        #   4 — pymupdf_unavailable
        #   5 — exception during extraction
        #   6 — sandbox requested but couldn't be applied
        if proc.returncode == 4:
            raise PyMuPDFUnavailable(
                "PyMuPDF4LLM is not installed in the worker environment",
                context={
                    "fix": "uv sync --extra parse",
                    "stderr": stderr_text[:500],
                },
            )
        if proc.returncode == 6:
            raise SandboxLoadFailed(
                "PyMuPDF worker refused to run: seccomp network-egress "
                "block was requested but could not be applied",
                context={
                    "fix": (
                        "either install pyseccomp via `uv sync --extra parse` "
                        "on a Linux host, or set "
                        "`parse.pymupdf_sandbox_network=false` if your "
                        "deployment genuinely needs network during parse"
                    ),
                    "stderr": stderr_text[:500],
                },
            )
        raise PyMuPDFCrashed(
            f"PyMuPDF worker exited with code {proc.returncode} on {source}",
            context={
                "source": str(source),
                "exit_code": proc.returncode,
                "stderr": stderr_text[:2000],
            },
            recoverable=True,
        )

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise PyMuPDFCrashed(
            "PyMuPDF worker produced unparseable output",
            context={
                "source": str(source),
                "stdout_prefix": (stdout or b"")[:500].decode(
                    "utf-8", errors="replace"
                ),
            },
        ) from e

    return PyMuPDFConversion.model_validate(payload)
