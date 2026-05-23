"""Docling adapter — primary parser for PDFs, DOCX, HTML, PPTX, XLSX.

Docling itself is invoked in a subprocess via `python -m
memex.parse.docling_worker`. Running it out-of-process contains its
crashes (segfaults, native-code aborts) — the parent agent keeps
running, and the caller surfaces the failure through the parser
circuit breaker. The subprocess additionally installs a seccomp-bpf
network-egress block before importing docling, so any phone-home
attempt fails at the kernel level. See `parse/docling_worker.py`,
`parse/sandbox.py`, and GUIDELINES.md Part VI "Sandboxed parsing".

The parent process never imports Docling; the worker module does.
This lets the agent layer be imported without the `[parse]` extras
installed.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import structlog
from pydantic import BaseModel

from memex.core.errors import MemexError

logger = structlog.get_logger(__name__)


class DoclingPageOutput(BaseModel):
    """Per-page Docling output — the rendered Markdown, the page
    number, and Docling's self-reported confidence (used by the
    parse-stage VLM-escalation router)."""

    page: int
    markdown: str
    confidence: float


class FigureMetadata(BaseModel):
    """Per-figure metadata serialized from the Docling worker's
    `doc.pictures` array.

    The image bytes themselves are NOT serialized across IPC — the
    chart-OCR backend re-renders each figure from the source PDF via
    `pypdfium2`, mirroring how `vlm_backend.py` renders pages. The
    bbox is in PDF user-space coordinates (origin bottom-left); the
    caller is responsible for any DPI / coord-space conversion.

    `caption` is whatever Docling associates with the figure (often
    empty or a one-line caption from the source).

    `classification` is Docling's `PictureClassifier` top-1 prediction
    (e.g. `bar_chart`, `line_chart`, `logo`, `flow_chart`,
    `photograph`, `engineering_drawing`, etc.) — used by the chart-OCR
    backend to skip non-chart figures before invoking the model. None
    when the worker had picture classification disabled (the v1
    payload format) or when no prediction was emitted.
    """

    page_no: int
    bbox: tuple[float, float, float, float]
    caption: str | None = None
    classification: str | None = None
    classification_confidence: float = 0.0


class DoclingConversion(BaseModel):
    """The full output the Docling worker returns via IPC — stitched
    Markdown, per-page outputs, figure / table / equation counts, and
    (when chart-OCR pre-filter is enabled) the FigureMetadata list."""

    markdown: str
    pages: list[DoclingPageOutput]
    docling_version: str | None = None
    figure_count: int = 0
    table_count: int = 0
    equation_count: int = 0
    # P3.3 Session 2: figure metadata for the chart-OCR pass. Default
    # empty so older callers / workers without the field stay
    # forward-compatible.
    figures: list[FigureMetadata] = []


class DoclingUnavailable(MemexError):
    """Docling is not installed. Install with `uv sync --extra parse`."""


class DoclingTimeout(MemexError):
    """Docling exceeded its per-document timeout."""


class DoclingCrashed(MemexError):
    """The Docling subprocess exited non-zero (segfault, abort, etc.)."""


class SandboxLoadFailed(MemexError):
    """The worker refused to run because the requested seccomp filter
    couldn't be installed. See `parse/sandbox.py`."""


_WORKER_MODULE = "memex.parse.docling_worker"


async def convert(
    source: Path,
    *,
    timeout_s: int,
    sandbox_network: bool = True,
    force_ocr: bool | None = None,
) -> DoclingConversion:
    """Convert `source` to markdown by spawning the Docling worker.

    Runs `python -m memex.parse.docling_worker <source>` as a child
    process. The child's stdout is the JSON-serialised
    `DoclingConversion`; non-zero exit signals a crash (segfault,
    abort, missing dep). We never import Docling in the parent — the
    crash containment is the whole point.

    With `sandbox_network=True` (default), the child also installs a
    seccomp network-egress block before importing docling — see
    `memex/parse/sandbox.py`. Disable only when the user has confirmed
    they need network access during parse; the README documents
    pre-fetching models with `huggingface-cli download` as the
    sandbox-friendly path.

    `force_ocr` overrides the `MEMEX_PARSE_DOCLING_OCR` env-var default
    for this single call. The PyMuPDF pre-filter's mixed-content
    routing path uses `force_ocr=True` so image-embedded text in
    charts, screenshots, and diagrams is OCR'd alongside Docling's
    native-text extraction. `None` (default) leaves the env-var
    setting alone.

    Wraps the call in `asyncio.wait_for(timeout_s)` so a hung Docling
    can't wedge the agent; on timeout we terminate the child.
    """
    log = logger.bind(
        source=str(source),
        timeout_s=timeout_s,
        sandbox_network=sandbox_network,
        force_ocr=force_ocr,
    )
    log.info("docling.start")

    env = os.environ.copy()
    env["MEMEX_PARSE_SANDBOX_NETWORK"] = "1" if sandbox_network else "0"
    if force_ocr is not None:
        env["MEMEX_PARSE_DOCLING_OCR"] = "1" if force_ocr else "0"

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
            raise DoclingTimeout(
                f"Docling exceeded {timeout_s}s on {source}",
                context={"source": str(source), "timeout_s": timeout_s},
                recoverable=True,
            ) from e
    except BaseException:
        # Any exit other than a clean `communicate()` return — including
        # `CancelledError` from a cancelling parent — needs to tear the
        # child down. `asyncio.wait_for` already cancels the inner task
        # on timeout, but `proc` may still be alive holding pipes.
        # SIGTERM, brief wait, then SIGKILL; pipes drain when the child
        # exits and Python's wait() reaps it (no zombies).
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
        # Worker exit codes:
        #   2 — usage error  (won't happen via this entry)
        #   3 — source not found
        #   4 — docling_unavailable
        #   5 — exception during conversion
        #   6 — sandbox requested but couldn't be applied
        if proc.returncode == 4:
            raise DoclingUnavailable(
                "Docling is not installed in the worker environment",
                context={
                    "fix": "uv sync --extra parse",
                    "stderr": stderr_text[:500],
                },
            )
        if proc.returncode == 6:
            raise SandboxLoadFailed(
                "Docling worker refused to run: seccomp network-egress "
                "block was requested but could not be applied",
                context={
                    "fix": (
                        "either install pyseccomp via `uv sync --extra parse` "
                        "on a Linux host, or set "
                        "`parse.docling_sandbox_network=false` if your "
                        "deployment genuinely needs network during parse"
                    ),
                    "stderr": stderr_text[:500],
                },
            )
        raise DoclingCrashed(
            f"Docling worker exited with code {proc.returncode} on {source}",
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
        raise DoclingCrashed(
            "Docling worker produced unparseable output",
            context={
                "source": str(source),
                "stdout_prefix": (stdout or b"")[:500].decode(
                    "utf-8", errors="replace"
                ),
            },
        ) from e

    return DoclingConversion.model_validate(payload)
