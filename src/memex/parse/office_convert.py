"""Office document → PDF conversion for the parse pipeline.

PowerPoint / Word / Excel (and their ODF cousins) can't be rendered by
pypdfium2, which the VLM-escalation and chart-OCR passes use to rasterise pages
and figure crops (both are PDF-only). So an Office source is converted to PDF
up front via headless LibreOffice, and the normal PDF pipeline runs on the
result — Docling, the VLM diagram transcription, and chart-OCR then all operate
on the PDF, so figure bboxes align and diagrams transcribe like any PDF.

The conversion is a real pipeline step, not a one-off: `parse_document` caches
the converted PDF in the document's vault dir (`converted.pdf`) and reuses it on
re-parse, so the PDF bytes — and therefore the content-addressed VLM / chart-OCR
cache keys — stay byte-stable across runs (LibreOffice stamps a fresh
CreationDate each conversion, so re-converting every parse would churn them).

Heavy work is a `soffice` subprocess via `asyncio.to_thread`; the module is pure
stdlib so it stays importable without the [parse]/[models] extras.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

import structlog

from memex.core.errors import MemexError

logger = structlog.get_logger(__name__)

# Office/ODF source suffixes routed through LibreOffice → PDF. Anything here is
# converted before parsing; everything else (.pdf/.md) takes its normal path.
OFFICE_SUFFIXES: frozenset[str] = frozenset(
    {".pptx", ".ppt", ".docx", ".doc", ".xlsx", ".xls", ".odp", ".odt", ".ods"}
)

# Candidate LibreOffice "program" dirs (hold soffice.bin + the bundled .so set).
# Pointing LD_LIBRARY_PATH at the right one works around environments where the
# `soffice` wrapper fails to add its own libs to the loader path (observed here:
# `libreglo.so: cannot open shared object file`, though the file is present in
# the program dir). Harmless where the wrapper already sets the path.
_LIBREOFFICE_PROGRAM_DIRS: tuple[str, ...] = (
    "/usr/lib/libreoffice/program",
    "/usr/lib64/libreoffice/program",
    "/opt/libreoffice/program",
)


class OfficeConversionError(MemexError):
    """LibreOffice was unavailable or failed to convert an Office doc to PDF."""


def _libreoffice_lib_dir() -> str | None:
    """Return the LibreOffice program dir (containing soffice.bin), or None."""
    for d in _LIBREOFFICE_PROGRAM_DIRS:
        if (Path(d) / "soffice.bin").is_file():
            return d
    return None


def _soffice_binary() -> str | None:
    """Resolve the `soffice`/`libreoffice` launcher on PATH, or None."""
    return shutil.which("soffice") or shutil.which("libreoffice")


async def convert_to_pdf(source: Path, out_dir: Path, *, timeout_s: float = 180.0) -> Path:
    """Convert `source` (an Office/ODF document) to PDF inside `out_dir`.

    Returns the produced PDF path (`out_dir/{source.stem}.pdf`). Raises
    `OfficeConversionError` if LibreOffice is missing, the conversion exits
    non-zero, times out, or produces no PDF. A per-conversion `UserInstallation`
    profile under `out_dir` keeps headless runs from clashing with a desktop
    LibreOffice instance's profile lock.
    """
    soffice = _soffice_binary()
    if soffice is None:
        raise OfficeConversionError(
            "LibreOffice (soffice) not found on PATH; it is required to parse "
            "Office documents (pptx/docx/xlsx). Install libreoffice.",
            context={"source": str(source), "suffix": source.suffix},
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    lib_dir = _libreoffice_lib_dir()
    if lib_dir:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = lib_dir + (os.pathsep + existing if existing else "")

    profile = out_dir / ".lo_profile"
    cmd = [
        soffice,
        "--headless",
        "--norestore",
        f"-env:UserInstallation=file://{profile}",
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        str(source),
    ]
    log = logger.bind(source=str(source), out_dir=str(out_dir))

    def _run() -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(  # noqa: S603 — fixed argv, no shell, trusted paths
            cmd, capture_output=True, timeout=timeout_s, check=False, env=env
        )

    try:
        proc = await asyncio.to_thread(_run)
    except subprocess.TimeoutExpired as e:
        raise OfficeConversionError(
            "LibreOffice conversion timed out",
            context={"source": str(source), "timeout_s": timeout_s},
            recoverable=True,
        ) from e
    finally:
        # The throwaway profile is large + useless after the run.
        shutil.rmtree(profile, ignore_errors=True)

    pdf = out_dir / f"{source.stem}.pdf"
    if proc.returncode != 0 or not pdf.is_file():
        raise OfficeConversionError(
            "LibreOffice failed to convert the document to PDF",
            context={
                "source": str(source),
                "returncode": proc.returncode,
                "stderr": proc.stderr[-600:].decode("utf-8", errors="replace"),
            },
        )
    log.info("office.converted", pdf=str(pdf), bytes=pdf.stat().st_size)
    return pdf
