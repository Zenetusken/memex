"""Unit tests for the webui source-preview PDF rasteriser (`parse/pdf_render.py`).

The render itself is also exercised live (the webui page route returns valid
PNGs); these pin the page-count + render happy path and — load-bearing for the
route guards — that a corrupt/out-of-range PDF surfaces as `PDFPreviewError`
(one type the route catches → degrades to no-pane, never a 500).
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path

import pytest

from memex.parse.pdf_render import PDFPreviewError, pdf_page_count, render_pdf_page_png

# A minimal but valid 1-page PDF (no xref table — pdfium rebuilds it). One empty
# 120×120 page, enough to page-count and rasterise.
_MINIMAL_PDF = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 120 120] >>
endobj
trailer
<< /Root 1 0 R >>
%%EOF
"""


def _write(tmp_path: Path, data: bytes) -> Path:
    p = tmp_path / "doc.pdf"
    p.write_bytes(data)
    return p


def test_page_count_and_render_minimal_pdf(tmp_path: Path) -> None:
    pdf = _write(tmp_path, _MINIMAL_PDF)
    assert pdf_page_count(pdf) == 1
    png = render_pdf_page_png(pdf, 0)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")  # PNG magic


def test_render_out_of_range_raises(tmp_path: Path) -> None:
    pdf = _write(tmp_path, _MINIMAL_PDF)
    with pytest.raises(PDFPreviewError):
        render_pdf_page_png(pdf, 5)


def test_corrupt_pdf_raises_preview_error(tmp_path: Path) -> None:
    """A truncated / non-PDF file must raise PDFPreviewError, not a raw
    PdfiumError (which the webui route guards wouldn't catch → would 500)."""
    pdf = _write(tmp_path, b"this is definitely not a pdf")
    with pytest.raises(PDFPreviewError):
        pdf_page_count(pdf)
    with pytest.raises(PDFPreviewError):
        render_pdf_page_png(pdf, 0)


def test_concurrent_renders_are_serialized(tmp_path: Path) -> None:
    """pypdfium2 is NOT thread-safe — concurrent renders SEGFAULTED the webui
    (a deck's ~30 lazy <img> page requests). The module lock must serialize them;
    this fires many parallel renders and asserts they all return valid PNGs (no
    crash, no corruption). A regression here would likely segfault the run — loud."""
    pdf = _write(tmp_path, _MINIMAL_PDF)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        out = list(ex.map(lambda _: render_pdf_page_png(pdf, 0), range(24)))
    assert len(out) == 24
    assert all(png.startswith(b"\x89PNG\r\n\x1a\n") for png in out)
