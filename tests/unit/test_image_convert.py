"""Unit tests for the standalone-image → 1-page-PDF converter (ADR-0020).

`convert_image_to_pdf` wraps a raster image into a single-page PDF for the scan→VLM route.
Pins: every supported pixel mode (RGB/L direct; RGBA/P/CMYK/LA via the RGB conversion — a PDF
image is rendered deterministically without an alpha channel) and every accepted format
(PNG/JPEG/WebP/BMP/TIFF/GIF) yields a VALID single-page PDF at `out_dir/{stem}.pdf`; a multi-frame
source yields its FIRST frame as one page (the v1 contract); corrupt bytes raise a typed
`ImageConversionError`. Uses real PIL against tiny in-memory fixtures (no GPU, no models).
"""

from __future__ import annotations

from pathlib import Path

import pypdfium2
import pytest
from PIL import Image

from memex.parse.image_convert import ImageConversionError, convert_image_to_pdf


def _page_count(pdf_path: Path) -> int:
    pdf = pypdfium2.PdfDocument(str(pdf_path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def _assert_single_page_pdf(pdf: Path) -> None:
    assert pdf.is_file()
    assert pdf.read_bytes()[:5] == b"%PDF-"
    assert _page_count(pdf) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA", "P", "CMYK", "LA"])
async def test_each_mode_converts_to_single_page_pdf(tmp_path: Path, mode: str) -> None:
    # TIFF is the fixture container — it holds every mode (PNG can't write CMYK/LA). The
    # converter decodes the mode and either saves it directly (RGB/L) or flattens to RGB
    # (RGBA/P/CMYK/LA — a PDF image is rendered deterministically without an alpha channel).
    src = tmp_path / f"{mode.lower()}.tif"
    Image.new(mode, (8, 6)).save(src, "TIFF")
    out = await convert_image_to_pdf(src, tmp_path / "out")
    assert out == tmp_path / "out" / f"{mode.lower()}.pdf"
    _assert_single_page_pdf(out)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fmt", "ext"),
    [
        ("PNG", "png"),
        ("JPEG", "jpg"),
        ("WEBP", "webp"),
        ("BMP", "bmp"),
        ("TIFF", "tif"),
        ("GIF", "gif"),
    ],
)
async def test_each_format_converts_to_single_page_pdf(tmp_path: Path, fmt: str, ext: str) -> None:
    src = tmp_path / f"shot.{ext}"
    Image.new("RGB", (10, 8), (120, 60, 30)).save(src, fmt)
    out = await convert_image_to_pdf(src, tmp_path / "out")
    _assert_single_page_pdf(out)


@pytest.mark.asyncio
async def test_multiframe_gif_yields_single_page(tmp_path: Path) -> None:
    # An animated GIF (2 frames) → a single-page PDF (the first frame, the v1 contract).
    src = tmp_path / "anim.gif"
    frames = [Image.new("P", (8, 8), 1), Image.new("P", (8, 8), 2)]
    frames[0].save(src, "GIF", save_all=True, append_images=frames[1:], duration=100)
    out = await convert_image_to_pdf(src, tmp_path / "out")
    _assert_single_page_pdf(out)


@pytest.mark.asyncio
async def test_corrupt_image_raises_typed_error(tmp_path: Path) -> None:
    src = tmp_path / "broken.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n not a real image, truncated garbage")
    with pytest.raises(ImageConversionError) as exc:
        await convert_image_to_pdf(src, tmp_path / "out")
    # The typed error carries actionable context (source + suffix), per the MemexError contract.
    assert exc.value.context["suffix"] == ".png"
    assert "broken.png" in str(exc.value.context["source"])


@pytest.mark.asyncio
async def test_out_dir_is_created(tmp_path: Path) -> None:
    src = tmp_path / "pic.png"
    Image.new("RGB", (4, 4)).save(src, "PNG")
    nested = tmp_path / "a" / "b" / "out"  # does not exist yet
    out = await convert_image_to_pdf(src, nested)
    _assert_single_page_pdf(out)
