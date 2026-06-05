"""Standalone image file → single-page PDF conversion for the parse pipeline (ADR-0020).

A standalone image (PNG/JPEG/WebP/BMP/TIFF/GIF) is a one-page scan: pypdfium2 — which the
VLM-escalation page rasteriser uses — is PDF-only, so the image is wrapped into a 1-page PDF up
front and run through the scan→VLM route UNCHANGED (`pipeline._parse_scan_with_vlm`), mirroring the
Office→PDF precedent (`office_convert.py`). The image renders back to a bitmap inside `convert_pages`
(`_render_page_to_image`), so figure handling, the VLM cache, and grounding are all the scan path.

Like the Office route, the converted PDF is cached in the document's vault dir (`converted.pdf`) and
reused on re-parse: `PIL.Image.save(..., "PDF")` stamps a fresh `time.gmtime()` CreationDate/ModDate
into the PDF Info trailer, so re-converting every parse would churn the bytes — and therefore the
content-addressed VLM cache key (`sha256(pdf_bytes)`) — exactly the LibreOffice CreationDate problem.
Caching keeps it byte-stable.

A camera/phone photo carries an EXIF Orientation tag that PIL does NOT auto-apply (it leaves the
pixels as-stored), so the image is run through `ImageOps.exif_transpose` first — otherwise a
photographed page (a named use case) would transcribe sideways. It is a pixel no-op when there is no
orientation tag.

Heavy work (PIL decode + save) runs under `asyncio.to_thread`; PIL is lazy-imported inside the
function so the module stays importable without the [parse] extra (the `office_convert`/`pdf_render`
discipline).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import structlog

from memex.core.errors import MemexError

logger = structlog.get_logger(__name__)

# Raster image suffixes routed through PIL → a 1-page PDF → the scan→VLM route. HEIC/AVIF are
# deliberately absent (they need a separate decode dependency; their `ftyp` brands also stay
# rejected at validation — ADR-0020). `.tif`/`.tiff` + `.jpg`/`.jpeg` are both spelled out.
IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
)

# DPI for the PDF page box: at 144, page_points = pixels/144*72 = pixels/2, so the scan route's
# scale-2.0 `_render_page_to_image` reproduces the image at its NATIVE pixel resolution (no waste
# up/downscale; the VLM processor's `max_pixels` still caps a very large image). A multi-frame
# source (multi-page TIFF / animated GIF) yields its FIRST frame in v1 (ADR-0020 revisit-when).
_PDF_RESOLUTION_DPI: float = 144.0


class ImageConversionError(MemexError):
    """A source image could not be decoded / wrapped into a PDF (corrupt, truncated, or an
    unsupported/over-large pixel grid)."""


async def convert_image_to_pdf(source: Path, out_dir: Path) -> Path:
    """Render an image file to a single-page PDF at `out_dir/{source.stem}.pdf`.

    The first frame is taken (a multi-page TIFF / animated GIF → page 1, v1). The EXIF Orientation
    tag is honoured (a phone photo is uprighted). Non-RGB/L modes (RGBA, palette, CMYK, LA) are
    converted to RGB — a PDF image XObject has no alpha channel. Raises `ImageConversionError` on a
    corrupt/unreadable/over-large image.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def _run() -> Path:
        # lazy — [parse] deps, same discipline as office_convert/pdf_render
        from PIL import Image, ImageOps

        out = out_dir / f"{source.stem}.pdf"
        try:
            with Image.open(source) as im:
                im.load()  # decode the first frame
                # Honour the EXIF Orientation tag: PIL leaves pixels AS-STORED, so a phone photo
                # (orientation 6/8) would otherwise transcribe SIDEWAYS — and "a photographed page"
                # is a named ADR-0020 use case. `exif_transpose` rotates to the intended upright
                # orientation and is a pixel no-op (a plain copy) when there's no orientation tag,
                # so the determinism of the non-EXIF path is unchanged. (`or im` guards the older-PIL
                # None return; pillow>=10 always returns an image.)
                upright = ImageOps.exif_transpose(im) or im
                # A PDF image XObject can't hold an alpha channel; RGBA/palette/CMYK/LA → RGB.
                # Grayscale "L" and "RGB" save directly.
                pdf_image = upright if upright.mode in ("RGB", "L") else upright.convert("RGB")
                pdf_image.save(out, "PDF", resolution=_PDF_RESOLUTION_DPI)
        except (OSError, ValueError, Image.DecompressionBombError) as e:
            # UnidentifiedImageError subclasses OSError (corrupt/not-an-image); ValueError = a bad
            # mode/param; DecompressionBombError = a maliciously huge pixel grid.
            raise ImageConversionError(
                "could not convert the image to a PDF (corrupt, truncated, or over-large)",
                context={"source": str(source), "suffix": source.suffix, "error": str(e)[:200]},
            ) from e
        return out

    pdf = await asyncio.to_thread(_run)
    logger.bind(source=str(source)).info("image.converted", pdf=str(pdf), bytes=pdf.stat().st_size)
    return pdf
