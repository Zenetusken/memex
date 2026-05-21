"""Chart-OCR pass over Docling figures (P3.3).

Renders each figure crop from the source PDF (via pypdfium2) and passes
it through the chart-OCR model loaded by `ModelRegistry.use("chart_ocr")`
— default `google/deplot` per the Session 1 verdict — with a strict
"linearise this chart as a table" instruction. The extracted markdown is
stitched into the parent document's markdown by replacing the
`<!-- image -->` placeholder with the image marker + the extracted data.

Mirrors `vlm_backend.py`'s pattern: image bytes never leave the
orchestrator process; the Docling worker stays CPU-only; one context
acquisition per document; per-figure exceptions return rather than
raise so the caller can route to fallback (leave the `<!-- image -->`
placeholder unchanged).

Heavy deps (pypdfium2, transformers, torch) are imported inside the
functions so the rest of the package stays importable without the
[parse] + [models] extras.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from memex.core.errors import MemexError
from memex.models.registry import get_registry
from memex.parse.docling_backend import FigureMetadata

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

# DePlot's published example uses this exact prompt; deviating produces
# worse extraction quality (the model is trained to expect this prompt
# token sequence).
_PROMPT = "Generate underlying data table of the figure below:"

# DePlot's table outputs typically run 50-300 tokens. 512 leaves
# headroom for dense charts; defends against runaway generation on
# under-trained inputs (e.g., chart imagery the model has never seen
# the genre of).
_MAX_NEW_TOKENS = 512

# Render DPI for figure crops. DePlot was trained on screenshots at
# roughly 150-200 DPI; we render at scale=2.5 ≈ 180 DPI by default. The
# downstream processor will resize to its own internal max-pixels
# budget. Override via env var if quality calls for it.
_DEFAULT_RENDER_SCALE = 2.5


class ChartOCRUnavailable(MemexError):
    """The chart-OCR stack is not installed (transformers / torch / pypdfium2)."""


class PDFFigureRenderError(MemexError):
    """A figure could not be cropped + rasterised to an image."""


@dataclass(frozen=True)
class ChartOCROutput:
    """Per-figure result. The `markdown` field is the model's raw
    output — typically a linearised table or `<insufficient data>`
    style refusal token sequence. Empty string means the model ran
    but produced nothing parseable; the caller can choose to elide
    such outputs from the stitched markdown."""

    page_no: int
    bbox: tuple[float, float, float, float]
    markdown: str


def _render_figure_to_image(
    pdf_path: Path,
    page_no: int,
    bbox: tuple[float, float, float, float],
    scale: float = _DEFAULT_RENDER_SCALE,
):
    """Render the bbox region on `page_no` to a PIL Image.

    Docling reports bboxes in PDF user-space coords (points; origin
    bottom-left). pypdfium2 renders pages bitmap-space (pixels;
    origin top-left). We render the full page at `scale`, then crop
    the bitmap to the bbox region with the y-axis flipped. This is
    simpler than passing crop bounds to pypdfium2 directly (which
    requires a different coord convention than Docling uses) and the
    extra full-page render cost is negligible compared to model
    inference.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as e:
        raise ChartOCRUnavailable(
            "pypdfium2 is required for chart figure rasterisation; "
            "install with `uv sync --extra parse`",
            context={"underlying": str(e)},
        ) from e

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        idx = page_no - 1
        if not 0 <= idx < len(doc):
            raise PDFFigureRenderError(
                f"page {page_no} out of range",
                context={"page": page_no, "page_count": len(doc)},
            )
        page = doc[idx]
        page_width = page.get_width()
        page_height = page.get_height()
        # Full-page bitmap.
        bitmap = page.render(scale=scale)
        full = bitmap.to_pil()

        # Convert PDF coords → pixel coords. PDF origin is bottom-left;
        # PIL origin is top-left. The bbox is (x0, y0, x1, y1) with
        # y0 < y1 in PDF coords; in pixel coords we flip.
        x0, y0, x1, y1 = bbox
        # Clamp to page bounds (Docling occasionally reports slightly
        # out-of-bounds bboxes for figures that bleed off the page).
        x0 = max(0.0, min(x0, page_width))
        x1 = max(0.0, min(x1, page_width))
        y0 = max(0.0, min(y0, page_height))
        y1 = max(0.0, min(y1, page_height))
        if x1 <= x0 or y1 <= y0:
            raise PDFFigureRenderError(
                f"degenerate bbox after clamping: ({x0},{y0},{x1},{y1})",
                context={"page": page_no, "bbox": bbox},
            )
        px = scale
        px_x0 = int(x0 * px)
        px_x1 = int(x1 * px)
        # Y flip: top of crop in pixel coords = (page_height - y1) * scale
        px_y0 = int((page_height - y1) * px)
        px_y1 = int((page_height - y0) * px)
        return full.crop((px_x0, px_y0, px_x1, px_y1))
    finally:
        doc.close()


def _chart_ocr_transcribe_sync(
    handle, image, prompt: str, max_new_tokens: int
) -> str:
    """Synchronous transcription; called via asyncio.to_thread.

    Mirrors DePlot's published example: processor builds the multi-
    modal batch, model.generate runs greedy decode, processor decodes
    the output ids. No chat-template wrapping (Pix2Struct doesn't
    have one — the prompt is just a text prefix).
    """
    import torch

    inputs = handle.processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(handle.model.device)

    with torch.inference_mode():
        outputs = handle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    decoded = handle.processor.decode(outputs[0], skip_special_tokens=True)
    return decoded.strip()


async def _extract_with_handle(
    handle: object,
    source_pdf: Path,
    figure: FigureMetadata,
    max_new_tokens: int,
) -> ChartOCROutput:
    """Internal: render + transcribe one figure given an already-
    acquired chart-OCR handle."""
    log = logger.bind(
        page=figure.page_no,
        bbox=figure.bbox,
        source=str(source_pdf),
    )
    log.info("chart_ocr.start")

    image = await asyncio.to_thread(
        _render_figure_to_image,
        source_pdf,
        figure.page_no,
        figure.bbox,
    )
    markdown = await asyncio.to_thread(
        _chart_ocr_transcribe_sync, handle, image, _PROMPT, max_new_tokens
    )

    log.info("chart_ocr.done", chars=len(markdown))
    return ChartOCROutput(
        page_no=figure.page_no,
        bbox=figure.bbox,
        markdown=markdown,
    )


async def chart_ocr_extract(
    *,
    source_pdf: Path,
    figures: list[FigureMetadata],
    max_new_tokens: int = _MAX_NEW_TOKENS,
) -> list[ChartOCROutput | Exception]:
    """Per-document batch extraction over a list of figures.

    Acquires `registry.use("chart_ocr")` once, iterates `figures`
    sequentially under that single context, releases at the end. Failures
    on individual figures are returned as exception objects in the
    result list (parallel to the input order); the caller decides
    whether to skip the figure or stitch a placeholder.

    Why a list and not a dict: the same `(page_no, bbox)` pair can in
    principle appear twice (rare; layout-engine pathology) and the
    caller wants to know which figure's output is which by position.
    The list preserves input order so `zip(figures, results)` is the
    contract.
    """
    if not figures:
        return []

    registry = get_registry()
    results: list[ChartOCROutput | Exception] = []
    async with registry.use("chart_ocr") as handle:
        for figure in figures:
            try:
                out = await _extract_with_handle(
                    handle, source_pdf, figure, max_new_tokens
                )
                results.append(out)
            except (ChartOCRUnavailable, PDFFigureRenderError) as e:
                results.append(e)
    return results
