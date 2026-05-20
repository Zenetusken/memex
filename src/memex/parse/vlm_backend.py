"""VLM fallback for pages Docling can't handle confidently.

Renders the source PDF page to an image (via pypdfium2) and passes it
through the VLM loaded by `ModelRegistry.use("vlm")` with a strict
"transcribe as clean markdown" instruction. The result replaces
Docling's output for that page; the manifest records the routing
decision either way.

Heavy deps (pypdfium2, transformers, torch) are imported inside the
functions so the rest of the package stays importable without the
[parse] + [models] extras.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from memex.core.errors import MemexError
from memex.models.registry import get_registry
from memex.parse.docling_backend import DoclingPageOutput

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

_PROMPT = (
    "You are converting a single document page to clean Markdown. "
    "Preserve structure:\n"
    "- Headings (#, ##, ###)\n"
    "- Tables (GFM table syntax)\n"
    "- Bulleted and numbered lists\n"
    "- Equations as LaTeX ($inline$ or $$display$$)\n"
    "- Code blocks (```fenced)\n\n"
    "Output ONLY Markdown for the page contents — no preface, "
    "no commentary, no closing remarks."
)
# A single page's transcribed Markdown is typically 500–800 tokens; 1024
# leaves headroom for table/code-heavy pages and defends against runaway
# generation. Tuned per the CUDA audit.
_MAX_NEW_TOKENS = 1024


class VLMUnavailable(MemexError):
    """The VLM stack is not installed (transformers / torch / pypdfium2)."""


class PDFRenderError(MemexError):
    """The page could not be rasterised to an image."""


def _render_page_to_image(pdf_path: Path, page_number: int):
    """Render a single 1-indexed page to a PIL Image at ~144 DPI.

    Qwen-VL processors use `max_pixels = 1280 * 28 * 28 ≈ 1.0 M` and
    downscale anything bigger. Rendering at scale=2.0 (~144 DPI ≈ 1.9 M
    pixels for letter pages) is half the rasterisation work of the
    previous scale=2.78 (~200 DPI ≈ 3.7 M pixels) while still leaving
    the processor a clean resize step. CUDA audit, item 9.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as e:
        raise VLMUnavailable(
            "pypdfium2 is required for VLM page rasterisation; "
            "install with `uv sync --extra parse`",
            context={"underlying": str(e)},
        ) from e

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        idx = page_number - 1
        if not 0 <= idx < len(doc):
            raise PDFRenderError(
                f"page {page_number} out of range",
                context={"page": page_number, "page_count": len(doc)},
            )
        page = doc[idx]
        bitmap = page.render(scale=2.0)
        return bitmap.to_pil()
    finally:
        doc.close()


def _vlm_transcribe_sync(handle, image, prompt: str, max_new_tokens: int) -> str:
    """Synchronous transcription; called via asyncio.to_thread."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = handle.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = handle.processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(handle.model.device)

    with torch.inference_mode():
        outputs = handle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Strip the prompt prefix from the decode.
    generated = outputs[:, inputs["input_ids"].shape[1] :]
    decoded = handle.processor.batch_decode(
        generated, skip_special_tokens=True
    )
    return (decoded[0] if decoded else "").strip()


async def _convert_with_handle(
    handle: object,
    source_pdf: Path,
    page_number: int,
    max_new_tokens: int,
) -> DoclingPageOutput:
    """Internal: render + transcribe one page given an already-acquired VLM."""
    log = logger.bind(page=page_number, source=str(source_pdf))
    log.info("vlm.start")

    image = await asyncio.to_thread(
        _render_page_to_image, source_pdf, page_number
    )
    markdown = await asyncio.to_thread(
        _vlm_transcribe_sync, handle, image, _PROMPT, max_new_tokens
    )

    log.info("vlm.done", chars=len(markdown))
    return DoclingPageOutput(
        page=page_number, markdown=markdown, confidence=1.0
    )


async def convert_page(
    *,
    source_pdf: Path,
    page_number: int,
    max_new_tokens: int = _MAX_NEW_TOKENS,
) -> DoclingPageOutput:
    """Single-page convenience: acquire the VLM, transcribe, release.

    For multi-page jobs from the same source, call `convert_pages`
    instead — one context acquisition per document is much cheaper
    than one per page.
    """
    registry = get_registry()
    async with registry.use("vlm") as handle:
        return await _convert_with_handle(
            handle, source_pdf, page_number, max_new_tokens
        )


async def convert_pages(
    *,
    source_pdf: Path,
    page_numbers: list[int],
    max_new_tokens: int = _MAX_NEW_TOKENS,
) -> dict[int, DoclingPageOutput | Exception]:
    """Per-document batch transcription. CUDA audit, item 8.

    Acquires `registry.use("vlm")` once, iterates `page_numbers`
    sequentially under that single context, releases at the end.
    Failures on individual pages are returned as exception objects in
    the result dict; the caller is responsible for routing each to
    either an escalated `PageDecision(engine="vlm")` or a fallback
    `PageDecision(engine="docling", rationale="VLM escalation failed: ...")`.

    Why this matters: today the `ModelRegistry` keeps the VLM
    resident after first load, so per-page calls don't reload. But
    once the registry grows OOM-driven eviction (Phase 4 hardening),
    per-page acquisition would thrash load/unload across a single
    document's parse. Acquiring once per document is the correct
    boundary regardless.
    """
    if not page_numbers:
        return {}

    registry = get_registry()
    results: dict[int, DoclingPageOutput | Exception] = {}
    async with registry.use("vlm") as handle:
        for page_number in page_numbers:
            try:
                results[page_number] = await _convert_with_handle(
                    handle, source_pdf, page_number, max_new_tokens
                )
            except (VLMUnavailable, PDFRenderError) as e:
                results[page_number] = e
    return results
