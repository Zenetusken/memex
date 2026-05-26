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
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import structlog

from memex.core.config import get_settings
from memex.core.errors import MemexError
from memex.models.registry import ChartOCRHandle, get_registry
from memex.parse.chart_ocr_cache import ChartOCRCache
from memex.parse.docling_backend import FigureMetadata

if TYPE_CHECKING:
    from PIL import Image

logger = structlog.get_logger(__name__)

# DePlot's published example uses this exact prompt; deviating produces
# worse extraction quality (the model is trained to expect this prompt
# token sequence).
_PROMPT_DEPLOT = "Generate underlying data table of the figure below:"

# VLM-style chart-extraction prompt with the UNREADABLE-escape-hatch
# pattern from the ChartHal (Sep 2025) hallucination benchmark and the
# Losing-the-Plot (Sep 2025) study on chart degradation. The
# `UNREADABLE` token + explicit "do NOT infer" clause is the most
# effective single technique reported for reducing fabrication on OOD
# chart imagery. Used when the chart-OCR backend is loaded as a VLM
# (e.g., Qwen2.5-VL-7B) rather than a Pix2Struct-derivative.
_PROMPT_VLM = (
    "Extract every data series from this chart as a markdown table. "
    "Columns: [x_label, series_name, value, unit]. "
    "Rules:\n"
    "1. Only output values you can READ DIRECTLY off the chart's axis "
    "labels, value labels, or legend. Do NOT infer values from the "
    "chart's visual shape or by interpolation.\n"
    "2. If the chart legend, axis labels, or values are ambiguous, "
    "illegible, missing, or this image is not a real data chart (e.g., "
    "a diagram, screenshot, photo, logo), output ONLY the literal "
    "token UNREADABLE on a single line.\n"
    "3. Never fabricate plausible-looking column headers (e.g., sports "
    "statistics, financial metrics) for charts whose subject you "
    "cannot identify.\n"
    "4. Return ONLY the markdown table or UNREADABLE — no commentary, "
    "no surrounding prose, no explanations.\n"
)

# When the VLM returns this literal token (case-insensitive, ignoring
# whitespace), the backend treats the extraction as a refusal — the
# stitch step will leave the `<!-- image -->` placeholder unchanged.
_UNREADABLE_TOKEN = "UNREADABLE"  # noqa: S105  # sentinel marker, not a credential

# DePlot-output post-processing patterns. The model's raw output uses
# `<0x0A>` (Pix2Struct's newline byte sequence) instead of actual `\n`
# characters; we normalise so the resulting markdown is readable by
# both humans and the downstream agent's literal-presence rule.
_PIX2STRUCT_NEWLINE_RE = re.compile(r"\s*<0x0A>\s*")

# Patterns indicating that DePlot collapsed a multi-series chart to a
# single column — the resulting table is too ambiguous to ground on
# (you can't tell which series each value belongs to). Reject these
# at the stitch step. Example: TSMC Lithography chart on the CUDA
# deck has "Power" and "Density" as Series1/Series2; DePlot extracts
# headers like `"TITLE | Series1 | Series2"` or just `"Series1"`,
# making "is 0.8 the power or density value?" unanswerable.
_AMBIGUOUS_HEADER_RE = re.compile(
    r"\bSeries\s*\d+\b",
    re.IGNORECASE,
)

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

# Minimum figure area (in PDF user-space squared points) to bother
# running chart-OCR on. Below this threshold, the figure is almost
# certainly a page-number badge, watermark, or decorative element
# rather than a real chart. ~140×140 points ≈ 2 inches square at
# 72 DPI, which is the minimum chart size we've seen on the slide-
# decks corpus.
_MIN_FIGURE_AREA_SQPT = 20000.0

# P3.3 v2: Docling's `DocumentPictureClassifier` (v2.5) class names we
# accept as "actual chart with extractable data." Anything else
# (logo, flow_chart, photograph, engineering_drawing, screenshot,
# icon, table, etc.) is dropped from the chart-OCR pass. This is the
# pre-filter that prevents DePlot's OOD-hallucination cascade on
# non-chart content. On the canonical CUDA deck this drops
# 245 picture objects → ~26 actual chart candidates (89% reduction).
# `flow_chart` is intentionally EXCLUDED — those are architecture /
# block diagrams without extractable rows+columns, and DePlot
# fabricates plausible-looking sports-statistics tables on them.
_CHART_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "bar_chart",
        "line_chart",
        "pie_chart",
        "scatter_plot",
        "box_plot",
        "tabular_chart",  # if Docling emits this distinct type
    }
)

# Minimum classifier confidence to trust the classification. Below
# this we treat the classification as "unknown" and fall back to the
# legacy "include everything that passes area filter" behavior — so a
# misclassification of a real chart as e.g. 'icon' at 0.3 confidence
# doesn't silently drop it. Tuned conservative; Docling's v2.5
# classifier is typically >0.85 confident when it has a clear answer.
_MIN_CLASSIFICATION_CONFIDENCE = 0.50


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
) -> Image.Image:
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
        # N7 (audit 2026-05-20): `bitmap.to_pil()` returns a view into
        # the bitmap's C-owned buffer, and PIL's `.crop()` is lazy —
        # it returns an Image that references the parent's pixel data
        # without copying. When `doc.close()` runs in the finally
        # block, the C buffer is freed; any subsequent read of the
        # cropped image (size, save, processor preprocessing) hits
        # freed memory. `.copy()` forces eager allocation + memcpy
        # BEFORE the doc closes so the returned crop owns its bytes.
        return full.crop((px_x0, px_y0, px_x1, px_y1)).copy()
    finally:
        doc.close()


def _is_vlm_handle(handle: ChartOCRHandle) -> bool:
    """True when the chart-OCR handle wraps a VLM-style model (Qwen-VL,
    LLaVA, etc.) rather than a Pix2Struct-style chart specialist. The
    inference path differs: VLMs use chat templates + messages;
    Pix2Struct uses a direct text+image processor call.
    """
    cls = type(handle.model).__name__
    return any(token in cls for token in ("VL", "Vision", "Llava", "Internvl"))


def _is_onechart_handle(handle: ChartOCRHandle) -> bool:
    """True when the chart-OCR handle wraps the OneChart model (P3.3-b).
    OneChart uses a custom Vary-derived architecture with a `.chat()`
    interface instead of `.generate()`, and emits an auxiliary
    `reliable_check` token to indicate self-consistency. The class
    name is `OneChartOPTForCausalLM` or similar — we match on
    "onechart" case-insensitively for robustness against minor naming
    changes in the upstream HF repo.
    """
    cls = type(handle.model).__name__.lower()
    return "onechart" in cls


def _is_nemotron_parse_handle(handle: ChartOCRHandle) -> bool:
    """True when the chart-OCR handle wraps the NVIDIA Nemotron-Parse
    model (P3.3-c, Path C). The class name is
    `NemotronParseForConditionalGeneration`."""
    cls = type(handle.model).__name__.lower()
    return "nemotronparse" in cls or "nemotron_parse" in cls


# Nemotron-Parse-v1.2 prompt prefix per the model card. The four
# task tokens declare:
#   <predict_bbox>     — emit bounding-box metadata
#   <predict_classes>  — emit element-class metadata
#   <output_markdown>  — format extracted text as markdown
#   <predict_no_text_in_pic> — skip OCR of text WITHIN picture
#       elements (we want the chart-block content, not in-image
#       text fragments)
_PROMPT_NEMOTRON_PARSE = (
    "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
)


# Nemotron-Parse output structure tokens. The model emits:
#   - `<bbox_*>` / `<class_*>` — element bounding boxes + classes
#   - `<x_0.XXXX>` / `<y_0.XXXX>` — normalised coordinate markers
#   - `<md_*>` / `<patch_*>` / `<extra_*>` — additional metadata
# We strip all of these at post-processing time so only the
# human-readable text content lands in the chart-extracted block.
_NEMOTRON_TAG_RE = re.compile(
    r"<(?:bbox|class|md|patch|extra|x|y)_[^>]*>",
    re.IGNORECASE,
)


def _is_unichart_handle(handle: ChartOCRHandle) -> bool:
    """True when the chart-OCR handle wraps a UniChart-family
    VisionEncoderDecoder model (P3.3-c, Path A). Detection: model is
    `VisionEncoderDecoderModel` AND has a `donut-swin` encoder. This
    catches `khhuang/chart-to-table` and any future UniChart variants
    without depending on model-id substring matching.
    """
    cls = type(handle.model).__name__
    if cls != "VisionEncoderDecoderModel":
        return False
    encoder_cls = type(getattr(handle.model, "encoder", None)).__name__.lower()
    return "donutswin" in encoder_cls or "donut_swin" in encoder_cls


# UniChart chart-to-table prompt. Per the model card
# (https://huggingface.co/khhuang/chart-to-table), this is the exact
# task token the model expects; deviating produces empty output.
_PROMPT_UNICHART = "<data_table_generation> <s_answer>"

# UniChart's table-serialization format:
#   row1col1 | row1col2 | row1col3 &&& row2col1 | row2col2 | row2col3
# We convert this to a markdown table for the agent's downstream
# consumption (markdown is the canonical chart-extracted block format
# in the rest of the pipeline).
_UNICHART_ROW_DELIM = "&&&"
_UNICHART_COL_DELIM = "|"


# OneChart's auxiliary self-consistency token. When the model's
# generated output contains this followed by "True"/"False" (or the
# model returns a tuple containing the boolean separately depending
# on the upstream API), the boolean indicates whether the chart
# extraction is internally consistent. We conservatively treat
# `reliable_check=False` as a model-confessed failure and drop the
# extraction — preserves HARD GATES at the cost of losing some
# borderline-quality extractions.
_ONECHART_RELIABLE_RE = re.compile(
    r"reliable[_\s-]*check\s*[:=]?\s*(?P<value>True|False|true|false)",
    re.IGNORECASE,
)


# OneChart's published prompt (per the paper + the HF repo's example
# code). Adjustments here change extraction quality; align with the
# upstream defaults unless A/B testing shows otherwise.
_PROMPT_ONECHART = "Convert the key information of the chart to a python dict:"


def _chart_ocr_transcribe_sync(
    handle: ChartOCRHandle, image: Image.Image, prompt: str, max_new_tokens: int
) -> str:
    """Synchronous transcription; called via asyncio.to_thread.

    Three inference paths, dispatched by handle type:
    - OneChart-style (P3.3-b, custom Vary-derived): calls the model's
      `.chat()` API with the image + tokenizer; parses the auxiliary
      `reliable_check` token and returns empty string when the model
      self-flags unreliable.
    - VLM-style (Qwen-VL etc.): uses chat-template + processor +
      generate, then slices off the prompt prefix from the output.
    - Pix2Struct-style (DePlot etc.): direct text+image processor
      call, decoded as-is.

    Calls `torch.cuda.empty_cache()` after inference to keep the
    caching allocator from fragmenting across many sequential figures.
    """
    import torch

    if _is_onechart_handle(handle):
        return _chart_ocr_transcribe_onechart(handle, image, max_new_tokens)

    if _is_unichart_handle(handle):
        return _chart_ocr_transcribe_unichart(handle, image, max_new_tokens)

    if _is_nemotron_parse_handle(handle):
        return _chart_ocr_transcribe_nemotron_parse(handle, image, max_new_tokens)

    is_vlm = _is_vlm_handle(handle)

    # The chart-OCR processor + model are model-specific classes whose
    # `apply_chat_template` / `__call__` / `batch_decode` / `decode`
    # kwargs aren't on the base `ProcessorMixin` stub, and transformers'
    # stub types `PreTrainedModel.generate` as a broken `Tensor | Module`
    # union. Both are genuinely-dynamic transformers boundaries → `Any`.
    processor: Any = handle.processor
    model: Any = handle.model

    if is_vlm:
        # VLM path: build a chat message with image + text prompt.
        # Mirrors `vlm_backend.py::_vlm_transcribe_sync` but with the
        # chart-extraction prompt (UNREADABLE escape hatch) instead of
        # the page-transcription prompt.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs: Any = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(model.device)
    else:
        # Pix2Struct path: direct text+image processor call.
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(model.device)

    # Cast float tensors to the model's dtype (BF16 per ADR-0006).
    # Pix2Struct processor produces FP32 pixel_values; BF16-loaded
    # model expects BF16. Integer tensors stay as-is.
    model_dtype = next(model.parameters()).dtype
    inputs = {k: v.to(model_dtype) if v.dtype.is_floating_point else v for k, v in inputs.items()}

    outputs: Any = None
    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        if is_vlm:
            # VLMs return prompt-prefix + completion; slice off the
            # prompt portion before decoding so we get just the model's
            # output, mirroring vlm_backend's pattern.
            generated = outputs[:, inputs["input_ids"].shape[1] :]
            decoded_list = processor.batch_decode(generated, skip_special_tokens=True)
            return cast(str, (decoded_list[0] if decoded_list else "").strip())
        decoded = processor.decode(outputs[0], skip_special_tokens=True)
        return cast(str, decoded.strip())
    finally:
        del inputs
        if outputs is not None:
            del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _chart_ocr_transcribe_onechart(
    handle: ChartOCRHandle, image: Image.Image, max_new_tokens: int
) -> str:
    """OneChart-specific transcription path.

    OneChart's HF repo ships a custom `.chat(tokenizer, image_file,
    ocr_type=...)` method that takes the tokenizer + image path/PIL
    and returns the generated text plus (for chart inputs) an
    auxiliary `reliable_check` boolean indicating self-consistency.
    The exact API signature has minor variation across revisions; we
    try the documented form first and fall through to a more defensive
    call shape on TypeError.

    We treat reliable_check=False as a model-confessed failure and
    return empty string — the stitch step then leaves the
    `<!-- image -->` placeholder unchanged. This is the P3.3-b
    conservative-by-default behavior; preserves HARD GATES on
    counterfactual queries that would otherwise be fed fabricated
    chart data.

    Output format: OneChart returns a python-dict-style string
    (e.g. `{'title': 'X', 'source': 'Y', 'values': {'A': 10, ...}}`).
    Convert to a markdown table for downstream agent consumption.
    """
    import tempfile

    import torch

    log = logger.bind(handle=type(handle.model).__name__)

    # OneChart exposes a custom Vary-derived `.chat()` method that isn't
    # on the base `PreTrainedModel` stub — a genuinely-dynamic
    # transformers boundary, accessed through `Any`.
    model: Any = handle.model

    # OneChart's `.chat(tokenizer, image_file, ...)` requires a
    # FILE PATH (string), not a PIL Image — its internal `load_image`
    # branches on `image_file.startswith('http')`. We render the PIL
    # image to a temp PNG and pass the path. Cleaned up on exit.
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            image.save(tmp.name, format="PNG")
            tmp.flush()
            try:
                with torch.inference_mode():
                    # OneChart's verified signature (HF-repo introspection
                    # 2026-05-23): `.chat(tokenizer, image_file,
                    # reliable_check=True, print_prompt=False)`. The
                    # `reliable_check=True` kwarg runs the model's
                    # built-in self-consistency check; outputs flagged
                    # as unreliable are detected via the
                    # `_ONECHART_RELIABLE_RE` post-pattern.
                    result: Any = model.chat(
                        handle.processor,  # tokenizer in OneChart's API
                        tmp.name,
                        reliable_check=True,
                    )
            except TypeError as e:
                # Upstream signature drift across HF revisions.
                log.warning(
                    "chart_ocr.onechart.signature_mismatch",
                    error=str(e)[:120],
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return ""
    except (OSError, RuntimeError) as e:
        # Temp-file I/O failure or runtime issue inside the model's
        # forward pass. Treat as figure-specific failure rather than
        # crashing the entire chart-OCR pass.
        log.warning("chart_ocr.onechart.io_or_runtime_error", error=str(e)[:120])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ""

    # The chat() result may be a single string, or a tuple of
    # (text, reliable_check) depending on upstream revision.
    if isinstance(result, tuple) and len(cast("tuple[Any, ...]", result)) >= 2:
        result_tuple = cast("tuple[Any, ...]", result)
        text = str(result_tuple[0])
        reliable = bool(result_tuple[1])
    else:
        text = str(cast("Any", result))
        # Parse the reliable_check from the text itself if the API
        # didn't return it as a separate value.
        match = _ONECHART_RELIABLE_RE.search(text)
        if match is None:
            # No reliable_check token at all — assume reliable for
            # backward compat with rev that doesn't emit one. The
            # conservative choice would be to assume unreliable, but
            # that would drop ALL extractions on a rev that doesn't
            # emit the token. Mid-ground: emit, and let downstream
            # ambiguous-header detection catch obvious failures.
            reliable = True
        else:
            reliable = match.group("value").lower() == "true"

    if not reliable:
        log.info("chart_ocr.onechart.reliable_check_false")
        return ""

    # Strip the reliable_check fragment from the text so it doesn't
    # leak into the final markdown.
    text = _ONECHART_RELIABLE_RE.sub("", text).strip()

    # Convert OneChart's python-dict-style output to a markdown table.
    # The output looks like `{'title': 'X', 'values': {'A': 10}}`. We
    # use ast.literal_eval (safe; no eval) and then format. If parsing
    # fails (e.g. malformed output), return the raw text — the agent
    # can still read it and the downstream ambiguous-header check
    # will drop anything obviously broken.
    try:
        import ast

        parsed = ast.literal_eval(text)
        return _onechart_dict_to_markdown(parsed)
    except (ValueError, SyntaxError):
        log.info("chart_ocr.onechart.dict_parse_failed", snippet=text[:120])
        return text


def _chart_ocr_transcribe_unichart(
    handle: ChartOCRHandle, image: Image.Image, max_new_tokens: int
) -> str:
    """UniChart Donut-style transcription path (Path A, P3.3-c).

    Model: `VisionEncoderDecoderModel` with `donut-swin` encoder +
    `mbart` decoder. Verified upstream inference recipe (model card
    on `khhuang/chart-to-table`):

      input_prompt = "<data_table_generation> <s_answer>"
      pixel_values = processor(img, random_padding=False).pixel_values
      decoder_input_ids = tokenizer(prompt, add_special_tokens=False)
      outputs = model.generate(
          pixel_values, decoder_input_ids,
          max_length=model.decoder.config.max_position_embeddings,
          num_beams=4, early_stopping=True,
          pad_token_id=..., eos_token_id=...,
          bad_words_ids=[[unk_token_id]],
      )

    The output is split on `<s_answer>` to extract the table portion.
    Tables use `&&&` for row delimiter, `|` for column delimiter.
    We convert this to a standard markdown table for the rest of the
    pipeline.

    `max_new_tokens` is honored by capping `max_length` to
    `min(model.decoder.config.max_position_embeddings, max_new_tokens + decoder_input_ids.shape[-1])`.
    """
    import torch

    log = logger.bind(handle=type(handle.model).__name__)

    # The DonutProcessor (its `.tokenizer` sub-object) + the
    # VisionEncoderDecoder model (`.decoder.config`, `.generate`) expose
    # attributes/kwargs absent from the base `ProcessorMixin` /
    # `PreTrainedModel` stubs — genuinely-dynamic transformers boundaries
    # routed through `Any`.
    processor: Any = handle.processor
    model: Any = handle.model

    try:
        with torch.inference_mode():
            pixel_values = processor(
                image.convert("RGB"),
                random_padding=False,
                return_tensors="pt",
            ).pixel_values.to(model.device, dtype=model.dtype)

            decoder_input_ids = processor.tokenizer(
                _PROMPT_UNICHART,
                add_special_tokens=False,
                return_tensors="pt",
                max_length=510,
            ).input_ids.to(model.device)

            # Bound generation: never exceed the model's
            # max_position_embeddings AND never exceed the caller's
            # max_new_tokens budget. Whichever is tighter wins.
            decoder_max = model.decoder.config.max_position_embeddings
            prompt_len = decoder_input_ids.shape[-1]
            max_length = min(decoder_max, prompt_len + max_new_tokens)

            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=4,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        sequence: str = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
            processor.tokenizer.pad_token, ""
        )

        if "<s_answer>" in sequence:
            table_raw = sequence.split("<s_answer>")[1].strip()
        else:
            table_raw = sequence.strip()
    except (RuntimeError, ValueError) as e:
        # Defensive: any inference-time failure (CUDA OOM during the
        # beam search, processor input shape mismatch, etc.) becomes
        # an empty extraction rather than crashing the parse pass.
        log.warning("chart_ocr.unichart.inference_failed", error=str(e)[:160])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ""
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _unichart_table_to_markdown(table_raw)


_LATEX_TABULAR_RE = re.compile(
    r"\\begin\{tabular\}\{[^}]*\}(.*?)(?:\\end\{tabular\}|\Z)",
    re.DOTALL,
)
_LATEX_MULTICOLUMN_RE = re.compile(
    r"\\multicolumn\{\d+\}\{[^}]*\}\{([^}]*)\}",
)
# Used by `_flatten_multicolumns` for the nested-brace case
# (`\multicolumn{2}{c}{Apr {x} Jun}` clips at the first `}` with the
# above regex). The opener-only regex finds where each `\multicolumn`
# starts; we then hand-balance braces to find the matching close.
# Established by the post-v7 verification audit (2026-05-23).
_LATEX_MULTICOLUMN_HEAD_RE = re.compile(r"\\multicolumn\{\d+\}\{[^}]*\}\{")


def _flatten_multicolumns(content: str) -> str:
    """Replace every `\\multicolumn{N}{spec}{content}` with its inner
    `content`, brace-balanced. Falls back to the regex-based
    flatten when no `\\multicolumn` is present (fast path).

    Brace-balancing is needed for cases like `\\multicolumn{2}{c}{Apr
    {x} Jun}` where the inner content contains nested braces — the
    regex `\\{([^}]*)\\}` would clip at the first inner `}` and emit
    `Apr {x`.
    """
    if "\\multicolumn" not in content:
        return content
    out: list[str] = []
    i = 0
    while i < len(content):
        m = _LATEX_MULTICOLUMN_HEAD_RE.search(content, i)
        if not m:
            out.append(content[i:])
            break
        out.append(content[i : m.start()])
        # Walk forward from m.end() balancing braces to find the close
        depth = 1
        j = m.end()
        while j < len(content) and depth > 0:
            ch = content[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth == 0:
            out.append(content[m.end() : j])
            i = j + 1
        else:
            # Unbalanced — append the rest as-is and stop.
            out.append(content[m.end() :])
            break
    return "".join(out)


# Matches `**On Time 22**`, `Late 8`, `Status 12.5%` — a label followed
# by a trailing number/percentage. Used to split chart-summary single-row
# tabulars into key-value lines the LLM can parse unambiguously.
#
# Post-audit hardening (2026-05-23 verification audit):
# - The value-half is bounded to 3 integer digits (`\d{1,3}(?:[,.]\d+)?`)
#   so 4-digit years like "March 2026" don't get misread as
#   `(label='March', value='2026')`. Counts in chart-summary tabulars
#   (22, 8, 30, etc.) fit comfortably; if a future chart has 4-digit
#   counts the row falls through to the markdown-table fallback, which
#   is acceptable — the misread was worse than the fallback.
# - Currency-prefixed cells (`$ 193,737`) fall through too: 193 is in
#   range but `,737` extends past the integer-cap pattern.
_LABEL_NUMBER_CELL_RE = re.compile(
    r"^\*{0,2}\s*([A-Za-z][\w\s]*?)\s+(-?\d{1,3}(?:[.,]\d{1,3})?\s*%?)\s*\*{0,2}\s*$"
)


def _split_label_number_cells(cells: list[str]) -> list[tuple[str, str]] | None:
    """If every cell in `cells` matches the `label <number>` pattern,
    return the split as `[(label, number), ...]`. Otherwise return
    None (caller falls back to the raw markdown table).

    Used for chart-summary single-row tabulars where Nemotron-Parse
    concatenates the legend label and its value into one cell:
        `**On Time 22**` → `("On Time", "22")`
        `**Late 8**` → `("Late", "8")`

    Post-audit (2026-05-23) hardening:
    - Date-only cells (`March 2026`) no longer split (the regex requires
      the label-half to contain a non-digit ending — `March 2026` →
      label=`March 2026 ` (with trailing space) which the rstrip
      normalizes, but the value-half then has to be a short number; the
      `{0,7}` cap rejects long fragments).
    - Currency-prefixed cells (`$ 193,737`) fall through to the table
      fallback because the value-half exceeds 8 chars.
    """
    splits: list[tuple[str, str]] = []
    for c in cells:
        if not c.strip():
            continue
        m = _LABEL_NUMBER_CELL_RE.match(c.strip())
        if not m:
            return None
        splits.append((m.group(1).strip(), m.group(2).strip()))
    return splits or None


def _latex_tabular_to_markdown(content: str) -> str:
    """Convert one LaTeX tabular body to a markdown table.

    Nemotron-Parse-v1.2 emits chart tabular data as
    ``\\begin{tabular}{cc} **On Time 22** & **Late 8** \\\\ \\end{tabular}``
    which the Qwen3-8B-AWQ assessor misreads. This converter
    flattens to the markdown equivalent, which the assessor reads
    natively.

    Behavior:
    - Single-row tabulars: emit ``| cell | cell |`` (no separator;
      a one-row "table" is just a pipe-separated line).
    - Multi-row tabulars: emit a full markdown table with the first
      non-empty row as the header.
    - ``\\multicolumn{N}{spec}{content}`` is flattened to just
      ``content`` (we drop the spanning). Brace-balanced via
      ``_flatten_multicolumns`` so nested braces in the content
      (``Apr {x} Jun``) survive — fix shipped post-v7 audit
      (2026-05-23).
    - Markdown ``**bold**`` markers are kept (valid markdown).
    """
    content = _flatten_multicolumns(content)
    rows_raw = re.split(r"\\\\", content)

    parsed: list[list[str]] = []
    for row in rows_raw:
        row = row.strip()
        if not row:
            continue
        cells = [c.strip() for c in row.split("&")]
        if any(cell for cell in cells):
            parsed.append(cells)

    if not parsed:
        return ""

    max_cols = max(len(r) for r in parsed)
    parsed = [r + [""] * (max_cols - len(r)) for r in parsed]

    if len(parsed) == 1:
        non_empty = [c for c in parsed[0] if c]
        if not non_empty:
            return ""
        # If every cell looks like `<label> <number>` (a chart-summary
        # row from Nemotron-Parse), emit as key-value bullets instead
        # of a pipe-row. Reason: a cell like `**On Time 22**` is
        # ambiguous to the LLM — could be a category label or a
        # label-value pair. Bullets disambiguate.
        split = _split_label_number_cells(non_empty)
        if split:
            return "\n".join(f"- {label}: {value}" for label, value in split)
        return "| " + " | ".join(non_empty) + " |"

    header = "| " + " | ".join(parsed[0]) + " |"
    separator = "| " + " | ".join(["---"] * max_cols) + " |"
    body = ["| " + " | ".join(r) + " |" for r in parsed[1:]]
    return "\n".join([header, separator, *body])


def _normalize_latex_tabulars(text: str) -> str:
    """Replace every ``\\begin{tabular}...\\end{tabular}`` block with a
    markdown table. Tolerates truncated tabulars (no closing tag) by
    converting whatever rows were emitted before the cutoff.
    """
    return _LATEX_TABULAR_RE.sub(
        lambda m: _latex_tabular_to_markdown(m.group(1)),
        text,
    )


def _unichart_table_to_markdown(raw: str) -> str:
    """Convert UniChart's `row1col1 | row1col2 &&& row2col1 | row2col2`
    format to a standard markdown table.

    The first row is treated as the header. If only one row is
    produced (degenerate output), still emit a single-row markdown
    table so the downstream agent can read it.
    """
    raw = raw.strip()
    if not raw:
        return ""

    rows = [r.strip() for r in raw.split(_UNICHART_ROW_DELIM) if r.strip()]
    if not rows:
        return ""

    parsed: list[list[str]] = []
    for row in rows:
        cells = [c.strip() for c in row.split(_UNICHART_COL_DELIM)]
        if cells:
            parsed.append(cells)

    if not parsed:
        return ""

    # Normalise column count to the longest row (pad with empty cells).
    max_cols = max(len(row) for row in parsed)
    parsed = [row + [""] * (max_cols - len(row)) for row in parsed]

    header = "| " + " | ".join(parsed[0]) + " |"
    separator = "| " + " | ".join(["---"] * max_cols) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in parsed[1:]]

    return "\n".join([header, separator, *body_rows])


def _chart_ocr_transcribe_nemotron_parse(
    handle: ChartOCRHandle, image: Image.Image, max_new_tokens: int
) -> str:
    """Nemotron-Parse-v1.2 inference path (Path C, P3.3-c).

    NVIDIA's 885M document parser. Per the model card recipe:
      inputs = processor(images=[image], text=task_prompt,
                         return_tensors="pt", add_special_tokens=False)
      generation_config = GenerationConfig.from_pretrained(model_path)
      outputs = model.generate(**inputs, generation_config=...)
      generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    The raw output contains `<bbox_*>` + `<class_*>` metadata tags
    interspersed with markdown content. We strip the tags via
    `_NEMOTRON_TAG_RE` so only the human-readable markdown lands
    in the chart-extracted block. The `<predict_no_text_in_pic>`
    task token in the prompt instructs the model to skip OCR of
    text within picture elements, focusing the output on chart-
    block content.

    Defensive try/except: RuntimeError (CUDA-side issue) or
    ValueError (processor mismatch) produce empty markdown rather
    than crashing the parse pass.
    """
    import torch

    log = logger.bind(handle=type(handle.model).__name__)

    # Nemotron-Parse's processor `__call__`/`batch_decode` kwargs and the
    # model's `.generate` / `.config` aren't on the base stubs — dynamic
    # transformers boundaries routed through `Any`.
    processor: Any = handle.processor
    model: Any = handle.model

    try:
        with torch.inference_mode():
            inputs = processor(
                images=[image.convert("RGB")],
                text=_PROMPT_NEMOTRON_PARSE,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(model.device)

            # Try to load the model's published GenerationConfig
            # (includes carefully-tuned stop tokens + repetition
            # penalty); fall back to inline kwargs if the import
            # fails (e.g. older transformers version doesn't have
            # the symbol).
            try:
                from transformers import GenerationConfig

                # `GenerationConfig.from_pretrained` returns Unknown in
                # transformers' stub; route the classmethod call through
                # an `Any`-typed alias so the member access is explicit.
                gen_config_cls: Any = GenerationConfig
                generation_config: Any = gen_config_cls.from_pretrained(
                    model.config._name_or_path,
                    trust_remote_code=True,
                )
                # Bound emission with the caller's max_new_tokens
                generation_config.max_new_tokens = max_new_tokens
                outputs = model.generate(**inputs, generation_config=generation_config)
            except (ImportError, OSError):
                # Fallback: explicit kwargs
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.1,
                )

        generated_text: str = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    except (RuntimeError, ValueError) as e:
        log.warning(
            "chart_ocr.nemotron_parse.inference_failed",
            error=str(e)[:160],
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ""
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Strip Nemotron-Parse structural tags (`<bbox_*>`, `<class_*>`,
    # etc.) so only markdown text content lands in the chart block.
    cleaned = _NEMOTRON_TAG_RE.sub(" ", generated_text)
    # Collapse multi-space runs introduced by tag removal.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    # Convert any `\begin{tabular}...\end{tabular}` blocks to markdown
    # tables. Raw LaTeX tabular is illegible to the downstream LLM
    # assessor: the v3 chart-OCR A/B confirmed the right chunk reaches
    # rank 1 in the reranker but assess_sufficiency misreads the LaTeX
    # cells as "no specific numbers." Markdown tables are LLM-native.
    cleaned = _normalize_latex_tabulars(cleaned)
    # Strip leading/trailing whitespace per line; drop empty lines.
    lines = [line.strip() for line in cleaned.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _onechart_dict_to_markdown(d: Any) -> str:
    """Convert OneChart's `{title, source, values: {label: value}}`
    dict to a markdown table. Handles the common shapes; falls back
    to a key-value table for unexpected structures.

    `d` is the result of `ast.literal_eval` on the model's dict-style
    output — a genuinely-dynamic structure, hence the explicit `Any`.
    """
    if not isinstance(d, dict):
        return str(d)
    d = cast("dict[Any, Any]", d)

    title = d.get("title", "")
    values = d.get("values", {})

    if not isinstance(values, dict) or not values:
        # No values dict — just dump key-value pairs as a 2-col table.
        rows = "\n".join(f"| {k} | {v} |" for k, v in d.items())
        return f"| key | value |\n| --- | --- |\n{rows}"

    header = f"# {title}\n\n" if title else ""
    values_dict = cast("dict[Any, Any]", values)
    table_rows = "\n".join(f"| {k} | {v} |" for k, v in values_dict.items())
    return f"{header}| label | value |\n| --- | --- |\n{table_rows}"


async def _extract_with_handle(
    handle: ChartOCRHandle,
    source_pdf: Path,
    figure: FigureMetadata,
    max_new_tokens: int,
) -> ChartOCROutput:
    """Internal: render + transcribe one figure given an already-
    acquired chart-OCR handle. Selects the prompt based on handle type
    (VLM-style gets the UNREADABLE-escape-hatch prompt; Pix2Struct-
    style gets DePlot's published prompt)."""
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
    # Prompt selection: OneChart + UniChart each use their own
    # published prompt (handled inside their respective transcribe
    # helpers); the prompt arg here is the fallback for the VLM /
    # Pix2Struct paths.
    if _is_onechart_handle(handle):
        prompt = _PROMPT_ONECHART  # passed through but not used directly
    elif _is_unichart_handle(handle):
        prompt = _PROMPT_UNICHART  # passed through but not used directly
    elif _is_nemotron_parse_handle(handle):
        prompt = _PROMPT_NEMOTRON_PARSE  # passed through
    else:
        prompt = _PROMPT_VLM if _is_vlm_handle(handle) else _PROMPT_DEPLOT
    markdown = await asyncio.to_thread(
        _chart_ocr_transcribe_sync, handle, image, prompt, max_new_tokens
    )

    # UNREADABLE-escape-hatch: when the VLM emits the literal token,
    # treat the extraction as a refusal and return an empty
    # ChartOCROutput. The stitch step skips empty results so the
    # `<!-- image -->` placeholder stays unchanged. This is the
    # primary hallucination-prevention signal for the VLM path.
    if markdown and _UNREADABLE_TOKEN in markdown.upper() and len(markdown.split()) < 5:
        log.info("chart_ocr.unreadable_refusal")
        markdown = ""

    # DePlot-output post-processing (P3.3 v2 Session 2 heuristic-cleanup
    # fallback). Two cheap interventions:
    # 1. Replace `<0x0A>` byte-sequence (Pix2Struct's newline encoding)
    #    with actual `\n` so the agent's literal-presence rule can
    #    parse the table rows as separate lines.
    # 2. Reject extractions whose headers contain "Series1/Series2/etc"
    #    — these are collapsed-multi-series outputs the agent cannot
    #    disambiguate (e.g., is 0.8 power or density?). Better to
    #    refuse than to ship ambiguous data.
    if markdown:
        markdown = _PIX2STRUCT_NEWLINE_RE.sub("\n", markdown)
        if _AMBIGUOUS_HEADER_RE.search(markdown):
            log.info(
                "chart_ocr.ambiguous_series_refusal",
                snippet=markdown[:100],
            )
            markdown = ""

    log.info("chart_ocr.done", chars=len(markdown))
    return ChartOCROutput(
        page_no=figure.page_no,
        bbox=figure.bbox,
        markdown=markdown,
    )


async def _extract_best_of_n(
    handle: ChartOCRHandle,
    source_pdf: Path,
    figure: FigureMetadata,
    max_new_tokens: int,
    samples: int,
) -> ChartOCROutput:
    """Take `samples` independent extraction draws of one figure and keep the
    one with the LONGEST markdown — a content-completeness proxy mirroring the
    VLM's best-of-N (`vlm_backend._convert_with_handle`). Chart-OCR is
    non-deterministic and a given draw can drop a row or trip the
    ambiguous-header / UNREADABLE refusal (→ empty markdown); the longest
    non-empty draw is the most complete. The chosen draw is then cached, so
    completeness is decided once. `samples <= 1` is exactly today's single-draw
    behaviour (no extra cost)."""
    best = await _extract_with_handle(handle, source_pdf, figure, max_new_tokens)
    n = max(1, samples)
    if n == 1:
        return best
    draft_chars = [len(best.markdown)]
    for _ in range(n - 1):
        candidate = await _extract_with_handle(handle, source_pdf, figure, max_new_tokens)
        draft_chars.append(len(candidate.markdown))
        if len(candidate.markdown) > len(best.markdown):
            best = candidate
    logger.bind(page=figure.page_no, bbox=figure.bbox).info(
        "chart_ocr.best_of_n", samples=n, draft_chars=draft_chars, chosen_chars=len(best.markdown)
    )
    return best


# Bump when the chart-OCR prompt/rendering changes in a way that should
# invalidate cached extractions (the model id is already in the key).
_CHART_OCR_CACHE_VERSION = "v1"


def _chart_cache_key(pdf_sha256: str, figure: FigureMetadata, model_id: str) -> tuple[str, str]:
    """Return (cache_key, bbox_key) for a figure's chart-OCR extraction. bbox is
    rounded to whole PDF points so trivial float jitter still hits; the model id
    + version bust the cache on a model/prompt change."""
    x0, y0, x1, y1 = figure.bbox
    bbox_key = f"{round(x0)}_{round(y0)}_{round(x1)}_{round(y1)}"
    return (
        f"{pdf_sha256}:{figure.page_no}:{bbox_key}:m={model_id}:v={_CHART_OCR_CACHE_VERSION}",
        bbox_key,
    )


async def chart_ocr_extract(
    *,
    source_pdf: Path,
    figures: list[FigureMetadata],
    max_new_tokens: int = _MAX_NEW_TOKENS,
    min_area_sqpt: float = _MIN_FIGURE_AREA_SQPT,
    chart_class_names: frozenset[str] = _CHART_CLASS_NAMES,
    min_classification_confidence: float = _MIN_CLASSIFICATION_CONFIDENCE,
    cache: ChartOCRCache | None = None,
    refresh: bool = False,
    extraction_samples: int = 1,
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

    # Filter tiny figures BEFORE acquiring the model. Page-number
    # badges, watermarks, and decorative elements are sub-threshold
    # and waste model inference time without producing useful chart
    # data. On the canonical CUDA deck this prunes ~80% of the 245
    # picture objects down to the ~40-50 real charts.
    #
    # IMPORTANT: we still return one result PER INPUT figure so the
    # caller's stitch step (`_stitch_chart_extractions`) can zip
    # results positionally against the markdown's `<!-- image -->`
    # placeholders. Filtered figures get a synthetic empty
    # ChartOCROutput so the count matches.
    results: list[ChartOCROutput | Exception] = []
    figures_to_extract: list[tuple[int, FigureMetadata]] = []
    for idx, figure in enumerate(figures):
        # ALWAYS append a placeholder so positional alignment with
        # the caller's `<!-- image -->` placeholders is preserved.
        # Filtered figures get an empty ChartOCROutput; the stitch
        # step skips empty results so the placeholder stays unchanged.
        results.append(ChartOCROutput(page_no=figure.page_no, bbox=figure.bbox, markdown=""))

        # Filter 1: area. Skip page-number badges, watermarks,
        # decorative elements below ~140×140 pt.
        x0, y0, x1, y1 = figure.bbox
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area < min_area_sqpt:
            continue

        # Filter 2: classification. When Docling's classifier emits a
        # confident prediction, accept ONLY the chart-like classes and
        # drop everything else (logos, flow charts, photographs,
        # engineering drawings, tables-as-images, icons). When the
        # confidence is below the floor, fall back to "extract" to
        # avoid dropping ambiguous-but-real charts. When the
        # classification is None (v1 worker / disabled feature), also
        # fall back to extract.
        if (
            figure.classification is not None
            and figure.classification_confidence >= min_classification_confidence
            and figure.classification not in chart_class_names
        ):
            continue

        figures_to_extract.append((idx, figure))

    if not figures_to_extract:
        return results

    # Cache check: serve hits, collect misses. Chart-OCR is non-deterministic,
    # so a re-parse otherwise drifts chart content; the cache replays the stored
    # extraction byte-for-byte (keyed by pdf bytes, page, bbox, model, version).
    # An all-hit batch skips the GPU acquisition entirely.
    pdf_sha256 = ""
    model_id = ""
    # (slot_idx, figure, cache_key, bbox_key); keys are "" when caching is off.
    to_run: list[tuple[int, FigureMetadata, str, str]] = []
    if cache is not None:
        pdf_sha256 = hashlib.sha256(source_pdf.read_bytes()).hexdigest()
        model_id = get_settings().models.chart_ocr
        if refresh:
            await cache.delete_by_pdf(pdf_sha256)
        hits = 0
        for slot_idx, figure in figures_to_extract:
            key, bbox_key = _chart_cache_key(pdf_sha256, figure, model_id)
            cached = await cache.get(key)
            if cached is not None:
                results[slot_idx] = ChartOCROutput(
                    page_no=figure.page_no, bbox=figure.bbox, markdown=cached
                )
                hits += 1
            else:
                to_run.append((slot_idx, figure, key, bbox_key))
        logger.info("chart_ocr.cache", hits=hits, misses=len(to_run))
        if not to_run:
            return results  # every figure cached — no GPU work
    else:
        to_run = [(slot_idx, figure, "", "") for slot_idx, figure in figures_to_extract]

    log = logger.bind(total_figures=len(figures), figures_to_extract=len(to_run))
    log.info("chart_ocr.batch.start")

    registry = get_registry()
    async with registry.use("chart_ocr") as handle:
        for slot_idx, figure, key, bbox_key in to_run:
            try:
                out = await _extract_best_of_n(
                    handle, source_pdf, figure, max_new_tokens, samples=extraction_samples
                )
                results[slot_idx] = out
                if cache is not None:
                    await cache.put(
                        key,
                        pdf_sha256=pdf_sha256,
                        page_no=figure.page_no,
                        bbox_key=bbox_key,
                        chart_ocr_model=model_id,
                        markdown=out.markdown,
                    )
            except (ChartOCRUnavailable, PDFFigureRenderError) as e:
                results[slot_idx] = e

    log.info("chart_ocr.batch.done")
    return results
