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
import re
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
_UNREADABLE_TOKEN = "UNREADABLE"

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


def _is_vlm_handle(handle) -> bool:
    """True when the chart-OCR handle wraps a VLM-style model (Qwen-VL,
    LLaVA, etc.) rather than a Pix2Struct-style chart specialist. The
    inference path differs: VLMs use chat templates + messages;
    Pix2Struct uses a direct text+image processor call.
    """
    cls = type(handle.model).__name__
    return any(token in cls for token in ("VL", "Vision", "Llava", "Internvl"))


def _is_onechart_handle(handle) -> bool:
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


def _is_unichart_handle(handle) -> bool:
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
    handle, image, prompt: str, max_new_tokens: int
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

    is_vlm = _is_vlm_handle(handle)

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
        text = handle.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = handle.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(handle.model.device)
    else:
        # Pix2Struct path: direct text+image processor call.
        inputs = handle.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(handle.model.device)

    # Cast float tensors to the model's dtype (BF16 per ADR-0006).
    # Pix2Struct processor produces FP32 pixel_values; BF16-loaded
    # model expects BF16. Integer tensors stay as-is.
    model_dtype = next(handle.model.parameters()).dtype
    inputs = {
        k: v.to(model_dtype) if v.dtype.is_floating_point else v
        for k, v in inputs.items()
    }

    try:
        with torch.inference_mode():
            outputs = handle.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        if is_vlm:
            # VLMs return prompt-prefix + completion; slice off the
            # prompt portion before decoding so we get just the model's
            # output, mirroring vlm_backend's pattern.
            generated = outputs[:, inputs["input_ids"].shape[1] :]
            decoded_list = handle.processor.batch_decode(
                generated, skip_special_tokens=True
            )
            return (decoded_list[0] if decoded_list else "").strip()
        decoded = handle.processor.decode(outputs[0], skip_special_tokens=True)
        return decoded.strip()
    finally:
        del inputs
        if "outputs" in dir():
            del outputs  # noqa: F821
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _chart_ocr_transcribe_onechart(handle, image, max_new_tokens: int) -> str:
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

    # OneChart's `.chat(tokenizer, image_file, ...)` requires a
    # FILE PATH (string), not a PIL Image — its internal `load_image`
    # branches on `image_file.startswith('http')`. We render the PIL
    # image to a temp PNG and pass the path. Cleaned up on exit.
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=True
        ) as tmp:
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
                    result = handle.model.chat(
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
    if isinstance(result, tuple) and len(result) >= 2:
        text = str(result[0])
        reliable = bool(result[1])
    else:
        text = str(result)
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


def _chart_ocr_transcribe_unichart(handle, image, max_new_tokens: int) -> str:
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

    try:
        with torch.inference_mode():
            pixel_values = handle.processor(
                image.convert("RGB"),
                random_padding=False,
                return_tensors="pt",
            ).pixel_values.to(handle.model.device, dtype=handle.model.dtype)

            decoder_input_ids = handle.processor.tokenizer(
                _PROMPT_UNICHART,
                add_special_tokens=False,
                return_tensors="pt",
                max_length=510,
            ).input_ids.to(handle.model.device)

            # Bound generation: never exceed the model's
            # max_position_embeddings AND never exceed the caller's
            # max_new_tokens budget. Whichever is tighter wins.
            decoder_max = handle.model.decoder.config.max_position_embeddings
            prompt_len = decoder_input_ids.shape[-1]
            max_length = min(decoder_max, prompt_len + max_new_tokens)

            outputs = handle.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                early_stopping=True,
                pad_token_id=handle.processor.tokenizer.pad_token_id,
                eos_token_id=handle.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=4,
                bad_words_ids=[[handle.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        sequence = handle.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(
            handle.processor.tokenizer.eos_token, ""
        ).replace(handle.processor.tokenizer.pad_token, "")

        if "<s_answer>" in sequence:
            table_raw = sequence.split("<s_answer>")[1].strip()
        else:
            table_raw = sequence.strip()
    except (RuntimeError, ValueError) as e:
        # Defensive: any inference-time failure (CUDA OOM during the
        # beam search, processor input shape mismatch, etc.) becomes
        # an empty extraction rather than crashing the parse pass.
        log.warning(
            "chart_ocr.unichart.inference_failed", error=str(e)[:160]
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ""
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _unichart_table_to_markdown(table_raw)


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


def _onechart_dict_to_markdown(d: object) -> str:
    """Convert OneChart's `{title, source, values: {label: value}}`
    dict to a markdown table. Handles the common shapes; falls back
    to a key-value table for unexpected structures.
    """
    if not isinstance(d, dict):
        return str(d)

    title = d.get("title", "")
    values = d.get("values", {})

    if not isinstance(values, dict) or not values:
        # No values dict — just dump key-value pairs as a 2-col table.
        rows = "\n".join(f"| {k} | {v} |" for k, v in d.items())
        return f"| key | value |\n| --- | --- |\n{rows}"

    header = f"# {title}\n\n" if title else ""
    table_rows = "\n".join(f"| {k} | {v} |" for k, v in values.items())
    return f"{header}| label | value |\n| --- | --- |\n{table_rows}"


async def _extract_with_handle(
    handle: object,
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
    if (
        markdown
        and _UNREADABLE_TOKEN in markdown.upper()
        and len(markdown.split()) < 5
    ):
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


async def chart_ocr_extract(
    *,
    source_pdf: Path,
    figures: list[FigureMetadata],
    max_new_tokens: int = _MAX_NEW_TOKENS,
    min_area_sqpt: float = _MIN_FIGURE_AREA_SQPT,
    chart_class_names: frozenset[str] = _CHART_CLASS_NAMES,
    min_classification_confidence: float = _MIN_CLASSIFICATION_CONFIDENCE,
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
        results.append(
            ChartOCROutput(
                page_no=figure.page_no, bbox=figure.bbox, markdown=""
            )
        )

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
            and figure.classification_confidence
            >= min_classification_confidence
            and figure.classification not in chart_class_names
        ):
            continue

        figures_to_extract.append((idx, figure))

    if not figures_to_extract:
        return results

    log = logger.bind(
        total_figures=len(figures),
        figures_to_extract=len(figures_to_extract),
    )
    log.info("chart_ocr.batch.start")

    registry = get_registry()
    async with registry.use("chart_ocr") as handle:
        for slot_idx, figure in figures_to_extract:
            try:
                out = await _extract_with_handle(
                    handle, source_pdf, figure, max_new_tokens
                )
                results[slot_idx] = out
            except (ChartOCRUnavailable, PDFFigureRenderError) as e:
                results[slot_idx] = e

    log.info("chart_ocr.batch.done")
    return results
