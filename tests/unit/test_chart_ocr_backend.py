"""P3.3 Session 3 unit tests for the chart-OCR backend.

Covers the per-figure batch flow without actually loading DePlot or
opening a real PDF. Stubs:
- `get_registry` → returns a fake that yields a stand-in handle
- `_render_figure_to_image` → returns a sentinel object instead of
  running pypdfium2
- `_chart_ocr_transcribe_sync` → returns canned markdown instead of
  running the model

Mirrors the `_FakeRegistry` pattern in `tests/unit/test_rerank.py`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from memex.parse import chart_ocr_backend
from memex.parse.chart_ocr_backend import (
    ChartOCROutput,
    PDFFigureRenderError,
    chart_ocr_extract,
)
from memex.parse.docling_backend import FigureMetadata


class _FakeModel:
    """Stand-in model class so `_is_vlm_handle(handle)` can call
    `type(handle.model).__name__` without crashing. Name doesn't match
    any VLM patterns ('VL', 'Vision', 'Llava', 'Internvl') so the
    Pix2Struct prompt branch is selected — which is fine because the
    test monkeypatches the transcribe call anyway."""


class _FakeChartOCRHandle:
    """Stand-in for ChartOCRHandle; never touched directly because the
    backend's `_chart_ocr_transcribe_sync` is monkeypatched."""

    def __init__(self) -> None:
        self.model = _FakeModel()
        self.processor = None


@asynccontextmanager
async def _yields(handle: object):
    yield handle


class _FakeRegistry:
    def use(self, name: str) -> Any:  # noqa: ARG002
        return _yields(_FakeChartOCRHandle())


@pytest.mark.asyncio
async def test_empty_figures_list_returns_empty_no_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the input list is empty, the backend MUST NOT acquire the
    registry handle — that's the cheap-path contract documented in the
    backend's docstring. We verify by asserting `get_registry` is never
    called.
    """
    calls = {"get_registry": 0}

    def _fake_get_registry() -> Any:
        calls["get_registry"] += 1
        return _FakeRegistry()

    monkeypatch.setattr(
        "memex.parse.chart_ocr_backend.get_registry", _fake_get_registry
    )

    out = await chart_ocr_extract(
        source_pdf=Path("/nonexistent.pdf"),
        figures=[],
    )

    assert out == []
    assert calls["get_registry"] == 0


@pytest.mark.asyncio
async def test_multiple_figures_extract_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3 figures → 3 ChartOCROutput results in the same order. The
    rendered image is never inspected; we just need the transcription
    stub to fire once per figure and the output to thread through.
    """
    figures = [
        FigureMetadata(
            page_no=1, bbox=(10.0, 20.0, 200.0, 180.0), caption="Fig 1"
        ),
        FigureMetadata(
            page_no=2, bbox=(50.0, 60.0, 250.0, 230.0), caption=None
        ),
        FigureMetadata(
            page_no=5, bbox=(0.0, 0.0, 500.0, 400.0), caption="Chart"
        ),
    ]

    monkeypatch.setattr(
        "memex.parse.chart_ocr_backend.get_registry", lambda: _FakeRegistry()
    )

    # Stub the heavy steps. The sentinel passed back is just to confirm
    # the call chain works; the transcribe stub returns per-page text.
    monkeypatch.setattr(
        chart_ocr_backend,
        "_render_figure_to_image",
        lambda pdf, page, bbox, scale=2.5: f"<image page={page}>",
    )

    def _fake_transcribe(handle, image, prompt, max_new_tokens):  # noqa: ARG001
        return f"| col1 | col2 |\n|---|---|\n| a | b |\n# from {image}"

    monkeypatch.setattr(
        chart_ocr_backend, "_chart_ocr_transcribe_sync", _fake_transcribe
    )

    out = await chart_ocr_extract(
        source_pdf=Path("/fake.pdf"),
        figures=figures,
        min_area_sqpt=0.0,  # bypass area filter so test bboxes go through
    )

    assert len(out) == 3
    for i, result in enumerate(out):
        assert isinstance(result, ChartOCROutput)
        assert result.page_no == figures[i].page_no
        assert result.bbox == figures[i].bbox
        assert f"from <image page={figures[i].page_no}>" in result.markdown


@pytest.mark.asyncio
async def test_classification_filter_skips_non_chart_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P3.3 v2: figures whose Docling-classifier `class_name` is NOT in
    the chart-class set (bar_chart/line_chart/etc.) MUST be skipped
    when classification confidence is above the floor. This is the
    pre-filter that prevents DePlot's OOD-hallucination cascade on
    non-chart content (logos, flow charts, photographs).
    """
    figures = [
        # Real chart — should be extracted.
        FigureMetadata(
            page_no=1,
            bbox=(0.0, 0.0, 500.0, 400.0),
            classification="bar_chart",
            classification_confidence=0.97,
        ),
        # Logo — should be skipped (top class is non-chart, high conf).
        FigureMetadata(
            page_no=2,
            bbox=(0.0, 0.0, 500.0, 400.0),
            classification="logo",
            classification_confidence=0.99,
        ),
        # Flow chart — architecture diagram, NOT extractable data.
        FigureMetadata(
            page_no=3,
            bbox=(0.0, 0.0, 500.0, 400.0),
            classification="flow_chart",
            classification_confidence=0.88,
        ),
        # Line chart — should be extracted.
        FigureMetadata(
            page_no=4,
            bbox=(0.0, 0.0, 500.0, 400.0),
            classification="line_chart",
            classification_confidence=0.92,
        ),
    ]

    transcribe_calls: list[int] = []

    monkeypatch.setattr(
        "memex.parse.chart_ocr_backend.get_registry", lambda: _FakeRegistry()
    )
    monkeypatch.setattr(
        chart_ocr_backend,
        "_render_figure_to_image",
        lambda pdf, page, bbox, scale=2.5: f"<image p={page}>",
    )

    def _fake_transcribe(handle, image, prompt, max_new_tokens):  # noqa: ARG001
        transcribe_calls.append(1)
        return f"| col | val |\n|---|---|\n| a | 1 |\n# {image}"

    monkeypatch.setattr(
        chart_ocr_backend, "_chart_ocr_transcribe_sync", _fake_transcribe
    )

    out = await chart_ocr_extract(
        source_pdf=Path("/fake.pdf"),
        figures=figures,
    )

    assert len(out) == 4
    # Exactly 2 figures (bar_chart at idx 0, line_chart at idx 3) ran
    # through the model. The logo and flow_chart were skipped.
    assert len(transcribe_calls) == 2
    # Per-figure outcomes:
    assert isinstance(out[0], ChartOCROutput) and out[0].markdown != ""
    assert isinstance(out[1], ChartOCROutput) and out[1].markdown == ""
    assert isinstance(out[2], ChartOCROutput) and out[2].markdown == ""
    assert isinstance(out[3], ChartOCROutput) and out[3].markdown != ""


@pytest.mark.asyncio
async def test_low_confidence_classification_falls_back_to_extract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the classifier's confidence is below the floor, we don't
    trust the prediction and run extraction anyway — preventing
    silent drops of real charts misclassified at low confidence (e.g.,
    a bar_chart labeled `icon` at 0.3 confidence).
    """
    figures = [
        # Classification says non-chart but confidence is low → extract.
        FigureMetadata(
            page_no=1,
            bbox=(0.0, 0.0, 500.0, 400.0),
            classification="icon",
            classification_confidence=0.30,
        ),
    ]

    transcribe_calls: list[int] = []

    monkeypatch.setattr(
        "memex.parse.chart_ocr_backend.get_registry", lambda: _FakeRegistry()
    )
    monkeypatch.setattr(
        chart_ocr_backend,
        "_render_figure_to_image",
        lambda pdf, page, bbox, scale=2.5: f"<image p={page}>",
    )

    def _fake_transcribe(handle, image, prompt, max_new_tokens):  # noqa: ARG001
        transcribe_calls.append(1)
        return "| col | val |\n|---|---|\n| a | 1 |"

    monkeypatch.setattr(
        chart_ocr_backend, "_chart_ocr_transcribe_sync", _fake_transcribe
    )

    out = await chart_ocr_extract(
        source_pdf=Path("/fake.pdf"),
        figures=figures,
    )

    assert len(out) == 1
    assert len(transcribe_calls) == 1  # Extracted despite icon label
    assert isinstance(out[0], ChartOCROutput) and out[0].markdown != ""


@pytest.mark.asyncio
async def test_no_classification_falls_back_to_extract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a worker emits FigureMetadata without classification (v1
    payload format), we extract — the absence of a classification
    must not silently drop the figure.
    """
    figures = [
        FigureMetadata(
            page_no=1,
            bbox=(0.0, 0.0, 500.0, 400.0),
            # classification field omitted — default None
        ),
    ]

    transcribe_calls: list[int] = []

    monkeypatch.setattr(
        "memex.parse.chart_ocr_backend.get_registry", lambda: _FakeRegistry()
    )
    monkeypatch.setattr(
        chart_ocr_backend,
        "_render_figure_to_image",
        lambda pdf, page, bbox, scale=2.5: "<image>",
    )

    def _fake_transcribe(handle, image, prompt, max_new_tokens):  # noqa: ARG001
        transcribe_calls.append(1)
        return "| col | val |\n|---|---|\n| a | 1 |"

    monkeypatch.setattr(
        chart_ocr_backend, "_chart_ocr_transcribe_sync", _fake_transcribe
    )

    out = await chart_ocr_extract(
        source_pdf=Path("/fake.pdf"),
        figures=figures,
    )

    assert len(transcribe_calls) == 1  # Extracted (no classification → trust)


@pytest.mark.asyncio
async def test_render_error_returned_per_figure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If `_render_figure_to_image` raises `PDFFigureRenderError` for
    a specific figure, that figure's slot in the result list holds the
    exception while other figures still process normally. The backend
    must not propagate the error out of the batch.
    """
    figures = [
        FigureMetadata(page_no=1, bbox=(10.0, 20.0, 200.0, 180.0)),
        FigureMetadata(page_no=2, bbox=(50.0, 60.0, 5.0, 5.0)),  # bad bbox
        FigureMetadata(page_no=3, bbox=(50.0, 60.0, 250.0, 230.0)),
    ]

    monkeypatch.setattr(
        "memex.parse.chart_ocr_backend.get_registry", lambda: _FakeRegistry()
    )

    def _selective_render(pdf, page, bbox, scale=2.5):  # noqa: ARG001
        if page == 2:
            raise PDFFigureRenderError(
                f"degenerate bbox for figure on page {page}",
                context={"page": page, "bbox": bbox},
            )
        return f"<image page={page}>"

    monkeypatch.setattr(
        chart_ocr_backend, "_render_figure_to_image", _selective_render
    )
    monkeypatch.setattr(
        chart_ocr_backend,
        "_chart_ocr_transcribe_sync",
        lambda h, img, p, m: "ok",
    )

    out = await chart_ocr_extract(
        source_pdf=Path("/fake.pdf"),
        figures=figures,
        min_area_sqpt=0.0,  # bypass area filter so test bboxes go through
    )

    assert len(out) == 3
    assert isinstance(out[0], ChartOCROutput)
    assert isinstance(out[1], PDFFigureRenderError)
    assert isinstance(out[2], ChartOCROutput)


# ----------------------------------------------------------------------
# P3.3-b — OneChart backend tests
# ----------------------------------------------------------------------


class _FakeImage:
    """Minimal stand-in for PIL.Image that supports `.save(path, format=...)`
    and stringifies meaningfully for fragment matching. The OneChart
    transcribe path saves the image to a temp PNG before invoking
    `.chat()`; the fake `.save()` writes a tiny placeholder file so the
    real OneChart-fake's `.chat()` would have something to load if the
    test ever pushed through to that point."""

    def __init__(self, page: int) -> None:
        self.page = page

    def save(self, path: object, format: str = "PNG") -> None:  # noqa: A002, ARG002
        # Write a 1-byte placeholder so the file exists on disk; the
        # OneChart-fake's `.chat()` never actually reads it.
        with open(str(path), "wb") as f:
            f.write(b"\x00")

    def convert(self, mode: str) -> "_FakeImage":  # noqa: ARG002
        # The UniChart transcribe path calls `image.convert("RGB")`
        # before passing to the processor. Return self to keep the
        # chain working.
        return self

    def __repr__(self) -> str:
        return f"<FakeImage page={self.page}>"


class _FakeOneChartModel:
    """Stand-in OneChart model. Class name contains 'OneChart' so
    `_is_onechart_handle` recognises it and dispatches to the
    `.chat()`-based transcription path instead of `.generate()`.

    Test instances configure `chat_return` to either a string or a
    `(text, reliable_check_bool)` tuple, matching the variation across
    OneChart HF-repo revisions.
    """

    def __init__(self, chat_return: object) -> None:
        self._chat_return = chat_return
        self.calls: list[dict[str, object]] = []

    def chat(
        self,
        tokenizer,
        image_file: str,
        reliable_check: bool = True,
        print_prompt: bool = False,
    ) -> object:
        """Mirror OneChart's verified `.chat()` signature (HF-repo
        introspection 2026-05-23): `(tokenizer, image_file,
        reliable_check=True, print_prompt=False)`. `image_file` is a
        path string — the backend writes a temp PNG and passes that.
        """
        self.calls.append(
            {
                "tokenizer": tokenizer,
                "image_file": image_file,
                "reliable_check": reliable_check,
                "print_prompt": print_prompt,
            }
        )
        if isinstance(self._chat_return, Exception):
            raise self._chat_return
        return self._chat_return


class _FakeOneChartHandle:
    """OneChart-shaped handle: `processor` slot holds the tokenizer
    (per the registry's OneChart branch); `model` is the fake with
    a `.chat()` method instead of `.generate()`."""

    def __init__(self, chat_return: object) -> None:
        self.model = _FakeOneChartModel(chat_return)
        self.processor = "fake-tokenizer"  # OneChart uses tokenizer not processor


def _onechart_registry(handle: _FakeOneChartHandle) -> Any:
    """Build a fake registry that yields the provided OneChart handle."""

    class _Reg:
        def use(self, name: str) -> Any:  # noqa: ARG002
            return _yields(handle)

    return _Reg()


def test_is_onechart_handle_detects_class_name() -> None:
    """`_is_onechart_handle` matches against model class name
    case-insensitively. This is the dispatch gate that selects the
    `.chat()` path."""
    from memex.parse.chart_ocr_backend import _is_onechart_handle

    handle = _FakeOneChartHandle(chat_return="x")
    assert _is_onechart_handle(handle) is True

    # Negative: a DePlot/Pix2Struct-style handle must NOT be detected
    # as OneChart.
    deplot_handle = _FakeChartOCRHandle()
    assert _is_onechart_handle(deplot_handle) is False


@pytest.mark.asyncio
async def test_onechart_success_returns_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OneChart returns a python-dict-style string + reliable_check
    True. The backend parses the dict, converts to markdown, and
    threads it through as the chunk's `markdown` field."""
    handle = _FakeOneChartHandle(
        chat_return=(
            "{'title': 'Transistor density', 'values': "
            "{'2004': 117, '2010': 1170, '2022': 50000}} "
            "reliable_check: True",
            True,
        )
    )
    monkeypatch.setattr(
        "memex.parse.chart_ocr_backend.get_registry",
        lambda: _onechart_registry(handle),
    )
    monkeypatch.setattr(
        chart_ocr_backend,
        "_render_figure_to_image",
        lambda pdf, page, bbox, scale=2.5: _FakeImage(page),
    )

    figures = [
        FigureMetadata(
            page_no=1,
            bbox=(0.0, 0.0, 500.0, 400.0),
            classification="bar_chart",
            classification_confidence=0.97,
        ),
    ]
    out = await chart_ocr_extract(
        source_pdf=Path("/fake.pdf"),
        figures=figures,
        min_area_sqpt=0.0,
    )
    assert len(out) == 1
    assert isinstance(out[0], ChartOCROutput)
    md = out[0].markdown
    assert "Transistor density" in md
    assert "2022" in md
    assert "50000" in md
    # The markdown is a table, not python dict syntax.
    assert "| label | value |" in md
    # reliable_check token leaked-out check — it must be stripped.
    assert "reliable_check" not in md


@pytest.mark.asyncio
async def test_onechart_unreliable_check_returns_empty_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When OneChart's `reliable_check=False`, the backend treats the
    extraction as a model-confessed failure and returns empty markdown.
    The stitch step then leaves the `<!-- image -->` placeholder
    unchanged. Preserves HARD GATES on counterfactual queries."""
    handle = _FakeOneChartHandle(
        chat_return=(
            "{'title': 'X', 'values': {'A': 999}}",
            False,  # reliable_check=False — model self-flags unreliable
        )
    )
    monkeypatch.setattr(
        "memex.parse.chart_ocr_backend.get_registry",
        lambda: _onechart_registry(handle),
    )
    monkeypatch.setattr(
        chart_ocr_backend,
        "_render_figure_to_image",
        lambda pdf, page, bbox, scale=2.5: _FakeImage(page),
    )

    figures = [
        FigureMetadata(
            page_no=1,
            bbox=(0.0, 0.0, 500.0, 400.0),
            classification="bar_chart",
            classification_confidence=0.97,
        ),
    ]
    out = await chart_ocr_extract(
        source_pdf=Path("/fake.pdf"),
        figures=figures,
        min_area_sqpt=0.0,
    )
    assert len(out) == 1
    assert isinstance(out[0], ChartOCROutput)
    assert out[0].markdown == ""  # empty → stitch leaves placeholder


@pytest.mark.asyncio
async def test_onechart_string_only_return_parses_reliable_from_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some OneChart revisions return only a string (not a tuple).
    The backend then scans the text for the `reliable_check` token
    and gates on it. Tests both True and False from-text variants.
    """
    # Case 1: text contains reliable_check=False → return empty.
    handle = _FakeOneChartHandle(
        chat_return="{'a': 1} reliable_check=False"
    )
    monkeypatch.setattr(
        "memex.parse.chart_ocr_backend.get_registry",
        lambda: _onechart_registry(handle),
    )
    monkeypatch.setattr(
        chart_ocr_backend,
        "_render_figure_to_image",
        lambda pdf, page, bbox, scale=2.5: _FakeImage(page),
    )

    figures = [
        FigureMetadata(
            page_no=1,
            bbox=(0.0, 0.0, 500.0, 400.0),
            classification="bar_chart",
            classification_confidence=0.97,
        ),
    ]
    out = await chart_ocr_extract(
        source_pdf=Path("/fake.pdf"),
        figures=figures,
        min_area_sqpt=0.0,
    )
    assert isinstance(out[0], ChartOCROutput)
    assert out[0].markdown == ""


# ----------------------------------------------------------------------
# P3.3-c Path A — UniChart Donut VisionEncoderDecoder tests
# ----------------------------------------------------------------------


def test_unichart_table_to_markdown_basic() -> None:
    """`row1col1 | row1col2 &&& row2col1 | row2col2` → markdown table.
    First row is the header by convention."""
    from memex.parse.chart_ocr_backend import _unichart_table_to_markdown

    raw = "Year | Revenue &&& 2020 | 1.2 &&& 2021 | 2.4"
    md = _unichart_table_to_markdown(raw)
    assert "| Year | Revenue |" in md
    assert "| --- | --- |" in md
    assert "| 2020 | 1.2 |" in md
    assert "| 2021 | 2.4 |" in md


def test_unichart_table_to_markdown_empty_returns_empty() -> None:
    """Degenerate inputs collapse to empty markdown so the stitch step
    leaves the placeholder unchanged."""
    from memex.parse.chart_ocr_backend import _unichart_table_to_markdown

    assert _unichart_table_to_markdown("") == ""
    assert _unichart_table_to_markdown("   ") == ""
    assert _unichart_table_to_markdown("&&&") == ""


def test_unichart_table_to_markdown_padding() -> None:
    """When rows have unequal column counts, the shorter rows are
    right-padded with empty cells so the markdown table is well-formed."""
    from memex.parse.chart_ocr_backend import _unichart_table_to_markdown

    raw = "A | B | C &&& 1 | 2 &&& X | Y | Z"
    md = _unichart_table_to_markdown(raw)
    # Header should set column count to 3
    assert "| A | B | C |" in md
    # Short row padded with empty cell
    assert "| 1 | 2 |  |" in md
    # Full row passes through
    assert "| X | Y | Z |" in md


def test_unichart_table_to_markdown_single_row() -> None:
    """One-row output: emit just the header row (no data). Better than
    silently swallowing the extraction."""
    from memex.parse.chart_ocr_backend import _unichart_table_to_markdown

    md = _unichart_table_to_markdown("Only | One | Row")
    assert "| Only | One | Row |" in md
    assert "| --- | --- | --- |" in md


class _FakeUniChartEncoder:
    """Stand-in encoder so `_is_unichart_handle` recognises the class
    name. Match against 'DonutSwin' family."""

    pass


_FakeUniChartEncoder.__name__ = "DonutSwinModel"


class _FakeUniChartModel:
    """Mimics a VisionEncoderDecoder model exposing the surface the
    UniChart transcribe path uses: `.encoder` (for dispatch detection),
    `.decoder.config.max_position_embeddings`, `.generate()`, `.device`,
    `.dtype`."""

    def __init__(self, sequences: object) -> None:
        # Class name must be VisionEncoderDecoderModel so the
        # `_is_unichart_handle` first guard passes.
        self.encoder = _FakeUniChartEncoder()
        self.decoder = type(
            "D", (), {"config": type("C", (), {"max_position_embeddings": 1536})}
        )()
        self._sequences = sequences
        # Stub device/dtype attributes used by the transcribe path.
        # Both must be torch-typed for the `.to(device, dtype=...)` call
        # the helper makes on the processor's pixel_values.
        import torch

        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def generate(self, *args, **kwargs):  # noqa: ARG002
        import torch
        # Return a namespace whose `.sequences` attribute is what the
        # processor.batch_decode() call will consume.
        return type("Out", (), {"sequences": torch.zeros(1, 10, dtype=torch.long)})()


_FakeUniChartModel.__name__ = "VisionEncoderDecoderModel"


class _FakeUniChartProcessor:
    """Mimics DonutProcessor: callable for image input, `.tokenizer`
    sub-object with `pad_token_id`/`eos_token_id`/`unk_token_id` +
    `pad_token`/`eos_token` strings + `__call__` for prompt tokenisation,
    `.batch_decode` for output."""

    def __init__(self, decoded: str) -> None:
        self._decoded = decoded

        class _Tok:
            pad_token_id = 0
            eos_token_id = 1
            unk_token_id = 2
            pad_token = "<pad>"
            eos_token = "</s>"

            def __call__(self_inner, text, **kw):  # noqa: ARG002, N805
                import torch

                return type("T", (), {"input_ids": torch.zeros(1, 3, dtype=torch.long)})()

        self.tokenizer = _Tok()

    def __call__(self, image, **kw):  # noqa: ARG002
        import torch

        return type("P", (), {"pixel_values": torch.zeros(1, 3, 32, 32)})()

    def batch_decode(self, sequences, **kw):  # noqa: ARG002
        return [self._decoded]


class _FakeUniChartHandle:
    def __init__(self, decoded: str) -> None:
        self.model = _FakeUniChartModel(sequences=None)
        self.processor = _FakeUniChartProcessor(decoded=decoded)


def _unichart_registry(handle: _FakeUniChartHandle) -> Any:
    class _Reg:
        def use(self, name: str) -> Any:  # noqa: ARG002
            return _yields(handle)

    return _Reg()


def test_is_unichart_handle_detects_class_name() -> None:
    """`_is_unichart_handle` matches `VisionEncoderDecoderModel` with
    a `donut-swin` encoder."""
    from memex.parse.chart_ocr_backend import _is_unichart_handle

    handle = _FakeUniChartHandle(decoded="")
    assert _is_unichart_handle(handle) is True

    # Negative: a Pix2Struct-style handle must NOT be detected as UniChart.
    deplot_handle = _FakeChartOCRHandle()
    assert _is_unichart_handle(deplot_handle) is False


@pytest.mark.asyncio
async def test_unichart_success_returns_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UniChart emits a `&&&`/`|` table; the backend converts it to
    markdown and threads through as the chunk's markdown."""
    decoded = (
        "<pad><pad><data_table_generation> <s_answer>"
        "Year | Revenue &&& 2020 | 1.2 &&& 2021 | 2.4</s>"
    )
    handle = _FakeUniChartHandle(decoded=decoded)
    monkeypatch.setattr(
        "memex.parse.chart_ocr_backend.get_registry",
        lambda: _unichart_registry(handle),
    )
    monkeypatch.setattr(
        chart_ocr_backend,
        "_render_figure_to_image",
        lambda pdf, page, bbox, scale=2.5: _FakeImage(page),
    )

    figures = [
        FigureMetadata(
            page_no=1,
            bbox=(0.0, 0.0, 500.0, 400.0),
            classification="bar_chart",
            classification_confidence=0.97,
        ),
    ]
    out = await chart_ocr_extract(
        source_pdf=Path("/fake.pdf"),
        figures=figures,
        min_area_sqpt=0.0,
    )
    assert len(out) == 1
    assert isinstance(out[0], ChartOCROutput)
    md = out[0].markdown
    assert "| Year | Revenue |" in md
    assert "| 2020 | 1.2 |" in md


@pytest.mark.asyncio
async def test_unichart_runtime_error_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RuntimeError during generate() (CUDA OOM mid-beam-search,
    processor input mismatch, etc.) must NOT crash the whole pass;
    the affected figure returns empty markdown."""

    class _ErrorHandle:
        def __init__(self) -> None:
            self.model = _FakeUniChartModel(sequences=None)
            # Override generate to raise RuntimeError.
            self.model.generate = lambda *a, **kw: (_ for _ in ()).throw(  # type: ignore
                RuntimeError("CUDA OOM mid-beam")
            )
            self.processor = _FakeUniChartProcessor(decoded="")

    handle = _ErrorHandle()
    monkeypatch.setattr(
        "memex.parse.chart_ocr_backend.get_registry",
        lambda: _unichart_registry(handle),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        chart_ocr_backend,
        "_render_figure_to_image",
        lambda pdf, page, bbox, scale=2.5: _FakeImage(page),
    )

    figures = [
        FigureMetadata(
            page_no=1,
            bbox=(0.0, 0.0, 500.0, 400.0),
            classification="bar_chart",
            classification_confidence=0.97,
        ),
    ]
    out = await chart_ocr_extract(
        source_pdf=Path("/fake.pdf"),
        figures=figures,
        min_area_sqpt=0.0,
    )
    assert isinstance(out[0], ChartOCROutput)
    assert out[0].markdown == ""


@pytest.mark.asyncio
async def test_onechart_chat_signature_mismatch_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A TypeError from OneChart's `.chat()` (signature drift across
    upstream revisions) must NOT crash the entire chart-OCR pass.
    Instead, the affected figure returns empty markdown and the
    siblings continue."""
    handle = _FakeOneChartHandle(
        chat_return=TypeError("chat() got unexpected keyword argument 'render'")
    )
    monkeypatch.setattr(
        "memex.parse.chart_ocr_backend.get_registry",
        lambda: _onechart_registry(handle),
    )
    monkeypatch.setattr(
        chart_ocr_backend,
        "_render_figure_to_image",
        lambda pdf, page, bbox, scale=2.5: _FakeImage(page),
    )

    figures = [
        FigureMetadata(
            page_no=1,
            bbox=(0.0, 0.0, 500.0, 400.0),
            classification="bar_chart",
            classification_confidence=0.97,
        ),
    ]
    out = await chart_ocr_extract(
        source_pdf=Path("/fake.pdf"),
        figures=figures,
        min_area_sqpt=0.0,
    )
    assert isinstance(out[0], ChartOCROutput)
    assert out[0].markdown == ""
