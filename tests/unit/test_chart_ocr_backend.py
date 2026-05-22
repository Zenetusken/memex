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
