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


class _FakeChartOCRHandle:
    """Stand-in for ChartOCRHandle; never touched directly because the
    backend's `_chart_ocr_transcribe_sync` is monkeypatched."""


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
            page_no=1, bbox=(10.0, 20.0, 100.0, 80.0), caption="Fig 1"
        ),
        FigureMetadata(
            page_no=2, bbox=(50.0, 60.0, 200.0, 150.0), caption=None
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
async def test_render_error_returned_per_figure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If `_render_figure_to_image` raises `PDFFigureRenderError` for
    a specific figure, that figure's slot in the result list holds the
    exception while other figures still process normally. The backend
    must not propagate the error out of the batch.
    """
    figures = [
        FigureMetadata(page_no=1, bbox=(10.0, 20.0, 100.0, 80.0)),
        FigureMetadata(page_no=2, bbox=(0.0, 0.0, 5.0, 5.0)),  # bad bbox
        FigureMetadata(page_no=3, bbox=(50.0, 60.0, 200.0, 150.0)),
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
