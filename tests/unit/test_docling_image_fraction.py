"""`_image_fraction_by_page` — the figure-area-dominance signal that
drives VLM escalation (`pipeline._route_and_escalate`). Pure geometry,
so it runs without docling installed (the worker imports docling lazily
inside `_convert_to_payload`)."""

from __future__ import annotations

from typing import Any

from memex.parse.docling_worker import _image_fraction_by_page


def _fig(page_no: int, x0: float, y0: float, x1: float, y1: float) -> dict[str, Any]:
    # bbox = [x0, y_bottom, x1, y_top] in PDF points (worker convention)
    return {"page_no": page_no, "bbox": [x0, y0, x1, y1]}


def test_no_figures_is_zero() -> None:
    assert _image_fraction_by_page([], {1: 10000.0}) == {1: 0.0}


def test_half_page_figure() -> None:
    # 100x100 page (area 10000); one 100x50 figure (area 5000) -> 0.5
    assert _image_fraction_by_page([_fig(1, 0, 0, 100, 50)], {1: 10000.0}) == {1: 0.5}


def test_multiple_figures_sum_per_page() -> None:
    figs = [_fig(1, 0, 0, 100, 25), _fig(1, 0, 50, 100, 75)]  # two 100x25 = 5000 total
    assert _image_fraction_by_page(figs, {1: 10000.0}) == {1: 0.5}


def test_overlapping_figures_clamped_to_one() -> None:
    figs = [_fig(1, 0, 0, 100, 100), _fig(1, 0, 0, 100, 100)]  # 2x full page
    assert _image_fraction_by_page(figs, {1: 10000.0}) == {1: 1.0}


def test_zero_area_page_is_zero_not_division_error() -> None:
    assert _image_fraction_by_page([_fig(1, 0, 0, 100, 100)], {1: 0.0}) == {1: 0.0}


def test_page_without_figures_among_others() -> None:
    out = _image_fraction_by_page([_fig(1, 0, 0, 100, 100)], {1: 10000.0, 2: 10000.0})
    assert out[1] == 1.0
    assert out[2] == 0.0  # page 2 has no figure
