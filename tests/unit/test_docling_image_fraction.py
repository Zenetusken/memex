"""`_image_fraction_by_page` — the figure-area-dominance signal that
drives VLM escalation (`pipeline._route_and_escalate`). Pure geometry,
so it runs without docling installed (the worker imports docling lazily
inside `_convert_to_payload`)."""

from __future__ import annotations

from typing import Any

from memex.parse.docling_worker import _CHART_CLASSES, _image_fraction_by_page


def _fig(page_no: int, x0: float, y0: float, x1: float, y1: float) -> dict[str, Any]:
    # bbox = [x0, y_bottom, x1, y_top] in PDF points (worker convention)
    return {"page_no": page_no, "bbox": [x0, y0, x1, y1]}


def _cfig(
    page_no: int, x0: float, y0: float, x1: float, y1: float, *, cls: str, conf: float = 0.9
) -> dict[str, Any]:
    # a figure carrying a classifier verdict (for the chart-exclusion path)
    return {
        "page_no": page_no,
        "bbox": [x0, y0, x1, y1],
        "classification": cls,
        "classification_confidence": conf,
    }


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


# --- chart-exclusion (charts are chart-OCR's domain, not the VLM's) ---


def test_confident_chart_figure_excluded() -> None:
    # a full-page bar_chart (conf 0.9) is excluded -> fraction 0 (no VLM escalation)
    out = _image_fraction_by_page([_cfig(1, 0, 0, 100, 100, cls="bar_chart")], {1: 10000.0})
    assert out == {1: 0.0}


def test_diagram_and_screenshot_figures_kept() -> None:
    # flow_chart / engineering_drawing / screenshot are the VLM's domain -> kept
    for cls in ("flow_chart", "engineering_drawing", "screenshot_from_computer"):
        out = _image_fraction_by_page([_cfig(1, 0, 0, 100, 100, cls=cls)], {1: 10000.0})
        assert out == {1: 1.0}, cls


def test_low_confidence_chart_is_kept() -> None:
    # below the confidence floor -> don't trust the chart label -> keep the figure
    out = _image_fraction_by_page([_cfig(1, 0, 0, 100, 100, cls="bar_chart", conf=0.3)], {1: 10000.0})
    assert out == {1: 1.0}


def test_unclassified_figure_kept() -> None:
    # classification absent (v1 worker / disabled) -> not a chart -> kept
    assert _image_fraction_by_page([_fig(1, 0, 0, 100, 100)], {1: 10000.0}) == {1: 1.0}


def test_chart_dominant_page_drops_below_escalation() -> None:
    # big chart (excluded) + small diagram (kept): only the diagram area counts,
    # so the page no longer trips the 0.20 image-area arm.
    figs = [
        _cfig(1, 0, 0, 100, 80, cls="bar_chart"),  # 8000, chart -> excluded
        _cfig(1, 0, 80, 100, 90, cls="flow_chart"),  # 1000, diagram -> kept
    ]
    out = _image_fraction_by_page(figs, {1: 10000.0})
    assert out[1] == 0.1  # 1000 / 10000 — below the 0.20 threshold


def test_worker_chart_set_matches_canonical() -> None:
    # the sandbox-local worker copy must stay equal to the canonical set
    from memex.core.types import CHART_CLASS_NAMES

    assert _CHART_CLASSES == CHART_CLASS_NAMES
