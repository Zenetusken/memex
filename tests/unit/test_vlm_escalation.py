"""VLM escalation routing in the parse pipeline (`_route_and_escalate`).

The escalation path sends below-confidence Docling pages through the VLM
(Qwen2.5-VL). On the 12 GB rig the VLM loads in-process, so the call is
wrapped in the vLLM-pause context manager and the VLM is unloaded before
vLLM restarts (symmetric with chart-OCR). These tests fake the pause (its
pkill/restart internals are exercised by the chart-OCR path) and assert
the escalation + unload wiring.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator
from pathlib import Path

import structlog

from memex.parse import pipeline as P
from memex.parse.docling_backend import (
    DoclingConversion,
    DoclingPageOutput,
    FigureMetadata,
)


def _conversion() -> DoclingConversion:
    return DoclingConversion(
        markdown="page one\n\npage two",
        pages=[
            DoclingPageOutput(page=1, markdown="page one", confidence=0.9),  # above threshold
            DoclingPageOutput(page=2, markdown="page two (low)", confidence=0.4),  # below
        ],
    )


async def test_escalation_wraps_pause_and_unloads_vlm(monkeypatch) -> None:
    entered: list[bool] = []

    @contextlib.asynccontextmanager
    async def fake_pause() -> AsyncGenerator[None]:
        entered.append(True)
        yield

    async def fake_vlm(*, source_pdf: Path, page_numbers: list[int], **_kw: object):
        assert page_numbers == [2]  # only the below-threshold page escalates
        return {2: DoclingPageOutput(page=2, markdown="VLM-corrected page two", confidence=0.95)}

    unloaded: list[str] = []

    class _FakeRegistry:
        async def unload(self, name: str) -> None:
            unloaded.append(name)

    monkeypatch.setattr(P, "_pause_vllm_for_gpu_parse", fake_pause)
    monkeypatch.setattr(P, "vlm_convert_pages", fake_vlm)
    monkeypatch.setattr("memex.models.registry.get_registry", lambda: _FakeRegistry())

    decisions, conv = await P._route_and_escalate(
        _conversion(),
        source=Path("scan.pdf"),
        threshold=0.65,
        image_area_threshold=0.5,
        disable_vlm=False,
        log=structlog.get_logger("test"),
    )

    by_page = {d.page: d for d in decisions}
    assert by_page[1].engine == "docling"  # high-confidence page kept
    assert by_page[2].engine == "vlm"  # low-confidence page escalated
    assert "VLM-corrected page two" in conv.markdown  # re-stitched into the doc
    assert entered == [True]  # the VLM call ran inside the vLLM-pause
    assert unloaded == ["vlm"]  # VLM unloaded before vLLM restarts


async def test_disable_vlm_skips_escalation_pause_and_unload(monkeypatch) -> None:
    entered: list[bool] = []

    @contextlib.asynccontextmanager
    async def fake_pause() -> AsyncGenerator[None]:
        entered.append(True)
        yield

    async def fake_vlm(*, source_pdf: Path, page_numbers: list[int], **_kw: object):
        raise AssertionError("vlm_convert_pages must not be called when disable_vlm=True")

    monkeypatch.setattr(P, "_pause_vllm_for_gpu_parse", fake_pause)
    monkeypatch.setattr(P, "vlm_convert_pages", fake_vlm)

    decisions, conv = await P._route_and_escalate(
        _conversion(),
        source=Path("scan.pdf"),
        threshold=0.65,
        image_area_threshold=0.5,
        disable_vlm=True,
        log=structlog.get_logger("test"),
    )

    assert all(d.engine == "docling" for d in decisions)  # nothing escalated
    assert entered == []  # vLLM never paused (no VLM work)
    assert conv.markdown == "page one\n\npage two"  # unchanged


async def test_image_dominant_page_escalates_despite_high_confidence(monkeypatch) -> None:
    """The dominant real-world trigger. Docling reports full confidence
    for a diagram/screenshot page (it read the title) while losing the
    figure content, so confidence never escalates it. A high
    `image_fraction` must escalate it anyway — and a confident page with
    only a small figure must NOT escalate.
    """

    @contextlib.asynccontextmanager
    async def fake_pause() -> AsyncGenerator[None]:
        yield

    async def fake_vlm(*, source_pdf: Path, page_numbers: list[int], **_kw: object):
        assert page_numbers == [2]  # only the image-dominant page escalates
        return {2: DoclingPageOutput(page=2, markdown="VLM diagram transcription", confidence=1.0)}

    class _FakeRegistry:
        async def unload(self, name: str) -> None:
            return None

    monkeypatch.setattr(P, "_pause_vllm_for_gpu_parse", fake_pause)
    monkeypatch.setattr(P, "vlm_convert_pages", fake_vlm)
    monkeypatch.setattr("memex.models.registry.get_registry", lambda: _FakeRegistry())

    conversion = DoclingConversion(
        markdown="text page\n\ndiagram page",
        pages=[
            # confident, only a small figure → kept
            DoclingPageOutput(page=1, markdown="text page", confidence=1.0, image_fraction=0.1),
            # confident BUT figure-dominant → escalated
            DoclingPageOutput(page=2, markdown="diagram page", confidence=1.0, image_fraction=0.6),
        ],
    )
    decisions, conv = await P._route_and_escalate(
        conversion,
        source=Path("deck.pdf"),
        threshold=0.65,
        image_area_threshold=0.3,
        disable_vlm=False,
        log=structlog.get_logger("test"),
    )

    by_page = {d.page: d for d in decisions}
    assert by_page[1].engine == "docling"  # confident + only a small figure
    assert by_page[2].engine == "vlm"  # image-dominant despite confidence 1.0
    assert "VLM diagram transcription" in conv.markdown


_BIG = (0.0, 0.0, 200.0, 200.0)  # 40000 sq-pt > _MIN_DIAGRAM_AREA_SQPT
_TINY = (0.0, 0.0, 50.0, 50.0)  # 2500 sq-pt < min → badge/watermark


def test_diagram_pages_selects_only_sized_confident_diagram_classes() -> None:
    """The classification arm marks a page only when it carries a diagram-
    class figure that is BOTH confidently classified AND large enough —
    decorative classes, low-trust labels, badge-sized figures, and
    unclassified figures are all excluded.
    """
    figures = [
        FigureMetadata(
            page_no=1, bbox=_BIG, classification="flow_chart", classification_confidence=0.9
        ),
        FigureMetadata(
            page_no=2, bbox=_BIG, classification="logo", classification_confidence=0.99
        ),  # decorative
        FigureMetadata(
            page_no=3, bbox=_BIG, classification="flow_chart", classification_confidence=0.3
        ),  # low-trust
        FigureMetadata(
            page_no=4,
            bbox=_TINY,
            classification="engineering_drawing",
            classification_confidence=0.95,
        ),  # too small
        FigureMetadata(
            page_no=5,
            bbox=_BIG,
            classification="engineering_drawing",
            classification_confidence=0.8,
        ),
        FigureMetadata(
            page_no=6, bbox=_BIG, classification=None, classification_confidence=0.0
        ),  # unclassified
    ]
    pages = P._diagram_pages(
        figures,
        classes=frozenset({"flow_chart", "engineering_drawing", "screenshot"}),
        min_confidence=0.5,
    )
    assert pages == {1, 5}


def test_diagram_pages_empty_classes_disables_the_arm() -> None:
    figures = [
        FigureMetadata(
            page_no=1, bbox=_BIG, classification="flow_chart", classification_confidence=0.9
        )
    ]
    assert P._diagram_pages(figures, classes=frozenset(), min_confidence=0.5) == set()


async def test_diagram_class_page_escalates_despite_confidence_and_low_image_fraction(
    monkeypatch,
) -> None:
    """A flow chart on a text-heavy slide: Docling-confident, image_fraction
    below threshold — but the classification arm routes it to the VLM (the
    chart-OCR pass excludes diagram classes, so otherwise it is lost).
    """

    @contextlib.asynccontextmanager
    async def fake_pause() -> AsyncGenerator[None]:
        yield

    async def fake_vlm(*, source_pdf: Path, page_numbers: list[int], **_kw: object):
        assert page_numbers == [2]  # only the flow-chart page escalates
        return {
            2: DoclingPageOutput(page=2, markdown="VLM flowchart transcription", confidence=1.0)
        }

    class _FakeRegistry:
        async def unload(self, name: str) -> None:
            return None

    monkeypatch.setattr(P, "_pause_vllm_for_gpu_parse", fake_pause)
    monkeypatch.setattr(P, "vlm_convert_pages", fake_vlm)
    monkeypatch.setattr("memex.models.registry.get_registry", lambda: _FakeRegistry())

    conversion = DoclingConversion(
        markdown="plain text page\n\nslide with a flowchart",
        pages=[
            DoclingPageOutput(
                page=1, markdown="plain text page", confidence=1.0, image_fraction=0.05
            ),
            DoclingPageOutput(
                page=2, markdown="slide with a flowchart", confidence=1.0, image_fraction=0.12
            ),
        ],
        figures=[
            FigureMetadata(
                page_no=2, bbox=_BIG, classification="flow_chart", classification_confidence=0.9
            ),
        ],
    )
    decisions, conv = await P._route_and_escalate(
        conversion,
        source=Path("lecture.pdf"),
        threshold=0.65,
        image_area_threshold=0.20,
        disable_vlm=False,
        log=structlog.get_logger("test"),
        diagram_classes=("flow_chart", "engineering_drawing", "screenshot"),
        diagram_min_confidence=0.5,
    )

    by_page = {d.page: d for d in decisions}
    assert by_page[1].engine == "docling"  # plain text, no diagram figure
    assert by_page[2].engine == "vlm"  # flow_chart arm fired (conf 1.0, image_fraction 0.12)
    assert "diagram-class figure" in by_page[2].rationale
    assert "VLM flowchart transcription" in conv.markdown


async def test_decorative_figure_does_not_escalate_via_diagram_arm(monkeypatch) -> None:
    """A confident text page carrying only a logo must NOT escalate — the
    classification arm is precise (logos/icons are not diagram classes).
    """

    @contextlib.asynccontextmanager
    async def fake_pause() -> AsyncGenerator[None]:
        yield

    async def fake_vlm(*, source_pdf: Path, page_numbers: list[int], **_kw: object):
        raise AssertionError("no page should escalate (logo is not a diagram class)")

    monkeypatch.setattr(P, "_pause_vllm_for_gpu_parse", fake_pause)
    monkeypatch.setattr(P, "vlm_convert_pages", fake_vlm)

    conversion = DoclingConversion(
        markdown="text with a logo",
        pages=[
            DoclingPageOutput(
                page=1, markdown="text with a logo", confidence=1.0, image_fraction=0.08
            )
        ],
        figures=[
            FigureMetadata(
                page_no=1, bbox=_BIG, classification="logo", classification_confidence=0.99
            )
        ],
    )
    decisions, conv = await P._route_and_escalate(
        conversion,
        source=Path("doc.pdf"),
        threshold=0.65,
        image_area_threshold=0.20,
        disable_vlm=False,
        log=structlog.get_logger("test"),
        diagram_classes=("flow_chart", "engineering_drawing", "screenshot"),
        diagram_min_confidence=0.5,
    )

    assert all(d.engine == "docling" for d in decisions)  # nothing escalated
    assert conv.markdown == "text with a logo"  # unchanged
