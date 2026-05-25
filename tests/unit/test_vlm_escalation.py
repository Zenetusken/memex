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
from memex.parse.docling_backend import DoclingConversion, DoclingPageOutput


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
