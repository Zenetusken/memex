"""ChartOCRCache + its wiring into `chart_ocr_backend.chart_ocr_extract`.

The cache makes the non-deterministic chart-OCR model reproducible by
construction: each figure is extracted once per `(pdf-bytes, page, bbox,
model, version)` key and reused thereafter. These tests cover the store
mechanics and the extract cache path (miss→extract→store, hit→reuse-skip-GPU,
refresh→re-extract, and empty-result caching) with a faked model (no torch).
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from memex.parse import chart_ocr_backend
from memex.parse.chart_ocr_backend import ChartOCROutput, chart_ocr_extract
from memex.parse.chart_ocr_cache import ChartOCRCache
from memex.parse.docling_backend import FigureMetadata

# ── store mechanics ──────────────────────────────────────────────────────


async def test_put_get_roundtrip(tmp_path: Path) -> None:
    cache = await ChartOCRCache.open(tmp_path)
    try:
        assert await cache.get("k1") is None  # miss
        await cache.put(
            "k1",
            pdf_sha256="abc",
            page_no=3,
            bbox_key="0_0_10_10",
            chart_ocr_model="m",
            markdown="| a | b |",
        )
        assert await cache.get("k1") == "| a | b |"
    finally:
        await cache.close()


async def test_insert_or_ignore_first_writer_wins(tmp_path: Path) -> None:
    cache = await ChartOCRCache.open(tmp_path)
    try:
        for md in ("first", "second"):
            await cache.put(
                "k1", pdf_sha256="abc", page_no=1, bbox_key="b", chart_ocr_model="m", markdown=md
            )
        assert await cache.get("k1") == "first"
    finally:
        await cache.close()


async def test_delete_by_pdf_scopes_to_one_document(tmp_path: Path) -> None:
    cache = await ChartOCRCache.open(tmp_path)
    try:
        await cache.put(
            "a:1", pdf_sha256="A", page_no=1, bbox_key="b", chart_ocr_model="m", markdown="x"
        )
        await cache.put(
            "a:2", pdf_sha256="A", page_no=2, bbox_key="b", chart_ocr_model="m", markdown="y"
        )
        await cache.put(
            "b:1", pdf_sha256="B", page_no=1, bbox_key="b", chart_ocr_model="m", markdown="z"
        )
        assert await cache.delete_by_pdf("A") == 2
        assert await cache.get("a:1") is None
        assert await cache.get("b:1") == "z"  # other document untouched
    finally:
        await cache.close()


async def test_persists_on_disk_across_reopen(tmp_path: Path) -> None:
    cache = await ChartOCRCache.open(tmp_path)
    await cache.put(
        "k", pdf_sha256="A", page_no=1, bbox_key="b", chart_ocr_model="m", markdown="kept"
    )
    await cache.close()
    assert (tmp_path / ".memex" / "chart_ocr_cache.sqlite").is_file()
    reopened = await ChartOCRCache.open(tmp_path)
    try:
        assert await reopened.get("k") == "kept"
    finally:
        await reopened.close()


# ── chart_ocr_extract wiring ─────────────────────────────────────────────


def _fig() -> FigureMetadata:
    # area 40000 (>= _MIN_FIGURE_AREA_SQPT) + a chart class so it passes the filters
    return FigureMetadata(
        page_no=1,
        bbox=(0.0, 0.0, 200.0, 200.0),
        classification="bar_chart",
        classification_confidence=0.9,
    )


def _fake_env(monkeypatch: Any, calls: list[int], markdown: str = "chart p1") -> None:
    async def fake_extract(
        handle: object, source_pdf: Path, figure: FigureMetadata, max_new_tokens: int
    ) -> ChartOCROutput:
        calls.append(figure.page_no)
        return ChartOCROutput(page_no=figure.page_no, bbox=figure.bbox, markdown=markdown)

    class _FakeRegistry:
        @contextlib.asynccontextmanager
        async def use(self, _name: str) -> AsyncGenerator[object]:
            yield object()

    monkeypatch.setattr(chart_ocr_backend, "_extract_with_handle", fake_extract)
    monkeypatch.setattr(chart_ocr_backend, "get_registry", lambda: _FakeRegistry())
    monkeypatch.setattr(
        chart_ocr_backend,
        "get_settings",
        lambda: SimpleNamespace(models=SimpleNamespace(chart_ocr="test-chart-ocr")),
    )


async def test_extract_misses_then_reuses(tmp_path: Path, monkeypatch: Any) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.7 fake bytes")
    cache = await ChartOCRCache.open(tmp_path)
    calls: list[int] = []
    _fake_env(monkeypatch, calls)
    try:
        # 1st: miss → model extracts.
        r1 = await chart_ocr_extract(source_pdf=pdf, figures=[_fig()], cache=cache)
        assert calls == [1]
        assert r1[0].markdown == "chart p1"  # type: ignore[union-attr]

        # 2nd: hit → model NOT invoked, byte-identical (reproducible).
        calls.clear()
        r2 = await chart_ocr_extract(source_pdf=pdf, figures=[_fig()], cache=cache)
        assert calls == []
        assert r2[0].markdown == "chart p1"  # type: ignore[union-attr]

        # refresh: bust this doc → re-extract.
        calls.clear()
        await chart_ocr_extract(source_pdf=pdf, figures=[_fig()], cache=cache, refresh=True)
        assert calls == [1]
    finally:
        await cache.close()


async def test_best_of_n_keeps_longest_draw(tmp_path: Path, monkeypatch: Any) -> None:
    """extraction_samples=N takes N independent draws and keeps the LONGEST
    markdown (completeness proxy; mirrors the VLM best-of-N). A shorter draw
    AND an empty/refused draw (len 0) both lose; the chosen draw is what the
    cache then freezes (so a re-parse replays the most-complete extraction)."""
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF best-of-n")
    cache = await ChartOCRCache.open(tmp_path)
    # Draw 2 is longest; draw 1 is shorter; draw 3 is an empty refusal.
    draws = iter(["short", "the longest most complete table", ""])

    async def fake_extract(
        handle: object, source_pdf: Path, figure: FigureMetadata, max_new_tokens: int
    ) -> ChartOCROutput:
        return ChartOCROutput(page_no=figure.page_no, bbox=figure.bbox, markdown=next(draws))

    class _FakeRegistry:
        @contextlib.asynccontextmanager
        async def use(self, _name: str) -> AsyncGenerator[object]:
            yield object()

    monkeypatch.setattr(chart_ocr_backend, "_extract_with_handle", fake_extract)
    monkeypatch.setattr(chart_ocr_backend, "get_registry", lambda: _FakeRegistry())
    monkeypatch.setattr(
        chart_ocr_backend,
        "get_settings",
        lambda: SimpleNamespace(models=SimpleNamespace(chart_ocr="test-chart-ocr")),
    )
    try:
        r = await chart_ocr_extract(
            source_pdf=pdf, figures=[_fig()], cache=cache, extraction_samples=3
        )
        assert r[0].markdown == "the longest most complete table"  # type: ignore[union-attr]
        # The chosen (longest) draw is cached → a re-parse replays it (hit).
        r2 = await chart_ocr_extract(
            source_pdf=pdf, figures=[_fig()], cache=cache, extraction_samples=3
        )
        assert r2[0].markdown == "the longest most complete table"  # type: ignore[union-attr]
    finally:
        await cache.close()


async def test_empty_extraction_is_cached(tmp_path: Path, monkeypatch: Any) -> None:
    """A "" extraction ("not a chart / no parseable data") is a legit,
    reproducible result and IS cached — so a re-parse replays it rather than
    re-running the model (unlike the VLM's min-length guard)."""
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF fake")
    cache = await ChartOCRCache.open(tmp_path)
    calls: list[int] = []
    _fake_env(monkeypatch, calls, markdown="")
    try:
        await chart_ocr_extract(source_pdf=pdf, figures=[_fig()], cache=cache)
        assert calls == [1]
        calls.clear()
        await chart_ocr_extract(source_pdf=pdf, figures=[_fig()], cache=cache)
        assert calls == []  # "" was cached → not re-run
    finally:
        await cache.close()
