"""scan→VLM parse route (spec: docs/specs/scan-vlm-parse.md). The VLM serving + cache +
page-count are faked; this pins (a) the transcribe→stitch→write→manifest end-to-end and
(b) the routing — a scan goes to the VLM when enabled, else falls through to Docling.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from memex.core.config import MemexSettings, set_settings
from memex.parse import pipeline
from memex.parse.docling_backend import DoclingPageOutput
from memex.parse.pipeline import (
    ParseResult,
    _parse_pdf,
    _parse_scan_with_vlm,
    _PreFilterDecision,
)


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemexSettings]:
    (tmp_path / "documents").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    s = MemexSettings()  # type: ignore[call-arg]
    set_settings(s)
    yield s
    set_settings(None)


class _FakeCache:
    @classmethod
    async def open(cls, vault_path: Path) -> _FakeCache:
        return cls()

    async def close(self) -> None:
        return None


class _FakePdf:
    def __init__(self, path: Any, n: int = 2) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def close(self) -> None:
        return None


@asynccontextmanager
async def _no_pause() -> AsyncIterator[None]:
    yield


def _seed_pdf(vault: Path, doc_id: str) -> Path:
    d = vault / "documents" / doc_id
    d.mkdir(parents=True, exist_ok=True)
    src = d / "source.pdf"
    src.write_bytes(b"%PDF-1.7 fake scan bytes")  # content unused (pypdfium2 + VLM faked)
    return src


@pytest.mark.asyncio
async def test_scan_route_transcribes_all_pages_and_writes_engine_scan(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = settings.vault_path
    doc_id = "5c4n0001-handwritten-note"
    src = _seed_pdf(vault, doc_id)
    seen: dict[str, Any] = {}

    async def fake_convert(
        *, source_pdf: Path, page_numbers: list[int], cache: Any = None, refresh_vlm: bool = False
    ) -> dict[int, DoclingPageOutput | Exception]:
        seen["page_numbers"] = page_numbers
        out: dict[int, DoclingPageOutput | Exception] = {
            p: DoclingPageOutput(
                page=p, markdown=f"# Page {p}\n\nTranscribed handwriting {p}.", confidence=1.0
            )
            for p in page_numbers
        }
        return out

    monkeypatch.setattr(pipeline, "vlm_convert_pages", fake_convert)
    monkeypatch.setattr(pipeline, "VLMTranscriptionCache", _FakeCache)
    monkeypatch.setattr(pipeline, "pause_vllm_for_gpu", _no_pause)
    monkeypatch.setattr("pypdfium2.PdfDocument", lambda path: _FakePdf(path, 2))

    result = await _parse_scan_with_vlm(vault, doc_id, src)

    assert result.engine == "scan"
    assert [p.engine for p in result.pages] == ["scan", "scan"]
    assert seen["page_numbers"] == [1, 2]  # 1-based, ALL pages (not just the first window)
    body = (vault / "documents" / f"{doc_id}.md").read_text(encoding="utf-8")
    assert "Transcribed handwriting 1." in body  # early page
    assert "Transcribed handwriting 2." in body  # AND the late page (no truncation)
    # Companion arc-3: the scan route builds the body AS the per-page join, so the page-boundary
    # round-trip always reconstructs → citation-grade char_start on every (content) page.
    from memex.vault.store import read_document

    assert all(p.char_start >= 0 for p in result.pages)
    starts = {p.page: p.char_start for p in result.pages}
    assert starts[1] == 0
    doc = await read_document(vault, doc_id)
    assert doc.body[starts[2] :].startswith("# Page 2")  # page 2's offset lands on its content


@pytest.mark.asyncio
async def test_parse_pdf_routes_scan_to_vlm_only_when_enabled(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = settings.vault_path
    doc_id = "5c4n0002"
    src = _seed_pdf(vault, doc_id)

    async def fake_prefilter(vp: Path, did: str, s: Path) -> _PreFilterDecision:
        return _PreFilterDecision(result=None, is_scan=True)

    async def fake_scan(vp: Path, did: str, s: Path, *, refresh_vlm: bool = False) -> ParseResult:
        return ParseResult(
            doc_id=did, correlation_id="t", engine="scan-route", pages=[], markdown_bytes=0
        )

    async def fake_docling(
        vp: Path, did: str, s: Path, *, force_ocr: Any = None, refresh_vlm: bool = False
    ) -> ParseResult:
        return ParseResult(
            doc_id=did, correlation_id="t", engine="docling-route", pages=[], markdown_bytes=0
        )

    monkeypatch.setattr(pipeline, "_parse_with_pymupdf", fake_prefilter)
    monkeypatch.setattr(pipeline, "_parse_scan_with_vlm", fake_scan)
    monkeypatch.setattr(pipeline, "_parse_with_docling", fake_docling)

    # VLM enabled → the scan goes to the VLM route.
    settings.parse.disable_vlm = False
    assert (await _parse_pdf(vault, doc_id, src)).engine == "scan-route"

    # VLM disabled (the default) → the gate is off; fall through to Docling (no regression).
    settings.parse.disable_vlm = True
    assert (await _parse_pdf(vault, doc_id, src)).engine == "docling-route"
