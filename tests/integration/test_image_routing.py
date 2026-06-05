"""Standalone image parse route (ADR-0020, spec: docs/specs/image-ingestion.md).

A `.png`/`.jpg`/… source is wrapped into a cached 1-page PDF (`convert_image_to_pdf`) and run
through the scan→VLM route UNCONDITIONALLY — routed DIRECT to `_parse_scan_with_vlm` (not
`_parse_pdf`), so the VLM is mandatory regardless of `disable_vlm` (the audio-route precedent).
The PIL conversion + the VLM serving/cache/page-count are faked; this pins (a) the routing through
`parse_document`, (b) `engine="image"` on the result + every page, (c) convert-once caching, (d)
provenance (the original image survives), and (e) the empty-transcription HARD-gate path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from memex.core.config import MemexSettings, set_settings
from memex.core.errors import ParseConfidenceTooLow
from memex.parse import pipeline
from memex.parse.docling_backend import DoclingPageOutput


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
    def __init__(self, path: Any, n: int = 1) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def close(self) -> None:
        return None


@asynccontextmanager
async def _no_pause() -> AsyncIterator[None]:
    yield


def _seed_png(vault: Path, doc_id: str) -> Path:
    doc_dir = vault / "documents" / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    src = doc_dir / "source.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n fake image bytes")  # content unused (PIL faked)
    return src


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    convert_calls: list[Path],
    transcription: str,
    n_pages: int = 1,
) -> None:
    async def fake_convert_image(source: Path, out_dir: Path) -> Path:
        convert_calls.append(source)
        pdf = out_dir / f"{source.stem}.pdf"
        pdf.write_bytes(b"%PDF-1.7 fake image-pdf")
        return pdf

    async def fake_vlm_convert(
        *, source_pdf: Path, page_numbers: list[int], cache: Any = None, refresh_vlm: bool = False
    ) -> dict[int, DoclingPageOutput | Exception]:
        return {
            p: DoclingPageOutput(page=p, markdown=transcription, confidence=1.0)
            for p in page_numbers
        }

    monkeypatch.setattr(pipeline, "convert_image_to_pdf", fake_convert_image)
    monkeypatch.setattr(pipeline, "vlm_convert_pages", fake_vlm_convert)
    monkeypatch.setattr(pipeline, "VLMTranscriptionCache", _FakeCache)
    monkeypatch.setattr(pipeline, "pause_vllm_for_gpu", _no_pause)
    monkeypatch.setattr("pypdfium2.PdfDocument", lambda path: _FakePdf(path, n_pages))


@pytest.mark.asyncio
async def test_image_routes_through_scan_vlm_with_engine_image(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = settings.vault_path
    doc_id = "1mg00001-diagram"
    _seed_png(vault, doc_id)
    convert_calls: list[Path] = []
    # disable_vlm is the DEFAULT True — the image route must IGNORE it (VLM-mandatory).
    settings.parse.disable_vlm = True
    _install_fakes(
        monkeypatch,
        convert_calls=convert_calls,
        transcription="# Topology\n\nA core switch links three access switches.",
    )

    result = await pipeline.parse_document(doc_id)

    assert result.engine == "image"
    assert [p.engine for p in result.pages] == ["image"]
    converted = vault / "documents" / doc_id / "converted.pdf"
    assert converted.is_file()
    assert convert_calls == [vault / "documents" / doc_id / "source.png"]
    body = (vault / "documents" / f"{doc_id}.md").read_text(encoding="utf-8")
    assert "A core switch links three access switches." in body
    # Provenance: the original image is preserved and NOT named source.pdf, so source
    # resolution still resolves the .png (the converted.pdf is a separate sibling).
    assert (vault / "documents" / doc_id / "source.png").is_file()


@pytest.mark.asyncio
async def test_image_converts_once_then_reuses_cache(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = settings.vault_path
    doc_id = "1mg00002"
    _seed_png(vault, doc_id)
    convert_calls: list[Path] = []
    _install_fakes(
        monkeypatch, convert_calls=convert_calls, transcription="# Note\n\nReadable content."
    )

    await pipeline.parse_document(doc_id)
    await pipeline.parse_document(doc_id)

    # The PIL conversion runs ONCE; the cached converted.pdf is reused on re-parse so the
    # PDF bytes (hence the content-addressed VLM cache key) stay byte-stable across runs.
    assert len(convert_calls) == 1


@pytest.mark.asyncio
async def test_blank_image_transcribes_nothing_and_refuses(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = settings.vault_path
    doc_id = "1mg00003-blank"
    _seed_png(vault, doc_id)
    convert_calls: list[Path] = []
    # A blank/unreadable image → the VLM transcribes only whitespace → no parts → the
    # structural HARD-gate path: ParseConfidenceTooLow, no junk 0-chunk doc written.
    _install_fakes(monkeypatch, convert_calls=convert_calls, transcription="   \n\n  ")

    with pytest.raises(ParseConfidenceTooLow):
        await pipeline.parse_document(doc_id)

    assert not (vault / "documents" / f"{doc_id}.md").is_file()
