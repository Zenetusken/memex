"""Office sources (pptx/docx/…) convert to a cached PDF and route through the
full PDF pipeline, so their figures + diagrams flow through the VLM/chart-OCR
passes (pypdfium2, which both renderers use, is PDF-only). The real LibreOffice
conversion + Docling pass are validated on the rig; here the converter and
`_parse_pdf` are faked to pin the routing + the converted-PDF caching.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from memex.core.config import MemexSettings, set_settings
from memex.parse import pipeline
from memex.parse.pipeline import ParseResult, parse_document


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemexSettings]:
    (tmp_path / "documents").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    s = MemexSettings()  # type: ignore[call-arg]
    set_settings(s)
    yield s
    set_settings(None)


def _seed_pptx(vault: Path, doc_id: str) -> Path:
    doc_dir = vault / "documents" / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    src = doc_dir / "source.pptx"
    src.write_bytes(b"PK\x03\x04 fake pptx bytes")  # zip magic; content unused (convert faked)
    return src


@pytest.mark.asyncio
async def test_office_source_converts_once_and_routes_through_pdf_pipeline(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = settings.vault_path
    doc_id = "abc123-deck"
    _seed_pptx(vault, doc_id)

    convert_calls: list[Path] = []

    async def fake_convert(source: Path, out_dir: Path, *, timeout_s: float = 180.0) -> Path:
        convert_calls.append(source)
        pdf = out_dir / f"{source.stem}.pdf"
        pdf.write_bytes(b"%PDF-1.7 fake converted")
        return pdf

    parse_calls: list[tuple[Path, bool]] = []

    async def fake_parse_pdf(
        vault_path: Path,
        did: str,
        source: Path,
        *,
        force_docling: bool = False,
        refresh_vlm: bool = False,
    ) -> ParseResult:
        parse_calls.append((source, force_docling))
        return ParseResult(
            doc_id=did, correlation_id="t", engine="docling", pages=[], markdown_bytes=0
        )

    monkeypatch.setattr(pipeline, "convert_to_pdf", fake_convert)
    monkeypatch.setattr(pipeline, "_parse_pdf", fake_parse_pdf)

    converted = vault / "documents" / doc_id / "converted.pdf"

    # First parse: convert pptx -> cached converted.pdf, route to the PDF
    # pipeline with force_docling (so the deck reaches the VLM/chart-OCR pass).
    await parse_document(doc_id)
    assert convert_calls == [vault / "documents" / doc_id / "source.pptx"]
    assert converted.is_file()
    assert parse_calls[-1] == (converted, True)
    # The original Office source is preserved (provenance) and is NOT named
    # source.pdf, so source resolution still finds the pptx.
    assert (vault / "documents" / doc_id / "source.pptx").is_file()

    # Second parse: reuse the cached converted.pdf — no re-conversion (keeps the
    # PDF bytes, hence the VLM/chart-OCR cache keys, byte-stable across parses).
    await parse_document(doc_id)
    assert len(convert_calls) == 1  # not converted again
    assert parse_calls[-1] == (converted, True)
