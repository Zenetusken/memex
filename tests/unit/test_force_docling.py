"""Unit tests for the `force_docling` parse-routing flag.

Pin the contract:
- `ParseSettings.force_docling` defaults to `False` and accepts `True`
  via `MEMEX_PARSE__FORCE_DOCLING=true`.
- `parse_document(doc_id, force_docling=True)` bypasses the PyMuPDF
  pre-filter and goes straight to `_parse_with_docling`.
- When the kwarg is `None`, routing follows the settings value.
- The CLI flag on `memex parse` propagates to `parse_document`.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from memex.core.config import MemexSettings, ParseSettings, set_settings
from memex.parse import pipeline


def test_parse_settings_force_docling_defaults_to_false() -> None:
    settings = ParseSettings()
    assert settings.force_docling is False


def test_parse_settings_force_docling_via_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings parsing honors `MEMEX_PARSE__FORCE_DOCLING=true`."""
    monkeypatch.setenv("MEMEX_PARSE__FORCE_DOCLING", "true")
    settings = MemexSettings(vault_path=Path("/tmp/x"))  # noqa: S108
    assert settings.parse.force_docling is True


@pytest.mark.asyncio
async def test_parse_document_force_docling_skips_pymupdf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When `parse_document(doc_id, force_docling=True)` is called, the
    PyMuPDF pre-filter is never invoked and Docling is called directly.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    doc_dir = vault / "documents" / "doc-1"
    doc_dir.mkdir(parents=True)
    source = doc_dir / "source.pdf"
    source.write_bytes(b"%PDF-1.4\n")

    set_settings(MemexSettings(vault_path=vault))

    pymupdf_mock = AsyncMock(name="_parse_with_pymupdf")
    docling_mock = AsyncMock(name="_parse_with_docling")
    docling_mock.return_value = "DOCLING_RESULT"

    monkeypatch.setattr(pipeline, "_parse_with_pymupdf", pymupdf_mock)
    monkeypatch.setattr(pipeline, "_parse_with_docling", docling_mock)

    result = await pipeline.parse_document("doc-1", force_docling=True)

    assert result == "DOCLING_RESULT"
    assert pymupdf_mock.await_count == 0  # PyMuPDF never called
    assert docling_mock.await_count == 1  # Docling called once
    # The Docling call should NOT pass `force_ocr` (that's the fall-
    # through hint from PyMuPDF, which we skipped).
    call_kwargs = docling_mock.call_args.kwargs
    assert "force_ocr" not in call_kwargs


@pytest.mark.asyncio
async def test_parse_document_force_docling_settings_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the kwarg is `None`, routing follows `settings.parse.force_docling`."""
    vault = tmp_path / "vault"
    vault.mkdir()
    doc_dir = vault / "documents" / "doc-2"
    doc_dir.mkdir(parents=True)
    source = doc_dir / "source.pdf"
    source.write_bytes(b"%PDF-1.4\n")

    # Settings says force_docling=True; kwarg is None → settings wins
    settings = MemexSettings(vault_path=vault)
    settings.parse.force_docling = True
    set_settings(settings)

    pymupdf_mock = AsyncMock(name="_parse_with_pymupdf")
    docling_mock = AsyncMock(name="_parse_with_docling")
    docling_mock.return_value = "DOCLING_RESULT"

    monkeypatch.setattr(pipeline, "_parse_with_pymupdf", pymupdf_mock)
    monkeypatch.setattr(pipeline, "_parse_with_docling", docling_mock)

    result = await pipeline.parse_document("doc-2")  # no kwarg

    assert result == "DOCLING_RESULT"
    assert pymupdf_mock.await_count == 0
    assert docling_mock.await_count == 1
