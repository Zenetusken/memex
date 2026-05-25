"""EmbeddingGemma native-prompt helpers — pure-function unit tests.

Covers `index/embed_prompts.py` (document_input / chunk_title /
native_prompts_enabled / EMBED_QUERY_PROMPT_NAME) plus the
recipe-version helper that lives in `index/pipeline.py`.
"""

from __future__ import annotations

import pytest

from memex.core.types import Chunk
from memex.index.embed_prompts import (
    EMBED_QUERY_PROMPT_NAME,
    chunk_title,
    document_input,
    native_prompts_enabled,
)
from memex.index.pipeline import _embed_recipe_version


def _chunk(
    *,
    text: str = "body",
    document_title: str = "Doc Title",
    heading_path: list[str] | None = None,
) -> Chunk:
    return Chunk(
        chunk_id="abc123",
        document_id="doc1",
        document_title=document_title,
        text=text,
        heading_path=heading_path if heading_path is not None else [],
    )


def test_embed_query_prompt_name_constant() -> None:
    assert EMBED_QUERY_PROMPT_NAME == "query"


def test_document_input_format() -> None:
    assert document_input("My Heading", "the body") == "title: My Heading | text: the body"


def test_chunk_title_deepest_heading() -> None:
    chunk = _chunk(heading_path=["Top", "Middle", "Deepest"])
    assert chunk_title(chunk) == "Deepest"


def test_chunk_title_falls_back_to_document_title() -> None:
    chunk = _chunk(heading_path=[], document_title="The Document")
    assert chunk_title(chunk) == "The Document"


def test_chunk_title_empty_heading_path_uses_doc_title() -> None:
    # heading_path empty + doc_title truthy → doc_title
    chunk = _chunk(heading_path=[], document_title="Fallback Title")
    assert chunk_title(chunk) == "Fallback Title"


def test_chunk_title_none_sentinel() -> None:
    # No heading, no doc title → the EmbeddingGemma "none" sentinel.
    chunk = _chunk(heading_path=[], document_title="")
    assert chunk_title(chunk) == "none"


def test_native_prompts_enabled_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMEX_EMBED_NATIVE_PROMPTS", raising=False)
    assert native_prompts_enabled() is True


def test_native_prompts_enabled_explicit_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMEX_EMBED_NATIVE_PROMPTS", "1")
    assert native_prompts_enabled() is True


def test_native_prompts_disabled_via_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMEX_EMBED_NATIVE_PROMPTS", "0")
    assert native_prompts_enabled() is False


def test_embed_recipe_version_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMEX_EMBED_NATIVE_PROMPTS", raising=False)
    assert _embed_recipe_version() == "v1-gemma-prompts"


def test_embed_recipe_version_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMEX_EMBED_NATIVE_PROMPTS", "0")
    assert _embed_recipe_version() == "v0"
