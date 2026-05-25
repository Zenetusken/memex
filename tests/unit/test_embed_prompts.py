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
    EMBED_QUERY_PROMPT_TEXT,
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


# ----------------------------------------------------------------------
# FIX 5 — non-empty + sanitized title
# ----------------------------------------------------------------------


def test_chunk_title_empty_string_heading_uses_doc_title() -> None:
    """A malformed `heading_path == [""]` (from a double-space `##  ` heading)
    must NOT yield an empty title (`title:  | text:`) — it falls back to the
    document_title."""
    chunk = _chunk(heading_path=[""], document_title="Fallback Title")
    assert chunk_title(chunk) == "Fallback Title"


def test_chunk_title_whitespace_heading_uses_doc_title() -> None:
    chunk = _chunk(heading_path=["   "], document_title="Fallback Title")
    assert chunk_title(chunk) == "Fallback Title"


def test_chunk_title_picks_deepest_nonempty_entry() -> None:
    """The deepest NON-EMPTY entry wins, even when a deeper entry is empty."""
    chunk = _chunk(heading_path=["Top", "Middle", ""], document_title="Doc")
    assert chunk_title(chunk) == "Middle"


def test_chunk_title_all_empty_path_empty_doc_title_is_none() -> None:
    chunk = _chunk(heading_path=["", "  "], document_title="")
    assert chunk_title(chunk) == "none"


def test_chunk_title_sanitizes_delimiter() -> None:
    """A heading containing the `" | "` delimiter would corrupt the title/text
    split — it's replaced with `" / "`."""
    chunk = _chunk(heading_path=["Revenue | Costs"], document_title="Doc")
    title = chunk_title(chunk)
    assert " | " not in title
    assert title == "Revenue / Costs"
    # And the assembled input has exactly one ` | ` (the format delimiter).
    assert document_input(title, "body").count(" | ") == 1


def test_chunk_title_sanitizes_newlines() -> None:
    chunk = _chunk(heading_path=["Line one\nLine two"], document_title="Doc")
    title = chunk_title(chunk)
    assert "\n" not in title
    assert title == "Line one Line two"


def test_chunk_title_clamps_long_heading() -> None:
    long_heading = "A" * 500
    chunk = _chunk(heading_path=[long_heading], document_title="Doc")
    title = chunk_title(chunk)
    assert len(title) <= 80


def test_chunk_title_doc_title_also_sanitized() -> None:
    """The document_title fallback is sanitized too."""
    chunk = _chunk(heading_path=[], document_title="Big | Filing")
    assert chunk_title(chunk) == "Big / Filing"


def test_embed_query_prompt_text_constant() -> None:
    assert EMBED_QUERY_PROMPT_TEXT == "task: search result | query: "


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
