"""The enrich prompts (v2) truncate the passage so an oversized chunk can't
blow the entity/citation context window and fail the whole chunk."""

from __future__ import annotations

from memex.prompts import render_prompt

_TRUNCATE = 6000  # the budget in extract_{entities,citations}/v2.md


def test_entity_prompt_truncates_oversized_passage() -> None:
    huge = "x" * 9000
    out = render_prompt("extract_entities", document_title="T", passage=huge)
    assert huge not in out  # not passed through whole
    assert "..." in out  # Jinja truncate ellipsis present
    # passage portion bounded near the budget (Jinja's word-boundary + leeway
    # handling lands within a small margin of the configured length)
    assert out.count("x") <= _TRUNCATE + 100


def test_citation_prompt_truncates_oversized_passage() -> None:
    huge = "y" * 9000
    out = render_prompt("extract_citations", document_title="T", passage=huge)
    assert huge not in out
    assert out.count("y") <= _TRUNCATE + 100


def test_short_passage_passes_through_untruncated() -> None:
    small = "A normal short passage about reflexivity and graphs."
    out = render_prompt("extract_entities", document_title="T", passage=small)
    assert small in out  # under budget → verbatim, no truncation


def test_loader_selects_v2() -> None:
    # The loader picks the highest on-disk version; v2 is active.
    out = render_prompt("extract_entities", document_title="T", passage="z")
    assert "extract_entities@v2" in out


# ----- Output-list bounds (prevent runaway → max_tokens truncation → fail) -----

import pytest  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from memex.enrich.citations import CitationCandidate, CitationList  # noqa: E402
from memex.enrich.entities import EntityList, ExtractedEntity  # noqa: E402


def _entity(i: int) -> ExtractedEntity:
    return ExtractedEntity(name=f"E{i}", kind="concept", confidence="high")


def test_entity_list_bounded_at_24() -> None:
    EntityList(entities=[_entity(i) for i in range(24)])  # ok
    with pytest.raises(ValidationError):
        EntityList(entities=[_entity(i) for i in range(25)])


def test_citation_list_bounded_at_24() -> None:
    mk = lambda i: CitationCandidate(surface_text=f"[{i}]", confidence="high")  # noqa: E731
    CitationList(citations=[mk(i) for i in range(24)])  # ok
    with pytest.raises(ValidationError):
        CitationList(citations=[mk(i) for i in range(25)])


def test_citation_surface_text_bounded() -> None:
    with pytest.raises(ValidationError):
        CitationCandidate(surface_text="x" * 201, confidence="low")
