"""P4.1 write-side wiring tests — `insert_wikilinks` emitting
`[[doc#section]]` anchors when the citation's context references a
target doc's heading.

The read-side primitives are in `tests/unit/test_wikilinks.py`. This
file covers the enrichment-stage write path: heading discovery via
`make_signature(body=...)` + section-anchor emission in
`insert_wikilinks(target_index=...)`.
"""

from __future__ import annotations

from datetime import date

import pytest

from memex.enrich.citations import (
    CitationIndex,
    ResolvedCitation,
    insert_wikilinks,
    make_signature,
)
from memex.vault.store import Frontmatter


def _frontmatter(title: str = "Sample Doc") -> Frontmatter:
    return Frontmatter(title=title, authors=["Alice Smith"], date=date(2024, 1, 1))


def _resolved(
    surface: str,
    target: str,
    *,
    confidence: float = 0.95,
) -> ResolvedCitation:
    return ResolvedCitation(
        surface_text=surface,
        target_doc_id=target,
        target_title=target,
        confidence=confidence,
        chunk_id="src-chunk-0",
        via="title",
    )


# ----------------------------------------------------------------------
# make_signature heading population
# ----------------------------------------------------------------------


def test_make_signature_extracts_headings_from_body() -> None:
    """`make_signature(body=...)` populates `headings` with the
    document's Markdown headings in order."""
    body = (
        "# Introduction\n\nIntro prose.\n\n"
        "## Methods\n\nMethods prose.\n\n"
        "## Results\n\nResults prose.\n"
    )
    sig = make_signature("doc-1", _frontmatter(), body)
    assert sig.headings == ["Introduction", "Methods", "Results"]


def test_make_signature_no_body_returns_empty_headings() -> None:
    """Back-compat: pre-P4.1 callers that don't pass `body` get an
    empty headings list (and `insert_wikilinks` falls through to
    plain `[[doc]]` links)."""
    sig = make_signature("doc-1", _frontmatter())
    assert sig.headings == []


def test_make_signature_skips_chart_block_h1s() -> None:
    """Inert `# H1` labels inside `[chart-extracted]` blocks (Nemotron-
    Parse chart-figure labels) must NOT pollute the headings list —
    same P3.3 v7 defense as the chunker uses."""
    body = (
        "## Real Section\n\n"
        "prose.\n\n"
        "[chart-extracted]\n"
        "# Inert Chart Label\n"
        "[/chart-extracted]\n\n"
        "## Another Real Section\n"
    )
    sig = make_signature("doc-1", _frontmatter(), body)
    assert "Inert Chart Label" not in sig.headings
    assert "Real Section" in sig.headings
    assert "Another Real Section" in sig.headings


# ----------------------------------------------------------------------
# insert_wikilinks section-anchor emission
# ----------------------------------------------------------------------


def _index_with_headings(target_id: str, headings: list[str]) -> CitationIndex:
    """Build a one-target CitationIndex with the given headings list."""
    idx = CitationIndex()
    # We could call make_signature with a fake body, but tests are
    # cleaner if we just set the field directly.
    sig = make_signature(target_id, _frontmatter())
    sig.headings = headings
    idx.by_id[target_id] = sig
    return idx


def test_insert_wikilinks_no_target_index_emits_plain_link() -> None:
    """Back-compat: when `target_index=None`, no section anchors are
    emitted — output matches the pre-P4.1 behaviour."""
    body = "See Smith 2024 for the Methods discussion."
    resolved = [_resolved("Smith 2024", "smith-2024")]
    new_body, count = insert_wikilinks(body, resolved)
    assert count == 1
    assert "[[smith-2024]]" in new_body
    assert "[[smith-2024#" not in new_body


def test_insert_wikilinks_emits_section_anchor_on_context_match() -> None:
    """When the target's headings include "Methods" AND the citation's
    context mentions "Methods", emit `[[doc#Methods]]`."""
    body = "See Smith 2024 for the Methods discussion in their paper."
    resolved = [_resolved("Smith 2024", "smith-2024")]
    idx = _index_with_headings("smith-2024", ["Methods", "Results"])
    new_body, count = insert_wikilinks(body, resolved, target_index=idx)
    assert count == 1
    assert "[[smith-2024#Methods]]" in new_body


def test_insert_wikilinks_no_anchor_when_no_heading_in_context() -> None:
    """When the citation's context mentions NONE of the target's
    headings, fall through to plain `[[doc]]`."""
    body = "Smith 2024 published an interesting paper on reflexivity."
    resolved = [_resolved("Smith 2024", "smith-2024")]
    idx = _index_with_headings("smith-2024", ["Methods", "Results"])
    new_body, count = insert_wikilinks(body, resolved, target_index=idx)
    assert count == 1
    assert "[[smith-2024]]" in new_body
    assert "[[smith-2024#" not in new_body


def test_insert_wikilinks_prefers_longest_matching_heading() -> None:
    """When multiple headings appear in context, pick the LONGEST one
    (most specific). Doc with headings ["Methods", "Methods: Data
    Movement"] + context "the Methods: Data Movement section" emits
    the longer anchor."""
    body = "Smith 2024's Methods: Data Movement section is foundational."
    resolved = [_resolved("Smith 2024", "smith-2024")]
    idx = _index_with_headings(
        "smith-2024", ["Methods", "Methods: Data Movement", "Results"]
    )
    new_body, count = insert_wikilinks(body, resolved, target_index=idx)
    assert count == 1
    assert "[[smith-2024#Methods: Data Movement]]" in new_body


def test_insert_wikilinks_skips_short_generic_headings() -> None:
    """Headings shorter than 4 chars (e.g. "Q1", "FY") are skipped —
    too generic, too likely to false-match. `min_length=4`."""
    body = "Smith 2024 mentions Q1 metrics that align with the data."
    resolved = [_resolved("Smith 2024", "smith-2024")]
    idx = _index_with_headings("smith-2024", ["Q1", "Quarterly Report"])
    new_body, count = insert_wikilinks(body, resolved, target_index=idx)
    assert count == 1
    # "Q1" was in context but is < 4 chars → skipped. No anchor.
    assert "[[smith-2024]]" in new_body
    assert "[[smith-2024#Q1]]" not in new_body


def test_insert_wikilinks_section_match_case_insensitive() -> None:
    """Heading "Methods" matches context "methods" (lowercase) — the
    matcher is case-insensitive."""
    body = "Smith 2024 covers methods at length in section 3."
    resolved = [_resolved("Smith 2024", "smith-2024")]
    idx = _index_with_headings("smith-2024", ["Methods"])
    new_body, count = insert_wikilinks(body, resolved, target_index=idx)
    assert count == 1
    # Emitted heading preserves its ORIGINAL casing from the headings list
    assert "[[smith-2024#Methods]]" in new_body


def test_insert_wikilinks_no_heading_match_outside_context_window() -> None:
    """The context window is ±300 chars around the surface_text. A
    heading that appears in the document but FAR from the citation
    doesn't trigger an anchor."""
    # Citation at the start; heading "Methods" appears at the very end
    # of a 1000-char body — outside the 300-char window.
    far_padding = "x" * 1000
    body = f"Smith 2024 is great.{far_padding}Methods are discussed."
    resolved = [_resolved("Smith 2024", "smith-2024")]
    idx = _index_with_headings("smith-2024", ["Methods"])
    new_body, count = insert_wikilinks(body, resolved, target_index=idx)
    assert count == 1
    # Far heading doesn't trigger anchor
    assert "[[smith-2024]]" in new_body
    assert "[[smith-2024#Methods]]" not in new_body


def test_insert_wikilinks_target_with_empty_headings_falls_through() -> None:
    """If the target's headings list is empty (e.g., doc has no
    Markdown headings, or `make_signature` was called without
    `body`), emit plain `[[doc]]`. No section-anchor attempt."""
    body = "Smith 2024 covers Methods at length."
    resolved = [_resolved("Smith 2024", "smith-2024")]
    idx = _index_with_headings("smith-2024", [])
    new_body, count = insert_wikilinks(body, resolved, target_index=idx)
    assert count == 1
    assert "[[smith-2024]]" in new_body


def test_insert_wikilinks_target_not_in_index_falls_through() -> None:
    """If the target doc isn't in `target_index` (shouldn't happen in
    production but defensive), emit plain `[[doc]]`."""
    body = "Smith 2024 covers Methods at length."
    resolved = [_resolved("Smith 2024", "smith-2024")]
    idx = _index_with_headings("other-doc", ["Methods"])  # different doc
    new_body, count = insert_wikilinks(body, resolved, target_index=idx)
    assert count == 1
    assert "[[smith-2024]]" in new_body
