"""Unit tests for the pymupdf4llm heading-level recovery in the worker.

pymupdf4llm (1.27.x) detects heading lines but collapses every level to
`## `. `_heading_size_to_level` + `_remap_heading_levels` re-derive the
real level from font size — touching only already-detected heading lines,
never adding/removing headings, and skipping fenced code.
"""

from __future__ import annotations

from typing import Any

from memex.parse.pymupdf_worker import (
    _heading_size_to_level,
    _remap_heading_levels,
)


class _FakePage:
    """Minimal stand-in: `get_text('dict')` returns one block whose lines
    each hold a single span `(text, size)`."""

    def __init__(self, spans: list[tuple[str, float]]) -> None:
        self._spans = spans

    def get_text(self, _kind: str) -> dict[str, Any]:
        return {
            "blocks": [{"lines": [{"spans": [{"text": t, "size": s}]} for t, s in self._spans]}]
        }


def test_size_to_level_ranks_above_body() -> None:
    # body=10 (most frequent); heading tiers 24>18>14 → levels 1,2,3.
    doc = [
        _FakePage(
            [("body", 10.0)] * 5
            + [("Title", 24.0), ("Sec A", 18.0), ("Sec B", 18.0), ("Sub", 14.0)]
        )
    ]
    assert _heading_size_to_level(doc) == {24: 1, 18: 2, 14: 3}


def test_size_to_level_empty_doc() -> None:
    assert _heading_size_to_level([_FakePage([])]) == {}


def test_remap_fixes_collapsed_levels() -> None:
    page = _FakePage(
        [("body text", 10.0), ("Title", 24.0), ("Section", 18.0), ("Subsection", 14.0)]
    )
    size_to_level = {24: 1, 18: 2, 14: 3}
    # pymupdf4llm collapsed all three to `## `.
    md = "## **Title**\n\nbody text\n\n## **Section**\n\n## **Subsection**\n"
    out = _remap_heading_levels(md, page, size_to_level)
    assert "# **Title**" in out and "## **Title**" not in out  # → H1
    assert "## **Section**" in out  # stays H2
    assert "### **Subsection**" in out  # → H3


def test_remap_skips_code_fences() -> None:
    page = _FakePage([("Real", 24.0), ("# not a heading", 10.0)])
    size_to_level = {24: 1}
    md = "## **Real**\n\n```\n# not a heading\n```\n"
    out = _remap_heading_levels(md, page, size_to_level)
    assert "# **Real**" in out and "## **Real**" not in out
    # the `# not a heading` inside the fence is untouched
    assert "```\n# not a heading\n```" in out


def test_remap_leaves_unmatched_heading_alone() -> None:
    # A heading whose text matches no span (e.g. a spurious table-header
    # heading) keeps pymupdf4llm's emitted level.
    page = _FakePage([("Body", 10.0)])
    out = _remap_heading_levels("## **Mystery Heading**\n", page, {24: 1})
    assert out == "## **Mystery Heading**\n"


def test_remap_noop_without_tiers() -> None:
    page = _FakePage([("Anything", 24.0)])
    md = "## **Anything**\n"
    assert _remap_heading_levels(md, page, {}) == md
