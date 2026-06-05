"""Companion arc-3 wiring: the parse + index round-trip helpers that record and consume each
page's citation-grade `char_start`.

Both helpers are PURE and SAFE BY CONSTRUCTION — they record/return page offsets ONLY when a
page-boundary marker round-trip reproduces the canonical body byte-for-byte, else they fall back
to the nav-grade path (an empty map / `None`). A wiring bug can therefore only lose page precision,
never churn the content-addressed chunk_ids. These pin both the happy path and the guard fallback.
"""

from __future__ import annotations

from memex.core.manifest import ChartExtraction
from memex.core.table_linearize import linearize_gfm_tables
from memex.core.text import reattach_chart_extractions
from memex.index.pipeline import _exact_page_intervals
from memex.parse.pipeline import _finalize_body, _finalize_body_with_page_starts

# ----- parse side: _finalize_body_with_page_starts -----------------------------------------------


def test_finalize_with_page_starts_records_offsets() -> None:
    per_page = [(1, "# A\n\nalpha"), (2, "# B\n\nbeta")]
    plain_md = "\n\n".join(seg for _, seg in per_page)
    body, starts = _finalize_body_with_page_starts(plain_md, per_page)
    # The body is ALWAYS the canonical finalized body — byte-identical to today's parse output.
    assert body == _finalize_body(plain_md)
    assert starts[1] == 0  # page 1 starts at the top
    assert body[starts[2] :].startswith("# B")  # page 2's recorded offset lands on its content


def test_finalize_with_page_starts_survives_table_linearization() -> None:
    # `_finalize_body` linearizes GFM tables (appends a `[table-rows]` block) — the marker rides it.
    p1 = "# One\n\nlead text"
    p2 = "# Two\n\n| Col | Val |\n|---|---|\n| x | 9 |"
    plain_md = "\n\n".join([p1, p2])
    body, starts = _finalize_body_with_page_starts(plain_md, [(1, p1), (2, p2)])
    assert body == _finalize_body(plain_md)
    assert body[starts[2] :].startswith("# Two")


def test_finalize_with_page_starts_guard_falls_back_on_mismatch() -> None:
    # If `per_page` does NOT reconstruct `plain_markdown`, the round-trip diverges → empty map
    # (nav-grade), but the body is STILL the canonical finalized `plain_markdown`.
    body, starts = _finalize_body_with_page_starts("# Real\n\ncontent", [(1, "# Different")])
    assert body == _finalize_body("# Real\n\ncontent")
    assert starts == {}


def test_finalize_with_page_starts_empty_per_page() -> None:
    body, starts = _finalize_body_with_page_starts("# Solo\n\ntext", [])
    assert body == _finalize_body("# Solo\n\ntext")
    assert starts == {}


# ----- index side: _exact_page_intervals ---------------------------------------------------------

_CHARTS = [ChartExtraction(placeholder_index=0, markdown="**Chart** | On 22 | Late 8")]
_P1 = "# Slide 1\n\n<!-- image -->\n\nIntro bullet."
_P2 = "# Slide 2\n\nbody two"
_PLAIN_BODY = "\n\n".join([_P1, _P2])  # the on-disk content-only `.md`
_INDEXED = linearize_gfm_tables(reattach_chart_extractions(_PLAIN_BODY, _CHARTS))


def test_exact_page_intervals_maps_through_transforms() -> None:
    boundaries = [(1, 0), (2, _PLAIN_BODY.index("# Slide 2"))]
    intervals = _exact_page_intervals(_PLAIN_BODY, _INDEXED, boundaries, _CHARTS, doc_id="d")
    assert intervals is not None
    assert [p for p, _, _ in intervals] == [1, 2]
    # The re-attached chart block shifts page 2 forward in the indexed body, yet the interval tracks it.
    assert _INDEXED[intervals[1][1] :].startswith("# Slide 2")
    assert intervals[0][1] <= _INDEXED.index("Intro bullet") < intervals[0][2]


def test_exact_page_intervals_guard_falls_back_on_bad_offset() -> None:
    # A boundary that does NOT sit on a clean block start corrupts the strip → byte mismatch → None.
    bad = [(1, 0), (2, 3)]  # offset 3 is mid-"# Slide 1"
    assert _exact_page_intervals(_PLAIN_BODY, _INDEXED, bad, _CHARTS, doc_id="d") is None


def test_exact_page_intervals_empty_boundaries() -> None:
    assert _exact_page_intervals(_PLAIN_BODY, _INDEXED, [], _CHARTS, doc_id="d") is None
