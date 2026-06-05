"""Citation-grade page-boundary marker scaffold (companion arc-3 foundation).

The marker rides the body transforms (`_finalize_body` at parse; `reattach_chart_extractions` +
`linearize_gfm_tables` at index) as a ruler, then is stripped. These pin the LOAD-BEARING golden
invariant — the stripped body is BYTE-IDENTICAL to the same body built without markers, so a re-index
that derives `Chunk.page` from the measured boundaries never churns content-addressed chunk_ids — plus
the helper offset math. The wiring that consumes these (parse records each page's `char_start`;
`index_document::_exact_page_intervals` maps them through the transforms) is covered by the parse +
index integration tests; the index-side marker placement golden is pinned here too.
"""

from __future__ import annotations

from memex.core.manifest import ChartExtraction
from memex.core.table_linearize import linearize_gfm_tables
from memex.core.text import (
    insert_page_markers_at,
    is_page_boundary_marker,
    mark_pages_for_measure,
    measure_and_strip_page_markers,
    reattach_chart_extractions,
)
from memex.parse.pipeline import _finalize_body

_P1 = "# Intro\n\nSome text on slide one.\n\n## Recap"
_P2 = "## Recap\n\nMore text on slide two.\n\n| A | B |\n|---|---|\n| 1 | 2 |"  # cross-page dup "## Recap"
_P3 = "### Deep\n\nFinal slide content here."


def _page_for_offset(intervals: list[tuple[int, int, int]], offset: int) -> int | None:
    for page_no, start, end in intervals:
        if start <= offset < end:
            return page_no
    return intervals[-1][0] if intervals else None


def test_is_page_boundary_marker() -> None:
    assert is_page_boundary_marker("<!--MEMEX_PAGE_BOUNDARY:3-->")
    assert is_page_boundary_marker("  <!--MEMEX_PAGE_BOUNDARY:0-->  ")
    assert not is_page_boundary_marker("<!-- image -->")
    assert not is_page_boundary_marker("# A heading")
    assert not is_page_boundary_marker("")


def test_round_trip_on_plain_body_is_identity_with_correct_intervals() -> None:
    per_page = [(1, _P1), (2, _P2), (3, _P3)]
    plain = "\n\n".join(seg for _, seg in per_page)
    clean, intervals = measure_and_strip_page_markers(mark_pages_for_measure(per_page))
    assert clean == plain  # the strip exactly reverses the marking
    assert [p for p, _, _ in intervals] == [1, 2, 3]
    # Each page's content sits inside its interval.
    assert clean[intervals[0][1] : intervals[0][2]].startswith("# Intro")
    assert clean[intervals[1][1] : intervals[1][2]].startswith("## Recap")
    assert clean[intervals[2][1] : intervals[2][2]].startswith("### Deep")
    # A chunk offset lands on the right page.
    assert _page_for_offset(intervals, plain.index("Final slide")) == 3
    assert _page_for_offset(intervals, plain.index("More text")) == 2


def test_golden_through_finalize_body_collapses_cross_page_duplicate() -> None:
    # _finalize_body collapses the duplicate "## Recap" across the page seam; WITH the collapse
    # exclusion the marker round-trip reproduces that EXACTLY (byte-identical → chunk_id-stable).
    per_page = [(1, _P1), (2, _P2), (3, _P3)]
    plain = _finalize_body("\n\n".join(seg for _, seg in per_page))
    assert plain.count("Recap") == 1  # the cross-page duplicate was collapsed
    marked_final = _finalize_body(mark_pages_for_measure(per_page))
    clean, intervals = measure_and_strip_page_markers(marked_final)
    assert clean == plain  # GOLDEN — the marker did not perturb the transform output
    assert [p for p, _, _ in intervals] == [1, 2, 3]


def test_golden_through_index_transforms() -> None:
    # The index transforms (chart re-attach + table linearization) shift offsets the most on
    # figure-heavy decks; the marker rides them inertly and strips back byte-identical.
    p1 = "# Slide 1\n\n<!-- image -->\n\nIntro bullet."
    p2 = "# Slide 2\n\n| Col | Val |\n|---|---|\n| x | 9 |"
    charts = [ChartExtraction(placeholder_index=0, markdown="**Chart** | On 22 | Late 8")]

    def _index(body: str) -> str:
        return linearize_gfm_tables(reattach_chart_extractions(body, charts))

    plain = _index("\n\n".join([p1, p2]))
    clean, intervals = measure_and_strip_page_markers(_index(mark_pages_for_measure([(1, p1), (2, p2)])))
    assert clean == plain  # GOLDEN through both index transforms
    # page 2 begins at "# Slide 2" in the post-transform body (after the inserted chart block).
    assert clean[intervals[1][1] :].startswith("# Slide 2")


def test_insert_page_markers_at_round_trips() -> None:
    # The INDEX-side placement: insert markers at known boundaries in a clean body, then measure+strip
    # → identity, and the boundaries are recovered.
    body = "# Slide 1\n\nbody one\n\n# Slide 2\n\nbody two"
    s2 = body.index("# Slide 2")
    marked = insert_page_markers_at(body, [(1, 0), (2, s2)])
    clean, intervals = measure_and_strip_page_markers(marked)
    assert clean == body
    assert intervals[0][:2] == (1, 0)
    assert clean[intervals[1][1] :].startswith("# Slide 2")


def test_insert_page_markers_at_golden_through_index_transforms() -> None:
    # The EXACT index path (`index_document::_exact_page_intervals`): insert markers at the
    # parse-recorded boundaries IN THE ON-DISK BODY, run the index transforms on the marked body,
    # strip → byte-identical to transforming the unmarked body (so chunk_ids stay stable), and the
    # recovered intervals attribute pages correctly DESPITE the chart block shifting page-2's start.
    p1 = "# Slide 1\n\n<!-- image -->\n\nIntro bullet."
    p2 = "# Slide 2\n\n| Col | Val |\n|---|---|\n| x | 9 |"
    charts = [ChartExtraction(placeholder_index=0, markdown="**Chart** | On 22 | Late 8")]

    def _index(body: str) -> str:
        return linearize_gfm_tables(reattach_chart_extractions(body, charts))

    # `plain_body` is the on-disk `.md` (content-only, markers absent); boundaries are page char_starts.
    plain_body = "\n\n".join([p1, p2])
    boundaries = [(1, 0), (2, plain_body.index("# Slide 2"))]
    indexed_plain = _index(plain_body)

    marked = insert_page_markers_at(plain_body, boundaries)
    clean, intervals = measure_and_strip_page_markers(_index(marked))
    assert clean == indexed_plain  # GOLDEN — the index-side marker rode reattach+linearize inertly
    assert [p for p, _, _ in intervals] == [1, 2]
    # Page 2's start moved forward by the re-attached chart block, yet the interval tracks it.
    assert clean[intervals[1][1] :].startswith("# Slide 2")
    assert _page_for_offset(intervals, indexed_plain.index("Intro bullet")) == 1
    assert _page_for_offset(intervals, indexed_plain.index("# Slide 2")) == 2


def test_empty_page_yields_zero_width_interval() -> None:
    # Consecutive markers (a page with no content between) → a zero-width interval (the chunker
    # attributes no chunk to an empty page; matches the legacy char_count=0 behaviour).
    marked = "<!--MEMEX_PAGE_BOUNDARY:1-->\n\n<!--MEMEX_PAGE_BOUNDARY:2-->\n\nreal content"
    clean, intervals = measure_and_strip_page_markers(marked)
    assert clean == "real content"
    assert intervals[0] == (1, 0, 0)  # empty page 1: zero width
    assert intervals[1] == (2, 0, len("real content"))
