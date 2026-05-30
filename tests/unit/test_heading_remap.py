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
    _looks_like_prose_heading,
    _normalise_breaks,
    _remap_heading_levels,
    strip_repeating_page_furniture,
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


# --- <br> normalisation: table-aware (regression for the row-shatter bug) ---


def test_normalise_breaks_non_table_becomes_paragraphs() -> None:
    # The original behaviour: a <br>-joined run (chart/picture text) → \n\n.
    out = _normalise_breaks("Web Search<br>Paraphrase<br>Open QA")
    assert out == "Web Search\n\nParaphrase\n\nOpen QA"


def test_normalise_breaks_keeps_table_row_on_one_line() -> None:
    # pymupdf4llm emits in-cell wraps as <br> INSIDE a pipe row; converting
    # them to \n\n would shatter the GFM table. They must collapse to a space.
    row = "|Gateway|99.95%|1|Brief restart during a<br>deploy.|"
    assert _normalise_breaks(row) == "|Gateway|99.95%|1|Brief restart during a deploy.|"


def test_normalise_breaks_table_survives_intact() -> None:
    # The quarterly-uptime-report failure mode: a <br>-wrapped header + body.
    raw = (
        "|<br>Service|<br>Uptime|<br>Incidents|Notes|\n"
        "|---|---|---|---|\n"
        "|Gateway|99.95%|1|Brief restart during a<br>deploy.|\n"
    )
    out = _normalise_breaks(raw)
    # Every row stays on its own single line — no row shattered across lines.
    assert "\n\n" not in out
    assert len([ln for ln in out.splitlines() if ln.strip()]) == 3  # header, delimiter, body
    # Cells reconstruct correctly once stripped.
    assert [c.strip() for c in out.splitlines()[0].strip("|").split("|")] == [
        "Service",
        "Uptime",
        "Incidents",
        "Notes",
    ]


# --- demotion of misdetected headings (ported from the Docling worker) ------


def test_looks_like_prose_heading() -> None:
    assert _looks_like_prose_heading("This is a body sentence that ends here.")  # >=4 words + .
    assert _looks_like_prose_heading(
        "one two three four five six seven eight nine ten "
        "eleven twelve thirteen fourteen fifteen sixteen"
    )  # >15 words
    assert not _looks_like_prose_heading("Overview")  # one word
    assert not _looks_like_prose_heading("Service Metrics")  # no terminal punctuation
    assert not _looks_like_prose_heading("Item 1.")  # ends "." but < 4 words (short label)
    assert not _looks_like_prose_heading("**Recommendations**")  # emphasis stripped → one word


def test_remap_demotes_body_font_prose_heading() -> None:
    # pymupdf4llm over-flagged a body sentence as a heading; its font is body
    # size (11, not the 22 heading tier) AND it reads as prose → demote.
    page = _FakePage(
        [
            ("Real Title", 22.0),
            ("Visibility readings were taken at dawn from the gallery deck.", 11.0),
        ]
    )
    md = "# Real Title\n\n## Visibility readings were taken at dawn from the gallery deck.\n"
    out = _remap_heading_levels(md, page, {22: 1})
    assert out.split("\n")[0] == "# Real Title"  # real heading kept + re-levelled
    vis = next(ln for ln in out.split("\n") if "Visibility" in ln)
    assert not vis.startswith("#")  # demoted to a plain paragraph
    assert vis == "Visibility readings were taken at dawn from the gallery deck."


def test_remap_keeps_short_body_font_heading() -> None:
    # A body-font flagged heading that is NOT prose-like stays a heading
    # (conservative — a short label like "Summary" isn't a sentence).
    page = _FakePage([("Summary", 11.0)])
    out = _remap_heading_levels("## Summary\n", page, {22: 1})
    assert out.strip() == "## Summary"


def test_remap_keeps_prose_heading_without_span_match() -> None:
    # A prose-like heading whose text matches no span (font unknown) is left
    # exactly as emitted — we can't confirm it's body-size, so stay safe.
    page = _FakePage([("Totally different text", 22.0)])
    md = "## A sentence that has no matching span but ends with a period.\n"
    out = _remap_heading_levels(md, page, {22: 1})
    assert out.strip().startswith("## ")


# ======================================================================
# audit-10 step 2b (W8): repeating page-furniture strip
# ======================================================================


def _hdr_pages(n: int) -> list[str]:
    # n pages, each with a running header (boundary) + unique body
    return [f"NIST SP 800-207\n\nBody content of page {i} with real prose." for i in range(n)]


def test_strips_running_header_recurring_on_most_pages() -> None:
    pages, stripped = strip_repeating_page_furniture(_hdr_pages(10))
    assert stripped == 10
    assert all("NIST SP 800-207" not in p for p in pages)
    assert all(f"page {i}" in p for i, p in enumerate(pages))  # body untouched


def test_position_aware_keeps_legit_midpage_occurrence() -> None:
    """A recurring header is stripped in the band, but the SAME text occurring MID-PAGE
    (outside the first/last band) is kept — the strip is position-aware."""
    # 7 non-blank lines: header(0) in the band; the citation at line 3 is OUTSIDE the band.
    pages = [
        f"RUNNING HEADER\n\nB{i}-1\n\nB{i}-2\n\nWe cite RUNNING HEADER mid-page.\n\nB{i}-4\n\nB{i}-5\n\nB{i}-6"
        for i in range(8)
    ]
    out, stripped = strip_repeating_page_furniture(pages)
    assert stripped == 8  # the band header on each page
    assert all(p.count("RUNNING HEADER") == 1 and "We cite RUNNING HEADER mid-page." in p for p in out)


def test_strips_multiline_running_header_band() -> None:
    """A TWO-line running header (e.g. NIST's "NIST SP 800-207" / "ZERO TRUST ARCHITECTURE")
    is caught by the band, not just the single first line."""
    pages = [f"NIST SP 800-207\n\nZERO TRUST ARCHITECTURE\n\nUnique body {i}." for i in range(10)]
    out, stripped = strip_repeating_page_furniture(pages)
    assert all("NIST SP 800-207" not in p and "ZERO TRUST ARCHITECTURE" not in p for p in out)
    assert all(f"Unique body {i}." in out[i] for i in range(10))  # body kept


def test_strips_bare_page_numbers_even_when_each_differs() -> None:
    pages = [f"Intro paragraph for page {i}.\n\n{i + 1}" for i in range(6)]  # footer = page number
    out, stripped = strip_repeating_page_furniture(pages)
    assert stripped == 6
    assert all(not p.rstrip().endswith(("1", "2", "3", "4", "5", "6")) or "page" in p for p in out)


def test_headings_and_tables_at_boundary_are_never_furniture() -> None:
    # structural boundary lines (heading / table) are never furniture even if they recur;
    # bodies are unique per page so only the structural-prefix guard is under test.
    pages = [f"## Section {i}\n\ntext {i}" for i in range(8)] + [
        f"| col |\n|---|\n| v{i} |" for i in range(8)
    ]
    out, stripped = strip_repeating_page_furniture(pages)
    assert stripped == 0
    assert out == pages


def test_short_doc_returned_unchanged() -> None:
    pages = ["Header\n\nbody", "Header\n\nbody"]  # < min_repeat
    assert strip_repeating_page_furniture(pages) == (pages, 0)


def test_rare_repeat_below_threshold_kept() -> None:
    # a boundary line on only 2 of 10 pages is NOT furniture (< max(3, 0.5*10)=5)
    pages = ["Shared top\n\nbody"] * 2 + [f"Unique {i}\n\nbody {i}" for i in range(8)]
    out, stripped = strip_repeating_page_furniture(pages)
    assert stripped == 0 and out == pages
