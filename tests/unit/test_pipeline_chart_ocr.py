"""P3.3 Session 4 unit tests for the pipeline's chart-OCR integration.

Focused tests for the helpers introduced in `parse/pipeline.py`:
- `_stitch_chart_extractions` — inject `[chart-extracted]` blocks at
  each `<!-- image -->` placeholder.
- The stitch's count-mismatch guard (logs and skips).

The full pipeline `_parse_with_docling` + vLLM pause/resume is
exercised by the integration test that runs the real chart-OCR
backend in Session 5; here we keep it light and fast.
"""

from __future__ import annotations

from typing import Literal

from memex.core.manifest import PageDecision
from memex.parse.chart_ocr_backend import (
    ChartOCROutput,
    PDFFigureRenderError,
)
from memex.parse.docling_backend import (
    DoclingConversion,
    DoclingPageOutput,
    FigureMetadata,
)
from memex.parse.pipeline import _figures_for_chart_ocr, _stitch_chart_extractions


def _conv(markdown: str) -> DoclingConversion:
    return DoclingConversion(
        markdown=markdown,
        pages=[DoclingPageOutput(page=1, markdown=markdown, confidence=0.95)],
        docling_version="fake-stitch",
        figure_count=0,
        table_count=0,
        equation_count=0,
    )


def test_stitch_replaces_each_image_placeholder_with_chart_block() -> None:
    """Two `<!-- image -->` placeholders + two ChartOCROutput results
    → both placeholders get a `[chart-extracted]` block appended. The
    placeholders themselves stay in the text (they're documentation
    of where the image was); the chart data follows immediately.
    """
    md = "# Doc\n\nFirst chart:\n\n<!-- image -->\n\nSecond chart:\n\n<!-- image -->\n"
    conversion = _conv(md)
    extractions: list[ChartOCROutput | Exception] = [
        ChartOCROutput(
            page_no=1,
            bbox=(0.0, 0.0, 100.0, 100.0),
            markdown="| year | density |\n|---|---|\n| 2020 | 100 |",
        ),
        ChartOCROutput(
            page_no=2,
            bbox=(50.0, 60.0, 200.0, 150.0),
            markdown="| flag | value |\n|---|---|\n| FP16 | 0.5x |",
        ),
    ]

    stitched = _stitch_chart_extractions(conversion, extractions)

    assert "<!-- image -->" in stitched.markdown
    assert "[chart-extracted]" in stitched.markdown
    assert "[/chart-extracted]" in stitched.markdown
    # Both extractions made it in, in order.
    assert "year | density" in stitched.markdown
    assert "flag | value" in stitched.markdown
    # First chart-extracted block appears BEFORE the second.
    first = stitched.markdown.index("year | density")
    second = stitched.markdown.index("flag | value")
    assert first < second
    # Count of blocks matches count of placeholders.
    assert stitched.markdown.count("[chart-extracted]") == 2
    assert stitched.markdown.count("[/chart-extracted]") == 2


def test_stitch_skips_extraction_errors() -> None:
    """An Exception in the extractions list leaves that placeholder
    unchanged; sibling placeholders with successful extractions still
    get stitched. The count must match (one extraction per
    placeholder, even if the extraction was an error)."""
    md = "<!-- image -->\n\nMiddle text\n\n<!-- image -->\n"
    conversion = _conv(md)
    extractions: list[ChartOCROutput | Exception] = [
        ChartOCROutput(page_no=1, bbox=(0.0, 0.0, 100.0, 100.0), markdown="ok-data"),
        PDFFigureRenderError("degenerate bbox", context={}),
    ]

    stitched = _stitch_chart_extractions(conversion, extractions)

    # First placeholder gets the block; second doesn't.
    assert stitched.markdown.count("[chart-extracted]") == 1
    assert "ok-data" in stitched.markdown
    # Both placeholders still in the markdown.
    assert stitched.markdown.count("<!-- image -->") == 2


def test_stitch_skips_empty_extraction() -> None:
    """An extraction whose markdown is empty (or whitespace-only)
    leaves the placeholder unchanged. The model legitimately returns
    nothing on charts it can't parse — we don't want to inject empty
    chart-extracted blocks into the document.
    """
    md = "<!-- image -->\n"
    conversion = _conv(md)
    extractions: list[ChartOCROutput | Exception] = [
        ChartOCROutput(page_no=1, bbox=(0.0, 0.0, 100.0, 100.0), markdown="   \n  "),
    ]

    stitched = _stitch_chart_extractions(conversion, extractions)

    assert "[chart-extracted]" not in stitched.markdown
    assert "<!-- image -->" in stitched.markdown


def test_stitch_count_mismatch_returns_unchanged() -> None:
    """When placeholder count != extraction count, the stitch refuses
    to align (alignment isn't reliable) and returns the conversion
    unchanged. Rare in practice — Docling emits one placeholder per
    picture — but if it ever happens we prefer no-stitch over
    misaligned-stitch.
    """
    md = "<!-- image -->\n\nText\n\n<!-- image -->\n\nMore text.\n"
    original_markdown = md
    conversion = _conv(md)
    extractions: list[ChartOCROutput | Exception] = [
        ChartOCROutput(page_no=1, bbox=(0.0, 0.0, 100.0, 100.0), markdown="data"),
        # Only one extraction for two placeholders → mismatch.
    ]

    stitched = _stitch_chart_extractions(conversion, extractions)
    assert stitched.markdown == original_markdown


def test_stitch_no_placeholders_no_extractions_returns_unchanged() -> None:
    """Empty case: no `<!-- image -->` in markdown, empty extractions
    list. The stitch is a no-op.
    """
    md = "# Doc\n\nAll prose.\n"
    conversion = _conv(md)
    stitched = _stitch_chart_extractions(conversion, [])
    assert stitched.markdown == md


def _fig(page_no: int) -> FigureMetadata:
    return FigureMetadata(
        page_no=page_no,
        bbox=(0.0, 0.0, 100.0, 100.0),
        classification="bar_chart",
        classification_confidence=0.9,
    )


def _decision(
    page: int, engine: Literal["docling", "vlm", "passthrough", "pymupdf"]
) -> PageDecision:
    return PageDecision(page=page, engine=engine, confidence=0.9, rationale="t")


def test_figures_for_chart_ocr_skips_escalated_pages() -> None:
    """Figures on VLM-escalated pages are dropped — their `<!-- image -->`
    placeholders were replaced by VLM prose, so chart-OCR'ing them would make
    the extraction count exceed the surviving placeholders and abort the whole
    stitch. Figures on docling pages survive, in document order."""
    figures = [_fig(1), _fig(4), _fig(4), _fig(5)]
    decisions = [_decision(1, "docling"), _decision(4, "vlm"), _decision(5, "docling")]
    kept = _figures_for_chart_ocr(figures, decisions)
    assert [f.page_no for f in kept] == [1, 5]  # both page-4 figures dropped


def test_figures_for_chart_ocr_passthrough_when_nothing_escalated() -> None:
    """No VLM escalation → every figure is eligible (the disable_vlm default
    path is unchanged)."""
    figures = [_fig(1), _fig(2), _fig(3)]
    decisions = [_decision(p, "docling") for p in (1, 2, 3)]
    kept = _figures_for_chart_ocr(figures, decisions)
    assert [f.page_no for f in kept] == [1, 2, 3]


def test_stitch_preserves_placeholder_token() -> None:
    """The `<!-- image -->` placeholder MUST stay in the markdown
    after stitching. It serves as the visible "there was an image
    here" marker for human readers; the chart-extracted block
    follows it. Removing the placeholder would break round-trips for
    users who edit their vault.
    """
    md = "<!-- image -->\n"
    conversion = _conv(md)
    extractions: list[ChartOCROutput | Exception] = [
        ChartOCROutput(page_no=1, bbox=(0.0, 0.0, 100.0, 100.0), markdown="data"),
    ]
    stitched = _stitch_chart_extractions(conversion, extractions)
    assert "<!-- image -->" in stitched.markdown
    assert "[chart-extracted]" in stitched.markdown
    # Placeholder comes before the chart-extracted block.
    assert stitched.markdown.index("<!-- image -->") < stitched.markdown.index("[chart-extracted]")


def test_collapse_toc_leaders_strips_pagination_artifacts() -> None:
    """audit-10 step 2c: dot-leader + trailing page number removed in prose AND table cells;
    fenced code untouched."""
    from memex.parse.pipeline import _collapse_toc_leaders

    assert _collapse_toc_leaders("|**1**|**Introduction ............ 1**|") == "|**1**|**Introduction**|"
    assert _collapse_toc_leaders("Figure 1: Access ......... 5") == "Figure 1: Access"
    assert _collapse_toc_leaders("plain prose, no leaders") == "plain prose, no leaders"
    fenced = "```\nx = a ...... b\n```"
    assert _collapse_toc_leaders(fenced) == fenced  # literal dots in code untouched


# ======================================================================
# audit-10 step 3 (W2/W15): engine-agnostic heading-hierarchy normalizer
# ======================================================================


def test_normalize_section_number_depth_builds_a_tree() -> None:
    """A flat wall of H2 (born-digital standard whose subsections share the heading font)
    gets re-nested by SECTION-NUMBER depth: `N`→H2, `N.N`→H3, `N.N.N`→H4. Bold-wrapped
    heading text (as both workers emit) is detected and preserved verbatim."""
    from memex.parse.pipeline import normalize_heading_levels

    md = (
        "## **1 Introduction**\n\n"
        "## **1.1 History**\n\n"
        "## **1.1.1 Origins**\n\n"
        "## **2 Tenets**\n"
    )
    out = normalize_heading_levels(md).split("\n")
    assert out[0] == "## **1 Introduction**"  # N → H2
    assert out[2] == "### **1.1 History**"  # N.N → H3 (text kept incl. bold)
    assert out[4] == "#### **1.1.1 Origins**"  # N.N.N → H4
    assert out[6] == "## **2 Tenets**"  # back to H2


def test_normalize_masthead_promoted_when_no_h1() -> None:
    """A doc whose headings never reach H1 and whose first heading is an unnumbered masthead
    gets that title promoted to H1; numbered sections nest beneath by their number."""
    from memex.parse.pipeline import normalize_heading_levels

    md = "## Generative Text Embeddings\n\n## Abstract\n\n## **3.1 Model Architecture**\n"
    out = normalize_heading_levels(md).split("\n")
    assert out[0] == "# Generative Text Embeddings"  # masthead → H1
    assert out[2] == "## Abstract"  # unnumbered section keeps engine level
    assert out[4] == "### **3.1 Model Architecture**"  # N.N → H3


def test_normalize_does_not_promote_when_h1_present() -> None:
    """An existing H1 (e.g. NIST's duplicated masthead) means no masthead promotion — the
    first heading keeps H1 and the rest nest by number."""
    from memex.parse.pipeline import normalize_heading_levels

    md = "# Zero Trust Architecture\n\n## **1 Introduction**\n\n## **1.1 History**\n"
    out = normalize_heading_levels(md).split("\n")
    assert out[0] == "# Zero Trust Architecture"  # untouched H1
    assert out[2] == "## **1 Introduction**"
    assert out[4] == "### **1.1 History**"


def test_normalize_item_and_appendix_anchor_at_h2() -> None:
    """10-K `Item N.` / `Appendix X` labels anchor at H2 regardless of the font-derived level
    (the mass-H6 case): the normalizer overrides the engine level for these."""
    from memex.parse.pipeline import normalize_heading_levels

    md = "###### Item 1. Business\n\n###### Item 7A. Market Risk\n\n###### Appendix A: Glossary\n"
    out = [ln for ln in normalize_heading_levels(md).split("\n") if ln.startswith("#")]
    assert out == ["## Item 1. Business", "## Item 7A. Market Risk", "## Appendix A: Glossary"]


def test_normalize_monotonic_guard_clamps_jumps() -> None:
    """A level that nests more than one deeper than its predecessor is clamped (H2 → H5
    becomes H2 → H3, then the next H6 → H4). An explicit H1 keeps masthead promotion out of
    the way so the clamp is isolated."""
    from memex.parse.pipeline import normalize_heading_levels

    md = "# Doc Title\n\n## Overview\n\n##### Buried Subsection\n\n###### Deeper Still\n"
    out = [ln for ln in normalize_heading_levels(md).split("\n") if ln.startswith("#")]
    assert out == ["# Doc Title", "## Overview", "### Buried Subsection", "#### Deeper Still"]


def test_normalize_leaves_code_fences_inert() -> None:
    """A `#`-comment inside a fenced code block is not a heading and is left untouched."""
    from memex.parse.pipeline import normalize_heading_levels

    md = "## **1 Setup**\n\n```python\n# this is a comment, not a heading\nx = 1\n```\n"
    out = normalize_heading_levels(md)
    assert "## **1 Setup**" in out
    assert "# this is a comment, not a heading" in out  # unchanged inside the fence


def test_normalize_year_prefix_is_not_a_section_number() -> None:
    """A 4-digit year leading a title must NOT be read as a section number (→ would force H2);
    the heading keeps its engine level."""
    from memex.parse.pipeline import normalize_heading_levels

    # With a proper H2 parent the year-led heading KEEPS its engine level (H3); had "2023" been
    # read as a section number it would have been forced to H2 (the visible distinction).
    md = "# Annual Report\n\n## Financials\n\n### 2023 Results and Outlook\n"
    out = normalize_heading_levels(md).split("\n")
    assert out[0] == "# Annual Report"
    assert out[2] == "## Financials"
    assert out[4] == "### 2023 Results and Outlook"  # year not mistaken for a section number


def test_normalize_no_headings_is_noop() -> None:
    from memex.parse.pipeline import normalize_heading_levels

    body = "Just prose.\n\nMore prose, no headings at all.\n"
    assert normalize_heading_levels(body) == body
