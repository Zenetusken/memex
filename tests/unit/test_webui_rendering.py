"""P4.1 webui rendering — server-side wikilink-to-anchor + heading-ID
transformations consumed by `_document_body.html`.

Read-side primitives (`core/wikilinks.py`) + write-side wiring
(`enrich.citations`) covered in their own test files; this exercises
the webui-specific slug / anchor / `<a>` emission contract.
"""

from __future__ import annotations

import re

from memex.webui.rendering import (
    TocEntry,
    extract_toc,
    render_body_html,
    slugify_heading,
)


# ----------------------------------------------------------------------
# slugify_heading
# ----------------------------------------------------------------------


def test_slugify_lowercase_with_dashes() -> None:
    """Standard heading slug: lowercase, spaces → dashes,
    alphanumerics only."""
    assert slugify_heading("BUSINESS OVERVIEW") == "business-overview"
    assert slugify_heading("Methods: Data Movement") == "methods-data-movement"
    assert slugify_heading("Bar chart") == "bar-chart"


def test_slugify_strips_special_chars() -> None:
    """Non-alphanumeric chars (other than dash + underscore) get
    replaced. `C# Programming Guide` → `c-programming-guide`."""
    assert slugify_heading("C# Programming Guide") == "c-programming-guide"
    assert slugify_heading("Q1 / FY2024 — overview") == "q1-fy2024-overview"


def test_slugify_collapses_multiple_dashes() -> None:
    """Multiple adjacent separators collapse to a single dash."""
    assert slugify_heading("Foo -- Bar   Baz") == "foo-bar-baz"


def test_slugify_strips_leading_trailing_dashes() -> None:
    """Leading/trailing punctuation gets cleaned off."""
    assert slugify_heading("  Heading  ") == "heading"
    assert slugify_heading("--Methods--") == "methods"


def test_slugify_empty_returns_empty() -> None:
    """Whitespace-only or symbol-only headings slug to empty (the
    caller skips emitting an anchor target for those)."""
    assert slugify_heading("") == ""
    assert slugify_heading("   ") == ""
    assert slugify_heading("###") == ""


# ----------------------------------------------------------------------
# render_body_html — wikilink rewriting
# ----------------------------------------------------------------------


def test_render_plain_wikilink_becomes_anchor_tag() -> None:
    """`[[doc-id]]` → `<a class="wikilink" href="/documents/doc-id">[[doc-id]]</a>`."""
    out = str(render_body_html("See [[doc-id]] for details."))
    assert '<a class="wikilink" href="/documents/doc-id">[[doc-id]]</a>' in out


def test_render_section_wikilink_emits_fragment() -> None:
    """`[[doc#Section Name]]` → `/documents/doc#section-name` (slugified)."""
    out = str(render_body_html("See [[my-doc#Methods]] for details."))
    assert 'href="/documents/my-doc#methods"' in out
    assert "[[my-doc#Methods]]" in out


def test_render_preserves_display_text_casing() -> None:
    """The displayed `[[doc#Section]]` text keeps original casing
    even though the URL fragment is lowercase-slugified."""
    out = str(render_body_html("[[doc#BUSINESS OVERVIEW]]"))
    assert "[[doc#BUSINESS OVERVIEW]]" in out
    assert 'href="/documents/doc#business-overview"' in out


def test_render_escapes_html_in_body() -> None:
    """Literal `<script>` in body must be escaped (XSS defense)."""
    out = str(render_body_html("<script>alert('xss')</script>"))
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_render_multiple_wikilinks_in_one_body() -> None:
    """Multiple wikilinks in the same body all rewrite, in document order."""
    body = "First [[doc-a]] then [[doc-b#Intro]] then plain text."
    out = str(render_body_html(body))
    assert 'href="/documents/doc-a"' in out
    assert 'href="/documents/doc-b#intro"' in out


def test_render_no_wikilinks_passes_through_escaped() -> None:
    """Body without wikilinks is just HTML-escaped (no `<a>` tags)."""
    out = str(render_body_html("plain prose with no wikilinks"))
    assert "<a" not in out
    assert "plain prose" in out


# ----------------------------------------------------------------------
# render_body_html — heading anchor target injection
# ----------------------------------------------------------------------


def test_render_heading_gets_anchor_target_span() -> None:
    """Each `## Heading` gets a `<span id="...">` injected before the
    `##` text so URL fragments scroll the heading into view."""
    body = "## BUSINESS OVERVIEW\n\nProse content."
    out = str(render_body_html(body))
    assert '<span id="business-overview" class="anchor-target"></span>' in out


def test_render_multiple_headings_get_distinct_ids() -> None:
    """Multiple headings each get their own anchor target with their
    own slug."""
    body = "## Introduction\n\n## Methods\n\n## Results\n"
    out = str(render_body_html(body))
    assert 'id="introduction"' in out
    assert 'id="methods"' in out
    assert 'id="results"' in out


def test_render_heading_levels_1_through_6_all_anchored() -> None:
    """ATX headings at all 6 levels get anchor targets."""
    body = "\n".join(f"{'#' * lvl} Heading L{lvl}" for lvl in range(1, 7))
    out = str(render_body_html(body))
    for lvl in range(1, 7):
        assert f'id="heading-l{lvl}"' in out


def test_render_skips_h1_inside_chart_extracted_block() -> None:
    """Inert `# H1` labels inside `[chart-extracted]` blocks (Nemotron-
    Parse chart-figure labels) must NOT get anchor targets — same
    defense as the chunker / `extract_heading_texts` use."""
    body = (
        "## Real Heading\n\n"
        "[chart-extracted]\n"
        "# Inert Chart Label\n"
        "[/chart-extracted]\n\n"
        "## Another Real Heading\n"
    )
    out = str(render_body_html(body))
    assert 'id="real-heading"' in out
    assert 'id="another-real-heading"' in out
    # The inert chart-figure label gets NO anchor target
    assert 'id="inert-chart-label"' not in out


def test_render_empty_heading_text_skipped() -> None:
    """A `## ` with only whitespace text gets no anchor target."""
    body = "##   \n\nProse."
    out = str(render_body_html(body))
    assert "anchor-target" not in out


# ----------------------------------------------------------------------
# Integration: wikilinks + headings in the same body
# ----------------------------------------------------------------------


def test_render_full_body_wikilinks_and_headings_coexist() -> None:
    """A realistic mixed body: wikilinks point at other-doc sections,
    own-doc headings get anchor targets — both transformations apply
    independently."""
    body = (
        "## Introduction\n\n"
        "See [[other-doc#Methods]] for related work.\n\n"
        "## Methods\n\n"
        "Our approach builds on [[other-doc]].\n"
    )
    out = str(render_body_html(body))
    # Wikilink with section → fragment
    assert 'href="/documents/other-doc#methods"' in out
    # Wikilink without section → no fragment
    assert 'href="/documents/other-doc"' in out
    # Own-doc headings have anchor targets
    assert 'id="introduction"' in out
    assert 'id="methods"' in out
    # Original markdown content preserved (escaped)
    assert "## Introduction" in out
    assert "## Methods" in out


def test_render_output_is_markup_object() -> None:
    """The return is a `markupsafe.Markup` so Jinja's auto-escape
    doesn't re-escape the `<a>` and `<span>` tags."""
    from markupsafe import Markup

    out = render_body_html("## H\n\n[[doc]]")
    assert isinstance(out, Markup)


# ----------------------------------------------------------------------
# Heading permalink (`#` link icon) emission — P4.1 polish (2026-05-23)
# ----------------------------------------------------------------------


def test_render_heading_emits_permalink_link() -> None:
    """Each heading also gets a `<a class="heading-link" href="#slug">#</a>`
    appended for in-page permalink (GitHub-style, hover-revealed via CSS)."""
    body = "## BUSINESS OVERVIEW\n\nProse."
    out = str(render_body_html(body))
    assert '<a class="heading-link" href="#business-overview"' in out
    assert ">#</a>" in out


def test_render_permalink_skipped_inside_chart_block() -> None:
    """Inert chart-block H1 labels get neither anchor target NOR
    permalink link — they aren't real document sections."""
    body = (
        "## Real Heading\n\n"
        "[chart-extracted]\n"
        "# Inert Label\n"
        "[/chart-extracted]\n"
    )
    out = str(render_body_html(body))
    # Real heading has permalink
    assert 'href="#real-heading"' in out
    # Inert chart label does not
    assert 'href="#inert-label"' not in out


# ----------------------------------------------------------------------
# extract_toc — TOC generation for the sidebar
# ----------------------------------------------------------------------


def test_extract_toc_returns_entries_in_document_order() -> None:
    """Headings appear in the TOC in the order they appear in the body."""
    body = (
        "# Top\n\n"
        "## Methods\n\n"
        "### Subsection\n\n"
        "## Results\n"
    )
    toc = extract_toc(body)
    assert [e.text for e in toc] == ["Top", "Methods", "Subsection", "Results"]


def test_extract_toc_records_heading_level() -> None:
    """The TOC entry's `level` matches the ATX depth (`#` = 1, `##` = 2)."""
    body = "# H1\n\n## H2\n\n### H3\n\n#### H4\n\n##### H5\n\n###### H6\n"
    toc = extract_toc(body)
    assert [e.level for e in toc] == [1, 2, 3, 4, 5, 6]


def test_extract_toc_slugifies_each_entry() -> None:
    """Each TOC entry's slug matches the document's heading anchor
    slug — so clicking the entry navigates to the right section."""
    body = "## BUSINESS OVERVIEW\n\n## Methods: Data Movement\n"
    toc = extract_toc(body)
    assert toc[0].slug == "business-overview"
    assert toc[1].slug == "methods-data-movement"


def test_extract_toc_skips_chart_block_h1s() -> None:
    """Same defense as `extract_heading_texts` and the chunker —
    inert chart-figure H1 labels don't appear in the TOC."""
    body = (
        "## Real One\n\n"
        "[chart-extracted]\n"
        "# Chart Label\n"
        "[/chart-extracted]\n\n"
        "## Real Two\n"
    )
    toc = extract_toc(body)
    assert [e.text for e in toc] == ["Real One", "Real Two"]


def test_extract_toc_empty_body_returns_empty() -> None:
    """Body without headings returns an empty list. Template hides
    the TOC sidebar when len(toc) < 3 anyway."""
    assert extract_toc("just prose, no headings here") == []
    assert extract_toc("") == []


def test_extract_toc_skips_empty_headings_and_symbol_only() -> None:
    """Headings whose slug would be empty (whitespace-only,
    symbol-only) are skipped — can't be link targets either."""
    body = "## Real Heading\n\n## ###\n\n## Another Real\n"
    toc = extract_toc(body)
    # The "###" heading has no alphanumerics → empty slug → skipped
    # (Also: with the tightened heading regex, `## ###` may or may not
    # match — but even if it does, slug ends up empty.)
    assert all(e.text != "###" or e.slug for e in toc)
    real = [e for e in toc if e.text in ("Real Heading", "Another Real")]
    assert len(real) == 2


def test_toc_entry_is_frozen_dataclass() -> None:
    """`TocEntry` is a frozen dataclass — immutable + hashable, so
    template caching and equality checks behave sanely."""
    a = TocEntry(level=2, text="Methods", slug="methods")
    b = TocEntry(level=2, text="Methods", slug="methods")
    assert a == b
    # Hashability
    assert {a, b} == {a}
