"""P4.1 webui rendering — server-side wikilink-to-anchor + heading-ID
transformations consumed by `_document_body.html`.

Read-side primitives (`core/wikilinks.py`) + write-side wiring
(`enrich.citations`) covered in their own test files; this exercises
the webui-specific slug / anchor / `<a>` emission contract.
"""

from __future__ import annotations

import re

from memex.webui.rendering import (
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
