"""P4.1 webui rendering — server-side wikilink-to-anchor + heading-ID
transformations consumed by `_document_body.html`.

Read-side primitives (`core/wikilinks.py`) + write-side wiring
(`enrich.citations`) covered in their own test files; this exercises
the webui-specific slug / anchor / `<a>` emission contract.
"""

from __future__ import annotations

from markupsafe import Markup

from memex.webui.rendering import (
    TocEntry,
    extract_toc,
    render_body_html,
    render_wikilink,
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
# No visible permalink glyph — raw-markdown fidelity (2026-05-23
# browser-review fix). The earlier `<a class="heading-link">#</a>`
# polluted the <pre> with a literal `#` that read as ATX closing-hash
# syntax. Removed; navigation stays via TOC + invisible anchor spans.
# ----------------------------------------------------------------------


def test_render_emits_no_visible_permalink_glyph() -> None:
    """No `heading-link` anchor / no injected `#` glyph after headings —
    the raw-markdown view must show ONLY the canonical markdown text
    (plus invisible anchor-target spans)."""
    body = "## BUSINESS OVERVIEW\n\nProse."
    out = str(render_body_html(body))
    assert "heading-link" not in out
    # The only `#` characters present are the literal `##` heading
    # prefix from the source — no trailing permalink `#` was injected.
    assert out.count("#") == out.count("##") * 2  # each `##` = two `#`
    # The invisible anchor target is still there for navigation.
    assert 'id="business-overview" class="anchor-target"' in out


def test_render_anchor_span_skipped_inside_chart_block() -> None:
    """Inert chart-block H1 labels get no anchor target — they aren't
    real document sections."""
    body = "## Real Heading\n\n[chart-extracted]\n# Inert Label\n[/chart-extracted]\n"
    out = str(render_body_html(body))
    assert 'id="real-heading"' in out
    assert 'id="inert-label"' not in out


# ----------------------------------------------------------------------
# extract_toc — TOC generation for the sidebar
# ----------------------------------------------------------------------


def test_extract_toc_returns_entries_in_document_order() -> None:
    """Headings appear in the TOC in the order they appear in the body."""
    body = "# Top\n\n## Methods\n\n### Subsection\n\n## Results\n"
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
    body = "## Real One\n\n[chart-extracted]\n# Chart Label\n[/chart-extracted]\n\n## Real Two\n"
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


# ----------------------------------------------------------------------
# Edge cases — duplicate headings, HTML-special chars, offset stability
# (2026-05-23 browser-review follow-up)
# ----------------------------------------------------------------------


def test_duplicate_headings_get_deduped_slugs_in_toc() -> None:
    """Three `Tips:` headings → slugs tips, tips-1, tips-2 so each TOC
    entry scrolls to its OWN section (not all to the first). The
    chart-types Tableau doc has exactly this pattern."""
    body = (
        "## Bar chart\n\n## Tips:\n\nprose\n\n"
        "## Line chart\n\n## Tips:\n\nprose\n\n"
        "## Pie chart\n\n## Tips:\n"
    )
    toc = extract_toc(body)
    tips_slugs = [e.slug for e in toc if e.text == "Tips:"]
    assert tips_slugs == ["tips", "tips-1", "tips-2"]


def test_duplicate_headings_get_unique_span_ids() -> None:
    """The rendered anchor-target spans must use the SAME deduped slugs
    as the TOC — unique `id=` per heading (duplicate IDs are invalid
    HTML; browser only scrolls to the first)."""
    body = "## Tips:\n\na\n\n## Tips:\n\nb\n\n## Tips:\n"
    out = str(render_body_html(body))
    assert 'id="tips"' in out
    assert 'id="tips-1"' in out
    assert 'id="tips-2"' in out
    # Exactly one of each — no duplicate id="tips"
    assert out.count('id="tips"') == 1
    assert out.count('id="tips-1"') == 1


def test_toc_and_render_slugs_stay_in_lockstep() -> None:
    """Every TOC fragment must match exactly one anchor-target span id
    in the rendered body — the shared `_walk_headings` guarantees it
    even with duplicates."""
    body = "## Intro\n\n## Methods\n\n## Methods\n\n## Results\n\n## Methods\n"
    toc = extract_toc(body)
    out = str(render_body_html(body))
    for entry in toc:
        assert f'id="{entry.slug}"' in out, f"TOC slug {entry.slug} has no span"
    # Methods appears 3x → methods, methods-1, methods-2
    methods_slugs = [e.slug for e in toc if e.text == "Methods"]
    assert methods_slugs == ["methods", "methods-1", "methods-2"]


def test_heading_with_ampersand_offset_stable() -> None:
    """A heading after an HTML-special char (`&`, `<`) must still get a
    correct anchor span. The old whole-body-escape approach drifted
    offsets here and misfired the chart-block filter. Line-by-line
    rendering fixes it."""
    # `<!-- image -->` contains `<` which escapes to &lt; (+3 chars).
    # A heading AFTER it must still be anchored correctly.
    body = "<!-- image -->\n\nAT&T earnings & growth\n\n## Revenue Section\n\nprose\n"
    out = str(render_body_html(body))
    # The < and & are escaped (XSS safety)
    assert "&lt;!-- image --&gt;" in out
    assert "AT&amp;T" in out
    # The heading still gets its anchor span despite the preceding
    # special chars shifting escaped-text offsets
    assert 'id="revenue-section"' in out
    assert "## Revenue Section" in out


def test_heading_inside_chart_block_after_special_chars_still_skipped() -> None:
    """The chart-block filter must hold even when HTML-special chars
    precede the chart block (the offset-drift bug's worst case). The
    `# Inert` heading inside the block gets NO anchor span."""
    body = (
        "## Real Before\n\n"
        "<!-- image with < and & chars -->\n\n"
        "[chart-extracted]\n# Inert Label\n[/chart-extracted]\n\n"
        "## Real After\n"
    )
    out = str(render_body_html(body))
    assert 'id="real-before"' in out
    assert 'id="real-after"' in out
    assert 'id="inert-label"' not in out


def test_unicode_heading_slug_french() -> None:
    """French headings (accented chars) slugify to lowercase unicode
    word chars — the CR350 course doc has these. Span id + TOC slug
    stay consistent."""
    body = "## Élève et Sécurité\n\nprose\n\n## Défense\n\nprose\n\n## Audit\n"
    toc = extract_toc(body)
    out = str(render_body_html(body))
    for entry in toc:
        assert f'id="{entry.slug}"' in out


def test_render_wikilink_and_heading_on_same_line_not_confused() -> None:
    """A heading line is never also a wikilink-rewrite target in a way
    that double-processes; and a wikilink on a prose line still renders
    while that line gets no anchor span."""
    body = "## Section\n\nSee [[other-doc#Methods]] here.\n"
    out = str(render_body_html(body))
    assert 'id="section"' in out
    assert 'href="/documents/other-doc#methods"' in out
    # The prose line with the wikilink is NOT a heading → no span on it
    # (only one anchor-target span total, for the heading)
    assert out.count("anchor-target") == 1


# ----------------------------------------------------------------------
# clean_heading_text — inline-markdown stripping for headings
# (2026-05-23, the chart-types `## [Tips:](url)` edge case)
# ----------------------------------------------------------------------


def test_clean_heading_text_strips_markdown_link() -> None:
    from memex.webui.rendering import clean_heading_text

    assert clean_heading_text("[Tips:](https://www.tableau.com/x)") == "Tips:"
    assert clean_heading_text("[Highlight table](https://x.com/y)") == "Highlight table"


def test_clean_heading_text_strips_bold_italic_code() -> None:
    from memex.webui.rendering import clean_heading_text

    assert clean_heading_text("**Bold Heading**") == "Bold Heading"
    assert clean_heading_text("*Italic*") == "Italic"
    assert clean_heading_text("`code` ref") == "code ref"


def test_clean_heading_text_plain_passes_through() -> None:
    from memex.webui.rendering import clean_heading_text

    assert clean_heading_text("Bar chart") == "Bar chart"
    assert clean_heading_text("BUSINESS OVERVIEW") == "BUSINESS OVERVIEW"


def test_markdown_link_heading_gets_clean_toc_text_and_slug() -> None:
    """A `## [Tips:](url)` heading shows "Tips:" in the TOC and slugs to
    `tips` — NOT the ugly `tips-https-www-...` from the raw link text."""
    body = "## [Tips:](https://www.tableau.com/resource/dos-and-donts)\n\nprose\n"
    toc = extract_toc(body)
    assert len(toc) == 1
    assert toc[0].text == "Tips:"
    assert toc[0].slug == "tips"
    out = str(render_body_html(body))
    assert 'id="tips"' in out


def test_markdown_link_heading_dedups_against_plain() -> None:
    """A `## [Tips:](url)` heading and a plain `## Tips:` heading both
    clean to "Tips:" → they dedup together (tips, tips-1)."""
    body = (
        "## Tips:\n\na\n\n## [Tips:](https://x.com/y)\n\nb\n\n## More content here so TOC shows\n"
    )
    toc = extract_toc(body)
    tips = [e for e in toc if e.text == "Tips:"]
    assert [e.slug for e in tips] == ["tips", "tips-1"]


# ----------------------------------------------------------------------
# render_wikilink — the public "Sources" Jinja filter (P4.1 emission)
# ----------------------------------------------------------------------


def test_render_wikilink_section_emits_slug_fragment() -> None:
    """`[[doc#Sec]]` → an `<a>` to `/documents/doc#sec` (section slugified). The
    visible label is "doc › Sec" (doc-id fallback when no title map); the raw
    `doc#Sec` is only the `title=` hover tooltip, NOT the rendered text."""
    out = str(render_wikilink("[[doc#Sec]]"))
    assert 'href="/documents/doc#sec"' in out
    assert 'class="wikilink"' in out
    assert "doc › Sec" in out  # by-title label
    assert 'title="doc#Sec"' in out  # raw id survives only as the tooltip
    assert "[[doc#Sec]]" not in out  # raw wikilink syntax no longer shown


def test_render_wikilink_uses_title_map() -> None:
    """Given a doc_id→title map, the label is the human TITLE (+ section); the
    href + tooltip keep the stable doc-id."""
    out = str(render_wikilink("[[6cf-notes#Intro]]", {"6cf-notes": "CS Notes"}))
    assert "CS Notes › Intro" in out  # title replaces the raw doc-id in the label
    assert 'href="/documents/6cf-notes#intro"' in out
    assert 'title="6cf-notes#Intro"' in out
    assert ">6cf-notes" not in out  # the doc-id is not the visible label


def test_render_wikilink_cleans_inline_markdown_in_section() -> None:
    """A parsed heading like `**Zero Trust Architecture**` must show as clean
    text (not literal asterisks) in the Sources label, and slug off the CLEANED
    text so the href matches the doc-page anchor (which `_walk_headings` builds
    from the cleaned label). The raw section stays in the `title=` tooltip."""
    out = str(render_wikilink("[[nist-207#**Zero Trust Architecture**]]", {"nist-207": "NIST 207"}))
    assert "NIST 207 › Zero Trust Architecture" in out  # no `**` in the label
    assert "**" not in out.split('title="')[0]  # no asterisks in the visible label/href
    assert 'href="/documents/nist-207#zero-trust-architecture"' in out  # slug off cleaned text


def test_render_wikilink_bare_doc_has_no_fragment() -> None:
    """A bare `[[doc]]` → `/documents/doc` with NO `#` fragment."""
    out = str(render_wikilink("[[doc]]"))
    assert 'href="/documents/doc"' in out
    assert "#" not in out


def test_render_wikilink_unparseable_is_escaped_raw() -> None:
    """Defensive: an input with no well-formed `[[...]]` is returned as
    escaped raw text, never raw HTML."""
    out = str(render_wikilink("not a wikilink <script>"))
    assert "<a" not in out
    assert "&lt;script&gt;" in out


def test_render_wikilink_returns_markup() -> None:
    """The filter returns `markupsafe.Markup` so Jinja won't re-escape
    the emitted `<a>`."""
    assert isinstance(render_wikilink("[[doc]]"), Markup)
