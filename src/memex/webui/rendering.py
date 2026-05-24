"""Server-side render helpers for the document view (P4.1 wikilink
section-anchor rendering).

The view-mode template (`templates/_document_body.html`) renders the
canonical markdown body inside a `<pre>` for visual fidelity. P4.1
extends that rendering with two server-side transformations:

1. **Wikilink → anchor tags.** `[[doc_id]]` and `[[doc_id#section]]`
   become real `<a>` tags pointing at the target document (and
   optionally the section fragment within it).

2. **Heading IDs.** Each Markdown heading (`# H1`, `## H2`, ...) gets
   an `id="..."` attribute on a pre-emitted `<span>` so the browser
   can scroll the wikilink's URL fragment into view.

Output is `markupsafe.Markup` (won't be re-escaped by Jinja). The
input is escaped first to prevent XSS from user-edited markdown that
contains literal `<script>` or other HTML.

Lives in `webui/` rather than `core/` because the slug → URL-fragment
convention is webui-specific; future MCP or CLI consumers of the
P4.1 anchors would need different rendering.
"""

from __future__ import annotations

import re

from markupsafe import Markup, escape

from memex.core.text import _MARKDOWN_HEADING_RE, chart_extracted_spans
from memex.core.wikilinks import parse_wikilink

# Same wikilink regex shape as `core/wikilinks._WIKILINK_RE` but re-
# declared here against the ESCAPED body (Jinja's escape replaces `<`
# with `&lt;` etc.; brackets `[` and `]` aren't escaped, so the regex
# still matches the same byte patterns).
_WIKILINK_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")

# Characters allowed in URL-fragment slugs. We keep letters, digits,
# hyphens, and underscores; everything else becomes a hyphen, then
# multiple hyphens collapse to one.
_SLUG_REMOVE_RE = re.compile(r"[^\w\s-]")
_SLUG_WS_RE = re.compile(r"[\s_]+")
_SLUG_DASH_RE = re.compile(r"-+")


def slugify_heading(text: str) -> str:
    """Convert a heading text to a URL fragment slug.

    `"BUSINESS OVERVIEW"` → `"business-overview"`
    `"Methods: Data Movement"` → `"methods-data-movement"`
    `"C# Programming Guide"` → `"c-programming-guide"`

    Slugs are lowercase, alphanumeric + dashes only. Used both when
    rewriting a `[[doc#section]]` wikilink to `/documents/{doc}#{slug}`
    AND when emitting `<span id="{slug}">` anchor targets above each
    rendered heading — keeps the two endpoints in lockstep.
    """
    text = _SLUG_REMOVE_RE.sub(" ", text).strip().lower()
    text = _SLUG_WS_RE.sub("-", text)
    text = _SLUG_DASH_RE.sub("-", text).strip("-")
    return text


def render_body_html(body: str) -> Markup:
    """Render a markdown body for the webui document view.

    Pipeline:
    1. HTML-escape the body (XSS safety).
    2. Replace each `[[...]]` wikilink with an `<a>` tag pointing at
       the target (with `#slug` fragment when a section is specified).
    3. Prepend an empty `<span id="...">` anchor target to each
       Markdown heading line so the URL fragment scrolls into view.
       Skips headings inside `[chart-extracted]` blocks (same defense
       as the chunker / `extract_heading_texts` use — inert chart-
       figure H1 labels shouldn't be link targets).

    Returns `markupsafe.Markup` so Jinja's auto-escape leaves the
    `<a>` and `<span>` tags intact. The wrapped `<pre>` still preserves
    the original whitespace for visual fidelity with the canonical
    markdown.
    """
    escaped = str(escape(body))
    escaped = _replace_wikilinks_with_anchors(escaped)
    escaped = _insert_heading_anchor_targets(body, escaped)
    return Markup(escaped)


def _replace_wikilinks_with_anchors(escaped_body: str) -> str:
    """Substitute each `[[doc]]` / `[[doc#section]]` in `escaped_body`
    with an `<a>` tag. Operates on already-HTML-escaped text — brackets
    survive the escape, so the regex still finds them."""
    def _sub(m: re.Match[str]) -> str:
        target = parse_wikilink(m.group(1))
        if target.section:
            slug = slugify_heading(target.section)
            href = f"/documents/{target.doc_id}#{slug}"
            display = f"[[{target.doc_id}#{target.section}]]"
        else:
            href = f"/documents/{target.doc_id}"
            display = f"[[{target.doc_id}]]"
        return (
            f'<a class="wikilink" href="{escape(href)}">'
            f"{escape(display)}"
            "</a>"
        )

    return _WIKILINK_RE.sub(_sub, escaped_body)


def _insert_heading_anchor_targets(
    original_body: str, escaped_body: str
) -> str:
    """Prepend `<span id="slug" class="anchor-target"></span>` to each
    Markdown heading in the ESCAPED body.

    The chart-block defense is keyed off the ORIGINAL (unescaped) body
    because `chart_extracted_spans` matches `[chart-extracted]` /
    `[/chart-extracted]` literals — escaping doesn't change byte
    offsets (brackets aren't escaped), so the offsets transfer cleanly.
    """
    chart_spans = chart_extracted_spans(original_body)

    def _sub(m: re.Match[str]) -> str:
        # m.start() is in the escaped body. Brackets aren't HTML-escaped
        # so offsets match the original body 1:1 for the purpose of
        # chart-span containment.
        if any(start <= m.start() < end for start, end in chart_spans):
            return m.group(0)
        text = m.group(2).strip()
        if not text:
            return m.group(0)
        slug = slugify_heading(text)
        if not slug:
            return m.group(0)
        return (
            f'<span id="{escape(slug)}" class="anchor-target"></span>'
            f"{m.group(0)}"
        )

    return _MARKDOWN_HEADING_RE.sub(_sub, escaped_body)
