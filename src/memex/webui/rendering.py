"""Server-side render helpers for the document view (P4.1 wikilink
section-anchor rendering + TOC generation).

The view-mode template (`templates/_document_body.html`) renders the
canonical markdown body inside a `<pre>` for visual fidelity. P4.1
extends that rendering with three server-side transformations:

1. **Wikilink → anchor tags.** `[[doc_id]]` and `[[doc_id#section]]`
   become real `<a>` tags pointing at the target document (and
   optionally the section fragment within it).

2. **Heading IDs + permalink icon.** Each Markdown heading
   (`# H1`, `## H2`, ...) gets an `id="..."` attribute on a
   pre-emitted `<span>` so the browser can scroll the wikilink's
   URL fragment into view; a `<a class="heading-link">#</a>` is
   appended for in-page permalink (GitHub-style, hover-revealed).

3. **Table of contents.** `extract_toc(body)` builds an ordered
   list of `TocEntry(level, text, slug)` from the doc's headings,
   feeding the sticky right-rail navigation in `document.html`.

Output is `markupsafe.Markup` (won't be re-escaped by Jinja). The
input is escaped first to prevent XSS from user-edited markdown that
contains literal `<script>` or other HTML.

Lives in `webui/` rather than `core/` because the slug → URL-fragment
convention is webui-specific; future MCP or CLI consumers of the
P4.1 anchors would need different rendering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from markupsafe import Markup, escape

from memex.core.text import (
    _MARKDOWN_HEADING_RE,
    chart_extracted_spans,
    is_inside_any_span,
)
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


# Inline-markdown patterns stripped from heading text before slugging
# / display. Headings in parsed docs sometimes carry link, bold, or
# code syntax — e.g. the Tableau guide has `## [Tips:](https://...)`
# headings. Without cleaning, those produce ugly slugs
# (`tips-https-www-tableau-com-...`) and raw `[Tips:](url)` TOC entries.
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")  # [text](url) → text
_MD_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")  # **text** → text
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")  # *text* → text
_MD_CODE_RE = re.compile(r"`([^`]+)`")  # `text` → text


def clean_heading_text(text: str) -> str:
    """Strip inline-markdown syntax (links, bold, italic, code) from a
    heading's text, leaving the human-readable label.

    `"[Tips:](https://x.com)"` → `"Tips:"`
    `"**Bold Heading**"` → `"Bold Heading"`
    `` "`code` ref" `` → `"code ref"`

    Used by `_walk_headings` so both the TOC display text AND the slug
    derive from the clean label — a `## [Tips:](url)` heading shows
    "Tips:" in the TOC and slugs to `tips` (then dedups against other
    `Tips:` sections). Order matters: links first (their text may
    contain bold/code), then bold, italic, code.
    """
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(r"\1", text)
    text = _MD_CODE_RE.sub(r"\1", text)
    return text.strip()


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

    Implemented line-by-line over the ORIGINAL body so HTML-escaping
    never drifts the byte offsets the heading / chart-block analysis
    depends on. (An earlier whole-body `escape()` → regex-substitute
    approach broke on any doc with a `<` or `&` before a chart block —
    e.g. the `<!-- image -->` markers everywhere in slide-deck markdown
    shifted offsets and misfired the chart-block filter. Fixed
    2026-05-23 alongside the duplicate-heading dedup.)
    """
    # Single source of truth: real headings (chart-block-aware, slugs
    # deduplicated) keyed by their line-start offset in the ORIGINAL
    # body. A heading line begins exactly at its `#` (the regex anchors
    # `^#` with no leading whitespace), so the match start == line start.
    slug_by_line_start = {h.start: h.slug for h in _walk_headings(body)}

    parts: list[str] = []
    offset = 0
    for line in body.splitlines(keepends=True):
        line_start = offset
        offset += len(line)
        if line.endswith("\n"):
            content, newline = line[:-1], "\n"
        else:
            content, newline = line, ""
        rendered = _replace_wikilinks_with_anchors(str(escape(content)))
        slug = slug_by_line_start.get(line_start)
        if slug:
            rendered = f'<span id="{escape(slug)}" class="anchor-target"></span>{rendered}'
        parts.append(rendered + newline)
    # Every segment was HTML-escaped per line above (see the offset-stable
    # escaping note); Markup just stops Jinja re-escaping our own markup.
    return Markup("".join(parts))  # noqa: S704


def _replace_wikilinks_with_anchors(escaped_line: str) -> str:
    """Substitute each `[[doc]]` / `[[doc#section]]` in `escaped_line`
    with an `<a>` tag. Operates on already-HTML-escaped text — brackets
    survive the escape, so the regex still finds them.

    The section anchor always points at the BASE slug (first occurrence
    of that heading text). Duplicate headings can't be disambiguated by
    text alone, so a `[[doc#Tips:]]` link resolves to the first "Tips:"
    section — which is what the dedup'd anchor IDs assign to the base
    slug. The 2nd+ occurrences get `-1`, `-2` suffixes that no wikilink
    targets (by design)."""

    def _sub(m: re.Match[str]) -> str:
        target = parse_wikilink(m.group(1))
        if target.section:
            slug = slugify_heading(target.section)
            href = f"/documents/{target.doc_id}#{slug}"
            display = f"[[{target.doc_id}#{target.section}]]"
        else:
            href = f"/documents/{target.doc_id}"
            display = f"[[{target.doc_id}]]"
        return f'<a class="wikilink" href="{escape(href)}">{escape(display)}</a>'

    return _WIKILINK_RE.sub(_sub, escaped_line)


# ----------------------------------------------------------------------
# Table of contents + heading walk
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TocEntry:
    """One entry in the document's table of contents.

    `level` is the ATX heading depth (1 for `#`, 2 for `##`, etc.) —
    used by the template to indent the entry visually.

    `text` is the heading's text in its original casing, for display.

    `slug` is the URL fragment that matches the in-document anchor
    target — clicking the TOC entry navigates to `#{slug}`, scrolling
    the corresponding `<span id="slug">` into view. Deduplicated: a
    document with three `Tips:` headings produces slugs `tips`,
    `tips-1`, `tips-2` so each TOC entry scrolls to its OWN section.
    """

    level: int
    text: str
    slug: str


@dataclass(frozen=True)
class _Heading:
    """Internal: one real heading with its original-body position,
    level, display text, and deduplicated slug. The single source of
    truth shared by `render_body_html` (anchor-span IDs) and
    `extract_toc` (TOC fragment links) so the two stay in lockstep."""

    start: int
    level: int
    text: str
    slug: str


def _walk_headings(body: str) -> list[_Heading]:
    """Walk every real Markdown heading in `body` in document order,
    returning `_Heading`s with deduplicated slugs.

    "Real" excludes:
    - headings inside `[chart-extracted]` blocks (inert chart-figure
      `# H1` labels — same defense as the chunker / `extract_heading_
      texts`)
    - headings whose text is empty or slugs to empty (whitespace-only
      or symbol-only — can't be link targets)

    Slug dedup is GitHub-style: the first `Methods` heading gets
    `methods`, the second `methods-1`, the third `methods-2`, etc.
    This guarantees unique `id=` attributes (duplicate IDs are invalid
    HTML and the browser only scrolls to the first) and lets each TOC
    entry scroll to its own section.

    All analysis runs on the ORIGINAL (unescaped) body so offsets are
    stable — callers map `_Heading.start` to original-body line starts.
    """
    chart_spans = chart_extracted_spans(body)
    seen: dict[str, int] = {}
    out: list[_Heading] = []
    for m in _MARKDOWN_HEADING_RE.finditer(body):
        if is_inside_any_span(m.start(), chart_spans):
            continue
        # Strip inline-markdown (links/bold/code) so the TOC label and
        # slug derive from the clean text — `## [Tips:](url)` → "Tips:".
        text = clean_heading_text(m.group(2).strip())
        if not text:
            continue
        base = slugify_heading(text)
        if not base:
            continue
        count = seen.get(base, 0)
        seen[base] = count + 1
        slug = base if count == 0 else f"{base}-{count}"
        out.append(
            _Heading(
                start=m.start(),
                level=len(m.group(1)),
                text=text,
                slug=slug,
            )
        )
    return out


def extract_toc(body: str) -> list[TocEntry]:
    """Build a flat list of TOC entries from `body`'s Markdown
    headings, in document order. Chart-block-aware + slug-deduplicated
    (shares `_walk_headings` with `render_body_html`, so every TOC
    fragment matches exactly one anchor-target span).

    Returns `[]` when the body has no navigable headings. Callers
    (`document.html`) hide the TOC when the list is < 3 or > 50 entries
    (too short to navigate / parse-noise).
    """
    return [TocEntry(level=h.level, text=h.text, slug=h.slug) for h in _walk_headings(body)]
