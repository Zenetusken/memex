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
from pydantic import ValidationError

from memex.core.text import (
    chart_extracted_spans,
    is_inside_any_span,
)
from memex.core.wikilinks import parse_wikilink

# Markdown ATX-heading matcher. Declared locally (not imported from
# `core/text._MARKDOWN_HEADING_RE`) to avoid reaching into another
# module's private surface — same pattern shape, kept in lockstep:
# horizontal-whitespace-only between the `#` run and the text so an
# empty `## ` line doesn't swallow the following paragraph.
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)

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


def render_body_html(body: str, *, headings: list[_Heading] | None = None) -> Markup:
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
    # `headings` may be passed in (by `render_body_and_toc`) to share ONE
    # walk with `extract_toc`; default None recomputes (standalone callers).
    walked = _walk_headings(body) if headings is None else headings
    slug_by_line_start = {h.start: h.slug for h in walked}

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


def _anchor_for_wikilink(inner: str) -> str:
    """Build the `<a class="wikilink">` HTML for one wikilink's inner
    text (the part between `[[` and `]]`, e.g. `doc#Section`).

    The single source of truth for the wikilink → anchor construction,
    shared by `_replace_wikilinks_with_anchors` (the body rewrite) and
    the public `render_wikilink` (the "Sources" list filter) so the two
    never drift. Returns already-HTML-escaped HTML — the `href` and the
    displayed label are escaped here.

    A `[[doc#section]]` links to `/documents/{doc}#{slug}` (the section
    slugified); a bare `[[doc]]` links to `/documents/{doc}` with no
    fragment. The section anchor always points at the BASE slug (first
    occurrence of that heading text) — duplicate headings can't be
    disambiguated by text alone, mirroring the dedup'd anchor IDs.
    """
    try:
        target = parse_wikilink(inner)
    except ValidationError:
        # A malformed inner (empty doc_id — `[[#section]]`, `[[ ]]`) can't form a
        # link; emit it as an inert escaped literal rather than 500-ing the body.
        return str(escape(f"[[{inner}]]"))
    if target.section:
        slug = slugify_heading(target.section)
        href = f"/documents/{target.doc_id}#{slug}"
        display = f"[[{target.doc_id}#{target.section}]]"
    else:
        href = f"/documents/{target.doc_id}"
        display = f"[[{target.doc_id}]]"
    return f'<a class="wikilink" href="{escape(href)}">{escape(display)}</a>'


def _replace_wikilinks_with_anchors(escaped_line: str) -> str:
    """Substitute each `[[doc]]` / `[[doc#section]]` in `escaped_line`
    with an `<a>` tag. Operates on already-HTML-escaped text — brackets
    survive the escape, so the regex still finds them.

    Delegates the per-match construction to `_anchor_for_wikilink` (the
    shared helper) so the body rewrite and the `render_wikilink` filter
    stay in lockstep."""
    return _WIKILINK_RE.sub(lambda m: _anchor_for_wikilink(m.group(1)), escaped_line)


def render_wikilink(wikilink: str, titles: dict[str, str] | None = None) -> Markup:
    """Render a single `[[doc]]` / `[[doc#section]]` wikilink string as
    an `<a class="wikilink">` anchor — the public Jinja filter behind the
    answer / summary "Sources" list (`templates/_answer.html`).

    The link LABEL is the **human document title** (+ `› section`) when a
    `doc_id → title` map is supplied (`titles`), instead of the raw
    `[[doc_id#section]]` syntax — the stable doc-id survives only as the
    `href` and a `title=` hover tooltip. With no/empty map (or an unknown
    doc-id) the label falls back to the doc-id, so it never renders blank.
    An unparseable input (no well-formed `[[...]]`) is returned as escaped
    raw text — defensive, never raw HTML.

    NB this DIVERGES from `_anchor_for_wikilink` (the document-BODY rewrite),
    which keeps the literal `[[doc#section]]` label for raw-markdown fidelity
    inside the `<pre>`. "Sources" is a presentation surface, so it reads by
    title; both still emit `class="wikilink"` + the same href shape. The raw
    input is parsed first, then every interpolation is escaped at output
    (single escape — no double-escaping).

    Returns `markupsafe.Markup` so Jinja's auto-escape leaves the emitted
    `<a>` intact.
    """
    match = _WIKILINK_RE.search(wikilink)
    if match is None:
        return Markup(str(escape(wikilink)))  # noqa: S704 — escaped above
    try:
        target = parse_wikilink(match.group(1))
    except ValidationError:
        # Empty doc_id (`[[#section]]`, `[[ ]]`) — fall back to the escaped literal.
        return Markup(str(escape(wikilink)))  # noqa: S704 — escaped above
    doc_label = (titles or {}).get(target.doc_id, target.doc_id)
    if target.section:
        # Clean inline-markdown out of the section for DISPLAY + slug (a parsed
        # heading like `**Zero Trust Architecture**` would otherwise show literal
        # asterisks and slug off the raw text — mismatching the doc-page anchor,
        # which `_walk_headings` builds from the cleaned label). The raw section
        # stays in the `title=` tooltip for traceability.
        section_label = clean_heading_text(target.section)
        href = f"/documents/{target.doc_id}#{slugify_heading(section_label)}"
        label = f"{doc_label} › {section_label}"  # "Title › Section"
        tip = f"{target.doc_id}#{target.section}"
    else:
        href = f"/documents/{target.doc_id}"
        label = doc_label
        tip = target.doc_id
    return Markup(  # noqa: S704 — all interpolations escaped
        f'<a class="wikilink" href="{escape(href)}" title="{escape(tip)}">{escape(label)}</a>'
    )


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


def extract_toc(body: str, *, headings: list[_Heading] | None = None) -> list[TocEntry]:
    """Build a flat list of TOC entries from `body`'s Markdown
    headings, in document order. Chart-block-aware + slug-deduplicated
    (shares `_walk_headings` with `render_body_html`, so every TOC
    fragment matches exactly one anchor-target span).

    Returns `[]` when the body has no navigable headings. Callers
    (`document.html`) hide the TOC when the list is < 3 or > 50 entries
    (too short to navigate / parse-noise). `headings` may be passed in
    (by `render_body_and_toc`) to share ONE walk with `render_body_html`.
    """
    walked = _walk_headings(body) if headings is None else headings
    return [TocEntry(level=h.level, text=h.text, slug=h.slug) for h in walked]


def render_body_and_toc(body: str) -> tuple[Markup, list[TocEntry]]:
    """Render the body HTML AND the TOC from a SINGLE heading walk — the document view needs
    both, and calling `render_body_html` + `extract_toc` separately walks the whole body twice
    (a full-body regex `finditer` + a `chart_extracted_spans` scan, ×2; on the 650 KB 10-K with
    501 headings that's a noticeable redundant cost; audit 2026-06-07). Pure CPU → call it via
    `asyncio.to_thread` from the route so it doesn't block the single-worker event loop."""
    walked = _walk_headings(body)
    return render_body_html(body, headings=walked), extract_toc(body, headings=walked)
