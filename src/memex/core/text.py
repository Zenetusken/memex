"""Cross-module text-manipulation helpers.

Lives in `core/` so both `index/` (chunker, fts_store) and `agents/`
(answer / verify prompt rendering) can import the same helper without
violating the documented module-import direction (`agents/ → core/`,
no `agents/ → index/`).
"""

from __future__ import annotations

import re

# Matches a `[chart-extracted]...[/chart-extracted]` block emitted by
# the P3.3 chart-OCR stitch step (see parse/pipeline.py::_stitch_chart_
# extractions). The dot-all flag lets `.` span newlines so multi-line
# extracted tables match.
_CHART_EXTRACTED_RE = re.compile(
    r"\[chart-extracted\].*?\[/chart-extracted\]",
    flags=re.DOTALL,
)

# Matches an orphan `[chart-extracted]` opener with no matching closer
# until end-of-text. Used by `chart_extracted_spans` as a fall-through:
# if a chunk got truncated mid-block (or a user-edited vault doc dropped
# the close tag), the section-splitting filter still needs to know the
# orphan range so inert chart-figure `# H1` labels don't get treated as
# document headings. The closer-only regex is a paired safety net.
_CHART_EXTRACTED_OPEN_RE = re.compile(r"\[chart-extracted\]", flags=re.DOTALL)
_CHART_EXTRACTED_CLOSE_RE = re.compile(r"\[/chart-extracted\]", flags=re.DOTALL)


def strip_chart_extracted_for_index(text: str) -> str:
    """Remove `[chart-extracted]...[/chart-extracted]` blocks.

    Used at the **index layer** (`index.fts_store::upsert`) only: strips
    the FTS body so chart-extracted dense numerical tokens (years,
    percentages, raw values) don't inflate BM25 term frequency for
    unrelated queries.

    Previously also used at the agent layer (P3.3 v3, v5, v6) to keep
    long chart-block headers from eating the answer/verify prompt's
    `truncate(N)` budget. The P3.3 v7 fix (commit a9e8326) removed the
    agent-layer strips: the v7 chart-OCR backend emits compact markdown
    (~50-150 chars per block) that fits comfortably alongside prose, and
    the strips were actively masking chart-content answering capability.

    The stored chunk text (LanceDB + chunks_meta + vault markdown)
    is preserved unchanged so display / MCP / future trace tooling
    can still surface the chart-OCR content if needed.

    Idempotent — text without chart blocks passes through unchanged.
    """
    return _CHART_EXTRACTED_RE.sub("", text)


def chart_extracted_spans(text: str) -> list[tuple[int, int]]:
    """Return `(start, end)` char offsets of each
    `[chart-extracted]...[/chart-extracted]` block in `text`.

    Used by `index.chunker._split_into_sections` so heading-detection
    can skip `# H1` lines that appear INSIDE chart-extracted blocks.
    Nemotron-Parse emits H1 prefixes for chart-figure-section labels
    (e.g. nvmath-python's 4 design principles as `# Minimal lead-time`,
    `# Inter-operability`, etc.) which the chunker would otherwise
    treat as document-section boundaries — splitting a single chart
    block across multiple chunks. Established by the P3.3 v7 chunker
    fix (2026-05-23) after chart-OCR validation on the nvmath-python
    architecture figure revealed only the last principle was reaching
    the reranker's top-5.

    Truncation-tolerant: if a `[chart-extracted]` opener has no matching
    closer (e.g. mid-chunk truncation, user-edited vault) the orphan
    span extends to end-of-text. A `[/chart-extracted]` closer with no
    opener extends from start-of-text to the closer position. Both
    defensive cases prevent silent regressions of the P3.3 v7 chunker
    fix the original audit (2026-05-23 post-v7) flagged.

    Returns the inclusive-exclusive `(start, end)` of each span in
    `text`. Empty list if no chart-block tags present. Order-stable
    (sorted by start offset).
    """
    spans: list[tuple[int, int]] = list(
        (m.start(), m.end()) for m in _CHART_EXTRACTED_RE.finditer(text)
    )
    consumed_open: set[int] = {start for start, _ in spans}
    consumed_close: set[int] = {end for _, end in spans}

    for m in _CHART_EXTRACTED_OPEN_RE.finditer(text):
        if m.start() not in consumed_open:
            spans.append((m.start(), len(text)))
    for m in _CHART_EXTRACTED_CLOSE_RE.finditer(text):
        if m.end() not in consumed_close:
            spans.append((0, m.end()))

    spans.sort(key=lambda s: s[0])
    return spans


def is_inside_any_span(offset: int, spans: list[tuple[int, int]]) -> bool:
    """True iff `offset` falls strictly within one of `spans` (where each
    span is `(start, end)`, end-exclusive). Used with
    `chart_extracted_spans` to filter heading matches inside chart
    blocks. O(len(spans)); fine for typical doc sizes (<100 chart
    blocks)."""
    return any(start <= offset < end for start, end in spans)


# ----- Table-RAG linearization helpers (Phase 1) -----------------------------
#
# A `[table-rows]...[/table-rows]` block is the markdown-KV linearization of a
# GFM table emitted by `parse/table_linearize.py::linearize_gfm_tables`. The
# 3-channel contract (see docs/specs/table-rag.md) routes each representation:
#   - `.md` on disk keeps BOTH the raw GFM table AND the `[table-rows]` block;
#   - dense chunks see them as SEPARATE chunks (the block is a distinct
#     paragraph unit);
#   - the FTS/BM25 body sees ONLY the `[table-rows]` block — the raw GFM table
#     it supersedes is stripped (`strip_superseded_gfm_tables`), the inverse of
#     the chart strip, so the KV rows (not the raw table) carry the BM25 signal.

_VALUE_CHARS: tuple[str, ...] = ("$", "%")


def looks_like_value(s: str) -> bool:
    """True if *s* reads as a numeric value (contains a digit or ``$``/``%``).

    Promoted verbatim from `parse/docling_tables._looks_like_value` so the
    table-linearizer (which lives in `parse/`) and the GFM-table strip (which
    lives here in `core/`, called from `index/fts_store`) share ONE heuristic
    without `index/ → parse/` or `core/ → parse/` import edges. The
    `parse/docling_tables` copy operates on `TableCell` flags and stays put;
    this one is the pure-text predicate.
    """
    if any(ch.isdigit() for ch in s):
        return True
    return any(ch in s for ch in _VALUE_CHARS)


# Matches a `[table-rows]...[/table-rows]` block (the table-linearization KV
# payload). Dot-all so the multi-line row body matches.
_TABLE_ROWS_RE = re.compile(
    r"\[table-rows\].*?\[/table-rows\]",
    flags=re.DOTALL,
)
_TABLE_ROWS_OPEN_RE = re.compile(r"\[table-rows\]", flags=re.DOTALL)
_TABLE_ROWS_CLOSE_RE = re.compile(r"\[/table-rows\]", flags=re.DOTALL)


def table_rows_spans(text: str) -> list[tuple[int, int]]:
    """Return `(start, end)` char offsets of each
    `[table-rows]...[/table-rows]` block in *text*.

    Mirrors `chart_extracted_spans` exactly — including the orphan-opener /
    orphan-closer truncation tolerance — because the chunker's char-split
    (`MAX_CHUNK_CHARS`) WILL split an oversized `[table-rows]` block across
    chunks (unlike chart blocks, which are exempt). A split half therefore
    carries only an opener (extends to end-of-text) or only a closer (extends
    from start-of-text), and any consumer that needs to recognise a partial
    block still gets the span. Order-stable (sorted by start offset).
    """
    spans: list[tuple[int, int]] = list((m.start(), m.end()) for m in _TABLE_ROWS_RE.finditer(text))
    consumed_open: set[int] = {start for start, _ in spans}
    consumed_close: set[int] = {end for _, end in spans}

    for m in _TABLE_ROWS_OPEN_RE.finditer(text):
        if m.start() not in consumed_open:
            spans.append((m.start(), len(text)))
    for m in _TABLE_ROWS_CLOSE_RE.finditer(text):
        if m.end() not in consumed_close:
            spans.append((0, m.end()))

    spans.sort(key=lambda s: s[0])
    return spans


# A GFM table-shape line group: a header pipe-row, a delimiter row, then one or
# more data pipe-rows. The delimiter row is the disambiguator (a run of
# pipes / dashes / colons / whitespace with at least one dash), mirroring
# `index.chunker._GFM_DELIM_RE`. Matched line-anchored + multiline so a table
# embedded in prose is found at its own line boundaries.
_GFM_TABLE_RE = re.compile(
    r"^[ \t]*\|.*\|[ \t]*\n"  # header pipe-row
    r"[ \t]*\|?[ \t:|-]*-[ \t:|-]*\|?[ \t]*\n"  # delimiter row (>=1 dash)
    r"(?:[ \t]*\|.*\|[ \t]*\n?)+",  # >=1 data pipe-row
    flags=re.MULTILINE,
)


def strip_superseded_gfm_tables(text: str) -> str:
    """Remove a GFM table that is immediately followed by a `[table-rows]`
    block (the table-linearization payload).

    Used at the **index layer** (`index.fts_store::upsert`) only, composed
    with `strip_chart_extracted_for_index`: the `[table-rows]` KV rows carry
    the BM25 signal, so leaving the raw GFM table in the FTS body would
    double-count every value token (term-frequency inflation — the same class
    of bug `strip_chart_extracted_for_index` fixes for chart blocks). This is
    its inverse: chart-strip drops the derived block and keeps the source;
    table-strip drops the *source* table and keeps the derived `[table-rows]`.

    Only strips a GFM table when a `[table-rows]` block follows it across an
    optional blank line — an un-linearized table (no following block) is left
    intact so its raw value tokens still reach BM25. Idempotent: text with no
    superseded table passes through unchanged.
    """

    def _replace(m: re.Match[str]) -> str:
        rest = text[m.end() :]
        # The linearizer separates the table from its block by a blank line
        # (`\n\n`); the table regex consumes a trailing newline, so tolerate
        # any further leading whitespace/newlines before the opener.
        if _TABLE_ROWS_OPEN_RE.match(rest.lstrip("\n \t") if rest else ""):
            # Drop the raw table but keep the structural newline so the
            # `[table-rows]` block stays a distinct paragraph in the body.
            return ""
        return m.group(0)

    return _GFM_TABLE_RE.sub(_replace, text)


# Matches a Markdown ATX heading line (1-6 `#` followed by space and
# heading text). Same shape as `index.chunker._HEADING_RE`; lifted
# here so `enrich.citations` can use it for section-anchor discovery
# without violating the `enrich/ → index/` import direction.
#
# Scoped to horizontal whitespace (`[ \t]`) rather than `\s` because
# the latter matches `\n` too, which would cause an empty heading like
# `## ` (no text) + blank line + `Prose.` to silently match as `##
# Prose.` (the regex `\s+(.+?)\s*$` swallows the newline and grabs
# the next-line's content). Post-P4.1 audit (2026-05-23 webui test
# pin) confirmed this was an actual hazard. The chunker uses the same
# regex via `index.chunker._HEADING_RE`; the tightening applies
# uniformly so empty-text headings no longer steal subsequent
# paragraphs.
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)


def extract_heading_texts(body: str, *, skip_chart_blocks: bool = True) -> list[str]:
    """Return the heading text (sans `#` prefix) of every Markdown
    heading in `body`, in document order.

    Used by the P4.1 wikilink section-anchor writer
    (`enrich.citations::insert_wikilinks`) to discover which headings
    the target document has, so the citation-insertion step can emit
    `[[doc#section]]` when the citation context references a
    specific section.

    `skip_chart_blocks=True` (default) filters out the inert
    `# H1` labels Nemotron-Parse emits INSIDE `[chart-extracted]`
    blocks (same defense as the P3.3 v7 chunker fix — see
    `chart_extracted_spans` above). Set to `False` only if you
    explicitly want chart-figure labels in the heading list
    (almost no caller should).

    Headings are stripped of leading/trailing whitespace. Empty
    or all-whitespace headings are skipped. Returns headings in
    document order; duplicates ARE preserved (a doc with two
    `## Methods` sections returns `["Methods", "Methods"]`) so
    callers that care about uniqueness can dedupe themselves.
    """
    spans = chart_extracted_spans(body) if skip_chart_blocks else []
    out: list[str] = []
    for m in _MARKDOWN_HEADING_RE.finditer(body):
        if is_inside_any_span(m.start(), spans):
            continue
        text = m.group(2).strip()
        if text:
            out.append(text)
    return out
