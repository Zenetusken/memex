"""Header-aware markdown chunker — see IMPLEMENTATION-PLAN.md §1.5 and §5.1.

Splits on markdown headings first (`##`, `###`, `####`), then on
paragraph boundaries, then by sentence to a target window. Each chunk
carries its heading path as metadata so citations can render
"Document title > 3.2 Methodology > 3.2.1 Cohort selection".

Token counting uses word-count as a rough proxy (real tokens are ~1.3x
words for English). When we adopt tiktoken or a model-specific
tokenizer this becomes more precise; for Phase 0 the approximation is
fine.

Target window + overlap come from `IndexSettings`
(`chunk_target_tokens`, `chunk_overlap_tokens`) so they can be tuned
per-rig without code edits. The module-level `TARGET_TOKENS` /
`OVERLAP_TOKENS` constants are kept as the default values for tests
and legacy callers; the live chunker reads settings on each call.
"""

from __future__ import annotations

import hashlib
import re

from memex.core.config import IndexSettings, get_settings
from memex.core.types import Chunk
from memex.vault.store import VaultDocument

# A heading line: 1–6 hashes, a space, then text.
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
# Conservative sentence boundary — splits on `.`, `!`, `?` followed by
# space and an uppercase letter or digit. Multilingual: the character
# class includes Latin-1 uppercase (À-Ö, Ø-Þ) + Œ + Ÿ so French
# (Élève, Ça, Êtes-vous, Œuvre), German (Ärger, Über), Spanish (Él,
# Único), Italian, and Portuguese sentence starts are recognised.
# ASCII A-Z + digit are first in the class for fast-path matching.
# Markdown's "paragraph" is two newlines.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞŒŸ])")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")

# Re-exported from core/text.py so existing callers keep working
# (`from memex.index.chunker import strip_chart_extracted_for_index`).
# The helper itself lives in core/ because both `index/` and
# `agents/` need it and the module-import direction forbids
# `agents/ → index/` (see CLAUDE.md).
from memex.core.text import (  # noqa: E402
    chart_extracted_spans,
    is_inside_any_span,
    strip_chart_extracted_for_index,
)

# Defaults — overridable per-call via IndexSettings. Kept as module
# constants so test helpers can reference them without importing the
# whole settings stack.
TARGET_TOKENS = IndexSettings.model_fields["chunk_target_tokens"].default
OVERLAP_TOKENS = IndexSettings.model_fields["chunk_overlap_tokens"].default
# Hard cap on a single chunk's budget, as a multiple of the target. A markdown
# table is one "paragraph" with no sentence boundaries, so the sentence-splitter
# can't break it — without this cap a 175-page 10-K's financial tables become
# single ~21K-char chunks that (a) blow the entity-extraction context window in
# enrich (vLLM rejects prompt+output > max-model-len → chunk_failed) and (b)
# exceed the reranker's attention window + the answer prompt's truncate budget.
# Units above the cap are force-split on line (row) boundaries — see
# `_force_split_oversized`.
MAX_CHUNK_MULTIPLIER = 3
# Hard cap on a single chunk's character length. A markdown table can be
# char-heavy but word-light (a financial 10-K table is ~21K chars but only
# ~493 budget-words), so the word-budget cap (`MAX_CHUNK_MULTIPLIER`) never
# fires and the table stays one oversized chunk. The char cap closes that
# gap: it matches the answer prompt's `truncate(1800)` so each table chunk is
# fully answer-visible, and it fits the reranker's attention window. Chart-
# extracted blocks are exempt (they're intentionally one chunk — see the
# `"[chart-extracted]" not in u` guard in `_split_section_into_chunks`).
MAX_CHUNK_CHARS = 1800

# A GFM table header: a pipe row followed by a delimiter row. The delimiter
# row is all of pipes / dashes / colons / whitespace AND contains at least
# one dash (so a plain `| a | b |` data row that happens to be all-punctuation
# doesn't match). Used by `_gfm_header` to repeat the column header onto each
# row-group when a table is force-split.
_GFM_DELIM_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def _word_count(s: str) -> int:
    return len(s.split())


def _budget_word_count(s: str) -> int:
    """Word count for chunker-budget purposes only.

    P3.3 v4 (audit 2026-05-22): excludes
    `[chart-extracted]...[/chart-extracted]` block contents so
    chart-OCR enrichment doesn't shift chunk boundaries away from
    the no-chart-OCR baseline.

    The chunker uses word count as a token-count proxy when deciding
    whether to close a chunk and start a new one (the
    `chunk_target_tokens` budget). Inline chart-extracted blocks
    (added by `parse/pipeline.py::_stitch_chart_extractions`) can be
    several hundred "words" — when a paragraph plus its post-figure
    chart block bumps the budget, the chunker closes the window
    early and pushes the next paragraph into a new chunk. The
    P3.3 v2/v3 eval data showed this is the dominant cause of the
    −6 prose-query regression on the CUDA deck (chunks shifted
    away from their paired prose).

    Counting chart-block words as ZERO for budget purposes keeps
    chunk boundaries at the no-chart-OCR baseline. The chart block
    still appears in the chunk's text (used by the agent at answer
    time), and FTS-side stripping (P3.3 v3, see fts_store.upsert)
    keeps the chart-numeric tokens out of BM25.
    """
    return _word_count(strip_chart_extracted_for_index(s))


def _heading_path_at(
    text: str,
    offset: int,
    *,
    chart_spans: list[tuple[int, int]] | None = None,
) -> list[str]:
    """The active heading stack at character offset `offset`.

    Walks every heading at or before `offset`, maintaining a stack
    keyed by hash count. Same chart-block-aware filter as
    `_split_into_sections` (P3.3 v7): `# H1`-style matches inside
    `[chart-extracted]` blocks are inert chart labels, not real
    document headings, and must not pollute the heading path of the
    chunks following the chart.

    `chart_spans` is an optional precomputed result of
    `chart_extracted_spans(text)`. Passing it lets the caller
    (`chunk_document`) compute spans ONCE per document instead of
    re-scanning the body on every heading-path call (post-v7 audit
    perf fix 2026-05-23). When omitted, recomputed locally for
    standalone callers + tests.
    """
    if chart_spans is None:
        chart_spans = chart_extracted_spans(text)
    stack: dict[int, str] = {}
    for m in _HEADING_RE.finditer(text):
        if m.start() > offset:
            break
        if is_inside_any_span(m.start(), chart_spans):
            continue
        level = len(m.group(1))
        # Drop any deeper headings — new heading at this level supersedes
        for deeper in [k for k in stack if k >= level]:
            del stack[deeper]
        stack[level] = m.group(2).strip()
    return [stack[k] for k in sorted(stack)]


def _stable_chunk_id(doc_id: str, text: str) -> str:
    # Content-addressing, not security — sha1 is fine and fast. The flag
    # tells hashlib (and bandit) this isn't a cryptographic use.
    digest = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
    return f"{doc_id}#{digest}"


def _split_into_sections(
    body: str,
    *,
    chart_spans: list[tuple[int, int]] | None = None,
) -> list[tuple[int, str]]:
    """Yield (start_offset, section_text) where section_text is the body
    between successive headings (or top-of-doc to first heading).

    P3.3 v7 (2026-05-23): heading-detection skips `# H1`-style matches
    that fall INSIDE `[chart-extracted]...[/chart-extracted]` blocks.
    Nemotron-Parse emits H1 labels for chart-figure-section names
    (e.g. nvmath-python's `# Minimal lead-time`, `# Inter-operability`)
    which the chunker would otherwise treat as document-section
    boundaries, splitting a single chart block across multiple chunks
    (the eval trace showed only the LAST principle reached the
    reranker's top-5).

    `chart_spans` is an optional precomputed result of
    `chart_extracted_spans(body)`. Passing it lets the caller
    (`chunk_document`) share spans with `_heading_path_at` — see
    that function's docstring + the post-v7 audit perf fix.
    """
    if chart_spans is None:
        chart_spans = chart_extracted_spans(body)
    headings = [
        m for m in _HEADING_RE.finditer(body) if not is_inside_any_span(m.start(), chart_spans)
    ]
    if not headings:
        return [(0, body)]
    sections: list[tuple[int, str]] = []
    # Preamble before first heading
    if headings[0].start() > 0:
        sections.append((0, body[: headings[0].start()]))
    for i, m in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        sections.append((m.start(), body[m.start() : end]))
    return sections


def _gfm_header(unit: str) -> str | None:
    """Return a GFM table's header block (header row + delimiter row) or `None`.

    A GitHub-flavoured-markdown table opens with a pipe row (`| A | B |`)
    followed by a delimiter row (`|---|---|`). When a table is force-split,
    every row-group past the first loses its column headers (they were only on
    the table's first line) — `| 2,345 | 1,890 |` with no idea what the columns
    mean. `_force_split_oversized` re-prepends this header to each later group
    so every table chunk is a valid, self-describing standalone GFM table.

    Detection: the unit's first two non-blank lines must be a pipe row (starts
    and ends with `|` after stripping) followed by a delimiter row (matches
    `_GFM_DELIM_RE` AND contains a dash). Returns the two lines joined by `\\n`;
    `None` for anything that isn't a GFM table (a non-pipe list, a one-line
    table, a chart block) so no synthetic header is prepended there.
    """
    non_blank = [ln for ln in unit.split("\n") if ln.strip()]
    if len(non_blank) < 2:
        return None
    first, second = non_blank[0], non_blank[1]
    fs = first.strip()
    if not (fs.startswith("|") and fs.endswith("|")):
        return None
    if "-" not in second or not _GFM_DELIM_RE.match(second):
        return None
    return first + "\n" + second


def _force_split_oversized(
    unit: str,
    *,
    target_tokens: int,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[str]:
    """Split a unit too large for the sentence-splitter on line boundaries.

    A markdown table is one paragraph with no sentence boundaries, so it
    arrives here whole. Splitting on `\\n` keeps each row (and each list item)
    intact: consecutive lines are packed greedily into groups, flushing when
    adding the next line would exceed EITHER `target_tokens` (word budget) OR
    `max_chars` (character budget — the load-bearing trigger for char-heavy /
    word-light financial tables). A single line that alone exceeds either
    budget (a pathological one-line table) is emitted whole rather than cut
    mid-row.

    GFM-table awareness: if `unit` opens with a GFM header (`_gfm_header`),
    that header block is prepended to every group AFTER the first (the first
    group already starts with it), so each row-group is a self-describing
    standalone table. The returned group text therefore is NOT necessarily a
    contiguous substring of `unit` — the caller (`_split_section_into_chunks`)
    handles offsets for the source-row portion explicitly. The group's source
    rows (group text minus any synthetic header) ARE contiguous in `unit`.
    """
    header = _gfm_header(unit)
    lines = unit.split("\n")
    row_groups: list[str] = []  # the source rows for each group (no synthetic header)
    buf: list[str] = []
    buf_tokens = 0
    buf_chars = 0
    for line in lines:
        lt = _budget_word_count(line)
        lc = len(line)
        # +1 for the joining "\n" once the buffer is non-empty.
        added_chars = lc + (1 if buf else 0)
        if buf and (buf_tokens + lt > target_tokens or buf_chars + added_chars > max_chars):
            row_groups.append("\n".join(buf))
            buf = []
            buf_tokens = 0
            buf_chars = 0
            added_chars = lc
        buf.append(line)
        buf_tokens += lt
        buf_chars += added_chars
    if buf:
        row_groups.append("\n".join(buf))

    if header is None:
        return row_groups
    # Prepend the synthetic header to every group after the first (the first
    # group's rows already begin with the header lines).
    return [rows if i == 0 else header + "\n" + rows for i, rows in enumerate(row_groups)]


def _split_section_into_chunks(
    section: str,
    section_offset: int,
    *,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    max_tokens_per_chunk: int | None = None,
) -> list[tuple[int, int, str]]:
    """Within a section, split into ~target_tokens windows with overlap.

    Returns a list of (char_start, char_end, text) relative to the
    enclosing document. `target_tokens` and `overlap_tokens` default
    to the module-level constants (which mirror `IndexSettings`
    defaults); callers via `chunk_document` thread the live settings.

    A unit whose own budget exceeds `max_tokens_per_chunk` (default
    `target_tokens * MAX_CHUNK_MULTIPLIER`) — a markdown table, which the
    sentence-splitter can't break — is force-split on line/row boundaries
    so no chunk blows the reranker window or the enrich context budget.
    """
    paragraphs = [p.strip() for p in _PARAGRAPH_RE.split(section) if p.strip()]
    if not paragraphs:
        return []
    if max_tokens_per_chunk is None:
        max_tokens_per_chunk = target_tokens * MAX_CHUNK_MULTIPLIER

    def _is_oversized(u: str) -> bool:
        # Word-budget oversize (a table the sentence-splitter can't break) OR
        # char-budget oversize (a char-heavy / word-light table that slips
        # under the word cap — e.g. a 21K-char/493-word 10-K table). The char
        # trigger never fires on a chart-extracted block: those are
        # intentionally one chunk, and `_budget_word_count` already zeroes
        # them so the word trigger can't fire either.
        if _budget_word_count(u) > max_tokens_per_chunk:
            return True
        return len(u) > MAX_CHUNK_CHARS and "[chart-extracted]" not in u

    # The output chunk tuples, built incrementally. Prose windows accumulate in
    # `windows` and are re-located against `section` (byte-identical to the
    # pre-table-fix behaviour); when an oversized unit is force-split, the
    # pending prose windows are flushed first, then the force-split row-groups
    # are appended as explicit `(cs, ce, text)` tuples (a header-repeated
    # group's text is NOT a contiguous substring of `section`, so it can't go
    # through the generic re-locate). The two paths share one `cursor`.
    chunks: list[tuple[int, int, str]] = []
    cursor = 0

    def _flush_windows(windows: list[list[str]]) -> None:
        # Re-locate each pending prose window into the source `section` so its
        # char_start/char_end refer to the enclosing document. Identical to the
        # original re-locate loop; only its scope changed (now flushable
        # mid-stream so force-split chunks can be interleaved in source order).
        nonlocal cursor
        for w in windows:
            joined = "\n\n".join(w)
            first = w[0]
            idx = section.find(first, cursor)
            if idx < 0:
                idx = cursor
            char_start = section_offset + idx
            char_end = char_start + len(joined)
            cursor = idx + len(first)
            chunks.append((char_start, char_end, joined))

    def _emit_force_split(unit: str) -> None:
        # Force-split an oversized unit on line/row boundaries and emit each
        # group as an explicit chunk tuple. For each group the ROWS (group text
        # minus any synthetic GFM header) ARE a contiguous substring of the
        # source, so they locate exactly; the synthetic header (groups >0) is
        # the only text outside `[char_start, char_end)`. This relaxes the
        # offset round-trip to the rows portion for header-repeated chunks.
        nonlocal cursor
        header = _gfm_header(unit)
        for i, group in enumerate(_force_split_oversized(unit, target_tokens=target_tokens)):
            # Group 0's text already begins with the (real, source-contiguous)
            # header — it is wholly contiguous in `section`. Groups >0 carry a
            # SYNTHETIC header prefix; strip it to recover the source rows.
            rows = group
            if header is not None and i > 0:
                rows = group[len(header) + 1 :]
            idx = section.find(rows, cursor)
            if idx < 0:
                idx = cursor
            char_start = section_offset + idx
            char_end = char_start + len(rows)
            cursor = idx + len(rows)
            chunks.append((char_start, char_end, group))

    # P3.3 v4: use `_budget_word_count` for all chunk-size decisions
    # (closes early-or-not, paragraph-oversize check). Chart-extracted
    # blocks contribute zero to the budget so chunk boundaries match
    # the no-chart-OCR baseline. The full paragraph text (including
    # chart blocks) is still appended to `cur` and ends up in the
    # final chunk text.
    windows: list[list[str]] = []
    cur: list[str] = []
    cur_tokens = 0
    for p in paragraphs:
        pt = _budget_word_count(p)
        if pt > target_tokens or _is_oversized(p):
            # Sentence-split oversized paragraphs.
            sentences = _SENTENCE_RE.split(p)
            for s in sentences:
                st = _budget_word_count(s)
                if _is_oversized(s):
                    # A unit the sentence-splitter couldn't break (a table).
                    # Flush the pending prose windows + in-progress window
                    # (preserving source order + the shared cursor), then emit
                    # the line-split row-groups directly with explicit offsets.
                    if cur:
                        windows.append(cur)
                        cur = []
                        cur_tokens = 0
                    _flush_windows(windows)
                    windows = []
                    _emit_force_split(s)
                    continue
                if cur and cur_tokens + st > target_tokens:
                    windows.append(cur)
                    overlap_words = " ".join(cur).split()[-overlap_tokens:]
                    cur = [" ".join(overlap_words)] if overlap_words else []
                    cur_tokens = _budget_word_count(cur[0]) if cur else 0
                cur.append(s)
                cur_tokens += st
        else:
            if cur and cur_tokens + pt > target_tokens:
                windows.append(cur)
                overlap_words = " ".join(cur).split()[-overlap_tokens:]
                cur = [" ".join(overlap_words)] if overlap_words else []
                cur_tokens = _budget_word_count(cur[0]) if cur else 0
            cur.append(p)
            cur_tokens += pt
    if cur:
        windows.append(cur)
    _flush_windows(windows)
    return chunks


def page_intervals(page_char_counts: list[tuple[int, int]]) -> list[tuple[int, int, int]] | None:
    """Cumulative `(page_no, char_start, char_end)` intervals against the joined
    body, computed from a `[(page_no, char_count), ...]` list (one tuple per page,
    in reading order). The body is `"\\n\\n".join(per_page_md)` — every parse path
    uses that delimiter — so each page interval starts at the cumulative sum of
    previous pages' char_counts + 2-char delimiters. Returns `None` when every
    page's char_count is 0 (the "page mapping unavailable" sentinel — legacy
    manifests written before `PageDecision.char_count` existed) so the caller
    knows to skip page attribution rather than collapsing every chunk to page 1.

    Pure + deterministic — pinned by unit tests.
    """
    if not page_char_counts or all(c == 0 for _, c in page_char_counts):
        return None
    intervals: list[tuple[int, int, int]] = []
    cursor = 0
    for i, (page_no, cc) in enumerate(page_char_counts):
        start = cursor
        end = start + cc
        intervals.append((page_no, start, end))
        cursor = end
        if i < len(page_char_counts) - 1:
            cursor += 2  # "\n\n" delimiter
    return intervals


def _page_for_offset(intervals: list[tuple[int, int, int]], offset: int) -> int | None:
    """Find the page whose interval contains `offset`. Linear scan — the
    intervals list is short (one entry per page; a typical doc has 10-100,
    a 49-page deck is the long tail). A chunk just past the LAST page's
    `end` (post-stitch drift; chart-OCR / table linearization can append
    content after the per-page char_counts were recorded) attributes to
    the LAST page rather than `None`."""
    if not intervals:
        return None
    for page_no, start, end in intervals:
        if start <= offset < end:
            return page_no
    # Past the last page's recorded end — attribute to the last page (drift
    # from post-stitch transforms; documented limitation).
    return intervals[-1][0]


def _time_range_for_offset(
    intervals: list[tuple[int, int, float, float]], offset: int
) -> tuple[float, float] | None:
    """The audio analogue of `_page_for_offset` (ADR-0017): find the transcript SEGMENT whose
    char-span `(char_start, char_end)` contains `offset` and return its `(start_s, end_s)` GLOBAL
    time range. A chunk past the last segment's end attributes to the last segment (post-finalize
    drift — navigation-grade, like the page mapping). `None` when there are no segments."""
    if not intervals:
        return None
    for char_start, char_end, start_s, end_s in intervals:
        if char_start <= offset < char_end:
            return (start_s, end_s)
    last = intervals[-1]
    return (last[2], last[3])


def chunk_document(
    doc: VaultDocument,
    *,
    page_char_counts: list[tuple[int, int]] | None = None,
    exact_page_intervals: list[tuple[int, int, int]] | None = None,
    segment_intervals: list[tuple[int, int, float, float]] | None = None,
) -> list[Chunk]:
    """Produce the canonical chunk list for a vault document.

    Reads target window + overlap from `IndexSettings` (live, not at
    import time) so env overrides take effect. The function is still
    deterministic given fixed settings + body: same input + same
    settings ⇒ same chunks and chunk_ids, so re-indexing is idempotent.

    Falls back to module-level defaults (`TARGET_TOKENS`,
    `OVERLAP_TOKENS`) when settings haven't been initialised — keeps
    pure-function tests that call the chunker directly working without
    needing a `set_settings` fixture.

    `page_char_counts` is optional `[(page_no, char_count), ...]` from
    `ParseStage.pages` — when provided AND any page has a non-zero
    char_count, each chunk's `char_start` is located by a linear scan over
    the page intervals and `Chunk.page` is populated (drives the webui's
    click-source→jump-to-PDF-page UX). When omitted or all-zero (legacy
    manifests), `Chunk.page` stays `None` and the webui falls back to
    section-only anchors. HARD-gate-neutral (a derived navigation
    metadata field; never alters retrieval or grounding).

    `exact_page_intervals` (companion arc-3, citation-grade) is the
    precise alternative: `[(page_no, char_start, char_end), ...]` already
    measured against THIS body (via the page-boundary marker round-trip
    in `index_document`), so each chunk's page is read directly without the
    `char_count`→interval derivation that assumes a uniform 2-char page
    delimiter and drifts on figure-heavy decks. Takes precedence over
    `page_char_counts`; same nav-grade / HARD-gate-neutral contract.

    `segment_intervals` is the audio analogue (ADR-0017): optional
    `[(char_start, char_end, start_s, end_s), ...]` from `ParseStage.segments`
    — when provided, each chunk's `Chunk.time_range` is set to the GLOBAL
    `(start_s, end_s)` of the transcript segment its `char_start` falls in
    (drives the webui's `[mm:ss]` source label). `None` for non-audio docs;
    same navigation-grade / HARD-gate-neutral contract as `page`.
    """
    try:
        settings = get_settings()
        target = settings.index.chunk_target_tokens
        overlap = settings.index.chunk_overlap_tokens
    except Exception:
        target = TARGET_TOKENS
        overlap = OVERLAP_TOKENS

    title = doc.frontmatter.title or doc.ref.doc_id
    # `exact_page_intervals` (companion arc-3 citation-grade map) — `(page_no, char_start, char_end)`
    # measured against THIS exact body — takes precedence over the nav-grade `char_count`-derived
    # intervals (which assume a uniform 2-char page delimiter and drift on figure-heavy decks).
    intervals = (
        exact_page_intervals
        if exact_page_intervals is not None
        else (page_intervals(page_char_counts) if page_char_counts else None)
    )
    # Post-audit perf fix (2026-05-23): compute chart-extracted spans
    # ONCE per document instead of N+1 times (once in
    # `_split_into_sections`, then once per `_heading_path_at` call
    # inside the inner loop). On a 200 KB slide deck at the default
    # chunk size, that's ~50-150× over-scan saved per document parse.
    chart_spans = chart_extracted_spans(doc.body)
    out: list[Chunk] = []
    for section_offset, section_text in _split_into_sections(doc.body, chart_spans=chart_spans):
        for cs, ce, text in _split_section_into_chunks(
            section_text,
            section_offset,
            target_tokens=target,
            overlap_tokens=overlap,
        ):
            heading_path = _heading_path_at(doc.body, cs, chart_spans=chart_spans)
            page = _page_for_offset(intervals, cs) if intervals else None
            time_range = (
                _time_range_for_offset(segment_intervals, cs) if segment_intervals else None
            )
            out.append(
                Chunk(
                    chunk_id=_stable_chunk_id(doc.ref.doc_id, text),
                    document_id=doc.ref.doc_id,
                    document_title=title,
                    text=text,
                    page=page,
                    char_start=cs,
                    char_end=ce,
                    heading_path=heading_path,
                    time_range=time_range,
                )
            )
    return out
