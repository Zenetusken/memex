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


def _force_split_oversized(unit: str, *, target_tokens: int) -> list[str]:
    """Split a unit too large for the sentence-splitter on line boundaries.

    A markdown table is one paragraph with no sentence boundaries, so it
    arrives here whole. Splitting on `\\n` keeps each row (and each list item)
    intact: consecutive lines are packed greedily into groups whose
    `_budget_word_count` stays within `target_tokens`. A single line that alone
    exceeds the budget (a pathological one-line table) is emitted whole rather
    than cut mid-row. Each returned group is a contiguous substring of `unit`,
    which keeps the caller's offset bookkeeping exact.
    """
    lines = unit.split("\n")
    groups: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for line in lines:
        lt = _budget_word_count(line)
        if buf and buf_tokens + lt > target_tokens:
            groups.append("\n".join(buf))
            buf = []
            buf_tokens = 0
        buf.append(line)
        buf_tokens += lt
    if buf:
        groups.append("\n".join(buf))
    return groups


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
        if pt > target_tokens:
            # Sentence-split oversized paragraphs.
            sentences = _SENTENCE_RE.split(p)
            for s in sentences:
                st = _budget_word_count(s)
                if st > max_tokens_per_chunk:
                    # A unit the sentence-splitter couldn't break (a table).
                    # Flush the pending window, then emit each line-split
                    # row-group as its OWN single-element window: `joined`
                    # then equals the group verbatim — a contiguous substring
                    # of `section` — so the re-locate loop's offsets stay exact.
                    if cur:
                        windows.append(cur)
                        cur = []
                        cur_tokens = 0
                    for group in _force_split_oversized(s, target_tokens=target_tokens):
                        windows.append([group])
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

    # Re-locate each window into the original section text so the
    # char_start/char_end refer to the source document.
    chunks: list[tuple[int, int, str]] = []
    cursor = 0
    for w in windows:
        joined = "\n\n".join(w)
        # Locate the first paragraph of the window in the source from `cursor`.
        first = w[0]
        idx = section.find(first, cursor)
        if idx < 0:
            idx = cursor
        char_start = section_offset + idx
        char_end = char_start + len(joined)
        cursor = idx + len(first)
        chunks.append((char_start, char_end, joined))
    return chunks


def chunk_document(doc: VaultDocument) -> list[Chunk]:
    """Produce the canonical chunk list for a vault document.

    Reads target window + overlap from `IndexSettings` (live, not at
    import time) so env overrides take effect. The function is still
    deterministic given fixed settings + body: same input + same
    settings ⇒ same chunks and chunk_ids, so re-indexing is idempotent.

    Falls back to module-level defaults (`TARGET_TOKENS`,
    `OVERLAP_TOKENS`) when settings haven't been initialised — keeps
    pure-function tests that call the chunker directly working without
    needing a `set_settings` fixture.
    """
    try:
        settings = get_settings()
        target = settings.index.chunk_target_tokens
        overlap = settings.index.chunk_overlap_tokens
    except Exception:
        target = TARGET_TOKENS
        overlap = OVERLAP_TOKENS

    title = doc.frontmatter.title or doc.ref.doc_id
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
            out.append(
                Chunk(
                    chunk_id=_stable_chunk_id(doc.ref.doc_id, text),
                    document_id=doc.ref.doc_id,
                    document_title=title,
                    text=text,
                    char_start=cs,
                    char_end=ce,
                    heading_path=heading_path,
                )
            )
    return out
