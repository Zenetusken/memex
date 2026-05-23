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
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
# Conservative sentence boundary — splits on `.`, `!`, `?` followed by space
# and uppercase or digit. Markdown's "paragraph" is two newlines.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")

# Matches a `[chart-extracted]...[/chart-extracted]` block emitted by
# the P3.3 chart-OCR stitch step (see parse/pipeline.py::_stitch_chart_
# extractions). The dot-all flag lets `.` span newlines so multi-line
# extracted tables match.
_CHART_EXTRACTED_RE = re.compile(
    r"\[chart-extracted\].*?\[/chart-extracted\]",
    flags=re.DOTALL,
)


def strip_chart_extracted_for_index(text: str) -> str:
    """Remove `[chart-extracted]...[/chart-extracted]` blocks from
    text destined for BM25 + dense-embedding indexing.

    Background (P3.3 v3 follow-up, audit 2026-05-21):
    The P3.3 chart-OCR pipeline injects `[chart-extracted]` blocks
    containing DePlot's extracted chart values (years, percentages,
    raw numerics). When those blocks land inside a chunk's text,
    they:
    - Inflate BM25 term frequency for dense numerical tokens that
      happen to appear in unrelated queries (a chart that lists
      "2014, 2018, 2022" inadvertently boosts a chunk's rank for
      ANY query mentioning those years), AND
    - Perturb the dense-embedding centroid by shifting it toward the
      numerical-table region of embedding space.

    The empirical effect on the CUDA deck: +1 chart-numeric win (Q21
    where the chart-extracted "1.6x" matched) but −6 prose
    regressions where chart-bearing chunks outranked the prose
    chunks the agent actually needed for queries about unrelated
    topics. Net −5 ANS vs the no-chart-OCR baseline.

    The fix: strip the blocks BEFORE indexing so the retrieval
    layer sees the chunk's prose only. The chunk's `Chunk.text`
    field is left intact, so when the chunk IS retrieved (via
    surrounding prose), the agent still sees the chart-extracted
    block in its context window. Chart-numeric queries still work
    when the prose around the chart matches the query (e.g., Q21's
    "NVRTC -minimal vs -O3" matches the slide title even with the
    chart body stripped from the index).

    The strip is also a defence in depth: future audits won't have
    to worry about adversarial chart content (a chart that
    extracts to e.g. medical metrics on an architecture deck)
    skewing retrieval.

    Idempotent — text without chart blocks passes through unchanged.
    Leading/trailing whitespace from the placeholder is preserved
    intentionally; the indexer doesn't care about exact whitespace.
    """
    return _CHART_EXTRACTED_RE.sub("", text)

# Defaults — overridable per-call via IndexSettings. Kept as module
# constants so test helpers can reference them without importing the
# whole settings stack.
TARGET_TOKENS = IndexSettings.model_fields["chunk_target_tokens"].default
OVERLAP_TOKENS = IndexSettings.model_fields["chunk_overlap_tokens"].default


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


def _heading_path_at(text: str, offset: int) -> list[str]:
    """The active heading stack at character offset `offset`.

    Walks every heading at or before `offset`, maintaining a stack
    keyed by hash count.
    """
    stack: dict[int, str] = {}
    for m in _HEADING_RE.finditer(text):
        if m.start() > offset:
            break
        level = len(m.group(1))
        # Drop any deeper headings — new heading at this level supersedes
        for deeper in [k for k in stack if k >= level]:
            del stack[deeper]
        stack[level] = m.group(2).strip()
    return [stack[k] for k in sorted(stack)]


def _stable_chunk_id(doc_id: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}#{digest}"


def _split_into_sections(body: str) -> list[tuple[int, str]]:
    """Yield (start_offset, section_text) where section_text is the body
    between successive headings (or top-of-doc to first heading)."""
    headings = list(_HEADING_RE.finditer(body))
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


def _split_section_into_chunks(
    section: str,
    section_offset: int,
    *,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[tuple[int, int, str]]:
    """Within a section, split into ~target_tokens windows with overlap.

    Returns a list of (char_start, char_end, text) relative to the
    enclosing document. `target_tokens` and `overlap_tokens` default
    to the module-level constants (which mirror `IndexSettings`
    defaults); callers via `chunk_document` thread the live settings.
    """
    paragraphs = [
        p.strip() for p in _PARAGRAPH_RE.split(section) if p.strip()
    ]
    if not paragraphs:
        return []

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
    out: list[Chunk] = []
    for section_offset, section_text in _split_into_sections(doc.body):
        for cs, ce, text in _split_section_into_chunks(
            section_text,
            section_offset,
            target_tokens=target,
            overlap_tokens=overlap,
        ):
            heading_path = _heading_path_at(doc.body, cs)
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
