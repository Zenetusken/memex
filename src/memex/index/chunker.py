"""Header-aware markdown chunker — see IMPLEMENTATION-PLAN.md §1.5 and §5.1.

Splits on markdown headings first (`##`, `###`, `####`), then on
paragraph boundaries, then by sentence to a target window. Each chunk
carries its heading path as metadata so citations can render
"Document title > 3.2 Methodology > 3.2.1 Cohort selection".

Token counting uses word-count as a rough proxy (real tokens are ~1.3x
words for English). When we adopt tiktoken or a model-specific
tokenizer this becomes more precise; for Phase 0 the approximation is
fine.
"""

from __future__ import annotations

import hashlib
import re

from memex.core.types import Chunk
from memex.vault.store import VaultDocument

# A heading line: 1–6 hashes, a space, then text.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
# Conservative sentence boundary — splits on `.`, `!`, `?` followed by space
# and uppercase or digit. Markdown's "paragraph" is two newlines.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")

TARGET_TOKENS = 600
OVERLAP_TOKENS = 100


def _word_count(s: str) -> int:
    return len(s.split())


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
) -> list[tuple[int, int, str]]:
    """Within a section, split into ~TARGET_TOKENS windows with overlap.

    Returns a list of (char_start, char_end, text) relative to the
    enclosing document.
    """
    paragraphs = [
        p.strip() for p in _PARAGRAPH_RE.split(section) if p.strip()
    ]
    if not paragraphs:
        return []

    windows: list[list[str]] = []
    cur: list[str] = []
    cur_tokens = 0
    for p in paragraphs:
        pt = _word_count(p)
        if pt > TARGET_TOKENS:
            # Sentence-split oversized paragraphs.
            sentences = _SENTENCE_RE.split(p)
            for s in sentences:
                st = _word_count(s)
                if cur and cur_tokens + st > TARGET_TOKENS:
                    windows.append(cur)
                    overlap_words = " ".join(cur).split()[-OVERLAP_TOKENS:]
                    cur = [" ".join(overlap_words)] if overlap_words else []
                    cur_tokens = _word_count(cur[0]) if cur else 0
                cur.append(s)
                cur_tokens += st
        else:
            if cur and cur_tokens + pt > TARGET_TOKENS:
                windows.append(cur)
                overlap_words = " ".join(cur).split()[-OVERLAP_TOKENS:]
                cur = [" ".join(overlap_words)] if overlap_words else []
                cur_tokens = _word_count(cur[0]) if cur else 0
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

    Pure function — no model calls, no I/O. The same input deterministically
    produces the same chunks (and the same chunk_ids), so re-indexing the
    same content is idempotent.
    """
    title = doc.frontmatter.title or doc.ref.doc_id
    out: list[Chunk] = []
    for section_offset, section_text in _split_into_sections(doc.body):
        for cs, ce, text in _split_section_into_chunks(section_text, section_offset):
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
