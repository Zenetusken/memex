"""Wikilink primitives — parser, extractor, section-aware resolver.

ADR-0003 commits to two wikilink shapes:
    [[doc_id]]
    [[doc_id#section]]

The plain form `[[doc_id]]` has been written by `enrich.citations::
insert_wikilinks` since v1. The section-anchor form
`[[doc_id#section]]` is the P4.1 primitive shipped here — the
read-side (parser + resolver) is ready for future write-side wiring
when a real citation-with-section use case lands (currently in Tier
5 'punt until a user needs it' per ROADMAP).

Lives in `core/` so any module can use it without breaking the
documented module-import direction (`agents/ → core/`; no
`core/ → index/`).

This module is pure-text helpers and a small pydantic model; no
filesystem, no model loads, no I/O of any kind.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from memex.core.types import Chunk

# Matches any `[[inner]]` where inner doesn't contain `[` or `]`. The
# `+?` makes it non-greedy so adjacent wikilinks don't merge:
#     `[[a]] and [[b]]` → ["a", "b"], not ["a]] and [[b"].
_WIKILINK_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")

# The character separating doc_id from the optional section anchor.
# Per ADR-0003: `[[doc_id#section]]`. Subsequent `#` characters in the
# section text are preserved (e.g., `[[doc#sec#sub]]` resolves to
# section="sec#sub"), which mirrors GitHub's anchor-link convention
# and lets section titles contain `#`.
_SECTION_SEP = "#"


class WikilinkTarget(BaseModel):
    """Parsed wikilink target.

    `doc_id` is the document identifier (the vault's `doc_id` —
    typically a short hex prefix + slug).

    `section` is the optional section-anchor text. Matched
    case-insensitively against the target's chunks' `heading_path`
    in `resolve_wikilink_section`.

    `raw` is the un-parsed inner text of the wikilink (the part
    between `[[` and `]]`), preserved for round-trip / error
    reporting / display.
    """

    doc_id: str = Field(min_length=1)
    section: str | None = None
    raw: str = Field(min_length=1)


def parse_wikilink(inner: str) -> WikilinkTarget:
    """Parse the inner text of a wikilink (without the surrounding
    `[[` `]]`).

    Examples:
        parse_wikilink("doc-id")
            → WikilinkTarget(doc_id="doc-id", section=None)
        parse_wikilink("doc-id#Section Title")
            → WikilinkTarget(doc_id="doc-id", section="Section Title")
        parse_wikilink("doc-id#")
            → WikilinkTarget(doc_id="doc-id", section=None)
            (an empty section is treated as absent)
        parse_wikilink("doc-id#nested#part")
            → WikilinkTarget(doc_id="doc-id", section="nested#part")
            (only the first `#` is the separator; subsequent `#`
            characters belong to the section text)

    Leading/trailing whitespace in either component is stripped.
    Raises pydantic ValidationError if `inner` is empty (caller
    should pre-validate or catch).
    """
    parts = inner.split(_SECTION_SEP, 1)
    doc_id = parts[0].strip()
    section = parts[1].strip() if len(parts) == 2 else None
    return WikilinkTarget(
        doc_id=doc_id,
        section=section if section else None,
        raw=inner,
    )


def format_wikilink(doc_id: str, section: str | None = None) -> str:
    """Build a wikilink string from a doc_id and optional section.

    `[[doc_id#section]]` (RAW heading text, NOT a slug) when `section`
    is non-empty after stripping, else `[[doc_id]]`. The emitted
    grammar round-trips through `parse_wikilink` for a clean section.

    Section is RAW heading text because `resolve_wikilink_section`
    matches it case-insensitively against `chunk.heading_path`, and
    `webui/rendering.py::slugify_heading` slugifies on demand for the
    URL fragment — consumers slugify, emission stays raw (e.g.
    `[[0e725ba0#Director Compensation]]`).

    Sanitization: if the (stripped) section contains `[` or `]`, fall
    back to the bare `[[doc_id]]` form. A `]` would terminate the
    read-side `_WIKILINK_RE` early, breaking the link. A `#` in the
    section is SAFE (parse splits on the first `#` only; doc_ids carry
    no `#`) and is preserved.
    """
    if section and section.strip():
        stripped = section.strip()
        if "[" not in stripped and "]" not in stripped:
            return f"[[{doc_id}#{stripped}]]"
    return f"[[{doc_id}]]"


def extract_wikilinks(body: str) -> list[WikilinkTarget]:
    """Find all wikilinks in `body`, returning their parsed targets
    in document order (by char_start of each match).

    Malformed wikilinks where the inner text is empty (`[[]]`) are
    skipped silently — the regex `[^\\[\\]]+?` requires at least one
    non-bracket char inside.

    Does NOT distinguish between wikilinks in prose vs. code blocks
    vs. tables. Callers that need to ignore wikilinks inside
    fenced-code regions should pre-process the body. For Memex's
    use case (citation-enriched markdown), wikilinks appear in
    prose; code blocks rarely contain literal `[[...]]` syntax.
    """
    return [parse_wikilink(m.group(1)) for m in _WIKILINK_RE.finditer(body)]


def resolve_wikilink_section(
    target: WikilinkTarget,
    chunks: list[Chunk],
) -> Chunk | None:
    """Find the chunk in `chunks` that contains the wikilink's
    section anchor.

    Match strategy:
    - If `target.section is None`: return the FIRST chunk of the
      target document (by `char_start`), the closest analog to
      "the document itself."
    - Otherwise: case-insensitive equality match of `target.section`
      against any heading in each chunk's `heading_path`. The first
      matching chunk (by `char_start`) wins.
    - Returns `None` if `target.doc_id` matches no chunks OR if
      `target.section` matches no chunk's heading path.

    `chunks` may be a global pool (covering many documents); the
    function filters to `target.doc_id` internally.

    This resolver does NOT use fuzzy matching or substring containment
    — section titles are matched exactly (case-insensitively). For
    looser matching, pre-process `target.section` before calling.
    """
    doc_chunks = [c for c in chunks if c.document_id == target.doc_id]
    if not doc_chunks:
        return None
    doc_chunks.sort(key=lambda c: c.char_start)

    if target.section is None:
        return doc_chunks[0]

    section_lc = target.section.casefold()
    for chunk in doc_chunks:
        for heading in chunk.heading_path:
            if heading.casefold() == section_lc:
                return chunk
    return None
