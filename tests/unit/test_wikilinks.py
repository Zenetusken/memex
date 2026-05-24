"""Unit tests for `core/wikilinks.py` — the P4.1 section-anchor parser
+ extractor + resolver primitives.

ADR-0003 spec: `[[doc_id]]` AND `[[doc_id#section]]`. The plain form
has been written by enrich since v1; the section-anchor variant is
the new addition (read side now; write side wires when a real
use case lands).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from memex.core.types import Chunk
from memex.core.wikilinks import (
    WikilinkTarget,
    extract_wikilinks,
    parse_wikilink,
    resolve_wikilink_section,
)


# ----------------------------------------------------------------------
# parse_wikilink
# ----------------------------------------------------------------------


def test_parse_wikilink_plain_doc_id() -> None:
    """`[[doc-id]]` → doc_id only, no section."""
    target = parse_wikilink("doc-id")
    assert target.doc_id == "doc-id"
    assert target.section is None
    assert target.raw == "doc-id"


def test_parse_wikilink_with_section() -> None:
    """`[[doc-id#Section Title]]` → doc_id + section."""
    target = parse_wikilink("doc-id#Section Title")
    assert target.doc_id == "doc-id"
    assert target.section == "Section Title"
    assert target.raw == "doc-id#Section Title"


def test_parse_wikilink_empty_section_is_none() -> None:
    """`[[doc-id#]]` (trailing `#`, empty section) → section=None.
    Avoids False-y but non-None section values that downstream code
    would have to special-case."""
    target = parse_wikilink("doc-id#")
    assert target.section is None


def test_parse_wikilink_section_with_hashes() -> None:
    """`[[doc#nested#part]]` — only the FIRST `#` is the separator;
    subsequent `#` characters are part of the section text. Matches
    GitHub's anchor-link convention; lets section titles contain `#`
    (e.g. C# / F# language references)."""
    target = parse_wikilink("doc#C# Programming Guide")
    assert target.doc_id == "doc"
    assert target.section == "C# Programming Guide"


def test_parse_wikilink_strips_whitespace() -> None:
    """Leading/trailing whitespace in either component is stripped —
    `[[ doc-id # Section ]]` → ("doc-id", "Section")."""
    target = parse_wikilink(" doc-id # Section Title ")
    assert target.doc_id == "doc-id"
    assert target.section == "Section Title"


def test_parse_wikilink_empty_inner_raises() -> None:
    """Empty inner text fails pydantic validation (doc_id min_length=1)."""
    with pytest.raises(ValidationError):
        parse_wikilink("")
    with pytest.raises(ValidationError):
        parse_wikilink("   ")


# ----------------------------------------------------------------------
# extract_wikilinks
# ----------------------------------------------------------------------


def test_extract_wikilinks_basic() -> None:
    """Multiple wikilinks in a markdown body, returned in document order."""
    body = "See [[doc-a]] and also [[doc-b#Methods]] for details."
    targets = extract_wikilinks(body)
    assert len(targets) == 2
    assert targets[0].doc_id == "doc-a"
    assert targets[0].section is None
    assert targets[1].doc_id == "doc-b"
    assert targets[1].section == "Methods"


def test_extract_wikilinks_adjacent_links_dont_merge() -> None:
    """`[[a]][[b]]` parses as two links, not one. Regex non-greedy
    `[^\\[\\]]+?` enforces this."""
    body = "[[a]][[b]] and [[c]]"
    targets = extract_wikilinks(body)
    assert [t.doc_id for t in targets] == ["a", "b", "c"]


def test_extract_wikilinks_ignores_malformed_empty() -> None:
    """`[[]]` (empty inner) is skipped — regex requires at least one
    non-bracket char."""
    body = "ok [[doc-a]] not-ok [[]] also-ok [[doc-b]]"
    targets = extract_wikilinks(body)
    assert [t.doc_id for t in targets] == ["doc-a", "doc-b"]


def test_extract_wikilinks_empty_body_returns_empty() -> None:
    assert extract_wikilinks("") == []
    assert extract_wikilinks("no wikilinks here") == []


def test_extract_wikilinks_with_punctuation_around() -> None:
    """Wikilinks adjacent to punctuation (parens, commas, periods)
    still parse correctly."""
    body = "See ([[doc-a]]), [[doc-b]]. Also [[doc-c#Sec]];"
    targets = extract_wikilinks(body)
    assert [t.doc_id for t in targets] == ["doc-a", "doc-b", "doc-c"]
    assert targets[2].section == "Sec"


# ----------------------------------------------------------------------
# resolve_wikilink_section
# ----------------------------------------------------------------------


def _chunk(
    doc_id: str,
    *,
    chunk_id: str = "c1",
    text: str = "body",
    heading_path: list[str] | None = None,
    char_start: int = 0,
) -> Chunk:
    """Test helper to build a Chunk with default values for the
    fields we don't care about in resolver tests."""
    return Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        document_title=doc_id,
        text=text,
        heading_path=heading_path or [],
        char_start=char_start,
    )


def test_resolve_no_section_returns_first_chunk_of_doc() -> None:
    """`[[doc-id]]` (no section) resolves to the FIRST chunk of the
    target document (by char_start)."""
    chunks = [
        _chunk("other-doc", chunk_id="x1", char_start=0),
        _chunk("my-doc", chunk_id="my-2", char_start=500),
        _chunk("my-doc", chunk_id="my-1", char_start=0),
    ]
    target = parse_wikilink("my-doc")
    resolved = resolve_wikilink_section(target, chunks)
    assert resolved is not None
    assert resolved.chunk_id == "my-1"


def test_resolve_section_matches_heading_path_exact() -> None:
    """Section anchor matches a chunk's heading_path entry
    case-insensitively."""
    chunks = [
        _chunk(
            "doc-a",
            chunk_id="a1",
            heading_path=["Introduction"],
            char_start=0,
        ),
        _chunk(
            "doc-a",
            chunk_id="a2",
            heading_path=["Methods", "Subsection"],
            char_start=200,
        ),
        _chunk(
            "doc-a",
            chunk_id="a3",
            heading_path=["Results"],
            char_start=500,
        ),
    ]
    target = parse_wikilink("doc-a#Methods")
    resolved = resolve_wikilink_section(target, chunks)
    assert resolved is not None
    assert resolved.chunk_id == "a2"


def test_resolve_section_case_insensitive() -> None:
    """`[[doc-a#methods]]` matches a chunk with `heading_path=["Methods"]`."""
    chunks = [
        _chunk("doc-a", chunk_id="a1", heading_path=["Methods"]),
    ]
    target = parse_wikilink("doc-a#METHODS")
    resolved = resolve_wikilink_section(target, chunks)
    assert resolved is not None
    assert resolved.chunk_id == "a1"


def test_resolve_section_matches_nested_heading() -> None:
    """A heading at any depth in heading_path matches the anchor."""
    chunks = [
        _chunk(
            "doc-a",
            chunk_id="a1",
            heading_path=["Top", "Middle", "Leaf Section"],
        ),
    ]
    target = parse_wikilink("doc-a#Leaf Section")
    resolved = resolve_wikilink_section(target, chunks)
    assert resolved is not None
    assert resolved.chunk_id == "a1"


def test_resolve_section_no_match_returns_none() -> None:
    """No chunk's heading_path contains the section anchor → None."""
    chunks = [
        _chunk("doc-a", chunk_id="a1", heading_path=["Introduction"]),
    ]
    target = parse_wikilink("doc-a#Conclusion")
    resolved = resolve_wikilink_section(target, chunks)
    assert resolved is None


def test_resolve_doc_id_not_in_chunks_returns_none() -> None:
    """No chunks match `target.doc_id` → None (regardless of section)."""
    chunks = [_chunk("doc-a", chunk_id="a1")]
    assert resolve_wikilink_section(parse_wikilink("missing"), chunks) is None
    assert resolve_wikilink_section(parse_wikilink("missing#Sec"), chunks) is None


def test_resolve_picks_first_matching_chunk_by_char_start() -> None:
    """When multiple chunks share a heading, the EARLIEST (lowest
    char_start) one wins."""
    chunks = [
        _chunk("doc-a", chunk_id="a3", heading_path=["Methods"], char_start=900),
        _chunk("doc-a", chunk_id="a2", heading_path=["Methods"], char_start=500),
        _chunk("doc-a", chunk_id="a1", heading_path=["Methods"], char_start=100),
    ]
    target = parse_wikilink("doc-a#Methods")
    resolved = resolve_wikilink_section(target, chunks)
    assert resolved is not None
    assert resolved.chunk_id == "a1"


def test_resolve_empty_chunks_returns_none() -> None:
    """An empty chunk pool returns None for any target."""
    assert resolve_wikilink_section(parse_wikilink("doc"), []) is None
    assert resolve_wikilink_section(parse_wikilink("doc#Sec"), []) is None
