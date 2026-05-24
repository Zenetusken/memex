"""Unit tests for `agents.answering._repair_claim_chunk_ids` — snapping
the answer LLM's occasionally-corrupted `source_chunk_id` emissions back
to real reranked chunk ids.

Motivated by the CR350 multi-doc eval (2026-05-23): the answer model
(Qwen3-8B-AWQ) sometimes drops the `docid#` prefix (emitting the bare
hash) or flips a single character in the long ids it's shown. Both break
the citation audit trail and the verifier's chunk lookup.
"""

from __future__ import annotations

from memex.agents.answering import (
    CitedClaim,
    _bounded_levenshtein,
    _repair_claim_chunk_ids,
)
from memex.core.types import Chunk


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=chunk_id.rsplit("#", 1)[0],
        document_title="doc",
        text="...",
    )


def _claim(source_chunk_id: str) -> CitedClaim:
    return CitedClaim(
        claim="a factual statement",
        source_chunk_id=source_chunk_id,
        confidence="high",
    )


# Two reranked chunks from different docs; distinct 10-hex hashes.
_RERANKED = [
    _chunk("7f7ff186-cr350-semaine-4-routeurs#bfc5b7e89a"),
    _chunk("7e7cd59e-cr350-semaine-5-attaques#c641e00478"),
    _chunk("5c9f3e0e-cr350-cours-3#508099af73"),
]


def test_bounded_levenshtein_basic() -> None:
    assert _bounded_levenshtein("abc", "abc", max_d=2) == 0
    assert _bounded_levenshtein("c641e00478", "d641e00478", max_d=2) == 1
    assert _bounded_levenshtein("abc", "abxyz", max_d=2) == 3  # > max_d
    # Early exit: length gap alone exceeds the bound.
    assert _bounded_levenshtein("a", "abcd", max_d=2) == 3


def test_exact_match_preserved() -> None:
    claims = [_claim("7f7ff186-cr350-semaine-4-routeurs#bfc5b7e89a")]
    repaired, stats = _repair_claim_chunk_ids(claims, _RERANKED)
    assert stats == {"exact": 1, "suffix": 0, "fuzzy": 0, "unresolved": 0}
    assert repaired[0].source_chunk_id == "7f7ff186-cr350-semaine-4-routeurs#bfc5b7e89a"


def test_bare_hash_repaired_via_suffix() -> None:
    """The xref-05 case: the model emitted just the hash, no `docid#`."""
    claims = [_claim("bfc5b7e89a")]
    repaired, stats = _repair_claim_chunk_ids(claims, _RERANKED)
    assert stats["suffix"] == 1
    assert repaired[0].source_chunk_id == "7f7ff186-cr350-semaine-4-routeurs#bfc5b7e89a"


def test_single_char_corruption_repaired_via_fuzzy() -> None:
    """The xref-02 case: `c641e00478` emitted as `d641e00478`."""
    claims = [_claim("d641e00478")]
    repaired, stats = _repair_claim_chunk_ids(claims, _RERANKED)
    assert stats["fuzzy"] == 1
    assert repaired[0].source_chunk_id == "7e7cd59e-cr350-semaine-5-attaques#c641e00478"


def test_corrupted_full_id_repaired_via_fuzzy() -> None:
    """A corrupted but prefix-bearing id fuzzy-matches on the hash."""
    claims = [_claim("7e7cd59e-cr350-semaine-5-attaques#c641e00479")]
    repaired, stats = _repair_claim_chunk_ids(claims, _RERANKED)
    assert stats["fuzzy"] == 1
    assert repaired[0].source_chunk_id == "7e7cd59e-cr350-semaine-5-attaques#c641e00478"


def test_far_off_id_left_unresolved() -> None:
    """A hash that matches nothing within distance 2 is left untouched."""
    claims = [_claim("ffffffffff")]
    repaired, stats = _repair_claim_chunk_ids(claims, _RERANKED)
    assert stats["unresolved"] == 1
    assert repaired[0].source_chunk_id == "ffffffffff"


def test_ambiguous_fuzzy_match_left_unresolved() -> None:
    """When two reranked hashes tie for closest, refuse to guess."""
    reranked = [
        _chunk("doc-a#aaaaaaaaaa"),
        _chunk("doc-b#aaaaaaaaab"),  # both distance 1 from the emitted hash
    ]
    claims = [_claim("aaaaaaaaac")]
    repaired, stats = _repair_claim_chunk_ids(claims, reranked)
    assert stats["unresolved"] == 1
    assert repaired[0].source_chunk_id == "aaaaaaaaac"


def test_mixed_batch_counts_each_branch() -> None:
    claims = [
        _claim("5c9f3e0e-cr350-cours-3#508099af73"),  # exact
        _claim("bfc5b7e89a"),  # suffix
        _claim("d641e00478"),  # fuzzy
        _claim("zzzzzzzzzz"),  # unresolved
    ]
    repaired, stats = _repair_claim_chunk_ids(claims, _RERANKED)
    assert stats == {"exact": 1, "suffix": 1, "fuzzy": 1, "unresolved": 1}
    assert [c.source_chunk_id for c in repaired] == [
        "5c9f3e0e-cr350-cours-3#508099af73",
        "7f7ff186-cr350-semaine-4-routeurs#bfc5b7e89a",
        "7e7cd59e-cr350-semaine-5-attaques#c641e00478",
        "zzzzzzzzzz",
    ]
