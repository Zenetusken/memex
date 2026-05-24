"""Audit follow-up — pin `reciprocal_rank_fusion`'s contract.

The test-organization audit (2026-05-22) flagged `retrieve/fusion.py`
as the only pure-function module on the query hot path with zero
direct unit tests. The function had been exercised only indirectly via
the agent's integration tests, which mock around it.

These tests pin:
- The RRF formula (1 / (k + rank)) so a future refactor can't silently
  change the weighting.
- The first-occurrence-metadata-wins rule (chunks in multiple rankings
  preserve the FIRST list's metadata, only the score is re-derived).
- The descending-score ordering invariant.
- The k parameter's role as a dampener.
- The empty + single-ranking edge cases that production occasionally
  hits (e.g., when one retrieval branch returns zero results).
"""

from __future__ import annotations

from itertools import pairwise

from memex.core.types import Chunk
from memex.retrieve.fusion import DEFAULT_K, reciprocal_rank_fusion


def _chunk(chunk_id: str, text: str = "") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-A",
        document_title="Doc A",
        text=text or f"text-{chunk_id}",
        char_start=0,
        char_end=10,
    )


def test_empty_rankings_returns_empty_list() -> None:
    """No input → no output. Production hits this when both BM25 and
    dense return empty (e.g., FTS5 query rejected as malformed)."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_single_ranking_preserves_order() -> None:
    """One ranked list in, same order out. The RRF formula is
    monotonically decreasing in rank, so the input's order is
    preserved exactly."""
    chunks = [_chunk(f"c{i}") for i in range(5)]
    result = reciprocal_rank_fusion([chunks])
    assert [c.chunk_id for c in result] == [c.chunk_id for c in chunks]
    # Scores strictly decreasing
    scores = [c.score for c in result]
    assert scores == sorted(scores, reverse=True)


def test_rrf_formula_default_k() -> None:
    """Score for a chunk at rank R in one list = 1 / (DEFAULT_K + R).
    Documents the formula explicitly so a future refactor that
    accidentally changes the weighting (e.g., to `1 / (k + r + 1)` or
    `1 / k**r`) breaks this test."""
    chunks = [_chunk("c0"), _chunk("c1"), _chunk("c2")]
    result = reciprocal_rank_fusion([chunks])
    assert result[0].score == 1.0 / (DEFAULT_K + 1)
    assert result[1].score == 1.0 / (DEFAULT_K + 2)
    assert result[2].score == 1.0 / (DEFAULT_K + 3)


def test_rrf_formula_custom_k() -> None:
    """`k` is a dampener — larger k → smaller per-rank score → more
    even fusion. Smaller k → first-rank dominates."""
    chunks = [_chunk("c0"), _chunk("c1")]
    aggressive = reciprocal_rank_fusion([chunks], k=1)
    gentle = reciprocal_rank_fusion([chunks], k=1000)
    assert aggressive[0].score == 1.0 / 2  # k=1 + rank=1
    assert gentle[0].score == 1.0 / 1001
    # Both still produce the same order, just different magnitudes.
    assert [c.chunk_id for c in aggressive] == [c.chunk_id for c in gentle]


def test_chunk_in_multiple_rankings_sums_contributions() -> None:
    """A chunk that appears in two lists at rank R1 and R2 fuses to
    `1/(k+R1) + 1/(k+R2)`. Critical: the fusion is ADDITIVE — chunks
    that BOTH retrievers found rank higher than chunks either alone
    found at the same position."""
    bm25 = [_chunk("shared"), _chunk("bm25_only")]
    dense = [_chunk("dense_only"), _chunk("shared")]
    result = reciprocal_rank_fusion([bm25, dense])

    by_id = {c.chunk_id: c for c in result}
    # shared: rank 1 in bm25 + rank 2 in dense
    expected_shared = 1.0 / (DEFAULT_K + 1) + 1.0 / (DEFAULT_K + 2)
    # bm25_only: rank 2 in bm25 only
    expected_bm25 = 1.0 / (DEFAULT_K + 2)
    # dense_only: rank 1 in dense only
    expected_dense = 1.0 / (DEFAULT_K + 1)

    assert by_id["shared"].score == expected_shared
    assert by_id["bm25_only"].score == expected_bm25
    assert by_id["dense_only"].score == expected_dense

    # And the shared chunk ranks first (its sum exceeds either solo
    # contribution).
    assert result[0].chunk_id == "shared"


def test_first_occurrence_metadata_wins() -> None:
    """When a chunk_id appears in multiple rankings, the FIRST list's
    Chunk instance is preserved — only the score is re-derived. This
    matters when dense and BM25 return chunks with the same
    chunk_id but slightly different `text` (e.g., FTS body stripped
    of [chart-extracted] blocks vs LanceDB full text)."""
    bm25_first = _chunk("shared", text="STRIPPED FTS text")
    dense_first = _chunk("shared", text="FULL dense text")
    # rankings[0] = bm25; rankings[1] = dense
    result = reciprocal_rank_fusion([[bm25_first], [dense_first]])
    assert len(result) == 1
    # The bm25 list was first, so its text is what propagates.
    assert result[0].text == "STRIPPED FTS text"
    # Score is the sum of contributions from both lists at rank 1.
    assert result[0].score == 2.0 / (DEFAULT_K + 1)


def test_no_duplicates_in_output() -> None:
    """Even when the same chunk_id appears in multiple rankings, the
    fused output has exactly ONE entry per chunk_id."""
    rankings = [
        [_chunk("a"), _chunk("b"), _chunk("c")],
        [_chunk("b"), _chunk("c"), _chunk("a")],
        [_chunk("c"), _chunk("a"), _chunk("b")],
    ]
    result = reciprocal_rank_fusion(rankings)
    chunk_ids = [c.chunk_id for c in result]
    assert sorted(chunk_ids) == ["a", "b", "c"]
    assert len(set(chunk_ids)) == 3  # no duplicates


def test_output_is_descending_score_order() -> None:
    """Caller depends on the result being sorted descending — agent's
    `retrieve` node takes the top-k from this list directly."""
    bm25 = [_chunk(f"b{i}") for i in range(10)]
    dense = [_chunk(f"d{i}") for i in range(10)]
    # One chunk appears in both lists.
    bm25.append(_chunk("shared"))
    dense.append(_chunk("shared"))
    result = reciprocal_rank_fusion([bm25, dense])
    scores = [c.score for c in result]
    # Allow equal-score ties; assert non-increasing across adjacent pairs.
    for prev, nxt in pairwise(scores):
        assert prev >= nxt


def test_rrf_does_not_mutate_inputs() -> None:
    """Input chunk objects must not be mutated — production code holds
    references to dense/BM25 results for separate downstream uses
    (e.g., logging the raw retrieval scores before fusion). The
    RRF must return COPIES with updated scores, not mutate in place."""
    original = _chunk("c0")
    original_score = original.score
    result = reciprocal_rank_fusion([[original]])
    # Score on the result is the fused score; original is unchanged.
    assert result[0].score == 1.0 / (DEFAULT_K + 1)
    assert original.score == original_score


def test_three_way_fusion() -> None:
    """The function accepts arbitrary-many rankings (production today
    fuses 2, but the contract should handle N)."""
    rankings = [
        [_chunk("a"), _chunk("b")],
        [_chunk("b"), _chunk("a")],
        [_chunk("c"), _chunk("a")],
    ]
    result = reciprocal_rank_fusion(rankings)
    by_id = {c.chunk_id: c for c in result}
    # 'a' appears in all three lists at ranks 1, 2, 2
    expected_a = 1.0 / (DEFAULT_K + 1) + 1.0 / (DEFAULT_K + 2) + 1.0 / (DEFAULT_K + 2)
    assert by_id["a"].score == expected_a
    assert result[0].chunk_id == "a"  # highest aggregated score
