"""Unit tests for the companion-merge alignment core (B1, ADR-0018): the PURE cosine + MaViLS
argmax/null/monotonic-tie-break, exercised with hand-built embeddings (no GPU, no stores)."""

from __future__ import annotations

import math

import pytest

from memex.core.types import Chunk
from memex.index.companion import align_blocks, cosine_matrix


def _t(chunk_id: str, t0: float = 0.0) -> Chunk:
    """A transcript chunk (carries a time_range)."""
    return Chunk(
        chunk_id=chunk_id, document_id="lec", document_title="Lecture", text="…",
        time_range=(t0, t0 + 30.0),
    )


def _d(chunk_id: str, page: int) -> Chunk:
    """A deck chunk on a given slide page."""
    return Chunk(chunk_id=chunk_id, document_id="deck", document_title="Deck", text="…", page=page)


def test_cosine_matrix_normalizes_and_shapes() -> None:
    # Un-normalized inputs still yield true cosine (dot of unit vectors).
    m = cosine_matrix([[2.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    assert m.shape == (1, 2)
    assert math.isclose(float(m[0][0]), 1.0, abs_tol=1e-5)  # parallel → 1
    assert math.isclose(float(m[0][1]), 0.0, abs_tol=1e-5)  # orthogonal → 0
    # Empty either side → correctly-shaped zeros (no crash).
    assert cosine_matrix([], [[1.0]]).shape == (0, 1)
    assert cosine_matrix([[1.0]], []).shape == (1, 0)


def test_align_argmax_picks_best_page() -> None:
    deck = [_d("d#1", 1), _d("d#2", 2), _d("d#3", 3)]
    p_emb = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
    tx = [_t("t#a")]
    t_emb = [[0, 1.0, 0]]  # matches page 2
    blocks, null_count = align_blocks(tx, deck, t_emb, p_emb, min_score=0.5)
    assert null_count == 0
    assert blocks[0].deck_page == 2 and blocks[0].deck_chunk_id == "d#2"
    assert math.isclose(blocks[0].score, 1.0, abs_tol=1e-5)
    assert blocks[0].time_range == (0.0, 30.0)  # cached anchor preserved


def test_align_null_below_floor() -> None:
    deck = [_d("d#1", 1), _d("d#2", 2)]
    p_emb = [[1.0, 0, 0], [0, 1.0, 0]]
    tx = [_t("t#x")]
    t_emb = [[1.0, 1.0, 1.0]]  # cosine 0.577 to each one-hot
    blocks, null_count = align_blocks(tx, deck, t_emb, p_emb, min_score=0.7)
    assert null_count == 1
    assert blocks[0].deck_chunk_id is None and blocks[0].deck_page is None
    assert blocks[0].score < 0.7


def test_monotonic_tiebreak_prefers_forward_on_near_tie() -> None:
    # page_prev becomes 2 after chunk A; chunk B is a NEAR-TIE whose argmax is page 1 (BACKWARD),
    # but page 3 (FORWARD) is within epsilon → the tie-break picks page 3, not the backward argmax.
    deck = [_d("d#1", 1), _d("d#2", 2), _d("d#3", 3)]
    p_emb = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
    a = _t("t#a", 0.0)
    b = _t("t#b", 30.0)
    # A clearly matches page 2 → page_prev = 2.
    # B: cosine to page1 = 0.71 (argmax), page3 = 0.70 (within 0.02), page2 = 0.
    t_emb = [[0, 1.0, 0], [0.71, 0.0, 0.70]]
    blocks, _ = align_blocks([a, b], deck, t_emb, p_emb, min_score=0.4, epsilon=0.02)
    assert blocks[0].deck_page == 2  # A
    assert blocks[1].deck_page == 3  # B → forward, NOT the backward argmax page 1


def test_clear_backward_jump_is_allowed() -> None:
    # When the backward match is CLEAR (not a near-tie), the argmax wins — a teacher CAN revisit.
    deck = [_d("d#1", 1), _d("d#2", 2), _d("d#3", 3)]
    p_emb = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
    a = _t("t#a", 0.0)  # → page 2
    b = _t("t#b", 30.0)  # clearly page 1 (0.97), page 3 far (0.2) → backward jump allowed
    t_emb = [[0, 1.0, 0], [0.97, 0.0, 0.2]]
    blocks, _ = align_blocks([a, b], deck, t_emb, p_emb, min_score=0.4, epsilon=0.02)
    assert blocks[0].deck_page == 2
    assert blocks[1].deck_page == 1  # clear backward argmax wins (no near-tie to override)


def test_empty_deck_all_null() -> None:
    tx = [_t("t#a"), _t("t#b")]
    blocks, null_count = align_blocks(tx, [], [[1.0], [1.0]], [], min_score=0.4)
    assert null_count == 2
    assert all(b.deck_chunk_id is None for b in blocks)


def test_parallel_length_validation() -> None:
    with pytest.raises(ValueError, match="parallel transcript_chunks"):
        align_blocks([_t("t#a")], [_d("d#1", 1)], [], [[1.0, 0]], min_score=0.4)
    with pytest.raises(ValueError, match="parallel deck_chunks"):
        align_blocks([_t("t#a")], [_d("d#1", 1)], [[1.0, 0]], [], min_score=0.4)
