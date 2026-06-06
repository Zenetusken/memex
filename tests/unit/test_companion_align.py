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
        chunk_id=chunk_id,
        document_id="lec",
        document_title="Lecture",
        text="…",
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


def test_keyframe_signal_is_primary_over_transcript_argmax() -> None:
    # The keyframe signal (ADR-0018 §13) for a chunk OVERRIDES its transcript-text argmax: t#a's text
    # matches page 1, but its keyframe entry assigns page 3 (the frame showed slide 3).
    deck = [_d("d#1", 1), _d("d#2", 2), _d("d#3", 3)]
    p_emb = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
    tx = [_t("t#a")]
    t_emb = [[1.0, 0, 0]]  # transcript text → page 1
    blocks, null_count = align_blocks(
        tx, deck, t_emb, p_emb, min_score=0.5, keyframe_signal={"t#a": ("d#3", 3, 0.93)}
    )
    assert null_count == 0
    assert (
        blocks[0].deck_page == 3 and blocks[0].deck_chunk_id == "d#3"
    )  # keyframe wins, not page 1
    assert blocks[0].score == 0.93


def test_keyframe_partial_falls_back_and_advances_page_prev() -> None:
    # t#a has a keyframe (page 2); t#b has NONE → transcript-text path, and its monotonic tie-break
    # must see page_prev=2 from the keyframe (a near-tie between page 1 and page 3 → forward page 3).
    deck = [_d("d#1", 1), _d("d#2", 2), _d("d#3", 3)]
    p_emb = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
    a, b = _t("t#a", 0.0), _t("t#b", 30.0)
    t_emb = [[1.0, 0, 0], [0.71, 0.0, 0.70]]  # t#a text → page1; t#b near-tie page1(.71)/page3(.70)
    blocks, _ = align_blocks(
        [a, b],
        deck,
        t_emb,
        p_emb,
        min_score=0.4,
        epsilon=0.02,
        keyframe_signal={"t#a": ("d#2", 2, 0.88)},
    )
    assert blocks[0].deck_page == 2  # t#a from the keyframe (not its text's page 1)
    assert blocks[1].deck_page == 3  # t#b falls back, tie-broken FORWARD from the keyframe's page 2


# ----- Arc 2: §13 monotonic DP + start_s time prior (opt-in) -----

_DECK3 = [_d("d#1", 1), _d("d#2", 2), _d("d#3", 3)]
_P3 = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]


def test_dp_basic_argmax_sanity() -> None:
    # With no jump/time pressure, the DP still lands the obvious argmax.
    blocks, null_count = align_blocks(
        [_t("t#a")], _DECK3, [[0, 1.0, 0]], _P3, min_score=0.4, use_dp=True, time_weight=0.0
    )
    assert null_count == 0
    assert blocks[0].deck_page == 2 and blocks[0].deck_chunk_id == "d#2"


def test_dp_backward_penalty_flips_where_greedy_goes_backward() -> None:
    # A → page 2 (page_prev=2). B's ARGMAX is page 1 (0.73, backward) over page 3 (0.68, forward) —
    # NOT a near-tie, so the GREEDY picks the backward argmax. The DP's asymmetric penalty (backward
    # 2× forward) makes page 3 the cheaper global choice → it flips B forward. This is the DP doing
    # something the greedy can't.
    a, b = _t("t#a", 0.0), _t("t#b", 30.0)
    t_emb = [[0, 1.0, 0], [0.73, 0.0, 0.68]]
    greedy, _ = align_blocks([a, b], _DECK3, t_emb, _P3, min_score=0.4)
    assert greedy[1].deck_page == 1  # greedy follows the backward argmax
    dp, _ = align_blocks(
        [a, b], _DECK3, t_emb, _P3, min_score=0.4, use_dp=True, lambda_jump=0.1, time_weight=0.0
    )
    assert dp[0].deck_page == 2
    assert dp[1].deck_page == 3  # DP penalizes the backward jump → forward wins


def test_dp_time_prior_pulls_a_late_chunk_forward() -> None:
    # A single LATE chunk (start_s 90 of a 120 s lecture → fraction 0.75 → expected slide ~index 1.5).
    # Its cosine slightly favors page 1 (0.72) over page 3 (0.70), so the greedy picks page 1; the
    # start_s time prior pulls it toward the later slide → the DP picks page 3.
    late = _t("t#late", 90.0)  # time_range (90, 120) → T_total 120
    t_emb = [[0.72, 0.0, 0.70]]
    greedy, _ = align_blocks([late], _DECK3, t_emb, _P3, min_score=0.4)
    assert greedy[0].deck_page == 1  # cosine argmax
    dp, _ = align_blocks(
        [late], _DECK3, t_emb, _P3, min_score=0.4, use_dp=True, lambda_jump=0.0, time_weight=0.2
    )
    assert dp[0].deck_page == 3  # the time prior overrides the marginal cosine lead


def test_dp_keyframe_anchor_is_fixed_and_advances_context() -> None:
    # Under the DP, a keyframe-PRIMARY chunk is a FIXED anchor: t#a is pinned to page 2 (its frame),
    # and t#b (transcript fallback, near-tie page1/page3) is tie-broken FORWARD from the anchor.
    a, b = _t("t#a", 0.0), _t("t#b", 30.0)
    t_emb = [[1.0, 0, 0], [0.71, 0.0, 0.70]]
    dp, _ = align_blocks(
        [a, b],
        _DECK3,
        t_emb,
        _P3,
        min_score=0.4,
        use_dp=True,
        lambda_jump=0.1,
        time_weight=0.0,
        keyframe_signal={"t#a": ("d#2", 2, 0.9)},
    )
    assert (
        dp[0].deck_page == 2 and dp[0].deck_chunk_id == "d#2" and dp[0].score == 0.9
    )  # anchor verbatim
    assert dp[1].deck_page == 3  # forward from the anchor's page 2 (backward to page 1 costs more)


def test_dp_null_below_floor_carries_context() -> None:
    # A below-floor chunk stays NULL under the DP (a tangent), and the page context carries: the
    # surrounding chunks still anchor to their pages.
    a = _t("t#a", 0.0)  # → page 1
    tangent = _t("t#mid", 30.0)  # all cosines below the floor → NULL
    c = _t("t#c", 60.0)  # → page 2
    t_emb = [[1.0, 0, 0], [0.3, 0.3, 0.3], [0, 1.0, 0]]
    dp, null_count = align_blocks(
        [a, tangent, c], _DECK3, t_emb, _P3, min_score=0.6, use_dp=True, time_weight=0.0
    )
    assert null_count == 1
    assert dp[1].deck_chunk_id is None and dp[1].deck_page is None  # the tangent is null
    assert dp[0].deck_page == 1 and dp[2].deck_page == 2  # neighbours assigned normally


def test_dp_off_is_the_greedy_path() -> None:
    # The default (use_dp=False) is byte-identical to the greedy — pin it explicitly on the
    # near-tie case so a future DP change can't silently alter the shipped default.
    a, b = _t("t#a", 0.0), _t("t#b", 30.0)
    t_emb = [[0, 1.0, 0], [0.71, 0.0, 0.70]]
    off, _ = align_blocks([a, b], _DECK3, t_emb, _P3, min_score=0.4, epsilon=0.02, use_dp=False)
    greedy, _ = align_blocks([a, b], _DECK3, t_emb, _P3, min_score=0.4, epsilon=0.02)
    assert [bl.deck_page for bl in off] == [bl.deck_page for bl in greedy] == [2, 3]
