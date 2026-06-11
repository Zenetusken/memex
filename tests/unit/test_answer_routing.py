"""Unit tests for the answering graph's `route_after_verify` + the
`_has_grounded_claims` helper — the partial-grounded ship logic (#262 /
compound-question over-refusal fix).

Pure functions over an `AnswerState`; no models, no I/O. These pin the routing
matrix precisely so a future change can't silently regress the
ship-the-grounded-subset behaviour or the counterfactual-still-refuses guard.
"""

from __future__ import annotations

from memex.agents.answering import (
    AnswerState,
    CitedClaim,
    DraftAnswer,
    VerificationResult,
    _has_grounded_claims,
    route_after_verify,
)


def _state(
    *,
    grounded: list[int],
    ungrounded: list[int],
    n_claims: int = 2,
    regenerate_attempts: int = 0,
    max_regenerate_attempts: int = 2,
    over_budget: bool = False,
    allow_partial_grounded: bool = True,
) -> AnswerState:
    claims = [
        CitedClaim(claim=f"Claim {i}", source_chunk_id=f"c{i}", confidence="high")
        for i in range(n_claims)
    ]
    return AnswerState(
        query="q",
        draft=DraftAnswer(summary="s", claims=claims),
        verification=VerificationResult(grounded=grounded, ungrounded=ungrounded),
        regenerate_attempts=regenerate_attempts,
        max_regenerate_attempts=max_regenerate_attempts,
        # over_budget() fires on tokens_used >= token_budget.
        tokens_used=9000 if over_budget else 0,
        token_budget=8000,
        allow_partial_grounded=allow_partial_grounded,
    )


# ---- _has_grounded_claims ----


def test_has_grounded_true_for_valid_index() -> None:
    assert _has_grounded_claims(_state(grounded=[0], ungrounded=[1])) is True


def test_has_grounded_false_when_none_grounded() -> None:
    assert _has_grounded_claims(_state(grounded=[], ungrounded=[0, 1])) is False


def test_has_grounded_false_for_phantom_index_only() -> None:
    # A grounded index past the claim list (verifier phantom) doesn't count.
    assert _has_grounded_claims(_state(grounded=[5], ungrounded=[0], n_claims=2)) is False


def test_has_grounded_false_without_draft() -> None:
    assert _has_grounded_claims(AnswerState(query="q")) is False


# ---- route_after_verify ----


def test_route_all_grounded_composes() -> None:
    assert route_after_verify(_state(grounded=[0, 1], ungrounded=[])) == "compose"


def test_route_mixed_with_budget_regenerates() -> None:
    # Attempts remain + under budget → try to ground the rest first.
    assert route_after_verify(_state(grounded=[0], ungrounded=[1])) == "regenerate"


def test_route_mixed_exhausted_ships_grounded_subset() -> None:
    # Regeneration exhausted, a grounded claim survives → ship it (compose),
    # NOT refuse. The compound-question fix.
    s = _state(grounded=[0], ungrounded=[1], regenerate_attempts=2, max_regenerate_attempts=2)
    assert route_after_verify(s) == "compose"


def test_route_mixed_exhausted_kill_switch_refuses() -> None:
    # Same state, partial-grounded disabled → restores all-or-nothing refuse.
    s = _state(
        grounded=[0],
        ungrounded=[1],
        regenerate_attempts=2,
        max_regenerate_attempts=2,
        allow_partial_grounded=False,
    )
    assert route_after_verify(s) == "refuse"


def test_route_zero_grounded_exhausted_refuses() -> None:
    # Nothing grounded → refuse even with partial-grounded on (counterfactual
    # safety: refusal_cf is unaffected).
    s = _state(grounded=[], ungrounded=[0, 1], regenerate_attempts=2, max_regenerate_attempts=2)
    assert route_after_verify(s) == "refuse"


def test_route_over_budget_with_grounded_ships_subset() -> None:
    # A late budget breach must not discard a valid grounded subset.
    s = _state(grounded=[0], ungrounded=[1], over_budget=True)
    assert route_after_verify(s) == "compose"


def test_route_over_budget_zero_grounded_refuses() -> None:
    s = _state(grounded=[], ungrounded=[0, 1], over_budget=True)
    assert route_after_verify(s) == "refuse"


def test_route_no_regenerate_when_attempts_exhausted_but_partial_off() -> None:
    # allow_partial off + exhausted + mixed → refuse (no regenerate, no ship).
    s = _state(
        grounded=[0],
        ungrounded=[1],
        regenerate_attempts=2,
        max_regenerate_attempts=2,
        allow_partial_grounded=False,
    )
    assert route_after_verify(s) == "refuse"


# ── mode-aware rerank top_k (ADR-0007 full-mode leverage) ──


def test_resolve_rerank_top_k_env_override_wins(monkeypatch) -> None:
    """An explicit MEMEX_RERANK_TOP_K is the operator escape hatch — it wins over
    the mode (and a bad value falls back to 5, never crashes the node)."""
    from memex.agents.answering import _resolve_rerank_top_k

    assert _resolve_rerank_top_k("12") == 12
    assert _resolve_rerank_top_k("0") == 1  # floored to ≥1
    assert _resolve_rerank_top_k("notanint") == 5  # parse failure → safe default


def test_resolve_rerank_top_k_follows_active_mode(monkeypatch) -> None:
    """With no env override, top_k comes from the ACTIVE co-residence mode:
    full deepens retrieval (18) while manual/fast keep 5 — so only an explicit
    `full` switch changes the common path."""
    from memex.agents.answering import _resolve_rerank_top_k
    from memex.core.config import MemexSettings, set_settings

    monkeypatch.delenv("MEMEX_RERANK_TOP_K", raising=False)
    for mode, expected in (("manual", 8), ("fast", 8), ("gpu_only", 8), ("full", 18)):
        s = MemexSettings()
        s.models.co_residence_mode = mode
        set_settings(s)
        assert _resolve_rerank_top_k(None) == expected, mode
