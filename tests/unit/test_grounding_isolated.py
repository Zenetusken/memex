"""`agents/grounding.py::ground_claims_isolated` — the bridge's per-claim (N=1) re-verification
that defeats the `verify_grounding/v2` BATCH-LENIENCY effect (2026-06-03). Pins the pure plumbing
(one `ground_claims` call per claim, order-preserving union, summed tokens, per-claim fail-open,
empty short-circuit) with `ground_claims` itself faked — no model, no prompt.
"""

from __future__ import annotations

import pytest

from memex.agents import grounding
from memex.agents.answering import CitedClaim
from memex.agents.grounding import ground_claims_isolated
from memex.core.types import Chunk


def _claim(text: str, cid: str) -> CitedClaim:
    return CitedClaim(claim=text, source_chunk_id=cid, confidence="high")


def _chunk(cid: str) -> Chunk:
    return Chunk(chunk_id=cid, document_id=cid.split("#")[0], document_title="Doc", text="body")


@pytest.mark.asyncio
async def test_empty_claims_short_circuits() -> None:
    kept, tokens = await ground_claims_isolated("s", [], [_chunk("d#a")], max_tokens=128)
    assert kept == []
    assert tokens == 0


@pytest.mark.asyncio
async def test_one_call_per_claim_order_preserved_tokens_summed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each claim gets ITS OWN single-claim `ground_claims` call; survivors keep INPUT order and
    tokens are summed across all calls (not just survivors')."""
    calls: list[str] = []

    async def _fake_ground(
        summary: str, claims: list[CitedClaim], chunks: list[Chunk], *, max_tokens: int
    ) -> tuple[list[CitedClaim], int]:
        assert len(claims) == 1, "isolated grounding must verify exactly one claim per call"
        c = claims[0]
        calls.append(c.source_chunk_id)
        # ground d#a and d#c (returns the claim + 5 tokens); drop d#b (returns [] + 3 tokens)
        return ([c], 5) if c.source_chunk_id in {"d#a", "d#c"} else ([], 3)

    monkeypatch.setattr(grounding, "ground_claims", _fake_ground)

    claims = [_claim("A", "d#a"), _claim("B", "d#b"), _claim("C", "d#c")]
    chunks = [_chunk("d#a"), _chunk("d#b"), _chunk("d#c")]
    kept, tokens = await ground_claims_isolated("ctx", claims, chunks, max_tokens=256)

    assert [c.source_chunk_id for c in kept] == ["d#a", "d#c"]  # order preserved, d#b dropped
    assert tokens == 5 + 3 + 5  # summed across ALL calls, including the dropped one
    assert len(calls) == 3  # exactly one call per claim


@pytest.mark.asyncio
async def test_all_chunks_passed_to_each_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every isolated call receives the FULL chunk set (not cited-chunk-only) — so sibling-chunk
    support is available to each claim."""
    seen_chunk_counts: list[int] = []

    async def _fake_ground(
        summary: str, claims: list[CitedClaim], chunks: list[Chunk], *, max_tokens: int
    ) -> tuple[list[CitedClaim], int]:
        seen_chunk_counts.append(len(chunks))
        return ([claims[0]], 1)

    monkeypatch.setattr(grounding, "ground_claims", _fake_ground)

    claims = [_claim("A", "d#a"), _claim("B", "d#b")]
    chunks = [_chunk("d#a"), _chunk("d#b"), _chunk("d#c")]
    await ground_claims_isolated("ctx", claims, chunks, max_tokens=256)

    assert seen_chunk_counts == [3, 3]  # all 3 chunks passed to BOTH calls


@pytest.mark.asyncio
async def test_per_claim_fail_open_drops_only_the_failing_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ground_claims` already returns ([], 0) on a ModelCallError, so a claim whose call fails
    drops ONLY itself — siblings still ground (the deliberate isolated-mode contract)."""

    async def _fake_ground(
        summary: str, claims: list[CitedClaim], chunks: list[Chunk], *, max_tokens: int
    ) -> tuple[list[CitedClaim], int]:
        c = claims[0]
        return ([], 0) if c.source_chunk_id == "d#a" else ([c], 4)  # d#a "failed" → dropped

    monkeypatch.setattr(grounding, "ground_claims", _fake_ground)

    claims = [_claim("A", "d#a"), _claim("B", "d#b")]
    kept, tokens = await ground_claims_isolated("ctx", claims, [_chunk("d#a"), _chunk("d#b")], max_tokens=256)

    assert [c.source_chunk_id for c in kept] == ["d#b"]  # A dropped, B survives
    assert tokens == 0 + 4
