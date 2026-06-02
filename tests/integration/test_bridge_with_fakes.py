"""Reason-then-ground bridge (`agents/bridge.py`, Surface §11) — heavy deps faked.

Pins the v1 CONTRACT with no GPU: only claims that survive the UNCHANGED verify gate are
presented as grounded; a claim whose cited id can't resolve to a reranked chunk is dropped
DETERMINISTICALLY before grounding; zero grounded returns the labelled analysis with no citation
chrome (never a refusal); an extraction OR grounding failure fails open to the analysis-only path.

The three faked seams: `bridge.reason_over_evidence` (Stage 1 — bypasses retrieval/rerank/reason),
`bridge.complete_structured` (Stage 1.5 — claim extraction), `grounding.complete_structured`
(Stage 2 — the per-claim verify). `repair_claim_chunk_ids`, the deterministic id filter, and
`ground_claims`' keep-rule all run for real.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from memex.agents import bridge, grounding
from memex.agents.answering import CitedClaim, DraftAnswer, RelevanceAssessment, VerificationResult
from memex.agents.bridge import reason_then_ground
from memex.core.config import MemexSettings, set_settings
from memex.core.errors import ModelCallError
from memex.core.types import Chunk


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemexSettings]:
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    s = MemexSettings()
    set_settings(s)
    yield s
    set_settings(None)


def _chunk(cid: str, text: str = "some grounded body text") -> Chunk:
    return Chunk(
        chunk_id=cid,
        document_id=cid.split("#")[0],
        document_title="Doc",
        text=text,
        heading_path=["Section"],
    )


def _claim(text: str, cid: str) -> CitedClaim:
    return CitedClaim(claim=text, source_chunk_id=cid, confidence="medium")


def _patch_stage1(monkeypatch: pytest.MonkeyPatch, analysis: str, reranked: list[Chunk]) -> None:
    async def _reason(question: str, **_kw: Any) -> tuple[str, list[Chunk], int]:
        return analysis, reranked, 10

    monkeypatch.setattr(bridge, "reason_over_evidence", _reason)


def _patch_extract(
    monkeypatch: pytest.MonkeyPatch, claims: list[CitedClaim], *, fail: bool = False
) -> None:
    async def _extract(*, schema: type, **_kw: Any) -> tuple[Any, int]:
        if fail:
            raise ModelCallError("extract boom")
        return DraftAnswer(summary="context", claims=claims), 12

    monkeypatch.setattr(bridge, "complete_structured", _extract)


def _patch_ground(
    monkeypatch: pytest.MonkeyPatch,
    *,
    grounded: list[int],
    ungrounded: list[int] | None = None,
    fail: bool = False,
    responsive: bool = True,
    relevance_reason: str = "addresses the question",
    relevance_fail: bool = False,
) -> list[str]:
    """Fake `grounding.complete_structured` — serves BOTH the per-claim verify (VerificationResult)
    AND the present-as-answer responsiveness gate (RelevanceAssessment), discriminated by schema.
    Returns a list that records each schema seen, so a test can assert the gate did/didn't fire."""
    seen: list[str] = []

    async def _verify(*, schema: type, **_kw: Any) -> tuple[Any, int]:
        if schema is RelevanceAssessment:
            seen.append("RelevanceAssessment")
            if relevance_fail:
                raise ModelCallError("relevance boom")
            return RelevanceAssessment(responsive=responsive, reason=relevance_reason), 4
        seen.append("VerificationResult")
        if fail:
            raise ModelCallError("verify boom")
        return VerificationResult(grounded=grounded, ungrounded=ungrounded or []), 8

    monkeypatch.setattr(grounding, "complete_structured", _verify)
    return seen


@pytest.mark.asyncio
async def test_only_grounded_claims_survive(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    reranked = [_chunk("d#a"), _chunk("d#b"), _chunk("d#c")]
    _patch_stage1(monkeypatch, "An analysis with three claims.", reranked)
    _patch_extract(
        monkeypatch,
        [_claim("claim A", "d#a"), _claim("claim B", "d#b"), _claim("claim C", "d#c")],
    )
    _patch_ground(monkeypatch, grounded=[0, 2], ungrounded=[1])

    ans = await reason_then_ground("How would you harden the DMZ?")

    assert ans.n_extracted == 3
    assert ans.n_grounded == 2
    assert [c.claim for c in ans.grounded_claims] == ["claim A", "claim C"]
    # claim B stays in the analysis only — never presented as grounded.
    assert all(c.claim != "claim B" for c in ans.grounded_claims)
    assert {c.chunk_id for c in ans.grounded_sources} == {"d#a", "d#c"}
    assert ans.analysis == "An analysis with three claims."


@pytest.mark.asyncio
async def test_ungrounded_claim_is_dropped(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    reranked = [_chunk("d#a")]
    _patch_stage1(monkeypatch, "Analysis.", reranked)
    _patch_extract(monkeypatch, [_claim("unsupported claim", "d#a")])
    _patch_ground(monkeypatch, grounded=[], ungrounded=[0])

    ans = await reason_then_ground("Q?")

    assert ans.n_extracted == 1
    assert ans.n_grounded == 0
    assert ans.grounded_claims == []
    assert ans.grounded_sources == []
    assert ans.analysis == "Analysis."  # the analysis is still returned


@pytest.mark.asyncio
async def test_unresolvable_chunk_id_dropped_before_grounding(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deterministic faithfulness filter drops a claim whose cited id can't resolve to a
    reranked chunk BEFORE grounding. Proven by making the verifier WILLING to ground both indices:
    if the bad claim reached grounding, n_grounded would be 2; the filter makes it 1."""
    reranked = [_chunk("d#a")]
    _patch_stage1(monkeypatch, "Analysis.", reranked)
    _patch_extract(
        monkeypatch,
        [_claim("good claim", "d#a"), _claim("fabricated claim", "ghostdoc#zzzzzzz")],
    )
    _patch_ground(monkeypatch, grounded=[0, 1])  # verifier would ground BOTH if asked

    ans = await reason_then_ground("Q?")

    assert ans.n_grounded == 1  # not 2 → the fabricated claim never reached the verifier
    assert [c.claim for c in ans.grounded_claims] == ["good claim"]


@pytest.mark.asyncio
async def test_zero_grounded_returns_analysis_not_refusal(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    reranked = [_chunk("d#a"), _chunk("d#b")]
    _patch_stage1(monkeypatch, "A useful but unverifiable analysis.", reranked)
    _patch_extract(monkeypatch, [_claim("claim A", "d#a"), _claim("claim B", "d#b")])
    _patch_ground(monkeypatch, grounded=[])

    ans = await reason_then_ground("Q?")

    assert ans.analysis == "A useful but unverifiable analysis."
    assert ans.grounded_claims == []
    assert ans.grounded_sources == []
    assert ans.n_extracted == 2
    assert ans.n_grounded == 0


@pytest.mark.asyncio
async def test_grounding_failure_fails_open(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    reranked = [_chunk("d#a")]
    _patch_stage1(monkeypatch, "Analysis.", reranked)
    _patch_extract(monkeypatch, [_claim("claim A", "d#a")])
    _patch_ground(monkeypatch, grounded=[0], fail=True)  # the verify call raises

    ans = await reason_then_ground("Q?")

    assert ans.n_grounded == 0  # fail-open: never ship ungrounded on a grounding failure
    assert ans.grounded_claims == []
    assert ans.analysis == "Analysis."


@pytest.mark.asyncio
async def test_extraction_failure_fails_open(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    reranked = [_chunk("d#a")]
    _patch_stage1(monkeypatch, "Analysis.", reranked)
    _patch_extract(monkeypatch, [], fail=True)  # extraction raises
    _patch_ground(monkeypatch, grounded=[0])

    ans = await reason_then_ground("Q?")

    assert ans.n_extracted == 0
    assert ans.n_grounded == 0
    assert ans.analysis == "Analysis."


# --- Present-as-answer (ADR-0016): the consented escalation presents the grounded subset AS an
# answer ONLY when it is non-empty AND responsive. The gate (assess_relevance) is added here, and
# ONLY here — the standalone path (default present_as_answer=False) never runs it. ---


@pytest.mark.asyncio
async def test_present_as_answer_responsive_is_presented(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    reranked = [_chunk("d#a"), _chunk("d#b")]
    _patch_stage1(monkeypatch, "Deep reasoning prose that is ungrounded.", reranked)
    _patch_extract(monkeypatch, [_claim("claim A", "d#a"), _claim("claim B", "d#b")])
    seen = _patch_ground(monkeypatch, grounded=[0, 1], responsive=True)

    ans = await reason_then_ground("Q?", present_as_answer=True)

    assert ans.presented is True
    assert ans.responsive is True
    # The headline is the DETERMINISTIC join of the grounded claims — never the analysis.
    assert ans.answer_headline == "claim A claim B"
    assert "reasoning prose" not in ans.answer_headline
    assert "RelevanceAssessment" in seen  # the responsiveness gate ran


@pytest.mark.asyncio
async def test_present_as_answer_non_responsive_falls_back(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    reranked = [_chunk("d#a")]
    _patch_stage1(monkeypatch, "Analysis.", reranked)
    _patch_extract(monkeypatch, [_claim("claim A", "d#a")])
    _patch_ground(
        monkeypatch, grounded=[0], responsive=False, relevance_reason="answers a related topic"
    )

    ans = await reason_then_ground("Q?", present_as_answer=True)

    assert ans.presented is False  # grounded, but not responsive → NOT presented as an answer
    assert ans.responsive is False
    assert ans.relevance_reason == "answers a related topic"
    # The grounded claim + analysis are still returned (the labelled-analysis fallback).
    assert [c.claim for c in ans.grounded_claims] == ["claim A"]
    assert ans.analysis == "Analysis."


@pytest.mark.asyncio
async def test_present_as_answer_zero_grounded_skips_gate(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero grounded → the responsiveness gate is NEVER called (nothing to present), and there is
    no refuse state — the labelled analysis is returned."""
    reranked = [_chunk("d#a")]
    _patch_stage1(monkeypatch, "Analysis.", reranked)
    _patch_extract(monkeypatch, [_claim("claim A", "d#a")])
    seen = _patch_ground(monkeypatch, grounded=[])

    ans = await reason_then_ground("Q?", present_as_answer=True)

    assert ans.presented is False
    assert ans.responsive is None  # gate not run
    assert "RelevanceAssessment" not in seen
    assert ans.analysis == "Analysis."


@pytest.mark.asyncio
async def test_present_as_answer_relevance_failure_fails_closed(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A responsiveness-gate ModelCallError fails CLOSED → not presented (falls back to the
    labelled analysis), never promoting an un-gated answer."""
    reranked = [_chunk("d#a")]
    _patch_stage1(monkeypatch, "Analysis.", reranked)
    _patch_extract(monkeypatch, [_claim("claim A", "d#a")])
    _patch_ground(monkeypatch, grounded=[0], relevance_fail=True)

    ans = await reason_then_ground("Q?", present_as_answer=True)

    assert ans.presented is False
    assert ans.responsive is False  # fail-closed sentinel
    assert [c.claim for c in ans.grounded_claims] == ["claim A"]  # subset still surfaced, labelled
    assert ans.analysis == "Analysis."


@pytest.mark.asyncio
async def test_standalone_default_never_runs_gate(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default path (present_as_answer=False) is byte-identical to before: the gate is never
    called, `responsive` stays None, and nothing is presented-as-answer."""
    reranked = [_chunk("d#a")]
    _patch_stage1(monkeypatch, "Analysis.", reranked)
    _patch_extract(monkeypatch, [_claim("claim A", "d#a")])
    seen = _patch_ground(monkeypatch, grounded=[0])

    ans = await reason_then_ground("Q?")

    assert ans.present_as_answer is False
    assert ans.presented is False
    assert ans.responsive is None
    assert ans.answer_headline == ""
    assert "RelevanceAssessment" not in seen  # standalone never runs the responsiveness gate


def test_bridge_isolated_from_ask_graph() -> None:
    """HARD-gate isolation: the `/ask` graph + the eval runner must NEVER import or construct the
    bridge — it is a fenced sibling, unreachable from `answer_query`/`run_eval`. Holds even with
    the present-as-answer escalation (ADR-0016): that reuses `assess_relevance`'s PROMPT + SCHEMA
    via `grounding.assess_responsiveness`, never the `/ask` graph node, so `answering` references
    neither the bridge nor a grounding import. If a future change wires the bridge into either,
    update this test on purpose (and re-examine the contract)."""
    import memex.agents.answering as answering_mod
    import memex.eval.runner as runner_mod

    for mod in (answering_mod, runner_mod):
        src = Path(mod.__file__).read_text()
        assert "reason_then_ground" not in src, f"{mod.__name__} references the bridge"
        assert "BridgeAnswer" not in src, f"{mod.__name__} references the bridge"
        assert "agents.bridge" not in src, f"{mod.__name__} imports the bridge"


@pytest.mark.asyncio
async def test_no_evidence_returns_analysis_only(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_stage1(monkeypatch, "Reasoned from expertise; vault was silent.", [])
    # extraction is SKIPPED when there are no reranked chunks; ground fake should never fire.
    _patch_ground(monkeypatch, grounded=[])

    ans = await reason_then_ground("Q?")

    assert ans.analysis == "Reasoned from expertise; vault was silent."
    assert ans.n_extracted == 0
    assert ans.grounded_claims == []
    assert ans.evidence == []
