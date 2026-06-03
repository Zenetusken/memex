"""Reason-then-ground bridge (`agents/bridge.py`, Surface §11) — heavy deps faked.

Pins the v1 CONTRACT with no GPU: only claims that survive the UNCHANGED verify gate are
presented as grounded; a claim whose cited id can't resolve to a reranked chunk is dropped
DETERMINISTICALLY before grounding; zero grounded returns the labelled analysis with no citation
chrome (never a refusal); an extraction OR grounding failure fails open to the analysis-only path.

The three faked seams: `bridge.reason_over_evidence` (Stage 1 — bypasses retrieval/rerank/reason),
`bridge.complete_structured` (Stage 1.5 — claim extraction), `grounding.complete_structured`
(Stage 2 — the per-claim verify). `repair_claim_chunk_ids`, the deterministic id filter, and
`ground_claims`/`ground_claims_isolated`' keep-rule all run for real.

The verify fake (`_patch_ground`) is claim-IDENTITY-aware (it parses the rendered prompt's Draft
lines), so it is correct under BOTH the batched gate AND the default ISOLATED re-verification (the
batch-leniency fix, 2026-06-03): each isolated call sees ONE claim and the fake returns `[0]`/`[]`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
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


# Matches a rendered `verify_grounding/v2` Draft line `- [N] claim text` (the claim section).
_CLAIM_LINE = re.compile(r"^- \[(\d+)\] (.+)$", re.MULTILINE)


def _patch_ground(
    monkeypatch: pytest.MonkeyPatch,
    *,
    grounded_keys: set[str] | None = None,
    grounds: Callable[[str, list[str], str], bool] | None = None,
    fail: bool = False,
    fail_keys: set[str] | None = None,
    responsive: bool = True,
    relevance_reason: str = "addresses the question",
    relevance_fail: bool = False,
) -> list[str]:
    """Fake `grounding.complete_structured` — serves BOTH the per-claim verify (VerificationResult)
    AND the present-as-answer responsiveness gate (RelevanceAssessment), discriminated by schema.

    The VerificationResult branch is **claim-IDENTITY-aware**: it parses the rendered prompt's Draft
    `- [i] text` lines and decides each claim INDEPENDENTLY, so it is correct under BOTH the batched
    gate (N claims → several indices) AND isolated re-verification (1 claim per call → `[0]`/`[]`).
    Grounding is decided per claim by `grounded_keys` (a set of claim texts that ground) or the more
    flexible `grounds(text, present, prompt)` predicate (e.g. "ground iff a sibling is present", to
    SIMULATE the batch-leniency). `fail_keys` raises a verify CALL iff any present claim is a
    fail-key (isolated → only that claim's call fails; batched → the whole call fails). Returns the
    list of schemas seen, so a test can assert the responsiveness gate did/didn't fire."""
    seen: list[str] = []

    def _decide(text: str, present: list[str], prompt: str) -> bool:
        if grounds is not None:
            return grounds(text, present, prompt)
        return text in (grounded_keys or set())

    async def _verify(*, schema: type, **_kw: object) -> tuple[Any, int]:
        if schema is RelevanceAssessment:
            seen.append("RelevanceAssessment")
            if relevance_fail:
                raise ModelCallError("relevance boom")
            return RelevanceAssessment(responsive=responsive, reason=relevance_reason), 4
        seen.append("VerificationResult")
        if fail:
            raise ModelCallError("verify boom")
        prompt_arg = _kw.get("prompt", "")
        prompt = prompt_arg if isinstance(prompt_arg, str) else str(prompt_arg)
        pairs = [(int(i), t.strip()) for i, t in _CLAIM_LINE.findall(prompt)]
        present = [t for _i, t in pairs]
        if fail_keys and any(t in fail_keys for t in present):
            raise ModelCallError("verify boom (per-claim)")
        grounded = [i for i, t in pairs if _decide(t, present, prompt)]
        ungrounded = [i for i, t in pairs if not _decide(t, present, prompt)]
        return VerificationResult(grounded=grounded, ungrounded=ungrounded), 8

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
    _patch_ground(monkeypatch, grounded_keys={"claim A", "claim C"})

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
    _patch_ground(monkeypatch, grounded_keys=set())

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
    reranked chunk BEFORE grounding. Proven by making the verifier WILLING to ground both claims:
    if the bad claim reached grounding, n_grounded would be 2; the filter makes it 1."""
    reranked = [_chunk("d#a")]
    _patch_stage1(monkeypatch, "Analysis.", reranked)
    _patch_extract(
        monkeypatch,
        [_claim("good claim", "d#a"), _claim("fabricated claim", "ghostdoc#zzzzzzz")],
    )
    _patch_ground(monkeypatch, grounded_keys={"good claim", "fabricated claim"})  # both willing

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
    _patch_ground(monkeypatch, grounded_keys=set())

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
    """A single claim whose verify call raises drops to zero grounded — never ship ungrounded.
    (Holds in both modes: isolated runs one call which raises; batched raises the only call.)"""
    reranked = [_chunk("d#a")]
    _patch_stage1(monkeypatch, "Analysis.", reranked)
    _patch_extract(monkeypatch, [_claim("claim A", "d#a")])
    _patch_ground(monkeypatch, grounded_keys={"claim A"}, fail=True)  # the verify call raises

    ans = await reason_then_ground("Q?")

    assert ans.n_grounded == 0  # fail-open: never ship ungrounded on a grounding failure
    assert ans.grounded_claims == []
    assert ans.analysis == "Analysis."


@pytest.mark.asyncio
async def test_isolated_per_claim_fail_open(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isolated mode's deliberate contract: one claim's verify raises → that claim drops, its
    SIBLINGS still ground (graceful per-claim degradation — the bridge has no refuse state)."""
    reranked = [_chunk("d#a"), _chunk("d#b")]
    _patch_stage1(monkeypatch, "Analysis.", reranked)
    _patch_extract(monkeypatch, [_claim("claim A", "d#a"), _claim("claim B", "d#b")])
    _patch_ground(monkeypatch, grounded_keys={"claim A", "claim B"}, fail_keys={"claim A"})

    ans = await reason_then_ground("Q?")

    assert ans.n_grounded == 1  # A's isolated call raised → A dropped; B still grounds
    assert [c.claim for c in ans.grounded_claims] == ["claim B"]


@pytest.mark.asyncio
async def test_batched_fail_open_drops_all(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kill-switch OFF (batched): a verify failure drops ALL claims (the original all-or-nothing
    fail-open) — the contrast to isolated per-claim fail-open above."""
    settings.agents.bridge_isolated_grounding_enabled = False
    reranked = [_chunk("d#a"), _chunk("d#b")]
    _patch_stage1(monkeypatch, "Analysis.", reranked)
    _patch_extract(monkeypatch, [_claim("claim A", "d#a"), _claim("claim B", "d#b")])
    _patch_ground(monkeypatch, grounded_keys={"claim A", "claim B"}, fail_keys={"claim A"})

    ans = await reason_then_ground("Q?")

    assert ans.n_grounded == 0  # one fail-key claim fails the whole batched call → both dropped


@pytest.mark.asyncio
async def test_extraction_failure_fails_open(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    reranked = [_chunk("d#a")]
    _patch_stage1(monkeypatch, "Analysis.", reranked)
    _patch_extract(monkeypatch, [], fail=True)  # extraction raises
    _patch_ground(monkeypatch, grounded_keys={"claim A"})

    ans = await reason_then_ground("Q?")

    assert ans.n_extracted == 0
    assert ans.n_grounded == 0
    assert ans.analysis == "Analysis."


# --- Isolated re-verification (the batch-leniency fix, 2026-06-03): each bridge claim is verified
# ALONE so the `verify_grounding/v2` gate can't rubber-stamp a plausible claim on batch coherence. ---


@pytest.mark.asyncio
async def test_isolated_grounding_drops_batch_only_claim(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The decisive proof: a claim that grounds ONLY when batched with a sibling (the fake grounds
    a claim iff >1 claim is present in the prompt — the batch-leniency simulacrum) is DROPPED under
    default-on isolated grounding (each claim is verified alone → present == [self])."""
    reranked = [_chunk("d#a"), _chunk("d#b")]
    _patch_stage1(monkeypatch, "Analysis.", reranked)
    _patch_extract(monkeypatch, [_claim("batch-only claim", "d#a"), _claim("sibling", "d#b")])
    _patch_ground(monkeypatch, grounds=lambda _text, present, _prompt: len(present) > 1)

    ans = await reason_then_ground("Q?")

    assert ans.n_grounded == 0  # isolated: each claim alone → the batch-only survivor is dropped
    assert ans.grounded_claims == []


@pytest.mark.asyncio
async def test_isolated_grounding_kill_switch_reverts_to_batch(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`bridge_isolated_grounding_enabled=False` reverts to the batched gate — the same batch-only
    claim is now KEPT (the prior, lenient behavior). The kill-switch is the instant revert."""
    settings.agents.bridge_isolated_grounding_enabled = False
    reranked = [_chunk("d#a"), _chunk("d#b")]
    _patch_stage1(monkeypatch, "Analysis.", reranked)
    _patch_extract(monkeypatch, [_claim("batch-only claim", "d#a"), _claim("sibling", "d#b")])
    _patch_ground(monkeypatch, grounds=lambda _text, present, _prompt: len(present) > 1)

    ans = await reason_then_ground("Q?")

    assert ans.n_grounded == 2  # batched: both present together → both ground (the old leniency)
    assert {c.claim for c in ans.grounded_claims} == {"batch-only claim", "sibling"}


@pytest.mark.asyncio
async def test_isolated_passes_all_chunks_not_cited_only(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isolated grounding passes ALL reranked chunks to each per-claim verify (NOT cited-chunk-only),
    so a claim whose support is in a SIBLING chunk still grounds. The fake grounds a claim iff the
    sibling chunk id `d#b` appears in the rendered prompt — proving all chunks were passed (guards
    against a future 'optimize to cited-chunk-only' regression that would false-negative)."""
    reranked = [_chunk("d#a", text="fragment"), _chunk("d#b", text="the supporting sentence")]
    _patch_stage1(monkeypatch, "Analysis.", reranked)
    _patch_extract(monkeypatch, [_claim("needs sibling", "d#a")])
    _patch_ground(monkeypatch, grounds=lambda _text, _present, prompt: "d#b" in prompt)

    ans = await reason_then_ground("Q?")

    assert ans.n_grounded == 1  # the sibling chunk was in the isolated prompt → grounded
    assert [c.claim for c in ans.grounded_claims] == ["needs sibling"]


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
    seen = _patch_ground(monkeypatch, grounded_keys={"claim A", "claim B"}, responsive=True)

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
        monkeypatch,
        grounded_keys={"claim A"},
        responsive=False,
        relevance_reason="answers a related topic",
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
    seen = _patch_ground(monkeypatch, grounded_keys=set())

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
    _patch_ground(monkeypatch, grounded_keys={"claim A"}, relevance_fail=True)

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
    seen = _patch_ground(monkeypatch, grounded_keys={"claim A"})

    ans = await reason_then_ground("Q?")

    assert ans.present_as_answer is False
    assert ans.presented is False
    assert ans.responsive is None
    assert ans.answer_headline == ""
    assert "RelevanceAssessment" not in seen  # standalone never runs the responsiveness gate


# --- Name-only GROUNDING backstop (ADR-0016 audit rec 1, upgraded 2026-06-03): the bridge DEMOTES
# from `grounded` any BEHAVIORAL claim cited to a chunk that merely NAMES the entity (a bare
# list/heading) — the SAME deterministic, MEMBERSHIP-AWARE rule (`claim_grounded_only_by_name`) the
# `/ask` verify node uses. It runs BEFORE the present/standalone split, so it shrinks `grounded_claims`
# itself (footer counts + labelled fallback + BOTH surfaces). Membership claims a name-list genuinely
# grounds are KEPT. The present-as-answer guard is kept as the membership-aware defense-in-depth layer. ---

_NAME_ONLY_TEXT = "### Contrôle d'accès\n- Role-Based Access Control (RBAC)\n- Attribute-Based Access Control (ABAC)"
_BEHAVIORAL_CLAIM = "RBAC assigns permissions based on a user's job function."
_MEMBERSHIP_CLAIM = "RBAC is one of the listed access-control models."


@pytest.mark.asyncio
async def test_name_only_behavioral_claim_demoted_from_grounded(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kill target: a BEHAVIORAL claim cited to a name-only chunk is now dropped from
    `grounded_claims` ITSELF (not just presentation); a prose-cited claim survives."""
    reranked = [_chunk("d#a", text=_NAME_ONLY_TEXT), _chunk("d#b", text="prose " * 12)]
    _patch_stage1(monkeypatch, "Analysis.", reranked)
    _patch_extract(monkeypatch, [_claim(_BEHAVIORAL_CLAIM, "d#a"), _claim("prose claim", "d#b")])
    _patch_ground(monkeypatch, grounded_keys={_BEHAVIORAL_CLAIM, "prose claim"}, responsive=True)

    ans = await reason_then_ground("Q?", present_as_answer=True)

    assert [c.claim for c in ans.grounded_claims] == ["prose claim"]  # behavioral name-only DROPPED
    assert [c.claim for c in ans.presented_claims] == ["prose claim"]
    assert ans.n_grounded == 1  # footer count reflects the demotion
    assert ans.presented is True


@pytest.mark.asyncio
async def test_name_only_membership_claim_kept_in_grounded(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The over-suppression fix: a MEMBERSHIP claim a name-list genuinely grounds STAYS in
    `grounded_claims` AND is PRESENTED (membership-first KEEP at both the backstop and the guard)."""
    reranked = [_chunk("d#a", text=_NAME_ONLY_TEXT)]
    _patch_stage1(monkeypatch, "Analysis.", reranked)
    _patch_extract(monkeypatch, [_claim(_MEMBERSHIP_CLAIM, "d#a")])
    _patch_ground(monkeypatch, grounded_keys={_MEMBERSHIP_CLAIM}, responsive=True)

    ans = await reason_then_ground("Q?", present_as_answer=True)

    assert [c.claim for c in ans.grounded_claims] == [_MEMBERSHIP_CLAIM]
    assert [c.claim for c in ans.presented_claims] == [_MEMBERSHIP_CLAIM]
    assert ans.presented is True


@pytest.mark.asyncio
async def test_name_only_behavioral_demoted_on_standalone(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The NEW coverage: the STANDALONE `/bridge` (present_as_answer defaults False) ALSO drops a
    behavioral name-only claim — the residual the old presentation-only guard never reached here."""
    reranked = [_chunk("d#a", text=_NAME_ONLY_TEXT), _chunk("d#b", text="prose " * 12)]
    _patch_stage1(monkeypatch, "Analysis.", reranked)
    _patch_extract(monkeypatch, [_claim(_BEHAVIORAL_CLAIM, "d#a"), _claim("prose claim", "d#b")])
    _patch_ground(monkeypatch, grounded_keys={_BEHAVIORAL_CLAIM, "prose claim"})

    ans = await reason_then_ground("Q?")  # standalone

    assert ans.present_as_answer is False
    assert [c.claim for c in ans.grounded_claims] == ["prose claim"]  # demoted on standalone too


@pytest.mark.asyncio
async def test_all_name_only_behavioral_falls_back_and_skips_gate(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the only grounded claim is behavioral-name-only, the backstop empties `grounded` →
    nothing to present → the responsiveness gate is SKIPPED and the surface falls back to analysis."""
    reranked = [_chunk("d#a", text=_NAME_ONLY_TEXT)]
    _patch_stage1(monkeypatch, "Analysis.", reranked)
    _patch_extract(monkeypatch, [_claim(_BEHAVIORAL_CLAIM, "d#a")])
    seen = _patch_ground(monkeypatch, grounded_keys={_BEHAVIORAL_CLAIM}, responsive=True)

    ans = await reason_then_ground("Q?", present_as_answer=True)

    assert ans.presented is False
    assert ans.responsive is None  # gate skipped — grounded emptied by the backstop
    assert ans.grounded_claims == []
    assert ans.presented_claims == []
    assert "RelevanceAssessment" not in seen


@pytest.mark.asyncio
async def test_name_only_guard_kill_switch_off(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`bridge_name_only_guard_enabled=False` reverts BOTH the grounding demotion and the guard — a
    behavioral name-only claim stays grounded and is presented (the prior behavior)."""
    settings.agents.bridge_name_only_guard_enabled = False
    reranked = [_chunk("d#a", text=_NAME_ONLY_TEXT)]
    _patch_stage1(monkeypatch, "Analysis.", reranked)
    _patch_extract(monkeypatch, [_claim(_BEHAVIORAL_CLAIM, "d#a")])
    _patch_ground(monkeypatch, grounded_keys={_BEHAVIORAL_CLAIM}, responsive=True)

    ans = await reason_then_ground("Q?", present_as_answer=True)

    assert [c.claim for c in ans.grounded_claims] == [_BEHAVIORAL_CLAIM]  # not demoted
    assert ans.presented is True
    assert [c.claim for c in ans.presented_claims] == [_BEHAVIORAL_CLAIM]


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
    _patch_ground(monkeypatch, grounded_keys=set())

    ans = await reason_then_ground("Q?")

    assert ans.analysis == "Reasoned from expertise; vault was silent."
    assert ans.n_extracted == 0
    assert ans.grounded_claims == []
    assert ans.evidence == []
