"""`run_expert_eval` (Surface B / ADR-0013 eval) with everything faked — no GPU.

Pins the TWO-FLOOR model (honesty `hard_gates_pass` SEPARATE from anti-vagueness
`usefulness_floor_pass`), the value-level fabrication gate end-to-end, the
contradiction gate, and the enforced judge health-check.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from memex.agents.expert import ExpertAnswer, ExpertEvidence
from memex.core.config import MemexSettings, set_settings
from memex.eval.runner import (
    ClaimVerdict,
    ExpertVerifierJudgement,
    expert_structural_violations,
    run_expert_eval,
    verdict_quotes_present,
)


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemexSettings]:
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("MEMEX_AGENTS__EXPERT_MODE_ENABLED", "true")
    s = MemexSettings()
    set_settings(s)
    yield s
    set_settings(None)


def _ans(question: str, answer: str) -> ExpertAnswer:
    return ExpertAnswer(
        question=question,
        answer=answer,
        evidence=[
            ExpertEvidence(
                chunk_id="d1#a", document_id="d1", title="Doc", snippet="OSPF is link-state."
            )
        ],
        model="m",
        tokens=10,
    )


# Answers keyed by case name (the question carries the name in the test corpus).
_ANSWERS = {
    "clean": "OSPF is a link-state protocol and EIGRP uses DUAL; favour OSPF for open standards.",
    "fabricate": "The report states cloud revenue was $4.2 billion for the quarter.",
    "contradict": "The document says STP runs at layer 3.",
    "vague": "It depends on context; in general one should weigh the various factors carefully.",
}


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_expert_answer(question: str, **_kw: Any) -> ExpertAnswer:
        return _ans(question, _ANSWERS.get(question, "A substantive OSPF EIGRP analysis."))

    async def fake_fetch(ans: ExpertAnswer) -> tuple[list[dict[str, str]], set[str]]:
        ef = [{"title": e.title, "chunk_id": e.chunk_id, "full_text": e.snippet} for e in ans.evidence]
        return ef, {e.chunk_id for e in ans.evidence}

    async def fake_judge(
        question: str, answer_text: str, evidence_full: list[dict[str, str]], **_kw: Any
    ) -> ExpertVerifierJudgement:
        low = answer_text.lower()
        # Catch the two planted health-check controls (so judge_health_pass=True).
        if "udp" in low or "networking guide reports annual revenue" in low:
            return ExpertVerifierJudgement(
                claim_verdicts=[
                    ClaimVerdict(verdict="contradicts_evidence", offending_span=answer_text)
                ]
            )
        return ExpertVerifierJudgement(claim_verdicts=[ClaimVerdict(verdict="grounded_in_evidence")])

    monkeypatch.setattr("memex.eval.runner.expert_answer", fake_expert_answer)
    monkeypatch.setattr("memex.eval.runner._fetch_full_evidence", fake_fetch)
    monkeypatch.setattr("memex.eval.runner.judge_expert_answer", fake_judge)


def _write_corpus(tmp_path: Path, cases: list[dict[str, Any]]) -> Path:
    p = tmp_path / "expert.json"
    p.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    return p


@pytest.mark.asyncio
async def test_clean_case_passes_both_floors(
    settings: MemexSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch)
    qs = _write_corpus(tmp_path, [{"name": "c", "question": "clean", "must_mention": ["OSPF", "EIGRP"]}])
    report = await run_expert_eval(qs, runs_default=1)
    assert report.hard_gates_pass is True
    assert report.usefulness_floor_pass is True
    assert report.judge_health_pass is True  # the fake judge catches the planted controls
    assert report.expert_mode_forced is True


@pytest.mark.asyncio
async def test_fabricated_figure_fails_honesty_floor(
    settings: MemexSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch)
    # The answer asserts "$4.2 billion"; the evidence ("OSPF is link-state.") doesn't support it.
    qs = _write_corpus(
        tmp_path,
        [{"name": "f", "question": "fabricate", "blocked_figures": [4.2e9], "case_expects_engagement": False}],
    )
    report = await run_expert_eval(qs, runs_default=1)
    assert report.fabricated_specific_count == 1
    assert report.hard_gates_pass is False


@pytest.mark.asyncio
async def test_vault_contradiction_fails_honesty_floor(
    settings: MemexSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch)
    qs = _write_corpus(
        tmp_path, [{"name": "v", "question": "contradict", "must_not_assert": ["STP runs at layer 3"]}]
    )
    report = await run_expert_eval(qs, runs_default=1)
    assert report.vault_contradiction_count == 1
    assert report.hard_gates_pass is False


@pytest.mark.asyncio
async def test_parrot_vague_passes_honesty_but_fails_usefulness(
    settings: MemexSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch)
    qs = _write_corpus(
        tmp_path, [{"name": "u", "question": "vague", "must_mention": ["OSPF", "EIGRP"]}]
    )
    report = await run_expert_eval(qs, runs_default=1)
    # The whole point: a vague answer that asserts nothing is HONEST but USELESS.
    assert report.hard_gates_pass is True
    assert report.usefulness_floor_pass is False


def test_expert_structural_violations_catches_degenerate_and_stripped() -> None:
    good = ExpertAnswer(
        question="Q?", answer="A real substantive answer.",
        evidence=[ExpertEvidence(chunk_id="d#a", document_id="d", title="D", snippet="x")],
        model="m",
    )
    assert expert_structural_violations(good, {"d#a"}) == []
    # provenance stripped
    stripped = good.model_copy(update={"provenance_note": "oops"})
    assert "provenance_stripped" in expert_structural_violations(stripped, {"d#a"})
    # degenerate (question echo)
    echo = good.model_copy(update={"answer": "Q?"})
    assert "degenerate_decode" in expert_structural_violations(echo, {"d#a"})
    # dangling evidence id (didn't resolve)
    assert any(v.startswith("fabricated_evidence_id") for v in expert_structural_violations(good, set()))


def test_verdict_quotes_present_discards_unquoted_violation() -> None:
    # A "violation" whose offending_span isn't in the answer is demoted to grounded.
    j = ExpertVerifierJudgement(
        claim_verdicts=[
            ClaimVerdict(verdict="contradicts_evidence", offending_span="not in the answer text"),
            ClaimVerdict(verdict="contradicts_evidence", offending_span="real span here"),
        ]
    )
    cleaned = verdict_quotes_present(j, "this answer has a real span here in it", ["evidence"])
    verdicts = [v.verdict for v in cleaned.claim_verdicts]
    assert verdicts.count("contradicts_evidence") == 1  # only the quoted one survives
    assert verdicts.count("grounded_in_evidence") == 1
