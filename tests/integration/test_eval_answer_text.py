"""The answer-TEXT-correctness threading in `run_eval` (`answer_must_mention`, audit-14).

`citation_precision` grades CITED CHUNKS, never the answer text — which hid the usage-class
wrong-answer class. The runner grades the text ONLY for annotated + answered + non-error
queries: `response.summary` must be accessed lazily inside that gate (a minimal fake with
only answered/claims/refusal_reason — the resilience-test shape — must stay valid on
un-annotated queries), the graded text is summary + joined claim texts, and the report
aggregates exclude every ungraded query.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from memex.core.errors import ModelCallError
from memex.eval.runner import run_eval


def _write_queries(tmp_path: Path, queries: list[dict[str, object]]) -> Path:
    qs = tmp_path / "queries.json"
    qs.write_text(json.dumps({"queries": queries}))
    return qs


def _claim(text: str, chunk_id: str = "d#1") -> SimpleNamespace:
    return SimpleNamespace(claim=text, source_chunk_id=chunk_id)


@pytest.mark.asyncio
async def test_annotated_answered_query_is_graded_and_aggregated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qs = _write_queries(
        tmp_path,
        [
            {
                "qid": "use-1",
                "question": "Which function calls is_known_safe_command?",
                "answer_must_mention": ["assess_command_safety"],
            },
            {
                "qid": "use-2",
                "question": "Which function calls apply_hunks?",
                "answer_must_mention": ["apply_patch", "missing_token"],
            },
        ],
    )

    async def fake_answer_query(question: str, **_kw: object) -> object:
        if "is_known_safe_command" in question:
            return SimpleNamespace(
                answered=True,
                summary="is_known_safe_command is called in assess_command_safety.",
                claims=[_claim("assess_command_safety calls it for auto-approval.")],
                refusal_reason=None,
            )
        return SimpleNamespace(
            answered=True,
            summary="apply_patch calls apply_hunks to apply the parsed hunks.",
            claims=[],
            refusal_reason=None,
        )

    monkeypatch.setattr("memex.eval.runner.answer_query", fake_answer_query)
    report = await run_eval(qs)

    r1 = next(r for r in report.per_query if r.qid == "use-1")
    assert r1.answer_mention_recall == 1.0
    assert r1.answer_text_correct is True
    r2 = next(r for r in report.per_query if r.qid == "use-2")
    assert r2.answer_mention_recall == 0.5  # apply_patch found, missing_token not
    assert r2.answer_text_correct is False
    assert report.answer_text_checked == 2
    assert report.answer_text_correct_count == 1
    assert report.mean_answer_mention_recall == 0.75


@pytest.mark.asyncio
async def test_slot_found_only_in_a_claim_still_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the claims-joined answer-text choice: the partial-grounded path rebuilds the
    summary FROM claims, and the key fact often lives in a claim, not the headline."""
    qs = _write_queries(
        tmp_path,
        [
            {
                "qid": "claim-only",
                "question": "Which function creates a RolloutRecorder?",
                "answer_must_mention": ["submission_loop"],
            }
        ],
    )

    async def fake_answer_query(question: str, **_kw: object) -> object:
        return SimpleNamespace(
            answered=True,
            summary="The recorder is created during session startup.",  # slot NOT here
            claims=[_claim("submission_loop constructs the RolloutRecorder.")],
            refusal_reason=None,
        )

    monkeypatch.setattr("memex.eval.runner.answer_query", fake_answer_query)
    report = await run_eval(qs)
    assert report.per_query[0].answer_mention_recall == 1.0
    assert report.per_query[0].answer_text_correct is True


@pytest.mark.asyncio
async def test_unannotated_refused_and_errored_queries_are_not_graded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qs = _write_queries(
        tmp_path,
        [
            # un-annotated → never graded, even though it answers; the fake for this
            # query deliberately has NO `.summary` (the resilience-test minimal shape) —
            # lazy access means no AttributeError.
            {"qid": "plain", "question": "plain"},
            # annotated but REFUSED → not graded (refusal accounting covers it)
            {
                "qid": "refused",
                "question": "refused",
                "answer_must_mention": ["whatever"],
            },
            # annotated but answer_query RAISES → error bucket, not graded
            {
                "qid": "boom",
                "question": "explode",
                "answer_must_mention": ["whatever"],
            },
        ],
    )

    async def fake_answer_query(question: str, **_kw: object) -> object:
        if question == "explode":
            raise ModelCallError("transient")
        if question == "refused":
            return SimpleNamespace(answered=False, claims=[], refusal_reason="no evidence")
        # the MINIMAL shape — no `.summary` attribute at all
        return SimpleNamespace(answered=True, claims=[], refusal_reason=None)

    monkeypatch.setattr("memex.eval.runner.answer_query", fake_answer_query)
    report = await run_eval(qs)

    for r in report.per_query:
        assert r.answer_mention_recall is None, r.qid
        assert r.answer_text_correct is None, r.qid
    assert report.answer_text_checked == 0
    assert report.answer_text_correct_count == 0
    assert report.mean_answer_mention_recall is None
    # the rest of the report is unaffected
    assert report.error_count == 1
    assert report.answered_count == 1


@pytest.mark.asyncio
async def test_fully_unannotated_corpus_is_value_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing corpus (no `answer_must_mention` anywhere) gets checked=0 / mean=None
    and IDENTICAL legacy metric values — the additive-fields contract."""
    qs = _write_queries(
        tmp_path,
        [
            {"qid": "a", "question": "a", "relevant_chunk_ids": ["d#1"]},
            {"qid": "cf", "question": "cf", "should_refuse": True},
        ],
    )

    async def fake_answer_query(question: str, **_kw: object) -> object:
        if question == "cf":
            return SimpleNamespace(answered=False, claims=[], refusal_reason="counterfactual")
        return SimpleNamespace(answered=True, claims=[_claim("fact.", "d#1")], refusal_reason=None)

    monkeypatch.setattr("memex.eval.runner.answer_query", fake_answer_query)
    report = await run_eval(qs)

    assert report.answer_text_checked == 0
    assert report.mean_answer_mention_recall is None
    assert report.answered_count == 1
    assert report.refused_count == 1
    assert report.refusal_rate_on_counterfactuals == 1.0
    assert report.mean_citation_precision == 1.0
    assert not math.isnan(report.mean_citation_precision_answered_only)
