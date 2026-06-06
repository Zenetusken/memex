"""The eval runner isolates a per-query failure into a DISTINCT error bucket.

Before 2026-06-06, `run_eval`'s `for q in queries` had no try/except, so one query's
`ModelCallError` (the slide-decks verify guided-decode truncation) aborted the WHOLE
suite. The fix records an errored query separately and excludes it from every metric —
crucially it must NOT be folded into "refused", which would false-pass
`refusal_rate_on_counterfactuals` on an errored counterfactual.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from memex.core.errors import ModelCallError
from memex.eval.runner import run_eval


@pytest.mark.asyncio
async def test_run_eval_isolates_per_query_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qs = tmp_path / "queries.json"
    qs.write_text(
        json.dumps(
            {
                "queries": [
                    # answerable → answers
                    {"qid": "ans-ok", "question": "good", "should_refuse": False,
                     "relevant_chunk_ids": []},
                    # counterfactual the agent WRONGLY answers → refusal_correct=False
                    {"qid": "cf-wrong", "question": "wrong", "should_refuse": True,
                     "relevant_chunk_ids": []},
                    # counterfactual whose answer_query RAISES → error bucket
                    {"qid": "cf-err", "question": "explode", "should_refuse": True,
                     "relevant_chunk_ids": []},
                ]
            }
        )
    )

    async def fake_answer_query(question: str, **_kw: object) -> object:
        if question == "explode":
            raise ModelCallError("verify guided-decode truncation")
        # both non-erroring queries ANSWER (cf-wrong is a counterfactual answered in error)
        return SimpleNamespace(answered=True, claims=[], refusal_reason=None)

    monkeypatch.setattr("memex.eval.runner.answer_query", fake_answer_query)

    report = await run_eval(qs)

    # The suite completes despite the raised query (no abort).
    assert report.query_count == 3
    assert report.error_count == 1
    # answered + refused + error == query_count; the errored query is NOT counted as refused
    assert report.answered_count == 2  # ans-ok + cf-wrong
    assert report.refused_count == 0  # cf-err is an ERROR, not a refusal
    # The decisive check: cf-err is EXCLUDED from refusal_cf. Only cf-wrong is an
    # evaluated counterfactual and it (wrongly) answered → 0/1 = 0.0. If cf-err were
    # folded into "refused" it would false-pass as a correct refusal → 1/2 = 0.5.
    assert report.refusal_rate_on_counterfactuals == 0.0

    boom = next(r for r in report.per_query if r.qid == "cf-err")
    assert boom.error is not None
    assert not boom.answered
    assert not boom.refusal_correct  # must not be recorded as a correct refusal
