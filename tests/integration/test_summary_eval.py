"""Integration test for the summary-eval runner (ADR-0008).

Fakes `summarize_document` so `run_summary_eval` is exercised end-to-end over a
tmp query set without a model: it must compute summarize/refuse correctness, the
mention-recall, and — the HARD gate — count any `must_not_assert` leak.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from memex.agents.answering import CitedClaim, FinalResponse, SectionSummary
from memex.eval import runner as run_mod


def _final(*, answered: bool) -> FinalResponse:
    if not answered:
        return FinalResponse(
            answered=False,
            refusal_reason="no indexed content",
            correlation_id="01CID",
            tokens_used=0,
            nodes_traversed=1,
            regenerate_attempts=0,
        )
    return FinalResponse(
        answered=True,
        summary="GTE is a general text embedding model trained with contrastive learning.",
        claims=[
            CitedClaim(
                claim="It uses multi-stage contrastive learning.",
                source_chunk_id="d#1",
                confidence="high",
            )
        ],
        sections=[
            SectionSummary(section_title="Intro", digest="An overview of embeddings.", key_points=[])
        ],
        correlation_id="01CID",
        tokens_used=10,
        nodes_traversed=2,
        regenerate_attempts=0,
    )


async def _fake_summarize(doc_id: str, **_kw: Any) -> FinalResponse:
    return _final(answered=doc_id != "empty-doc")


@pytest.mark.asyncio
async def test_run_summary_eval_scores_recall_leaks_and_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query_set = tmp_path / "summary.json"
    query_set.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "doc_id": "gte",
                        "must_mention": ["contrastive", "embedding"],
                        "must_not_assert": ["reinforcement learning"],
                        "should_summarize": True,
                    },
                    {
                        # The fake's text DOES say "contrastive" → a must_not_assert leak.
                        "doc_id": "leak",
                        "must_not_assert": ["contrastive"],
                        "should_summarize": True,
                    },
                    {"doc_id": "empty-doc", "should_summarize": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_mod, "summarize_document", _fake_summarize)

    report = await run_mod.run_summary_eval(query_set)

    assert report.case_count == 3
    assert report.summarized_count == 2  # gte + leak answered; empty-doc refused
    assert report.summarize_correct_count == 3  # all three matched their label
    assert report.hallucination_count == 1  # the "leak" case leaked "contrastive"
    # answered-only recall: gte 1.0 (both terms present) + leak 1.0 (no must_mention)
    assert report.mean_mention_recall == 1.0
    leak = next(r for r in report.per_doc if r.doc_id == "leak")
    assert leak.violations == ["contrastive"]
    empty = next(r for r in report.per_doc if r.doc_id == "empty-doc")
    assert empty.summarize_correct and not empty.summarized
