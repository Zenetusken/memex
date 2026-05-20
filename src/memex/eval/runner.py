"""Eval runner — drives the agent over a query set and scores the output.

The full 125-document corpus described in `docs/eval-corpus-plan.md` is
a separate deliverable on its own roadmap; this runner is the engine
that consumes whatever ground-truth `tests/evals/<category>/queries.json`
lays down. For Phase 2 we ship the framework and a single smoke test
that exercises the harness against a synthetic three-doc corpus.

`memex eval --quick` runs a sampled subset for fast iteration.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

from memex.agents.answering import answer_query
from memex.eval.scoring import (
    CitationPrecisionInput,
    citation_precision,
)

logger = structlog.get_logger(__name__)


class EvalQuery(BaseModel):
    """One eval entry: the question, optionally with ground-truth chunk ids
    and a flag for "this should be refused" counterfactuals.
    """

    qid: str
    question: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    should_refuse: bool = False


class EvalQueryResult(BaseModel):
    qid: str
    question: str
    answered: bool
    citation_precision: float
    refusal_correct: bool
    refusal_reason: str | None = None


class EvalReport(BaseModel):
    run_id: str
    started_at: datetime
    finished_at: datetime
    query_count: int
    answered_count: int
    refused_count: int
    mean_citation_precision: float
    refusal_rate_on_counterfactuals: float
    per_query: list[EvalQueryResult] = Field(default_factory=list)


def _load_queries(query_set_path: Path) -> list[EvalQuery]:
    payload = json.loads(query_set_path.read_text(encoding="utf-8"))
    return [EvalQuery(**q) for q in payload["queries"]]


async def run_eval(
    query_set: Path,
    *,
    quick: bool = False,
    sample_fraction: float = 0.2,
) -> EvalReport:
    """Run every query in `query_set` against the configured agent.

    With `quick=True` the runner samples `sample_fraction` of queries
    deterministically (every Nth entry) so the same `--quick` call
    always evaluates the same slice — invaluable for triaging.
    """
    import ulid

    queries = _load_queries(query_set)
    if quick:
        step = max(1, int(1 / sample_fraction))
        queries = queries[::step]

    started = datetime.now(UTC)
    log = logger.bind(query_count=len(queries), quick=quick)
    log.info("eval.start")

    results: list[EvalQueryResult] = []
    counterfactual_count = 0
    refusal_correct_count = 0

    for q in queries:
        response = await answer_query(q.question)
        cp = citation_precision(
            CitationPrecisionInput(
                cited_chunk_ids=[c.source_chunk_id for c in response.claims],
                relevant_chunk_ids=set(q.relevant_chunk_ids),
            )
        )
        refusal_correct = q.should_refuse == (not response.answered)
        if q.should_refuse:
            counterfactual_count += 1
            if refusal_correct:
                refusal_correct_count += 1
        results.append(
            EvalQueryResult(
                qid=q.qid,
                question=q.question,
                answered=response.answered,
                citation_precision=cp,
                refusal_correct=refusal_correct,
                refusal_reason=response.refusal_reason,
            )
        )

    finished = datetime.now(UTC)
    report = EvalReport(
        run_id=str(ulid.ULID()),
        started_at=started,
        finished_at=finished,
        query_count=len(results),
        answered_count=sum(1 for r in results if r.answered),
        refused_count=sum(1 for r in results if not r.answered),
        mean_citation_precision=(
            sum(r.citation_precision for r in results) / len(results)
            if results
            else 0.0
        ),
        refusal_rate_on_counterfactuals=(
            refusal_correct_count / counterfactual_count
            if counterfactual_count
            else 1.0
        ),
        per_query=results,
    )
    log.info(
        "eval.done",
        query_count=report.query_count,
        mean_citation_precision=report.mean_citation_precision,
    )
    return report
