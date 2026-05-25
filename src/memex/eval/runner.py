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
from typing import Any, cast

import structlog
from pydantic import BaseModel, Field

from memex.agents.answering import answer_query
from memex.core.config import get_settings
from memex.eval.scoring import (
    CitationPrecisionInput,
    ParseQualityScores,
    citation_precision,
    score_parse_quality,
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
    """Per-query verdict from a single eval run — whether the agent
    answered, the citation precision against the labeled chunks, and
    (for refusals) the refusal-correctness flag against the
    `should_refuse` label."""

    qid: str
    question: str
    answered: bool
    citation_precision: float
    refusal_correct: bool
    refusal_reason: str | None = None


class EvalReport(BaseModel):
    """Aggregate report from one eval invocation — counts, the two
    `mean_citation_precision` variants (all queries vs answered-only),
    refusal-rate-on-counterfactuals, and the per-query verdicts."""

    run_id: str
    started_at: datetime
    finished_at: datetime
    query_count: int
    answered_count: int
    refused_count: int
    mean_citation_precision: float
    # Refused queries score 1.0 on `mean_citation_precision` because
    # they cite nothing (no false positives). That inflates the
    # headline when most queries are refusals. The "answered-only"
    # variant is the honest signal of citation quality on the queries
    # the agent actually attempted. NaN when zero queries answered.
    mean_citation_precision_answered_only: float
    refusal_rate_on_counterfactuals: float
    per_query: list[EvalQueryResult] = Field(default_factory=list[EvalQueryResult])


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
    answered_results = [r for r in results if r.answered]
    report = EvalReport(
        run_id=str(ulid.ULID()),
        started_at=started,
        finished_at=finished,
        query_count=len(results),
        answered_count=len(answered_results),
        refused_count=sum(1 for r in results if not r.answered),
        mean_citation_precision=(
            sum(r.citation_precision for r in results) / len(results) if results else 0.0
        ),
        mean_citation_precision_answered_only=(
            sum(r.citation_precision for r in answered_results) / len(answered_results)
            if answered_results
            else float("nan")
        ),
        refusal_rate_on_counterfactuals=(
            refusal_correct_count / counterfactual_count if counterfactual_count else 1.0
        ),
        per_query=results,
    )
    log.info(
        "eval.done",
        query_count=report.query_count,
        mean_citation_precision=report.mean_citation_precision,
        mean_citation_precision_answered_only=report.mean_citation_precision_answered_only,
    )
    return report


# ---------------------------------------------------------------------------
# Parse-quality eval (CER / WER / structural-F1 vs hand-curated ground truth)
# ---------------------------------------------------------------------------
#
# This is the parse-fidelity half of the eval matrix (docs/eval-corpus-plan.md
# §Scoring) — distinct from the answering-quality half above. It consumes the
# documented per-document corpus layout:
#
#     <corpus_dir>/<category>/<doc_id>/
#         ground-truth.md     # the expected markdown (required)
#         manifest.json        # optional: doc_id + scoring thresholds
#         predicted.md         # optional: overrides the vault as the predicted
#
# The *predicted* markdown is what Memex produced. By default it's read from
# the vault by doc_id (so the operator ingests the corpus, then runs this);
# a `predicted.md` sibling overrides that for offline / test use.


class ParseEvalDoc(BaseModel):
    """Per-document parse-quality verdict: the three metrics plus the
    manifest thresholds (when present) and whether all were met."""

    doc_id: str
    cer: float
    wer: float
    structural_f1_headings: float
    structural_f1_tables: float
    structural_f1_equations: float
    cer_threshold: float | None = None
    wer_threshold: float | None = None
    structural_f1_threshold: float | None = None
    structural_f1_tables_threshold: float | None = None
    structural_f1_equations_threshold: float | None = None
    passed: bool


class ParseEvalReport(BaseModel):
    """Aggregate parse-quality report — per-doc scores + means + how many
    met every declared threshold. `errors` lists docs that couldn't be
    scored (no predicted markdown found)."""

    run_id: str
    started_at: datetime
    finished_at: datetime
    doc_count: int
    passed_count: int
    mean_cer: float
    mean_wer: float
    mean_structural_f1_headings: float
    mean_structural_f1_tables: float
    mean_structural_f1_equations: float
    per_doc: list[ParseEvalDoc] = Field(default_factory=list[ParseEvalDoc])
    errors: list[str] = Field(default_factory=list[str])


def _resolve_predicted(doc_dir: Path, doc_id: str, vault_path: Path) -> str | None:
    """The predicted markdown for `doc_id`: a `predicted.md` sibling if
    present (offline / test seam), else the doc's body from the vault.
    Returns None when neither resolves (caller records an error)."""
    override = doc_dir / "predicted.md"
    if override.is_file():
        return override.read_text(encoding="utf-8")
    md_path = vault_path / "documents" / f"{doc_id}.md"
    if md_path.is_file():
        return md_path.read_text(encoding="utf-8")
    return None


def _passes_thresholds(scores: ParseQualityScores, scoring: dict[str, float]) -> bool:
    """All declared thresholds met? CER/WER are upper bounds, structural
    F1 is a lower bound. Missing thresholds don't gate."""
    if "cer_threshold" in scoring and scores.cer > scoring["cer_threshold"]:
        return False
    if "wer_threshold" in scoring and scores.wer > scoring["wer_threshold"]:
        return False
    if (
        "structural_f1_threshold" in scoring
        and scores.structural_f1_headings < scoring["structural_f1_threshold"]
    ):
        return False
    if (
        "structural_f1_tables_threshold" in scoring
        and scores.structural_f1_tables < scoring["structural_f1_tables_threshold"]
    ):
        return False
    if (
        "structural_f1_equations_threshold" in scoring
        and scores.structural_f1_equations < scoring["structural_f1_equations_threshold"]
    ):
        return False
    return True


async def run_parse_eval(corpus_dir: Path, *, vault_path: Path | None = None) -> ParseEvalReport:
    """Score every ground-truthed document under `corpus_dir`.

    Walks for `ground-truth.md` files, reads each doc's predicted markdown
    (vault by doc_id, or a `predicted.md` override), and computes CER /
    WER / structural-F1 against the ground truth — checked against the
    `manifest.json` `scoring` thresholds when present. Docs without a
    resolvable predicted markdown are recorded in `errors`, not fatal.
    """
    import ulid

    if vault_path is None:
        vault_path = get_settings().vault_path

    started = datetime.now(UTC)
    log = logger.bind(corpus_dir=str(corpus_dir))
    log.info("parse_eval.start")

    per_doc: list[ParseEvalDoc] = []
    errors: list[str] = []

    for gt_path in sorted(corpus_dir.rglob("ground-truth.md")):
        doc_dir = gt_path.parent
        manifest_path = doc_dir / "manifest.json"
        # `Any` because the manifest is dynamic JSON (json.loads → Any).
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        doc_id = str(manifest.get("doc_id") or doc_dir.name)
        scoring_raw = manifest.get("scoring")
        scoring: dict[str, float] = {}
        if isinstance(scoring_raw, dict):
            # Per the eval-corpus-plan manifest schema, `scoring` maps
            # metric → numeric threshold; trust it via cast.
            scoring = {str(k): float(v) for k, v in cast("dict[str, float]", scoring_raw).items()}

        predicted = _resolve_predicted(doc_dir, doc_id, vault_path)
        if predicted is None:
            errors.append(f"{doc_id}: no predicted markdown (not in vault, no predicted.md)")
            log.warning("parse_eval.no_predicted", doc_id=doc_id)
            continue

        reference = gt_path.read_text(encoding="utf-8")
        scores = score_parse_quality(predicted, reference)
        passed = _passes_thresholds(scores, scoring)
        per_doc.append(
            ParseEvalDoc(
                doc_id=doc_id,
                cer=scores.cer,
                wer=scores.wer,
                structural_f1_headings=scores.structural_f1_headings,
                structural_f1_tables=scores.structural_f1_tables,
                structural_f1_equations=scores.structural_f1_equations,
                cer_threshold=scoring.get("cer_threshold"),
                wer_threshold=scoring.get("wer_threshold"),
                structural_f1_threshold=scoring.get("structural_f1_threshold"),
                structural_f1_tables_threshold=scoring.get("structural_f1_tables_threshold"),
                structural_f1_equations_threshold=scoring.get("structural_f1_equations_threshold"),
                passed=passed,
            )
        )
        log.info(
            "parse_eval.doc",
            doc_id=doc_id,
            cer=scores.cer,
            wer=scores.wer,
            structural_f1=scores.structural_f1_headings,
            passed=passed,
        )

    n = len(per_doc)
    report = ParseEvalReport(
        run_id=str(ulid.ULID()),
        started_at=started,
        finished_at=datetime.now(UTC),
        doc_count=n,
        passed_count=sum(1 for d in per_doc if d.passed),
        mean_cer=sum(d.cer for d in per_doc) / n if n else 0.0,
        mean_wer=sum(d.wer for d in per_doc) / n if n else 0.0,
        mean_structural_f1_headings=(
            sum(d.structural_f1_headings for d in per_doc) / n if n else 0.0
        ),
        mean_structural_f1_tables=(sum(d.structural_f1_tables for d in per_doc) / n if n else 0.0),
        mean_structural_f1_equations=(
            sum(d.structural_f1_equations for d in per_doc) / n if n else 0.0
        ),
        per_doc=per_doc,
        errors=errors,
    )
    log.info(
        "parse_eval.done",
        doc_count=report.doc_count,
        passed_count=report.passed_count,
        mean_cer=report.mean_cer,
        errors=len(errors),
    )
    return report
