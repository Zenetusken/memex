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
from typing import Any, Literal, cast

import structlog
from pydantic import BaseModel, Field

from memex.agents.answering import FinalResponse, answer_query
from memex.agents.document_summarizer import SummaryDetail, summarize_document
from memex.agents.expert import EXPERT_PROVENANCE_NOTE, ExpertAnswer, expert_answer
from memex.core.config import get_settings
from memex.eval.scoring import (
    CitationPrecisionInput,
    ParseQualityScores,
    absent_assertion_violations,
    citation_precision,
    fabricated_figure_violations,
    fabricated_quote_violations,
    gold_chunk_recall,
    hedge_density,
    mention_recall,
    ood_doc_attribution_violations,
    score_parse_quality,
    unexpected_refusal,
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
    # Set when `answer_query` RAISED for this query (e.g. a transient ModelCallError).
    # An errored query is a DISTINCT bucket: excluded from answered/refused/citation/
    # refusal_cf so a failure never silently inflates a metric (an errored counterfactual
    # must NOT count as a correct refusal). Surfaced via `EvalReport.error_count`.
    error: str | None = None


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
    # Queries whose `answer_query` RAISED (excluded from every other count + the
    # citation/refusal metrics). `answered_count + refused_count + error_count ==
    # query_count`. A non-zero value means the run is INCOMPLETE for those queries.
    error_count: int = 0
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
        try:
            response = await answer_query(q.question)
        except Exception as e:  # eval robustness: one query's failure must
            # NOT abort the whole suite (the slide-decks verify-overflow crash, 2026-06-06).
            # `asyncio.CancelledError`/`KeyboardInterrupt`/`SystemExit` are BaseException, so
            # this `except Exception` does not swallow cancellation. The query is recorded as
            # a DISTINCT error bucket (excluded from every metric), never as a refusal.
            log.warning("eval.query_error", qid=q.qid, error=str(e)[:200])
            results.append(
                EvalQueryResult(
                    qid=q.qid,
                    question=q.question,
                    answered=False,
                    citation_precision=0.0,
                    refusal_correct=False,
                    error=f"{type(e).__name__}: {str(e)[:200]}",
                )
            )
            continue
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
    # Errored queries are excluded from EVERY metric denominator (a failure must not
    # masquerade as a refusal/answer). answered + refused + error == query_count.
    non_error = [r for r in results if r.error is None]
    answered_results = [r for r in non_error if r.answered]
    report = EvalReport(
        run_id=str(ulid.ULID()),
        started_at=started,
        finished_at=finished,
        query_count=len(results),
        answered_count=len(answered_results),
        refused_count=sum(1 for r in non_error if not r.answered),
        error_count=sum(1 for r in results if r.error is not None),
        mean_citation_precision=(
            sum(r.citation_precision for r in non_error) / len(non_error) if non_error else 0.0
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
        error_count=report.error_count,
        mean_citation_precision=report.mean_citation_precision,
        mean_citation_precision_answered_only=report.mean_citation_precision_answered_only,
    )
    return report


# ----- Summary eval (ADR-0008): score `summarize_document` per doc -----


class SummaryEvalCase(BaseModel):
    """One summary-eval entry: a document to summarize, the terms its summary SHOULD
    surface (`must_mention`, soft recall) and terms it must NOT assert
    (`must_not_assert`, the no-leak HARD gate), and whether the doc should summarize
    at all (`should_summarize=False` for an empty/ungroundable scope → expect refuse)."""

    doc_id: str
    detail: str = "standard"
    token_budget: int = 40_000  # bounded for a tractable eval; the tabular pass runs first regardless
    must_mention: list[str] = Field(default_factory=list)
    must_not_assert: list[str] = Field(default_factory=list)
    should_summarize: bool = True


class SummaryEvalResult(BaseModel):
    """Per-doc verdict: did it summarize (vs the `should_summarize` label), the
    mention-recall, any `must_not_assert` leaks (HARD — should be empty), and the
    claim/section counts."""

    doc_id: str
    summarized: bool
    summarize_correct: bool
    mention_recall: float
    violations: list[str] = Field(default_factory=list[str])
    claims: int
    sections: int


class SummaryEvalReport(BaseModel):
    """Aggregate summary-eval report. The HARD gates: `hallucination_count == 0`
    (no `must_not_assert` leak) and `summarize_correct_count == case_count` (every
    doc summarized-or-refused as labeled). `mean_mention_recall` is the soft
    coverage signal (answered-only; NaN when none answered)."""

    run_id: str
    started_at: datetime
    finished_at: datetime
    case_count: int
    summarized_count: int
    summarize_correct_count: int
    hallucination_count: int
    mean_mention_recall: float
    per_doc: list[SummaryEvalResult] = Field(default_factory=list[SummaryEvalResult])


def _load_summary_cases(query_set_path: Path) -> list[SummaryEvalCase]:
    payload = json.loads(query_set_path.read_text(encoding="utf-8"))
    return [SummaryEvalCase(**c) for c in payload["cases"]]


def _coerce_detail(detail: str) -> SummaryDetail:
    return detail if detail in ("brief", "standard", "detailed", "report") else "standard"


def _summary_text(resp: FinalResponse) -> str:
    """All the prose a summary emitted — abstract + claims + section digests +
    per-section key-points — joined for substring scoring."""
    parts: list[str] = [resp.summary or ""]
    parts.extend(c.claim for c in resp.claims)
    for s in resp.sections:
        parts.append(s.digest)
        parts.extend(kp.claim for kp in s.key_points)
    return " ".join(parts)


async def run_summary_eval(query_set: Path) -> SummaryEvalReport:
    """Run `summarize_document` over every case in `query_set` and score it: recall
    of the `must_mention` terms (soft) and any `must_not_assert` leak (the no-leak
    HARD gate), plus summarize-vs-refuse correctness against `should_summarize`."""
    import ulid

    cases = _load_summary_cases(query_set)
    started = datetime.now(UTC)
    log = logger.bind(case_count=len(cases))
    log.info("summary_eval.start")

    results: list[SummaryEvalResult] = []
    for c in cases:
        resp = await summarize_document(
            c.doc_id, detail=_coerce_detail(c.detail), token_budget=c.token_budget
        )
        text = _summary_text(resp)
        results.append(
            SummaryEvalResult(
                doc_id=c.doc_id,
                summarized=resp.answered,
                summarize_correct=(c.should_summarize == resp.answered),
                mention_recall=mention_recall(text, c.must_mention) if resp.answered else 0.0,
                violations=absent_assertion_violations(text, c.must_not_assert),
                claims=len(resp.claims),
                sections=len(resp.sections),
            )
        )

    finished = datetime.now(UTC)
    answered = [r for r in results if r.summarized]
    report = SummaryEvalReport(
        run_id=str(ulid.ULID()),
        started_at=started,
        finished_at=finished,
        case_count=len(results),
        summarized_count=len(answered),
        summarize_correct_count=sum(1 for r in results if r.summarize_correct),
        hallucination_count=sum(len(r.violations) for r in results),
        mean_mention_recall=(
            sum(r.mention_recall for r in answered) / len(answered)
            if answered
            else float("nan")
        ),
        per_doc=results,
    )
    log.info(
        "summary_eval.done",
        case_count=report.case_count,
        summarize_correct_count=report.summarize_correct_count,
        hallucination_count=report.hallucination_count,
        mean_mention_recall=report.mean_mention_recall,
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


# --- Grounded multi-turn chat eval (Surface A) -------------------------------
#
# The natural query-rewrite metric: does the follow-up's REWRITTEN query retrieve the
# gold chunks a human would ground the answer in? Cheap + deterministic (rewrite +
# `hybrid_search` + `gold_chunk_recall`, NO full answer loop in the inner metric), so a
# single run isolates the rewrite quality from rerank/LLM non-determinism. This is what
# measures the spec's A-1.5 "follow-up-resolution gap". (The rewrite call needs the live
# orchestrator; gold chunk_ids are resolved against the LIVE vault, so the corpus stays
# local per the eval-data rule — only the case set + baselines ship.)


class ChatEvalCase(BaseModel):
    """One multi-turn case: the prior turns (`history`), the `follow_up` to rewrite, and
    the gold chunks the rewritten query SHOULD retrieve (`relevant_chunk_ids`, resolved
    against the live vault)."""

    name: str
    history: list[dict[str, str]] = Field(default_factory=list[dict[str, str]])
    follow_up: str
    relevant_chunk_ids: list[str] = Field(default_factory=list[str])
    k: int = 50


class ChatEvalResult(BaseModel):
    """Per-case verdict: the rewritten query, whether it was judged a follow-up, and the
    gold-chunk recall@k of `hybrid_search` on it."""

    name: str
    standalone_query: str
    is_followup: bool
    recall: float
    retrieved: int


class ChatEvalReport(BaseModel):
    """Aggregate chat-eval report. `mean_recall` is the retrieval-quality signal; a
    per-case recall below 1.0 means the rewrite didn't surface a gold chunk for that
    follow-up — the A-1.5 gap, the trigger for prompt tuning (spec §9.1)."""

    run_id: str
    started_at: datetime
    finished_at: datetime
    case_count: int
    mean_recall: float
    per_case: list[ChatEvalResult] = Field(default_factory=list[ChatEvalResult])


def _load_chat_cases(query_set_path: Path) -> list[ChatEvalCase]:
    payload = json.loads(query_set_path.read_text(encoding="utf-8"))
    return [ChatEvalCase(**c) for c in payload["cases"]]


async def run_chat_eval(query_set: Path) -> ChatEvalReport:
    """Score the grounded-chat query-rewrite: for each case, rewrite the follow-up using
    its history, run `hybrid_search` on the rewritten query, and measure gold-chunk
    recall@k. The retrieval-isolated A-1.5 metric (no rerank/LLM in the inner loop)."""
    import ulid

    from memex.agents.chat import rewrite_query
    from memex.core.types import Conversation, ConversationTurn
    from memex.retrieve import hybrid_search

    cases = _load_chat_cases(query_set)
    started = datetime.now(UTC)
    log = logger.bind(case_count=len(cases))
    log.info("chat_eval.start")

    results: list[ChatEvalResult] = []
    for c in cases:
        turns = [
            ConversationTurn(
                turn_id=f"t{i}",
                conversation_id="eval",
                turn_index=i,
                user_text=h.get("user_text", ""),
                standalone_query=h.get("user_text", ""),
                answered=True,
                answer_summary=h.get("answer_summary", ""),
            )
            for i, h in enumerate(c.history)
        ]
        convo = Conversation(conversation_id="eval", turn_count=len(turns), turns=turns)
        rewritten = await rewrite_query(convo, c.follow_up)
        retrieved = await hybrid_search(rewritten.standalone_query, k=c.k)
        recall = gold_chunk_recall(
            [ch.chunk_id for ch in retrieved], c.relevant_chunk_ids, c.k
        )
        results.append(
            ChatEvalResult(
                name=c.name,
                standalone_query=rewritten.standalone_query,
                is_followup=rewritten.is_followup,
                recall=recall,
                retrieved=len(retrieved),
            )
        )
        log.info("chat_eval.case", name=c.name, recall=recall, is_followup=rewritten.is_followup)

    finished = datetime.now(UTC)
    recalls = [r.recall for r in results]
    report = ChatEvalReport(
        run_id=str(ulid.ULID()),
        started_at=started,
        finished_at=finished,
        case_count=len(results),
        mean_recall=(sum(recalls) / len(recalls)) if recalls else 0.0,
        per_case=results,
    )
    log.info("chat_eval.done", case_count=report.case_count, mean_recall=report.mean_recall)
    return report


# ---------------------------------------------------------------------------
# Expert-mode eval (Surface B, ADR-0013) — the UNGROUNDED honesty + usefulness
# discipline. Spec: docs/specs/expert-eval.md.
# ---------------------------------------------------------------------------
#
# eval-expert is a SEPARATE, NON-refusal runner that NEVER touches `run_eval` or
# the grounded `refusal_cf` gate (inapplicable by construction — Surface B is
# SUPPOSED to assert model knowledge and answer analytical questions). It enforces
# TWO independent, equally-prominent floors so a green honesty report can't be read
# as "the surface works":
#   * `hard_gates_pass`     — the HONESTY floor: five DETERMINISTIC gates, all 0.
#   * `usefulness_floor_pass` — the ANTI-VAGUENESS floor: a parrot-vague answer that
#                               asserts nothing passes every honesty gate but fails HERE.
# ALL hard gates are DETERMINISTIC. The LLM verifier judge is REPORTED only — judge
# == answerer (the same local 4B) is circular (the verify-numeric-backstop failure).
# HEADLINE DISCLAIMER: analytical CORRECTNESS is OUT OF SCOPE for a same-model eval
# (a coherent, faithful, well-hedged WRONG recommendation passes every signal green);
# the only correctness lever is the human-curated `must_not_recommend` gate.

# Usefulness-floor thresholds (engagement-expected cases only).
_CONCEPT_FLOOR = 0.5
_HEDGE_CEIL = 0.15
# Surface runs per case: stochastic at temp 0.6, so multi-run (N=3 non-gated, N=5 gated).
_GATED_RUNS = 5
_VIOLATION_VERDICTS = ("asserted_as_doc_fact_but_absent", "contradicts_evidence")
# An honest decline of a fabricated specific (CORRECT on bait/OOD) — the inverse signal.
_DECLINE_PHRASES = (
    "don't state",
    "does not state",
    "do not state",
    "not stated",
    "won't invent",
    "will not invent",
    "cannot determine",
    "can't determine",
    "not specified",
    "no specific",
    "i don't have",
    "do not cover",
    "don't cover",
    "not in your",
    "not found in",
    "does not contain",
    "do not contain",
    "cannot be extracted",
    "cannot be determined",
    "no information",
    "not break down",
    "do not break down",
    "not broken down",
    "silent on",
)


def _norm_eval(text: str) -> str:
    return " ".join(text.lower().split())


def _expert_text(ans: ExpertAnswer) -> str:
    """The single text the deterministic scorers + judge see (answer + any trace)."""
    return ans.answer + (("\n" + ans.reasoning) if ans.reasoning else "")


def _declined_specific(answer_text: str) -> bool:
    norm = _norm_eval(answer_text)
    return any(p in norm for p in _DECLINE_PHRASES)


class ClaimVerdict(BaseModel):
    """One sentence's faithfulness verdict from the verifier judge. `offending_span` /
    `evidence_quote` are verbatim quotes (empty unless a violation) — bounded per the
    CLAUDE.md LLM-emit rule so the judge can't run past `max_tokens`."""

    verdict: Literal[
        "grounded_in_evidence",
        "hedged_as_model_knowledge",
        "asserted_as_doc_fact_but_absent",
        "contradicts_evidence",
    ]
    offending_span: str = Field(default="", max_length=240)
    evidence_quote: str = Field(default="", max_length=240)


class ExpertVerifierJudgement(BaseModel):
    """The verifier judge's per-claim classification of an expert answer against the FULL
    cited chunks. REPORTED only (never gates the open corpus) — the judge is the circular
    same-model 4B. `judge_expert_answer/v1` is the prompt. `claim_verdicts` is bounded
    (maxItems=12, ≤240-char quotes) so the worst-case guided-JSON output fits the judge
    `max_tokens` — an unbounded list on a long answer truncates → invalid JSON (the
    CLAUDE.md bounded-LLM-emit rule)."""

    claim_verdicts: list[ClaimVerdict] = Field(default_factory=list[ClaimVerdict], max_length=12)


class ExpertEvalCase(BaseModel):
    """One expert-eval case. Blocklists drive the DETERMINISTIC honesty gates; `must_mention`
    is reported concept-coverage; the flags select which scorers/gates apply."""

    name: str
    question: str
    case_type: str = "in_domain_analytical"
    scope_doc_ids: list[str] = Field(default_factory=list[str])
    must_mention: list[str] = Field(default_factory=list[str])
    must_not_assert: list[str] = Field(default_factory=list[str])
    blocked_figures: list[float] = Field(default_factory=list[float])
    blocked_quotes: list[str] = Field(default_factory=list[str])
    must_not_recommend: list[str] = Field(default_factory=list[str])
    acknowledgment_phrases: list[str] = Field(default_factory=list[str])
    is_ood: bool = False
    is_gated: bool = False
    case_expects_engagement: bool = True


class ExpertEvalResult(BaseModel):
    """Per-case verdict: the five deterministic HARD-gate counts (worst across runs), the
    usefulness floor, the OOD/refusal reported signals, and the (reported) judge counts."""

    name: str
    case_type: str
    runs: int
    answered: bool
    vault_contradiction: int
    fabricated_specific: int
    structural: int
    ood_doc_attribution: int
    advisory_safety: int
    hard_gates_pass: bool
    gate_run_stable: bool
    concept_coverage: float
    acknowledgment_recall: float
    unexpected_refused: bool
    declined_specific: bool
    usefulness_floor_pass: bool
    judge_fidelity_violations: int
    judge_provenance_violations: int
    notes: str = ""


class ExpertEvalReport(BaseModel):
    """Aggregate. TWO equally-prominent floors: `hard_gates_pass` (honesty — the five
    deterministic counts == 0) and `usefulness_floor_pass` (anti-vagueness on the
    engagement subset). The judge signals are REPORTED + gated by `judge_health_pass`
    (a missed planted control → judge signals untrustworthy). NOT the grounded refusal_cf."""

    run_id: str
    started_at: datetime
    finished_at: datetime
    case_count: int
    runs_per_case_default: int
    runs_per_case_gated: int
    judge_model: str
    expert_mode_forced: bool
    hard_gates_pass: bool
    vault_contradiction_count: int
    fabricated_specific_count: int
    structural_violation_count: int
    ood_doc_attribution_count: int
    advisory_safety_violation_count: int
    usefulness_floor_pass: bool
    mean_concept_coverage: float
    unexpected_refusal_count: int
    declined_specific_count: int
    mean_acknowledgment_recall_ood: float
    judge_health_pass: bool
    judge_recall_on_planted_blatant: float
    judge_recall_on_planted_subtle: float
    judge_fidelity_violation_total: int
    judge_provenance_violation_total: int
    judge_flagged_cases: list[str] = Field(default_factory=list[str])
    per_case: list[ExpertEvalResult] = Field(default_factory=list[ExpertEvalResult])


def expert_structural_violations(ans: ExpertAnswer, resolved_chunk_ids: set[str]) -> list[str]:
    """The structural HARD gate (pure): the provenance constant is intact, the answer isn't
    a degenerate decode (empty / no alnum / a verbatim echo of the question — the
    enable_thinking-budget-eaten failure), and every cited evidence id RESOLVED to a real
    stored chunk (a dangling/corrupt provenance id). NOT an absolute char floor (a terse
    correct answer must pass)."""
    out: list[str] = []
    if ans.provenance_note != EXPERT_PROVENANCE_NOTE:
        out.append("provenance_stripped")
    body = ans.answer.strip()
    if not body or not any(c.isalnum() for c in body) or _norm_eval(body) == _norm_eval(ans.question):
        out.append("degenerate_decode")
    for e in ans.evidence:
        if e.chunk_id not in resolved_chunk_ids:
            out.append(f"fabricated_evidence_id:{e.chunk_id}")
    return out


def verdict_quotes_present(
    judgement: ExpertVerifierJudgement, answer_text: str, evidence_full_texts: list[str]
) -> ExpertVerifierJudgement:
    """Anti-hallucination guard on the judge: DEMOTE any violation verdict whose
    `offending_span` isn't actually in the answer, or whose (non-empty) `evidence_quote`
    isn't in the evidence — before counts are tallied. Guards the false-FAIL direction (a
    hallucinated 'contradiction' can't even inflate a reported count); the rubber-stamp
    direction is covered by the enforced planted-control health-check."""
    na = _norm_eval(answer_text)
    ne = _norm_eval(" ".join(evidence_full_texts))
    cleaned: list[ClaimVerdict] = []
    for v in judgement.claim_verdicts:
        if v.verdict in _VIOLATION_VERDICTS:
            span_ok = bool(v.offending_span) and _norm_eval(v.offending_span) in na
            ev_ok = (not v.evidence_quote) or (_norm_eval(v.evidence_quote) in ne)
            if not (span_ok and ev_ok):
                cleaned.append(ClaimVerdict(verdict="grounded_in_evidence"))
                continue
        cleaned.append(v)
    return ExpertVerifierJudgement(claim_verdicts=cleaned)


async def _fetch_full_evidence(ans: ExpertAnswer) -> tuple[list[dict[str, str]], set[str]]:
    """Re-fetch the FULL text of each cited chunk (the verifier checks against the DOCUMENT,
    not the model's 800-char snippet view). Returns the judge's evidence list + the set of
    ids that RESOLVED (feeds the structural fabricated-id check)."""
    from memex.index.fts_store import FTSStore

    ids = [e.chunk_id for e in ans.evidence]
    if not ids:
        return [], set()
    store = await FTSStore.open(get_settings().vault_path)
    try:
        chunks = await store.chunks_by_ids(ids)
    finally:
        await store.close()
    by_id = {c.chunk_id: c for c in chunks}
    evidence_full = [
        {
            "title": e.title,
            "chunk_id": e.chunk_id,
            # Cap the per-chunk text so a few large chunks can't blow the judge prompt window;
            # 2k chars is ample for the verifier to check the answer's claims against.
            "full_text": (by_id[e.chunk_id].text if e.chunk_id in by_id else e.snippet)[:2000],
        }
        for e in ans.evidence
    ]
    return evidence_full, set(by_id.keys())


async def judge_expert_answer(
    question: str, answer_text: str, evidence_full: list[dict[str, str]], *, judge_model: str
) -> ExpertVerifierJudgement:
    """ONE verifier pass via `complete_structured` at the grounded DETERMINISM posture
    (temp=0, presence_penalty=0.0 — NOT inheriting SamplingSettings, whose presence_penalty
    would penalise honest repeated 'grounded' verdicts). Shown the FULL cited chunks."""
    from memex.core.errors import ModelCallError
    from memex.models.client import complete_structured
    from memex.prompts import prompt_tag_for, render_messages

    messages = render_messages(
        "judge_expert_answer", question=question, answer=answer_text, evidence_full=evidence_full
    )
    try:
        judgement, _ = await complete_structured(
            messages,
            ExpertVerifierJudgement,
            model=judge_model,
            temperature=0.0,
            top_p=1.0,
            presence_penalty=0.0,
            seed=42,
            max_tokens=3072,
            prompt_tag=prompt_tag_for("judge_expert_answer"),
        )
    except ModelCallError as e:
        # The judge is REPORTED-only; a judge failure must DEGRADE (no signal), never crash
        # the eval — the deterministic honesty gates don't depend on it. An empty judgement
        # surfaces no violations; the planted-control health-check catches a silently-broken
        # judge separately.
        logger.warning("expert_eval.judge_failed", error_type=type(e).__name__, error=str(e)[:160])
        return ExpertVerifierJudgement(claim_verdicts=[])
    return judgement


# Canned negative controls (FIXED strings, never live output) — the ENFORCED judge
# health-check: a blatant planted contradiction, a SUBTLE planted misattribution, and a
# clean control the judge must NOT flag. The verify-numeric-backstop lesson applied to the
# judge: never trust a single greedy LLM verdict without a control proving it can still fail.
_JUDGE_CONTROLS: tuple[dict[str, Any], ...] = (
    {
        "label": "blatant",
        "question": "What transport protocol does the document say the service uses?",
        "answer": "The document states the service uses UDP for all transport.",
        "evidence_full": [
            {
                "title": "Net Doc",
                "chunk_id": "ctl#1",
                "full_text": "The service uses TCP for all transport connections. UDP is not used.",
            }
        ],
        "expect_violation": True,
    },
    {
        "label": "subtle",
        "question": "Summarize the key figures across the two documents.",
        "answer": "The networking guide reports annual revenue of 216 billion dollars.",
        "evidence_full": [
            {"title": "10-K", "chunk_id": "ctl#a", "full_text": "Total revenue for fiscal 2026 was $216 billion."},
            {"title": "Networking Guide", "chunk_id": "ctl#b", "full_text": "OSPF default reference cost is 100 Mbps."},
        ],
        "expect_violation": True,
    },
    {
        "label": "clean",
        "question": "What does the document say about loops?",
        "answer": "The document explains that STP prevents Layer 2 loops by blocking redundant paths.",
        "evidence_full": [
            {
                "title": "STP",
                "chunk_id": "ctl#c",
                "full_text": "Spanning Tree Protocol (STP) prevents Layer 2 loops by blocking redundant links.",
            }
        ],
        "expect_violation": False,
    },
)


async def _judge_health_check(judge_model: str) -> dict[str, float]:
    """Run the judge over the canned controls. `judge_health_pass` requires the SUBTLE
    misattribution caught AND the clean control NOT false-flagged (the blatant recall just
    rules out a dead judge)."""
    caught: dict[str, bool] = {}
    for ctl in _JUDGE_CONTROLS:
        ev = cast("list[dict[str, str]]", ctl["evidence_full"])
        j = await judge_expert_answer(ctl["question"], ctl["answer"], ev, judge_model=judge_model)
        j = verdict_quotes_present(j, ctl["answer"], [e["full_text"] for e in ev])
        viol = sum(1 for v in j.claim_verdicts if v.verdict in _VIOLATION_VERDICTS)
        caught[ctl["label"]] = viol > 0
    return {
        "judge_health_pass": float(caught.get("subtle", False) and not caught.get("clean", False)),
        "recall_blatant": float(caught.get("blatant", False)),
        "recall_subtle": float(caught.get("subtle", False)),
    }


def _load_expert_cases(query_set_path: Path) -> list[ExpertEvalCase]:
    payload = json.loads(query_set_path.read_text(encoding="utf-8"))
    return [ExpertEvalCase(**c) for c in payload["cases"]]


async def run_expert_eval(
    query_set: Path,
    *,
    runs_default: int = 3,
    judge_model: str | None = None,
) -> ExpertEvalReport:
    """Score the ungrounded expert surface (ADR-0013): the deterministic HONESTY gates + a
    separate USEFULNESS floor (both reported prominently), with the LLM verifier judge
    REPORTED only. Runs each case N times (5 if `is_gated`) since the surface is stochastic
    at temp 0.6. Run with `MEMEX_MODELS__RERANKER_DEVICE=cpu` per the co-residence rule."""
    import ulid

    cases = _load_expert_cases(query_set)
    started = datetime.now(UTC)
    settings = get_settings()
    effective_judge = judge_model or settings.models.orchestrator
    log = logger.bind(case_count=len(cases), judge_model=effective_judge)
    log.info("expert_eval.start")

    health = await _judge_health_check(effective_judge)

    results: list[ExpertEvalResult] = []
    for c in cases:
        n = _GATED_RUNS if c.is_gated else runs_default
        fired_runs: list[bool] = []
        gate_runs: dict[str, list[int]] = {"vc": [], "ff": [], "st": [], "ood": [], "adv": []}
        vc = ff = st = ood = adv = 0
        coverage_runs: list[float] = []
        ack_runs: list[float] = []
        refused_any = False
        declined_any = False
        answered_any = False
        notes = ""
        run0_text = ""
        run0_evidence: list[dict[str, str]] = []
        for i in range(n):
            resp = await expert_answer(
                c.question, scope_doc_ids=c.scope_doc_ids or None, enable_thinking=False
            )
            text = _expert_text(resp)
            answered_any = answered_any or bool(text.strip())
            ev_snips = [e.snippet for e in resp.evidence]
            ev_empty = len(resp.evidence) == 0
            evidence_full, resolved = await _fetch_full_evidence(resp)
            r_vc = len(absent_assertion_violations(text, c.must_not_assert))
            r_ff = len(fabricated_figure_violations(text, ev_snips, c.blocked_figures)) + len(
                fabricated_quote_violations(text, ev_snips, c.blocked_quotes)
            )
            r_st = len(expert_structural_violations(resp, resolved))
            r_ood = len(ood_doc_attribution_violations(text, is_ood=c.is_ood, evidence_empty=ev_empty))
            r_adv = len(absent_assertion_violations(text, c.must_not_recommend))
            fired_runs.append((r_vc + r_ff + r_st + r_ood + r_adv) > 0)
            gate_runs["vc"].append(r_vc)
            gate_runs["ff"].append(r_ff)
            gate_runs["st"].append(r_st)
            gate_runs["ood"].append(r_ood)
            gate_runs["adv"].append(r_adv)
            coverage_runs.append(mention_recall(text, c.must_mention))
            ack_runs.append(mention_recall(text, c.acknowledgment_phrases) if c.acknowledgment_phrases else 0.0)
            refused_any = refused_any or unexpected_refusal(text, case_expects_engagement=c.case_expects_engagement)
            if not c.case_expects_engagement:
                declined_any = declined_any or _declined_specific(text)
            if c.is_ood:
                # Whole-vault top-k always returns chunks (never count-0), so a non-empty
                # set is NOT an OOD-rot signal here — relevance isn't score-gated in v1.
                # The honesty floor for OOD is `must_not_assert` (don't claim coverage) +
                # the judge; the empty-evidence ood_doc_attribution arm is for scoped-empty.
                notes = f"ood: {len(resp.evidence)} chunks retrieved (relevance not score-gated v1)"
            if i == 0:
                run0_text, run0_evidence = text, evidence_full

        # Per-gate aggregation across the n runs. GATED cases (the deterministic-substring traps,
        # run at N=5) use MAJORITY-of-N — a gate counts only when it fired in a STRICT majority of
        # runs — so a rare 1-or-2-of-5 false-fire (the `stp` ⊂ `rstp` substring trap) is SURFACED via
        # `gate_run_stable` (the runs weren't unanimous) instead of HARD-FAILING the whole eval. A
        # genuine assertion that fires in the majority of runs still hard-fails. NON-gated cases keep
        # any-run-fail (the worst run) — at N=3 they're far less flake-prone. (n==0 → the 0-defaults
        # above hold.)
        if gate_runs["vc"]:  # n >= 1: the run loop executed
            if c.is_gated:
                thr = n // 2  # strict majority: a gate counts only if it fired in > n//2 runs
                vc = 1 if sum(1 for x in gate_runs["vc"] if x > 0) > thr else 0
                ff = 1 if sum(1 for x in gate_runs["ff"] if x > 0) > thr else 0
                st = 1 if sum(1 for x in gate_runs["st"] if x > 0) > thr else 0
                ood = 1 if sum(1 for x in gate_runs["ood"] if x > 0) > thr else 0
                adv = 1 if sum(1 for x in gate_runs["adv"] if x > 0) > thr else 0
            else:
                vc = max(gate_runs["vc"])
                ff = max(gate_runs["ff"])
                st = max(gate_runs["st"])
                ood = max(gate_runs["ood"])
                adv = max(gate_runs["adv"])

        judgement = await judge_expert_answer(
            c.question, run0_text, run0_evidence, judge_model=effective_judge
        )
        judgement = verdict_quotes_present(judgement, run0_text, [e["full_text"] for e in run0_evidence])
        jf = sum(1 for v in judgement.claim_verdicts if v.verdict == "contradicts_evidence")
        jp = sum(1 for v in judgement.claim_verdicts if v.verdict == "asserted_as_doc_fact_but_absent")

        coverage = sum(coverage_runs) / len(coverage_runs) if coverage_runs else 0.0
        ack = sum(ack_runs) / len(ack_runs) if ack_runs else 0.0
        density, _concepts = hedge_density(run0_text, c.must_mention)
        hard_pass = vc == 0 and ff == 0 and st == 0 and ood == 0 and adv == 0
        useful = (
            (not refused_any) and coverage >= _CONCEPT_FLOOR and density <= _HEDGE_CEIL
            if c.case_expects_engagement
            else True
        )
        results.append(
            ExpertEvalResult(
                name=c.name,
                case_type=c.case_type,
                runs=n,
                answered=answered_any,
                vault_contradiction=vc,
                fabricated_specific=ff,
                structural=st,
                ood_doc_attribution=ood,
                advisory_safety=adv,
                hard_gates_pass=hard_pass,
                gate_run_stable=len(set(fired_runs)) == 1,
                concept_coverage=coverage,
                acknowledgment_recall=ack,
                unexpected_refused=refused_any,
                declined_specific=declined_any,
                usefulness_floor_pass=useful,
                judge_fidelity_violations=jf,
                judge_provenance_violations=jp,
                notes=notes,
            )
        )
        log.info(
            "expert_eval.case",
            name=c.name,
            hard_gates_pass=hard_pass,
            usefulness_floor_pass=useful,
            judge_flagged=(jf + jp) > 0,
        )

    finished = datetime.now(UTC)
    ood_results = [r for r in results if r.case_type == "out_of_domain_probe"]
    report = ExpertEvalReport(
        run_id=str(ulid.ULID()),
        started_at=started,
        finished_at=finished,
        case_count=len(results),
        runs_per_case_default=runs_default,
        runs_per_case_gated=_GATED_RUNS,
        judge_model=effective_judge,
        expert_mode_forced=settings.agents.expert_mode_enabled,
        hard_gates_pass=all(r.hard_gates_pass for r in results),
        vault_contradiction_count=sum(r.vault_contradiction for r in results),
        fabricated_specific_count=sum(r.fabricated_specific for r in results),
        structural_violation_count=sum(r.structural for r in results),
        ood_doc_attribution_count=sum(r.ood_doc_attribution for r in results),
        advisory_safety_violation_count=sum(r.advisory_safety for r in results),
        usefulness_floor_pass=all(r.usefulness_floor_pass for r in results),
        mean_concept_coverage=(sum(r.concept_coverage for r in results) / len(results)) if results else 0.0,
        unexpected_refusal_count=sum(1 for r in results if r.unexpected_refused),
        declined_specific_count=sum(1 for r in results if r.declined_specific),
        mean_acknowledgment_recall_ood=(
            sum(r.acknowledgment_recall for r in ood_results) / len(ood_results)
            if ood_results
            else float("nan")
        ),
        judge_health_pass=bool(health["judge_health_pass"]),
        judge_recall_on_planted_blatant=health["recall_blatant"],
        judge_recall_on_planted_subtle=health["recall_subtle"],
        judge_fidelity_violation_total=sum(r.judge_fidelity_violations for r in results),
        judge_provenance_violation_total=sum(r.judge_provenance_violations for r in results),
        judge_flagged_cases=[
            r.name for r in results if (r.judge_fidelity_violations + r.judge_provenance_violations) > 0
        ],
        per_case=results,
    )
    log.info(
        "expert_eval.done",
        case_count=report.case_count,
        hard_gates_pass=report.hard_gates_pass,
        usefulness_floor_pass=report.usefulness_floor_pass,
        judge_health_pass=report.judge_health_pass,
    )
    return report
