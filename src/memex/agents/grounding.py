"""Shared answer-quality gates reused OUTSIDE the `/ask` graph — the per-claim grounding check
(`ground_claims`) and the whole-answer responsiveness check (`assess_responsiveness`).

Both the document summarizer (the GROUND step of MAP→GROUND→REDUCE) and the reason-then-ground
bridge (Surface §11) need the SAME deterministic-as-possible per-claim grounding: wrap a set of
`CitedClaim`s in a `DraftAnswer`, run the UNCHANGED `verify_grounding/v2` prompt against the chunks
each claim cites, and keep only the claims the verifier confirms are supported. This is the exact
logic that used to live inline in `document_summarizer._ground_points`; it is hoisted here verbatim
so the bridge reuses the identical gate (same prompt, same schema, same conservative keep rule)
WITHOUT importing a private symbol across modules and WITHOUT touching the `/ask` `answering.verify`
node (which keeps its own inline bounded schema — the answer graph stays byte-identical).

`assess_responsiveness` is the responsiveness twin: the consented A→B escalation, when it presents
the grounded subset AS an answer (ADR-0016), needs the SAME responsiveness verdict the `/ask`
`assess_relevance` node applies — "do these grounded claims answer the SPECIFIC question, or a
related-but-different one?". We render the UNCHANGED `assess_relevance@v1` prompt + `RelevanceAssessment`
schema here rather than refactor the `/ask` node to call this helper: making `answering` import
`grounding` would create an `answering↔grounding` cycle (this module already imports FROM answering)
AND would edit a HARD-gate node for zero behavioural gain. The single source of truth is the prompt
file + the schema; a small call wrapper duplicated is the correct cost (cf. `bounded_verification`).

Neither helper is the `/ask` `verify` node: they do NOT carry the numeric-aggregate backstop (that
demotion lives in the node). Callers that can surface computed-table-figure claims must keep those
out of scope — both consumers do (the summarizer accepts this today; the bridge's extractor is told
to avoid bare computed figures).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import structlog
from pydantic import Field, create_model

from memex.agents.answering import DraftAnswer, RelevanceAssessment, VerificationResult
from memex.core.errors import ModelCallError
from memex.models.client import complete_structured
from memex.prompts import render_prompt

if TYPE_CHECKING:
    from memex.agents.answering import CitedClaim
    from memex.core.types import Chunk

logger = structlog.get_logger(__name__)


def bounded_verification(n: int) -> type[VerificationResult]:
    """Per-call `VerificationResult` with the index lists bounded to `n`
    (mirrors `answering.verify`). xgrammar enforces the list bounds."""
    return create_model(
        "VerificationResult",
        __base__=VerificationResult,
        grounded=(Annotated[list[int], Field(max_length=n)], Field(default_factory=list)),
        ungrounded=(Annotated[list[int], Field(max_length=n)], Field(default_factory=list)),
        ungrounded_reasons=(
            Annotated[list[Annotated[str, Field(max_length=250)]], Field(max_length=n)],
            Field(default_factory=list),
        ),
    )


async def ground_claims(
    summary: str,
    claims: list[CitedClaim],
    chunks: list[Chunk],
    *,
    max_tokens: int,
) -> tuple[list[CitedClaim], int]:
    """Keep only the claims the verifier confirms are supported by their cited chunk —
    reusing `verify_grounding/v2` exactly (the claims ARE `CitedClaim`s, so we wrap them
    in a `DraftAnswer` and run the same prompt). Conservative: a claim survives only if
    explicitly grounded (missing/contested → dropped). Fail-open to `([], 0)` on a grounding
    `ModelCallError` — NEVER ship ungrounded on a grounding failure.

    `summary` is verifier CONTEXT only (it grounds the CLAIMS, not the summary); callers pass
    a short, already-bounded synthesis (e.g. the summarizer passes `digest[:300]`).
    """
    if not claims:
        return [], 0
    draft = DraftAnswer(summary=summary, claims=claims)
    chunk_by_id = {c.chunk_id: c for c in chunks}
    prompt = render_prompt("verify_grounding", draft=draft, chunk_by_id=chunk_by_id)
    n = len(claims)
    try:
        bounded, tokens = await complete_structured(
            prompt=prompt,
            schema=bounded_verification(n),
            max_tokens=max_tokens,
            prompt_tag="verify_grounding@v2",
        )
    except ModelCallError as e:
        # If grounding itself fails, drop the claims (never ship ungrounded).
        logger.warning("grounding.failed", error=str(e)[:160])
        return [], 0
    grounded_idx = [i for i in bounded.grounded if 0 <= i < n]
    ungrounded_idx = {i for i in bounded.ungrounded if 0 <= i < n}
    kept = [claims[i] for i in grounded_idx if i not in ungrounded_idx]
    return kept, tokens


async def assess_responsiveness(
    query: str,
    summary: str,
    claim_texts: list[str],
) -> tuple[RelevanceAssessment, int]:
    """Whole-answer responsiveness gate, reusable OUTSIDE the `/ask` graph — does the
    already-grounded `summary` + `claim_texts` answer the SPECIFIC question, or a
    related-but-different one? Runs the UNCHANGED `assess_relevance@v1` prompt + schema (the
    same one the `/ask` `assess_relevance` node uses; the node is left untouched, see the
    module docstring on the import-cycle reason).

    FAIL-CLOSED on a `ModelCallError` → `(RelevanceAssessment(responsive=False, …), 0)`: a gate
    error must never PROMOTE the bridge's grounded subset to a presented answer (the bridge-shaped
    equivalent of `/ask` failing loud — it falls back to the labelled-analysis surface instead).
    """
    prompt = render_prompt(
        "assess_relevance",
        query=query,
        summary=summary,
        claims=claim_texts,
    )
    try:
        relevance, tokens = await complete_structured(
            prompt=prompt,
            schema=RelevanceAssessment,
            prompt_tag="assess_relevance@v1",
        )
    except ModelCallError as e:
        logger.warning("relevance.failed", error=str(e)[:160])
        return RelevanceAssessment(responsive=False, reason="responsiveness check failed"), 0
    return relevance, tokens
