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

import asyncio
from typing import TYPE_CHECKING, Annotated

import structlog
from pydantic import Field, create_model

from memex.agents.answering import DraftAnswer, RelevanceAssessment, VerificationResult
from memex.core.errors import ModelCallError
from memex.models.client import complete_structured
from memex.prompts import prompt_tag_for, render_prompt

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
            prompt_tag=prompt_tag_for("verify_grounding"),
        )
    except ModelCallError as e:
        # If grounding itself fails, drop the claims (never ship ungrounded).
        logger.warning("grounding.failed", error=str(e)[:160])
        return [], 0
    grounded_idx = [i for i in bounded.grounded if 0 <= i < n]
    ungrounded_idx = {i for i in bounded.ungrounded if 0 <= i < n}
    kept = [claims[i] for i in grounded_idx if i not in ungrounded_idx]
    return kept, tokens


# Cap on concurrent isolated verify calls — matches `enrich/pipeline.py`'s `_MAX_CONCURRENT`
# precedent (concurrent `complete_structured` against the single shared orchestrator). The bridge's
# `groundable` is <= 8 (DraftAnswer.claims max_length), so this rarely binds; it is a polite ceiling.
_ISOLATED_GROUND_CONCURRENCY = 4


async def ground_claims_isolated(
    summary: str,
    claims: list[CitedClaim],
    chunks: list[Chunk],
    *,
    max_tokens: int,
) -> tuple[list[CitedClaim], int]:
    """`ground_claims`, but each claim verified ALONE (one verify call per claim, run concurrently)
    — the BRIDGE's defeat for the `verify_grounding/v2` BATCH-LENIENCY effect: the gate grounds a
    plausible behavioral claim more readily inside a coherent BATCH than in isolation (measured live
    4/5 batched vs 0/5 isolated, same claims/chunk; genuine support grounds 3/3 either way). Reuses
    the UNCHANGED gate via `ground_claims` at N=1 — same prompt, same conservative keep rule, no new
    grounding path.

    The SAME full `chunks` set is passed to EVERY call (NOT cited-chunk-only): the evidence variable
    is held constant so only batch→single changes, and a claim whose support straddles a chunk
    boundary still sees its sibling chunk (no false-negative class introduced).

    Per-claim FAIL-OPEN: `ground_claims` returns `([], 0)` on a `ModelCallError`, so one failed claim
    drops ONLY itself (siblings still ground) — graceful degradation, valid because the bridge has no
    refuse state (vs the batched all-or-nothing fail-open). Survivors are returned in INPUT order and
    tokens are summed. Bridge-only (the summarizer + the `/ask` verify node keep batched grounding).
    """
    if not claims:
        return [], 0
    sem = asyncio.Semaphore(_ISOLATED_GROUND_CONCURRENCY)

    async def _one(claim: CitedClaim) -> tuple[list[CitedClaim], int]:
        async with sem:
            return await ground_claims(summary, [claim], chunks, max_tokens=max_tokens)

    results = await asyncio.gather(*(_one(c) for c in claims))
    kept = [c for c, (surv, _t) in zip(claims, results, strict=True) if surv]
    tokens = sum(t for _surv, t in results)
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
            prompt_tag=prompt_tag_for("assess_relevance"),
        )
    except ModelCallError as e:
        logger.warning("relevance.failed", error=str(e)[:160])
        return RelevanceAssessment(responsive=False, reason="responsiveness check failed"), 0
    return relevance, tokens
