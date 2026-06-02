"""The reason-then-ground bridge (Surface §11) — joins Surface A (grounded) and B (ungrounded).

Reason freely over retrieved evidence (Stage 1, the shared expert core), extract the discrete
claims that reasoning made (Stage 1.5), then run each through the UNCHANGED `verify_grounding/v2`
gate (Stage 2) so only vault-supported claims are presented as CITED — the rest stay inside the
labelled ungrounded analysis. Gives expert-mode reach a trust floor WITHOUT touching the `/ask`
`answer_query` graph (which is never imported here).

Deliberate v1 contract decisions (see the §11 plan):
- **verify-only, NO `assess_relevance`.** That gate judges whole-answer responsiveness; the
  bridge's grounded subset is "which reasoned claims are vault-supported," not "a direct answer,"
  so `assess_relevance` would over-refuse. It is reserved for the future consented A→B escalation.
- **Zero grounded ≠ refuse.** The analysis is still useful; return it labelled with an empty
  grounded subset (no citation chrome). The bridge has no refuse state.
- **Faithfulness guard (two layers):** an extractor-not-generator prompt PLUS a deterministic
  drop of any extracted claim whose `source_chunk_id` (after `repair`) doesn't resolve to a
  reranked chunk — closes the "fabricated-but-coincidentally-grounded" hole the verifier is blind to.
- **Known gap (matches the summarizer):** the numeric-aggregate backstop lives in the `/ask`
  `verify` NODE, not in `ground_claims`, so bare computed-figure claims are out of v1 scope (the
  extractor prompt is told to avoid them).
"""

from __future__ import annotations

from collections.abc import Callable

import structlog
import ulid
from pydantic import BaseModel

from memex.agents.answering import CitedClaim, DraftAnswer, repair_claim_chunk_ids
from memex.agents.expert import (
    ExpertEvidence,
    reason_over_evidence,
    to_evidence,
)
from memex.agents.grounding import ground_claims
from memex.core.config import get_settings
from memex.core.errors import ModelCallError
from memex.core.types import Chunk
from memex.models.client import complete_structured, split_think
from memex.observability import bind_run_context, clear_run_context
from memex.prompts import render_messages

logger = structlog.get_logger(__name__)

# Default evidence chunks reasoned over + grounded against (matches expert mode).
_EVIDENCE_K = 8
# Per-call extraction output budget — DraftAnswer (≤600 summary + ≤8×~435-char claims ≈ 1.1k).
_EXTRACT_MAX_TOKENS = 1024
# The grounding budget — match the summarizer's `verify_grounding/v2` call exactly.
_GROUND_MAX_TOKENS = 768

BRIDGE_PROVENANCE_NOTE = (
    "Reason-then-ground: the ANALYSIS is ungrounded model reasoning; only the GROUNDED CLAIMS "
    "passed the same vault-grounding check as a normal /ask answer. Treat the analysis as claims "
    "to verify; the grounded claims are anchored to their cited sources."
)


class BridgeAnswer(BaseModel):
    """Result of a reason-then-ground pass (Surface §11): an ungrounded analysis PLUS the subset
    of its claims that survived the unchanged `/ask` grounding gate (presented as cited)."""

    question: str
    analysis: str
    grounded_claims: list[CitedClaim] = []
    grounded_sources: list[Chunk] = []
    evidence: list[ExpertEvidence] = []
    provenance_note: str = BRIDGE_PROVENANCE_NOTE
    n_extracted: int = 0
    n_grounded: int = 0
    model: str = ""
    tokens: int = 0
    scope_doc_ids: list[str] = []
    correlation_id: str = ""


async def reason_then_ground(
    question: str,
    *,
    scope_doc_ids: list[str] | None = None,
    evidence_k: int = _EVIDENCE_K,
    correlation_id: str | None = None,
    on_phase: Callable[[str], None] | None = None,
) -> BridgeAnswer:
    """Reason over the vault's evidence, then ground the reasoned claims (Surface §11)."""
    correlation_id = correlation_id or str(ulid.ULID())
    # Defensive reset BEFORE the try; the bind lives inside so `finally` always unbinds it.
    clear_run_context()
    settings = get_settings()
    # `models.reasoner` is the same RESERVED hook expert mode uses (ADR-0013): default None →
    # the orchestrator reasons; a set id must already be the served model. Stamped for display.
    model = settings.models.reasoner or settings.models.orchestrator
    scope = list(dict.fromkeys(d.strip() for d in (scope_doc_ids or []) if d.strip()))

    def _emit(phase: str) -> None:
        if on_phase is None:
            return
        try:
            on_phase(phase)
        except Exception:
            logger.warning("bridge.on_phase_failed", phase=phase)

    try:
        bind_run_context(correlation_id, query_preview=f"bridge {question[:60]}")

        # Stage 1 — reason (ungrounded) over retrieved evidence. Emits Retrieving/Reasoning.
        raw_text, reranked, t_reason = await reason_over_evidence(
            question,
            scope_doc_ids=scope,
            evidence_k=evidence_k,
            enable_thinking=False,
            on_phase=on_phase,
        )
        _trace, analysis = split_think(raw_text)
        analysis = analysis or raw_text
        evidence = [to_evidence(c) for c in reranked]

        _emit("Grounding claims")

        # Stage 1.5 — extract the discrete claims the analysis explicitly made (structured).
        # Fail-open: an extraction failure yields zero claims → the analysis-only path.
        candidates: list[CitedClaim] = []
        t_extract = 0
        if reranked:
            try:
                draft, t_extract = await complete_structured(
                    prompt=render_messages(
                        "extract_claims",
                        question=question,
                        analysis=analysis,
                        evidence=evidence,
                    ),
                    schema=DraftAnswer,
                    max_tokens=_EXTRACT_MAX_TOKENS,
                    prompt_tag="extract_claims@v1",
                )
                candidates = draft.claims
            except ModelCallError as e:
                logger.warning("bridge.extract_failed", error=str(e)[:160])

        # Stage 2 — repair cited ids, then deterministically drop any claim whose id doesn't
        # resolve to a reranked chunk (faithfulness/groundability floor the verifier is blind
        # to), THEN the UNCHANGED `verify_grounding/v2` gate keeps only the supported claims.
        reranked_ids = {c.chunk_id for c in reranked}
        repaired, _stats = repair_claim_chunk_ids(candidates, reranked)
        groundable = [c for c in repaired if c.source_chunk_id in reranked_ids]
        grounded, t_ground = await ground_claims(
            analysis[:300], groundable, reranked, max_tokens=_GROUND_MAX_TOKENS
        )
        grounded_ids = {gc.source_chunk_id for gc in grounded}
        grounded_sources = [c for c in reranked if c.chunk_id in grounded_ids]

        logger.info(
            "bridge.done",
            extracted=len(candidates),
            groundable=len(groundable),
            grounded=len(grounded),
        )
        return BridgeAnswer(
            question=question,
            analysis=analysis,
            grounded_claims=grounded,
            grounded_sources=grounded_sources,
            evidence=evidence,
            n_extracted=len(candidates),
            n_grounded=len(grounded),
            model=model,
            tokens=t_reason + t_extract + t_ground,
            scope_doc_ids=scope,
            correlation_id=correlation_id,
        )
    finally:
        clear_run_context()
