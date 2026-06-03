"""The reason-then-ground bridge (Surface §11) — joins Surface A (grounded) and B (ungrounded).

Reason freely over retrieved evidence (Stage 1, the shared expert core), extract the discrete
claims that reasoning made (Stage 1.5), then run each through the UNCHANGED `verify_grounding/v2`
gate (Stage 2) so only vault-supported claims are presented as CITED — the rest stay inside the
labelled ungrounded analysis. Gives expert-mode reach a trust floor WITHOUT touching the `/ask`
`answer_query` graph (which is never imported here).

Deliberate contract decisions (see the §11 plan + ADR-0016):
- **Standalone: verify-only, NO `assess_relevance`.** That gate judges whole-answer
  responsiveness; the standalone bridge's grounded subset is "which reasoned claims are
  vault-supported," not "a direct answer," so `assess_relevance` would over-refuse.
- **Consented escalation, present-as-answer (`present_as_answer=True`, ADR-0016): verify +
  `assess_relevance`.** When the consented A→B escalation presents the grounded subset AS the
  answer (the spec-reserved "the bridge output IS the answer"), it DOES add the responsiveness
  gate — the grounded claims are promoted to a presented grounded answer ONLY when they are
  non-empty AND responsive (`BridgeAnswer.presented`). Otherwise it falls back to the labelled
  analysis. The presented body is the grounded `CitedClaim`s; the ungrounded analysis is never
  promoted (the surface fences it). The `/ask` `answer_query` graph is still never imported.
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
from memex.agents.grounding import assess_responsiveness, ground_claims, ground_claims_isolated
from memex.core.config import get_settings
from memex.core.errors import ModelCallError
from memex.core.text import claim_grounded_only_by_name
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

    # Present-as-answer (ADR-0016, the consented A→B escalation): when the caller asked to present
    # the grounded subset AS a direct answer, these carry the responsiveness verdict. On the
    # standalone path they stay at their defaults (the gate is never run) — the surface then renders
    # the labelled-analysis view, byte-identical to before.
    present_as_answer: bool = False
    responsive: bool | None = None  # None = the gate was not run (standalone / zero-grounded)
    relevance_reason: str = ""
    # Deterministic synthesis of the PRESENTED claims — the `assess_responsiveness` input. NEVER
    # the ungrounded analysis or the extractor's free summary (no ungrounded text reaches the answer).
    answer_headline: str = ""
    # The name-only-filtered subset actually shown when presenting (ADR-0016 audit rec 1):
    # `presented_claims ⊆ grounded_claims`, with claims whose cited chunk only NAMES the entity
    # (a bare list/heading) held back. The presented surface renders THIS; the labelled-analysis
    # fallback + the footer counts keep `grounded_claims` (the full gate output). Empty on the
    # standalone path (gate not run).
    presented_claims: list[CitedClaim] = []

    @property
    def presented(self) -> bool:
        """True only when the consented escalation asked to present-as-answer AND a non-empty
        PRESENTABLE subset survived the name-only guard AND it is responsive — the single switch
        the surfaces branch on."""
        return self.present_as_answer and bool(self.presented_claims) and self.responsive is True


async def reason_then_ground(
    question: str,
    *,
    scope_doc_ids: list[str] | None = None,
    evidence_k: int = _EVIDENCE_K,
    present_as_answer: bool = False,
    correlation_id: str | None = None,
    on_phase: Callable[[str], None] | None = None,
) -> BridgeAnswer:
    """Reason over the vault's evidence, then ground the reasoned claims (Surface §11).

    `present_as_answer` (ADR-0016, the consented A→B escalation): when True AND the grounded
    subset is non-empty, additionally run the `assess_relevance` responsiveness gate so the
    surface can present the grounded claims AS a direct answer (`BridgeAnswer.presented`). The
    standalone surface leaves it False → the gate is never run, the labelled-analysis view is
    byte-identical to before.
    """
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
                    prompt_tag="extract_claims@v2",
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
        # ISOLATED re-verification (default ON) defeats the `verify_grounding/v2` BATCH-LENIENCY
        # effect — the gate grounds a plausible behavioral claim more readily inside a coherent
        # BATCH than alone (measured 4/5 batched vs 0/5 isolated). Verify each claim at N=1; the
        # kill-switch reverts to the batched gate. Bridge-only (summarizer + /ask keep batched).
        isolated = settings.agents.bridge_isolated_grounding_enabled
        ground = ground_claims_isolated if isolated else ground_claims
        grounded, t_ground = await ground(
            analysis[:300], groundable, reranked, max_tokens=_GROUND_MAX_TOKENS
        )
        # NAME-ONLY GROUNDING BACKSTOP (membership-aware): the `verify_grounding/v2` gate
        # intermittently grounds a BEHAVIORAL claim against a chunk that merely NAMES the entity (a
        # bare list/heading) — the residual the isolated re-verification doesn't catch (it still
        # grounds 5 such claims at N=1). Demote any grounded claim grounded ONLY by name (the SAME
        # deterministic rule the `/ask` verify node uses, `claim_grounded_only_by_name`):
        # membership/existence + unrecognised predicates are KEPT (fail-open). DEMOTION-ONLY + the
        # bridge has no refuse state ⇒ this only moves a claim into the labelled analysis. Shrinks
        # `grounded` ITSELF (before the present/standalone split), so the footer counts, the
        # labelled fallback, AND both bridge surfaces reflect it. `ground_claims` (summarizer +
        # `/ask`) is untouched. Same kill-switch as the (now membership-aware) presentation guard.
        if settings.agents.bridge_name_only_guard_enabled and grounded:
            chunk_by_id = {c.chunk_id: c for c in reranked}
            kept = [
                c
                for c in grounded
                if not (
                    (ch := chunk_by_id.get(c.source_chunk_id)) is not None
                    and claim_grounded_only_by_name(c.claim, ch.text)
                )
            ]
            if len(kept) != len(grounded):
                logger.info(
                    "bridge.name_only_grounding_demoted",
                    demoted=len(grounded) - len(kept),
                    grounded=len(grounded),
                )
                grounded = kept
        grounded_ids = {gc.source_chunk_id for gc in grounded}
        grounded_sources = [c for c in reranked if c.chunk_id in grounded_ids]

        # Present-as-answer (ADR-0016): ONLY the consented escalation, and ONLY when a non-empty
        # PRESENTABLE subset survives — promote it to a presented answer IFF it is also responsive.
        # NAME-ONLY GUARD (audit rec 1), now MEMBERSHIP-AWARE: hold back a grounded claim grounded
        # only by name (the same `claim_grounded_only_by_name` rule). After the grounding backstop
        # above this is an idempotent no-op for behavioural claims (already demoted) and correctly
        # KEEPS membership claims a name-list genuinely grounds — kept as defense-in-depth. The
        # headline is a DETERMINISTIC join of the PRESENTABLE claims — never the ungrounded
        # analysis, never the extractor's free summary.
        responsive: bool | None = None
        relevance_reason = ""
        answer_headline = ""
        presented_claims: list[CitedClaim] = []
        t_relevance = 0
        if present_as_answer and grounded:
            src_by_id = {c.chunk_id: c for c in grounded_sources}
            guard_on = settings.agents.bridge_name_only_guard_enabled
            presentable = [
                c
                for c in grounded
                if not (
                    guard_on
                    and c.source_chunk_id in src_by_id
                    and claim_grounded_only_by_name(c.claim, src_by_id[c.source_chunk_id].text)
                )
            ]
            held_back = len(grounded) - len(presentable)
            if held_back:
                logger.info(
                    "bridge.name_only_held_back", held_back=held_back, grounded=len(grounded)
                )
            # Guard the gate on `presentable` (not `grounded`): an all-filtered case skips the
            # responsiveness call entirely (no phantom `responsive=True` on a result that falls back).
            if presentable:
                answer_headline = " ".join(c.claim for c in presentable)
                verdict, t_relevance = await assess_responsiveness(
                    question, answer_headline, [c.claim for c in presentable]
                )
                responsive = verdict.responsive
                relevance_reason = verdict.reason
                presented_claims = presentable

        logger.info(
            "bridge.done",
            extracted=len(candidates),
            groundable=len(groundable),
            grounded=len(grounded),
            isolated_grounding=isolated,
            present_as_answer=present_as_answer,
            presented=len(presented_claims),
            responsive=responsive,
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
            tokens=t_reason + t_extract + t_ground + t_relevance,
            scope_doc_ids=scope,
            correlation_id=correlation_id,
            present_as_answer=present_as_answer,
            responsive=responsive,
            relevance_reason=relevance_reason,
            answer_headline=answer_headline,
            presented_claims=presented_claims,
        )
    finally:
        clear_run_context()
