"""Surface B — the UNGROUNDED reasoning expert mode (ADR-0013).

This is the deliberate INVERSE of the grounded `/ask` + chat surfaces. Where those
answer ONLY what the vault supports (refusal_cf=1.0, zero hallucination, every claim
grounded by a cited chunk), this surface answers ANALYTICAL / ADVISORY / SYNTHESIS
questions from the model's own knowledge, *reasoned over* retrieved evidence — relaxing
literal grounding to "supported by the evidence set + sound reasoning". It is fenced
behind `AgentsSettings.expert_mode_enabled` (default off) and surfaced only on CLI
(`memex expert`) + webui (`/expert`), NEVER on the grounded path and NEVER on MCP.

The contract inversion is explicit and load-bearing:
  - It calls `complete_reasoning` (free-text, thinking mode), NOT `complete_structured`.
  - It NEVER runs `verify` / `assess_relevance` — there is no grounding gate here.
  - Every answer carries `EXPERT_PROVENANCE_NOTE` so the surface labels it "model
    knowledge, not your verified vault answer".

Pipeline = retrieve (hybrid) → rerank → reason. Retrieval is for CONTEXT, not grounding:
the evidence anchors and sharpens the analysis, but the model may reason beyond it and
must say so when it does (see `prompts/expert_answer/v1.md`).
"""

from __future__ import annotations

from collections.abc import Callable

import structlog
import ulid
from pydantic import BaseModel

from memex.core.config import get_settings
from memex.core.types import Chunk
from memex.models.client import complete_reasoning, split_think
from memex.observability import bind_run_context, clear_run_context
from memex.prompts import prompt_tag_for, render_messages
from memex.retrieve import (
    cross_encoder_rerank,
    hybrid_search,
    hybrid_search_in_docs,
)

logger = structlog.get_logger(__name__)

# How many candidates to retrieve before reranking down to the evidence set.
_CANDIDATE_K = 50
# Default evidence chunks passed to the reasoner.
_EVIDENCE_K = 8
# Per-chunk snippet length in the evidence block — enough to be useful context
# without flooding the reasoning window.
_SNIPPET_CHARS = 800
# Room for a thorough reasoned analysis. Generous because expert answers are
# discursive (reasoning + conclusion), but bounded so a runaway can't hang.
_ANSWER_MAX_TOKENS = 1536

# The standing provenance label every expert answer carries. Surfaces stamp it so the
# user always knows this is ungrounded model knowledge, NOT a verified vault answer.
EXPERT_PROVENANCE_NOTE = (
    "Ungrounded expert analysis — the model's own knowledge reasoned over your "
    "documents, not a verified answer grounded in your vault. Treat specific facts "
    "as claims to check against the cited sources."
)


class ExpertEvidence(BaseModel):
    """One retrieved chunk shown to the reasoner as context (NOT a grounding cite)."""

    chunk_id: str
    document_id: str
    title: str
    section: str | None = None
    page: int | None = None
    snippet: str


class ExpertAnswer(BaseModel):
    """The result of an ungrounded expert reasoning pass (Surface B).

    `answer` is the reasoned response; `reasoning` is a `<think>` trace IF the model
    emitted a tagged one (normally None on the live 4B — its CoT is inline in `answer`).
    `evidence` is the retrieved context the reasoning drew on, surfaced for the user to
    verify against. `provenance_note` is always `EXPERT_PROVENANCE_NOTE`.
    """

    question: str
    answer: str
    reasoning: str | None = None
    evidence: list[ExpertEvidence] = []
    provenance_note: str = EXPERT_PROVENANCE_NOTE
    model: str = ""
    tokens: int = 0
    scope_doc_ids: list[str] = []
    correlation_id: str = ""


def to_evidence(chunk: Chunk) -> ExpertEvidence:
    section = chunk.heading_path[-1] if chunk.heading_path else None
    snippet = chunk.text.strip()
    if len(snippet) > _SNIPPET_CHARS:
        snippet = snippet[:_SNIPPET_CHARS].rstrip() + " …"
    return ExpertEvidence(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        title=chunk.document_title or chunk.document_id,
        section=section,
        page=chunk.page,
        snippet=snippet,
    )


async def reason_over_evidence(
    question: str,
    *,
    scope_doc_ids: list[str] | None = None,
    evidence_k: int = _EVIDENCE_K,
    enable_thinking: bool = False,
    on_phase: Callable[[str], None] | None = None,
) -> tuple[str, list[Chunk], int]:
    """Stage 1 of BOTH expert mode and the reason-then-ground bridge (Surface §11): retrieve
    (optionally scoped) → rerank → ONE free-text reasoning pass over the evidence. Returns
    `(raw_text, reranked, tokens)` — `reranked` are the REAL `Chunk`s (expert renders them as
    `ExpertEvidence`; the bridge grounds its extracted claims against them), and `raw_text` is
    the un-split model output (the caller runs `split_think`).

    Observability-agnostic: the CALLER owns run-context binding (so the bound `correlation_id`
    threads these logs) and the model-id choice. `on_phase` (if given) receives the
    "Retrieving evidence" / "Reasoning" progress phases; a failing sink never aborts the call.
    """

    def _emit(phase: str) -> None:
        if on_phase is None:
            return
        try:
            on_phase(phase)
        except Exception:
            logger.warning("reason.on_phase_failed", phase=phase)

    # Blank-strip + dedup (order-preserving), mirroring the grounded scope path.
    scope = list(dict.fromkeys(d.strip() for d in (scope_doc_ids or []) if d.strip()))

    _emit("Retrieving evidence")
    if scope:
        candidates = await hybrid_search_in_docs(question, scope, k=_CANDIDATE_K)
    else:
        candidates = await hybrid_search(question, k=_CANDIDATE_K)
    reranked = (
        await cross_encoder_rerank(question, candidates, top_k=evidence_k) if candidates else []
    )
    evidence = [to_evidence(c) for c in reranked]
    logger.info("reason.retrieved", candidates=len(candidates), evidence=len(evidence))

    _emit("Reasoning")
    messages = render_messages("expert_answer", question=question, evidence=evidence)
    text, tokens = await complete_reasoning(
        messages,
        enable_thinking=enable_thinking,
        max_tokens=_ANSWER_MAX_TOKENS,
        prompt_tag=prompt_tag_for("expert_answer"),
    )
    logger.info("reason.done", tokens=tokens, chars=len(text))
    return text, reranked, tokens


async def expert_answer(
    question: str,
    *,
    scope_doc_ids: list[str] | None = None,
    evidence_k: int = _EVIDENCE_K,
    enable_thinking: bool = False,
    correlation_id: str | None = None,
    on_phase: Callable[[str], None] | None = None,
) -> ExpertAnswer:
    """Answer `question` in ungrounded expert mode (Surface B, ADR-0013).

    A thin wrapper over the shared `reason_over_evidence` Stage-1 core: it owns the run-context
    lifecycle + the model-id label, then packages the reasoning into an `ExpertAnswer`. Does NOT
    verify or gate — the answer is the model's reasoned opinion, labelled `EXPERT_PROVENANCE_NOTE`.

    `enable_thinking` defaults to FALSE: on the live 4B (verified 2026-06-01) the thinking mode
    emits a verbose, UNTAGGED scratchpad that consumes the whole token budget before the answer
    and can't be cleanly split; the model still reasons in its prose. The kwarg (with `split_think`)
    is plumbed for a future model that emits a separable trace.
    """
    correlation_id = correlation_id or str(ulid.ULID())
    # Defensively reset any leaked prior context BEFORE the try; the bind lives INSIDE the try
    # so `finally: clear_run_context()` always unbinds it (a pre-try raise can't leak it).
    clear_run_context()
    settings = get_settings()
    # `models.reasoner` is a RESERVED hook (ADR-0013, UNUSED in v1): when set it retargets the
    # reasoning call to that id, but v1 does NOT serve it — it must already be the live daemon's
    # served model (no auto swap-in; a mis-set id 404s). Default None → the orchestrator answers.
    model = settings.models.reasoner or settings.models.orchestrator
    scope = list(dict.fromkeys(d.strip() for d in (scope_doc_ids or []) if d.strip()))

    try:
        bind_run_context(correlation_id, query_preview=f"expert {question[:60]}")
        text, reranked, tokens = await reason_over_evidence(
            question,
            scope_doc_ids=scope,
            evidence_k=evidence_k,
            enable_thinking=enable_thinking,
            on_phase=on_phase,
        )
        trace, body = split_think(text)
        return ExpertAnswer(
            question=question,
            answer=body or text,
            reasoning=trace,
            evidence=[to_evidence(c) for c in reranked],
            model=model,
            tokens=tokens,
            scope_doc_ids=scope,
            correlation_id=correlation_id,
        )
    finally:
        clear_run_context()
