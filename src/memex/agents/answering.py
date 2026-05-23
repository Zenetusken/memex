"""
Memex Answering Agent
=====================

A LangGraph state machine for the answering pipeline. Receives a query,
retrieves and reranks candidates, drafts an answer with explicit
citations, verifies that every claim is grounded in retrieved chunks,
and either returns the answer or refuses with a useful reason.

Design principles (from GUIDELINES.md):

  - Typed state. All inter-node communication is via the pydantic
    AnswerState model, never raw dicts.
  - Explicit edges. Routing is in functions named route_*, not buried
    in node logic.
  - Bounded loops. Every loop has a budget (regenerate_attempts,
    nodes_traversed, tokens_used). Exceeding budget routes to refuse,
    never to silent failure.
  - Refusal is first-class. The graph has a refuse node, not just
    exception handling. A useful refusal is the right answer when the
    corpus cannot support a confident one.
  - Verification is mandatory. Every drafted answer passes through
    grounding verification before reaching the user.
  - Observable. The correlation_id flows through structlog ContextVars
    and a Langfuse CallbackHandler attached to the graph; logs and
    traces are joined by that single ID (ADR-0004).

Usage:

    response = await answer_query("What does Smith 2024 say about reflexivity?")
    if response.answered:
        print(response.summary)
        for claim in response.claims:
            print(f"  - {claim.claim} [source: {claim.source_chunk_id}]")
    else:
        print(f"Refused: {response.refusal_reason}")
"""

from __future__ import annotations

import threading
from typing import Annotated, Literal, TypedDict

import structlog
import ulid
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, create_model

from memex.core.errors import AnswerStateInvariantError
from memex.core.text import strip_chart_extracted_for_index
from memex.core.types import Chunk
from memex.models.client import complete_structured
from memex.observability.tracing import (
    bind_run_context,
    callback_handler,
    clear_run_context,
    run_attributes,
)
from memex.prompts import render_prompt
from memex.retrieve import (
    cross_encoder_rerank,
    hybrid_search,
    hybrid_search_in_docs,
)

logger = structlog.get_logger(__name__)


# Node return contract. `total=False` because each node updates only
# the fields it touches; typos in keys fail at type-check time instead
# of becoming silent state drops.
class AnswerStateUpdate(TypedDict, total=False):
    candidates: list[Chunk]
    reranked: list[Chunk]
    graph_expanded_doc_ids: list[str]
    sufficiency: SufficiencyAssessment
    draft: DraftAnswer
    verification: VerificationResult
    regenerate_attempts: int
    nodes_traversed: int
    tokens_used: int
    final: FinalResponse


# =============================================================================
# Types — every cross-node payload is one of these.
# =============================================================================


class CitedClaim(BaseModel):
    """A single factual claim with the chunk that supports it."""

    claim: str = Field(description="One factual statement, self-contained.")
    source_chunk_id: str = Field(description="The chunk_id that supports the claim.")
    confidence: Literal["high", "medium", "low"]


class DraftAnswer(BaseModel):
    """The model's structured draft. Verified before it reaches the user."""

    summary: str = Field(description="One or two sentences overviewing the answer.")
    claims: list[CitedClaim] = Field(
        description="Detailed claims, each with a citation."
    )


class SufficiencyAssessment(BaseModel):
    """Did retrieval surface enough material to answer the query?"""

    sufficient: bool
    reason: str = Field(
        description="If insufficient, what is missing? If sufficient, why?",
        max_length=500,
    )

    # P3.3 v6 (audit 2026-05-22): xgrammar enforces this max_length at
    # the grammar level, bounding emission so the model can't run away
    # in the `reason` field and trip max_tokens before closing the
    # JSON object. The crash this fixes: on counterfactual queries
    # like "What is the energy cost of FP128..." (which the deck
    # doesn't cover), Qwen3-4B-AWQ would emit a 2000+ char ramble in
    # `reason` explaining why the chunks were insufficient — past
    # max_tokens=640, JSON gets cut off, schema-validate fails,
    # ModelCallError aborts the eval. 500 chars is plenty for "one
    # sentence" per the prompt.


class VerificationResult(BaseModel):
    """Per-claim grounding verdict from the verifier pass.

    Indices refer to 0-based positions in the draft's `claims` list.
    Matching by index — rather than echoing claim text — avoids silent
    failures when the verifier paraphrases (which LLMs do routinely).
    """

    grounded: list[int] = Field(
        description="0-indexed positions in draft.claims whose cited chunk supports the claim.",
    )
    ungrounded: list[int] = Field(
        description="0-indexed positions whose cited chunk does NOT support the claim.",
    )
    ungrounded_reasons: list[str] = Field(
        default_factory=list,
        description="Optional, parallel to `ungrounded`: one short reason per ungrounded claim.",
    )

    @property
    def all_grounded(self) -> bool:
        return not self.ungrounded


class FinalResponse(BaseModel):
    """What gets returned to the caller (CLI, web UI, MCP server)."""

    answered: bool
    summary: str | None = None
    claims: list[CitedClaim] = []
    refusal_reason: str | None = None
    used_chunks: list[Chunk] = []

    # Observability fields — always populated, useful for trace correlation.
    correlation_id: str
    tokens_used: int
    nodes_traversed: int
    regenerate_attempts: int


class AnswerState(BaseModel):
    """The graph's working state. Every node reads and writes this."""

    # --- Input ---
    query: str
    correlation_id: str = Field(default_factory=lambda: str(ulid.ULID()))

    # --- Retrieval ---
    candidates: list[Chunk] = []
    reranked: list[Chunk] = []
    # Documents added to the candidate pool by `expand_graph` — kept on
    # state so the trace + `FinalResponse.used_chunks` consumers can
    # tell graph-expanded chunks from lexically-retrieved ones.
    graph_expanded_doc_ids: list[str] = []

    # --- Graph-expansion budgets ---
    # Toggle the whole expansion step off (e.g. when the graph store
    # isn't available or for benchmarking pure flat retrieval).
    graph_expansion_enabled: bool = True
    # How many distinct source documents from the initial retrieval
    # to seed graph traversal from. Smaller = less noise + fewer
    # graph queries; larger = wider reach.
    graph_expansion_budget: int = 3
    # Chunks pulled from each neighbour document via filtered hybrid
    # search. The reranker is the quality gate downstream.
    chunks_per_neighbor: int = 2

    # --- Reasoning intermediates ---
    sufficiency: SufficiencyAssessment | None = None
    draft: DraftAnswer | None = None
    verification: VerificationResult | None = None

    # --- Budgets (the difference between an agent and a runaway loop) ---
    regenerate_attempts: int = 0
    max_regenerate_attempts: int = 2

    nodes_traversed: int = 0
    max_nodes_traversed: int = 20

    tokens_used: int = 0
    token_budget: int = 8000

    # --- Output ---
    final: FinalResponse | None = None

    def over_budget(self) -> bool:
        return (
            self.nodes_traversed >= self.max_nodes_traversed
            or self.tokens_used >= self.token_budget
        )


# =============================================================================
# Nodes — each is async, takes state, returns a dict of state updates.
# =============================================================================


async def retrieve(state: AnswerState) -> AnswerStateUpdate:
    """Hybrid search: BM25 (SQLite FTS5) + dense (LanceDB) + RRF fusion."""
    log = logger.bind(node="retrieve")
    log.info("start", query_len=len(state.query))
    candidates = await hybrid_search(state.query, k=50)
    log.info("done", candidate_count=len(candidates))
    return {
        "candidates": candidates,
        "nodes_traversed": state.nodes_traversed + 1,
    }


async def expand_graph(state: AnswerState) -> AnswerStateUpdate:
    """Best-effort wrapper around `_expand_graph_impl`.

    Graph expansion is strictly additive — the agent must answer
    correctly when the graph layer is unavailable for any reason
    (ryugraph not installed, settings not initialised in a test
    fixture, transient graph-store I/O failure, ...). We catch the
    *expected* failure modes here and log; programming bugs
    (`AttributeError` after a schema change, `NameError`, etc.)
    propagate so we don't silently degrade on a real defect.
    """
    from memex.core.errors import ConfigurationError, MemexError

    try:
        return await _expand_graph_impl(state)
    except (ImportError, ConfigurationError, MemexError, OSError) as exc:
        logger.bind(node="expand_graph").warning(
            "expand_graph.skipped",
            reason=type(exc).__name__,
            error=str(exc),
        )
        return {"nodes_traversed": state.nodes_traversed + 1}


async def _expand_graph_impl(state: AnswerState) -> AnswerStateUpdate:
    """Augment the candidate pool with chunks from entity-sharing docs.

    The agent's flat retrieval (BM25 + dense + RRF) finds documents
    that lexically or semantically match the query. The graph store
    holds `MENTIONS` and `CITES` edges between documents (written by
    the enrich stage). This node uses the entity-sharing neighbours
    today: it takes the top N unique source documents from
    `state.candidates`, asks the graph for their one-hop
    `shares_entity` neighbours, and pulls the BM25/dense-fused top
    chunks from each neighbour via `hybrid_search_in_docs`. Those
    chunks are appended to `state.candidates` so the next node
    (`rerank`) scores them against the query on equal footing.

    CITES traversal is queued in the v1.x backlog — see
    `index/graph_store.py:neighbors`. When that lands, the relation
    field on the returned `GraphNeighbor` will distinguish the two.

    Hard-stops early when `graph_expansion_enabled=False` or when
    retrieval returned no candidates. Expected failure modes (missing
    graph store, missing settings, store I/O errors) are swallowed by
    the `expand_graph` wrapper above; programming bugs propagate.
    """
    log = logger.bind(node="expand_graph")

    if not state.graph_expansion_enabled or not state.candidates:
        log.info("skip", reason="disabled_or_empty")
        return {"nodes_traversed": state.nodes_traversed + 1}

    # Top N unique source documents in retrieval order.
    seen_docs: list[str] = []
    for c in state.candidates:
        if c.document_id not in seen_docs:
            seen_docs.append(c.document_id)
        if len(seen_docs) >= state.graph_expansion_budget:
            break

    log.info("start", source_docs=len(seen_docs))

    from memex.core.config import get_settings
    from memex.index.graph_store import GraphStore

    vault_path = get_settings().vault_path
    store = await GraphStore.open(vault_path)

    neighbor_doc_ids: list[str] = []
    seen_neighbors: set[str] = set()
    try:
        for doc_id in seen_docs:
            neighbours = await store.neighbors(
                doc_id, limit=state.graph_expansion_budget
            )
            for n in neighbours:
                if n.doc_id in seen_neighbors:
                    continue
                seen_neighbors.add(n.doc_id)
                neighbor_doc_ids.append(n.doc_id)
    finally:
        await store.close()

    # Drop neighbours that are already in the candidate pool — they're
    # already in front of the reranker.
    existing_doc_ids = {c.document_id for c in state.candidates}
    new_doc_ids = [d for d in neighbor_doc_ids if d not in existing_doc_ids]

    if not new_doc_ids:
        log.info("no_new_neighbors")
        return {"nodes_traversed": state.nodes_traversed + 1}

    # Fetch per-doc top chunks. We ask for `chunks_per_neighbor * N`
    # in one pooled call; uneven distribution across docs is fine —
    # the reranker filters.
    k = state.chunks_per_neighbor * len(new_doc_ids)
    fetched = await hybrid_search_in_docs(state.query, new_doc_ids, k=k)

    # De-dup against existing candidates by chunk_id.
    existing_chunk_ids = {c.chunk_id for c in state.candidates}
    new_chunks = [c for c in fetched if c.chunk_id not in existing_chunk_ids]

    log.info(
        "done",
        neighbour_docs=len(new_doc_ids),
        added_chunks=len(new_chunks),
    )
    return {
        "candidates": state.candidates + new_chunks,
        "graph_expanded_doc_ids": new_doc_ids,
        "nodes_traversed": state.nodes_traversed + 1,
    }


async def rerank(state: AnswerState) -> AnswerStateUpdate:
    """Second-stage reranking. Backend selected by settings
    (`reranker_backend`): bge cross-encoder by default, Qwen3-Reranker
    when flagged. See `retrieve/rerank.py`.
    """
    import os

    log = logger.bind(node="rerank")
    log.info("start", candidate_count=len(state.candidates))
    # Default top_k=5 fits a 4096-token assembly budget when each chunk
    # is truncated at 1800 chars (~450 tokens) in the answer prompt:
    # 5 × 1800 × ~0.25 tok/char ≈ 2250 tokens for chunks + ~500 for
    # scaffolding + 1024 for output = ~3774 tokens, comfortably under
    # max-model-len=4096. The earlier `top_k=10` default paired with
    # an aggressive `truncate(700)` in the answer prompt clipped most
    # chunks to ~32% of their content (median chunk is 2172 chars);
    # bumping truncate to 1800 and dropping top_k to 5 trades retrieval
    # breadth for grounding fidelity, which the eval showed was the
    # winning trade. Users with longer-context model variants can raise
    # MEMEX_RERANK_TOP_K to recover breadth.
    try:
        top_k = int(os.environ.get("MEMEX_RERANK_TOP_K", "5"))
    except ValueError:
        top_k = 5
    top_k = max(1, top_k)
    reranked = await cross_encoder_rerank(state.query, state.candidates, top_k=top_k)
    log.info("done", reranked_count=len(reranked))
    return {
        "reranked": reranked,
        "nodes_traversed": state.nodes_traversed + 1,
    }


async def assess(state: AnswerState) -> AnswerStateUpdate:
    """Ask the orchestrator: do these chunks contain enough to answer?"""
    log = logger.bind(node="assess")
    log.info("start", reranked_count=len(state.reranked))

    # Short-circuit: if retrieval came back empty, no model call needed.
    if not state.reranked:
        log.info("short_circuit", reason="empty_reranked")
        return {
            "sufficiency": SufficiencyAssessment(
                sufficient=False,
                reason="No relevant content found in the vault for this query.",
            ),
            "nodes_traversed": state.nodes_traversed + 1,
        }

    # P3.3 v6 (audit 2026-05-22): the assess prompt has the same
    # `truncate(1200)` interaction with chart-extracted blocks as
    # the answer prompt. If chart noise eats the truncate budget,
    # the assess model only sees chart noise and can return
    # `sufficient: false` because it never reaches the prose +
    # Docling tables in the chunk. Strip the chart blocks here too
    # so all three prompt stages (assess / answer / verify) see the
    # same content view.
    stripped_chunks_for_assess = [
        c.model_copy(update={"text": strip_chart_extracted_for_index(c.text)})
        for c in state.reranked
    ]
    prompt = render_prompt(
        "assess_sufficiency",
        query=state.query,
        chunks=stripped_chunks_for_assess,
    )
    sufficiency, tokens = await complete_structured(
        prompt=prompt,
        schema=SufficiencyAssessment,
        prompt_tag="assess_sufficiency@v1",
    )

    return {
        "sufficiency": sufficiency,
        "tokens_used": state.tokens_used + tokens,
        "nodes_traversed": state.nodes_traversed + 1,
    }


async def answer(state: AnswerState) -> AnswerStateUpdate:
    """Generate a draft answer with explicit citations.

    On regeneration, includes feedback about which claims were
    ungrounded in the previous attempt.
    """
    log = logger.bind(node="answer")
    log.info("start", regenerate_attempt=state.regenerate_attempts)

    feedback = ""
    if (
        state.draft is not None
        and state.verification is not None
        and state.verification.ungrounded
    ):
        lines: list[str] = []
        for n, idx in enumerate(state.verification.ungrounded):
            if 0 <= idx < len(state.draft.claims):
                text = state.draft.claims[idx].claim
                reason = (
                    state.verification.ungrounded_reasons[n]
                    if n < len(state.verification.ungrounded_reasons)
                    else None
                )
                lines.append(f"- {text}" + (f" ({reason})" if reason else ""))
        if lines:
            feedback = (
                "The previous attempt included claims not supported by the cited chunks:\n"
                + "\n".join(lines)
                + "\nRevise: only make claims that the provided chunks directly support. "
                "Drop or rephrase any claim that goes beyond what the chunks state."
            )

    # P3.3 v5 (audit 2026-05-22 trace): strip `[chart-extracted]`
    # blocks from the chunk text before rendering the prompt.
    # Rationale: the v3 FTS-side strip prevented BM25 perturbation
    # but the chunks the agent reads STILL have chart blocks
    # (LanceDB stored full text + RRF preserves dense chunks first).
    # The truncate(1800) filter in the prompt template was getting
    # eaten by a chart block at the TOP of the chunk, pushing the
    # actual answer table (`| FP16 | FMA | 0.5x |`) past the
    # truncation cut-off. Stripping at prompt-render time keeps the
    # chunk's stored text intact (display, MCP UX, embedding signal)
    # while letting the answer LLM see the prose + Docling tables
    # the agent actually needs.
    stripped_chunks = [
        c.model_copy(update={"text": strip_chart_extracted_for_index(c.text)})
        for c in state.reranked
    ]
    prompt = render_prompt(
        "answer",
        query=state.query,
        chunks=stripped_chunks,
        feedback=feedback,
    )
    draft, tokens = await complete_structured(
        prompt=prompt,
        schema=DraftAnswer,
        prompt_tag="answer@v1",
    )

    return {
        "draft": draft,
        "tokens_used": state.tokens_used + tokens,
        "nodes_traversed": state.nodes_traversed + 1,
    }


async def verify(state: AnswerState) -> AnswerStateUpdate:
    """Independent grounding check.

    For each claim in the draft, ask: does the cited chunk actually
    support this claim? This is a separate model call with a focused
    prompt, not the same call that generated the draft.
    """
    log = logger.bind(node="verify")
    log.info(
        "start",
        claim_count=len(state.draft.claims) if state.draft else 0,
    )
    if state.draft is None:
        raise AnswerStateInvariantError(
            "verify called without a draft on AnswerState",
            context={"correlation_id": state.correlation_id, "node": "verify"},
        )

    # Empty-draft short-circuit. When the answer node legitimately
    # returns zero claims (the literal-presence rule in answer/v2
    # fires for queries whose answer isn't in the chunks), there is
    # nothing to verify. Calling the model with an empty `Draft
    # (indexed):` section is also dangerous: the model has no signal
    # to bound the `grounded`/`ungrounded` arrays (both are
    # `list[int]` with no length constraint in `VerificationResult`),
    # and xgrammar's strict JSON-schema mode then permits arbitrarily
    # long emissions. Under certain prompt phrasings this degenerates
    # into a runaway integer sequence (`grounded: [0, 1, 2, ...]+`
    # hitting `max_tokens=1024` with `finish_reason: length`) →
    # schema-validation failure → ModelCallError. Short-circuiting
    # both fixes that pathology AND saves the round-trip — an empty
    # draft IS the refusal signal from the answer stage; downstream
    # `compose` will correctly route to refuse.
    if not state.draft.claims:
        log.info("empty_draft_shortcircuit")
        return {
            "verification": VerificationResult(
                grounded=[], ungrounded=[], ungrounded_reasons=[]
            ),
            "nodes_traversed": state.nodes_traversed + 1,
        }

    # P3.3 v5: same strip as the answer node — the verify prompt's
    # literal-presence check needs to see the same chunk text the
    # answer model used. Otherwise the verifier might reject a
    # legitimate claim because the chart block ate the truncate
    # budget and the actual support text is past the cut-off.
    chunk_by_id = {
        c.chunk_id: c.model_copy(
            update={"text": strip_chart_extracted_for_index(c.text)}
        )
        for c in state.reranked
    }

    prompt = render_prompt(
        "verify_grounding",
        draft=state.draft,
        chunk_by_id=chunk_by_id,
    )

    # Bounded-schema construction. The default `VerificationResult` has
    # unbounded `list[int]` fields, which lets xgrammar emit arbitrarily
    # many integers when the model gets stuck (the same pathology as the
    # empty-draft case, but here it surfaces when the model loses track
    # of how many claims exist — observed on Qwen3-4B-AWQ in P4.2
    # Session 3a). Constraining the arrays to `max_length=n` where n is
    # the actual claim count lets xgrammar reject runaway emission at
    # the grammar level. Pydantic schemas are class-level, so we build
    # a dynamic subclass per call.
    n = len(state.draft.claims)
    # The dynamically-constructed schema retains `__name__ ==
    # "VerificationResult"` so test fakes that key canned responses by
    # schema name continue to match — see FakeLLM.__call__ in
    # tests/integration/test_answering_with_fakes.py.
    BoundedVerificationResult = create_model(
        "VerificationResult",
        grounded=(
            Annotated[list[int], Field(max_length=n)],
            Field(description=f"0-indexed positions in draft.claims (0..{n - 1}) whose cited chunk supports the claim."),
        ),
        ungrounded=(
            Annotated[list[int], Field(max_length=n)],
            Field(description=f"0-indexed positions in draft.claims (0..{n - 1}) whose cited chunk does NOT support the claim."),
        ),
        ungrounded_reasons=(
            Annotated[list[str], Field(max_length=n)],
            Field(default_factory=list, description="Optional, parallel to `ungrounded`: one short reason per ungrounded claim."),
        ),
    )

    bounded, tokens = await complete_structured(
        prompt=prompt,
        schema=BoundedVerificationResult,
        prompt_tag="verify_grounding@v2",
    )

    verification = VerificationResult(
        grounded=list(bounded.grounded),
        ungrounded=list(bounded.ungrounded),
        ungrounded_reasons=list(bounded.ungrounded_reasons),
    )

    return {
        "verification": verification,
        "tokens_used": state.tokens_used + tokens,
        "nodes_traversed": state.nodes_traversed + 1,
    }


async def regenerate(state: AnswerState) -> AnswerStateUpdate:
    """Bump the regeneration counter.

    No model call here — this is a routing-prep step the verify node's
    conditional edge routes to before going back to answer. The prior
    draft is intentionally left on `state.draft` so the answer node can
    look up the text of each ungrounded claim by index when building
    feedback for the next attempt.
    """
    log = logger.bind(node="regenerate")
    log.info("start", new_attempt=state.regenerate_attempts + 1)
    return {
        "regenerate_attempts": state.regenerate_attempts + 1,
        "nodes_traversed": state.nodes_traversed + 1,
    }


async def refuse(state: AnswerState) -> AnswerStateUpdate:
    """Compose a useful refusal explaining what's missing.

    Refusals are not failures — they're the right answer when the
    corpus can't support a confident one. The refusal carries the
    retrieved chunks so the user can see what *was* found.
    """
    log = logger.bind(node="refuse")
    log.info("start")

    if state.sufficiency and not state.sufficiency.sufficient:
        reason = state.sufficiency.reason
    elif state.verification and state.verification.ungrounded:
        reason = (
            f"I drafted an answer but couldn't ground "
            f"{len(state.verification.ungrounded)} claim(s) "
            f"after {state.regenerate_attempts} attempt(s). "
            "I won't return a partially hallucinated answer."
        )
    elif state.over_budget():
        reason = (
            "I exceeded my reasoning budget for this query. "
            "Try rephrasing or narrowing the question."
        )
    else:
        reason = "I couldn't construct a confident answer from your corpus."

    new_nodes = state.nodes_traversed + 1
    final = FinalResponse(
        answered=False,
        refusal_reason=reason,
        used_chunks=state.reranked,
        correlation_id=state.correlation_id,
        tokens_used=state.tokens_used,
        nodes_traversed=new_nodes,
        regenerate_attempts=state.regenerate_attempts,
    )
    return {"final": final, "nodes_traversed": new_nodes}


async def compose(state: AnswerState) -> AnswerStateUpdate:
    """Final formatting of a verified answer.

    Drops any claims that didn't pass verification (defense in depth —
    routing should already guarantee `all_grounded`, but if a partial
    verdict ever slips through we don't ship ungrounded claims).
    """
    log = logger.bind(node="compose")
    log.info("start")
    if state.draft is None or state.verification is None:
        raise AnswerStateInvariantError(
            "compose called without a verified draft on AnswerState",
            context={
                "correlation_id": state.correlation_id,
                "node": "compose",
                "has_draft": state.draft is not None,
                "has_verification": state.verification is not None,
            },
        )

    grounded_indices = set(state.verification.grounded)
    surviving_claims = [
        claim
        for i, claim in enumerate(state.draft.claims)
        if i in grounded_indices
    ]

    new_nodes = state.nodes_traversed + 1

    # Defense in depth: routing should have guaranteed all_grounded, but if
    # the verifier returned no grounded indices (or out-of-range ones) we
    # refuse rather than ship a summary with no supporting claims.
    if not surviving_claims:
        final = FinalResponse(
            answered=False,
            refusal_reason=(
                "Verification returned no grounded claims; I won't ship a "
                "summary without citations to back it."
            ),
            used_chunks=state.reranked,
            correlation_id=state.correlation_id,
            tokens_used=state.tokens_used,
            nodes_traversed=new_nodes,
            regenerate_attempts=state.regenerate_attempts,
        )
        return {"final": final, "nodes_traversed": new_nodes}

    used_chunk_ids = {c.source_chunk_id for c in surviving_claims}
    used_chunks = [c for c in state.reranked if c.chunk_id in used_chunk_ids]

    final = FinalResponse(
        answered=True,
        summary=state.draft.summary,
        claims=surviving_claims,
        used_chunks=used_chunks,
        correlation_id=state.correlation_id,
        tokens_used=state.tokens_used,
        nodes_traversed=new_nodes,
        regenerate_attempts=state.regenerate_attempts,
    )
    return {"final": final, "nodes_traversed": new_nodes}


# =============================================================================
# Routing — conditional edges, each returns the name of the next node.
# =============================================================================


def route_after_assess(state: AnswerState) -> Literal["answer", "refuse"]:
    """Sufficient -> answer. Insufficient or over-budget -> refuse."""
    if state.over_budget():
        return "refuse"
    if state.sufficiency and state.sufficiency.sufficient:
        return "answer"
    return "refuse"


def route_after_verify(
    state: AnswerState,
) -> Literal["compose", "regenerate", "refuse"]:
    """Grounded -> compose. Ungrounded with budget -> regenerate. Otherwise -> refuse.

    Compose is checked *before* the budget gate: if verification passed
    we already paid for a valid answer, and `compose` does no model
    work, so a late budget breach shouldn't force us to discard it.
    """
    if state.verification and state.verification.all_grounded:
        return "compose"
    if state.over_budget():
        return "refuse"
    if state.regenerate_attempts < state.max_regenerate_attempts:
        return "regenerate"
    return "refuse"


# =============================================================================
# Graph construction
# =============================================================================


_COMPILED_GRAPH: CompiledStateGraph | None = None
# N6 (audit 2026-05-20): gate the lazy compile so concurrent first-call
# threads don't both pay the compile cost (and don't race to write the
# global). `threading.Lock` rather than `asyncio.Lock` because callers
# include sync code paths (test setup, daemon bootstrap) and because
# concurrent threads via `asyncio.to_thread` could otherwise hit a true
# data race on the global. Single event-loop callers are already safe
# under cooperative scheduling — the lock is uncontended in that case.
_COMPILED_GRAPH_LOCK = threading.Lock()


def get_compiled_graph() -> CompiledStateGraph:
    """Return the singleton compiled answering graph.

    Cached by module state so we pay the compile cost once per process,
    not once per query. Tests that need a fresh graph (e.g. after
    monkey-patching a node) call `reset_compiled_graph()`.

    Thread-safe lazy init via `_COMPILED_GRAPH_LOCK`: double-checked
    locking pattern — the fast path (cache hit) skips the lock; the
    slow path (first call) takes the lock and re-checks before
    building, so a concurrent compiler doesn't waste work.
    """
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is not None:
        return _COMPILED_GRAPH
    with _COMPILED_GRAPH_LOCK:
        # Re-check under the lock — another thread may have compiled
        # while we were waiting.
        if _COMPILED_GRAPH is None:
            _COMPILED_GRAPH = build_answering_graph()
        return _COMPILED_GRAPH


def reset_compiled_graph() -> None:
    """Drop the cached compiled graph. For tests."""
    global _COMPILED_GRAPH
    with _COMPILED_GRAPH_LOCK:
        _COMPILED_GRAPH = None


def build_answering_graph() -> CompiledStateGraph:
    """Construct and compile the answering state machine.

    Topology is static, so one compile per process is correct; the
    caching is in `get_compiled_graph()` so tests can opt out via
    `reset_compiled_graph()`.

        START
          |
          v
      retrieve --> expand_graph --> rerank --> assess
                                                 |
                                       +---------+---------+
                                       |                   |
                                    answer              refuse
                                       |                   |
                                       v                   v
                                    verify              END
                                       |
                             +---------+---------+
                             |         |         |
                         compose   regenerate  refuse
                             |         |         |
                             v         v         v
                           END     (back to    END
                                    answer)

    `expand_graph` augments the candidate pool with chunks from
    documents the graph store says are related (one-hop CITES /
    shared-entity neighbours). It is a no-op when the graph store
    isn't installed, when expansion is disabled on AnswerState, or
    when retrieval returned no candidates.
    """
    g = StateGraph(AnswerState)

    # Nodes
    g.add_node("retrieve", retrieve)
    g.add_node("expand_graph", expand_graph)
    g.add_node("rerank", rerank)
    g.add_node("assess", assess)
    g.add_node("answer", answer)
    g.add_node("verify", verify)
    g.add_node("regenerate", regenerate)
    g.add_node("refuse", refuse)
    g.add_node("compose", compose)

    # Linear edges
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "expand_graph")
    g.add_edge("expand_graph", "rerank")
    g.add_edge("rerank", "assess")
    g.add_edge("answer", "verify")
    g.add_edge("regenerate", "answer")  # the loop
    g.add_edge("compose", END)
    g.add_edge("refuse", END)

    # Conditional edges
    g.add_conditional_edges(
        "assess",
        route_after_assess,
        {"answer": "answer", "refuse": "refuse"},
    )
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "compose": "compose",
            "regenerate": "regenerate",
            "refuse": "refuse",
        },
    )

    return g.compile()


# =============================================================================
# Public entry point
# =============================================================================


async def answer_query(
    query: str,
    *,
    token_budget: int = 8000,
    max_regenerate_attempts: int = 2,
    graph_expansion_enabled: bool = True,
    graph_expansion_budget: int = 3,
    chunks_per_neighbor: int = 2,
) -> FinalResponse:
    """Run the answering graph for a single query.

    This is the function the CLI, MCP server, and web UI all call.
    Everything else is implementation detail.

    Observability contract (ADR-0004):
      - `correlation_id` (ULID) is bound into structlog ContextVars so
        every nested log line in this run carries it without manual
        propagation.
      - A Langfuse `CallbackHandler` is attached to the graph
        invocation so every node, edge, and model call appears under
        one trace keyed by the same `correlation_id`. Logs and traces
        are joined by that ID.

    Graph-expansion knobs (v1.x): `graph_expansion_enabled` toggles the
    one-hop citation-graph augmentation between retrieval and rerank;
    `graph_expansion_budget` is the number of source documents whose
    neighbours are explored; `chunks_per_neighbor` is the max chunks
    pulled from each neighbour. See `expand_graph` for the design.
    """
    initial = AnswerState(
        query=query,
        token_budget=token_budget,
        max_regenerate_attempts=max_regenerate_attempts,
        graph_expansion_enabled=graph_expansion_enabled,
        graph_expansion_budget=graph_expansion_budget,
        chunks_per_neighbor=chunks_per_neighbor,
    )

    clear_run_context()  # belt-and-suspenders: a missed clear elsewhere
    bind_run_context(initial.correlation_id, query_preview=query[:80])
    try:
        graph = get_compiled_graph()
        final_state = await graph.ainvoke(
            initial,
            config={
                "callbacks": [callback_handler()],
                "metadata": run_attributes(
                    initial.correlation_id, "answer_query"
                ),
            },
        )

        # LangGraph returns the final state as a dict-like with all fields.
        response = final_state.get("final")
        if response is None:
            raise AnswerStateInvariantError(
                "Answering graph terminated without producing a FinalResponse",
                context={"correlation_id": initial.correlation_id},
            )
        return response
    finally:
        clear_run_context()
