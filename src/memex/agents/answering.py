# pyright: reportConstantRedefinition=false
# `_COMPILED_GRAPH` is an uppercase module-level singleton intentionally
# rebound by the lazy-init helpers (see `get_compiled_graph` /
# `reset_compiled_graph`). Pyright's "uppercase = literal constant"
# heuristic doesn't fit the cache-singleton pattern.

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

import math
import re
import threading
from collections.abc import Callable
from typing import Annotated, Literal, TypedDict

import structlog
import ulid
from langchain_core.callbacks.base import BaseCallbackHandler
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, create_model

from memex.agents.artifact_scope import (
    ArtifactReference,
    ResolvedArtifactScope,
    detect_artifact_reference,
    resolve_scope,
)
from memex.agents.table_sql import coerce_number, describe_aggregate
from memex.core.errors import AnswerStateInvariantError
from memex.core.text import STOPWORDS, atomise, claim_grounded_only_by_name
from memex.core.types import Chunk, RelatedDocument, StoredTable, TableQueryResult
from memex.core.wikilinks import format_wikilink
from memex.models.client import complete_structured
from memex.observability.tracing import (
    bind_run_context,
    callback_handler,
    clear_run_context,
    run_attributes,
)
from memex.prompts import render_messages, render_prompt
from memex.retrieve import (
    cross_encoder_rerank,
    hybrid_search,
    hybrid_search_in_docs,
)

logger = structlog.get_logger(__name__)


# The answering graph's top-level node names — the canonical "steps" a progress
# UI surfaces. Used by `_NodeProgressHandler` to filter the callback firehose to
# real node starts; the webui maps each to a human phase label.
_PROGRESS_NODES = frozenset(
    {
        "retrieve",
        "resolve_artifact_scope",
        "expand_graph",
        "rerank",
        "query_tables",
        "assess",
        "answer",
        "regenerate",
        "verify",
        "assess_relevance",
        "compose",
        "refuse",
    }
)


class _NodeProgressHandler(BaseCallbackHandler):
    """Observe-only LangGraph callback that reports each top-level node as it
    STARTS, for the webui's live progress indicator (`answer_query(on_node=…)`).

    It is appended NEXT TO the Langfuse handler and never reads or mutates the
    graph state, so it cannot affect routing or the answering HARD gates — the
    worst a misbehaving sink can do is be swallowed (see below).

    Filtering (verified on langgraph 1.2.0 / langchain-core 1.4.0): a top-level
    graph node's `on_chain_start` carries `metadata["langgraph_node"]`, a
    `graph:step:N` tag, and `kwargs["name"] == langgraph_node`. Nested
    routers/runnables inside a node carry `seq:step:` tags or a mismatched
    `name`, and the root `LangGraph` event has no node metadata — all excluded.
    """

    def __init__(self, on_node: Callable[[str], None]) -> None:
        self._on_node = on_node

    def on_chain_start(
        self,
        serialized: object,
        inputs: object,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        # `serialized`/`inputs` are part of the base signature, unused here; the
        # node name + the discriminating tag/name come via metadata/tags/kwargs.
        node = (metadata or {}).get("langgraph_node")
        if not isinstance(node, str) or node not in _PROGRESS_NODES:
            return
        if not any(t.startswith("graph:step:") for t in (tags or [])):
            return  # a nested runnable inside the node (seq:step:…), not the node start
        if kwargs.get("name") != node:
            return
        try:
            self._on_node(node)
        except Exception:
            # reason: a progress-UI sink must NEVER abort an answering run.
            logger.warning("progress.on_node_failed", node=node)


# Node return contract. `total=False` because each node updates only
# the fields it touches; typos in keys fail at type-check time instead
# of becoming silent state drops.
class AnswerStateUpdate(TypedDict, total=False):
    """The shape every answering-graph node returns: a partial state
    update. `total=False` means each node only declares the fields it
    actually touches; LangGraph merges them onto the running state."""

    candidates: list[Chunk]
    reranked: list[Chunk]
    graph_expanded_doc_ids: list[str]
    artifact_scope_doc_ids: list[str]
    sufficiency: SufficiencyAssessment
    draft: DraftAnswer
    verification: VerificationResult
    relevance: RelevanceAssessment
    regenerate_attempts: int
    nodes_traversed: int
    tokens_used: int
    final: FinalResponse


# Keyword heuristic for the `query_tables` gate (Table-RAG Phase 2). Cheap
# precondition deciding whether to ATTEMPT a SQL query — the §4 injection gate
# (recompute / verbatim-rows) is the real safety boundary, so a broad gate is
# fine: an attempt that can't be verified simply no-ops. Aggregation /
# superlative / filtered-set shapes are what row-lookup retrieval misses.
_TABLE_QUERY_KEYWORDS: tuple[str, ...] = (
    "total",
    "sum",
    "average",
    "avg",
    "mean",
    "count",
    "how many",
    "number of",
    "highest",
    "lowest",
    "largest",
    "smallest",
    "biggest",
    "maximum",
    "minimum",
    "max ",
    "min ",
    "most",
    "least",
    "greater than",
    "less than",
    "more than",
    "fewer than",
    "at least",
    "at most",
    "combined",
    "across all",
    "which segment",
    "which had",
    "rank",
    "top ",
)


def _query_is_table_shaped(query: str) -> bool:
    """Cheap keyword test: does the query look aggregation/superlative/filter
    shaped (the shapes row-lookup retrieval misses)?"""
    q = query.lower()
    return any(kw in q for kw in _TABLE_QUERY_KEYWORDS)


# =============================================================================
# Types — every cross-node payload is one of these.
# =============================================================================


class CitedClaim(BaseModel):
    """A single factual claim with the chunk that supports it."""

    # Path C tightening (2026-05-22): bounds reduced from earlier
    # hardening pass so xgrammar-forced JSON-close at max_length
    # leaves enough budget under `max_tokens` to actually close the
    # object cleanly. Math: a `claims` array at max_items=8 with
    # each claim at max_length=300 + chunk_id at max_length=80 +
    # ~52-char JSON overhead = ~3,456 chars (≈865 tokens) for the
    # array alone. Plus summary at max_length=300 = ~75 tokens.
    # Comfortably under max_tokens=1024.
    claim: str = Field(
        description="One factual statement, self-contained.",
        max_length=300,
    )
    source_chunk_id: str = Field(
        description="The chunk_id that supports the claim.",
        max_length=80,
    )
    confidence: Literal["high", "medium", "low"]


class DraftAnswer(BaseModel):
    """The model's structured draft. Verified before it reaches the user."""

    # The summary is the headline ANSWER the user reads (`.ans-answer`). The
    # cap is a SAFETY ceiling, not the target length — the prompt still asks for
    # "one or two sentences". 300 was too tight once `full` mode's deeper retrieval
    # (top-k 18) gives the model enough context to write a thorough 3-clause answer:
    # it hit exactly 300 and xgrammar force-closed the JSON string MID-WORD
    # ("…and policyEn"). 600 lets a thorough headline finish cleanly while the cap
    # still guards against a runaway; the answer node raises `max_tokens` to match
    # so xgrammar can always close (600 summary + 8×~435-char claims ≈ 1.1k tokens).
    summary: str = Field(
        description="One or two sentences overviewing the answer.",
        max_length=600,
    )
    claims: list[CitedClaim] = Field(
        description="Detailed claims, each with a citation.",
        max_length=8,
    )


class SectionSummary(BaseModel):
    """A grounded digest of one heading-section — the MAP step's output and an
    element of a document summary (ADR-0008). `key_points` are bounded
    `CitedClaim`s grounded against the section's own chunks; `digest` is a
    faithful synthesis (the no-hallucination gate is on the cited points)."""

    section_title: str = Field(
        description="The section's heading (or the document title for a flat doc).",
        max_length=200,
    )
    digest: str = Field(
        description="A faithful 1-3 sentence synthesis of the section.",
        max_length=600,
    )
    key_points: list[CitedClaim] = Field(
        description="The section's key points, each cited to a chunk in this section.",
        max_length=8,
    )


class DocAbstract(BaseModel):
    """The REDUCE step's output: a whole-document overview synthesized from the
    (already grounded) section digests (ADR-0008).

    The overview is emitted as a bounded LIST of complete sentences, not one
    free-form string — the same shape the rest of the summarizer uses (ADR-0009).
    `maxItems` bounds the COUNT and each sentence is short enough to end at a
    natural boundary, so the joined `abstract` can never force-close MID-WORD the
    way a single capped string does (the "policyEn" failure). The `abstract`
    property joins them, so `_reduce` / `FinalResponse.summary` / the webui are
    unchanged."""

    sentences: list[Annotated[str, Field(max_length=300)]] = Field(
        description="The whole-document overview as 2-8 complete sentences.",
        max_length=8,
    )

    @property
    def abstract(self) -> str:
        """The overview as joined prose (what callers read)."""
        return " ".join(s.strip() for s in self.sentences if s.strip())


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


class RelevanceAssessment(BaseModel):
    """Does the VERIFIED draft actually RESPOND to the specific question?

    Grounding (the verify pass) confirms each claim is supported by its
    cited chunk. This is the orthogonal check: a claim can be perfectly
    grounded yet answer a RELATED-BUT-DIFFERENT question than the one
    asked. The motivating case (slide-decks-30): the deck's "C++ and
    Python abstractions for tensor cores" content grounds a claim about
    library usage, which the agent passed off as the answer to "which
    language writes CUDA *kernels*" — grounded, but non-responsive. This
    gate runs after verify, before compose, and routes a non-responsive
    answer to `refuse`. It is deliberately conservative (default
    responsive) so it removes only clear question/answer topic mismatches,
    not legitimate answers that need light reading.
    """

    responsive: bool
    reason: str = Field(
        description="One sentence: does the answer address the specific question, or a related-but-different one?",
        max_length=500,
    )


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
        """True when no claim was rejected by the verifier — answer can
        ship without rewrites."""
        return not self.ungrounded


class ReportStructure(BaseModel):
    """Adaptive paragraph structure for a `report` summary (ADR-0010): the 0-based
    section indices (in reading order) at which a NEW paragraph begins. Lets the model
    COMBINE related adjacent sections and choose logical breaks instead of a fixed
    every-N split. Boundary-selection (NOT free index assignment) cannot drop or
    duplicate a section; the result is validated + size-capped, with a deterministic
    fallback to mechanical batching. Grouping ONLY — never grounding."""

    paragraph_starts: list[int] = Field(
        max_length=16,
        description=(
            "0-based section indices where a new paragraph begins (index 0 starts the "
            "first paragraph). Group adjacent sections that share a theme together."
        ),
    )


class ReportConfidence(BaseModel):
    """Faithfulness confidence of a `report` summary against its source digests
    (ADR-0010) — INFORM-ONLY, never the HARD gate. A hybrid signal: semantic embedding
    cosine + lexical content-overlap, scored per paragraph against the grounded digests
    that paragraph was built from, then aggregated. Higher = the generated prose tracks
    the source more closely; a low score flags a paragraph worth a second look, not a
    fabrication (the `must_not_assert` eval remains the no-hallucination gate)."""

    overall: float = Field(description="Aggregate confidence in [0,1] (mean of per_paragraph).")
    embedding: float = Field(description="Mean semantic cosine of each paragraph vs its digests.")
    lexical: float = Field(description="Mean lexical content-overlap of each paragraph vs its digests.")
    per_paragraph: list[float] = Field(
        description="Per-paragraph combined confidence, in paragraph order."
    )


class FinalResponse(BaseModel):
    """What gets returned to the caller (CLI, web UI, MCP server)."""

    answered: bool
    summary: str | None = None
    claims: list[CitedClaim] = []
    refusal_reason: str | None = None
    used_chunks: list[Chunk] = []
    # Deterministic, grounded `[[doc_id#section]]` wikilinks for the
    # chunks this answer cited — derived in `compose` from the GROUNDED
    # used_chunks (NOT LLM-emitted). Empty on a refusal (a refusal cited
    # nothing). Surfaces a navigable "Sources" list to MCP/webui/CLI.
    wikilinks: list[str] = []

    # Per-section grounded digests for a document SUMMARY (ADR-0008,
    # `agents/document_summarizer.py`). Empty `[]` on the answer path (the
    # common case) — only `summarize_document` populates it. HARD-gate-neutral:
    # each section's `key_points` are grounded `CitedClaim`s, surfaced for the
    # webui's collapsible per-section view.
    sections: list[SectionSummary] = []

    # Faithfulness confidence for a `report`-detail summary (ADR-0010) — the hybrid
    # embedding+lexical alignment of each generated paragraph against the grounded
    # digests it was built from. INFORM-ONLY (surfaced + logged, never gates); None
    # on every other path (answers + non-report summaries).
    report_confidence: ReportConfidence | None = None

    # Documents retrieval was deterministically RE-SCOPED to because the query
    # named a specific artifact (#256). Empty = the full-corpus path (the common
    # case). Surfaced so the re-scope is auditable: on a refusal it explains WHY
    # the pool was narrowed (e.g. "scoped to the firewall doc, which has no VLAN
    # range" → refuse). HARD-gate-neutral — derived from state, never alters
    # answered/claims/refusal. (ADR-0004: observable at every layer.)
    artifact_scope_doc_ids: list[str] = []

    # "Explore connections" discovery: documents the entity graph relates to the docs THIS
    # answer cited. The AGENT leaves this empty `[]` — populating it would add a graph open to
    # every `answer_query` (incl. the eval path). The SURFACES (MCP/CLI `ask`, webui /ask panel)
    # enrich it post-hoc from `used_chunks` via `retrieve.related_documents_for_answer`, a
    # read-only graph query. HARD-gate-neutral by construction (never alters
    # answered/claims/refusal). Empty on a refusal (cited nothing) or when no graph is present.
    related_documents: list[RelatedDocument] = []

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
    # Documents the candidate pool was RE-SCOPED to by `resolve_artifact_scope`.
    # Two sources: (#256) the query NAMED a specific artifact → inferred scope, OR
    # the user EXPLICITLY selected docs (the doc-picker → `scope_doc_ids` below).
    # Empty = the full-corpus path (the common case). Surfaced on `FinalResponse`
    # for the trace + the webui "Scoped to …" note.
    artifact_scope_doc_ids: list[str] = []
    # EXPLICIT user-selected document scope (the Notebook-LM-style doc-picker).
    # When non-empty, `resolve_artifact_scope` scopes retrieval to exactly these
    # docs — no inference, no qualifier/single-token gates — and it TAKES
    # PRECEDENCE over an inferred artifact reference (the user said which docs).
    # Empty = fall back to artifact inference / the full-corpus path. Input only.
    scope_doc_ids: list[str] = []

    # The grounded multi-turn chat's bounded prior-chunk carry (Surface A,
    # docs/specs/grounded-agentic-chat.md): chunk_ids from the immediately-prior
    # answered turn, re-admitted in `retrieve` as extra rerank CANDIDATES so a tight
    # follow-up keeps its referents. Input only; default empty → byte-identical to a
    # bare `/ask`. A carried chunk only reaches grounding if it SURVIVES rerank against
    # the (rewritten) query — `verify` is untouched, so grounding stays safe.
    prior_carry_chunk_ids: list[str] = []

    # --- Graph-expansion budgets ---
    # Toggle the whole expansion step off (e.g. when the graph store
    # isn't available or for benchmarking pure flat retrieval).
    graph_expansion_enabled: bool = True
    # How many distinct source documents from the initial retrieval
    # to seed graph traversal from. Smaller = less noise + fewer
    # graph queries; larger = wider reach.
    graph_expansion_budget: int = 3
    # How many neighbour DOCUMENTS to pull per seed (the graph query's
    # LIMIT). Previously `graph_expansion_budget` was overloaded as BOTH
    # the seed count AND this per-seed neighbour cap — two unrelated
    # quantities, so a densely-connected hub silently truncated to the
    # seed count. Split out so each is tunable independently; the default
    # (3) preserves the prior effective behaviour exactly.
    neighbors_per_doc: int = 3
    # Chunks pulled from each neighbour document via filtered hybrid
    # search. The reranker is the quality gate downstream.
    chunks_per_neighbor: int = 2

    # --- Reasoning intermediates ---
    sufficiency: SufficiencyAssessment | None = None
    draft: DraftAnswer | None = None
    verification: VerificationResult | None = None
    relevance: RelevanceAssessment | None = None

    # --- Budgets (the difference between an agent and a runaway loop) ---
    regenerate_attempts: int = 0
    max_regenerate_attempts: int = 2

    # When True, a draft with SOME grounded + some ungrounded claims, once
    # regeneration is exhausted, ships the grounded SUBSET (relevance-gated;
    # ungrounded claims dropped) rather than refusing the whole answer — fixes
    # the compound-question over-refusal. Set from `agents.partial_grounded_answers`
    # in `answer_query` (fail-open True). A zero-grounded verdict always refuses.
    allow_partial_grounded: bool = True

    # The deterministic numeric-grounding backstop (2026-05-31). When True, the
    # verify node demotes a grounded claim whose principal LARGE figure is absent
    # from its cited TABLE chunk (a computed aggregate the LLM verifier accepts
    # via the literal-table-row loophole). Set from
    # `agents.numeric_grounding_backstop_enabled` in `answer_query` (fail-open).
    numeric_grounding_backstop: bool = True

    # The deterministic NAME-ONLY grounding backstop (2026-06-03). When True, the verify node
    # demotes a grounded BEHAVIORAL/property/comparative claim whose cited chunk merely NAMES the
    # subject (a bare list/heading; `core/text.is_name_only_chunk` + `claim_asserts_behavior`) —
    # the entity-name-presence loophole. Fail-open + demotion-only (membership/unknown claims kept).
    # Set from `agents.name_only_grounding_backstop_enabled` in `answer_query` (fail-open).
    name_only_grounding_backstop: bool = True

    nodes_traversed: int = 0
    max_nodes_traversed: int = 20

    tokens_used: int = 0
    token_budget: int = 8000

    # --- Output ---
    final: FinalResponse | None = None

    def over_budget(self) -> bool:
        """True when the agent has exhausted either the step budget
        (`max_nodes_traversed`) or the token budget (`token_budget`).
        Routing predicates short-circuit to `refuse` when this fires."""
        return (
            self.nodes_traversed >= self.max_nodes_traversed
            or self.tokens_used >= self.token_budget
        )


# =============================================================================
# Nodes — each is async, takes state, returns a dict of state updates.
# =============================================================================


async def retrieve(state: AnswerState) -> AnswerStateUpdate:
    """Hybrid search: BM25 (SQLite FTS5) + dense (LanceDB) + RRF fusion.

    When the grounded multi-turn chat supplies a bounded prior-chunk carry
    (`state.prior_carry_chunk_ids`), those chunks are fetched verbatim from the FTS
    store and unioned into the candidate pool — the SAME candidate-merge seam
    `expand_graph` uses (the documented `agents/ → index/` lazy-store-open edge). They
    must still SURVIVE rerank against the (rewritten) query to reach grounding, so a
    stale referent is reranked out and `verify` is unchanged. Empty carry → byte-identical.
    """
    log = logger.bind(node="retrieve")
    log.info("start", query_len=len(state.query))
    candidates = await hybrid_search(state.query, k=50)

    if state.prior_carry_chunk_ids:
        from memex.core.config import get_settings
        from memex.index.fts_store import FTSStore

        existing = {c.chunk_id for c in candidates}
        carry_ids = [cid for cid in state.prior_carry_chunk_ids if cid not in existing]
        if carry_ids:
            store = await FTSStore.open(get_settings().vault_path)
            try:
                carried = await store.chunks_by_ids(carry_ids)
            finally:
                await store.close()
            candidates = candidates + carried
            log.info("carry_merged", carried=len(carried))

    log.info("done", candidate_count=len(candidates))
    return {
        "candidates": candidates,
        "nodes_traversed": state.nodes_traversed + 1,
    }


async def resolve_artifact_scope(state: AnswerState) -> AnswerStateUpdate:
    """Deterministic artifact→doc re-scope (#256).

    If the query NAMES a specific artifact ("the firewall diagram", "le diagramme
    de coupe-feu"), resolve it to the document(s) it lives in and REPLACE the
    candidate pool with chunks scoped there — the named artifact acts as an
    automatic doc-selection. The downstream gates then answer from the right
    source or refuse naturally.

    Conservative + fail-open by construction: no named artifact, no confident
    resolution, a disabled flag, or any resolver/store error all leave
    `state.candidates` untouched (the full-corpus path). The node can therefore
    only NARROW retrieval for clearly artifact-named queries; it never turns an
    answerable open/multi-file query into a refusal, never adds a chunk, and
    never relaxes a gate. Worst case is a conservative false-refuse on a
    mis-resolved artifact query — never a hallucination, never a wrongly-answered
    counterfactual.

    Runs between `retrieve` and `expand_graph`; `expand_graph` then short-circuits
    while a scope is active (it would otherwise re-admit other documents' chunks
    and defeat the determinism guarantee).
    """
    import sqlite3

    from memex.core.config import get_settings
    from memex.core.errors import ConfigurationError, MemexError

    log = logger.bind(node="resolve_artifact_scope")
    bump = state.nodes_traversed + 1

    # EXPLICIT user doc-selection (the doc-picker) WINS over inference: scope
    # retrieval to exactly the selected docs (dedup, preserve order, drop blanks).
    # No detection, no gates — the user said which docs. Worst case is an empty
    # scoped pool (all-bogus ids, or selected docs with no chunk matching the
    # query) → the downstream gates refuse cleanly: HARD-gate-safe by construction
    # (only narrows; never adds a chunk, never relaxes a gate), exactly like #256.
    user_scope = [d for d in dict.fromkeys(state.scope_doc_ids) if d.strip()]
    if user_scope:
        scoped = await hybrid_search_in_docs(state.query, user_scope, k=50)
        log.info(
            "scoped",
            via="user-selected",
            doc_ids=user_scope,
            scoped_count=len(scoped),
            replaced_from=len(state.candidates),
        )
        return {
            "candidates": scoped,
            "artifact_scope_doc_ids": user_scope,
            "nodes_traversed": bump,
        }

    # Detection is pure + cheap and needs no settings — run it FIRST so the
    # common no-artifact query (the overwhelming majority) takes a path
    # byte-identical to the pre-#256 pipeline, never even reading settings.
    ref = detect_artifact_reference(state.query)
    if ref is None:
        return {"nodes_traversed": bump}

    try:
        if not get_settings().agents.artifact_scope_enabled:
            return {"nodes_traversed": bump}  # kill-switch: full revert
        scope = await _resolve_artifact_scope_via_corpus(ref)
    except (ImportError, ConfigurationError, MemexError, OSError, sqlite3.Error) as exc:
        # FAIL-OPEN: an unconfigured settings read (e.g. a non-bootstrapped
        # fixture) or any resolver/store error must never turn an answerable
        # query into a refusal — fall back to the full-corpus candidate pool.
        log.warning("skipped", reason=type(exc).__name__, error=str(exc))
        return {"nodes_traversed": bump}

    if not scope.doc_ids:
        log.info("no_scope", via=scope.via, arttype=ref.arttype)
        return {"artifact_scope_doc_ids": [], "nodes_traversed": bump}

    doc_ids = list(scope.doc_ids)
    # REPLACE (not top-up): topping up would re-admit the wrong-source chunk and
    # reintroduce the bug. An empty scoped pool is the correct outcome for a named
    # artifact whose doc can't answer — the downstream gates then refuse cleanly.
    scoped = await hybrid_search_in_docs(state.query, doc_ids, k=50)
    log.info(
        "scoped",
        via=scope.via,
        arttype=ref.arttype,
        doc_ids=doc_ids,
        scoped_count=len(scoped),
        replaced_from=len(state.candidates),
    )
    return {
        "candidates": scoped,
        "artifact_scope_doc_ids": doc_ids,
        "nodes_traversed": bump,
    }


async def _resolve_artifact_scope_via_corpus(
    ref: ArtifactReference,
) -> ResolvedArtifactScope:
    """Open the FTS store and resolve `ref` against the whole corpus.

    Lazy-open inside the call (same `agents/ → index/` lazy-store edge as
    `expand_graph`/`query_tables`); the agent owns no threaded store handle.
    `resolve_scope` is pure given the injected per-token search.
    """
    from memex.core.config import get_settings
    from memex.index.fts_store import FTSStore

    store = await FTSStore.open(get_settings().vault_path)
    try:

        async def _search(token: str, k: int) -> list[Chunk]:
            return await store.search(token, k=k)

        return await resolve_scope(ref, _search)
    finally:
        await store.close()


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

    if state.artifact_scope_doc_ids:
        # The query named a specific artifact and retrieval was re-scoped to its
        # doc(s) (#256). Graph expansion pulls in OTHER documents' chunks, which
        # would re-admit the wrong-source evidence the re-scope deliberately
        # removed — defeating the determinism guarantee. The scoped docs are
        # already fully retrieved (hybrid_search_in_docs, k=50), so skip.
        log.info(
            "skip", reason="artifact_scope_active", scope_docs=len(state.artifact_scope_doc_ids)
        )
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
            neighbours = await store.neighbors(doc_id, limit=state.neighbors_per_doc)
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


def _resolve_rerank_top_k(env_value: str | None) -> int:
    """How many reranked chunks the answer prompt grounds against. Resolution:
    an explicit `MEMEX_RERANK_TOP_K` wins (the operator escape hatch; a bad value
    falls back to 5), otherwise the ACTIVE co-residence mode's `retrieval_top_k`
    (ADR-0007) — the concrete way `full` mode leverages its 24,576 window: it
    grounds against ~18 chunks vs `fast`/`manual`'s 5. HARD-gate-neutral: more
    chunks only add candidate evidence; the grounding + refusal gates downstream
    are unchanged, so a deeper pool can never turn a refusal into a hallucination.
    Always ≥ 1. Falls back to 5 when settings aren't initialised (a unit-test
    context with no `set_settings`) — matching the prior env-default behaviour."""
    from memex.core.config import get_settings
    from memex.core.errors import MemexError
    from memex.core.resources import resolve_profile

    if env_value is not None:
        try:
            return max(1, int(env_value))
        except ValueError:
            return 5
    try:
        s = get_settings()
    except MemexError:
        return 5  # settings not initialised (e.g. a unit test) → historical default
    profile = resolve_profile(
        s.models.co_residence_mode,
        embedder_device=s.models.embedder_device,
        reranker_device=s.models.reranker_device,
    )
    return max(1, profile.retrieval_top_k)


async def rerank(state: AnswerState) -> AnswerStateUpdate:
    """Second-stage reranking. Backend selected by settings
    (`reranker_backend`): bge cross-encoder by default, Qwen3-Reranker
    when flagged. See `retrieve/rerank.py`.
    """
    import os

    log = logger.bind(node="rerank")
    log.info("start", candidate_count=len(state.candidates))
    # `top_k` is the count of reranked chunks the answer prompt grounds against.
    # It's resolved from the active co-residence mode (ADR-0007): the `fast`/6,144
    # window holds ~5 truncated chunks (5 × 1800 chars ≈ 2250 tokens + scaffold +
    # output, comfortably under 6,144), while `full`/24,576 grounds against ~18 —
    # the concrete way full mode LEVERAGES its larger window (deeper retrieval =
    # more evidence per answer; the grounding/refusal gate downstream is unchanged,
    # so more chunks can only add evidence, never a hallucination). `manual` keeps
    # the historical default of 5. An explicit `MEMEX_RERANK_TOP_K` overrides the
    # mode (the operator escape hatch — e.g. to recover breadth on a custom window).
    env_top_k = os.environ.get("MEMEX_RERANK_TOP_K")
    top_k = _resolve_rerank_top_k(env_top_k)
    log.info("top_k_resolved", top_k=top_k, source="env" if env_top_k is not None else "mode")
    reranked = await cross_encoder_rerank(state.query, state.candidates, top_k=top_k)
    log.info("done", reranked_count=len(reranked))
    return {
        "reranked": reranked,
        "nodes_traversed": state.nodes_traversed + 1,
    }


def _render_kv_row(header: list[str], cells: list[str]) -> str:
    """Render one table row as a compact markdown-KV line: `h0=c0, h1=c1, …`.

    A local renderer (not the private `table_linearize._linearize_row`) so the
    synthetic-chunk evidence reads like the Phase-1 `[table-rows]` KV the
    answer/verify prompts already understand. A cell with no matching header
    column is keyed by 1-based position; an empty header cell uses `colN`.
    """
    n = max(len(header), len(cells))

    def _hdr(j: int) -> str:
        return header[j] if j < len(header) and header[j] else f"col{j + 1}"

    def _val(j: int) -> str:
        return cells[j] if j < len(cells) else ""

    return ", ".join(f"{_hdr(j)}={_val(j)}" for j in range(n))


# Max chars of the synthetic chunk's evidence text. Bounded < the smallest
# (assess) truncate budget (1200) so the contributing rows always survive
# into the assess/answer/verify prompts (spec §3 / round-1 B1c).
_SYNTHETIC_TEXT_MAX = 900

# Max chars of the source-table caption injected into a row/superlative synthetic
# chunk's framing line (ar-15 fix): a section is unbounded doc text, so cap it to
# keep the framing from eating the mandatory answer row's budget. 120 > any real
# table title.
_SYNTHETIC_TABLE_NAME_MAX = 120


def _build_synthetic_chunk(result: TableQueryResult) -> Chunk:
    """Build the single synthetic `Chunk` injected into `state.reranked` for a
    `TableQueryResult` (Table-RAG Phase 2, spec §3).

    The evidence — source/contributing rows as KV lines — comes FIRST and is
    bounded to `_SYNTHETIC_TEXT_MAX` so it survives the downstream truncate
    budgets; the SQL string is appended LAST (informational, clippable). The
    chunk_id suffix `sql0001` is 7 chars and contains the non-hex letters
    `s/q/l`, so `repair_claim_chunk_ids` can't fuzzy-collide it with a real
    10-hex-char chunk hash.
    """
    header = result.header
    lines: list[str] = []
    if result.kind == "aggregate":
        # Lead with the validated aggregate framing, then the basis rows. The
        # value lives in THIS framing line, so the aggregate path is unaffected
        # by row-dropping — even if every basis row overflows, the answer (the
        # scalar) survives. The framing SELF-DESCRIBES the aggregate (`SUM of
        # Fees Earned or Paid in Cash ($) = 956250`) so the answer node connects
        # the scalar to the queried quantity — a bare "Aggregate result = X"
        # reads as un-labelled and the literal-presence rule refuses it.
        agg_val = result.aggregate_value
        val_str = f"{agg_val:g}" if agg_val is not None else "n/a"
        label = describe_aggregate(result) or "Aggregate result"
        lines.append(f"{label} = {val_str} over {len(result.contributing_rows)} rows:")
        for cells in result.contributing_rows:
            candidate = "\n".join([*lines, _render_kv_row(header, cells)])
            # Stop adding rows once we'd exceed the evidence budget (leaving
            # room for the trailing SQL line). Keep at least the framing line.
            if len(candidate) > _SYNTHETIC_TEXT_MAX:
                break
            lines.append(_render_kv_row(header, cells))
        text = "\n".join(lines)
    else:
        # kind="rows": the VALUE is in the row(s), not the framing — so the row
        # must never be dropped. The superlative framing line (the only claim,
        # "the highest/lowest X") is emitted ONLY when its row is present;
        # otherwise we'd ship an unsupported extremum claim with no value.
        # Always include at least the first contributing row, truncating its KV
        # text to fit the budget if it's pathologically wide, so the value is
        # never lost.
        # Name the SOURCE table by its caption/section so the answer node can
        # map a generic column header ("Total ($)") to the queried quantity
        # ("total compensation"): without it the row reads as an anonymous
        # "Total ($)" the No-substitute rule conservatively refuses (ar-15 —
        # the "Director Compensation for Fiscal 2026" table's Total column IS
        # total compensation, but the answer LLM can't confirm that from a
        # bare column label). The caption is VERBATIM doc text carrying no
        # numeric value ⇒ HARD-gate-safe (it adds context, never a figure).
        # The aggregate framing already self-describes via `describe_aggregate`
        # ("SUM of Fees Earned or Paid in Cash ($)"), so this is row-path only.
        table_name = (result.section or (result.heading_path[-1] if result.heading_path else "") or "").strip().strip("*").strip()
        # Bound the caption like every other LLM-facing string: a section is doc
        # text of unbounded length, and the framing line's budget is reserved
        # up-front (framing_cost below) — an unbounded caption would eat the
        # mandatory answer row's char budget. 120 fits any real table title.
        table_name = table_name[:_SYNTHETIC_TABLE_NAME_MAX]
        table_phrase = f'the "{table_name}" table' if table_name else "this table"

        superlative_framing: str | None = None
        if result.superlative is not None:
            # Verified extremum framing — the returned row was independently
            # confirmed to hold the {highest|lowest} value of this column, so
            # the agent can attribute the superlative (the framing is grounded,
            # not an unchecked claim). Held back until we know the row fits.
            col_label, direction = result.superlative
            superlative_framing = f"Row with the {direction} {col_label} in {table_phrase}:"

        # The framing line that WILL be prepended (superlative if verified, else
        # the neutral "Matching rows:"). Reserve room for it + its newline in
        # every row-budget check so the assembled `framing\nrows` total can't
        # exceed _SYNTHETIC_TEXT_MAX. The superlative framing is only used when
        # at least the first row survives (so an extremum claim always has a
        # value beneath it).
        neutral_framing = f"Matching rows in {table_phrase}:"
        framing_for_budget = superlative_framing if superlative_framing else neutral_framing
        framing_cost = len(framing_for_budget) + 1  # + the joining newline

        row_lines: list[str] = []
        for idx, cells in enumerate(result.contributing_rows):
            kv = _render_kv_row(header, cells)
            if idx == 0:
                # First row is mandatory — it carries the answer. Truncate its
                # KV to whatever the budget allows after reserving the framing.
                budget = _SYNTHETIC_TEXT_MAX - framing_cost
                if len(kv) > budget:
                    kv = kv[: max(0, budget)]
                row_lines.append(kv)
                continue
            candidate = "\n".join([*row_lines, kv])
            if len(candidate) + framing_cost > _SYNTHETIC_TEXT_MAX:
                break
            row_lines.append(kv)

        if row_lines and superlative_framing is not None:
            lines.append(superlative_framing)
        else:
            # No rows survived (defensive) OR no verified superlative → neutral
            # framing with no unsupported extremum claim.
            lines.append(neutral_framing)
        lines.extend(row_lines)
        text = "\n".join(lines)
    # Append as much of the trailing `[sql]` line as fits within the TOTAL
    # _SYNTHETIC_TEXT_MAX budget — truncate the SQL tail, never the evidence,
    # so the whole chunk stays under the smallest (1200) assess truncate and
    # the contributing rows always survive (spec §3/§4).
    sql_line = f"\n[sql] {result.sql}"
    remaining = _SYNTHETIC_TEXT_MAX - len(text)
    if remaining > 0:
        text = text + sql_line[:remaining]
    return Chunk(
        chunk_id=f"{result.doc_id}#sql0001",
        document_id=result.doc_id,
        document_title=result.document_title,
        text=text,
        char_start=result.char_start,
        char_end=result.char_end,
        score=0.0,
        rerank_score=None,
        heading_path=result.heading_path,
    )


# Phase-2 table-SQL prompt-size guard. A real filing (the NVIDIA 10-K) has ~74
# stored tables; rendering every one's schema into the generate_table_sql prompt
# overflows the 6144-token context. The retrieval already surfaced the relevant
# region, so we pass only the tables whose char-span is at/near a reranked chunk
# of the target doc, capped — turning "all the doc's tables" into "the handful
# the query actually retrieved". (GPU-acceptance finding 2026-05-24.)
_TABLE_PROXIMITY_MARGIN = 2000  # chars; ~one chunk of slop, catches the adjacent [table-rows] chunk
_TABLE_CANDIDATE_CAP = 6


def _span_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    """Char gap between spans [a0,a1) and [b0,b1): 0 if they overlap/touch."""
    if a1 >= b0 and b1 >= a0:
        return 0
    return b0 - a1 if b0 > a1 else a0 - b1


def _relevant_tables(
    tables: list[StoredTable], reranked: list[Chunk], doc_id: str
) -> list[StoredTable]:
    """Tables of *doc_id* whose span is within `_TABLE_PROXIMITY_MARGIN` of a
    reranked chunk of the same doc, nearest first, capped. Empty when the
    retrieved chunks aren't near any table (→ the node no-ops)."""
    spans = [
        (c.char_start, c.char_end)
        for c in reranked
        if c.document_id == doc_id and c.char_end > c.char_start
    ]
    if not spans:
        return []
    scored = sorted(
        (
            (min(_span_gap(t.char_start, t.char_end, s0, s1) for s0, s1 in spans), i, t)
            for i, t in enumerate(tables)
        ),
        key=lambda x: (x[0], x[1]),
    )
    return [t for gap, _, t in scored if gap <= _TABLE_PROXIMITY_MARGIN][:_TABLE_CANDIDATE_CAP]


# Decouple Table-RAG recall from chunk-proximity. `_relevant_tables` only offers
# the SQL generator tables whose char-span sits near a RERANKED chunk — so a
# clearly-named table whose prose chunk didn't rerank into the pool is never
# queried (the 10-K's "Director Compensation" fees table for a "total director
# fees" query: its chunk ranks below the exec-comp sections, leaving the table
# ~60k chars from any reranked chunk, so the SQL-gen LLM gets the wrong tables
# and returns empty SQL → the query false-refuses despite a perfectly clean,
# summable Fees column). The query-relevance path below ALSO offers a table
# whose section heading + header columns lexically match the query. Lexical
# (the shared `core/text` atomise + STOPWORDS), no LLM, no embedding. Additive
# to recall: the §4 recompute / row-verbatim injection gate in `query_doc_tables`
# stays the safety boundary, so an extra (even off-topic) candidate table can
# only let the model find the right one — it can never ship an unverified value.
_TABLE_QUERY_MATCH_MIN = 2  # significant query terms a table's section+header must share


def _significant_terms(text: str) -> set[str]:
    """Lowercased, stopword-filtered atomic tokens of a phrase (single-char
    tokens like the `s` of `NVIDIA's` dropped) — the same `core/text` tokenizer
    artifact_scope/FTS use, so terms match the same way."""
    return {a for raw in text.split() for a in atomise(raw) if len(a) > 1 and a not in STOPWORDS}


def _query_matched_tables(tables: list[StoredTable], query: str) -> list[StoredTable]:
    """Tables whose section heading + header columns share at least
    `_TABLE_QUERY_MATCH_MIN` significant terms with the query, ranked by overlap
    (desc, ties by document order), capped. The query-relevance recall path that
    doesn't depend on the table's prose chunk reranking."""
    q_terms = _significant_terms(query)
    if not q_terms:
        return []
    scored: list[tuple[int, int, StoredTable]] = []
    for i, t in enumerate(tables):
        sig_terms = _significant_terms(t.section + " " + " ".join(t.header))
        overlap = len(q_terms & sig_terms)
        if overlap >= _TABLE_QUERY_MATCH_MIN:
            scored.append((overlap, i, t))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [t for _, _, t in scored][:_TABLE_CANDIDATE_CAP]


def _select_tables(
    tables: list[StoredTable], reranked: list[Chunk], query: str, doc_id: str
) -> list[StoredTable]:
    """The tables of *doc_id* to offer the SQL generator: those near a reranked
    chunk (spatial proximity, `_relevant_tables`) UNION those whose section/header
    matches the query (`_query_matched_tables`) — query-matched first (more
    on-topic for a table question), deduped by `table_id`, capped at
    `_TABLE_CANDIDATE_CAP`. When proximity already finds the right table the two
    overlap and dedup to the same set (no change); the query-match path only adds
    recall when retrieval didn't rerank the table's chunk."""
    matched = _query_matched_tables(tables, query)
    proximity = _relevant_tables(tables, reranked, doc_id)
    out: list[StoredTable] = []
    seen: set[str] = set()
    for t in [*matched, *proximity]:
        if t.table_id not in seen:
            seen.add(t.table_id)
            out.append(t)
    return out[:_TABLE_CANDIDATE_CAP]


async def query_tables(state: AnswerState) -> AnswerStateUpdate:
    """Text-to-SQL over a relevant document's structured tables (Phase 2).

    Runs between `rerank` and `assess`. Gate: a reranked chunk is a
    `[table-rows]` chunk OR comes from a doc that has stored tables, AND the
    query is aggregation/superlative/filter shaped. On a gated-in query it
    asks `query_doc_tables` for a SQL answer over the most-relevant doc's
    tables and, on a result, injects ONE synthetic Chunk into `state.reranked`
    (returning the FULL augmented list so the plain `reranked` field is
    replaced, not wiped). Every failure path no-ops — the §4 injection gate
    inside `query_doc_tables` is the safety boundary, so a broad attempt that
    can't be verified simply adds nothing and the query proceeds normally.
    """
    import json
    import sqlite3

    from pydantic import ValidationError

    from memex.agents.table_sql import query_doc_tables
    from memex.core.config import get_settings
    from memex.core.errors import MemexError
    from memex.index.table_store import TableStore

    log = logger.bind(node="query_tables")

    if not state.reranked or not _query_is_table_shaped(state.query):
        log.info("skip", reason="empty_or_not_table_shaped")
        return {"nodes_traversed": state.nodes_traversed + 1}

    # The whole table-query attempt is best-effort: Table-RAG is strictly
    # additive, so any failure (a `ModelCallError` from `query_doc_tables`'
    # `complete_structured`, a `sqlite3.Error`/JSON/pydantic error reading the
    # store) must no-op and let the query proceed on the normal reranked set —
    # the agent then answers or refuses as usual. We catch the EXPECTED failure
    # modes (matching `query_doc_tables`'s own safety boundary) and re-raise
    # cancellation; programming bugs propagate. Mirrors `expand_graph`.
    try:
        vault_path = get_settings().vault_path
        store = await TableStore.open(vault_path)
        try:
            # Gate (a): which reranked docs have stored tables? Iterate reranked
            # docs in order; the first doc with tables is the query target.
            seen_docs: list[str] = []
            for c in state.reranked:
                if c.document_id not in seen_docs:
                    seen_docs.append(c.document_id)

            # Pick the first reranked doc whose stored tables are relevant to the
            # query — either NEAR the retrieved chunks (proximity) OR matching the
            # query's section/header terms (query-relevance). Both paths are capped
            # so passing all of a 74-table doc can't overflow the SQL-gen prompt. A
            # doc with tables neither near the retrieved region nor query-matching
            # is skipped, as is one with no stored tables.
            target_tables: list[StoredTable] = []
            target_title = ""
            any_doc_has_tables = False
            for doc_id in seen_docs:
                tables = await store.tables_for_document(doc_id)
                if not tables:
                    continue
                any_doc_has_tables = True
                near = _select_tables(tables, state.reranked, state.query, doc_id)
                if near:
                    target_tables = near
                    target_title = next(
                        (c.document_title for c in state.reranked if c.document_id == doc_id),
                        "",
                    )
                    break

            if not target_tables:
                log.info(
                    "skip",
                    reason="no_relevant_tables" if any_doc_has_tables else "no_stored_tables",
                )
                return {"nodes_traversed": state.nodes_traversed + 1}

            result = await query_doc_tables(state.query, target_tables)
        finally:
            await store.close()

        if result is None:
            log.info("no_table_answer")
            return {"nodes_traversed": state.nodes_traversed + 1}

        synthetic = _build_synthetic_chunk(
            result.model_copy(update={"document_title": result.document_title or target_title})
        )
    except (MemexError, OSError, sqlite3.Error, json.JSONDecodeError, ValidationError) as exc:
        log.info("query_tables.skipped", reason=type(exc).__name__, error=str(exc))
        return {"nodes_traversed": state.nodes_traversed + 1}

    log.info("injected", kind=result.kind, chunk_id=synthetic.chunk_id)
    # FULL augmented list — `reranked` is a plain field, so this REPLACES it
    # (returning only `[synthetic]` would wipe the real chunks). The spread
    # form is the same new-list replacement as `state.reranked + [synthetic]`.
    return {
        "reranked": [*state.reranked, synthetic],
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

    # P3.3 v7 (2026-05-23 force-docling A/B trace): keep
    # `[chart-extracted]` blocks visible to the assessor. The v3-v6
    # defenses stripped them to prevent verbose LaTeX from eating
    # the `truncate(1200)` budget; the v7 chart-OCR backend emits
    # compact markdown tables / key-value bullets (~50-150 chars per
    # block) which fit within the truncate budget alongside prose.
    # Stripping was masking the chart-OCR upside on chart-content
    # questions (the right chunk reaches rank-1 then gets blanked
    # before the LLM sees it).
    prompt = render_prompt(
        "assess_sufficiency",
        query=state.query,
        chunks=state.reranked,
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


# --- Citation-id repair -------------------------------------------------------

# Max edit distance between an emitted hash and a real chunk-hash for a
# fuzzy repair to fire. Chunk hashes are 10 random hex chars, so two
# distinct ones landing within distance 2 is vanishingly unlikely — the
# uniqueness guard below makes a wrong snap unlikelier still.
_CHUNK_ID_FUZZY_MAX_DISTANCE = 2


def _bounded_levenshtein(a: str, b: str, *, max_d: int) -> int:
    """Levenshtein edit distance, short-circuiting at `max_d + 1`.

    We only care whether two short chunk-hashes are *near* each other;
    the early exit keeps this cheap (reranked sets are ≤ ~7 chunks).
    """
    if abs(len(a) - len(b)) > max_d:
        return max_d + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_best = i
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            row_best = min(row_best, cur[-1])
        if row_best > max_d:
            return max_d + 1
        prev = cur
    return prev[-1]


def repair_claim_chunk_ids(
    claims: list[CitedClaim], reranked: list[Chunk]
) -> tuple[list[CitedClaim], dict[str, int]]:
    """Snap each claim's emitted `source_chunk_id` to a real reranked
    chunk_id, repairing transcription errors from the answer LLM.

    The answer prompt shows the model the full `docid#hash` chunk id,
    but on long ids the model occasionally (a) drops the `docid#`
    prefix and emits the bare hash, or (b) flips a character. Left
    uncorrected those break the citation audit trail
    (`compose` matches `source_chunk_id` against reranked chunk_ids to
    build `FinalResponse.used_chunks`) AND the `verify` chunk lookup
    (`chunk_by_id[claim.source_chunk_id]`). Surfaced by the CR350
    multi-doc eval (2026-05-23): xref-05 emitted bare hashes, xref-02
    emitted a single-char-corrupted hash.

    Repair order, decreasing safety:
      1. exact   — id already matches a reranked chunk_id: keep.
      2. suffix  — emitted hash equals the `#<hash>` of exactly one
                   reranked chunk_id (bare-hash emission): repair.
      3. fuzzy   — emitted hash is within `_CHUNK_ID_FUZZY_MAX_DISTANCE`
                   of exactly one reranked hash, with a strictly closer
                   match than any other (no tie): repair.
    Anything else is left untouched — a dangling citation the verifier
    will then mark ungrounded. Repairing here (in `answer`, before
    `verify`) means the grounding check sees the corrected id too.

    Returns `(repaired_claims, stats)` where stats counts each branch.
    """
    valid_ids = [c.chunk_id for c in reranked]
    valid_set = set(valid_ids)
    hash_of = {cid: cid.rsplit("#", 1)[-1] for cid in valid_ids}

    stats = {"exact": 0, "suffix": 0, "fuzzy": 0, "unresolved": 0}
    repaired: list[CitedClaim] = []

    for claim in claims:
        emitted = claim.source_chunk_id
        if emitted in valid_set:
            stats["exact"] += 1
            repaired.append(claim)
            continue

        emitted_hash = emitted.rsplit("#", 1)[-1]
        suffix_matches = [cid for cid in valid_ids if hash_of[cid] == emitted_hash]
        if len(suffix_matches) == 1:
            stats["suffix"] += 1
            repaired.append(claim.model_copy(update={"source_chunk_id": suffix_matches[0]}))
            continue

        scored = sorted(
            (
                _bounded_levenshtein(
                    emitted_hash,
                    hash_of[cid],
                    max_d=_CHUNK_ID_FUZZY_MAX_DISTANCE,
                ),
                cid,
            )
            for cid in valid_ids
        )
        if (
            scored
            and scored[0][0] <= _CHUNK_ID_FUZZY_MAX_DISTANCE
            and (len(scored) == 1 or scored[1][0] > scored[0][0])
        ):
            stats["fuzzy"] += 1
            repaired.append(claim.model_copy(update={"source_chunk_id": scored[0][1]}))
            continue

        stats["unresolved"] += 1
        repaired.append(claim)

    return repaired, stats


def _is_context_overflow(error: Exception) -> bool:
    """True iff a `ModelCallError` is a vLLM context-length 400 — the rendered
    prompt plus the requested output exceeds the model's window (distinct from
    other 400s / real I/O failures by the 'maximum context length' phrasing the
    OpenAI-compatible server returns). The answer node degrades on this rather
    than letting it abort the whole run."""
    return "maximum context length" in str(error).lower()


async def answer(state: AnswerState) -> AnswerStateUpdate:
    """Generate a draft answer with explicit citations.

    On regeneration, includes feedback about which claims were
    ungrounded in the previous attempt.
    """
    log = logger.bind(node="answer")
    log.info("start", regenerate_attempt=state.regenerate_attempts)

    feedback = ""
    if state.draft is not None and state.verification is not None and state.verification.ungrounded:
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

    # P3.3 v7 (2026-05-23): keep `[chart-extracted]` blocks visible
    # to the answer LLM. See assess() above for the rationale. The
    # markdown-table emission from chart_ocr_backend._latex_tabular_
    # to_markdown + _split_label_number_cells keeps chart blocks
    # compact enough to live alongside prose within truncate(1800).
    # Generate the draft, DEGRADING on a context-length overflow instead of
    # crashing the run. vLLM raises a 400 when the rendered chunks + the 1800-tok
    # output reservation exceed the window (e.g. a few dense 10-K table chunks in
    # the 6144 fast window — the eval-aborting case). Drop the lowest-ranked REAL
    # chunk and retry (NEVER the synthetic Table-RAG `#sql0001` chunk, which holds
    # the aggregate/superlative answer): verify rebuilds `chunk_by_id` from the
    # FULL `state.reranked`, so grounding for the kept chunks is unaffected. If
    # even the single top chunk overflows (pathological), refuse via an empty
    # draft (verify short-circuits → `route_after_verify` → refuse) — never crash.
    from memex.core.errors import ModelCallError

    answer_chunks = list(state.reranked)
    draft: DraftAnswer | None = None
    tokens = 0
    while draft is None:
        messages = render_messages(
            "answer",
            query=state.query,
            chunks=answer_chunks,
            feedback=feedback,
        )
        try:
            draft, tokens = await complete_structured(
                prompt=messages,
                schema=DraftAnswer,
                # Explicit (above the 1024 default): the summary cap is 600 + up
                # to 8 claims (~435 chars each) ≈ 4.1k chars ≈ ~1.1-1.6k tokens
                # worst-case, so xgrammar needs room to CLOSE the JSON or the
                # draft truncates invalid. 1800 clears that with margin and still
                # fits the fast 6,144 window. The summary-cap bump (300→600) fixed
                # the mid-word "policyEn" cut full mode's richer answers exposed.
                max_tokens=1800,
                prompt_tag="answer@v3",
            )
        except ModelCallError as e:
            if not _is_context_overflow(e):
                raise
            # Drop the lowest-ranked REAL chunk — NEVER the synthetic Table-RAG
            # `#sql0001` chunk. It's appended LAST (so a naive drop-the-last would
            # discard it FIRST) but it carries the aggregate/superlative answer, so
            # dropping it defeats the table query (regressed ar-14/ar-15 under
            # overflow at top_k=5/6144). Find the lowest-ranked non-synthetic chunk.
            drop_idx = next(
                (
                    i
                    for i in range(len(answer_chunks) - 1, -1, -1)
                    if not answer_chunks[i].chunk_id.endswith("#sql0001")
                ),
                None,
            )
            if drop_idx is None:
                # Only the synthetic chunk remains and it STILL overflows
                # (pathological — it's bounded < the smallest truncate budget):
                # refuse via an empty draft rather than crash the run.
                log.warning("context_overflow_refuse", chunks=len(answer_chunks))
                draft = DraftAnswer(
                    summary="No answer could be generated within the model's context window.",
                    claims=[],
                )
                tokens = 0
                break
            dropped = answer_chunks.pop(drop_idx)
            log.info(
                "context_overflow_retry",
                remaining=len(answer_chunks),
                dropped_chunk=dropped.chunk_id,
            )

    # Repair corrupted citation ids before they reach verify/compose.
    # The answer LLM sometimes mangles the long `docid#hash` ids it's
    # shown (bare hash, single-char flip); snapping them back to real
    # reranked chunk_ids keeps the audit trail + grounding check honest.
    if draft.claims and state.reranked:
        repaired, repair_stats = repair_claim_chunk_ids(draft.claims, state.reranked)
        if repair_stats["suffix"] or repair_stats["fuzzy"] or repair_stats["unresolved"]:
            log.info("chunk_id_repair", **repair_stats)
        draft = draft.model_copy(update={"claims": repaired})

    return {
        "draft": draft,
        "tokens_used": state.tokens_used + tokens,
        "nodes_traversed": state.nodes_traversed + 1,
    }


# --- Numeric-grounding backstop (2026-05-31) --------------------------------
# A deterministic post-verify gate (in `verify`) that demotes a grounded claim
# whose PRINCIPAL large figure is absent from its cited TABLE chunk — the
# verbatim-cell half of the Table-RAG fabrication boundary (agents/table_sql.py)
# ported onto the free-text verify path. It closes the verify_grounding
# aggregate-numeric FALSE-POSITIVE: the LLM verifier rubber-stamps a SUMMED table
# total via the "literal reading of a table row / structural adjacency" rule
# (the live `$159,748,343` 10-K fabrication; the recorded `$16,042M` vs `16384`).
# It is engineered to NOT false-drop legitimate numeric answers:
#   * fires ONLY on a claim citing a chunk that CONTAINS a markdown table (the
#     cross-cell-aggregate site) — prose-cited rounded headlines ("$216 billion")
#     are exempt;
#   * EXEMPTS the synthetic Table-RAG `#sql0001` chunk (its aggregate is already
#     independently recompute-gated, and `%g` framing renders a >=1e6 value
#     unreadable so a verbatim re-check would wrongly drop it);
#   * scopes to a LARGE figure (|value| >= 1e4) so years (2026), small counts,
#     percentages, and `x`-suffixed values never enter scope;
#   * matches a cell under a thousands/millions/billions DENOMINATION shift (a
#     power of 1000) so "$22.5 billion" grounds in a "22,500" $-millions cell and
#     "$16,042 million" in a "16,042" cell — but DELIBERATELY NOT an arbitrary
#     x10/x100 shift (that would launder a mis-scaled fabrication).
# Demotion-only ⇒ HARD-gate-safe by construction (can only refuse, never admit).
# Kill-switch: AgentsSettings.numeric_grounding_backstop_enabled (default True).

# A number-like token: optional currency, optional accounting parens / sign,
# digits with thousands separators, optional decimal, optional %/scale suffix.
# Greedy enough to keep "$159,748,343", "$22.5 billion", "(1,234)", "215,938"
# whole. coerce_number does the real parsing; this just isolates candidates.
_NUMBER_TOKEN_RE = re.compile(
    # A scale WORD may sit after a space ("$22.5 billion"); a scale LETTER must
    # attach directly ("2.5B") — and both end on a word boundary so the "t" of a
    # following word ("$50,000 to …") is never mistaken for a trillion suffix.
    r"[$€£]?\(?[+-]?\d[\d,]*(?:\.\d+)?\)?(?:\s*(?:thousand|million|billion|trillion)\b|[KMBT]\b)?",
    re.IGNORECASE,
)
# A GFM table separator row (`|---|---|`, optional leading/trailing pipe +
# alignment colons) — the structural signature that a chunk carries a table.
_TABLE_SEPARATOR_RE = re.compile(r"(?m)^[ \t]*\|?[ \t]*:?-{1,}:?[ \t]*(?:\|[ \t]*:?-{1,}:?[ \t]*)+\|?[ \t]*$")
# The index-time `[table-rows]` linearization marker (W1, linearize_gfm_tables)
# — a second, parse-engine-independent table signature.
_TABLE_ROWS_MARKER = "[table-rows]"
# Below this magnitude a figure is out of scope (years, small counts/ordinals);
# the fabrication shape this gate targets is a large financial aggregate.
_NUMERIC_BACKSTOP_MIN_MAGNITUDE = 1e4
# Only a thousands/millions/billions denomination bridges a claim figure to a
# cell — NOT an arbitrary x10/x100 (which would be a mis-scaled fabrication).
_NUMERIC_DENOMINATION_FACTORS = (1.0, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9)
# A claim phrased as an arithmetic SUM EXPRESSION of large (thousands-separated)
# figures — `19,166,424 + 18,034,343 + ...`. This is a computed aggregate whose
# RESULT the doc never states, even though each summand is a verbatim cell — so
# the "any figure supported" rule would wrongly keep it (the ar-16 evasion). The
# first operand must carry a thousands separator so a bare "2025 + 2026" (years)
# never matches.
_SUM_EXPRESSION_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*\+\s*[$€£(]?\s*\d")


def _chunk_has_markdown_table(chunk_text: str) -> bool:
    """True iff the chunk carries a GFM table (a `|---|` separator row) or its
    index-time `[table-rows]` linearization — the only site where a cross-cell
    aggregate is fabricated. A prose-only chunk returns False (exempt)."""
    return _TABLE_ROWS_MARKER in chunk_text or _TABLE_SEPARATOR_RE.search(chunk_text) is not None


def _chunk_numbers(chunk_text: str) -> list[float]:
    """Every number-like token in the FULL chunk text parsed via `coerce_number`
    (covers GFM cells, the `[table-rows]` KV linearization, and prose figures)."""
    out: list[float] = []
    for tok in _NUMBER_TOKEN_RE.findall(chunk_text):
        value = coerce_number(tok)
        if value is not None:
            out.append(value)
    return out


def _claim_scoped_figures(claim_text: str) -> list[float]:
    """Every LARGE figure the claim asserts (|`coerce_number`| >= 1e4). Years,
    small counts, and `x`/%-only values are out of scope (the LLM verdict stands
    untouched). A claim with NO such figure returns `[]` → the gate skips it.

    The gate demotes only when EVERY one of these is unsupported — so a claim
    stating a rounded headline AND an exact figure (e.g. "~800M … 788M") is kept
    on the strength of the verbatim 788M, never false-dropped on the rounded one."""
    out: list[float] = []
    for tok in _NUMBER_TOKEN_RE.findall(claim_text):
        value = coerce_number(tok)
        if value is None or abs(value) < _NUMERIC_BACKSTOP_MIN_MAGNITUDE:
            continue
        out.append(value)
    return out


def _claim_is_sum_expression(claim_text: str) -> bool:
    """True if the claim presents an arithmetic SUM of large figures
    (`19,166,424 + 18,034,343 + ...`) — a computed aggregate whose result the doc
    never states, so it must be demoted even though each summand is a verbatim
    cell (the "any figure supported" rule would otherwise keep it)."""
    return _SUM_EXPRESSION_RE.search(claim_text) is not None


def _figure_supported_by_chunk(figure: float, chunk_numbers: list[float]) -> bool:
    """True iff `figure` equals some chunk number, allowing only a
    thousands/millions/billions denomination shift (a power of 1000), under the
    Table-RAG aggregate-gate tolerance `|a - b| <= max(1, 1e-6 * |b|)`. A
    computed aggregate (no single cell at any clean unit scale) returns False."""
    if not math.isfinite(figure):
        return False
    for cell in chunk_numbers:
        if not math.isfinite(cell):
            continue
        for factor in _NUMERIC_DENOMINATION_FACTORS:
            target = cell * factor
            if abs(figure - target) <= max(1.0, 1e-6 * abs(target)):
                return True
    return False


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
            "verification": VerificationResult(grounded=[], ungrounded=[], ungrounded_reasons=[]),
            "nodes_traversed": state.nodes_traversed + 1,
        }

    # P3.3 v7 (2026-05-23): same as assess()/answer() — preserve
    # chart-extracted blocks in the verifier's view so chart-content
    # citations are checkable. The compact markdown emission means
    # chart blocks no longer eat the truncate budget.
    chunk_by_id = {c.chunk_id: c for c in state.reranked}

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
        __base__=VerificationResult,
        grounded=(
            Annotated[list[int], Field(max_length=n)],
            Field(
                description=f"0-indexed positions in draft.claims (0..{n - 1}) whose cited chunk supports the claim."
            ),
        ),
        ungrounded=(
            Annotated[list[int], Field(max_length=n)],
            Field(
                description=f"0-indexed positions in draft.claims (0..{n - 1}) whose cited chunk does NOT support the claim."
            ),
        ),
        # Hardening (audit 2026-05-22): bound each reason string's
        # length too, not just the outer list. Same pathology as the
        # v6 SufficiencyAssessment.reason: the model can ramble per
        # ungrounded claim, exhausting max_tokens before closing the
        # JSON. 250 chars per "one short reason" is generous.
        ungrounded_reasons=(
            Annotated[
                list[Annotated[str, Field(max_length=250)]],
                Field(max_length=n),
            ],
            Field(
                default_factory=list,
                description="Optional, parallel to `ungrounded`: one short reason per ungrounded claim.",
            ),
        ),
    )

    # `__base__=VerificationResult` above makes the dynamic model a true
    # subclass, so pyright infers `bounded` as VerificationResult and the
    # `.grounded` / `.ungrounded` / `.ungrounded_reasons` accesses below
    # stay typed — no cast needed.
    bounded, tokens = await complete_structured(
        prompt=prompt,
        schema=BoundedVerificationResult,
        prompt_tag="verify_grounding@v2",
    )

    # Defensive index-filtering (P3.3 v7 trace, 2026-05-23). The verifier
    # can emit phantom indices > n-1 when distracted by chunk content not
    # in the draft (observed: 1-claim draft, verifier returned
    # `grounded: [0], ungrounded: [1]` referencing "Pareto chart" from
    # the chunk text but not the draft). The xgrammar `max_length=n`
    # bounds list length, not index values. Filter to 0..n-1.
    valid_grounded = [i for i in bounded.grounded if 0 <= i < n]
    valid_ungrounded = [i for i in bounded.ungrounded if 0 <= i < n]
    valid_reasons = [
        bounded.ungrounded_reasons[k]
        for k, idx in enumerate(bounded.ungrounded)
        if 0 <= idx < n and k < len(bounded.ungrounded_reasons)
    ]
    # Post-audit defense (2026-05-23 verification audit): if the verifier
    # over-emitted reasons relative to ungrounded indices (e.g. ungrounded
    # had 1 entry, reasons had 3), the excess is silently dropped by the
    # enumerate-zip above. Log when this happens so wrong-reason-attribution
    # surfaces in observability; the dropped reasons would otherwise mislead
    # the regenerate-feedback loop.
    overflow = max(0, len(bounded.ungrounded_reasons) - len(bounded.ungrounded))
    if overflow:
        log.info(
            "verify.reasons_overflow_dropped",
            overflow_count=overflow,
            ungrounded_count=len(bounded.ungrounded),
            reasons_count=len(bounded.ungrounded_reasons),
        )
    # Any claim index missing from BOTH lists → treat as ungrounded
    # (conservative: don't let a verifier omission falsely mark a claim
    # as grounded by default).
    mentioned = set(valid_grounded) | set(valid_ungrounded)
    missing = [i for i in range(n) if i not in mentioned]
    if missing:
        log.info("verify.indices_missing", missing=missing, draft_claim_count=n)
        valid_ungrounded.extend(missing)
        valid_reasons.extend(["Verifier omitted this claim from its response."] * len(missing))
    phantom_g = [i for i in bounded.grounded if not (0 <= i < n)]
    phantom_u = [i for i in bounded.ungrounded if not (0 <= i < n)]
    if phantom_g or phantom_u:
        log.info(
            "verify.phantom_indices_dropped",
            phantom_grounded=phantom_g,
            phantom_ungrounded=phantom_u,
            draft_claim_count=n,
        )
    # A claim index the verifier put in BOTH lists must not ship as grounded —
    # `compose` keys solely on `verification.grounded`, so without this an index
    # the verifier ALSO flagged ungrounded would still be emitted. Treat ungrounded
    # as authoritative (consistent with the missing-index "treat-as-ungrounded" stance).
    contested = set(valid_grounded) & set(valid_ungrounded)
    if contested:
        log.info("verify.contested_indices_demoted", contested=sorted(contested))
        valid_grounded = [i for i in valid_grounded if i not in contested]

    # Numeric-grounding backstop (2026-05-31): a 4th deterministic demotion,
    # mirroring the phantom/missing/contested filters above. Demote a claim the
    # LLM marked grounded when it asserts a PRINCIPAL large figure that is absent
    # from its cited TABLE chunk under a clean unit-denomination shift — i.e. a
    # computed aggregate the verifier accepted via the literal-table-row
    # loophole. See `_figure_supported_by_chunk` et al. above. Demotion-only ⇒
    # HARD-gate-safe; the existing route_after_verify/compose/refuse handle a
    # zero-grounded result (→ refuse). Fail-open kill-switch on AnswerState.
    if state.numeric_grounding_backstop:
        numeric_demoted: list[int] = []
        for i in valid_grounded:
            claim = state.draft.claims[i]
            # The Table-RAG synthetic aggregate is recompute-gated upstream AND
            # its %g framing renders a >=1e6 value unreadable to coerce_number,
            # so a verbatim re-check would wrongly drop it — exempt by id.
            if claim.source_chunk_id.endswith("#sql0001"):
                continue
            chunk = chunk_by_id.get(claim.source_chunk_id)
            if chunk is None or not _chunk_has_markdown_table(chunk.text):
                continue  # prose-only / dangling: not a cross-cell-aggregate site
            figures = _claim_scoped_figures(claim.claim)
            if not figures:
                continue  # no large figure → out of scope
            chunk_numbers = _chunk_numbers(chunk.text)
            # Demote when EVERY asserted large figure is unsupported (a single
            # verbatim figure, even alongside a rounded one, keeps the claim) OR
            # the claim is an arithmetic SUM EXPRESSION (`a + b + ...`) — a computed
            # aggregate whose result the doc never states, even though each summand
            # is a verbatim cell (the ar-16 "19,166,424 + 18,034,343 + …" evasion).
            unsupported = not any(_figure_supported_by_chunk(f, chunk_numbers) for f in figures)
            if unsupported or _claim_is_sum_expression(claim.claim):
                numeric_demoted.append(i)
        if numeric_demoted:
            log.info("verify.numeric_aggregate_demoted", demoted=numeric_demoted)
            demote = set(numeric_demoted)
            valid_grounded = [i for i in valid_grounded if i not in demote]
            already_ungrounded = set(valid_ungrounded)
            for i in numeric_demoted:
                if i not in already_ungrounded:
                    valid_ungrounded.append(i)
                    valid_reasons.append(
                        "Claim asserts a large numeric value not present in the cited "
                        "table; a computed aggregate is not a literal cell reading."
                    )

    # Name-only grounding backstop (2026-06-03): a 5th deterministic demotion, same shape as the
    # numeric one. The single greedy `verify_grounding/v2` call grounds a BEHAVIORAL claim against
    # a chunk that merely NAMES the entity (a bare list/heading) via its "structural adjacency is
    # sufficient" rule — measured 3/3 (e.g. "RBAC assigns permissions by role" cited to a slide
    # that only LISTS "RBAC"). Demote a still-grounded claim whose cited chunk is name-only AND
    # whose predicate is behavioral/property/comparative. FAIL-OPEN + demotion-only ⇒ over-refusal-
    # safe BY CONSTRUCTION: a membership/existence claim (a name-list DOES support it) and any
    # unrecognised phrasing are KEPT, so the worst case is the status-quo over-grounding, never a
    # new refusal; a zero-grounded result routes to `refuse` as usual. `is_name_only_chunk` returns
    # False for any table/chart chunk, so the Table-RAG/chart literal-read rule is untouched.
    # The demotion RULE (`claim_grounded_only_by_name`) is shared verbatim with the bridge's
    # name-only demotion (`core/text.py`) — one source of truth. Kill-switch on AnswerState.
    if state.name_only_grounding_backstop:
        name_only_demoted: list[int] = []
        for i in valid_grounded:
            claim = state.draft.claims[i]
            chunk = chunk_by_id.get(claim.source_chunk_id)
            if chunk is None:
                continue  # dangling chunk: not the loophole site
            if claim_grounded_only_by_name(claim.claim, chunk.text):
                name_only_demoted.append(i)
        if name_only_demoted:
            log.info("verify.name_only_demoted", demoted=name_only_demoted)
            demote = set(name_only_demoted)
            valid_grounded = [i for i in valid_grounded if i not in demote]
            already_ungrounded = set(valid_ungrounded)
            for i in name_only_demoted:
                if i not in already_ungrounded:
                    valid_ungrounded.append(i)
                    valid_reasons.append(
                        "Cited chunk only NAMES the subject (a bare list/heading); it does not "
                        "state the behavior/property the claim asserts."
                    )

    verification = VerificationResult(
        grounded=valid_grounded,
        ungrounded=valid_ungrounded,
        ungrounded_reasons=valid_reasons,
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

    if state.relevance and not state.relevance.responsive:
        reason = (
            "The retrieved material addresses a related topic but not your "
            f"specific question. {state.relevance.reason}"
        )
    elif state.sufficiency and not state.sufficiency.sufficient:
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
        artifact_scope_doc_ids=state.artifact_scope_doc_ids,
        correlation_id=state.correlation_id,
        tokens_used=state.tokens_used,
        nodes_traversed=new_nodes,
        regenerate_attempts=state.regenerate_attempts,
    )
    return {"final": final, "nodes_traversed": new_nodes}


async def assess_relevance(state: AnswerState) -> AnswerStateUpdate:
    """Responsiveness gate — runs after verify (grounded path), before compose.

    Grounding confirms each claim is supported by its chunk; this confirms the
    grounded answer actually RESPONDS to the specific question (vs a related-
    but-different one). Catches the conflation where adjacent-topic content is
    passed off as the answer. Conservative by design (defaults responsive) so
    it removes only clear question/answer topic mismatches. See
    `RelevanceAssessment`. Non-responsive routes to `refuse`.
    """
    log = logger.bind(node="assess_relevance")
    log.info("start")
    # Only reached on the grounded path, so draft + verification are present;
    # guard defensively and pass through as responsive if they somehow aren't.
    if state.draft is None or state.verification is None:
        return {
            "relevance": RelevanceAssessment(responsive=True, reason="no grounded draft to assess"),
            "nodes_traversed": state.nodes_traversed + 1,
        }
    grounded = set(state.verification.grounded)
    grounded_claims = [c.claim for i, c in enumerate(state.draft.claims) if i in grounded]
    prompt = render_prompt(
        "assess_relevance",
        query=state.query,
        summary=state.draft.summary,
        claims=grounded_claims,
    )
    relevance, tokens = await complete_structured(
        prompt=prompt,
        schema=RelevanceAssessment,
        prompt_tag="assess_relevance@v1",
    )
    log.info("relevance", responsive=relevance.responsive)
    return {
        "relevance": relevance,
        "tokens_used": state.tokens_used + tokens,
        "nodes_traversed": state.nodes_traversed + 1,
    }


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
        claim for i, claim in enumerate(state.draft.claims) if i in grounded_indices
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
            artifact_scope_doc_ids=state.artifact_scope_doc_ids,
            correlation_id=state.correlation_id,
            tokens_used=state.tokens_used,
            nodes_traversed=new_nodes,
            regenerate_attempts=state.regenerate_attempts,
        )
        return {"final": final, "nodes_traversed": new_nodes}

    used_chunk_ids = {c.source_chunk_id for c in surviving_claims}
    used_chunks = [c for c in state.reranked if c.chunk_id in used_chunk_ids]

    # Deterministic, grounded wikilinks: one `[[doc_id#section]]` per
    # cited chunk (section = deepest heading_path entry, raw text), deduped
    # preserving first-seen order so a "Sources" list doesn't repeat. Pure
    # derivation from the already-grounded used_chunks — no model call, no
    # behaviour change to answered/claims (HARD-gate-neutral).
    wikilinks: list[str] = []
    seen_wikilinks: set[str] = set()
    for chunk in used_chunks:
        section = chunk.heading_path[-1] if chunk.heading_path else None
        wikilink = format_wikilink(chunk.document_id, section)
        if wikilink not in seen_wikilinks:
            seen_wikilinks.add(wikilink)
            wikilinks.append(wikilink)

    # Summary safety for a PARTIAL-grounded ship: the model's `summary`
    # synthesizes the WHOLE draft, so if verification dropped any claim it may
    # assert the dropped (ungrounded) content. Rebuild the summary from the
    # surviving grounded claims so no ungrounded assertion reaches the headline.
    # When every claim survived, keep the model's richer one-line synthesis.
    if len(surviving_claims) < len(state.draft.claims):
        summary = " ".join(c.claim for c in surviving_claims)
        log.info(
            "partial_grounded_ship",
            grounded=len(surviving_claims),
            dropped=len(state.draft.claims) - len(surviving_claims),
        )
    else:
        summary = state.draft.summary

    final = FinalResponse(
        answered=True,
        summary=summary,
        claims=surviving_claims,
        used_chunks=used_chunks,
        wikilinks=wikilinks,
        artifact_scope_doc_ids=state.artifact_scope_doc_ids,
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


def _has_grounded_claims(state: AnswerState) -> bool:
    """True iff at least one draft claim was marked grounded with a VALID index.

    Mirrors `compose`'s surviving-claims computation (a grounded index must fall
    within `draft.claims`), so it answers exactly "would compose ship anything?"
    """
    if state.draft is None or state.verification is None:
        return False
    n = len(state.draft.claims)
    return any(0 <= i < n for i in state.verification.grounded)


def route_after_verify(
    state: AnswerState,
) -> Literal["compose", "regenerate", "refuse"]:
    """Route a verified draft.

    - Fully grounded -> compose (via the relevance gate). Checked *before* the
      budget gate: we already paid for a valid answer and `compose` does no model
      work, so a late budget breach shouldn't discard it.
    - Mixed/none grounded, with budget + attempts left -> regenerate (try to
      ground the rest; the answer node is fed the ungrounded-claim feedback).
    - Regeneration exhausted (or over budget): if a NON-EMPTY grounded subset
      remains, ship it -> "compose" (the relevance gate still vets responsiveness;
      `compose` drops the ungrounded claims and rebuilds the summary from the
      survivors). Only a ZERO-grounded verdict refuses. This is the fix for the
      compound-question over-refusal: a question whose groundable half the corpus
      supports answers that half instead of refusing the whole thing because the
      other half can't be grounded. Counterfactuals (nothing grounded) still
      refuse, so `refusal_cf` is unaffected. Gated by `allow_partial_grounded`
      (default on; `MEMEX_AGENTS__PARTIAL_GROUNDED_ANSWERS=false` restores
      all-or-nothing).
    """
    if state.verification and state.verification.all_grounded:
        return "compose"
    # Try to ground the remaining claims while budget + attempts remain.
    if not state.over_budget() and state.regenerate_attempts < state.max_regenerate_attempts:
        return "regenerate"
    # Retry exhausted / over budget. Ship the grounded subset if any survives.
    if state.allow_partial_grounded and _has_grounded_claims(state):
        return "compose"
    return "refuse"


def route_after_relevance(state: AnswerState) -> Literal["compose", "refuse"]:
    """Responsive -> compose. Non-responsive -> refuse.

    A non-responsive verdict means the draft was grounded but answered a
    related-but-different question than the one asked (the conflation guard).
    Conservative: only refuses on an explicit non-responsive verdict.
    """
    if state.relevance and not state.relevance.responsive:
        return "refuse"
    return "compose"


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
      retrieve --> expand_graph --> rerank --> query_tables --> assess
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
                    assess_relevance regenerate refuse
                             |         |         |
                  +----------+----+ (back to    END
                  |               |  answer)
              compose          refuse
                  |               |
                  v               v
                END             END

    `assess_relevance` is the responsiveness gate: a grounded draft that
    answers a related-but-different question than asked (a conflation)
    routes to `refuse` instead of `compose`. `expand_graph` augments the
    candidate pool with chunks from
    documents the graph store says are related (one-hop CITES /
    shared-entity neighbours). It is a no-op when the graph store
    isn't installed, when expansion is disabled on AnswerState, or
    when retrieval returned no candidates.
    """
    g = StateGraph(AnswerState)

    # Nodes
    g.add_node("retrieve", retrieve)
    g.add_node("resolve_artifact_scope", resolve_artifact_scope)
    g.add_node("expand_graph", expand_graph)
    g.add_node("rerank", rerank)
    g.add_node("query_tables", query_tables)
    g.add_node("assess", assess)
    g.add_node("answer", answer)
    g.add_node("verify", verify)
    g.add_node("regenerate", regenerate)
    g.add_node("refuse", refuse)
    g.add_node("assess_relevance", assess_relevance)
    g.add_node("compose", compose)

    # Linear edges
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "resolve_artifact_scope")
    g.add_edge("resolve_artifact_scope", "expand_graph")
    g.add_edge("expand_graph", "rerank")
    g.add_edge("rerank", "query_tables")
    g.add_edge("query_tables", "assess")
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
            "compose": "assess_relevance",  # grounded -> responsiveness gate -> compose
            "regenerate": "regenerate",
            "refuse": "refuse",
        },
    )
    g.add_conditional_edges(
        "assess_relevance",
        route_after_relevance,
        {"compose": "compose", "refuse": "refuse"},
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
    scope_doc_ids: list[str] | None = None,
    prior_carry_chunk_ids: list[str] | None = None,
    correlation_id: str | None = None,
    on_node: Callable[[str], None] | None = None,
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

    `scope_doc_ids` (the doc-picker): an EXPLICIT list of document ids to scope
    retrieval to — takes precedence over inferred artifact-scope (#256). Empty /
    None = the full-corpus path. The applied scope is surfaced on
    `FinalResponse.artifact_scope_doc_ids`.

    `prior_carry_chunk_ids` (the grounded multi-turn chat): chunk_ids from the
    immediately-prior answered turn, re-admitted as extra rerank candidates in
    `retrieve` so a tight follow-up keeps its referents (Surface A,
    `docs/specs/grounded-agentic-chat.md`). None/[] → byte-identical to a bare `/ask`.
    A carried chunk only reaches grounding if it survives rerank; `verify` is unchanged.

    `correlation_id` (optional): a caller-supplied id used instead of a fresh
    ULID — it drives the structlog/Langfuse binding AND
    `FinalResponse.correlation_id` (the webui keys its progress registry by it).
    `on_node` (optional): an observe-only sink invoked with each top-level node
    name as it STARTS, for a live progress UI (`_NodeProgressHandler`); it never
    touches graph state or routing. Both default off → CLI/MCP unchanged.
    """
    # Partial-grounded ship is a settings policy toggle (default on). Read it
    # fail-open so a non-bootstrapped fixture keeps the default behaviour.
    from memex.core.config import get_settings
    from memex.core.errors import ConfigurationError, MemexError

    try:
        allow_partial_grounded = get_settings().agents.partial_grounded_answers
    except (ConfigurationError, MemexError):
        allow_partial_grounded = True

    # The numeric-grounding backstop is a settings policy toggle (default on),
    # read fail-open so a non-bootstrapped fixture keeps the default behaviour.
    try:
        numeric_grounding_backstop = get_settings().agents.numeric_grounding_backstop_enabled
    except (ConfigurationError, MemexError):
        numeric_grounding_backstop = True

    # The name-only grounding backstop is the analogous settings policy toggle (default on),
    # read fail-open so a non-bootstrapped fixture keeps the default behaviour.
    try:
        name_only_grounding_backstop = get_settings().agents.name_only_grounding_backstop_enabled
    except (ConfigurationError, MemexError):
        name_only_grounding_backstop = True

    # Graph expansion is the param ANDed with the settings kill-switch (default on),
    # read fail-open — so `MEMEX_AGENTS__GRAPH_EXPANSION_ENABLED=false` disables it
    # globally (for the earns-its-keep A/B) while an explicit param=False still wins.
    try:
        graph_expansion_enabled = (
            graph_expansion_enabled and get_settings().agents.graph_expansion_enabled
        )
    except (ConfigurationError, MemexError):
        pass

    initial = AnswerState(
        query=query,
        token_budget=token_budget,
        max_regenerate_attempts=max_regenerate_attempts,
        graph_expansion_enabled=graph_expansion_enabled,
        graph_expansion_budget=graph_expansion_budget,
        chunks_per_neighbor=chunks_per_neighbor,
        allow_partial_grounded=allow_partial_grounded,
        numeric_grounding_backstop=numeric_grounding_backstop,
        name_only_grounding_backstop=name_only_grounding_backstop,
        scope_doc_ids=scope_doc_ids or [],
        # The grounded multi-turn chat's bounded prior-chunk carry (Surface A). Default
        # None/[] → byte-identical to a bare `/ask`; `retrieve` unions these into the
        # candidate pool where they must survive rerank to reach grounding.
        prior_carry_chunk_ids=prior_carry_chunk_ids or [],
        # A caller-supplied correlation_id (e.g. the webui's progress key) drives
        # both the structlog/Langfuse binding and FinalResponse.correlation_id, so
        # logs + trace + the progress registry stay joined (ADR-0004). None → mint
        # a fresh ULID here (matching the field's default_factory).
        correlation_id=correlation_id if correlation_id is not None else str(ulid.ULID()),
    )

    clear_run_context()  # belt-and-suspenders: a missed clear elsewhere
    bind_run_context(initial.correlation_id, query_preview=query[:80])
    try:
        graph = get_compiled_graph()
        # Langfuse trace + (opt-in) the webui's live node-progress observer. The
        # progress handler is observe-only — appended after Langfuse, it never
        # touches state or routing, so the HARD gates are unaffected.
        callbacks: list[BaseCallbackHandler] = [callback_handler()]
        if on_node is not None:
            callbacks.append(_NodeProgressHandler(on_node))
        final_state = await graph.ainvoke(
            initial,
            config={
                "callbacks": callbacks,
                "metadata": run_attributes(initial.correlation_id, "answer_query"),
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
