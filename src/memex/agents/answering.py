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

import threading
from typing import Annotated, Literal, TypedDict

import structlog
import ulid
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, create_model

from memex.core.errors import AnswerStateInvariantError
from memex.core.types import Chunk, StoredTable, TableQueryResult
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
    sufficiency: SufficiencyAssessment
    draft: DraftAnswer
    verification: VerificationResult
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

    # Path C tightening (2026-05-22): see CitedClaim for the budget
    # rationale. "One or two sentences" overview legitimately runs
    # 50-200 chars; 300 is generous headroom while leaving room for
    # up to 8 claims under max_tokens=1024.
    summary: str = Field(
        description="One or two sentences overviewing the answer.",
        max_length=300,
    )
    claims: list[CitedClaim] = Field(
        description="Detailed claims, each with a citation.",
        max_length=8,
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
        """True when no claim was rejected by the verifier — answer can
        ship without rewrites."""
        return not self.ungrounded


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
            neighbours = await store.neighbors(doc_id, limit=state.graph_expansion_budget)
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


def _build_synthetic_chunk(result: TableQueryResult) -> Chunk:
    """Build the single synthetic `Chunk` injected into `state.reranked` for a
    `TableQueryResult` (Table-RAG Phase 2, spec §3).

    The evidence — source/contributing rows as KV lines — comes FIRST and is
    bounded to `_SYNTHETIC_TEXT_MAX` so it survives the downstream truncate
    budgets; the SQL string is appended LAST (informational, clippable). The
    chunk_id suffix `sql0001` is 7 chars and contains the non-hex letters
    `s/q/l`, so `_repair_claim_chunk_ids` can't fuzzy-collide it with a real
    10-hex-char chunk hash.
    """
    header = result.header
    lines: list[str] = []
    if result.kind == "aggregate":
        op_col = result.sql
        # Lead with the validated aggregate framing, then the basis rows. The
        # value lives in THIS framing line, so the aggregate path is unaffected
        # by row-dropping — even if every basis row overflows, the answer (the
        # scalar) survives.
        agg_val = result.aggregate_value
        val_str = f"{agg_val:g}" if agg_val is not None else "n/a"
        lines.append(f"Aggregate result = {val_str} over {len(result.contributing_rows)} rows:")
        _ = op_col
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
        superlative_framing: str | None = None
        if result.superlative is not None:
            # Verified extremum framing — the returned row was independently
            # confirmed to hold the {highest|lowest} value of this column, so
            # the agent can attribute the superlative (the framing is grounded,
            # not an unchecked claim). Held back until we know the row fits.
            col_label, direction = result.superlative
            superlative_framing = f"Row with the {direction} {col_label} in this table:"

        # The framing line that WILL be prepended (superlative if verified, else
        # the neutral "Matching rows:"). Reserve room for it + its newline in
        # every row-budget check so the assembled `framing\nrows` total can't
        # exceed _SYNTHETIC_TEXT_MAX. The superlative framing is only used when
        # at least the first row survives (so an extremum claim always has a
        # value beneath it).
        neutral_framing = "Matching rows:"
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

            # Pick the first reranked doc that has stored tables NEAR the
            # retrieved chunks (proximity-scoped — passing all of a 74-table doc
            # overflows the SQL-gen prompt). A doc with tables only far from the
            # retrieved region is skipped, as is one with no stored tables.
            target_tables: list[StoredTable] = []
            target_title = ""
            any_doc_has_tables = False
            for doc_id in seen_docs:
                tables = await store.tables_for_document(doc_id)
                if not tables:
                    continue
                any_doc_has_tables = True
                near = _relevant_tables(tables, state.reranked, doc_id)
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


def _repair_claim_chunk_ids(
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
    messages = render_messages(
        "answer",
        query=state.query,
        chunks=state.reranked,
        feedback=feedback,
    )
    draft, tokens = await complete_structured(
        prompt=messages,
        schema=DraftAnswer,
        prompt_tag="answer@v3",
    )

    # Repair corrupted citation ids before they reach verify/compose.
    # The answer LLM sometimes mangles the long `docid#hash` ids it's
    # shown (bare hash, single-char flip); snapping them back to real
    # reranked chunk_ids keeps the audit trail + grounding check honest.
    if draft.claims and state.reranked:
        repaired, repair_stats = _repair_claim_chunk_ids(draft.claims, state.reranked)
        if repair_stats["suffix"] or repair_stats["fuzzy"] or repair_stats["unresolved"]:
            log.info("chunk_id_repair", **repair_stats)
        draft = draft.model_copy(update={"claims": repaired})

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

    final = FinalResponse(
        answered=True,
        summary=state.draft.summary,
        claims=surviving_claims,
        used_chunks=used_chunks,
        wikilinks=wikilinks,
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
    g.add_node("query_tables", query_tables)
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
