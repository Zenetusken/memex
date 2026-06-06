"""Grounded multi-turn agentic chat — Surface A (`docs/specs/grounded-agentic-chat.md`).

A conversational, multi-turn version of `/ask` that STAYS grounded. Every answer
turn runs the unchanged `answer_query` graph (so `refusal_cf=1.0` / 0-hallucination
apply per turn); this module adds only memory + the bounded prior-chunk carry on top.

The per-turn flow (`answer_turn`):
  1. load the conversation (sqlite sidecar);
  2. **query rewrite** (control layer) — a small guided-JSON call rewrites a
     follow-up into a self-contained query using a COMPACT history (deterministic,
     reasoning suppressed by the grammar); fail-open to the literal message;
  3. **bounded prior-chunk carry** — the immediately-prior answered turn's cited
     chunk_ids (≤5) re-admitted as extra rerank candidates (relevance without
     accumulation; a carried chunk only grounds if it survives rerank);
  4. answer through the UNCHANGED `answer_query` graph;
  5. persist the turn, then compact the oldest turn into the running summary.

The reasoning is confined to the control-layer rewrite; grounding is safe across
turns by construction (`answer_query`'s `verify` only ever sees the current turn's
reranked chunks). This module NEVER imports a free-text/reasoning primitive and
NEVER sets `enable_thinking` — that is fenced to Surface B (ADR-0013).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

import structlog
from pydantic import BaseModel, Field

from memex.agents.answering import FinalResponse, answer_query
from memex.core.config import get_settings
from memex.core.conversation_store import ConversationStore
from memex.core.errors import ConfigurationError, ModelCallError
from memex.core.types import Conversation, ConversationTurn
from memex.models.client import complete_structured
from memex.prompts import prompt_tag_for, render_messages

logger = structlog.get_logger(__name__)

# How many recent turns are kept VERBATIM in the rewrite context (older turns fold
# into the running summary). The carry seeds from the immediately-prior answered turn.
_RECENT_TURNS = 4
_PRIOR_CARRY_MAX = 5  # bounded prior-chunk carry — the immediately-prior answered turn only
_RUNNING_SUMMARY_MAX = 1200  # the compacted history cap (chars)
_ANSWER_SUMMARY_MAX = 600  # stored per-turn answer headline (rewrite context only)


class StandaloneQuery(BaseModel):
    """The control-layer query-rewrite output. Bounded per the `max_length` mandate so
    xgrammar enforces the cap (the rewrite is a guided-JSON call → deterministic, the
    4B's CoT suppressed by the grammar). A mis-rewrite degrades to a grounded-but-non-
    responsive answer → the existing relevance gate refuses — never a hallucination."""

    standalone_query: str = Field(
        description="The user's latest message rewritten as a fully self-contained query.",
        max_length=400,
    )
    is_followup: bool = Field(
        description="True if the message depended on prior turns to be understood.",
    )
    referents_resolved: list[Annotated[str, Field(max_length=80)]] = Field(
        default_factory=list,
        description="Each resolved reference as '<token> -> <expansion>', for audit.",
        max_length=6,
    )


class ConversationDigest(BaseModel):
    """The running-summary compaction output — a bounded list of short sentences (the
    summarizer's natural-boundaries-bound-the-count idiom). Conversation metadata, never
    an answer, so no grounding gate applies."""

    sentences: list[Annotated[str, Field(max_length=200)]] = Field(
        default_factory=list,
        description="The compacted conversation history as 1-6 short sentences.",
        max_length=6,
    )


class ChatTurnResult(BaseModel):
    """One completed chat turn: the grounded `FinalResponse`, the persisted turn record,
    and the resolved standalone query (surfaced so the user can see what was searched)."""

    response: FinalResponse
    turn: ConversationTurn
    standalone_query: str
    is_followup: bool


async def rewrite_query(conversation: Conversation, user_text: str) -> StandaloneQuery:
    """The control-layer query rewrite — a deterministic guided-JSON call that resolves a
    follow-up's references into a self-contained query using the compact history.

    Public so the `eval-chat` harness can score retrieval on `.standalone_query`. Raises
    `ModelCallError` on a model failure (the chat loop wraps it fail-open; the eval surfaces it).
    """
    recent = conversation.turns[-_RECENT_TURNS:]
    prompt = render_messages(
        "rewrite_followup",
        running_summary=conversation.running_summary,
        recent_turns=recent,
        user_text=user_text,
    )
    result, _tokens = await complete_structured(
        prompt=prompt,
        schema=StandaloneQuery,
        prompt_tag=prompt_tag_for("rewrite_followup"),
    )
    return result


async def _rewrite_followup(convo: Conversation, user_text: str) -> tuple[str, bool]:
    """Rewrite `user_text` into `(standalone_query, is_followup)`. Fail-open: a model error
    (or an empty rewrite) falls back to the literal message as a non-follow-up — a failed
    rewrite never blocks the answer, it only loses reference resolution for this turn."""
    try:
        result = await rewrite_query(convo, user_text)
    except ModelCallError:
        logger.warning("chat.rewrite_failed", fallback="literal_message")
        return user_text, False
    standalone = result.standalone_query.strip() or user_text
    return standalone, result.is_followup


async def _digest(prior_summary: str, turns_to_fold: list[ConversationTurn]) -> str:
    """Fold `turns_to_fold` into `prior_summary` → a new running summary (capped).

    Fail-open: a model error keeps the prior summary unchanged (history compaction is
    best-effort metadata, never load-bearing for grounding)."""
    prompt = render_messages(
        "conversation_digest",
        prior_summary=prior_summary,
        turns_to_fold=turns_to_fold,
    )
    try:
        result, _tokens = await complete_structured(
            prompt=prompt,
            schema=ConversationDigest,
            prompt_tag=prompt_tag_for("conversation_digest"),
        )
    except ModelCallError:
        # Fail-open WITHOUT bloating: a string of digest failures must not let the
        # running summary grow unbounded (it feeds the rewrite prompt and eats its
        # budget). Mechanically fold the evicted turns' headlines and re-cap, so the
        # window stays bounded and the evicted context isn't silently lost.
        logger.warning("chat.digest_failed", fallback="mechanical_truncate")
        folded = " ".join(t.answer_summary for t in turns_to_fold if t.answer_summary)
        return f"{prior_summary} {folded}".strip()[:_RUNNING_SUMMARY_MAX] or prior_summary
    joined = " ".join(s.strip() for s in result.sentences if s.strip()).strip()
    return joined[:_RUNNING_SUMMARY_MAX] or prior_summary


async def _maybe_compact(store: ConversationStore, conversation_id: str) -> None:
    """After an append, fold the single turn that just left the verbatim window into the
    running summary — incremental, so the digest input stays bounded (prior summary + 1
    turn). No-op until the conversation exceeds the recent window."""
    convo = await store.load(conversation_id)
    if convo is None or convo.turn_count <= _RECENT_TURNS:
        return
    # Exactly one turn just fell out of the last-_RECENT_TURNS window.
    evicted_index = convo.turn_count - _RECENT_TURNS - 1
    if evicted_index < 0 or evicted_index >= len(convo.turns):
        return
    evicted = convo.turns[evicted_index]
    new_summary = await _digest(convo.running_summary, [evicted])
    if new_summary != convo.running_summary:
        await store.update_running_summary(conversation_id, new_summary)


async def answer_turn(
    conversation_id: str,
    user_text: str,
    *,
    scope_doc_ids: list[str] | None = None,
    correlation_id: str | None = None,
    on_node: Callable[[str], None] | None = None,
) -> ChatTurnResult:
    """Answer one grounded chat turn in an existing conversation.

    The conversation must already exist (the surface creates it via
    `ConversationStore.create_conversation` and passes its id). Opens the store lazily
    from `settings.vault_path` so it is surface-agnostic (identical from CLI + webui).

    `scope_doc_ids` (optional) overrides the conversation-level scope pin for this turn;
    `None` uses the conversation's stored `scope_doc_ids`. `correlation_id` + `on_node`
    thread straight into `answer_query` (the webui's progress hook).
    """
    log = logger.bind(node="chat.answer_turn", conversation_id=conversation_id)
    settings = get_settings()
    store = await ConversationStore.open(settings.vault_path)
    try:
        convo = await store.load(conversation_id)
        if convo is None:
            raise ConfigurationError(
                "no such conversation",
                context={"conversation_id": conversation_id},
            )

        effective_scope = scope_doc_ids if scope_doc_ids is not None else convo.scope_doc_ids

        # 1-2. Query rewrite (only with history) + the bounded prior-chunk carry.
        if convo.turns:
            standalone_query, is_followup = await _rewrite_followup(convo, user_text)
        else:
            standalone_query, is_followup = user_text, False

        carry_ids: list[str] = []
        if is_followup:
            answered_turns = [t for t in convo.turns if t.answered]
            if answered_turns:
                carry_ids = answered_turns[-1].cited_chunk_ids[:_PRIOR_CARRY_MAX]

        log.info(
            "turn.start",
            is_followup=is_followup,
            carry=len(carry_ids),
            scoped=len(effective_scope),
        )

        # 3-4. Answer through the UNCHANGED grounded graph.
        response = await answer_query(
            standalone_query,
            scope_doc_ids=effective_scope or None,
            # `[] or None` → None; `retrieve`'s `if state.prior_carry_chunk_ids:` treats
            # both as no-carry, so an empty carry is byte-identical to a bare /ask.
            prior_carry_chunk_ids=carry_ids or None,
            correlation_id=correlation_id,
            on_node=on_node,
        )

        # 5. Persist the turn, then compact the oldest out-of-window turn.
        summary_text = (response.summary or response.refusal_reason or "")[:_ANSWER_SUMMARY_MAX]
        turn = await store.append_turn(
            conversation_id,
            user_text=user_text,
            standalone_query=standalone_query,
            is_followup=is_followup,
            answered=response.answered,
            answer_summary=summary_text,
            cited_chunk_ids=[c.chunk_id for c in response.used_chunks],
            response_json=response.model_dump_json(),
            correlation_id=response.correlation_id,
        )
        await _maybe_compact(store, conversation_id)

        log.info("turn.done", answered=response.answered, turn_index=turn.turn_index)
        return ChatTurnResult(
            response=response,
            turn=turn,
            standalone_query=standalone_query,
            is_followup=is_followup,
        )
    finally:
        await store.close()
