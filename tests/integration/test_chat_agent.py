"""Grounded multi-turn chat agent (`agents/chat.py`) + the additive carry-merge.

No GPU: `complete_structured` (rewrite/digest) and `answer_query` are faked, and the
`retrieve` node's prior-chunk carry is exercised with a faked hybrid_search + FTS store.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from memex.agents.answering import CitedClaim, FinalResponse
from memex.agents.chat import ConversationDigest, StandaloneQuery, answer_turn
from memex.core.config import MemexSettings, set_settings
from memex.core.conversation_store import ConversationStore
from memex.core.types import Chunk


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemexSettings]:
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    s = MemexSettings()
    set_settings(s)
    yield s
    set_settings(None)


def _final(summary: str, chunk_ids: list[str], *, answered: bool = True) -> FinalResponse:
    used = [Chunk(chunk_id=c, document_id="d", document_title="D", text="x") for c in chunk_ids]
    return FinalResponse(
        answered=answered,
        summary=summary if answered else None,
        refusal_reason=None if answered else "no grounded answer",
        claims=(
            [CitedClaim(claim=summary, source_chunk_id=chunk_ids[0], confidence="high")]
            if answered and chunk_ids
            else []
        ),
        used_chunks=used,
        correlation_id="cid",
        tokens_used=1,
        nodes_traversed=1,
        regenerate_attempts=0,
    )


class _FakeChatLLM:
    """Stand-in for `complete_structured` on the rewrite + digest calls."""

    def __init__(self) -> None:
        self.rewrites: list[StandaloneQuery] = []

    async def __call__(
        self, *, prompt: object, schema: type, **_kw: object
    ) -> tuple[Any, int]:
        if schema is StandaloneQuery:
            return self.rewrites.pop(0), 5
        if schema is ConversationDigest:
            return ConversationDigest(sentences=["digested"]), 5
        raise AssertionError(f"unexpected schema {schema!r}")


async def _new_conversation(settings: MemexSettings) -> str:
    store = await ConversationStore.open(settings.vault_path)
    convo = await store.create_conversation()
    await store.close()
    return convo.conversation_id


async def test_first_turn_no_carry_then_followup_carries(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str] | None] = []

    async def fake_answer_query(query: str, **kw: object) -> FinalResponse:
        captured.append(kw.get("prior_carry_chunk_ids"))  # type: ignore[arg-type]
        return _final(f"answer to {query}", ["d#a", "d#b"])

    llm = _FakeChatLLM()
    monkeypatch.setattr("memex.agents.chat.answer_query", fake_answer_query)
    monkeypatch.setattr("memex.agents.chat.complete_structured", llm)

    cid = await _new_conversation(settings)

    # Turn 1: no history → no rewrite, no carry.
    r1 = await answer_turn(cid, "What is X?")
    assert r1.is_followup is False
    assert r1.standalone_query == "What is X?"
    assert captured[-1] is None

    # Turn 2: a follow-up → rewrite resolves it; carry = turn 1's cited chunks.
    llm.rewrites.append(
        StandaloneQuery(standalone_query="what is the latency of X", is_followup=True)
    )
    r2 = await answer_turn(cid, "and its latency?")
    assert r2.is_followup is True
    assert r2.standalone_query == "what is the latency of X"
    assert captured[-1] == ["d#a", "d#b"]


async def test_rewrite_failure_falls_open_to_literal(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from memex.core.errors import ModelCallError

    seen: list[str] = []

    async def fake_answer_query(query: str, **kw: object) -> FinalResponse:
        seen.append(query)
        return _final(f"a:{query}", ["d#a"])

    async def boom(*, prompt: object, schema: type, **_kw: object) -> tuple[Any, int]:
        raise ModelCallError("rewrite down", context={})

    monkeypatch.setattr("memex.agents.chat.answer_query", fake_answer_query)
    monkeypatch.setattr("memex.agents.chat.complete_structured", boom)

    cid = await _new_conversation(settings)
    await answer_turn(cid, "first")
    # Second turn: rewrite raises → fall open to the literal message, non-followup.
    r2 = await answer_turn(cid, "follow up here")
    assert r2.is_followup is False
    assert r2.standalone_query == "follow up here"
    assert seen[-1] == "follow up here"


async def test_digest_compaction_after_window(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_answer_query(query: str, **kw: object) -> FinalResponse:
        return _final(f"a:{query}", ["d#a"])

    llm = _FakeChatLLM()
    monkeypatch.setattr("memex.agents.chat.answer_query", fake_answer_query)
    monkeypatch.setattr("memex.agents.chat.complete_structured", llm)

    cid = await _new_conversation(settings)
    await answer_turn(cid, "q0")
    for i in range(1, 5):  # turns 1..4 are follow-ups
        llm.rewrites.append(StandaloneQuery(standalone_query=f"q{i}", is_followup=True))
        await answer_turn(cid, f"q{i}")

    store = await ConversationStore.open(settings.vault_path)
    loaded = await store.load(cid)
    await store.close()
    assert loaded is not None
    assert loaded.turn_count == 5
    # turn_count (5) > _RECENT_TURNS (4) → the 5th append evicted turn 0 → digest fired.
    assert loaded.running_summary == "digested"


async def test_retrieve_node_merges_prior_carry(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from memex.agents.answering import AnswerState, retrieve
    from memex.index.fts_store import FTSStore

    fresh = Chunk(chunk_id="fresh#1", document_id="d", document_title="D", text="f")
    carried = Chunk(chunk_id="carry#1", document_id="d2", document_title="D2", text="c")

    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return [fresh]

    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)

    class _FakeFTS:
        async def chunks_by_ids(self, ids: list[str]) -> list[Chunk]:
            return [carried] if "carry#1" in ids else []

        async def close(self) -> None:
            pass

    async def _open(cls: type, vault_path: object) -> _FakeFTS:
        return _FakeFTS()

    monkeypatch.setattr(FTSStore, "open", classmethod(_open))

    # With a carry: both the fresh hit and the carried chunk are candidates.
    out = await retrieve(AnswerState(query="q", prior_carry_chunk_ids=["carry#1"]))
    assert {c.chunk_id for c in out["candidates"]} == {"fresh#1", "carry#1"}

    # Empty carry → byte-identical (only the fresh hit; the FTS store is never opened).
    out2 = await retrieve(AnswerState(query="q"))
    assert {c.chunk_id for c in out2["candidates"]} == {"fresh#1"}

    # Overlap: a carried id that is ALSO a fresh hit is deduped (kept once).
    out3 = await retrieve(AnswerState(query="q", prior_carry_chunk_ids=["fresh#1", "carry#1"]))
    ids3 = [c.chunk_id for c in out3["candidates"]]
    assert sorted(ids3) == ["carry#1", "fresh#1"]
    assert ids3.count("fresh#1") == 1


async def test_refusal_turn_does_not_carry_to_next_followup(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The safety invariant: a REFUSED turn must NOT carry chunks to the next follow-up
    (the carry seeds only from answered turns). Validates the `t.answered` filter."""
    captured: list[list[str] | None] = []

    async def fake_answer_query(query: str, **kw: object) -> FinalResponse:
        captured.append(kw.get("prior_carry_chunk_ids"))  # type: ignore[arg-type]
        if len(captured) == 1:
            # Turn 0 REFUSES, but its reranked pool (used_chunks) is non-empty.
            return _final("refused", ["d#a", "d#b"], answered=False)
        return _final(f"a:{query}", ["d#c"])

    llm = _FakeChatLLM()
    monkeypatch.setattr("memex.agents.chat.answer_query", fake_answer_query)
    monkeypatch.setattr("memex.agents.chat.complete_structured", llm)

    cid = await _new_conversation(settings)
    await answer_turn(cid, "first question")  # refuses
    llm.rewrites.append(StandaloneQuery(standalone_query="the follow-up", is_followup=True))
    await answer_turn(cid, "and the follow-up?")
    # The prior turn was a refusal → no answered turn to carry from → carry is empty.
    assert captured[-1] is None


async def test_digest_failure_keeps_running_summary_bounded(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A1: a string of digest failures must NOT let running_summary grow unbounded — the
    mechanical-truncate fallback keeps it capped (and non-empty) so the rewrite budget holds."""
    from memex.core.errors import ModelCallError

    async def fake_answer_query(query: str, **kw: object) -> FinalResponse:
        # A long summary (FinalResponse.summary is unbounded; the per-turn answer_summary
        # is capped to 600 at persist) with no claims (CitedClaim.claim caps at 300).
        return FinalResponse(
            answered=True,
            summary="S" * 600,
            claims=[],
            used_chunks=[Chunk(chunk_id="d#a", document_id="d", document_title="D", text="x")],
            correlation_id="cid",
            tokens_used=1,
            nodes_traversed=1,
            regenerate_attempts=0,
        )

    class _DigestAlwaysFails:
        async def __call__(self, *, prompt: object, schema: type, **_kw: object) -> tuple[Any, int]:
            if schema is StandaloneQuery:
                return StandaloneQuery(standalone_query="q", is_followup=True), 5
            raise ModelCallError("digest down", context={})

    monkeypatch.setattr("memex.agents.chat.answer_query", fake_answer_query)
    monkeypatch.setattr("memex.agents.chat.complete_structured", _DigestAlwaysFails())

    cid = await _new_conversation(settings)
    for i in range(8):  # >4 turns → compaction fires repeatedly; the digest always fails
        await answer_turn(cid, f"q{i}")

    store = await ConversationStore.open(settings.vault_path)
    convo = await store.load(cid)
    await store.close()
    assert convo is not None
    assert convo.turn_count == 8
    from memex.agents.chat import _RUNNING_SUMMARY_MAX

    assert 0 < len(convo.running_summary) <= _RUNNING_SUMMARY_MAX


async def test_run_chat_eval_scores_rewrite_recall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from memex.agents.chat import StandaloneQuery
    from memex.eval.runner import run_chat_eval

    async def fake_rewrite(conversation: Any, user_text: str) -> StandaloneQuery:
        return StandaloneQuery(
            standalone_query=f"rewritten: {user_text}", is_followup=bool(conversation.turns)
        )

    async def fake_hybrid(query: str, k: int = 50) -> list[Chunk]:
        cid = "gold#1" if "timers" in query else "other#1"
        return [Chunk(chunk_id=cid, document_id="d", document_title="D", text="x")]

    monkeypatch.setattr("memex.agents.chat.rewrite_query", fake_rewrite)
    monkeypatch.setattr("memex.retrieve.hybrid_search", fake_hybrid)

    qs = tmp_path / "q.json"
    qs.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "name": "hit",
                        "history": [{"user_text": "STP timers", "answer_summary": "x"}],
                        "follow_up": "the timers?",
                        "relevant_chunk_ids": ["gold#1"],
                    },
                    {
                        "name": "miss",
                        "history": [],
                        "follow_up": "something else",
                        "relevant_chunk_ids": ["gold#2"],
                    },
                ]
            }
        )
    )
    report = await run_chat_eval(qs)
    assert report.case_count == 2
    by_name = {r.name: r for r in report.per_case}
    assert by_name["hit"].recall == 1.0
    assert by_name["hit"].is_followup is True  # had history
    assert by_name["miss"].recall == 0.0
    assert by_name["miss"].is_followup is False  # no history
    assert report.mean_recall == 0.5
