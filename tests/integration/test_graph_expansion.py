"""Citation-graph reasoning during answering — v1.x regression net.

The agent's `expand_graph` node augments the retrieval candidate pool
with chunks from documents the graph store says are related (one-hop
CITES + entity-shared neighbours). These tests cover three branches:

1. **Default skip** — when the graph store isn't reachable (the
   in-tree default for tests, since ryugraph is an optional dep) the
   node is a no-op and the agent answers normally from flat retrieval
   only.
2. **Disabled by flag** — `graph_expansion_enabled=False` skips
   without touching the graph store at all.
3. **Real expansion** — when a graph store is mocked in, the node
   pulls neighbour chunks via `hybrid_search_in_docs` and the
   reranker sees the augmented pool. The agent's answer can cite
   chunks from a neighbour document that flat retrieval missed.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from memex.agents.answering import (
    AnswerState,
    Chunk,
    CitedClaim,
    DraftAnswer,
    RelevanceAssessment,
    SufficiencyAssessment,
    VerificationResult,
    answer_query,
    reset_compiled_graph,
)


@pytest.fixture(autouse=True)
def _fresh_graph() -> Iterator[None]:
    reset_compiled_graph()
    yield
    reset_compiled_graph()


# ----- Shared fake LLM (canned by schema) -----


def _make_fake_llm(
    *,
    sufficient: bool = True,
    cited_chunk_id: str = "primary#a",
    summary: str = "Summary text.",
) -> Any:
    """A `complete_structured` stand-in that routes by schema.

    Tests use the returned function with `monkeypatch.setattr(
    "memex.agents.answering.complete_structured", fake)`.
    """

    async def _fake(*, prompt: str, schema: type, **_kw: Any) -> tuple[Any, int]:
        if schema is SufficiencyAssessment:
            return (
                SufficiencyAssessment(
                    sufficient=sufficient,
                    reason="ok" if sufficient else "not enough material",
                ),
                5,
            )
        if schema is DraftAnswer:
            return (
                DraftAnswer(
                    summary=summary,
                    claims=[
                        CitedClaim(
                            claim="A grounded claim.",
                            source_chunk_id=cited_chunk_id,
                            confidence="high",
                        )
                    ],
                ),
                30,
            )
        if schema is VerificationResult or schema.__name__ == VerificationResult.__name__:
            return VerificationResult(grounded=[0], ungrounded=[]), 10
        if schema.__name__ == "RelevanceAssessment":
            return RelevanceAssessment(responsive=True, reason="ok"), 6
        raise AssertionError(f"unexpected schema {schema}")

    return _fake


def _make_fake_render() -> Any:
    return lambda name, **_kw: f"[fake {name}]"


# ----- Branch 1: no graph available → no-op, agent still answers -----


@pytest.mark.asyncio
async def test_no_graph_available_falls_back_to_flat_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the graph store can't be opened (ImportError because
    ryugraph isn't installed, or any other failure), `expand_graph`
    swallows the error and the agent runs flat retrieval only."""

    candidates = [
        Chunk(
            chunk_id="primary#a",
            document_id="primary",
            document_title="Primary",
            text="Primary document body.",
            score=0.9,
        )
    ]

    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return list(candidates)

    async def _hybrid_in_docs(query: str, doc_ids: list[str], *, k: int) -> list[Chunk]:
        raise AssertionError("expand_graph should not call this when graph is unavailable")

    async def _rerank(query: str, cands: list[Chunk], top_k: int = 10) -> list[Chunk]:
        return list(cands[:top_k])

    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.hybrid_search_in_docs", _hybrid_in_docs)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    monkeypatch.setattr("memex.agents.answering.render_prompt", _make_fake_render())
    monkeypatch.setattr("memex.agents.answering.complete_structured", _make_fake_llm())

    response = await answer_query("What does the primary doc say?")

    assert response.answered is True
    assert len(response.claims) == 1
    assert response.claims[0].source_chunk_id == "primary#a"


# ----- Branch 2: explicit disable via state flag -----


@pytest.mark.asyncio
async def test_graph_expansion_disabled_skips_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`answer_query(graph_expansion_enabled=False)` must not touch the
    graph store at all — even the import path."""
    candidates = [
        Chunk(
            chunk_id="primary#a",
            document_id="primary",
            document_title="Primary",
            text="Primary document body.",
        )
    ]

    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return list(candidates)

    async def _hybrid_in_docs(query: str, doc_ids: list[str], *, k: int) -> list[Chunk]:
        raise AssertionError("disabled — should not be called")

    async def _rerank(query, cands, top_k=10):  # type: ignore[no-untyped-def]
        return list(cands[:top_k])

    def _explode_graph(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("disabled — graph store should not be opened")

    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.hybrid_search_in_docs", _hybrid_in_docs)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    monkeypatch.setattr("memex.agents.answering.render_prompt", _make_fake_render())
    monkeypatch.setattr("memex.agents.answering.complete_structured", _make_fake_llm())
    # If `expand_graph` reaches the graph-store import, this will raise.
    monkeypatch.setattr("memex.index.graph_store.GraphStore.open", _explode_graph)

    response = await answer_query(
        "What does the primary doc say?",
        graph_expansion_enabled=False,
    )
    assert response.answered is True


# ----- Branch 3: real expansion pulls neighbour chunks -----


class _FakeGraphStore:
    """In-memory neighbours table. Replaces `GraphStore.open`."""

    def __init__(self, neighbours: dict[str, list[Any]]) -> None:
        self._neighbours = neighbours
        self.queries: list[tuple[str, int]] = []
        self.closed = False

    @classmethod
    def make_opener(cls, neighbours: dict[str, list[Any]]):
        instance = cls(neighbours)

        async def _open(_vault_path: Any) -> _FakeGraphStore:
            return instance

        return _open, instance

    async def neighbors(self, doc_id: str, *, limit: int = 50):
        self.queries.append((doc_id, limit))
        return self._neighbours.get(doc_id, [])

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_expand_graph_pulls_chunks_from_neighbour_docs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Flat retrieval finds `primary#a` (a chunk in 'primary'). The
    graph store says 'primary' cites 'cited'. `expand_graph` must
    fetch chunks from 'cited' via `hybrid_search_in_docs` and append
    them to the candidate pool. The agent's answer can then cite a
    chunk from 'cited'."""
    from memex.core.config import MemexSettings, set_settings
    from memex.index.graph_store import GraphNeighbor

    # `expand_graph` calls `get_settings().vault_path`. Stand up a
    # minimal settings to make that call succeed. graph_expansion is OFF by
    # default (2026-05-28 audit — see config), so this test (which exercises
    # the expansion mechanism) opts it back IN explicitly.
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("MEMEX_AGENTS__GRAPH_EXPANSION_ENABLED", "true")
    settings = MemexSettings()  # type: ignore[call-arg]
    set_settings(settings)

    # The initial retrieval surfaces a chunk from 'primary'.
    primary_chunk = Chunk(
        chunk_id="primary#a",
        document_id="primary",
        document_title="Primary",
        text="Primary content; cites Smith 2024.",
    )

    # 'cited' is the document the graph says 'primary' cites. Flat
    # retrieval doesn't surface it (returns empty when asked filtered
    # to cited.doc_id — wait, we mock _in_docs explicitly).
    cited_chunk = Chunk(
        chunk_id="cited#k",
        document_id="cited",
        document_title="Smith 2024",
        text="Smith's argument about reflexivity in research design.",
    )

    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return [primary_chunk]

    in_docs_calls: list[tuple[str, list[str], int]] = []

    async def _hybrid_in_docs(query: str, doc_ids: list[str], *, k: int) -> list[Chunk]:
        in_docs_calls.append((query, list(doc_ids), k))
        if "cited" in doc_ids:
            return [cited_chunk]
        return []

    async def _rerank(query, cands, top_k=10):  # type: ignore[no-untyped-def]
        # Pass-through; preserve order so `cited#k` is visible to assess.
        return list(cands[:top_k])

    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.hybrid_search_in_docs", _hybrid_in_docs)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    monkeypatch.setattr("memex.agents.answering.render_prompt", _make_fake_render())
    # The fake LLM cites `cited#k` — only possible if expand_graph
    # actually surfaced it into the pool.
    monkeypatch.setattr(
        "memex.agents.answering.complete_structured",
        _make_fake_llm(cited_chunk_id="cited#k"),
    )

    # Wire the fake GraphStore.open so `expand_graph` returns a
    # neighbour entry pointing at the 'cited' doc.
    fake_open, fake_store = _FakeGraphStore.make_opener(
        {
            "primary": [
                GraphNeighbor(
                    doc_id="cited",
                    title="Smith 2024",
                    relation="shares_entity",
                    via="reflexivity",
                )
            ]
        }
    )
    monkeypatch.setattr("memex.index.graph_store.GraphStore.open", fake_open)

    try:
        response = await answer_query("What does Smith argue about reflexivity?")
    finally:
        set_settings(None)

    # The agent's final answer cites the chunk from 'cited' — which
    # would have been impossible without graph expansion.
    assert response.answered is True
    assert len(response.claims) == 1
    assert response.claims[0].source_chunk_id == "cited#k"

    # The graph store was queried for 'primary's neighbours.
    assert fake_store.queries
    assert fake_store.queries[0][0] == "primary"

    # `expand_graph` called `hybrid_search_in_docs` with the new doc
    # whitelist.
    assert in_docs_calls
    assert "cited" in in_docs_calls[0][1]


@pytest.mark.asyncio
async def test_expand_graph_dedups_neighbour_already_in_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """If the graph names a neighbour that's already in the candidate
    pool (lexical retrieval already found it), `expand_graph` must
    NOT re-fetch it — we don't want duplicate chunks dominating the
    reranker."""
    from memex.core.config import MemexSettings, set_settings
    from memex.index.graph_store import GraphNeighbor

    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    settings = MemexSettings()  # type: ignore[call-arg]
    set_settings(settings)

    primary_chunk = Chunk(
        chunk_id="primary#a",
        document_id="primary",
        document_title="Primary",
        text="content one",
    )
    cited_chunk_already = Chunk(
        chunk_id="cited#k",
        document_id="cited",
        document_title="Cited (already in pool)",
        text="content two",
    )

    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return [primary_chunk, cited_chunk_already]

    in_docs_calls: list[Any] = []

    async def _hybrid_in_docs(query, doc_ids, *, k):  # type: ignore[no-untyped-def]
        in_docs_calls.append(list(doc_ids))
        return []

    async def _rerank(query, cands, top_k=10):  # type: ignore[no-untyped-def]
        return list(cands[:top_k])

    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.hybrid_search_in_docs", _hybrid_in_docs)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    monkeypatch.setattr("memex.agents.answering.render_prompt", _make_fake_render())
    monkeypatch.setattr(
        "memex.agents.answering.complete_structured",
        _make_fake_llm(cited_chunk_id="primary#a"),
    )

    fake_open, _store = _FakeGraphStore.make_opener(
        {
            "primary": [
                GraphNeighbor(
                    doc_id="cited",
                    title="Cited",
                    relation="cites",
                    via=None,
                )
            ],
            "cited": [
                GraphNeighbor(
                    doc_id="primary",
                    title="Primary",
                    relation="cites",
                    via=None,
                )
            ],
        }
    )
    monkeypatch.setattr("memex.index.graph_store.GraphStore.open", fake_open)

    try:
        response = await answer_query("anything")
    finally:
        set_settings(None)

    assert response.answered is True
    # Both 'primary' and 'cited' are already in `state.candidates`, so
    # the dedup logic should drop them and not call `hybrid_search_in_docs`
    # at all (the only new doc ids list would be empty).
    for call_doc_ids in in_docs_calls:
        # If the function got called, its doc_ids list must be empty —
        # but the implementation skips the call entirely when there are
        # no new doc ids, so this list should be empty (no calls).
        assert call_doc_ids == [], f"unexpected refetch of {call_doc_ids}"
    assert in_docs_calls == []  # cleanest assertion: not called at all


# ----- Empty candidates short-circuit -----


@pytest.mark.asyncio
async def test_expand_graph_skips_when_retrieval_returned_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No retrieval candidates → no source docs to expand from →
    no-op. The agent then short-circuits at `assess` (empty reranked)
    and refuses politely."""

    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return []

    async def _hybrid_in_docs(query, doc_ids, *, k):  # type: ignore[no-untyped-def]
        raise AssertionError("should not be called on empty retrieval")

    async def _rerank(query, cands, top_k=10):  # type: ignore[no-untyped-def]
        return list(cands)

    def _explode_open(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("graph store should not be opened on empty candidates")

    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.hybrid_search_in_docs", _hybrid_in_docs)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    monkeypatch.setattr("memex.agents.answering.render_prompt", _make_fake_render())
    monkeypatch.setattr("memex.agents.answering.complete_structured", _make_fake_llm())
    monkeypatch.setattr("memex.index.graph_store.GraphStore.open", _explode_open)

    response = await answer_query("an unanswerable question")
    # The assess node's short-circuit fires and produces an
    # insufficiency-based refusal.
    assert response.answered is False
    assert "No relevant content" in (response.refusal_reason or "")


# ----- AnswerState defaults -----


def test_answer_state_graph_expansion_defaults() -> None:
    """The defaults are sensible for the reference rig."""
    s = AnswerState(query="hello")
    assert s.graph_expansion_enabled is True
    assert s.graph_expansion_budget == 3
    assert s.chunks_per_neighbor == 2
    assert s.graph_expanded_doc_ids == []
