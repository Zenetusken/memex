"""The /ask threading of the code-only FTS term-query (Phase-3 Lever A).

`hybrid_search` / `hybrid_search_in_docs` must pass `term_query=True` to the BM25 arm ONLY when
the flag is on AND the query NAMES a code identifier — a prose query (or the kill-switch) keeps
the unchanged phrase-wrap. This pins the wiring without a GPU/store (the heavy I/O is faked, the
end-to-end recovery is the audit-13 production A/B).
"""

from __future__ import annotations

import asyncio

import pytest

from memex.core.config import MemexSettings, set_settings
from memex.retrieve import hybrid


class _FakeFTS:
    last_term: bool | None = None

    @classmethod
    async def open(cls, _path: object) -> _FakeFTS:
        return cls()

    async def search(self, query: str, *, k: int, term_query: bool = False) -> list:
        _FakeFTS.last_term = term_query
        return []

    async def search_in_docs(
        self, query: str, *, doc_ids: list[str], k: int, term_query: bool = False
    ) -> list:
        _FakeFTS.last_term = term_query
        return []

    async def close(self) -> None:
        pass


class _FakeVec:
    @classmethod
    async def open(cls, _path: object) -> _FakeVec:
        return cls()

    async def search(self, _emb: object, *, k: int) -> list:
        return []

    async def search_in_docs(self, _emb: object, *, doc_ids: list[str], k: int) -> list:
        return []

    async def close(self) -> None:
        pass


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hybrid, "FTSStore", _FakeFTS)
    monkeypatch.setattr(hybrid, "VectorStore", _FakeVec)

    async def _fake_embed(_q: str) -> list[float]:
        return [0.0, 0.0]

    monkeypatch.setattr(hybrid, "_embed_query", _fake_embed)


_CODE_Q = "Which function calls is_known_safe_command?"
_PROSE_Q = "What does Smith argue about reflexivity?"


def test_term_query_threaded_for_code_query_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch)
    monkeypatch.delenv("MEMEX_AGENTS__CODE_TERM_QUERY_ENABLED", raising=False)
    set_settings(MemexSettings())  # default ON
    asyncio.run(hybrid.hybrid_search(_CODE_Q))
    assert _FakeFTS.last_term is True
    asyncio.run(hybrid.hybrid_search_in_docs(_CODE_Q, ["doc1"], k=50))
    assert _FakeFTS.last_term is True


def test_phrase_wrap_for_prose_query_even_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch)
    set_settings(MemexSettings())  # default ON, but a prose query must NOT trigger
    asyncio.run(hybrid.hybrid_search(_PROSE_Q))
    assert _FakeFTS.last_term is False


def test_kill_switch_forces_phrase_wrap_for_code_query(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch)
    monkeypatch.setenv("MEMEX_AGENTS__CODE_TERM_QUERY_ENABLED", "false")
    set_settings(MemexSettings())
    asyncio.run(hybrid.hybrid_search(_CODE_Q))
    assert _FakeFTS.last_term is False
    monkeypatch.delenv("MEMEX_AGENTS__CODE_TERM_QUERY_ENABLED", raising=False)
    set_settings(MemexSettings())
