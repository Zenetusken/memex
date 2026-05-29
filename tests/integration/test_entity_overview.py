"""Integration test for `retrieve.entity.entity_overview` — the graph+FTS orchestrator
behind entity-centric retrieval (ADR-0011).

Fakes `GraphStore.open` + `FTSStore.open` (no ryugraph, no SQLite) to pin the three paths:
resolved entity → passages SCOPED to the mentioning docs; unknown entity → whole-corpus
FTS fallback; graph unavailable (ImportError) → fail-open to the same fallback, never raises.
The real-graph Cypher is covered by test_entity_profile.py (opt-in real ryugraph).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from memex.core.config import MemexSettings, set_settings
from memex.core.types import Chunk
from memex.index.graph_store import EntityMention, EntityProfile, EntitySuggestion


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemexSettings]:
    (tmp_path / "documents").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    s = MemexSettings()  # type: ignore[call-arg]
    set_settings(s)
    yield s
    set_settings(None)


def _chunk(cid: str, doc: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, document_id=doc, document_title=doc.title(), text=text)


class _FakeFTS:
    """Records which search path was taken + returns canned chunks."""

    def __init__(self) -> None:
        self.scoped_doc_ids: list[str] | None = None
        self.corpus_query: str | None = None

    @classmethod
    async def open(cls, vault_path: Any) -> _FakeFTS:
        return cls()

    async def search_in_docs(self, query: str, *, doc_ids: list[str], k: int) -> list[Chunk]:
        self.scoped_doc_ids = doc_ids
        return [_chunk("d1#a", "d1", f"scoped passage about {query}")]

    async def search(self, query: str, *, k: int) -> list[Chunk]:
        self.corpus_query = query
        return [_chunk("dx#a", "dx", f"corpus passage about {query}")]

    async def close(self) -> None:
        return None


def _install_fts(monkeypatch: pytest.MonkeyPatch) -> _FakeFTS:
    """Patch FTSStore.open to return ONE shared fake we can inspect afterwards."""
    fake = _FakeFTS()
    monkeypatch.setattr("memex.retrieve.entity.FTSStore.open", staticmethod(lambda vp: _ready(fake)))
    return fake


async def _ready(value: Any) -> Any:
    return value


@pytest.mark.asyncio
async def test_resolved_entity_scopes_passages_to_mention_docs(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = EntityProfile(
        query_name="STP",
        matched_names=["STP"],
        kinds=["concept"],
        doc_count=2,
        mentions=[EntityMention(doc_id="d1", title="Doc 1"), EntityMention(doc_id="d2", title="Doc 2")],
        cooccurring=[],
        resolved=True,
    )

    class _FakeGraph:
        @classmethod
        async def open(cls, vault_path: Any) -> _FakeGraph:
            return cls()

        async def entity_profile(self, name: str, *, max_docs: int, max_cooccurring: int) -> EntityProfile:
            return profile

        async def close(self) -> None:
            return None

    monkeypatch.setattr("memex.index.graph_store.GraphStore.open", staticmethod(_FakeGraph.open))
    fake_fts = _install_fts(monkeypatch)

    from memex.retrieve import entity_overview

    overview = await entity_overview("STP")
    assert overview.profile.resolved is True
    assert overview.passages_scoped is True
    assert fake_fts.scoped_doc_ids == ["d1", "d2"]  # scoped to the mentioning docs
    assert fake_fts.corpus_query is None  # the corpus fallback was NOT used
    assert overview.passages


@pytest.mark.asyncio
async def test_suggestions_survive_orchestrator_and_passages_stay_scoped(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resolved profile carrying `suggestions` (the acronym↔expansion bridge) passes
    through the orchestrator untouched, AND the FTS passages stay scoped to the EXACT
    mention docs — a suggestion is a link, never a passage-scope-widener."""
    profile = EntityProfile(
        query_name="DNS",
        matched_names=["DNS"],
        kinds=["concept"],
        doc_count=1,
        mentions=[EntityMention(doc_id="d1", title="Doc 1")],
        cooccurring=[],
        resolved=True,
        suggestions=[
            EntitySuggestion(name="Domain Name System", kind="concept", doc_count=3, relation="expansion")
        ],
    )

    class _FakeGraph:
        @classmethod
        async def open(cls, vault_path: Any) -> _FakeGraph:
            return cls()

        async def entity_profile(self, name: str, *, max_docs: int, max_cooccurring: int) -> EntityProfile:
            return profile

        async def close(self) -> None:
            return None

    monkeypatch.setattr("memex.index.graph_store.GraphStore.open", staticmethod(_FakeGraph.open))
    fake_fts = _install_fts(monkeypatch)

    from memex.retrieve import entity_overview

    overview = await entity_overview("DNS")
    assert [s.name for s in overview.profile.suggestions] == ["Domain Name System"]
    assert overview.passages_scoped is True
    assert fake_fts.scoped_doc_ids == ["d1"]  # scoped to the EXACT mention, not the suggestion's docs
    assert fake_fts.corpus_query is None


@pytest.mark.asyncio
async def test_unknown_entity_falls_back_to_whole_corpus(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    unresolved = EntityProfile(
        query_name="nope", matched_names=[], kinds=[], doc_count=0,
        mentions=[], cooccurring=[], resolved=False,
    )

    class _FakeGraph:
        @classmethod
        async def open(cls, vault_path: Any) -> _FakeGraph:
            return cls()

        async def entity_profile(self, name: str, *, max_docs: int, max_cooccurring: int) -> EntityProfile:
            return unresolved

        async def close(self) -> None:
            return None

    monkeypatch.setattr("memex.index.graph_store.GraphStore.open", staticmethod(_FakeGraph.open))
    fake_fts = _install_fts(monkeypatch)

    from memex.retrieve import entity_overview

    overview = await entity_overview("nope")
    assert overview.profile.resolved is False
    assert overview.passages_scoped is False
    assert fake_fts.corpus_query == "nope"  # whole-corpus fallback
    assert fake_fts.scoped_doc_ids is None


@pytest.mark.asyncio
async def test_unresolved_with_suggestions_uses_corpus_fallback(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An UNRESOLVED name that still carries a bridge ('Did you mean?') goes through the
    whole-corpus FTS fallback (not scoped), and the suggestions ride through untouched."""
    unresolved = EntityProfile(
        query_name="DNS", matched_names=[], kinds=[], doc_count=0,
        mentions=[], cooccurring=[], resolved=False,
        suggestions=[
            EntitySuggestion(name="Domain Name System", kind="concept", doc_count=3, relation="acronym")
        ],
    )

    class _FakeGraph:
        @classmethod
        async def open(cls, vault_path: Any) -> _FakeGraph:
            return cls()

        async def entity_profile(self, name: str, *, max_docs: int, max_cooccurring: int) -> EntityProfile:
            return unresolved

        async def close(self) -> None:
            return None

    monkeypatch.setattr("memex.index.graph_store.GraphStore.open", staticmethod(_FakeGraph.open))
    fake_fts = _install_fts(monkeypatch)

    from memex.retrieve import entity_overview

    overview = await entity_overview("DNS")
    assert overview.profile.resolved is False
    assert overview.passages_scoped is False
    assert fake_fts.corpus_query == "DNS"  # whole-corpus fallback
    assert fake_fts.scoped_doc_ids is None
    assert [s.name for s in overview.profile.suggestions] == ["Domain Name System"]


@pytest.mark.asyncio
async def test_graph_unavailable_fails_open(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ImportError from GraphStore.open (ryugraph absent) → corpus FTS fallback, no raise."""

    def _boom(vault_path: Any) -> Any:
        raise ImportError("ryugraph not installed")

    monkeypatch.setattr("memex.index.graph_store.GraphStore.open", staticmethod(_boom))
    fake_fts = _install_fts(monkeypatch)

    from memex.retrieve import entity_overview

    overview = await entity_overview("STP")
    assert overview.profile.resolved is False  # uniform stand-in
    assert overview.passages_scoped is False
    assert overview.profile.suggestions == []  # no graph ⇒ no bridge, never a stray suggestion
    assert fake_fts.corpus_query == "STP"
    assert overview.passages  # still got passages
