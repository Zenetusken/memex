"""The shared related-documents aggregation (`retrieve/related.py`) behind the /ask Related
panel + scope-set suggestions (webui) and the MCP/CLI `ask` payloads. Faked GraphStore (the
class-method patch the webui/MCP tests also use) — no live ryugraph needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from memex.core.types import RelatedDocument
from memex.retrieve import related_documents_for_answer, related_documents_for_seeds


class _FakeStore:
    @classmethod
    async def open(cls, vault_path: object) -> _FakeStore:
        return cls()

    async def related_documents(
        self, doc_id: str, *, limit: int = 10, max_entities: int = 8
    ) -> list[RelatedDocument]:
        if doc_id == "a":
            return [
                RelatedDocument(doc_id="c", title="C", score=3.0, shared_entities=["DNS"]),
                RelatedDocument(doc_id="b", title="B", score=2.0, shared_entities=["x"]),  # a seed → excluded
            ]
        if doc_id == "b":
            return [
                RelatedDocument(doc_id="c", title="C", score=4.5, shared_entities=["firewall"]),  # higher → wins
                RelatedDocument(doc_id="d", title="D", score=1.0, shared_entities=["y"]),
            ]
        return []

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_related_for_seeds_merges_dedups_excludes_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("memex.index.graph_store.GraphStore.open", staticmethod(_FakeStore.open))
    out = await related_documents_for_seeds(Path("/x"), ["a", "b"], out_limit=6)
    by_id = {r.doc_id: r for r in out}
    assert "b" not in by_id and "a" not in by_id  # seed docs excluded
    assert by_id["c"].score == 4.5  # dedup keeps the higher-score relation (b's, "firewall")
    assert by_id["c"].shared_entities == ["firewall"]
    assert "d" in by_id
    # re-ranked by score desc: c (4.5) before d (1.0)
    assert [r.doc_id for r in out] == ["c", "d"]


@pytest.mark.asyncio
async def test_related_for_answer_answered_only_and_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("memex.index.graph_store.GraphStore.open", staticmethod(_FakeStore.open))
    # answered: seeds from the distinct cited doc_ids (a appears twice), excludes the seeds.
    out = await related_documents_for_answer(Path("/x"), ["a", "a", "b"], answered=True)
    assert {r.doc_id for r in out} == {"c", "d"}
    # a refusal cited nothing groundable → empty (no graph query)
    assert await related_documents_for_answer(Path("/x"), ["a", "b"], answered=False) == []
    # answered but no cited docs → empty
    assert await related_documents_for_answer(Path("/x"), [], answered=True) == []


@pytest.mark.asyncio
async def test_related_fail_open_when_graph_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(vault_path: object) -> object:
        raise ImportError("ryugraph not installed")

    monkeypatch.setattr("memex.index.graph_store.GraphStore.open", staticmethod(_boom))
    assert await related_documents_for_seeds(Path("/x"), ["a"]) == []
    assert await related_documents_for_answer(Path("/x"), ["a"], answered=True) == []
