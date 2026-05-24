"""Retitle — metadata-only title fan-out, no re-embed.

`index.retitle_document` changes a document's title everywhere it is
denormalized (frontmatter + FTS + vector + graph) WITHOUT re-chunking
or re-embedding. The title isn't part of the embedded text or the
chunk id, so the vectors and chunk_ids are invariant under a rename.

These tests pin the two properties that make the feature worth having
over a force-reindex: (1) the embedder is never called, and (2) the
chunk_ids are unchanged. LanceDB + sentence-transformers are faked;
the vault + manifest are real (tmp dirs).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from memex.core.config import MemexSettings, set_settings
from memex.core.types import Chunk
from memex.ingest.pipeline import ingest_markdown_passthrough
from memex.vault.store import read_document


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemexSettings]:
    (tmp_path / "documents").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    s = MemexSettings()  # type: ignore[call-arg]
    set_settings(s)
    yield s
    set_settings(None)


class _FakeFTSStore:
    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}
        self.title_updates: list[tuple[str, str]] = []

    @classmethod
    def make_opener(cls) -> tuple[Any, _FakeFTSStore]:
        inst = cls()

        async def _open(_vault_path: Any) -> _FakeFTSStore:
            return inst

        return _open, inst

    async def chunk_ids_for_document(self, doc_id: str) -> set[str]:
        return {cid for cid, c in self.chunks.items() if c.document_id == doc_id}

    async def delete_document(self, doc_id: str) -> int:
        n = len(self.chunks)
        self.chunks = {cid: c for cid, c in self.chunks.items() if c.document_id != doc_id}
        return n - len(self.chunks)

    async def delete_chunks(self, chunk_ids: list[str]) -> int:
        return sum(self.chunks.pop(c, None) is not None for c in chunk_ids)

    async def upsert(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self.chunks[c.chunk_id] = c

    async def update_document_title(self, doc_id: str, title: str) -> int:
        self.title_updates.append((doc_id, title))
        n = 0
        for c in self.chunks.values():
            if c.document_id == doc_id:
                c.document_title = title
                n += 1
        return n

    async def close(self) -> None:
        return None


class _FakeVectorStore:
    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}
        self.embeddings: dict[str, list[float]] = {}
        self.title_updates: list[tuple[str, str]] = []

    @classmethod
    def make_opener(cls) -> tuple[Any, _FakeVectorStore]:
        inst = cls()

        async def _open(_vault_path: Any) -> _FakeVectorStore:
            return inst

        return _open, inst

    async def delete_document(self, doc_id: str) -> int:
        n = len(self.chunks)
        keep = [cid for cid, c in self.chunks.items() if c.document_id != doc_id]
        self.chunks = {cid: self.chunks[cid] for cid in keep}
        self.embeddings = {cid: self.embeddings[cid] for cid in keep if cid in self.embeddings}
        return n - len(self.chunks)

    async def delete_chunks(self, chunk_ids: list[str]) -> int:
        removed = 0
        for cid in chunk_ids:
            if self.chunks.pop(cid, None) is not None:
                removed += 1
            self.embeddings.pop(cid, None)
        return removed

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for c, e in zip(chunks, embeddings, strict=True):
            self.chunks[c.chunk_id] = c
            self.embeddings[c.chunk_id] = e

    async def update_document_title(self, doc_id: str, title: str) -> int:
        self.title_updates.append((doc_id, title))
        n = 0
        for c in self.chunks.values():
            if c.document_id == doc_id:
                c.document_title = title
                n += 1
        return n

    async def close(self) -> None:
        return None


@pytest.fixture
def fake_stores(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    fts_open, fts = _FakeFTSStore.make_opener()
    vec_open, vec = _FakeVectorStore.make_opener()
    monkeypatch.setattr("memex.index.pipeline.FTSStore.open", fts_open)
    monkeypatch.setattr("memex.index.pipeline.VectorStore.open", vec_open)

    embed_calls: list[list[str]] = []

    async def _fake_embed(chunks: list[Chunk]) -> list[list[float]]:
        embed_calls.append([c.chunk_id for c in chunks])
        return [[float(i)] * 768 for i in range(len(chunks))]

    monkeypatch.setattr("memex.index.pipeline._embed_chunks", _fake_embed)

    # No graph store — exercise the ImportError-degrades path.
    async def _no_graph(_vault_path: Any) -> None:
        raise ImportError("ryugraph not installed (test stub)")

    monkeypatch.setattr("memex.index.pipeline.GraphStore.open", _no_graph)
    return {"fts": fts, "vec": vec, "embed_calls": embed_calls}


async def _seed_and_index(body: str, stem: str) -> str:
    from memex.index.pipeline import index_document

    ref = await ingest_markdown_passthrough(body, source_stem=stem)
    await index_document(ref.doc_id)
    return ref.doc_id


@pytest.mark.asyncio
async def test_retitle_fans_out_without_reembedding(
    settings: MemexSettings, fake_stores: dict[str, Any]
) -> None:
    from memex.index.pipeline import retitle_document

    doc_id = await _seed_and_index("# Doc\n\nFirst paragraph.\n\nSecond paragraph.\n", "old_stem")
    embeds_before = len(fake_stores["embed_calls"])
    chunk_ids_before = set(fake_stores["fts"].chunks)
    assert chunk_ids_before  # sanity: something got indexed

    result = await retitle_document(doc_id, "A Clean Human Title")

    # No re-embed: the embedder was not called again.
    assert len(fake_stores["embed_calls"]) == embeds_before
    # Chunk ids are invariant under a rename.
    assert set(fake_stores["fts"].chunks) == chunk_ids_before
    assert set(fake_stores["vec"].chunks) == chunk_ids_before
    # Title propagated to every denormalized copy.
    assert result.frontmatter_changed is True
    assert result.fts_rows == len(chunk_ids_before)
    assert result.vector_rows == len(chunk_ids_before)
    assert all(
        c.document_title == "A Clean Human Title" for c in fake_stores["fts"].chunks.values()
    )
    assert all(
        c.document_title == "A Clean Human Title" for c in fake_stores["vec"].chunks.values()
    )
    # Frontmatter (source of truth) updated on disk.
    doc = await read_document(settings.vault_path, doc_id)
    assert doc.frontmatter.title == "A Clean Human Title"


@pytest.mark.asyncio
async def test_retitle_same_title_repairs_stores_without_sha_change(
    settings: MemexSettings, fake_stores: dict[str, Any]
) -> None:
    """Re-running with the current title is a no-op on the frontmatter
    (sha stable) but still refreshes the derived stores — so retitle
    doubles as a 'repair a stale denormalized title' tool."""
    from memex.index.pipeline import retitle_document

    doc_id = await _seed_and_index("# Doc\n\nBody text here.\n", "stem")
    doc = await read_document(settings.vault_path, doc_id)
    current_title = doc.frontmatter.title
    assert current_title is not None
    sha_before = doc.ref.content_sha256

    # Simulate drift: a store row carries a stale title.
    for c in fake_stores["fts"].chunks.values():
        c.document_title = "STALE"

    result = await retitle_document(doc_id, current_title)

    assert result.frontmatter_changed is False
    after = await read_document(settings.vault_path, doc_id)
    assert after.ref.content_sha256 == sha_before  # no rewrite
    # Stores were repaired to the canonical title anyway.
    assert all(c.document_title == current_title for c in fake_stores["fts"].chunks.values())


@pytest.mark.asyncio
async def test_retitle_rejects_empty_title(
    settings: MemexSettings, fake_stores: dict[str, Any]
) -> None:
    from memex.core.errors import ConfigurationError
    from memex.index.pipeline import retitle_document

    doc_id = await _seed_and_index("# Doc\n\nBody.\n", "stem")
    with pytest.raises(ConfigurationError):
        await retitle_document(doc_id, "   ")
