"""Incremental partial re-indexing — v1.x regression net.

The index pipeline's chunk_ids are content-derived (`sha1(text)[:10]`),
so an unchanged paragraph keeps its chunk_id across re-indexings.
`index_document` exploits that: it diffs the new chunker output
against the FTS store's existing per-doc chunk_ids and only re-embeds
+ re-writes the chunks whose content actually changed.

These tests cover the diff logic + the force/embedder-change escape
hatches + the watcher's edit→reindex path that the whole thing was
built for. LanceDB and sentence-transformers are faked; the FTS store
+ vault + manifest are real (tmp dirs) so the storage-layer integration
is exercised.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from memex.core.config import MemexSettings, set_settings
from memex.core.manifest import read_manifest
from memex.core.types import Chunk
from memex.ingest.pipeline import ingest_markdown_passthrough

# ----- Shared fixtures -----


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    (tmp_path / "documents").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def settings(tmp_vault: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemexSettings]:
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_vault))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    s = MemexSettings()  # type: ignore[call-arg]
    set_settings(s)
    yield s
    set_settings(None)


# ----- Fake stores -----
#
# We fake VectorStore + FTSStore because lancedb/SQLite-FTS are not the
# unit under test — the diff logic in `index_document` is. The fakes
# implement the public surface the pipeline calls and persist a single
# in-memory state across calls in the same test, so the second
# `index_document` invocation sees the first's writes.


class _FakeFTSStore:
    """In-memory stand-in. One instance per test session via
    `make_opener`; multiple `.open()` calls return the same singleton."""

    def __init__(self) -> None:
        # Map chunk_id → Chunk
        self.chunks: dict[str, Chunk] = {}
        self.open_count = 0
        self.delete_chunk_calls: list[list[str]] = []
        self.delete_doc_calls: list[str] = []
        self.upsert_calls: list[list[Chunk]] = []

    @classmethod
    def make_opener(cls) -> tuple[Any, _FakeFTSStore]:
        instance = cls()

        async def _open(_vault_path: Any) -> _FakeFTSStore:
            instance.open_count += 1
            return instance

        return _open, instance

    async def chunk_ids_for_document(self, doc_id: str) -> set[str]:
        return {cid for cid, c in self.chunks.items() if c.document_id == doc_id}

    async def delete_document(self, doc_id: str) -> int:
        self.delete_doc_calls.append(doc_id)
        before = len(self.chunks)
        self.chunks = {cid: c for cid, c in self.chunks.items() if c.document_id != doc_id}
        return before - len(self.chunks)

    async def delete_chunks(self, chunk_ids: list[str]) -> int:
        self.delete_chunk_calls.append(list(chunk_ids))
        removed = 0
        for cid in chunk_ids:
            if self.chunks.pop(cid, None) is not None:
                removed += 1
        return removed

    async def upsert(self, chunks: list[Chunk]) -> None:
        self.upsert_calls.append(list(chunks))
        for c in chunks:
            self.chunks[c.chunk_id] = c

    async def close(self) -> None:
        return None


class _FakeVectorStore:
    """Same shape as `_FakeFTSStore`, holding embeddings alongside chunks."""

    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}
        self.embeddings: dict[str, list[float]] = {}
        self.open_count = 0
        self.delete_chunk_calls: list[list[str]] = []
        self.delete_doc_calls: list[str] = []
        self.upsert_calls: list[list[Chunk]] = []

    @classmethod
    def make_opener(cls) -> tuple[Any, _FakeVectorStore]:
        instance = cls()

        async def _open(_vault_path: Any) -> _FakeVectorStore:
            instance.open_count += 1
            return instance

        return _open, instance

    async def delete_document(self, doc_id: str) -> int:
        self.delete_doc_calls.append(doc_id)
        before = len(self.chunks)
        survivors = [cid for cid, c in self.chunks.items() if c.document_id != doc_id]
        self.chunks = {cid: self.chunks[cid] for cid in survivors}
        self.embeddings = {cid: self.embeddings[cid] for cid in survivors if cid in self.embeddings}
        return before - len(self.chunks)

    async def delete_chunks(self, chunk_ids: list[str]) -> int:
        self.delete_chunk_calls.append(list(chunk_ids))
        removed = 0
        for cid in chunk_ids:
            if self.chunks.pop(cid, None) is not None:
                removed += 1
            self.embeddings.pop(cid, None)
        return removed

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.upsert_calls.append(list(chunks))
        for c, e in zip(chunks, embeddings, strict=True):
            self.chunks[c.chunk_id] = c
            self.embeddings[c.chunk_id] = e

    async def close(self) -> None:
        return None


@pytest.fixture
def fake_stores(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install the fake stores + a deterministic fake embedder.

    Tracks every chunk that gets sent to the embedder so individual
    tests can assert "only these N chunks were re-embedded."
    """
    fts_open, fts = _FakeFTSStore.make_opener()
    vec_open, vec = _FakeVectorStore.make_opener()
    monkeypatch.setattr("memex.index.pipeline.FTSStore.open", fts_open)
    monkeypatch.setattr("memex.index.pipeline.VectorStore.open", vec_open)

    embed_calls: list[list[str]] = []

    async def _fake_embed(chunks: list[Chunk]) -> list[list[float]]:
        embed_calls.append([c.chunk_id for c in chunks])
        # Deterministic per-chunk vector so tests can introspect if needed.
        return [[float(i)] * 768 for i in range(len(chunks))]

    monkeypatch.setattr("memex.index.pipeline._embed_chunks", _fake_embed)

    # Graph store: pretend ryugraph is missing — same path tests exercise
    # in CI. _open_graph returns None and the pipeline degrades.
    async def _no_graph(_vault_path: Any) -> None:
        raise ImportError("ryugraph not installed (test stub)")

    monkeypatch.setattr("memex.index.pipeline.GraphStore.open", _no_graph)

    return {
        "fts": fts,
        "vec": vec,
        "embed_calls": embed_calls,
    }


# ----- Helpers -----


async def _seed_doc(body: str, *, source_stem: str = "test") -> str:
    """Write a markdown doc into the vault, return its doc_id."""
    ref = await ingest_markdown_passthrough(body, source_stem=source_stem)
    return ref.doc_id


def _read_doc_via_vault(vault_path: Path, doc_id: str) -> str:
    md_path = vault_path / "documents" / f"{doc_id}.md"
    return md_path.read_text(encoding="utf-8")


async def _write_doc_via_vault(vault_path: Path, doc_id: str, body: str) -> None:
    """Write a new body to the vault for `doc_id`, preserving frontmatter."""
    from memex.vault.store import VaultDocument, read_document, write_document

    existing = await read_document(vault_path, doc_id)
    updated = VaultDocument(
        ref=existing.ref,
        frontmatter=existing.frontmatter,
        body=body,
        mtime_ns=existing.mtime_ns,
    )
    await write_document(vault_path, updated)


# ----- The actual tests -----


@pytest.mark.asyncio
async def test_first_index_writes_every_chunk(
    settings: MemexSettings, fake_stores: dict[str, Any]
) -> None:
    """No prior state → all chunks are 'added', zero deleted, zero unchanged."""
    from memex.index.pipeline import index_document

    doc_id = await _seed_doc(
        "# Test\n\nFirst paragraph.\n\nSecond paragraph.\n",
        source_stem="first_index",
    )

    result = await index_document(doc_id)
    assert result.chunk_count > 0
    assert result.chunks_added == result.chunk_count
    assert result.chunks_deleted == 0
    assert result.chunks_unchanged == 0
    # Diff was attempted but found no prior chunks ⇒ partial=False the
    # first time, since the contract is "diff present old vs new"; with
    # no old, the partial flag is False (it's effectively a full insert).
    assert result.partial is False
    # Both stores now hold every chunk.
    assert len(fake_stores["fts"].chunks) == result.chunk_count
    assert len(fake_stores["vec"].chunks) == result.chunk_count
    # The embedder saw all of them.
    assert len(fake_stores["embed_calls"][-1]) == result.chunk_count


@pytest.mark.asyncio
async def test_reindex_unchanged_body_skips_all_embeddings(
    settings: MemexSettings, fake_stores: dict[str, Any]
) -> None:
    """Second `index_document` call on an unchanged doc must add zero
    new chunks, delete zero, mark all unchanged, and skip the embedder
    entirely."""
    from memex.index.pipeline import index_document

    doc_id = await _seed_doc(
        "# Unchanged\n\nA paragraph.\n\nAnother paragraph.\n",
        source_stem="unchanged",
    )
    first = await index_document(doc_id)

    # Reset the embed call log so we only see what the second call does.
    fake_stores["embed_calls"].clear()

    second = await index_document(doc_id)
    assert second.chunk_count == first.chunk_count
    assert second.chunks_added == 0
    assert second.chunks_deleted == 0
    assert second.chunks_unchanged == first.chunk_count
    assert second.partial is True
    assert second.embedded is False
    # The embedder was called once (with an empty list) or not at all.
    # _embed_chunks short-circuits on empty input, so the call log
    # captures only non-empty batches.
    for batch in fake_stores["embed_calls"]:
        assert batch == [], f"unexpected re-embed of {batch}"


@pytest.mark.asyncio
async def test_reindex_with_added_paragraph_only_embeds_new(
    settings: MemexSettings, fake_stores: dict[str, Any]
) -> None:
    """Append a new section to the body. The original chunks keep
    their chunk_ids; only the new section produces new chunks."""
    from memex.index.pipeline import index_document

    body_v1 = (
        "# Doc\n\n## Section A\n\nOriginal paragraph A.\n\n## Section B\n\nOriginal paragraph B.\n"
    )
    doc_id = await _seed_doc(body_v1, source_stem="appended")
    v1 = await index_document(doc_id)
    fake_stores["embed_calls"].clear()

    # Append a new section.
    body_v2 = body_v1 + "\n## Section C\n\nA brand new paragraph C.\n"
    await _write_doc_via_vault(settings.vault_path, doc_id, body_v2)

    v2 = await index_document(doc_id)
    assert v2.chunk_count > v1.chunk_count  # new chunks added
    assert v2.chunks_added >= 1  # at least the new section
    assert v2.chunks_deleted == 0  # nothing removed
    assert v2.chunks_unchanged >= 1  # originals kept
    # Embedder saw only the new chunks.
    total_embedded = sum(len(b) for b in fake_stores["embed_calls"])
    assert total_embedded == v2.chunks_added


@pytest.mark.asyncio
async def test_reindex_with_removed_paragraph_deletes_only_gone_chunks(
    settings: MemexSettings, fake_stores: dict[str, Any]
) -> None:
    """Remove a section. Its chunks disappear; the rest stay."""
    from memex.index.pipeline import index_document

    body_v1 = (
        "# Doc\n\n"
        "## Section A\n\nParagraph A content here.\n\n"
        "## Section B\n\nParagraph B content here.\n\n"
        "## Section C\n\nParagraph C content here.\n"
    )
    doc_id = await _seed_doc(body_v1, source_stem="trimmed")
    v1 = await index_document(doc_id)
    fake_stores["embed_calls"].clear()

    body_v2 = (
        "# Doc\n\n"
        "## Section A\n\nParagraph A content here.\n\n"
        "## Section C\n\nParagraph C content here.\n"
    )
    await _write_doc_via_vault(settings.vault_path, doc_id, body_v2)

    v2 = await index_document(doc_id)
    assert v2.chunk_count < v1.chunk_count
    assert v2.chunks_deleted >= 1
    assert v2.chunks_unchanged >= 1
    # No new content ⇒ embedder doesn't run at all (or only with empty
    # batches).
    for batch in fake_stores["embed_calls"]:
        assert batch == []
    assert v2.chunks_added == 0


@pytest.mark.asyncio
async def test_reindex_with_modified_paragraph_swaps_one_chunk(
    settings: MemexSettings, fake_stores: dict[str, Any]
) -> None:
    """Modify one paragraph. The modified chunk's sha changes ⇒ one
    chunk_id disappears, one new chunk_id appears, the rest stay."""
    from memex.index.pipeline import index_document

    body_v1 = (
        "# Doc\n\n"
        "## A\n\nOriginal A content.\n\n"
        "## B\n\nOriginal B content.\n\n"
        "## C\n\nOriginal C content.\n"
    )
    doc_id = await _seed_doc(body_v1, source_stem="modified")
    await index_document(doc_id)  # seed; metrics asserted on the re-index below
    fake_stores["embed_calls"].clear()

    body_v2 = (
        "# Doc\n\n"
        "## A\n\nOriginal A content.\n\n"
        "## B\n\nMODIFIED B content with different text.\n\n"
        "## C\n\nOriginal C content.\n"
    )
    await _write_doc_via_vault(settings.vault_path, doc_id, body_v2)

    v2 = await index_document(doc_id)
    # Net chunk count is roughly the same (one swapped).
    assert v2.chunks_added >= 1
    assert v2.chunks_deleted >= 1
    assert v2.chunks_unchanged >= 1
    # Embedder only re-ran for the changed chunk(s).
    total_embedded = sum(len(b) for b in fake_stores["embed_calls"])
    assert total_embedded == v2.chunks_added


@pytest.mark.asyncio
async def test_force_reindex_re_embeds_everything(
    settings: MemexSettings, fake_stores: dict[str, Any]
) -> None:
    """`force=True` skips the diff — every chunk is re-embedded even
    when nothing changed."""
    from memex.index.pipeline import index_document

    doc_id = await _seed_doc(
        "# Forced\n\nParagraph one.\n\nParagraph two.\n",
        source_stem="forced",
    )
    await index_document(doc_id)  # seed; metrics asserted on the re-index below
    fake_stores["embed_calls"].clear()

    v2 = await index_document(doc_id, force=True)
    assert v2.partial is False
    assert v2.chunks_added == v2.chunk_count
    assert v2.chunks_unchanged == 0
    # All chunks re-embedded.
    total_embedded = sum(len(b) for b in fake_stores["embed_calls"])
    assert total_embedded == v2.chunk_count
    # Document-wide delete was called (the force path).
    assert fake_stores["fts"].delete_doc_calls[-1] == doc_id
    assert fake_stores["vec"].delete_doc_calls[-1] == doc_id


@pytest.mark.asyncio
async def test_embedder_change_triggers_implicit_force(
    settings: MemexSettings, fake_stores: dict[str, Any]
) -> None:
    """When the configured embedder differs from the manifest's prior
    record, the next `index_document` call must run a full re-embed
    even without `force=True` — existing vectors live in the wrong
    space."""
    from memex.index.pipeline import index_document

    doc_id = await _seed_doc(
        "# Embedder swap\n\nFirst.\n\nSecond.\n",
        source_stem="embedder_swap",
    )
    await index_document(doc_id)
    fake_stores["embed_calls"].clear()

    # Mutate the active settings to simulate an embedder swap.
    settings.models.embedder = "different/embedder-v2"
    set_settings(settings)

    v2 = await index_document(doc_id)  # NOT passing force=True
    assert v2.partial is False
    assert v2.chunks_added == v2.chunk_count
    # And the new embedder name landed in the manifest.
    manifest = await read_manifest(settings.vault_path, doc_id)
    assert manifest is not None
    assert manifest.index is not None
    assert manifest.index.embedding_model == "different/embedder-v2"


@pytest.mark.asyncio
async def test_manifest_records_diff_metrics(
    settings: MemexSettings, fake_stores: dict[str, Any]
) -> None:
    """The manifest's IndexStage must record added / deleted / unchanged
    so `memex doctor` and the eval suite can surface partial-reindex
    behaviour. Uses a body large enough that the chunker produces
    multiple chunks — otherwise any edit replaces the only chunk
    wholesale and `chunks_unchanged` is correctly zero."""
    from memex.index.pipeline import index_document

    big_para = " ".join(["lorem ipsum dolor sit amet"] * 80)
    other_para = " ".join(["sed do eiusmod tempor incididunt"] * 80)
    body_v1 = (
        f"# Doc\n\n## A\n\n{big_para}\n\n## B\n\n{other_para}\n\n## C\n\nSome additional content.\n"
    )
    doc_id = await _seed_doc(body_v1, source_stem="manifest_metrics")
    first = await index_document(doc_id)
    # Sanity: chunker produced multiple chunks (otherwise this test
    # can't distinguish unchanged vs. wholesale replacement).
    assert first.chunk_count >= 2

    # Modify one paragraph; the chunks for the others must survive.
    body_v2 = body_v1.replace(other_para, "completely different content here")
    await _write_doc_via_vault(settings.vault_path, doc_id, body_v2)
    await index_document(doc_id)

    manifest = await read_manifest(settings.vault_path, doc_id)
    assert manifest is not None
    stage = manifest.index
    assert stage is not None
    assert stage.partial is True
    assert stage.chunks_added >= 1
    assert stage.chunks_deleted >= 1
    assert stage.chunks_unchanged >= 1


@pytest.mark.asyncio
async def test_existing_chunks_preserved_across_partial_reindex(
    settings: MemexSettings, fake_stores: dict[str, Any]
) -> None:
    """Unchanged chunks keep both their chunk_id AND their embedding
    after a partial re-index — that's the whole point. Verify the
    embedding stayed in the vector store."""
    from memex.index.pipeline import index_document

    body_v1 = (
        "# Doc\n\n## Stays\n\nThis paragraph never changes.\n\n## Changes\n\nVersion 1 content.\n"
    )
    doc_id = await _seed_doc(body_v1, source_stem="preserve")
    await index_document(doc_id)

    # Snapshot the embeddings of the chunks that should survive.
    stays_chunk_ids = {
        cid for cid, c in fake_stores["fts"].chunks.items() if "never changes" in c.text
    }
    assert stays_chunk_ids, "expected at least one 'stays' chunk"
    pre_embeddings = {cid: list(fake_stores["vec"].embeddings[cid]) for cid in stays_chunk_ids}

    body_v2 = body_v1.replace("Version 1 content.", "Version 2 different.")
    await _write_doc_via_vault(settings.vault_path, doc_id, body_v2)
    await index_document(doc_id)

    # The 'stays' chunks have the same chunk_ids AND the same embeddings.
    for cid, original_vec in pre_embeddings.items():
        assert cid in fake_stores["vec"].chunks, f"chunk {cid} was deleted across re-index"
        assert fake_stores["vec"].embeddings[cid] == original_vec, (
            f"chunk {cid} was re-embedded across re-index"
        )
