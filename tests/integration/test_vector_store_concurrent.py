"""P3.3 follow-up — Filler N1: LanceDB concurrent-search smoke test.

LanceDB 0.30 split its API into async query construction (`.search`,
`.to_pydantic` are awaitable) and sync builder methods (`.limit`,
`.where` are chained sync). The agent's `retrieve` node fires
`asyncio.gather` for parallel BM25 + dense calls, and inside the dense
path we call `VectorStore.search` — but the codebase has had no
direct test that the async LanceDB surface tolerates **multiple
concurrent reads** against a SINGLE `VectorStore` (i.e., one shared
LanceDB connection).

This test populates a small vector store, fires N parallel `.search()`
calls via `asyncio.gather`, and verifies:
- All calls complete without exceptions
- Each call returns the expected number of results
- Results are deterministic across the burst (same query → same
  chunk_ids, same order)
- The store remains usable after the burst (post-condition: a final
  sequential call still works)

The test uses a real LanceDB-on-disk store (the production code path),
not a fake — the point is to exercise the actual async surface that
the agent uses in production. Skip semantics: if `lancedb` isn't
installed, the test skips gracefully.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from memex.core.types import Chunk

# Skip the whole module if lancedb isn't installed. The test only
# makes sense against the real LanceDB surface.
lancedb = pytest.importorskip("lancedb")

from memex.index.vector_store import VectorStore  # noqa: E402


def _chunk(chunk_id: str, document_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title=f"Doc {document_id}",
        text=text,
        char_start=0,
        char_end=len(text),
    )


@pytest.mark.asyncio
async def test_concurrent_search_burst_returns_consistent_results(
    tmp_path: Path,
) -> None:
    """20 concurrent `.search()` calls on the same VectorStore must
    all complete successfully and produce consistent results.

    Without this guarantee, the agent's hybrid_search (which awaits a
    dense + BM25 pair via asyncio.gather) could race silently under
    multi-query workloads — the production daemon serves multiple
    concurrent /ask requests on the same shared VectorStore.
    """
    vault_path = tmp_path / "vault"
    vault_path.mkdir(parents=True)

    store = await VectorStore.open(vault_path)

    # Populate with deterministic embeddings: a 32-dim vector where
    # the chunk's int suffix is the index of the only non-zero element.
    # Query for [1.0, 0, 0, ...] should match chunk 0 first; querying
    # for [0, 1.0, 0, ...] matches chunk 1, etc.
    dim = 768  # EmbeddingGemma-300m output dim (the production embedder)
    n = 12
    chunks = [_chunk(f"chunk-{i:03d}", f"doc-{i:03d}", f"text body {i}") for i in range(n)]
    embeddings = []
    for i in range(n):
        emb = [0.0] * dim
        emb[i] = 1.0  # i-th chunk's embedding is one-hot at position i
        embeddings.append(emb)

    await store.upsert(chunks, embeddings)

    # Build 20 concurrent queries. Each query asks for top-3.
    # Queries cycle through the chunk indices so we get a mix of
    # different "winner" chunks across the burst.
    burst_size = 20
    queries: list[list[float]] = []
    expected_top1: list[str] = []
    for i in range(burst_size):
        chunk_idx = i % n
        q = [0.0] * dim
        q[chunk_idx] = 1.0
        queries.append(q)
        expected_top1.append(f"chunk-{chunk_idx:03d}")

    # Fire all queries concurrently. asyncio.gather raises if any
    # individual task raises, so an exception here = race / corruption.
    results = await asyncio.gather(
        *(store.search(q, k=3) for q in queries),
        return_exceptions=False,
    )

    # Every call returned exactly 3 chunks; no None / empty list.
    assert len(results) == burst_size
    for i, r in enumerate(results):
        assert len(r) == 3, f"query {i} returned {len(r)} results, expected 3"

    # Each query's top-1 should match the deterministic expectation.
    # (LanceDB's L2 distance ordering: the query vector has exactly
    # one non-zero at position chunk_idx; the matching chunk's
    # embedding has the same. Distance 0 vs others' distance √2.)
    actual_top1 = [r[0].chunk_id for r in results]
    assert actual_top1 == expected_top1, (
        "concurrent reads produced inconsistent top-1 results — "
        "race condition in LanceDB async surface?"
    )

    # Post-condition: store remains usable after the burst. If the
    # connection got into a bad state under concurrent reads, a
    # subsequent sequential call should reveal it.
    final_query = [0.0] * dim
    final_query[0] = 1.0
    final = await store.search(final_query, k=3)
    assert len(final) == 3
    assert final[0].chunk_id == "chunk-000"

    # ALSO post-condition: the second-best match should be one of the
    # other chunks (sanity that the store isn't returning the same
    # chunk multiple times under load).
    assert final[1].chunk_id != "chunk-000"

    await store.close()


@pytest.mark.asyncio
async def test_concurrent_search_in_docs_burst_consistent(
    tmp_path: Path,
) -> None:
    """The `search_in_docs` variant (whitelist-filtered dense search)
    is used by the agent's `expand_graph` node. Concurrent reads
    against it must also be safe — same threading reality as the
    primary search method.
    """
    vault_path = tmp_path / "vault"
    vault_path.mkdir(parents=True)

    store = await VectorStore.open(vault_path)

    dim = 768
    n_per_doc = 5
    doc_ids = ["doc-A", "doc-B", "doc-C"]
    chunks: list[Chunk] = []
    embeddings: list[list[float]] = []
    for doc_idx, doc in enumerate(doc_ids):
        for i in range(n_per_doc):
            cid = f"{doc}-chunk-{i:02d}"
            chunks.append(_chunk(cid, doc, f"text {doc} {i}"))
            emb = [0.0] * dim
            # Unique embedding per chunk: position = doc_idx * n_per_doc + i
            emb[doc_idx * n_per_doc + i] = 1.0
            embeddings.append(emb)
    await store.upsert(chunks, embeddings)

    # 15 concurrent queries; each restricts to a single doc.
    burst_size = 15
    coros = []
    expected_docs = []
    for i in range(burst_size):
        doc_idx = i % len(doc_ids)
        doc = doc_ids[doc_idx]
        q = [0.0] * dim
        # Target a chunk within `doc`: position = doc_idx * n_per_doc + (i mod n_per_doc)
        q[doc_idx * n_per_doc + (i % n_per_doc)] = 1.0
        coros.append(store.search_in_docs(q, doc_ids=[doc], k=2))
        expected_docs.append(doc)

    results = await asyncio.gather(*coros)

    assert len(results) == burst_size
    for i, r in enumerate(results):
        # Each result has at most k=2 entries; ALL results must come
        # from the whitelisted doc.
        assert 1 <= len(r) <= 2
        for chunk in r:
            assert chunk.document_id == expected_docs[i], (
                f"concurrent search_in_docs returned chunk from doc "
                f"{chunk.document_id} when whitelisted to {expected_docs[i]}"
            )

    await store.close()


@pytest.mark.asyncio
async def test_concurrent_searches_across_multiple_stores_are_independent(
    tmp_path: Path,
) -> None:
    """When two separate VectorStore instances are opened on different
    vaults, concurrent searches against them must NOT cross-contaminate.
    Tests for any global state in lancedb's async connection layer.
    """
    vault_a = tmp_path / "vault-a"
    vault_b = tmp_path / "vault-b"
    vault_a.mkdir(parents=True)
    vault_b.mkdir(parents=True)

    store_a = await VectorStore.open(vault_a)
    store_b = await VectorStore.open(vault_b)

    dim = 768
    chunks_a = [_chunk(f"A-{i}", "doc-A", f"text A{i}") for i in range(5)]
    chunks_b = [_chunk(f"B-{i}", "doc-B", f"text B{i}") for i in range(5)]
    embs_a = []
    embs_b = []
    for i in range(5):
        ea = [0.0] * dim
        eb = [0.0] * dim
        ea[i] = float(i + 1)
        eb[i] = float(i + 1)
        embs_a.append(ea)
        embs_b.append(eb)

    await store_a.upsert(chunks_a, embs_a)
    await store_b.upsert(chunks_b, embs_b)

    q = [1.0] + [0.0] * (dim - 1)
    # Fire interleaved searches against both stores.
    results = await asyncio.gather(
        store_a.search(q, k=3),
        store_b.search(q, k=3),
        store_a.search(q, k=3),
        store_b.search(q, k=3),
    )

    # Results from store_a must contain only A-* chunks; store_b only B-*.
    for i, r in enumerate(results):
        prefix = "A" if i % 2 == 0 else "B"
        for chunk in r:
            assert chunk.chunk_id.startswith(prefix), (
                f"Result {i} (store_{prefix}) returned cross-store chunk {chunk.chunk_id}"
            )

    await store_a.close()
    await store_b.close()
