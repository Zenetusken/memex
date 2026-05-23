"""End-to-end indexing pipeline — chunk → embed → write.

Reads from the vault via `memex.vault`, runs the chunker (pure
function), pulls the embedder via `ModelRegistry`, writes to both the
vector store (LanceDB), the FTS store (SQLite), and the graph store
(RyuGraph). Idempotent on `doc_id`.

**Incremental re-indexing.** Chunk IDs are content-derived
(`sha1(text)[:10]`), so an unchanged paragraph keeps its `chunk_id`
across re-indexings. On every call this module computes the diff
between the existing per-doc chunk_ids in the FTS store and the new
chunker output, then only embeds and writes the *added* chunks while
selectively deleting the *removed* ones. Unchanged chunks (with their
embeddings) stay in place. This makes editing one paragraph of a
50-chunk document a ~6% rewrite instead of a 100% rewrite.

`force=True` (or a detected embedder-model change) skips the diff and
re-embeds every chunk. The watcher's `default_reaction` benefits
automatically — small markdown edits trigger small index updates.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import structlog
import ulid
from pydantic import BaseModel, Field

from memex.core.config import get_settings
from memex.core.manifest import (
    IndexStage,
    now_utc,
    read_manifest,
    update_manifest,
)
from memex.core.types import Chunk
from memex.index.chunker import chunk_document
from memex.index.fts_store import FTSStore
from memex.index.graph_store import GraphStore
from memex.index.vector_store import EMBEDDING_DIM, VectorStore
from memex.models.registry import get_registry
from memex.vault.store import list_documents, read_document

logger = structlog.get_logger(__name__)


class IndexResult(BaseModel):
    """Return value of `index_document` — the post-index chunk count,
    whether any embedding actually happened, and the partial-reindex
    diff (added / deleted / unchanged) so callers can render a
    meaningful progress line."""

    doc_id: str
    chunk_count: int                # total chunks now in the index for this doc
    embedded: bool                  # True if at least one chunk was embedded
    chunks_added: int = 0           # new chunks vs. prior state
    chunks_deleted: int = 0         # chunks that disappeared
    chunks_unchanged: int = 0       # chunks whose chunk_id (and thus text) survived
    partial: bool = False           # whether this was a diff-based re-index


class FailureItem(BaseModel):
    """Per-document failure record returned by `reindex_vault`.

    Carries structured fields instead of stringifying the exception so
    the CLI (and any other caller) can render a useful table of doc_id
    × error type × message × context.
    """

    doc_id: str
    error_type: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)


class ReindexReport(BaseModel):
    """Aggregate report from `reindex_vault` — total docs processed,
    chunks written, and a structured per-document failure list."""

    documents_processed: int
    chunks_written: int
    failures: list[FailureItem]


async def _embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    """Run the chunks through the embedder. Empty list short-circuits.

    `batch_size=32` is the safe default when vLLM is co-resident on
    a 12 GB card (8B-AWQ leaves ~2.5 GB for the embedder + reranker
    + activations, and batch=128 spikes the activations enough to
    OOM during the encode call). The earlier 128 was tuned for a
    standalone embedder run with the orchestrator off-process and
    no other GPU consumers. Override via `MEMEX_INDEX_EMBED_BATCH`
    if you want to push throughput on rigs with more headroom.
    """
    if not chunks:
        return []
    batch_size = int(os.environ.get("MEMEX_INDEX_EMBED_BATCH", "32"))
    registry = get_registry()
    # P3.3 v3/v4: leave the dense-embedding text UNSTRIPPED of
    # `[chart-extracted]` blocks. The FTS body IS stripped (see
    # FTSStore.upsert) — that's where the term-frequency
    # perturbation lived. Dense embedding is more diffuse; empirically
    # (v4b eval 2026-05-22), stripping it loses Q21's chart-OCR win
    # without recovering any prose queries. Keep chart content in the
    # embedded text so chart-numeric queries can match via dense
    # vector similarity.
    async with registry.use("embedder") as embedder:
        # SentenceTransformer.encode is sync + heavy; offload.
        def _encode() -> Any:
            return embedder.encode(
                [c.text for c in chunks],
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )

        embeddings = await asyncio.to_thread(_encode)
    return [list(map(float, row)) for row in embeddings]


async def index_document(doc_id: str, *, force: bool = False) -> IndexResult:
    """Chunk, embed, and write derived state for one document.

    By default, runs a diff against the FTS store's existing
    chunk_ids for this doc and only re-embeds the chunks whose
    content actually changed. `force=True` skips the diff and
    re-embeds everything (used by `reindex --force` and triggered
    automatically when the configured embedder model differs from
    the one recorded in the prior manifest — the existing vectors
    would otherwise live in the wrong vector space).
    """
    settings = get_settings()
    log = logger.bind(doc_id=doc_id)
    log.info("index.start", force=force)

    correlation_id = str(ulid.ULID())
    start = time.monotonic()

    doc = await read_document(settings.vault_path, doc_id)
    new_chunks = chunk_document(doc)
    new_by_id = {c.chunk_id: c for c in new_chunks}
    new_ids = set(new_by_id.keys())

    # Embedder-model-change detection. If the configured embedder differs
    # from the one recorded the last time we indexed this doc, the
    # existing vectors are in the wrong space — force a full re-embed.
    if not force:
        prior = await read_manifest(settings.vault_path, doc_id)
        if (
            prior is not None
            and prior.index is not None
            and prior.index.embedding_model != settings.models.embedder
        ):
            log.info(
                "index.embedder_changed",
                prior=prior.index.embedding_model,
                current=settings.models.embedder,
            )
            force = True

    # Open the three stores under a single AsyncExitStack so a failure
    # opening any one of them closes the others cleanly. The graph
    # store is optional (ImportError when ryugraph isn't installed);
    # the other two are required.
    async with AsyncExitStack() as stack:
        vstore = await VectorStore.open(settings.vault_path)
        stack.push_async_callback(vstore.close)
        fstore = await FTSStore.open(settings.vault_path)
        stack.push_async_callback(fstore.close)
        gstore = await _open_graph(settings.vault_path)
        if gstore is not None:
            stack.push_async_callback(gstore.close)

        if force:
            existing_ids: set[str] = set()
            to_add_ids = new_ids
            to_delete_ids: set[str] = set()
            unchanged_count = 0
            partial = False
            # Wipe everything this doc had — chunks may have been removed
            # AND we're re-embedding every survivor anyway.
            await vstore.delete_document(doc_id)
            await fstore.delete_document(doc_id)
        else:
            existing_ids = await fstore.chunk_ids_for_document(doc_id)
            to_add_ids = new_ids - existing_ids
            to_delete_ids = existing_ids - new_ids
            unchanged_count = len(existing_ids & new_ids)
            partial = bool(existing_ids)
            # Surgical removal — only delete the chunks that disappeared.
            if to_delete_ids:
                await vstore.delete_chunks(list(to_delete_ids))
                await fstore.delete_chunks(list(to_delete_ids))

        # Preserve chunker order for the new chunks we have to write.
        chunks_to_add = [c for c in new_chunks if c.chunk_id in to_add_ids]

        # Only embed the chunks we actually need to write.
        embeddings = await _embed_chunks(chunks_to_add)

        if chunks_to_add:
            await vstore.upsert(chunks_to_add, embeddings)
            await fstore.upsert(chunks_to_add)

        # Register the document node in the graph; enrich adds the
        # MENTIONS / CITES edges when it runs. We do this on every
        # index call so the graph stays in sync with the markdown
        # vault even when no chunks changed.
        if gstore is not None:
            await gstore.upsert_document(
                doc_id, doc.frontmatter.title or doc_id
            )

    duration_ms = int((time.monotonic() - start) * 1000)
    chunks_added = len(chunks_to_add)
    chunks_deleted = len(to_delete_ids)

    await update_manifest(
        settings.vault_path,
        doc_id,
        index=IndexStage(
            correlation_id=correlation_id,
            indexed_at=now_utc(),
            embedding_model=settings.models.embedder,
            embedding_dim=EMBEDDING_DIM,
            chunk_count=len(new_chunks),
            chunks_added=chunks_added,
            chunks_deleted=chunks_deleted,
            chunks_unchanged=unchanged_count,
            partial=partial,
            duration_ms=duration_ms,
        ),
        correlation_id=correlation_id,
    )

    log.info(
        "index.done",
        chunks=len(new_chunks),
        added=chunks_added,
        deleted=chunks_deleted,
        unchanged=unchanged_count,
        partial=partial,
        duration_ms=duration_ms,
    )
    return IndexResult(
        doc_id=doc_id,
        chunk_count=len(new_chunks),
        embedded=bool(chunks_to_add),
        chunks_added=chunks_added,
        chunks_deleted=chunks_deleted,
        chunks_unchanged=unchanged_count,
        partial=partial,
    )


async def remove_document(doc_id: str) -> None:
    """Drop the document's derived state from all three indexes.

    The canonical markdown under `vault/documents/{doc_id}.md` is left
    alone; this is the "clear my derived state" operation that
    `memex reindex --force` and the watcher's delete path call.
    """
    settings = get_settings()
    async with AsyncExitStack() as stack:
        vstore = await VectorStore.open(settings.vault_path)
        stack.push_async_callback(vstore.close)
        fstore = await FTSStore.open(settings.vault_path)
        stack.push_async_callback(fstore.close)
        gstore = await _open_graph(settings.vault_path)
        if gstore is not None:
            stack.push_async_callback(gstore.close)

        await vstore.delete_document(doc_id)
        await fstore.delete_document(doc_id)
        if gstore is not None:
            await gstore.delete_document(doc_id)


async def _open_graph(vault_path: Path) -> GraphStore | None:
    """Open the graph store, returning None if RyuGraph isn't installed.

    The agent's citation-graph traversal uses this, but the rest of
    the pipeline degrades gracefully — vector + FTS retrieval still
    works without it. The user sees a `graph.open.unavailable` warning
    and can fix by `uv sync` if they expected the graph to be present.
    """
    try:
        return await GraphStore.open(vault_path)
    except ImportError as e:
        logger.warning(
            "graph.open.unavailable",
            reason=str(e),
            fix="uv sync ensures ryugraph is installed",
        )
        return None


async def reindex_vault(*, force: bool = False) -> ReindexReport:
    """Rebuild the indexes from the canonical Markdown (ADR-0003).

    `force=True` deletes the derived-state files entirely before
    iterating documents. Per-doc `index_document(force=True)` then
    re-embeds every chunk (no diff to compute — the stores are empty).
    Without `--force`, this is effectively a per-doc incremental
    re-index that catches any drift between the on-disk markdown and
    the index.
    """
    settings = get_settings()
    if force:
        # Bypass any persistence checks; recursively drop the .memex/{embeddings,search,graph}.*
        # We don't touch traces/ or manifests/ — they're independent derived state.
        derived = settings.vault_path / ".memex"
        for target in ("embeddings.lance", "search.sqlite", "graph.ryu"):
            path = derived / target
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file():
                        child.unlink()
                for child in sorted(path.rglob("*"), reverse=True):
                    if child.is_dir():
                        child.rmdir()
                path.rmdir()

    from memex.core.errors import MemexError

    processed = 0
    chunks_total = 0
    failures: list[FailureItem] = []
    async for ref in list_documents(settings.vault_path):
        try:
            result = await index_document(ref.doc_id, force=force)
            processed += 1
            chunks_total += result.chunk_count
        except asyncio.CancelledError:
            raise
        except Exception as e:
            ctx: dict[str, Any] = {}
            if isinstance(e, MemexError):
                ctx = dict(e.context)
            failures.append(
                FailureItem(
                    doc_id=ref.doc_id,
                    error_type=type(e).__name__,
                    message=str(e),
                    context=ctx,
                )
            )
            logger.warning(
                "reindex.doc_failed",
                doc_id=ref.doc_id,
                error_type=type(e).__name__,
                error=str(e),
            )
    return ReindexReport(
        documents_processed=processed,
        chunks_written=chunks_total,
        failures=failures,
    )
