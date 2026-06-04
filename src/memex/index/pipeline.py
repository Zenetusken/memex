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
from memex.core.errors import ConfigurationError
from memex.core.manifest import (
    ChartExtraction,
    IndexStage,
    now_utc,
    read_manifest,
    update_manifest,
)
from memex.core.table_linearize import linearize_gfm_tables
from memex.core.text import reattach_chart_extractions
from memex.core.types import Chunk
from memex.index.chunker import chunk_document
from memex.index.embed_prompts import (
    chunk_title,
    document_input,
    native_prompts_enabled,
)
from memex.index.fts_store import FTSStore
from memex.index.graph_store import GraphStore
from memex.index.table_store import TableStore, extract_tables
from memex.index.vector_store import EMBEDDING_DIM, VectorStore
from memex.models.registry import get_registry
from memex.vault.store import (
    VaultDocument,
    list_documents,
    make_ref,
    read_document,
    write_document,
)

logger = structlog.get_logger(__name__)


class IndexResult(BaseModel):
    """Return value of `index_document` — the post-index chunk count,
    whether any embedding actually happened, and the partial-reindex
    diff (added / deleted / unchanged) so callers can render a
    meaningful progress line."""

    doc_id: str
    chunk_count: int  # total chunks now in the index for this doc
    embedded: bool  # True if at least one chunk was embedded
    chunks_added: int = 0  # new chunks vs. prior state
    chunks_deleted: int = 0  # chunks that disappeared
    chunks_unchanged: int = 0  # chunks whose chunk_id (and thus text) survived
    partial: bool = False  # whether this was a diff-based re-index


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
    # EmbeddingGemma native doc prompt (`title: … | text: …`) when enabled
    # (default ON); else bare `chunk.text` (the A/B / revert path). The
    # prompt wraps ONLY the transient `encode` input — the Chunk objects
    # flowing to the stores keep their original `.text`.
    if native_prompts_enabled():
        inputs = [document_input(chunk_title(c), c.text) for c in chunks]
    else:
        inputs = [c.text for c in chunks]
    async with registry.use("embedder") as embedder:
        # SentenceTransformer.encode is sync + heavy; offload. `inputs` is
        # the first positional param in sentence-transformers 5.5.1.
        def _encode() -> Any:
            return embedder.encode(
                inputs,
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )

        embeddings = await asyncio.to_thread(_encode)
    return [list(map(float, row)) for row in embeddings]


def _embed_recipe_version() -> str:
    """The embedding-recipe version tag recorded in the manifest.

    Bumping this versus the prior manifest value auto-forces a full
    re-embed (query- and doc-side embeddings must change together for a
    bi-encoder). `"v0"` is the back-compat bare baseline; native prompts
    bump to `"v1-gemma-prompts"`. Toggling native prompts off self-heals
    back to `"v0"`.
    """
    return "v1-gemma-prompts" if native_prompts_enabled() else "v0"


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
    # One manifest read serves both (a) the parse stage's per-page char counts
    # below (chunker page attribution → the webui's click-source→jump-to-PDF-page
    # UX) and (b) the embedder/recipe drift checks further down. A second
    # `read_manifest` call would race a concurrent watcher / re-parse — single
    # snapshot keeps the chunker, the force-detection, and the recorded recipe
    # all consistent.
    prior_manifest = await read_manifest(settings.vault_path, doc_id)
    # Legacy manifests written before `PageDecision.char_count` existed carry
    # all-zero counts → `chunk_document` treats that as "no mapping available"
    # and leaves `Chunk.page` as `None` (the webui falls back to section-only
    # anchors; no regression). Same fallback when there's no manifest yet (very
    # first index call).
    page_char_counts: list[tuple[int, int]] | None = None
    segment_intervals: list[tuple[int, int, float, float]] | None = None
    chart_extractions: list[ChartExtraction] = []
    if prior_manifest is not None and prior_manifest.parse is not None:
        page_char_counts = [(p.page, p.char_count) for p in prior_manifest.parse.pages]
        chart_extractions = prior_manifest.parse.chart_extractions
        if prior_manifest.parse.segments:  # audio route (ADR-0017): transcript time-attribution
            segment_intervals = [
                (s.char_start, s.char_end, s.start_s, s.end_s) for s in prior_manifest.parse.segments
            ]
    # Re-attach the chart-OCR `[chart-extracted]` blocks (from the parse manifest sidecar) at the
    # `<!-- image -->` positions, then re-derive the `[table-rows]` linearization — NEITHER lives
    # in the vault `.md`, which is content-only since audit-10. Both transforms reproduce the SAME
    # body the chunker saw when the `.md` carried the blocks → chunk_ids stay stable whether the
    # on-disk `.md` is already clean (new parse, non-empty sidecar) or still-polluted (pre-migration,
    # empty sidecar → re-attach is a no-op on the already-inline blocks). The chunk text thus carries
    # the chart content for retrieval/answering while the vault file (and the webui raw view) stay
    # clean. Order matches parse time: chart re-attach (the old stitch) THEN table linearization.
    reattached_body = reattach_chart_extractions(doc.body, chart_extractions)
    indexed_doc = doc.model_copy(update={"body": linearize_gfm_tables(reattached_body)})
    new_chunks = chunk_document(
        indexed_doc, page_char_counts=page_char_counts, segment_intervals=segment_intervals
    )
    new_by_id = {c.chunk_id: c for c in new_chunks}
    new_ids = set(new_by_id.keys())

    # Snapshot the embedding-recipe version ONCE for this index call and reuse
    # it for both the force-detection AND the manifest write below. Reading
    # `_embed_recipe_version()` separately at each site (with awaits between)
    # would be a TOCTOU: the env toggle could flip mid-call so the manifest
    # records a recipe that doesn't match the embeddings actually written. One
    # snapshot makes the force-check and the recorded recipe consistent.
    recipe = _embed_recipe_version()

    # Embedder-model-change detection. If the configured embedder differs
    # from the one recorded the last time we indexed this doc, the
    # existing vectors are in the wrong space — force a full re-embed.
    if not force:
        prior = prior_manifest
        if prior is not None and prior.index is not None:
            if prior.index.embedding_model != settings.models.embedder:
                log.info(
                    "index.embedder_changed",
                    prior=prior.index.embedding_model,
                    current=settings.models.embedder,
                )
                force = True
            # Orthogonal to the embedder swap: an embedding-recipe change
            # (e.g. enabling/disabling native prompts) also invalidates the
            # existing vectors — query+doc embeddings must share a space.
            if prior.index.embedding_recipe_version != recipe:
                log.info(
                    "index.recipe_changed",
                    prior=prior.index.embedding_recipe_version,
                    current=recipe,
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
        tstore = await TableStore.open(settings.vault_path)
        stack.push_async_callback(tstore.close)
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

        # Table-RAG Phase 2: extract the well-formed GFM tables from the
        # document body and replace this doc's structured-table set. This
        # is a full per-doc replacement (extract_tables scans the whole
        # body), so it's correct on both the force path and the diff path —
        # it always reflects the current body. The agent's `query_tables`
        # node reads these at answer time. On a body with no tables this is
        # a cheap DELETE (clears any stale rows from a prior version).
        await tstore.upsert_document(
            doc_id,
            extract_tables(
                doc_id, doc.body, split_merged=settings.agents.table_column_split_enabled
            ),
        )

        # Register the document node in the graph; enrich adds the
        # MENTIONS / CITES edges when it runs. We do this on every
        # index call so the graph stays in sync with the markdown
        # vault even when no chunks changed.
        if gstore is not None:
            await gstore.upsert_document(doc_id, doc.frontmatter.title or doc_id)

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
            embedding_recipe_version=recipe,
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


class RetitleResult(BaseModel):
    """Return value of `retitle_document` — the new title plus the
    per-store fan-out counts, so the caller can confirm every
    denormalized copy was updated."""

    doc_id: str
    old_title: str | None
    new_title: str
    frontmatter_changed: bool  # False when the title was already current
    fts_rows: int  # chunks_meta rows retitled
    vector_rows: int  # vector rows retitled
    graph_updated: bool  # graph node title set (False if no graph store)


async def retitle_document(doc_id: str, new_title: str) -> RetitleResult:
    """Change a document's title everywhere it is denormalized — the
    frontmatter (source of truth) plus the FTS, vector, and graph
    copies — *without* re-chunking or re-embedding.

    The title is pure metadata: `chunk.text` (the embedded input) and
    `chunk_id` (`hash(doc_id, text)`) are both functions of the body
    only, so a title change never invalidates a vector or a chunk id.
    That is what makes this cheap and safe — unlike a force-reindex,
    which deletes-then-re-embeds every chunk (wasteful, and a
    delete-before-write window if the re-embed fails).

    The frontmatter write is skipped when the title is already current,
    but the derived stores are still refreshed — so this doubles as a
    "repair a stale denormalized title" operation. Each store update is
    idempotent; on a partial failure the frontmatter remains the source
    of truth and a later `index --force` (or re-running this) reconciles.
    """
    settings = get_settings()
    correlation_id = str(ulid.ULID())
    log = logger.bind(doc_id=doc_id, correlation_id=correlation_id)

    new_title = new_title.strip()
    if not new_title:
        raise ConfigurationError(
            "retitle requires a non-empty title",
            context={"doc_id": doc_id},
        )

    doc = await read_document(settings.vault_path, doc_id)
    old_title = doc.frontmatter.title
    log.info("retitle.start", old_title=old_title, new_title=new_title)

    # 1. Frontmatter (source of truth) — only rewrite if it actually
    #    changed, to keep the content sha stable on a no-op / repair run.
    frontmatter_changed = old_title != new_title
    if frontmatter_changed:
        updated = VaultDocument(
            ref=make_ref(
                settings.vault_path,
                doc_id,
                content_sha256=doc.ref.content_sha256,  # write recomputes
                source_path=doc.ref.source_path,
            ),
            frontmatter=doc.frontmatter.model_copy(update={"title": new_title}),
            body=doc.body,
            mtime_ns=doc.mtime_ns,
        )
        written = await write_document(settings.vault_path, updated)
        await update_manifest(
            settings.vault_path,
            doc_id,
            content_sha256=written.content_sha256,
            correlation_id=correlation_id,
        )

    # 2. Derived stores — metadata-only column updates, no re-embed.
    async with AsyncExitStack() as stack:
        vstore = await VectorStore.open(settings.vault_path)
        stack.push_async_callback(vstore.close)
        fstore = await FTSStore.open(settings.vault_path)
        stack.push_async_callback(fstore.close)
        gstore = await _open_graph(settings.vault_path)
        if gstore is not None:
            stack.push_async_callback(gstore.close)

        fts_rows = await fstore.update_document_title(doc_id, new_title)
        vector_rows = await vstore.update_document_title(doc_id, new_title)
        if gstore is not None:
            await gstore.upsert_document(doc_id, new_title)

    log.info(
        "retitle.done",
        frontmatter_changed=frontmatter_changed,
        fts_rows=fts_rows,
        vector_rows=vector_rows,
        graph_updated=gstore is not None,
    )
    return RetitleResult(
        doc_id=doc_id,
        old_title=old_title,
        new_title=new_title,
        frontmatter_changed=frontmatter_changed,
        fts_rows=fts_rows,
        vector_rows=vector_rows,
        graph_updated=gstore is not None,
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
        tstore = await TableStore.open(settings.vault_path)
        stack.push_async_callback(tstore.close)
        gstore = await _open_graph(settings.vault_path)
        if gstore is not None:
            stack.push_async_callback(gstore.close)

        await vstore.delete_document(doc_id)
        await fstore.delete_document(doc_id)
        await tstore.delete_document(doc_id)
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
        # scope_sets.json and conversations.sqlite are deliberately NOT here either:
        # they're USER-authored (not regenerable from the markdown), so a full rebuild
        # must preserve them (the latter is the grounded multi-turn chat's history).
        derived = settings.vault_path / ".memex"
        for target in (
            "embeddings.lance",
            "search.sqlite",
            "tables.sqlite",
            "graph.ryu",
            "vlm_cache.sqlite",
            "chart_ocr_cache.sqlite",
            "asr_cache.sqlite",  # ADR-0017 parse-time ASR transcription cache (derived)
            "companion_alignments.json",  # ADR-0018 transcript↔deck alignment sidecar (derived)
        ):
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
    # Compact the vector store once after the bulk rebuild: the per-doc upserts
    # above each left fragments + deletion tombstones that a flat KNN scan would
    # otherwise read on every query (the one unbounded latency/size term). Best-
    # effort inside VectorStore.optimize — never fails the reindex.
    if processed:
        vstore = await VectorStore.open(settings.vault_path)
        try:
            await vstore.optimize()
        finally:
            await vstore.close()
    return ReindexReport(
        documents_processed=processed,
        chunks_written=chunks_total,
        failures=failures,
    )
