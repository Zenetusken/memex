"""Per-document manifest — the audit artifact per IMPLEMENTATION-PLAN §2.4.

A manifest accumulates one section per pipeline stage that touches a
document (ingest, parse, enrich, index). Each section records the
correlation_id under which it ran, the model/prompt versions in play,
and a duration. Manifests live at
`vault/.memex/manifests/{doc_id}.json` and are written atomically.

`memex doctor` walks manifests to find stale state (markdown mtime
later than the recorded index time, etc.).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from memex.core.errors import VaultIntegrityError


class IngestStage(BaseModel):
    """Audit record for the ingest step — what file came in, how big,
    what mime, and (when rejected) why."""

    correlation_id: str
    ingested_at: datetime
    source_path: str
    source_size_bytes: int
    detected_mime: str
    rejection_reason: str | None = None


class PageDecision(BaseModel):
    """Per-page parse engine routing record.

    `char_count` is the per-page markdown length AT JOIN TIME (before any
    post-stitch transforms like chart-OCR or GFM-table linearization) — used
    by the chunker to attribute each chunk to its dominant source page (for
    the click-source→jump-to-PDF-page UX in the webui). `0` is the legacy
    default: docs parsed before this field was added carry `char_count=0`
    across all pages, which the chunker reads as "page mapping unavailable"
    and falls back to section-anchor-only navigation (no regression). Newly
    parsed/re-parsed docs populate it.

    `char_start` is the page's CITATION-GRADE start offset in the FINAL `doc.body`
    (post-`_finalize_body`), recorded via the page-boundary marker round-trip
    (companion arc-3). `-1` is the legacy/unknown default: a manifest where ANY
    page is `-1` keeps the nav-grade `char_count`-derived attribution (drifts on
    figure-heavy decks); a manifest where EVERY page has `char_start >= 0` lets
    `index_document` map these boundaries through the chart-reattach/linearize
    transforms to attribute `Chunk.page` against the exact chunked body. A re-parse
    upgrades a legacy doc.
    """

    page: int
    engine: Literal["docling", "vlm", "passthrough", "pymupdf", "scan", "image"]
    confidence: float
    rationale: str = ""
    duration_ms: int = 0
    char_count: int = 0
    char_start: int = -1


class ChartExtraction(BaseModel):
    """One chart-OCR block, stored in the parse manifest sidecar (audit-10 W-series follow-on).

    The canonical vault `.md` is content-only (a `<!-- image -->` placeholder where each chart
    was), so the chart-OCR output — which is NON-re-derivable from the `.md` (the chart image is
    gone; the OCR is non-deterministic + cached) — is persisted here and RE-ATTACHED at index time
    (`core/text.reattach_chart_extractions`) at the placeholder positions, reproducing the historical
    stitched body byte-for-byte so chunk_ids stay stable. The `[table-rows]` pattern, sidecar-backed.

    `placeholder_index` is the 0-based ordinal of the `<!-- image -->` placeholder (in document
    order) this block attaches AFTER; `markdown` is the verbatim chart-OCR markdown (what the old
    parse stitch inserted). A skipped figure (empty/errored extraction) emits NO entry, so the
    stored blocks are a SUBSET of placeholders — the explicit ordinal keeps each pinned exactly.
    """

    placeholder_index: int
    markdown: str


class TranscriptSegment(BaseModel):
    """One ASR transcript segment — the audio analogue of `PageDecision`, stored in the parse
    manifest sidecar (audio route, ADR-0017 / `docs/specs/audio-asr-route.md`).

    Audio has no pages, so the audio route leaves `ParseStage.pages` empty and records these
    instead. `start_s`/`end_s` are **global** seconds vs the whole file (the anchor the eventual
    companion-slide-merge aligns on); `char_start`/`char_end` is the segment's span in the
    canonical `.md` AT JOIN TIME (pre-transform), used by the chunker to attribute each
    transcript chunk a time range (the time analogue of `PageDecision.char_count → Chunk.page`).
    A failed/empty chunk is recorded with `confidence=0.0` + the error in `rationale` — never a
    silent drop. The companion-deck link is DOCUMENT-level (a wikilink / CITES edge), NOT a field
    here (a standalone ingest has no companion, and the merge aligns on the time-range)."""

    index: int
    char_start: int
    char_end: int
    start_s: float
    end_s: float
    language: str = ""
    confidence: float = 1.0
    rationale: str = ""


class ParseStage(BaseModel):
    """Audit record for the parse step — per-page routing decisions
    (Docling / VLM / PyMuPDF / passthrough), counts of figures /
    tables / equations, and any crash diagnostics."""

    correlation_id: str
    parsed_at: datetime
    parser_version: str
    docling_version: str | None = None
    pymupdf_version: str | None = None
    pages: list[PageDecision] = Field(default_factory=list[PageDecision])
    figure_count: int = 0
    table_count: int = 0
    equation_count: int = 0
    duration_ms: int = 0
    crashed: bool = False
    crash_message: str | None = None
    # Chart-OCR sidecar (audit-10 follow-on): the chart blocks stripped from the canonical
    # `.md` at parse, re-attached at index time. Empty default → legacy manifests load
    # unchanged + the re-attach is a no-op for non-chart docs.
    chart_extractions: list[ChartExtraction] = Field(default_factory=list[ChartExtraction])
    # Audio route (ADR-0017): per-transcript-segment records (timestamps + char-spans). Empty
    # for non-audio docs; legacy manifests load unchanged (the re-attach/chunker no-op without it).
    segments: list[TranscriptSegment] = Field(default_factory=list[TranscriptSegment])


class EnrichStage(BaseModel):
    """Audit record for the enrich step — entity + citation counts,
    wikilinks inserted into the markdown, and the prompt versions
    used (so an eval regression can be tied to a specific prompt
    change)."""

    correlation_id: str
    enriched_at: datetime
    entity_count: int
    citation_count: int = 0
    wikilinks_inserted: int = 0
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    duration_ms: int = 0


class IndexStage(BaseModel):
    """Audit record for the index step — what embedder was used, how
    many chunks landed, and the partial-reindex diff (chunks added /
    deleted / unchanged)."""

    correlation_id: str
    indexed_at: datetime
    embedding_model: str
    embedding_dim: int
    # Embedding-recipe tag (e.g. native-prompt toggle). Back-compat default
    # "v0" matches on-disk manifests; a mismatch vs the current recipe
    # auto-forces a full re-embed (see pipeline._embed_recipe_version).
    embedding_recipe_version: str = "v0"
    chunk_count: int
    duration_ms: int = 0
    # Incremental re-indexing breakdown. Defaults to 0 so existing
    # manifests (written before partial re-indexing landed) load fine.
    chunks_added: int = 0
    chunks_deleted: int = 0
    chunks_unchanged: int = 0
    partial: bool = False  # True if this was a diff-based re-index


class Manifest(BaseModel):
    """The audit record for a single vault document.

    Stages are optional because the pipeline is incremental — a freshly
    ingested doc has only `ingest`; a re-index produces `ingest +
    parse + index`; etc. Every stage records the `correlation_id` it
    ran under so logs and Langfuse traces can be joined.
    """

    doc_id: str
    content_sha256: str
    correlation_ids: list[str] = Field(default_factory=list)
    ingest: IngestStage | None = None
    parse: ParseStage | None = None
    enrich: EnrichStage | None = None
    index: IndexStage | None = None

    def add_correlation_id(self, correlation_id: str) -> None:
        """Append a new run's correlation_id to the manifest's
        history list, skipping duplicates. Lets a single doc's full
        ingest/parse/enrich/index trace be reconstructed from
        Langfuse later."""
        if correlation_id not in self.correlation_ids:
            self.correlation_ids.append(correlation_id)


def _manifest_path(vault_path: Path, doc_id: str) -> Path:
    return vault_path / ".memex" / "manifests" / f"{doc_id}.json"


async def read_manifest(vault_path: Path, doc_id: str) -> Manifest | None:
    """Load a manifest if it exists; return None otherwise."""
    path = _manifest_path(vault_path, doc_id)
    if not path.exists():
        return None
    text = await asyncio.to_thread(path.read_text, encoding="utf-8")
    return Manifest.model_validate_json(text)


async def write_manifest(vault_path: Path, manifest: Manifest) -> Path:
    """Atomically write a manifest to disk."""
    path = _manifest_path(vault_path, manifest.doc_id)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    content = manifest.model_dump_json(indent=2, exclude_none=True)

    def _write() -> None:
        fd, tmp = tempfile.mkstemp(prefix=f"{path.name}.tmp.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    await asyncio.to_thread(_write)
    return path


async def delete_manifest(vault_path: Path, doc_id: str) -> None:
    """Remove a document's manifest file (missing-ok).

    Used by `memex remove` so that re-ingesting the same source later starts
    from a clean parse/index state rather than inheriting the removed
    document's stage record."""
    path = _manifest_path(vault_path, doc_id)
    await asyncio.to_thread(path.unlink, missing_ok=True)


async def update_manifest(
    vault_path: Path,
    doc_id: str,
    *,
    content_sha256: str | None = None,
    ingest: IngestStage | None = None,
    parse: ParseStage | None = None,
    enrich: EnrichStage | None = None,
    index: IndexStage | None = None,
    correlation_id: str | None = None,
) -> Manifest:
    """Read, mutate, write. Creates the manifest if it doesn't exist.

    Idempotent on each field — passing the same `parse` block twice
    overwrites with the new one. The `correlation_id` list grows on
    every call that provides one.
    """
    existing = await read_manifest(vault_path, doc_id)
    if existing is None:
        if content_sha256 is None:
            raise VaultIntegrityError(
                "creating a new manifest requires content_sha256",
                context={"doc_id": doc_id, "vault_path": str(vault_path)},
            )
        existing = Manifest(doc_id=doc_id, content_sha256=content_sha256)
    elif content_sha256 is not None:
        existing.content_sha256 = content_sha256

    if ingest is not None:
        existing.ingest = ingest
    if parse is not None:
        existing.parse = parse
    if enrich is not None:
        existing.enrich = enrich
    if index is not None:
        existing.index = index
    if correlation_id is not None:
        existing.add_correlation_id(correlation_id)

    await write_manifest(vault_path, existing)
    return existing


def now_utc() -> datetime:
    """Current time, UTC + tz-aware. Use this anywhere a manifest
    stage records `_at` timestamps so the JSON serialisation stays
    consistent across hosts."""
    return datetime.now(UTC)
