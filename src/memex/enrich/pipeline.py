"""Enrich stage — entity extraction + citation resolution + wikilink insertion.

Per-chunk LLM calls extract `Entity` and `CitationCandidate` records;
the document-level pass deduplicates entities by `(lower(name), kind)`,
resolves citations against the vault's other documents via a heuristic
matcher (`memex.enrich.citations.resolve_candidate`), writes `MENTIONS`
+ `CITES` edges to the graph store, substitutes high-confidence
`[[doc_id]]` wikilinks into the canonical markdown, and updates the
manifest's `EnrichStage` with `entity_count`, `citation_count`,
`wikilinks_inserted`, and the prompt versions in play.

Citation resolution doesn't go through the LLM a second time — the
graph + frontmatter authoritatively own the matching. See
`memex/enrich/citations.py` for the resolver design (title /
author-year / token-overlap scoring with a 0.70 threshold).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import structlog
import ulid
from pydantic import BaseModel, Field

from memex.core.config import get_settings
from memex.core.errors import ModelCallError
from memex.core.manifest import (
    ChartExtraction,
    EnrichStage,
    now_utc,
    read_manifest,
    update_manifest,
)
from memex.core.table_linearize import linearize_gfm_tables
from memex.core.text import reattach_chart_extractions
from memex.core.types import Chunk
from memex.enrich.citations import (
    CitationCandidate,
    CitationIndex,
    CitationList,
    CitationListCompact,
    ResolvedCitation,
    insert_wikilinks,
    make_signature,
    resolve_candidate,
)
from memex.enrich.course_refs import extract_course_references
from memex.enrich.entities import (
    Entity,
    EntityList,
    EntityListCompact,
    dedupe,
    merge_entities,
)
from memex.enrich.ner_otter import extract_chunk_entities, otter_backend_enabled
from memex.index.chunker import chunk_document
from memex.index.graph_store import GraphStore, open_graph_for_write
from memex.models.client import complete_structured
from memex.prompts import active_version, prompt_tag_for, render_prompt
from memex.vault.store import (
    VaultDocument,
    list_documents,
    make_ref,
    read_document,
    write_document,
)

logger = structlog.get_logger(__name__)

_ENTITY_PROMPT_NAME = "extract_entities"
# The active extract_entities prompt truncates the passage (`{{ passage | truncate(6000) }}`)
# so an oversized chunk (a big table that escaped the chunker cap, or an un-migrated vault
# doc) can't blow the entity-extraction context window and fail the whole chunk. See the
# chunker `MAX_CHUNK_MULTIPLIER` cap, the primary fix.
_CITATION_PROMPT_NAME = "extract_citations"
# Per-call output budget for entity/citation extraction, kept safely under the
# 6144 orchestrator model-len: a dense passage prompt runs ~1800-2700 tokens
# (the `truncate(6000)`-char passage tokenises denser for numeric tables), so
# 3072 completion + worst-case prompt stays < 6144. We therefore CANNOT cure a
# truncation by raising this — it would overflow the model-len. The densest
# numeric chunks, whose bounded 24-item JSON can still exceed 3072 and truncate
# (ModelCallError "did not match the requested schema"), are instead handled by
# the compact-schema retry in `_extract_with_fallback` below.
_ENRICH_MAX_TOKENS = 3072

_MAX_CONCURRENT = 4  # per-chunk extraction parallelism


class EnrichResult(BaseModel):
    """Summary of one enrich run — entities + citations extracted,
    wikilinks inserted into the markdown, total chunks touched. Used
    as the return type of `enrich_document` and serialized into the
    manifest's `EnrichStage`."""

    doc_id: str
    correlation_id: str
    entities: list[Entity] = Field(default_factory=list[Entity])
    citations: list[ResolvedCitation] = Field(default_factory=list[ResolvedCitation])
    wikilinks_inserted: int = 0
    chunk_count: int = 0
    duration_ms: int = 0


async def _build_citation_index(vault_path: Path, *, skip_doc_id: str) -> CitationIndex:
    """Index every other document in the vault for citation matching.

    The body is passed to `make_signature` so each `DocSignature` carries
    its target's Markdown headings — feeds the P4.1 section-anchor
    writer in `insert_wikilinks`.
    """
    idx = CitationIndex()
    async for ref in list_documents(vault_path):
        if ref.doc_id == skip_doc_id:
            continue
        doc = await read_document(vault_path, ref.doc_id)
        idx.by_id[ref.doc_id] = make_signature(ref.doc_id, doc.frontmatter, doc.body)
    return idx


async def _extract_with_fallback[T: BaseModel](
    *, prompt: str, full_schema: type[T], compact_schema: type[T], prompt_tag: str
) -> T:
    """One structured extraction call, with a compact-schema retry.

    On a dense chunk the full (max_length=24) list's JSON can run past
    `_ENRICH_MAX_TOKENS` and truncate, so `complete_structured` raises
    `ModelCallError` ("did not match the requested schema"). Raising
    `max_tokens` can't fix it — the 6144 model-len already bounds
    prompt+completion — so retry ONCE with the half-cap `compact_schema`,
    whose worst-case output fits with margin: the chunk enriches with its
    top dozen items instead of being dropped. A second failure propagates
    to the caller's `gather`, which logs `enrich.chunk_failed`.
    """
    try:
        result, _ = await complete_structured(
            prompt=prompt, schema=full_schema, prompt_tag=prompt_tag, max_tokens=_ENRICH_MAX_TOKENS
        )
        return result
    except ModelCallError:
        logger.info("enrich.compact_retry", prompt_tag=prompt_tag)
        result, _ = await complete_structured(
            prompt=prompt,
            schema=compact_schema,
            prompt_tag=prompt_tag,
            max_tokens=_ENRICH_MAX_TOKENS,
        )
        return result


async def _extract_chunk(chunk: Chunk, title: str) -> tuple[list[Entity], list[CitationCandidate]]:
    """Entities + citations for one chunk, in parallel.

    Entities come from the LLM (with a compact-schema fallback so a dense chunk that
    truncates still enriches with its top items) OR, when `enrich_ner_backend="otter"`,
    from the OTTER span NER (`ner_otter.extract_chunk_entities`, returning `list[Entity]`
    directly — the same shape `merge_entities` produces, so the document-level dedupe +
    graph write are unchanged). Citations ALWAYS stay on the LLM (OTTER does NER only).
    The LLM calls go through `complete_structured`, generic over `schema`, so pyright
    keeps the chain typed.
    """
    citation_prompt = render_prompt(_CITATION_PROMPT_NAME, document_title=title, passage=chunk.text)
    citation_task = _extract_with_fallback(
        prompt=citation_prompt,
        full_schema=CitationList,
        compact_schema=CitationListCompact,
        prompt_tag=prompt_tag_for(_CITATION_PROMPT_NAME),
    )

    if otter_backend_enabled():
        entities, citation_raw = await asyncio.gather(extract_chunk_entities(chunk), citation_task)
    else:
        entity_prompt = render_prompt(_ENTITY_PROMPT_NAME, document_title=title, passage=chunk.text)
        entity_task = _extract_with_fallback(
            prompt=entity_prompt,
            full_schema=EntityList,
            compact_schema=EntityListCompact,
            prompt_tag=prompt_tag_for(_ENTITY_PROMPT_NAME),
        )
        entity_raw, citation_raw = await asyncio.gather(entity_task, citation_task)
        # `complete_structured` is typed to return the schema instance (the compact
        # variants subclass the full schema), so this can only fire if a test fake
        # breaks the contract — surface explicitly rather than crash.
        if not isinstance(entity_raw, EntityList):  # type: ignore[reportUnnecessaryIsInstance]  # runtime contract guard
            raise ModelCallError(
                "Entity extraction returned unexpected payload type",
                context={"got": type(entity_raw).__name__, "expected": "EntityList"},
            )
        entities = merge_entities(chunk, entity_raw)

    if not isinstance(citation_raw, CitationList):  # type: ignore[reportUnnecessaryIsInstance]  # runtime contract guard
        raise ModelCallError(
            "Citation extraction returned unexpected payload type",
            context={"got": type(citation_raw).__name__, "expected": "CitationList"},
        )
    return entities, list(citation_raw.citations)


async def enrich_document(doc_id: str) -> EnrichResult:
    """Extract entities + citations, write graph edges + wikilinks +
    manifest.

    Wikilinks are inserted only for high-confidence resolutions; the
    write goes through `vault.write_document` (atomic) and the
    manifest's `content_sha256` is updated in the same call so the
    watcher's `_confirm_user_edit` correctly treats the self-write as
    Memex's own (per IMPLEMENTATION-PLAN §2.3).
    """
    settings = get_settings()
    correlation_id = str(ulid.ULID())
    log = logger.bind(doc_id=doc_id, correlation_id=correlation_id)
    log.info("enrich.start")

    start = time.monotonic()

    doc = await read_document(settings.vault_path, doc_id)
    # Chunk the SAME bytes `index_document` chunks (#394): re-attach the chart-OCR
    # `[chart-extracted]` blocks from the parse manifest sidecar (NON-re-derivable from the
    # content-only `.md`) then re-derive the `[table-rows]` linearization, in parse order
    # (chart re-attach THEN table linearize). `chunk_id` is content-addressed (sha1 of chunk
    # text), so WITHOUT this a chart-bearing chunk's enrich chunk_id != index's re-attached
    # chunk_id and the MENTIONS attested chunk_id never resolves in FTS. `reattach(body, [])`
    # is identity → a no-op for the non-chart docs (back-compat). Only the chunking copy is
    # transformed: `doc.body` stays CLEAN for the course-ref + wikilink writes below, which
    # operate on / rewrite the canonical content-only `.md`. We do NOT thread the manifest's
    # page_char_counts (unlike index_document) — `chunk_id` is content-addressed on text only,
    # page-independent, so it's irrelevant to the attestation parity this fixes.
    prior_manifest = await read_manifest(settings.vault_path, doc_id)
    chart_extractions: list[ChartExtraction] = []
    if prior_manifest is not None and prior_manifest.parse is not None:
        chart_extractions = prior_manifest.parse.chart_extractions
    reattached_body = reattach_chart_extractions(doc.body, chart_extractions)
    chunking_doc = doc.model_copy(update={"body": linearize_gfm_tables(reattached_body)})
    chunks = chunk_document(chunking_doc)
    title = doc.frontmatter.title or doc_id

    # Citation index excludes the current doc (we don't link to ourself).
    citation_index = await _build_citation_index(settings.vault_path, skip_doc_id=doc_id)

    semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

    async def _run(c: Chunk) -> tuple[list[Entity], list[CitationCandidate]]:
        async with semaphore:
            return await _extract_chunk(c, title)

    # `return_exceptions=True` so one bad chunk doesn't abort the whole
    # document mid-enrich. We log per-chunk failures, treat them as
    # "this chunk contributed no entities and no citations," and let
    # the partial result through. Aborting the whole doc would leave
    # the graph in a half-enriched state (some chunks' MENTIONS/CITES
    # written, others not).
    raw_results = await asyncio.gather(*(_run(c) for c in chunks), return_exceptions=True)

    # Flatten entities + citations, keep chunk attribution on candidates.
    entities_flat: list[Entity] = []
    citation_candidates: list[tuple[str, CitationCandidate]] = []
    chunk_failures = 0
    for chunk, result in zip(chunks, raw_results, strict=True):
        if isinstance(result, BaseException):
            chunk_failures += 1
            log.warning(
                "enrich.chunk_failed",
                chunk_id=chunk.chunk_id,
                error_type=type(result).__name__,
                error=str(result)[:200],
            )
            continue
        ents, cits = result
        entities_flat.extend(ents)
        for c in cits:
            citation_candidates.append((chunk.chunk_id, c))

    entities = dedupe(entities_flat)

    # Resolve citations against the vault index. Each candidate resolves
    # independently; unresolved candidates are dropped from the output
    # (they go into structlog at debug level for forensics).
    resolved: list[ResolvedCitation] = []
    for chunk_id, candidate in citation_candidates:
        match = resolve_candidate(candidate, chunk_id, citation_index, skip_doc_id=doc_id)
        if match is not None:
            resolved.append(match)
        else:
            log.debug(
                "enrich.citation_unresolved",
                surface=candidate.surface_text[:80],
                model_confidence=candidate.confidence,
            )

    # Course cross-references: a pure-text pass that resolves "Cours N"
    # / "Semaine N" mentions to sibling lecture docs in the same course
    # series (no LLM). These fire on lecture/syllabus content where the
    # academic-citation resolver emits nothing — the syllabus's course
    # program is the natural hub linking to each lecture.
    course_refs = extract_course_references(
        doc.body,
        self_doc_id=doc_id,
        self_title=title,
        index=citation_index,
    )
    if course_refs:
        log.info("enrich.course_refs", count=len(course_refs))
        resolved.extend(course_refs)

    # Wikilink insertion: rewrites the body atomically.
    # `target_index` enables P4.1 section-anchor emission — when a
    # citation's surrounding context mentions a heading of the target
    # doc, the wikilink becomes `[[doc#heading]]` instead of `[[doc]]`.
    new_body, wikilinks_inserted = insert_wikilinks(doc.body, resolved, target_index=citation_index)
    if new_body != doc.body:
        updated_ref = make_ref(
            settings.vault_path,
            doc_id,
            content_sha256=doc.ref.content_sha256,  # placeholder; write recomputes
            source_path=doc.ref.source_path,
        )
        updated = VaultDocument(
            ref=updated_ref,
            frontmatter=doc.frontmatter,
            body=new_body,
            mtime_ns=doc.mtime_ns,
        )
        written = await write_document(settings.vault_path, updated)
        new_content_sha = written.content_sha256
    else:
        new_content_sha = doc.ref.content_sha256

    # Write to the graph. We open + close per call so concurrent enrich runs don't share a
    # connection. `open_graph_for_write` adds the cross-process lock policy: a brief reader
    # race (a webui discovery read holding ryugraph's exclusive dir lock) is RETRIED then
    # re-raised — a writer must not silently skip the write (that would drop this doc's
    # entities). ryugraph-absent → None (the graph is optional; enrich output still lands in
    # the vault). The `GraphStore` annotation keeps the class importable as the test seam.
    graph: GraphStore | None = await open_graph_for_write(settings.vault_path)

    if graph is not None:
        try:
            await graph.upsert_document(doc_id, title)
            # REPLACE, don't append: clear this doc's prior MENTIONS so a re-enrich (esp. a
            # backend switch to OTTER, whose entity set differs wholesale) doesn't leave the
            # previous extractor's stale entities alongside the new ones.
            await graph.clear_mentions(doc_id)
            for ent in entities:
                eid = await graph.upsert_entity(ent.name, ent.kind)
                # A representative attested chunk (the first the entity was found in; dedupe
                # unions chunk_ids first-seen-first) so discovery can surface the exact passage.
                rep_chunk = ent.chunk_ids[0] if ent.chunk_ids else None
                await graph.link_mentions(doc_id, eid, ent.confidence, chunk_id=rep_chunk)
            # Ensure each DISTINCT target document node exists before linking —
            # upsert once per target (not once per citation), so a doc cited N
            # times doesn't re-MERGE the same node + emit N upsert log lines.
            targets = {cit.target_doc_id: cit.target_title for cit in resolved}
            for tid, ttitle in targets.items():
                await graph.upsert_document(tid, ttitle)
            for cit in resolved:
                await graph.link_cites(
                    from_doc_id=doc_id,
                    to_doc_id=cit.target_doc_id,
                    surface_text=cit.surface_text[:200],
                    confidence=cit.confidence,
                )
        finally:
            await graph.close()

    duration_ms = int((time.monotonic() - start) * 1000)
    await update_manifest(
        settings.vault_path,
        doc_id,
        content_sha256=new_content_sha,
        enrich=EnrichStage(
            correlation_id=correlation_id,
            enriched_at=now_utc(),
            entity_count=len(entities),
            citation_count=len(resolved),
            wikilinks_inserted=wikilinks_inserted,
            prompt_versions={
                _ENTITY_PROMPT_NAME: active_version(_ENTITY_PROMPT_NAME),
                _CITATION_PROMPT_NAME: active_version(_CITATION_PROMPT_NAME),
            },
            duration_ms=duration_ms,
        ),
        correlation_id=correlation_id,
    )

    log.info(
        "enrich.done",
        chunk_count=len(chunks),
        chunk_failures=chunk_failures,
        entity_count=len(entities),
        citation_count=len(resolved),
        wikilinks_inserted=wikilinks_inserted,
        duration_ms=duration_ms,
    )

    return EnrichResult(
        doc_id=doc_id,
        correlation_id=correlation_id,
        entities=entities,
        citations=resolved,
        wikilinks_inserted=wikilinks_inserted,
        chunk_count=len(chunks),
        duration_ms=duration_ms,
    )
