"""Enrich stage — entity extraction, citation resolution, wikilink insertion.

Per-chunk LLM extraction (one call for entities, one for citations,
parallelised via `asyncio.gather`); entity dedupe across the document;
citation matching against the vault's other documents via a
title/author-year/token-overlap scorer; graph upserts (`MENTIONS`,
`CITES`); high-confidence resolutions become `[[doc_id]]` wikilinks
substituted into the canonical markdown.

See GUIDELINES.md Part II "The pipeline" and IMPLEMENTATION-PLAN §1.4.
"""

from memex.enrich.citations import (
    CitationCandidate,
    CitationConfidence,
    CitationIndex,
    CitationList,
    ResolvedCitation,
    insert_wikilinks,
    make_signature,
    resolve_candidate,
)
from memex.enrich.course_refs import extract_course_references
from memex.enrich.entities import (
    Entity,
    EntityConfidence,
    EntityKind,
    EntityList,
    ExtractedEntity,
    dedupe,
    merge_entities,
)
from memex.enrich.pipeline import EnrichResult, enrich_document

__all__ = [
    "CitationCandidate",
    "CitationConfidence",
    "CitationIndex",
    "CitationList",
    "EnrichResult",
    "Entity",
    "EntityConfidence",
    "EntityKind",
    "EntityList",
    "ExtractedEntity",
    "ResolvedCitation",
    "dedupe",
    "enrich_document",
    "extract_course_references",
    "insert_wikilinks",
    "make_signature",
    "merge_entities",
    "resolve_candidate",
]
