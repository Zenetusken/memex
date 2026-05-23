"""Entity extraction — one LLM call per chunk, dedup across the document.

The extractor returns `Entity` instances with name, kind, confidence,
and (when locatable) a character span into the markdown body. The
pipeline aggregates per-chunk results, normalizes the entity ID,
dedupes by `(kind, lower(name))`, and writes the result to the graph
store via `MENTIONS` edges.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from memex.core.types import Chunk

EntityKind = Literal[
    "person", "org", "place", "concept", "method", "tool", "other"
]
EntityConfidence = Literal["high", "medium", "low"]


_CONFIDENCE_TO_FLOAT: dict[EntityConfidence, float] = {
    "high": 0.95,
    "medium": 0.70,
    "low": 0.45,
}


class ExtractedEntity(BaseModel):
    """Model-emitted entity record."""

    # Hardening (audit 2026-05-22 follow-up to v6 schema bound): cap
    # the LLM-emit strings. Entity `name` legitimately runs ~3-50
    # chars; allow some headroom for full names of organisations or
    # multi-word concepts. `span_text` should be a quoted phrase from
    # the passage — 200 chars covers a long sentence fragment.
    name: str = Field(min_length=1, max_length=120)
    kind: EntityKind
    confidence: EntityConfidence
    span_text: str = Field(default="", max_length=200)


class EntityList(BaseModel):
    """Top-level output schema for the extract_entities prompt."""

    entities: list[ExtractedEntity] = Field(default_factory=list)


class EntityExtractionInput(BaseModel):
    """Documented input shape for the extract_entities prompt."""

    document_title: str
    passage: str


class Entity(BaseModel):
    """Post-processed entity, ready for the graph + manifest."""

    name: str
    kind: EntityKind
    confidence: float
    chunk_ids: list[str] = Field(default_factory=list)
    char_span: tuple[int, int] | None = None


def _key(name: str, kind: EntityKind) -> tuple[str, EntityKind]:
    return (name.strip().lower(), kind)


def merge_entities(
    chunk: Chunk, raw: EntityList
) -> list[Entity]:
    """Per-chunk: convert ExtractedEntity → Entity, locating spans within
    the chunk text.
    """
    out: list[Entity] = []
    for ex in raw.entities:
        span: tuple[int, int] | None = None
        if ex.span_text:
            idx = chunk.text.find(ex.span_text)
            if idx >= 0:
                span = (
                    chunk.char_start + idx,
                    chunk.char_start + idx + len(ex.span_text),
                )
        out.append(
            Entity(
                name=ex.name.strip(),
                kind=ex.kind,
                confidence=_CONFIDENCE_TO_FLOAT[ex.confidence],
                chunk_ids=[chunk.chunk_id],
                char_span=span,
            )
        )
    return out


def dedupe(entities: list[Entity]) -> list[Entity]:
    """Document-level merge by `(lower(name), kind)`.

    The first-seen `name` (canonical case as the model returned it) is
    preserved; confidences are max'd; chunk_ids are unioned.
    """
    merged: dict[tuple[str, EntityKind], Entity] = {}
    for e in entities:
        k = _key(e.name, e.kind)
        if k not in merged:
            merged[k] = e.model_copy()
            continue
        existing = merged[k]
        existing.confidence = max(existing.confidence, e.confidence)
        for cid in e.chunk_ids:
            if cid not in existing.chunk_ids:
                existing.chunk_ids.append(cid)
        # First-seen char_span wins (it's an example, not a unique location).
    return list(merged.values())
