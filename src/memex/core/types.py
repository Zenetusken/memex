"""Shared cross-module types.

When a module needs to expose a type that other modules import, it
goes here rather than into the producing module — keeps the import
graph acyclic and explicit. Module-internal types stay in their own
module.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrievable unit of document content with citation metadata.

    The same shape is produced by `memex.index` (when chunking a
    document for storage), consumed by `memex.retrieve` (BM25 + dense
    candidates + reranked results), and threaded through the answering
    agent's state. Lives here so neither side has to import from the
    other.
    """

    chunk_id: str
    document_id: str
    document_title: str
    text: str
    page: int | None = None
    char_start: int = 0
    char_end: int = 0
    score: float = 0.0
    rerank_score: float | None = None
    heading_path: list[str] = Field(default_factory=list)
