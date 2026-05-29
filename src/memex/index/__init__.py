"""Index stage — writers for LanceDB (vectors), SQLite FTS5 (BM25), RyuGraph (graph).

This stage is idempotent and fully regenerable from the Markdown vault
(ADR-0003). `memex reindex` deletes the index files and rebuilds from
`vault/documents/`. Content-derived chunk IDs (`sha1(text)[:10]`)
enable incremental partial re-indexing: a one-paragraph edit only
rewrites the changed chunk, not the whole document. See GUIDELINES.md
Part II and Part IV.
"""

from memex.index.chunker import chunk_document
from memex.index.embed_prompts import (
    EMBED_QUERY_PROMPT_NAME,
    chunk_title,
    document_input,
    native_prompts_enabled,
)
from memex.index.fts_store import FTSStore
from memex.index.graph_store import (
    CoOccurringEntity,
    EntityMention,
    EntityProfile,
    GraphNeighbor,
    GraphStore,
    RelatedDocument,
)
from memex.index.pipeline import (
    IndexResult,
    ReindexReport,
    RetitleResult,
    index_document,
    reindex_vault,
    remove_document,
    retitle_document,
)
from memex.index.table_store import TableStore, extract_tables
from memex.index.vector_store import EMBEDDING_DIM, VectorStore

__all__ = [
    "EMBEDDING_DIM",
    "EMBED_QUERY_PROMPT_NAME",
    "CoOccurringEntity",
    "EntityMention",
    "EntityProfile",
    "FTSStore",
    "GraphNeighbor",
    "GraphStore",
    "IndexResult",
    "ReindexReport",
    "RelatedDocument",
    "RetitleResult",
    "TableStore",
    "VectorStore",
    "chunk_document",
    "chunk_title",
    "document_input",
    "extract_tables",
    "index_document",
    "native_prompts_enabled",
    "reindex_vault",
    "remove_document",
    "retitle_document",
]
