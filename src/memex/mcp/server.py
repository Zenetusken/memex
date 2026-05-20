"""MCP server — the public API of Memex (see IMPLEMENTATION-PLAN §1.9).

Five tools, all backed by code that Phases 0–4 shipped:

- `search(query, k)` → hybrid retrieval + rerank
- `ask(question)` → the full answering agent (Langfuse-traced)
- `get_document(doc_id)` → canonical markdown + frontmatter
- `list_documents()` → every DocumentRef in the vault (no bodies)
- `get_graph_neighbors(doc_id, limit)` → one-hop graph neighbours

Transports: stdio for desktop MCP clients (Claude Code, Cursor, ...) and
streamable HTTP for network-local agents (behind `--transport http`).
HTTP has no auth — bind to localhost only and treat it as a developer
affordance until a network-edge auth model lands.

The tool functions are also exported at module top-level so tests can
exercise them directly without spinning up an MCP transport.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from mcp.server.fastmcp import FastMCP

from memex.agents.answering import FinalResponse, answer_query
from memex.core.config import get_settings
from memex.core.types import Chunk
from memex.retrieve import cross_encoder_rerank, hybrid_search
from memex.vault.store import (
    DocumentRef,
    VaultDocument,
    list_documents,
    read_document,
)

if TYPE_CHECKING:
    from memex.index.graph_store import GraphNeighbor

logger = structlog.get_logger(__name__)

# The FastMCP instance is module-level so tool registration is a side effect
# of import. CLI lifecycle calls `serve_stdio` / `serve_http` on this same
# instance.
server = FastMCP(name="memex")


# ----- Tools -----


async def search(query: str, k: int = 10) -> list[Chunk]:
    """Hybrid search over the vault: BM25 (FTS5) ⊕ dense (LanceDB) → RRF
    → cross-encoder rerank. Returns the top `k` chunks with their
    document title, page, heading path, and rerank score.

    Returns typed `Chunk` instances rather than dicts — the MCP
    transport serializes pydantic models to JSON on the wire while
    Python callers get a fully-typed object back.

    Use this when you want raw retrieval; use `ask` for a grounded
    answer.
    """
    log = logger.bind(tool="search", query_len=len(query), k=k)
    log.info("mcp.tool.start")
    candidates = await hybrid_search(query, k=max(50, k * 5))
    reranked = await cross_encoder_rerank(query, candidates, top_k=k)
    log.info("mcp.tool.done", returned=len(reranked))
    return reranked


async def ask(question: str) -> FinalResponse:
    """Answer `question` using the full agent: retrieve → rerank →
    assess → answer → verify, with mandatory grounding before any
    claim is returned. A refusal is a first-class outcome.

    The returned `FinalResponse` includes `correlation_id`, which is
    also the Langfuse trace id — paste it into the Langfuse UI to see
    every chunk retrieved, every model call made, and every claim
    verified.
    """
    log = logger.bind(tool="ask")
    log.info("mcp.tool.start")
    response = await answer_query(question)
    log.info(
        "mcp.tool.done",
        answered=response.answered,
        correlation_id=response.correlation_id,
    )
    return response


async def get_document(doc_id: str) -> VaultDocument:
    """Fetch a vault document by its content-derived `doc_id`.

    Returns the canonical markdown body, the YAML frontmatter parsed
    into structured fields, and the `DocumentRef` (paths, content
    hash).
    """
    settings = get_settings()
    log = logger.bind(tool="get_document", doc_id=doc_id)
    log.info("mcp.tool.start")
    doc = await read_document(settings.vault_path, doc_id)
    log.info("mcp.tool.done")
    return doc


async def list_documents_tool() -> list[DocumentRef]:
    """List every document in the vault. Cheap — does not load bodies.

    Exposed as `list_documents` over MCP; the suffix on the Python
    name avoids shadowing the import from `memex.vault.store`.
    """
    settings = get_settings()
    log = logger.bind(tool="list_documents")
    log.info("mcp.tool.start")
    refs: list[DocumentRef] = []
    async for ref in list_documents(settings.vault_path):
        refs.append(ref)
    log.info("mcp.tool.done", count=len(refs))
    return refs


async def get_graph_neighbors(
    doc_id: str, limit: int = 50
) -> list["GraphNeighbor"]:
    """Documents one hop away in the entity graph.

    Surfaces documents that share entities with `doc_id` via the
    RyuGraph store (`shares_entity` relation). Returns up to `limit`
    neighbors with `doc_id`, `title`, the relation kind, and the
    `via` field (the shared entity name) when present.

    CITES-edge traversal is written by the enrich stage but not yet
    read by this tool — see `index/graph_store.py:neighbors` for the
    Cypher and the v1.x backlog for the extension.

    Returns an empty list if RyuGraph isn't installed; the graph is
    optional for retrieval.
    """
    from memex.index.graph_store import GraphStore

    settings = get_settings()
    log = logger.bind(tool="get_graph_neighbors", doc_id=doc_id)
    log.info("mcp.tool.start")
    try:
        store = await GraphStore.open(settings.vault_path)
    except ImportError as e:
        log.warning("mcp.graph_unavailable", reason=str(e))
        return []
    try:
        neighbors = await store.neighbors(doc_id, limit=limit)
    finally:
        await store.close()
    log.info("mcp.tool.done", count=len(neighbors))
    return neighbors


# Register with FastMCP. We do this after defining the functions so the
# above docstrings show up verbatim in MCP tool-introspection responses.
server.tool()(search)
server.tool()(ask)
server.tool()(get_document)
server.tool(name="list_documents")(list_documents_tool)
server.tool()(get_graph_neighbors)


# ----- Transports -----


async def serve_stdio() -> None:
    """Run the MCP server over stdio. Blocks until the client disconnects."""
    logger.info("mcp.serve.stdio.start")
    await server.run_stdio_async()


async def serve_http(host: str = "127.0.0.1", port: int = 7424) -> None:
    """Run the MCP server over streamable HTTP. Bind to localhost only
    unless you have a Phase 4 auth model in place — there is no
    authentication on this transport today.
    """
    # FastMCP exposes its uvicorn config via `.settings`; mutate before
    # calling `run_streamable_http_async`.
    server.settings.host = host
    server.settings.port = port
    logger.info("mcp.serve.http.start", host=host, port=port)
    await server.run_streamable_http_async()
