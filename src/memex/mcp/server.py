"""MCP server — the public API of Memex (see IMPLEMENTATION-PLAN §1.9).

Seven tools, all backed by code that Phases 0–4 shipped:

- `search(query, k)` → hybrid retrieval + rerank
- `ask(question, scope_doc_ids?, scope_set?)` → the full answering agent (Langfuse-traced)
- `summarize(doc_id, instruction?, detail?)` → structured grounded document summary
- `get_document(doc_id)` → canonical markdown + frontmatter
- `list_documents()` → every DocumentRef in the vault (no bodies)
- `list_scope_sets()` → every saved document scope set (name + doc ids)
- `get_graph_neighbors(doc_id, limit)` → one-hop graph neighbours

Transports: stdio for desktop MCP clients (Claude Code, Cursor, ...) and
streamable HTTP for network-local agents (behind `--transport http`).
HTTP has no auth — bind to localhost only and treat it as a developer
affordance until a network-edge auth model lands.

The tool functions are also exported at module top-level so tests can
exercise them directly without spinning up an MCP transport.
"""

from __future__ import annotations

import structlog
from mcp.server.fastmcp import FastMCP

from memex.agents.answering import FinalResponse, answer_query
from memex.agents.document_summarizer import SummaryDetail, summarize_document
from memex.core.config import get_settings
from memex.core.errors import ConfigurationError
from memex.core.scope_sets import ScopeSet, get_scope_set, list_scope_sets
from memex.core.types import Chunk
from memex.index.graph_store import GraphNeighbor, RelatedDocument
from memex.mcp.auth import BearerAuthMiddleware, validate_bind
from memex.retrieve import cross_encoder_rerank, hybrid_search
from memex.vault.store import (
    DocumentRef,
    VaultDocument,
    list_documents,
    read_document,
)

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


async def ask(
    question: str,
    scope_doc_ids: list[str] | None = None,
    scope_set: str | None = None,
) -> FinalResponse:
    """Answer `question` using the full agent: retrieve → rerank →
    assess → answer → verify, with mandatory grounding before any
    claim is returned. A refusal is a first-class outcome.

    `scope_doc_ids` (optional): restrict retrieval to these document ids — the
    answer is grounded ONLY in the named docs (a refusal results if they don't
    contain it). Omit for the full-corpus search. The applied scope is echoed on
    `FinalResponse.artifact_scope_doc_ids`.

    `scope_set` (optional): the name of a saved scope set (see `list_scope_sets`)
    whose documents are added to the scope. An unknown name raises rather than
    silently searching the whole corpus.

    The returned `FinalResponse` includes `correlation_id`, which is
    also the Langfuse trace id — paste it into the Langfuse UI to see
    every chunk retrieved, every model call made, and every claim
    verified.
    """
    log = logger.bind(tool="ask")
    scope_ids = list(scope_doc_ids or [])
    if scope_set and scope_set.strip():
        settings = get_settings()
        found = await get_scope_set(settings.vault_path, scope_set)
        if found is None:
            available = [s.name for s in await list_scope_sets(settings.vault_path)]
            raise ConfigurationError(
                f"no scope set named {scope_set!r}",
                context={"requested": scope_set, "available": available},
            )
        scope_ids.extend(found.doc_ids)
    merged = list(dict.fromkeys(scope_ids))  # ordered de-dup of doc + set overlap
    log.info("mcp.tool.start", scope_doc_ids=merged, scope_set=scope_set or "")
    response = await answer_query(question, scope_doc_ids=merged or None)
    log.info(
        "mcp.tool.done",
        answered=response.answered,
        correlation_id=response.correlation_id,
    )
    return response


async def summarize(
    doc_id: str, instruction: str | None = None, detail: str = "standard"
) -> FinalResponse:
    """Summarise a document — a structured, GROUNDED summary (ADR-0008): an
    `summary` abstract + cited `claims` (key-points) + per-section `sections`
    digests, every point grounded to a source chunk (it refuses rather than
    fabricate). `detail` tunes length: "brief" | "standard" | "detailed" |
    "report" (a multi-paragraph body). Same quality regardless of the server's
    co-residence mode.

    Use this for "summarize this doc"; use `ask` for a specific question.
    """
    log = logger.bind(tool="summarize", doc_id=doc_id)
    log.info("mcp.tool.start", detail=detail)
    level: SummaryDetail = (
        detail if detail in ("brief", "standard", "detailed", "report") else "standard"
    )
    response = await summarize_document(doc_id, instruction=instruction, detail=level)
    log.info("mcp.tool.done", answered=response.answered, correlation_id=response.correlation_id)
    return response


async def list_scope_sets_tool() -> list[ScopeSet]:
    """List every saved scope set in the vault — name, document ids, and
    timestamps. A scope set is a named document collection; pass its `name` to
    `ask(scope_set=...)` to scope an answer to those documents.

    Exposed as `list_scope_sets` over MCP; the suffix on the Python name avoids
    shadowing the import from `memex.core.scope_sets`.
    """
    settings = get_settings()
    log = logger.bind(tool="list_scope_sets")
    log.info("mcp.tool.start")
    sets = await list_scope_sets(settings.vault_path)
    log.info("mcp.tool.done", count=len(sets))
    return sets


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


async def get_graph_neighbors(doc_id: str, limit: int = 50) -> list[GraphNeighbor]:
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


async def related_documents(doc_id: str, limit: int = 10) -> list[RelatedDocument]:
    """Explore connections: documents related to `doc_id` via SHARED ENTITIES,
    ranked by entity SPECIFICITY.

    Unlike `get_graph_neighbors` (one-hop neighbours, unranked — so generic shared
    entities like 'IP' or an author's name dominate), this scores each related document
    by the inverse-document-frequency of the entities it shares with `doc_id`: a rare,
    specific shared concept is a strong link; a near-universal one is filtered out as
    noise. Returns up to `limit` documents, each with a `score` and the connecting
    `shared_entities` (most-specific first) — the meaningful "what else in my corpus
    relates to this" surface.

    Returns an empty list if RyuGraph isn't installed (the graph is optional).
    """
    from memex.index.graph_store import GraphStore

    settings = get_settings()
    log = logger.bind(tool="related_documents", doc_id=doc_id)
    log.info("mcp.tool.start")
    try:
        store = await GraphStore.open(settings.vault_path)
    except ImportError as e:
        log.warning("mcp.graph_unavailable", reason=str(e))
        return []
    try:
        related = await store.related_documents(doc_id, limit=limit)
    finally:
        await store.close()
    log.info("mcp.tool.done", count=len(related))
    return related


# Register with FastMCP. We do this after defining the functions so the
# above docstrings show up verbatim in MCP tool-introspection responses.
server.tool()(search)
server.tool()(ask)
server.tool()(summarize)
server.tool()(get_document)
server.tool(name="list_documents")(list_documents_tool)
server.tool(name="list_scope_sets")(list_scope_sets_tool)
server.tool()(get_graph_neighbors)
server.tool()(related_documents)


# ----- Transports -----


async def serve_stdio() -> None:
    """Run the MCP server over stdio. Blocks until the client disconnects."""
    logger.info("mcp.serve.stdio.start")
    await server.run_stdio_async()


async def serve_http(host: str = "127.0.0.1", port: int = 7424) -> None:
    """Run the MCP server over streamable HTTP.

    When `McpSettings.auth_token` is set (env: MEMEX_MCP__AUTH_TOKEN),
    every request must carry `Authorization: Bearer <token>` and
    `BearerAuthMiddleware` rejects everything else with 401. When the
    token is unset, the bind is restricted: loopback hosts get a
    startup WARN and run unauthenticated; non-loopback hosts are
    refused at startup via `validate_bind`. See docs/deploy/mcp-http.md.
    """
    settings = get_settings()
    has_token = settings.mcp.auth_token is not None
    validate_bind(host, has_token=has_token)

    server.settings.host = host
    server.settings.port = port

    if not has_token:
        logger.warning(
            "mcp.serve.http.unauthenticated",
            host=host,
            port=port,
            hint=(
                "set MEMEX_MCP__AUTH_TOKEN to require Bearer auth; "
                "this loopback-only mode is a developer affordance"
            ),
        )
        logger.info("mcp.serve.http.start", host=host, port=port, auth=False)
        await server.run_streamable_http_async()
        return

    # Authenticated path: wrap the FastMCP ASGI app with bearer-auth
    # middleware and serve via uvicorn directly. FastMCP's
    # `run_streamable_http_async` builds the app + runs uvicorn in one
    # call; we split that here so the middleware can layer on between.
    import uvicorn

    auth_token = settings.mcp.auth_token
    if auth_token is None:
        # Unreachable: the `not has_token` branch above returns first. This
        # guard narrows `auth_token` for the type checker without a bare
        # `assert` (which `-O` strips) and stays within the typed-error rule.
        raise ConfigurationError(
            "authenticated serve_http reached with no auth token",
            context={"host": host, "port": port},
        )

    app = server.streamable_http_app()
    app.add_middleware(
        BearerAuthMiddleware,
        expected_token=auth_token.get_secret_value(),
    )
    logger.info("mcp.serve.http.start", host=host, port=port, auth=True)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    await uvicorn.Server(config).serve()
