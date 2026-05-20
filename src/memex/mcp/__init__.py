"""MCP server — the public API of Memex.

Anything an external agent (Claude Code, Cursor, custom MCP clients)
should be able to do is exposed here as an MCP tool. Anything not
exposed is internal. Runs over stdio (desktop agent integration) or
streamable HTTP (network-local agents). See GUIDELINES.md Part V "MCP
server" and IMPLEMENTATION-PLAN.md §1.9.
"""

# Note: the FastMCP instance is intentionally NOT re-exported from this
# package under the name `server` — that would shadow the
# `memex.mcp.server` module path used by `monkeypatch.setattr` in the
# integration tests. Callers that want the instance import it
# explicitly: `from memex.mcp.server import server`.

from memex.mcp.server import (
    ask,
    get_document,
    list_documents_tool,
    search,
    serve_http,
    serve_stdio,
)

__all__ = [
    "ask",
    "get_document",
    "list_documents_tool",
    "search",
    "serve_http",
    "serve_stdio",
]
