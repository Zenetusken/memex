# MCP HTTP transport — auth + deployment

`memex serve mcp --transport http` exposes the five MCP tools (`search`, `ask`, `get_document`, `list_documents`, `get_graph_neighbors`) over a streamable HTTP endpoint. For everything beyond `localhost:7424`, you need to set an auth token.

## Why bearer tokens

Memex is local-first / single-user. The minimum-viable auth that unblocks non-loopback use is a single static bearer token loaded from env. It composes cleanly with a reverse proxy if you need anything more elaborate later — per-client tokens, scopes, OAuth, mTLS. Putting Caddy / Traefik / nginx in front of Memex with proper auth, and turning Memex's own token off, is a fully supported deployment.

## Quickstart

1. **Generate a token**

   ```sh
   memex mcp generate-token
   # → ze1Q9k…ZW (43 urlsafe chars)
   ```

   The token goes to stdout with no trailing newline-noise, so you can pipe it into a clipboard helper or secrets manager:

   ```sh
   memex mcp generate-token | pbcopy          # macOS
   memex mcp generate-token | wl-copy         # Wayland
   memex mcp generate-token > ~/.memex.token  # write to a 0600 file
   ```

2. **Configure**

   Either env (preferred for one-off / containerised deployments):

   ```sh
   export MEMEX_MCP__AUTH_TOKEN="ze1Q9k…ZW"
   ```

   …or persist in `~/.config/memex/config.toml`:

   ```toml
   [mcp]
   auth_token = "ze1Q9k…ZW"
   ```

3. **Start the server**

   ```sh
   memex serve mcp --transport http --host 0.0.0.0 --port 7424
   ```

   Memex refuses to start with a non-loopback bind unless `MEMEX_MCP__AUTH_TOKEN` is set, so this command fails-fast if you forget step 2.

4. **Call from a client**

   ```sh
   curl -H "Authorization: Bearer $MEMEX_MCP__AUTH_TOKEN" \
        https://my-memex.example.com/mcp/
   ```

   For Claude Desktop / Code, configure the MCP endpoint with custom headers:

   ```json
   {
     "mcpServers": {
       "memex": {
         "url": "https://my-memex.example.com/mcp/",
         "headers": {
           "Authorization": "Bearer ze1Q9k…ZW"
         }
       }
     }
   }
   ```

## What the server does

- **Token set** — every request must carry `Authorization: Bearer <token>`. Verification is constant-time (`hmac.compare_digest`). Missing/malformed/wrong gets `401 Unauthorized` with `WWW-Authenticate: Bearer realm="memex-mcp"` and a structured WARNING log (remote IP + reason, never the token itself).
- **Token unset on loopback** — server starts with a clear startup WARNING and runs unauthenticated. Use this only for local development.
- **Token unset on non-loopback** — server refuses to bind, prints a `ConfigurationError` with a fix hint, and exits non-zero. This is intentional: a misconfigured production deployment should be obvious, not silent.

## Token rotation

There's no in-process rotation. To rotate:

1. Generate a new token (`memex mcp generate-token`).
2. Update `MEMEX_MCP__AUTH_TOKEN` in your env / config.
3. Restart `memex serve mcp`.
4. Update clients with the new token.

Old tokens stop working immediately on restart.

## What's not in scope here

- **Token expiry** — tokens are static until rotated.
- **Multi-token / per-client scopes** — one token, full access. Put a proxy in front if you need granularity.
- **OAuth flow** — overkill for single-user. FastMCP supports it (`AuthSettings` + `OAuthAuthorizationServerProvider`); the integration point is `mcp/server.py:serve_http` if you want to wire it.
- **mTLS** — terminate at the proxy layer.

## What runs without auth regardless

- **stdio transport** (`memex serve mcp --transport stdio`) — runs over a parent-child pipe; the parent already has full process access, so an auth check would be theatre. This is the transport used by desktop MCP clients (Claude Code, Cursor).
- **The web UI** (`memex serve web`) — has its own auth story (currently none; same loopback-only assumption). See `webui/CLAUDE.md` for the plan.
