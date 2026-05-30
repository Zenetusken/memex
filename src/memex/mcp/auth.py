"""Bearer-token auth for the MCP HTTP transport.

A single static token, loaded from `McpSettings.auth_token`, gates
every request via `Authorization: Bearer <token>` with constant-time
comparison. When the token is unset the bind is restricted: loopback
addresses are allowed (with a startup warning); non-loopback binds
are refused at startup.

Logging is mandatory and the token is never written to a log line.
"""

from __future__ import annotations

import hmac
import ipaddress
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from memex.core.errors import ConfigurationError

logger = structlog.get_logger(__name__)

_LOOPBACK_HOST_LITERALS = frozenset({"localhost"})


def is_loopback(host: str) -> bool:
    """True iff `host` resolves only to loopback addresses.

    Accepts the string `localhost`, IPv4 / IPv6 literals (`127.0.0.1`,
    `::1`, `127.0.0.7`), and falls through to False on hostnames or
    unparseable input. `0.0.0.0` is *not* loopback — it's a wildcard
    that listens on every interface and must be refused without auth.
    """
    if not host:
        return False
    if host.lower() in _LOOPBACK_HOST_LITERALS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_bind(host: str, *, has_token: bool) -> None:
    """Refuse to bind a non-loopback host without an auth token.

    Called from `serve_http` before the server starts. The hint in the
    `ConfigurationError.context` tells the user exactly which env var
    to set and references the token-generation CLI.
    """
    if has_token or is_loopback(host):
        return
    raise ConfigurationError(
        "Refusing to start the MCP HTTP transport on a non-loopback "
        "address without an auth token. Set MEMEX_MCP__AUTH_TOKEN "
        "(generate one with `memex mcp generate-token`) or bind to "
        "127.0.0.1.",
        context={"host": host, "fix": "MEMEX_MCP__AUTH_TOKEN=<secret>"},
    )


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Require `Authorization: Bearer <expected_token>` on every request.

    - Constant-time string comparison (`hmac.compare_digest`).
    - 401 with `WWW-Authenticate: Bearer realm="memex-mcp"` on
      missing/invalid token.
    - Logs failures at WARNING with remote IP + reason; successes at
      DEBUG. Never logs the token.
    """

    def __init__(self, app: Any, *, expected_token: str) -> None:
        super().__init__(app)
        if not expected_token:
            raise ValueError("expected_token must be non-empty")
        self._expected = expected_token

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Starlette middleware entry point — extracts the
        `Authorization: Bearer <token>` header and compares
        constant-time against the configured token. Rejects with 401
        on missing / malformed / invalid; otherwise forwards to the
        next handler."""
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return self._unauthorized(request, reason="missing_or_malformed")
        # Compare on the UTF-8 bytes: str compare_digest raises TypeError on a
        # non-ASCII token (attacker-controlled) → a 500 instead of a clean 401.
        # Bytes compare stays constant-time and is byte-identical for ASCII tokens.
        if not hmac.compare_digest(token.encode("utf-8"), self._expected.encode("utf-8")):
            return self._unauthorized(request, reason="invalid_token")
        logger.debug(
            "mcp.auth.ok",
            remote=request.client.host if request.client else None,
            path=request.url.path,
        )
        return await call_next(request)

    def _unauthorized(self, request: Request, *, reason: str) -> Response:
        logger.warning(
            "mcp.auth.failed",
            remote=request.client.host if request.client else None,
            path=request.url.path,
            reason=reason,
        )
        return JSONResponse(
            {"error": "unauthorized", "reason": reason},
            status_code=401,
            headers={"WWW-Authenticate": 'Bearer realm="memex-mcp"'},
        )
