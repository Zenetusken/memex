"""Langfuse tracing — see ADR-0004.

Targets the Langfuse v4 Python SDK. The v2 `langfuse.callback` import
path is gone; the v3+ path is `langfuse.langchain`. Constructor kwargs
for `CallbackHandler` (trace_name, metadata, tags) were also removed in
v3 — per-run attributes flow via the LangGraph `config.metadata` dict
using `langfuse_*` prefixed keys.
"""

from __future__ import annotations

from typing import Any

import structlog
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from memex.core.config import ObservabilitySettings
from memex.core.errors import ConfigurationError

_LANGFUSE: Langfuse | None = None


def configure_tracing(settings: ObservabilitySettings) -> None:
    """Initialise the Langfuse SDK singleton. Call once at startup.

    Misconfiguration fails here, not silently at the first model call.
    If `langfuse_enabled` is False, no SDK is created and downstream
    callback handlers degrade to no-op behaviour.
    """
    global _LANGFUSE
    if not settings.langfuse_enabled:
        _LANGFUSE = None
        return

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        raise ConfigurationError(
            "Langfuse is enabled but public/secret keys are not configured.",
            context={
                "fix": "Set MEMEX_OBSERVABILITY__LANGFUSE_PUBLIC_KEY and "
                "MEMEX_OBSERVABILITY__LANGFUSE_SECRET_KEY, or set "
                "MEMEX_OBSERVABILITY__LANGFUSE_ENABLED=false.",
            },
        )

    _LANGFUSE = Langfuse(
        host=settings.langfuse_host,
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
    )


def get_langfuse() -> Langfuse | None:
    """Return the configured Langfuse client, or None if tracing is disabled.

    Callers that need flush/shutdown semantics use this; per-run code
    should not — it goes through `callback_handler` instead.
    """
    return _LANGFUSE


def callback_handler() -> CallbackHandler:
    """Build a bare LangChain CallbackHandler for the current run.

    In Langfuse v4 the handler takes no constructor kwargs; per-run
    attributes (session id, tags, trace name) flow through the
    LangGraph `config.metadata` dict using `langfuse_*` prefixed
    keys. Build the metadata dict via `run_attributes()` below.
    """
    return CallbackHandler()


def run_attributes(
    correlation_id: str,
    name: str,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the `config.metadata` payload for a LangGraph `ainvoke`.

    The structlog side reads `correlation_id`; the Langfuse v4 side
    reads the `langfuse_*` prefixed keys.
    """
    payload: dict[str, Any] = {
        "correlation_id": correlation_id,
        "langfuse_session_id": correlation_id,
        "langfuse_trace_name": name,
        "langfuse_tags": [name],
    }
    if extra:
        payload.update(extra)
    return payload


def bind_run_context(correlation_id: str, **fields: Any) -> None:
    """Bind ContextVars so every log line in the current run carries
    `correlation_id` and any additional fields.

    Pair with `clear_run_context()` in a `finally` at the run's exit.
    """
    structlog.contextvars.bind_contextvars(
        correlation_id=correlation_id, **fields
    )


def clear_run_context() -> None:
    """Clear all ContextVars bound by `bind_run_context`."""
    structlog.contextvars.clear_contextvars()
