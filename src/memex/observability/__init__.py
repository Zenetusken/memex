"""Observability — structlog logs + Langfuse traces, joined by correlation_id.

See ADR-0004 for the full architecture.
"""

from memex.observability.logging import configure_logging
from memex.observability.tracing import (
    bind_run_context,
    callback_handler,
    clear_run_context,
    configure_tracing,
    get_langfuse,
    run_attributes,
)

__all__ = [
    "bind_run_context",
    "callback_handler",
    "clear_run_context",
    "configure_logging",
    "configure_tracing",
    "get_langfuse",
    "run_attributes",
]
