"""structlog configuration — see ADR-0004."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(
    *,
    json_output: bool = True,
    level: int = logging.INFO,
) -> None:
    """Configure structlog process-wide. Call once at startup.

    `json_output=True` (default) emits one JSON object per log line for
    the disk log stream and downstream tooling (jq, log aggregators,
    Langfuse import). `json_output=False` uses ConsoleRenderer for an
    interactive dev experience.

    Run-scope context (correlation_id, query preview, document id) is
    bound via `memex.observability.bind_run_context` at the entry point
    of each run; it flows through `contextvars` to every nested log call.

    Logs go to stderr, not stdout. The CLI promises that piped stdout is
    pure JSON data (GUIDELINES.md Part V); a structlog line on stdout
    would corrupt `memex ask … | jq` and similar workflows.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            (
                structlog.processors.JSONRenderer()
                if json_output
                else structlog.dev.ConsoleRenderer()
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
