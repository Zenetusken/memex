"""Event bus types — see GUIDELINES.md Part II "Event-driven communication".

Stages communicate via typed events, not direct calls. The bus is
in-process (single-machine application; no Redis, no Kafka), but the
discipline of events keeps modules decoupled and the system observable.

Events are persisted to SQLite in a rolling 30-day window for debugging,
recovery, and evaluation replay.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Literal

import ulid
from pydantic import BaseModel, Field

EventStage = Literal["ingest", "parse", "enrich", "index", "retrieve"]


class ErrorInfo(BaseModel):
    """Error context attached to an event when a stage fails."""

    error_type: str
    message: str
    context: dict[str, object] = Field(default_factory=dict)
    recoverable: bool = False
    stack: str | None = None


class MemexEvent(BaseModel):
    """One event in the pipeline.

    `correlation_id` is the most important field — a ULID generated at
    ingest (for documents) or query receipt (for retrievals) that flows
    through every event, log line, Langfuse trace, and model call for
    that unit of work. Given a `correlation_id`, you can reconstruct
    everything that happened.
    """

    event_id: str = Field(default_factory=lambda: str(ulid.ULID()))
    correlation_id: str
    stage: EventStage
    event_type: str  # e.g. "document.parsed", "document.enriched"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, object] = Field(default_factory=dict)
    error: ErrorInfo | None = None

    def payload_json(self) -> str:
        """Serialise payload for the events-table TEXT column."""
        return json.dumps(self.payload, default=str)

    def error_json(self) -> str | None:
        """Serialise error info for the events-table TEXT column, or None."""
        return self.error.model_dump_json() if self.error else None
