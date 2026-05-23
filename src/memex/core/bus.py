# pyright: reportConstantRedefinition=false
# `_BUS` is an uppercase module-level singleton intentionally rebound
# by `set_bus()` for test isolation and `reset_bus()` for teardown.

"""In-process event bus — see GUIDELINES.md Part II and IMPLEMENTATION-PLAN.md §2.1.

Asyncio pubsub with synchronous append-to-SQLite as the persistence
sidecar. Single-machine application: no Redis, no Kafka, no broker.

The bus is the only allowed inter-stage communication channel: parse
emits `document.parsed`, enrich subscribes; enrich emits
`document.enriched`, index subscribes; etc. Direct cross-module
function calls between stages are a layering violation.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Self

import structlog

from memex.core.errors import ConfigurationError
from memex.core.events import EventStage, MemexEvent

logger = structlog.get_logger(__name__)

_RETENTION_DAYS = 30


class EventBus:
    """Process-singleton asyncio pubsub with optional SQLite persistence.

    Subscribers receive events via per-subscription `asyncio.Queue`s.
    A slow subscriber back-pressures itself (the bus uses
    `put_nowait`); if the queue is full, the event is logged as
    `bus.dropped` and the publisher continues. Pipelines must not
    wedge on observability.
    """

    def __init__(self, *, db_path: Path | None = None, queue_size: int = 1024):
        self._db_path = db_path
        self._db: sqlite3.Connection | None = None
        self._queue_size = queue_size
        self._subscribers: list[
            tuple[
                EventStage | None,
                str | None,
                asyncio.Queue[MemexEvent],
            ]
        ] = []
        self._lock = asyncio.Lock()

    def start(self) -> Self:
        """Open the SQLite connection. Synchronous so bootstrap can call
        it from a non-async entry point. Idempotent.

        Idempotent: calling twice on the same instance is a no-op
        on the second call.
        """
        if self._db is not None:
            return self
        if self._db_path is not None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(
                self._db_path,
                isolation_level=None,
                check_same_thread=False,
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                "event_id TEXT PRIMARY KEY, "
                "correlation_id TEXT NOT NULL, "
                "stage TEXT NOT NULL, "
                "event_type TEXT NOT NULL, "
                "timestamp TEXT NOT NULL, "
                "payload TEXT NOT NULL, "
                "error TEXT"
                ")"
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS events_correlation "
                "ON events(correlation_id)"
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS events_timestamp "
                "ON events(timestamp)"
            )
        return self

    def stop(self) -> None:
        """Close the SQLite connection. Synchronous; idempotent."""
        if self._db is not None:
            self._db.close()
            self._db = None

    async def __aenter__(self) -> Self:
        return self.start()

    async def __aexit__(self, *_exc: object) -> None:
        self.stop()

    async def publish(self, event: MemexEvent) -> None:
        """Persist (if configured) then fan out to subscribers."""
        if self._db is not None:
            # SQLite write is sync; dispatch to a worker thread so the
            # event loop isn't blocked by fsync. Keep this small — every
            # pipeline stage publishes, so this is hot.
            db = self._db
            row = (
                event.event_id,
                event.correlation_id,
                event.stage,
                event.event_type,
                event.timestamp.isoformat(),
                event.payload_json(),
                event.error_json(),
            )

            def _insert() -> None:
                db.execute(
                    "INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?, ?, ?, ?)",
                    row,
                )

            await asyncio.to_thread(_insert)

        async with self._lock:
            subscribers = list(self._subscribers)

        for stage, event_type, q in subscribers:
            if stage is not None and stage != event.stage:
                continue
            if event_type is not None and event_type != event.event_type:
                continue
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "bus.dropped",
                    event_type=event.event_type,
                    correlation_id=event.correlation_id,
                )

    @asynccontextmanager
    async def subscribe(
        self,
        *,
        stage: EventStage | None = None,
        event_type: str | None = None,
    ) -> AsyncIterator[asyncio.Queue[MemexEvent]]:
        """Subscribe to events matching the (optional) filters.

        Yields the queue; iterate with `await q.get()` in the caller.
        Unsubscribes automatically on exit.
        """
        q: asyncio.Queue[MemexEvent] = asyncio.Queue(maxsize=self._queue_size)
        entry = (stage, event_type, q)
        async with self._lock:
            self._subscribers.append(entry)
        try:
            yield q
        finally:
            async with self._lock:
                self._subscribers.remove(entry)

    async def replay(
        self,
        *,
        since: datetime,
        until: datetime | None = None,
    ) -> AsyncIterator[MemexEvent]:
        """Yield persisted events in the time window. Requires persistence."""
        if self._db is None:
            return
        until = until or datetime.now(UTC)
        cursor = self._db.execute(
            "SELECT correlation_id, stage, event_type, timestamp, payload, error, event_id "
            "FROM events WHERE timestamp >= ? AND timestamp <= ? "
            "ORDER BY timestamp",
            (since.isoformat(), until.isoformat()),
        )
        for row in cursor:
            yield MemexEvent(
                event_id=row[6],
                correlation_id=row[0],
                stage=row[1],
                event_type=row[2],
                timestamp=datetime.fromisoformat(row[3]),
                payload=json.loads(row[4]),
                error=json.loads(row[5]) if row[5] else None,
            )

    async def prune(self, *, retention_days: int = _RETENTION_DAYS) -> int:
        """Delete events older than `retention_days`. Returns rows deleted."""
        if self._db is None:
            return 0
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        cursor = self._db.execute(
            "DELETE FROM events WHERE timestamp < ?",
            (cutoff.isoformat(),),
        )
        return cursor.rowcount


# Lazily constructed module-level singleton. The CLI / daemon entrypoint
# is responsible for opening it inside `async with`; tests construct their
# own instances and never touch the singleton.

_BUS: EventBus | None = None


def get_bus() -> EventBus:
    """Return the process bus. Configure via `set_bus()` at startup."""
    if _BUS is None:
        raise ConfigurationError(
            "EventBus is not initialised",
            context={
                "fix": "call set_bus(EventBus(...).start()) from the entry point",
            },
        )
    return _BUS


def set_bus(bus: EventBus | None) -> None:
    """Install the process bus. Pass None to detach (tests)."""
    global _BUS
    _BUS = bus


