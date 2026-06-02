"""SQLite-backed persistence for grounded multi-turn chat conversations.

Surface A of `docs/specs/grounded-agentic-chat.md`. A conversation is a thread of
turns; each turn records the user message, the rewritten standalone query, the
answer summary (or refusal reason), the cited chunk IDs (ids ONLY — never chunk
text), and an opaque `FinalResponse.model_dump_json()` for re-render.

**User data, not regenerable derived state** — this store is deliberately NOT in
the `reindex_vault(force=True)` teardown allow-list (the `scope_sets.json`
precedent): a full rebuild must preserve a user's conversations. Lives in `core/`
(no Memex deps): `core/errors` + `core/sqlite_tuning` + `core/types` + stdlib only — so the surfaces
(`webui/`, `cli/`) import it without inverting an edge, and `response_json` stays
opaque (core/ never imports `agents/FinalResponse`).

Mirrors `index/fts_store.py`: sync SQLite (`isolation_level=None`, WAL via
`apply_sqlite_pragmas`), async wrapper through `asyncio.to_thread`, an `asyncio.Lock`
gating the multi-statement appends so concurrent turns don't interleave.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import structlog
import ulid

from memex.core.errors import VaultIntegrityError
from memex.core.sqlite_tuning import apply_sqlite_pragmas
from memex.core.types import Conversation, ConversationTurn

logger = structlog.get_logger(__name__)

_TITLE_MAX = 60  # a conversation's auto-derived title is the first message, truncated

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id   TEXT PRIMARY KEY,
    title             TEXT NOT NULL DEFAULT '',
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    turn_count        INTEGER NOT NULL DEFAULT 0,
    running_summary   TEXT NOT NULL DEFAULT '',
    scope_doc_ids     TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id           TEXT PRIMARY KEY,
    conversation_id   TEXT NOT NULL,
    turn_index        INTEGER NOT NULL,
    created_at        TEXT NOT NULL,
    user_text         TEXT NOT NULL,
    standalone_query  TEXT NOT NULL,
    is_followup       INTEGER NOT NULL DEFAULT 0,
    answered          INTEGER NOT NULL DEFAULT 0,
    answer_summary    TEXT NOT NULL DEFAULT '',
    cited_chunk_ids   TEXT NOT NULL DEFAULT '[]',
    response_json     TEXT,
    correlation_id    TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);

CREATE INDEX IF NOT EXISTS turns_convo ON turns(conversation_id, turn_index);
"""


def _now() -> str:
    """ISO-8601 UTC timestamp (sortable, timezone-explicit)."""
    return datetime.now(UTC).isoformat()


def _row_to_conversation(row: sqlite3.Row) -> Conversation:
    return Conversation(
        conversation_id=row["conversation_id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        turn_count=row["turn_count"],
        running_summary=row["running_summary"],
        scope_doc_ids=json.loads(row["scope_doc_ids"]),
    )


def _row_to_turn(row: sqlite3.Row) -> ConversationTurn:
    return ConversationTurn(
        turn_id=row["turn_id"],
        conversation_id=row["conversation_id"],
        turn_index=row["turn_index"],
        created_at=row["created_at"],
        user_text=row["user_text"],
        standalone_query=row["standalone_query"],
        is_followup=bool(row["is_followup"]),
        answered=bool(row["answered"]),
        answer_summary=row["answer_summary"],
        cited_chunk_ids=json.loads(row["cited_chunk_ids"]),
        response_json=row["response_json"],
        correlation_id=row["correlation_id"],
    )


class ConversationStore:
    """Sync SQLite under the hood; async wrapper via `asyncio.to_thread`."""

    def __init__(self, db: sqlite3.Connection):
        self._db = db
        # Gate the multi-statement appends (insert turn + bump conversation counters)
        # so two concurrent turns on the same conversation can't interleave. Reads stay
        # unlocked (SQLite serializes individual statements in autocommit mode).
        self._lock = asyncio.Lock()

    @classmethod
    async def open(cls, vault_path: Path) -> ConversationStore:
        """Open (or create) the conversation database under
        `{vault_path}/.memex/conversations.sqlite`."""
        path = vault_path / ".memex" / "conversations.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        def _connect() -> sqlite3.Connection:
            db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
            db.row_factory = sqlite3.Row
            apply_sqlite_pragmas(db)
            db.executescript(_SCHEMA)
            return db

        db = await asyncio.to_thread(_connect)
        return cls(db)

    async def close(self) -> None:
        await asyncio.to_thread(self._db.close)

    async def create_conversation(
        self, *, title: str = "", scope_doc_ids: list[str] | None = None
    ) -> Conversation:
        """Create a fresh, empty conversation and return it."""
        cid = str(ulid.ULID())
        now = _now()
        scope = list(dict.fromkeys(d.strip() for d in (scope_doc_ids or []) if d.strip()))

        def _write() -> None:
            self._db.execute(
                "INSERT INTO conversations "
                "(conversation_id, title, created_at, updated_at, turn_count, "
                " running_summary, scope_doc_ids) VALUES (?, ?, ?, ?, 0, '', ?)",
                (cid, title, now, now, json.dumps(scope)),
            )

        async with self._lock:
            await asyncio.to_thread(_write)
        return Conversation(
            conversation_id=cid,
            title=title,
            created_at=now,
            updated_at=now,
            scope_doc_ids=scope,
        )

    async def load(self, conversation_id: str) -> Conversation | None:
        """Load a conversation + its turns (ordered by `turn_index`), or `None` if absent."""

        def _read() -> Conversation | None:
            row = self._db.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()
            if row is None:
                return None
            convo = _row_to_conversation(row)
            turn_rows = self._db.execute(
                "SELECT * FROM turns WHERE conversation_id = ? ORDER BY turn_index",
                (conversation_id,),
            ).fetchall()
            convo.turns = [_row_to_turn(r) for r in turn_rows]
            return convo

        return await asyncio.to_thread(_read)

    async def append_turn(
        self,
        conversation_id: str,
        *,
        user_text: str,
        standalone_query: str,
        is_followup: bool,
        answered: bool,
        answer_summary: str,
        cited_chunk_ids: list[str],
        response_json: str | None,
        correlation_id: str | None,
    ) -> ConversationTurn:
        """Append one turn and bump the conversation's counters — one locked transaction.

        `turn_index` is the conversation's current `turn_count` (read inside the lock so
        concurrent appends can't collide). The first turn (index 0) also seeds an empty
        title from the user message.
        """
        turn_id = str(ulid.ULID())
        now = _now()

        def _write() -> int:
            row = self._db.execute(
                "SELECT turn_count, title FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                # TOCTOU: the caller loaded this conversation, then a concurrent
                # `delete_conversation` removed it before this locked append. Typed +
                # context-carrying per the Errors rule (a bare KeyError can't be matched).
                raise VaultIntegrityError(
                    "conversation not found (deleted concurrently?)",
                    context={"conversation_id": conversation_id},
                )
            turn_index = int(row["turn_count"])
            self._db.execute(
                "INSERT INTO turns "
                "(turn_id, conversation_id, turn_index, created_at, user_text, "
                " standalone_query, is_followup, answered, answer_summary, "
                " cited_chunk_ids, response_json, correlation_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    turn_id,
                    conversation_id,
                    turn_index,
                    now,
                    user_text,
                    standalone_query,
                    1 if is_followup else 0,
                    1 if answered else 0,
                    answer_summary,
                    json.dumps(cited_chunk_ids),
                    response_json,
                    correlation_id,
                ),
            )
            # Seed the title from the first message when none was set.
            if turn_index == 0 and not str(row["title"]).strip():
                self._db.execute(
                    "UPDATE conversations SET title = ? WHERE conversation_id = ?",
                    (user_text.strip()[:_TITLE_MAX], conversation_id),
                )
            self._db.execute(
                "UPDATE conversations SET turn_count = turn_count + 1, updated_at = ? "
                "WHERE conversation_id = ?",
                (now, conversation_id),
            )
            return turn_index

        async with self._lock:
            turn_index = await asyncio.to_thread(_write)
        return ConversationTurn(
            turn_id=turn_id,
            conversation_id=conversation_id,
            turn_index=turn_index,
            created_at=now,
            user_text=user_text,
            standalone_query=standalone_query,
            is_followup=is_followup,
            answered=answered,
            answer_summary=answer_summary,
            cited_chunk_ids=cited_chunk_ids,
            response_json=response_json,
            correlation_id=correlation_id,
        )

    async def set_scope(self, conversation_id: str, scope_doc_ids: list[str]) -> None:
        """Pin the per-conversation document scope (set on the first turn; later turns
        inherit it). Deduped + blank-stripped, like the doc-picker path."""
        scope = list(dict.fromkeys(d.strip() for d in scope_doc_ids if d.strip()))

        def _write() -> None:
            self._db.execute(
                "UPDATE conversations SET scope_doc_ids = ?, updated_at = ? "
                "WHERE conversation_id = ?",
                (json.dumps(scope), _now(), conversation_id),
            )

        async with self._lock:
            await asyncio.to_thread(_write)

    async def update_running_summary(self, conversation_id: str, running_summary: str) -> None:
        """Replace the conversation's compacted running summary (the digest output)."""

        def _write() -> None:
            self._db.execute(
                "UPDATE conversations SET running_summary = ?, updated_at = ? "
                "WHERE conversation_id = ?",
                (running_summary, _now(), conversation_id),
            )

        async with self._lock:
            await asyncio.to_thread(_write)

    async def list_conversations(self, *, limit: int = 50) -> list[Conversation]:
        """Recent conversations (most-recently-updated first), WITHOUT their turns."""

        def _read() -> list[Conversation]:
            rows = self._db.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [_row_to_conversation(r) for r in rows]

        return await asyncio.to_thread(_read)

    async def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and all its turns (one locked transaction)."""

        def _write() -> None:
            self._db.execute("DELETE FROM turns WHERE conversation_id = ?", (conversation_id,))
            self._db.execute(
                "DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,)
            )

        async with self._lock:
            await asyncio.to_thread(_write)
