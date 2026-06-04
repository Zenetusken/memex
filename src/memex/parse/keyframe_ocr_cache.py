"""Content-addressed cache for lecture-video KEYFRAME OCR (companion-merge, ADR-0018 §13).

The keyframe-OCR accuracy lever (recover MaViLS's strong frame-text signal) OCRs one
video frame per transcript chunk via the VLM, whose greedy output is NON-deterministic
(the same BF16/AWQ FP-accumulation flip as `vlm_cache.py`). Left uncached, every
`link-slides --use-video` re-run would re-OCR the frames and drift the derived alignment
sidecar. This cache makes the frame OCR reproducible BY CONSTRUCTION: each frame is OCR'd
once per `(source-video bytes, timestamp, model, prompt)` key; every subsequent run reuses
the stored markdown.

Keyed on `(video_sha256, timestamp_ms, model, prompt_sha8)` — the VIDEO + TIMESTAMP, NOT
the decoded frame's PNG bytes: a frame decode can vary at the pixel level (codec/seek
non-determinism) while denoting the same moment, and the video+timestamp is checkable
without re-decoding. Regenerable derived state under `vault/.memex/` (ADR-0003), dropped by
`reindex --force`.

Mirrors `parse/vlm_cache.py`: sync `sqlite3` under `asyncio.to_thread`, multi-statement
writes gated by an `asyncio.Lock`.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import structlog

from memex.core.sqlite_tuning import apply_sqlite_pragmas

logger = structlog.get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS keyframe_ocr_cache (
    cache_key    TEXT PRIMARY KEY,
    video_sha256 TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    vlm_model    TEXT NOT NULL,
    prompt_sha8  TEXT NOT NULL,
    markdown     TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS keyframe_cache_video ON keyframe_ocr_cache(video_sha256);
"""


class KeyframeOCRCache:
    """Sync SQLite under the hood; async wrapper via `asyncio.to_thread`."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db
        # Gate multi-statement writes (same rationale as VLMTranscriptionCache._lock):
        # autocommit connection, so SQLite's kernel mutex serializes individual
        # statements, but the lock keeps a future multi-statement write atomic.
        self._lock = asyncio.Lock()

    @classmethod
    async def open(cls, vault_path: Path) -> KeyframeOCRCache:
        """Open (or create) the cache db under
        `{vault_path}/.memex/keyframe_ocr_cache.sqlite` and return a ready store."""
        path = vault_path / ".memex" / "keyframe_ocr_cache.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        def _connect() -> sqlite3.Connection:
            db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
            apply_sqlite_pragmas(db)  # WAL + cache + mmap (ADR-0003 derived state)
            db.executescript(_SCHEMA)
            return db

        db = await asyncio.to_thread(_connect)
        return cls(db)

    async def get(self, cache_key: str) -> str | None:
        """Return the cached OCR markdown for `cache_key`, or None on miss."""

        def _read() -> str | None:
            row = self._db.execute(
                "SELECT markdown FROM keyframe_ocr_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            return None if row is None else str(row[0])

        return await asyncio.to_thread(_read)

    async def put(
        self,
        cache_key: str,
        *,
        video_sha256: str,
        timestamp_ms: int,
        vlm_model: str,
        prompt_sha8: str,
        markdown: str,
    ) -> None:
        """Store a frame OCR. `INSERT OR IGNORE` — first writer wins (concurrent draws
        of the same key came from the same video/timestamp/model/prompt)."""
        created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        async with self._lock:

            def _write() -> None:
                self._db.execute(
                    "INSERT OR IGNORE INTO keyframe_ocr_cache "
                    "(cache_key, video_sha256, timestamp_ms, vlm_model, prompt_sha8, "
                    "markdown, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        cache_key,
                        video_sha256,
                        timestamp_ms,
                        vlm_model,
                        prompt_sha8,
                        markdown,
                        created_at,
                    ),
                )

            await asyncio.to_thread(_write)

    async def delete_by_video(self, video_sha256: str) -> int:
        """Drop every cached frame for a source video. Returns the number of rows deleted."""
        async with self._lock:

            def _delete() -> int:
                cur = self._db.execute(
                    "DELETE FROM keyframe_ocr_cache WHERE video_sha256 = ?",
                    (video_sha256,),
                )
                return cur.rowcount

            return await asyncio.to_thread(_delete)

    async def close(self) -> None:
        await asyncio.to_thread(self._db.close)
