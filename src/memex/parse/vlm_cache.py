"""Content-addressed cache for VLM page transcriptions.

The VLM (Qwen2.5-VL-AWQ) transcribes a rendered PDF page, but its greedy
output is NON-deterministic — BF16's 7-bit mantissa lets near-tied
top-token logits flip under FP accumulation-order variance (AWQ split-K
atomics, the SDPA kernel), and an early flip cascades through the rest of
the generation. So re-parsing the same page yields a different
transcription, which churns the content-addressed
`chunk_id = sha1(doc_id + chunk.text)` and drifts eval anchors.

This cache makes transcription reproducible BY CONSTRUCTION: the VLM is
invoked once per unique `(source-pdf bytes, page, model, prompt)` key;
every subsequent parse reuses the stored markdown. Regenerable derived
state under `vault/.memex/` (ADR-0003), dropped by `reindex --force`.

Mirrors `index/fts_store.py`: sync `sqlite3` under `asyncio.to_thread`,
multi-statement writes gated by an `asyncio.Lock`. See
`docs/specs/vlm-transcription-cache.md`.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vlm_page_cache (
    cache_key   TEXT PRIMARY KEY,
    pdf_sha256  TEXT NOT NULL,
    page_no     INTEGER NOT NULL,
    vlm_model   TEXT NOT NULL,
    prompt_sha8 TEXT NOT NULL,
    markdown    TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS vlm_cache_pdf ON vlm_page_cache(pdf_sha256);
"""


class VLMTranscriptionCache:
    """Sync SQLite under the hood; async wrapper via `asyncio.to_thread`."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db
        # Gate multi-statement writes (same rationale as FTSStore._lock):
        # the connection is autocommit (`isolation_level=None`), so SQLite's
        # kernel mutex serializes individual statements, but the lock keeps
        # a future multi-statement write atomic. Reads stay unlocked.
        self._lock = asyncio.Lock()

    @classmethod
    async def open(cls, vault_path: Path) -> VLMTranscriptionCache:
        """Open (or create) the cache db under
        `{vault_path}/.memex/vlm_cache.sqlite` and return a ready store.
        """
        path = vault_path / ".memex" / "vlm_cache.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        def _connect() -> sqlite3.Connection:
            db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
            db.executescript(_SCHEMA)
            return db

        db = await asyncio.to_thread(_connect)
        return cls(db)

    async def get(self, cache_key: str) -> str | None:
        """Return the cached transcription for `cache_key`, or None on miss."""

        def _read() -> str | None:
            row = self._db.execute(
                "SELECT markdown FROM vlm_page_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            return None if row is None else str(row[0])

        return await asyncio.to_thread(_read)

    async def put(
        self,
        cache_key: str,
        *,
        pdf_sha256: str,
        page_no: int,
        vlm_model: str,
        prompt_sha8: str,
        markdown: str,
    ) -> None:
        """Store a transcription. `INSERT OR IGNORE` — first writer wins
        (concurrent draws of the same key came from the same page/model/
        prompt, so either is equally valid)."""
        created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        async with self._lock:

            def _write() -> None:
                self._db.execute(
                    "INSERT OR IGNORE INTO vlm_page_cache "
                    "(cache_key, pdf_sha256, page_no, vlm_model, prompt_sha8, "
                    "markdown, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (cache_key, pdf_sha256, page_no, vlm_model, prompt_sha8, markdown, created_at),
                )

            await asyncio.to_thread(_write)

    async def delete_by_pdf(self, pdf_sha256: str) -> int:
        """Drop every cached page for a source PDF (the `--refresh-vlm`
        bust). Returns the number of rows deleted."""
        async with self._lock:

            def _delete() -> int:
                cur = self._db.execute(
                    "DELETE FROM vlm_page_cache WHERE pdf_sha256 = ?",
                    (pdf_sha256,),
                )
                return cur.rowcount

            return await asyncio.to_thread(_delete)

    async def close(self) -> None:
        await asyncio.to_thread(self._db.close)
