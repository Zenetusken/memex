"""Content-addressed cache for chart-OCR figure extractions.

The chart-OCR model (Nemotron-Parse, a VLM-style extractor) has the same
greedy non-determinism as the page VLM — BF16 / AWQ accumulation-order
variance flips near-tied logits, and an early flip cascades. So re-parsing a
chart-heavy doc yields drifting chart-content markdown, which churns the
content-addressed `chunk_id` and, worse, silently shifts whether a
chart-content answer is even derivable run-to-run (it plagued the
slide-decks / chart-types / annual-report re-baselines).

This cache makes chart-OCR reproducible BY CONSTRUCTION: each figure is
extracted once per unique `(source-pdf bytes, page, bbox, model, cache
version)` key; every subsequent parse reuses the stored markdown. Regenerable
derived state under `vault/.memex/` (ADR-0003), dropped by `reindex --force`.

Mirrors `parse/vlm_cache.py` exactly (sync `sqlite3` under `asyncio.to_thread`,
writes gated by an `asyncio.Lock`). The caller computes the key (so this stays
dependency-free).
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chart_ocr_cache (
    cache_key      TEXT PRIMARY KEY,
    pdf_sha256     TEXT NOT NULL,
    page_no        INTEGER NOT NULL,
    bbox_key       TEXT NOT NULL,
    chart_ocr_model TEXT NOT NULL,
    markdown       TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS chart_ocr_cache_pdf ON chart_ocr_cache(pdf_sha256);
"""


class ChartOCRCache:
    """Sync SQLite under the hood; async wrapper via `asyncio.to_thread`."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db
        # Gate multi-statement writes (same rationale as VLMTranscriptionCache /
        # FTSStore): autocommit connection, so SQLite serializes statements;
        # the lock keeps a future multi-statement write atomic. Reads unlocked.
        self._lock = asyncio.Lock()

    @classmethod
    async def open(cls, vault_path: Path) -> ChartOCRCache:
        """Open (or create) the cache db under
        `{vault_path}/.memex/chart_ocr_cache.sqlite` and return a ready store."""
        path = vault_path / ".memex" / "chart_ocr_cache.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        def _connect() -> sqlite3.Connection:
            db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
            db.executescript(_SCHEMA)
            return db

        db = await asyncio.to_thread(_connect)
        return cls(db)

    async def get(self, cache_key: str) -> str | None:
        """Return the cached extraction for `cache_key`, or None on miss.

        NB an empty-string extraction ("not a chart / no parseable data") is a
        legit, reproducible result and IS cached — so a hit can return ""."""

        def _read() -> str | None:
            row = self._db.execute(
                "SELECT markdown FROM chart_ocr_cache WHERE cache_key = ?",
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
        bbox_key: str,
        chart_ocr_model: str,
        markdown: str,
    ) -> None:
        """Store an extraction. `INSERT OR IGNORE` — first writer wins."""
        created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        async with self._lock:

            def _write() -> None:
                self._db.execute(
                    "INSERT OR IGNORE INTO chart_ocr_cache "
                    "(cache_key, pdf_sha256, page_no, bbox_key, chart_ocr_model, "
                    "markdown, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        cache_key,
                        pdf_sha256,
                        page_no,
                        bbox_key,
                        chart_ocr_model,
                        markdown,
                        created_at,
                    ),
                )

            await asyncio.to_thread(_write)

    async def delete_by_pdf(self, pdf_sha256: str) -> int:
        """Drop every cached figure for a source PDF. Returns rows deleted."""
        async with self._lock:

            def _delete() -> int:
                cur = self._db.execute(
                    "DELETE FROM chart_ocr_cache WHERE pdf_sha256 = ?",
                    (pdf_sha256,),
                )
                return cur.rowcount

            return await asyncio.to_thread(_delete)

    async def close(self) -> None:
        await asyncio.to_thread(self._db.close)
