"""Content-addressed cache for ASR per-chunk transcriptions (ADR-0017).

The ASR backend's greedy decode is reproducible for a fixed input + hardware + library
version but NOT bit-exact across CUDA/cuDNN/library upgrades, so — exactly like the VLM cache
— this cache (not decode determinism) is what keeps the content-addressed `chunk_id` stable
across a re-parse. The unit is the **VAD chunk**, which is deterministic from the audio bytes +
VAD params, so its key is known BEFORE transcription (and a re-parse re-runs the same VAD →
same chunks → cache hits). The cached value is that chunk's emitted segments (text +
chunk-local timestamps) serialized as JSON.

The key is `sha256(audio_bytes):chunk_index:model:cfg` where `cfg` is an 8-char hash over the
DECODING params (backend, beam_size, language, temperature, VAD/chunk window) — so a
decoding-param change is a clean MISS, never a stale replay (the one extra key component vs the
VLM cache, which has a prompt instead).

Mirrors `parse/vlm_cache.py`: sync `sqlite3` under `asyncio.to_thread`, multi-statement writes
gated by an `asyncio.Lock`. Regenerable derived state under `vault/.memex/` (ADR-0003), dropped
by `reindex --force`. See `docs/specs/audio-asr-route.md` §9.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import structlog

from memex.core.sqlite_tuning import apply_sqlite_pragmas

logger = structlog.get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS asr_chunk_cache (
    cache_key     TEXT PRIMARY KEY,
    audio_sha256  TEXT NOT NULL,
    chunk_index   INTEGER NOT NULL,
    asr_model     TEXT NOT NULL,
    cfg_sha8      TEXT NOT NULL,
    segments_json TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS asr_cache_audio ON asr_chunk_cache(audio_sha256);
"""


def cfg_sha8(decoding_cfg: Mapping[str, object]) -> str:
    """8-char hash over the DECODING params (order-independent). Folded into the cache key so a
    param change (beam_size, language, temperature, VAD/chunk window, backend) is a clean miss."""
    canonical = json.dumps(decoding_cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def build_asr_cache_key(*, audio_sha256: str, chunk_index: int, model: str, cfg: str) -> str:
    """The per-VAD-chunk cache key: `sha256(audio):chunk_index:model:cfg` (`cfg` from `cfg_sha8`).
    Input-derived (the VAD chunk boundaries are deterministic), so it is known pre-transcription."""
    return f"{audio_sha256}:{chunk_index}:{model}:{cfg}"


class ASRTranscriptionCache:
    """Sync SQLite under the hood; async wrapper via `asyncio.to_thread`."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db
        # Gate multi-statement writes (same rationale as FTSStore._lock / VLMTranscriptionCache):
        # the connection is autocommit, so SQLite serializes individual statements, but the lock
        # keeps a future multi-statement write atomic. Reads stay unlocked.
        self._lock = asyncio.Lock()

    @classmethod
    async def open(cls, vault_path: Path) -> ASRTranscriptionCache:
        """Open (or create) the cache db under `{vault_path}/.memex/asr_cache.sqlite`."""
        path = vault_path / ".memex" / "asr_cache.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        def _connect() -> sqlite3.Connection:
            db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
            apply_sqlite_pragmas(db)  # WAL + cache + mmap (ADR-0003 derived state)
            db.executescript(_SCHEMA)
            return db

        db = await asyncio.to_thread(_connect)
        return cls(db)

    async def get(self, cache_key: str) -> str | None:
        """Return the cached `segments_json` for `cache_key`, or None on miss."""

        def _read() -> str | None:
            row = self._db.execute(
                "SELECT segments_json FROM asr_chunk_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            return None if row is None else str(row[0])

        return await asyncio.to_thread(_read)

    async def put(
        self,
        cache_key: str,
        *,
        audio_sha256: str,
        chunk_index: int,
        asr_model: str,
        cfg_sha8: str,
        segments_json: str,
    ) -> None:
        """Store a chunk's transcribed segments. `INSERT OR IGNORE` — first writer wins
        (concurrent draws of the same key came from the same chunk/model/cfg)."""
        created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        async with self._lock:

            def _write() -> None:
                self._db.execute(
                    "INSERT OR IGNORE INTO asr_chunk_cache "
                    "(cache_key, audio_sha256, chunk_index, asr_model, cfg_sha8, "
                    "segments_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        cache_key,
                        audio_sha256,
                        chunk_index,
                        asr_model,
                        cfg_sha8,
                        segments_json,
                        created_at,
                    ),
                )

            await asyncio.to_thread(_write)

    async def delete_by_audio(self, audio_sha256: str) -> int:
        """Drop every cached chunk for a source audio file (the `--refresh-asr` bust). Returns
        the number of rows deleted."""
        async with self._lock:

            def _delete() -> int:
                cur = self._db.execute(
                    "DELETE FROM asr_chunk_cache WHERE audio_sha256 = ?",
                    (audio_sha256,),
                )
                return cur.rowcount

            return await asyncio.to_thread(_delete)

    async def close(self) -> None:
        await asyncio.to_thread(self._db.close)
