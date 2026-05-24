"""Cross-process advisory file lock for vault writes — P1.5.

Layers `fcntl.LOCK_EX` on top of the in-process `asyncio.Lock` so two
`memex` processes editing the same `doc_id` can't race past each
other. Linux + macOS get the real `flock`; Windows falls back to a
one-time WARN + no-op (matches the `pyseccomp` Linux-only gate
already in `pyproject.toml`).

The lock file lives at `vault/.memex/locks/{doc_id}.lock`, mirroring
the manifest pattern. The file itself is empty; the lock is the
fcntl advisory state on the open fd.

Usage (inside `write_document` / `delete_document`):

    async with _lock_for(doc_id):                  # within-process
        async with doc_file_lock(vault_path, doc_id):  # cross-process
            ... atomic write ...
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

_HAS_FCNTL = sys.platform != "win32"
if _HAS_FCNTL:
    import fcntl


def _locks_dir(vault_path: Path) -> Path:
    d = vault_path / ".memex" / "locks"
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    return d


def _lock_path(vault_path: Path, doc_id: str) -> Path:
    return _locks_dir(vault_path) / f"{doc_id}.lock"


def _acquire_blocking(path: Path) -> int:
    """Open `path` and take an exclusive flock. Returns the fd.

    Blocks until acquired. The caller is responsible for calling
    `_release(fd)` to unlock + close. On exception during flock, the
    fd is closed before re-raising.
    """
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)  # type: ignore[name-defined]
    except Exception:
        os.close(fd)
        raise
    return fd


def _release(fd: int) -> None:
    """Release the flock and close the fd. Idempotent re: close."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)  # type: ignore[name-defined]
    finally:
        os.close(fd)


@contextlib.asynccontextmanager
async def doc_file_lock(vault_path: Path, doc_id: str) -> AsyncGenerator[None]:
    """Hold `fcntl.LOCK_EX` on `.memex/locks/{doc_id}.lock` for the body.

    On Linux + macOS: blocks until the exclusive lock is acquired,
    running the flock syscall in a worker thread so the event loop
    stays responsive. Emits `vault.lock.contended` at INFO if the
    wait crosses 100 ms; on acquisition emits `vault.lock.acquired`
    with the total wait. On Windows (no fcntl): logs a one-time WARN
    and yields immediately — multi-process topology isn't supported
    there.
    """
    if not _HAS_FCNTL:
        _warn_no_fcntl_once()
        yield
        return

    path = _lock_path(vault_path, doc_id)
    start = time.monotonic()
    contention_logged = False

    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, _acquire_blocking, path)
    while True:
        done, _ = await asyncio.wait({future}, timeout=0.1)
        if future in done:
            fd = future.result()
            break
        if not contention_logged:
            logger.info(
                "vault.lock.contended",
                doc_id=doc_id,
                waited_ms=int((time.monotonic() - start) * 1000),
            )
            contention_logged = True

    waited_ms = int((time.monotonic() - start) * 1000)
    if waited_ms >= 100:
        logger.info("vault.lock.acquired", doc_id=doc_id, waited_ms=waited_ms)
    try:
        yield
    finally:
        await asyncio.to_thread(_release, fd)


_warned_no_fcntl = False


def _warn_no_fcntl_once() -> None:
    global _warned_no_fcntl
    if _warned_no_fcntl:
        return
    _warned_no_fcntl = True
    logger.warning(
        "vault.lock.no_fcntl",
        platform=sys.platform,
        hint=(
            "fcntl not available; cross-process vault locking is "
            "disabled. Run only one memex process at a time on this "
            "platform."
        ),
    )


def cleanup_lock_file(vault_path: Path, doc_id: str) -> None:
    """Remove the lock file for `doc_id`. Best-effort.

    Safe to call from inside `doc_file_lock`'s body — flock is held
    via the open fd, not the path, so unlinking the path doesn't
    release the lock for the still-open fd. A future
    `doc_file_lock(...)` will create a fresh file at the same path.
    """
    try:
        _lock_path(vault_path, doc_id).unlink(missing_ok=True)
    except OSError:
        pass
