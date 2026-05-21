"""Cross-process vault-lock integration tests (P1.5).

These spawn a real subprocess that holds `fcntl.LOCK_EX` on the lock
file and verify the parent's `write_document` call blocks until the
subprocess releases — exercising the actual cross-process race, not
a mock. Linux + macOS only; the tests `skip` on Windows.
"""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
from pathlib import Path

import pytest

from memex.vault._file_lock import _lock_path
from memex.vault.store import (
    Frontmatter,
    VaultDocument,
    hash_bytes,
    make_ref,
    read_document,
    write_document,
)


def _build_doc(vault: Path, doc_id: str, body: str) -> VaultDocument:
    return VaultDocument(
        ref=make_ref(
            vault, doc_id, content_sha256=hash_bytes(body.encode("utf-8"))
        ),
        frontmatter=Frontmatter(title=doc_id),
        body=body,
        mtime_ns=0,
    )


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / "documents").mkdir()
    return tmp_path


@pytest.fixture
def lock_holder_script(tmp_path: Path) -> Path:
    """A standalone Python script that takes the lock + sleeps.

    Prints `locked` to stdout once the flock is held so the parent
    knows it's safe to start its contended write. Exits cleanly.
    """
    script = tmp_path / "lock_holder.py"
    script.write_text(
        textwrap.dedent("""
            import fcntl, os, sys, time
            path = sys.argv[1]
            hold_seconds = float(sys.argv[2])
            fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            sys.stdout.write("locked\\n")
            sys.stdout.flush()
            time.sleep(hold_seconds)
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        """).lstrip()
    )
    return script


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl not available")
@pytest.mark.asyncio
async def test_write_document_blocks_on_cross_process_lock(
    vault: Path, lock_holder_script: Path
) -> None:
    """A subprocess holds the lock for ~1 s; the parent's
    `write_document` on the same `doc_id` must block until the
    subprocess releases. Wall time of the parent's write is the
    proof.
    """
    doc_id = "00000006-xproc"
    # Seed the doc so the contended write has something to overwrite.
    seed_ref = await write_document(vault, _build_doc(vault, doc_id, "seed"))
    lock_path = _lock_path(vault, doc_id)
    # Sanity: the seed write should have created the lock file.
    assert lock_path.exists()

    hold_seconds = 1.0
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(lock_holder_script),
        str(lock_path),
        str(hold_seconds),
        stdout=asyncio.subprocess.PIPE,
    )
    assert proc.stdout is not None
    # Wait for the subprocess to confirm it holds the flock.
    ready = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
    assert ready.strip() == b"locked"

    start = asyncio.get_event_loop().time()
    new_ref = await write_document(
        vault, _build_doc(vault, doc_id, "after-lock-released")
    )
    elapsed = asyncio.get_event_loop().time() - start
    # Allow some slop for subprocess teardown + scheduling.
    assert elapsed >= hold_seconds - 0.2, (
        f"parent didn't wait long enough: {elapsed:.3f}s vs hold {hold_seconds}s"
    )
    # And it shouldn't have waited much longer than the hold + a small
    # margin — if it did, the lock wasn't promptly released.
    assert elapsed < hold_seconds + 2.0

    await proc.wait()

    on_disk = await read_document(vault, doc_id)
    assert "after-lock-released" in on_disk.body
    assert new_ref.content_sha256 != seed_ref.content_sha256


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl not available")
@pytest.mark.asyncio
async def test_concurrent_writes_within_process_dont_corrupt(
    vault: Path,
) -> None:
    """`asyncio.gather` three writes on the same `doc_id`. The
    asyncio.Lock layer serialises them in-process so the inner
    flock can't deadlock against itself; all three must complete
    cleanly and one of their bodies must end up on disk.
    """
    doc_id = "00000007-same-process"
    await write_document(vault, _build_doc(vault, doc_id, "seed"))

    async def writer(body: str) -> None:
        await write_document(vault, _build_doc(vault, doc_id, body))

    await asyncio.gather(writer("a"), writer("b"), writer("c"))
    on_disk = await read_document(vault, doc_id)
    # Whichever write ran last wins; all three completed without raising.
    assert on_disk.body in {"a", "b", "c"}


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl not available")
@pytest.mark.asyncio
async def test_delete_document_unlinks_lock_file(vault: Path) -> None:
    """After `delete_document`, the lock file must be unlinked so the
    `.memex/locks/` directory doesn't accumulate dead entries.
    """
    from memex.vault.store import delete_document

    doc_id = "00000008-delete-cleanup"
    await write_document(vault, _build_doc(vault, doc_id, "v1"))
    lock_path = _lock_path(vault, doc_id)
    assert lock_path.exists()

    await delete_document(vault, doc_id)
    assert not lock_path.exists()
