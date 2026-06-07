"""Cross-process ryugraph lock-resilience tests.

ryugraph takes an EXCLUSIVE lock on the graph directory on ANY open (root-caused by
`scripts/ryugraph_consistency_probe.py`), so a brief reader/writer race across processes
raises `RuntimeError: ... Could not set lock on file`. These tests spawn a REAL subprocess
that holds the lock and verify the policy helpers in `index/graph_store.py`:

  - `open_graph_for_read`  → FAILS OPEN to None under contention (a discovery read must
    degrade gracefully, never 500 the webui).
  - `open_graph_for_write` → RETRIES a brief window then RE-RAISES (a writer must not
    silently drop a doc's entities).

Mirrors `test_vault_cross_process_lock.py` (a standalone holder script that signals
readiness on stdout). Linux/macOS only; needs ryugraph installed (else skipped).
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path

import pytest

from memex.index.graph_store import (
    is_graph_lock_error,
    open_graph_for_read,
    open_graph_for_write,
)


def test_is_graph_lock_error_classifier() -> None:
    """Matches ryugraph's lock message; rejects other RuntimeErrors + non-RuntimeErrors —
    a bare RuntimeError also wraps corruption/schema errors that must NOT be swallowed."""
    assert is_graph_lock_error(
        RuntimeError("IO exception: Could not set lock on file : /x/graph.ryu")
    )
    assert is_graph_lock_error(RuntimeError("could not set lock"))  # case-insensitive
    assert not is_graph_lock_error(RuntimeError("Table Document does not exist"))
    assert not is_graph_lock_error(RuntimeError("Binder exception: corrupt catalog"))
    assert not is_graph_lock_error(ValueError("Could not set lock"))  # wrong type
    assert not is_graph_lock_error(ImportError("No module named 'ryugraph'"))


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / "documents").mkdir()
    (tmp_path / ".memex").mkdir()
    return tmp_path


@pytest.fixture
def lock_holder_script(tmp_path: Path) -> Path:
    """A standalone script that opens the graph (acquiring ryugraph's exclusive dir lock),
    prints `locked`, holds for N seconds, then exits (process exit frees the lock)."""
    script = tmp_path / "graph_lock_holder.py"
    script.write_text(
        textwrap.dedent("""
            import asyncio, sys, time
            from pathlib import Path
            from memex.index.graph_store import GraphStore

            async def main() -> None:
                vault = Path(sys.argv[1])
                hold_seconds = float(sys.argv[2])
                store = await GraphStore.open(vault)  # acquires the exclusive dir lock
                sys.stdout.write("locked\\n")
                sys.stdout.flush()
                time.sleep(hold_seconds)
                await store.close()

            asyncio.run(main())
        """).lstrip()
    )
    return script


async def _spawn_holder(script: Path, vault: Path, hold_seconds: float) -> asyncio.subprocess.Process:
    """Launch the holder and block until it has acquired the lock (prints `locked`)."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(script), str(vault), str(hold_seconds),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdout is not None
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=30.0)
    assert line.strip() == b"locked", f"holder failed to lock: {line!r}"
    return proc


@pytest.mark.skipif(sys.platform == "win32", reason="cross-process lock test is POSIX-only")
@pytest.mark.asyncio
async def test_open_graph_for_read_fails_open_under_cross_process_lock(
    vault: Path, lock_holder_script: Path
) -> None:
    """A discovery read while another process holds the lock → None (fail-open), no raise."""
    pytest.importorskip("ryugraph")
    holder = await _spawn_holder(lock_holder_script, vault, hold_seconds=3.0)
    try:
        # Retry budget (~0.2s) is far below the 3s hold → must give up and fail open.
        store = await open_graph_for_read(vault, retries=2, backoff_s=0.1)
        assert store is None
    finally:
        await holder.wait()


@pytest.mark.skipif(sys.platform == "win32", reason="cross-process lock test is POSIX-only")
@pytest.mark.asyncio
async def test_open_graph_for_write_retries_then_succeeds(
    vault: Path, lock_holder_script: Path
) -> None:
    """A writer racing a BRIEF reader window retries and eventually opens."""
    pytest.importorskip("ryugraph")
    holder = await _spawn_holder(lock_holder_script, vault, hold_seconds=0.8)
    try:
        # Default write budget (~2s) outlasts the 0.8s hold → the retry succeeds.
        store = await open_graph_for_write(vault)
        assert store is not None
        await store.close()
    finally:
        await holder.wait()


@pytest.mark.skipif(sys.platform == "win32", reason="cross-process lock test is POSIX-only")
@pytest.mark.asyncio
async def test_open_graph_for_write_reraises_when_persistently_locked(
    vault: Path, lock_holder_script: Path
) -> None:
    """A writer that exhausts its retries against a persistent lock RE-RAISES (never a
    silent skip — that would drop the doc's entities)."""
    pytest.importorskip("ryugraph")
    holder = await _spawn_holder(lock_holder_script, vault, hold_seconds=3.0)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            # Tiny budget (~0.1s) ≪ the 3s hold → exhausts retries → re-raises.
            await open_graph_for_write(vault, retries=1, backoff_s=0.05)
        assert is_graph_lock_error(excinfo.value)
    finally:
        await holder.wait()
