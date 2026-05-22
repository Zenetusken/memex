"""Filler N6 — verify `get_compiled_graph()` is safe under concurrent
first-call by multiple threads.

The audit (2026-05-20) flagged that `_COMPILED_GRAPH` was being
written without a lock on first compile. In the previous code, two
threads entering `get_compiled_graph()` simultaneously could both
observe `_COMPILED_GRAPH is None`, both call `build_answering_graph()`
(an expensive compile), and both write to the global. The visible
work is wasted, and the second writer's reference replaces the first
silently.

The fix adds a `threading.Lock` + double-checked locking: the fast
path (cache hit) skips the lock; the slow path takes it and re-checks.

These tests pin the contract.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from memex.agents import answering


def test_concurrent_get_compiled_graph_builds_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """10 threads hitting `get_compiled_graph()` simultaneously must
    cause exactly ONE call to `build_answering_graph()`. Without the
    lock, multiple threads would race past the `None` check and
    duplicate the compile work.

    Stub `build_answering_graph` with a counting sentinel so we can
    verify the call count directly. A small `time.sleep` inside the
    stub widens the race window so the test reliably catches
    regressions on machines fast enough to schedule all 10 threads
    before the first compile completes.
    """
    import time

    call_count = 0
    count_lock = threading.Lock()
    sentinel = MagicMock(name="compiled_graph")

    def _fake_build() -> object:
        nonlocal call_count
        with count_lock:
            call_count += 1
        # Widen the race window — without this, the first thread might
        # finish before the second even enters the function on fast
        # machines, hiding a regression.
        time.sleep(0.05)
        return sentinel

    monkeypatch.setattr(answering, "build_answering_graph", _fake_build)
    # Start from a clean slate.
    answering.reset_compiled_graph()

    n_threads = 10
    results: list[object] = []

    def _hit() -> None:
        results.append(answering.get_compiled_graph())

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(_hit) for _ in range(n_threads)]
        for f in futures:
            f.result()

    # Critical invariant: only ONE compile happened.
    assert call_count == 1, (
        f"build_answering_graph called {call_count} times; expected "
        "exactly 1 — concurrent compile race not gated"
    )
    # All callers got the SAME compiled graph instance.
    assert len(results) == n_threads
    assert all(r is sentinel for r in results)

    answering.reset_compiled_graph()


def test_get_compiled_graph_fast_path_no_lock_contention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the first compile, repeated calls must return the cached
    graph WITHOUT calling `build_answering_graph()` again. The
    double-checked-locking fast path should observe the cached
    reference and skip the lock entirely.
    """
    call_count = 0
    sentinel = MagicMock(name="compiled_graph")

    def _fake_build() -> object:
        nonlocal call_count
        call_count += 1
        return sentinel

    monkeypatch.setattr(answering, "build_answering_graph", _fake_build)
    answering.reset_compiled_graph()

    # Prime the cache.
    first = answering.get_compiled_graph()
    assert call_count == 1
    assert first is sentinel

    # 100 subsequent calls — none should trigger another compile.
    for _ in range(100):
        result = answering.get_compiled_graph()
        assert result is sentinel
    assert call_count == 1, "fast path triggered rebuild — cache miss?"

    answering.reset_compiled_graph()


def test_reset_compiled_graph_clears_under_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`reset_compiled_graph()` must hold the lock so it can't race
    against a concurrent first-call compile. After reset, the next
    `get_compiled_graph()` rebuilds.
    """
    call_count = 0

    def _fake_build() -> object:
        nonlocal call_count
        call_count += 1
        return MagicMock(name=f"build-{call_count}")

    monkeypatch.setattr(answering, "build_answering_graph", _fake_build)
    answering.reset_compiled_graph()

    answering.get_compiled_graph()
    assert call_count == 1

    answering.reset_compiled_graph()
    answering.get_compiled_graph()
    assert call_count == 2, "reset did not clear the cache"

    answering.reset_compiled_graph()
