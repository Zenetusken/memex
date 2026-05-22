"""Filler N3 — `configure_client` httpx pool linger.

Audit item: a previous version of `configure_client` simply reassigned
the module-global `_client`, leaking the prior httpx connection pool.
AsyncOpenAI wraps an httpx AsyncClient with persistent keep-alive
connections; without explicit `.close()` they linger until GC reaps the
orphan, at which point httpx emits a ResourceWarning. In a long-running
daemon that re-reads config (e.g., SIGHUP-driven reconfig), this slowly
exhausts the file-descriptor budget.

The fix: best-effort schedule `.close()` on the running event loop. If
no loop is running, log a warning so the operator knows the cleanup
deferred to GC.

These tests verify both code paths.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from memex.core.config import InferenceSettings
from memex.models import client as client_module


def _settings(base_url: str = "http://test:8000/v1") -> InferenceSettings:
    return InferenceSettings(
        base_url=base_url,
        api_key="test-key",
        request_timeout_s=30.0,
    )


@pytest.mark.asyncio
async def test_configure_client_schedules_close_when_loop_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When called from within an active asyncio loop, configure_client
    MUST schedule the prior client's `close()` as a task. This is the
    daemon-SIGHUP reconfiguration path — clean handoff with no
    connection-pool linger.
    """
    # Sentinel: track that close() was scheduled. Real AsyncOpenAI is
    # imported through langfuse's wrapper; we stub the constructor to
    # return a MagicMock so we don't open real HTTP connections.
    close_calls: list[int] = []

    class _FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True
            close_calls.append(1)

    monkeypatch.setattr(client_module, "AsyncOpenAI", _FakeClient)
    monkeypatch.setattr(client_module, "_client", None)

    # First call — no prior client to close, just sets the singleton.
    client_module.configure_client(_settings())
    first = client_module._client
    assert first is not None
    assert close_calls == []  # nothing to close yet

    # Second call — should schedule the close of the FIRST client.
    client_module.configure_client(_settings("http://test:8001/v1"))
    # Yield to the loop so the scheduled task actually runs.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert close_calls == [1], (
        "configure_client must schedule the prior client's .close() "
        "via the running event loop"
    )
    assert first.closed is True

    # Subsequent reconfigure should close the second client too.
    second = client_module._client
    client_module.configure_client(_settings("http://test:8002/v1"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert close_calls == [1, 1]
    assert second.closed is True


def test_configure_client_logs_warning_when_no_loop_running(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When called from sync context with no running loop, the prior
    client cannot be closed (close() is async). We log a warning so
    the operator knows GC will eventually reclaim the socket. This is
    the test-reconfiguration path — acceptable because each test process
    exits anyway.
    """
    close_calls: list[int] = []

    class _FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def close(self) -> None:
            close_calls.append(1)

    monkeypatch.setattr(client_module, "AsyncOpenAI", _FakeClient)
    monkeypatch.setattr(client_module, "_client", None)

    # First call sets up the client (no prior to close).
    client_module.configure_client(_settings())
    assert close_calls == []

    # Capture structlog output to confirm the warning fires.
    warnings: list[dict[str, object]] = []
    original_warning = client_module.logger.warning

    def _capture_warning(event: str, **kwargs: object) -> None:
        warnings.append({"event": event, **kwargs})
        original_warning(event, **kwargs)

    monkeypatch.setattr(client_module.logger, "warning", _capture_warning)

    # Second call from sync context — no running loop. Schedule should
    # fail; warning should fire; close should NOT be called (best-effort).
    client_module.configure_client(_settings("http://test:8001/v1"))

    assert close_calls == [], (
        "without a running loop, close() cannot be scheduled — "
        "must NOT be called synchronously"
    )
    assert any(
        w.get("event") == "configure_client.no_loop_for_cleanup"
        for w in warnings
    ), (
        "configure_client must emit a warning when called sync with a "
        f"prior client present; saw events: {[w.get('event') for w in warnings]}"
    )


def test_configure_client_first_call_does_not_warn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first call to configure_client (when no prior client exists)
    must NOT emit the no-loop warning — there's nothing to close. Keeps
    the warning signal clean for actual leaks."""

    class _FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def close(self) -> None:
            pass

    monkeypatch.setattr(client_module, "AsyncOpenAI", _FakeClient)
    monkeypatch.setattr(client_module, "_client", None)

    warnings: list[dict[str, object]] = []

    def _capture_warning(event: str, **kwargs: object) -> None:
        warnings.append({"event": event, **kwargs})

    monkeypatch.setattr(client_module.logger, "warning", _capture_warning)

    client_module.configure_client(_settings())

    assert not any(
        w.get("event") == "configure_client.no_loop_for_cleanup"
        for w in warnings
    ), "first call should not warn — there's no prior client to close"
