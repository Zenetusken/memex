"""Filler N5 — verify `_pid_alive` returns True on EPERM.

The audit (2026-05-20) flagged that `_pid_alive` returns True when
`os.kill(pid, 0)` raises EPERM. This is **intentional**: EPERM means
the process exists but is owned by another uid (or lives in a
different PID namespace). The function answers "does the kernel show
this PID as alive", not "can we kill it". The supervisor's `stop()`
path acquires its own EPERM from the real signal call and bails out
cleanly.

These tests pin the contract so a future refactor doesn't accidentally
flip the semantics.
"""

from __future__ import annotations

import errno

import pytest

from memex.daemon import supervisor


def test_pid_alive_returns_true_on_eperm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cross-uid case: `os.kill(other_user_pid, 0)` raises EPERM.
    `_pid_alive` must report alive — the process IS alive, just not
    signallable by us. Used by `start()` to detect that an existing
    vLLM may be running under a different user (container, re-install
    under different account) and refuse to spawn a duplicate.
    """

    def _raise_eperm(*_a: object, **_kw: object) -> None:
        e = OSError("operation not permitted")
        e.errno = errno.EPERM
        raise e

    monkeypatch.setattr(supervisor.os, "kill", _raise_eperm)
    assert supervisor._pid_alive(12345) is True


def test_pid_alive_returns_false_on_esrch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reaped-process case: `os.kill(dead_pid, 0)` raises ESRCH.
    `_pid_alive` must report not-alive so the supervisor can spawn a
    fresh daemon.
    """

    def _raise_esrch(*_a: object, **_kw: object) -> None:
        e = OSError("no such process")
        e.errno = errno.ESRCH
        raise e

    monkeypatch.setattr(supervisor.os, "kill", _raise_esrch)
    assert supervisor._pid_alive(99999) is False


def test_pid_alive_returns_true_on_signal_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The healthy-process case: `os.kill(our_pid, 0)` succeeds (returns
    None). `_pid_alive` reports alive. This is the path 99% of `start()`
    calls hit on a running daemon."""
    monkeypatch.setattr(supervisor.os, "kill", lambda *_a, **_kw: None)
    assert supervisor._pid_alive(100) is True


def test_pid_alive_returns_false_for_zero_or_negative_pid() -> None:
    """Defensive case: a pid file containing 0 or -1 (corrupted state)
    must report not-alive without ever calling kill(0, 0) — which would
    signal the entire process group, an unintended side effect.
    """
    assert supervisor._pid_alive(0) is False
    assert supervisor._pid_alive(-1) is False


def test_pid_alive_returns_false_on_unknown_errno(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive case: an unrecognised errno (kernel oddity, weird
    container) is treated as not-alive. Prevents the supervisor from
    deadlocking against a phantom PID it can't classify.
    """

    def _raise_eio(*_a: object, **_kw: object) -> None:
        e = OSError("io error")
        e.errno = errno.EIO  # never expected from kill(pid, 0)
        raise e

    monkeypatch.setattr(supervisor.os, "kill", _raise_eio)
    assert supervisor._pid_alive(5000) is False


def test_pid_alive_eperm_emits_debug_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When EPERM fires, a structured debug event is emitted so an
    operator chasing cross-uid mismatches has a breadcrumb."""

    def _raise_eperm(*_a: object, **_kw: object) -> None:
        e = OSError("operation not permitted")
        e.errno = errno.EPERM
        raise e

    monkeypatch.setattr(supervisor.os, "kill", _raise_eperm)

    captured: list[dict[str, object]] = []

    def _capture_debug(event: str, **kwargs: object) -> None:
        captured.append({"event": event, **kwargs})

    monkeypatch.setattr(supervisor.logger, "debug", _capture_debug)

    assert supervisor._pid_alive(7777) is True
    assert any(c.get("event") == "pid_alive.eperm" and c.get("pid") == 7777 for c in captured)


# ── _argv_matches_daemon: the PID-identity check must survive the script's `exec` ──
# `serve-vllm.sh` ends in `exec uv run … vllm serve …`, so the tracked PID's cmdline
# is the vLLM invocation, NOT the script path. These pin that the identity check
# recognises the exec'd daemon (the regression that silently broke `mode set`).

from pathlib import Path  # noqa: E402

_SCRIPT = Path("/home/u/project/scripts/serve-vllm.sh")


def _argv(*tokens: str) -> list[bytes]:
    """Build a NUL-split-style argv (list of bytes) like `/proc/<pid>/cmdline`."""
    return [t.encode("utf-8") for t in tokens]


def test_argv_matches_execd_uv_run_vllm_serve() -> None:
    # The real regression cmdline: the script exec'd `uv run … vllm serve … --port 8000`.
    args = _argv(
        "uv", "run", "--extra", "serve", "vllm", "serve", "Qwen/Qwen3-8B-AWQ",
        "--host", "127.0.0.1", "--port", "8000", "--max-model-len", "24576",
    )
    assert supervisor._argv_matches_daemon(args, _SCRIPT, 8000) is True


def test_argv_matches_bare_vllm_serve() -> None:
    args = _argv("vllm", "serve", "Qwen/Qwen3-8B-AWQ", "--host", "127.0.0.1", "--port", "8000")
    assert supervisor._argv_matches_daemon(args, _SCRIPT, 8000) is True


def test_argv_matches_vllm_full_path() -> None:
    args = _argv("/home/u/project/.venv/bin/vllm", "serve", "Qwen/Q", "--port", "8000")
    assert supervisor._argv_matches_daemon(args, _SCRIPT, 8000) is True


def test_argv_matches_script_name_when_no_exec() -> None:
    # If the script is run without exec (or pre-exec), the script name still matches.
    args = _argv("bash", str(_SCRIPT))
    assert supervisor._argv_matches_daemon(args, _SCRIPT, 8000) is True


def test_argv_rejects_unrelated_recycled_process() -> None:
    args = _argv("nginx", "-g", "daemon off;")
    assert supervisor._argv_matches_daemon(args, _SCRIPT, 8000) is False


def test_argv_rejects_vllm_serve_on_a_different_port() -> None:
    # Port pins identity: a *second* vLLM on another port is not our daemon.
    args = _argv("vllm", "serve", "other/model", "--port", "9999")
    assert supervisor._argv_matches_daemon(args, _SCRIPT, 8000) is False


def test_argv_accepts_vllm_serve_when_port_unknown() -> None:
    # base_url with no parseable port → accept the vllm-serve signature alone.
    args = _argv("vllm", "serve", "Qwen/Q", "--host", "127.0.0.1", "--port", "8000")
    assert supervisor._argv_matches_daemon(args, _SCRIPT, None) is True
