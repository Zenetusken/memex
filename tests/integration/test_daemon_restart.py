"""`daemon.supervisor.restart` — applies a co-residence mode's orchestrator
posture (ADR-0007) by setting the vLLM env vars between a stop and a start.

Fakes `stop`/`start` so no real vLLM process is touched.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from memex.core.config import MemexSettings
from memex.daemon import supervisor
from memex.daemon.supervisor import DaemonStatus


@pytest.mark.asyncio
async def test_restart_sets_env_then_stop_then_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    # delenv via monkeypatch so its teardown cleans up restart's direct os.environ writes.
    monkeypatch.delenv("MEMEX_VLLM_GPU_FRACTION", raising=False)
    monkeypatch.delenv("MEMEX_VLLM_MAX_MODEL_LEN", raising=False)
    settings = MemexSettings()  # type: ignore[call-arg]

    calls: list[str] = []

    def _fake_stop(s: MemexSettings) -> DaemonStatus:
        calls.append("stop")
        # restart sets the env AFTER stop — assert it's not set yet here.
        assert "MEMEX_VLLM_GPU_FRACTION" not in os.environ
        return DaemonStatus(
            pid=None,
            alive=False,
            reachable=False,
            base_url=s.inference.base_url,
            pid_file="p",
            log_file="l",
        )

    async def _fake_start(s: MemexSettings) -> DaemonStatus:
        calls.append("start")
        # restart set the env BEFORE start so the serve script picks it up.
        assert os.environ["MEMEX_VLLM_GPU_FRACTION"] == "0.8"
        assert os.environ["MEMEX_VLLM_MAX_MODEL_LEN"] == "24576"
        return DaemonStatus(
            pid=123,
            alive=True,
            reachable=True,
            base_url=s.inference.base_url,
            pid_file="p",
            log_file="l",
        )

    monkeypatch.setattr(supervisor, "stop", _fake_stop)
    monkeypatch.setattr(supervisor, "start", _fake_start)

    status = await supervisor.restart(settings, gpu_fraction=0.8, max_model_len=24576)

    assert calls == ["stop", "start"]  # stop strictly before start
    assert status.reachable is True


@pytest.mark.asyncio
async def test_restart_without_overrides_leaves_env_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    monkeypatch.delenv("MEMEX_VLLM_GPU_FRACTION", raising=False)
    settings = MemexSettings()  # type: ignore[call-arg]

    def _fake_stop(s: MemexSettings) -> DaemonStatus:
        return DaemonStatus(
            pid=None,
            alive=False,
            reachable=False,
            base_url=s.inference.base_url,
            pid_file="p",
            log_file="l",
        )

    async def _fake_start(s: MemexSettings) -> DaemonStatus:
        return DaemonStatus(
            pid=1,
            alive=True,
            reachable=True,
            base_url=s.inference.base_url,
            pid_file="p",
            log_file="l",
        )

    monkeypatch.setattr(supervisor, "stop", _fake_stop)
    monkeypatch.setattr(supervisor, "start", _fake_start)

    await supervisor.restart(settings)  # no gpu_fraction / max_model_len
    assert "MEMEX_VLLM_GPU_FRACTION" not in os.environ  # untouched when not overridden
