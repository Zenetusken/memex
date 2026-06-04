"""Tests for the dynamic-VRAM-manager lifecycle hardening: the `pause_vllm_for_gpu` restart-retry (the
load-bearing reliability fix), the boot GPU pre-flight, registry.unload_all, and the holder enumerator.
All heavy I/O (subprocess, sleeps, reachability) is faked — no GPU, no real vLLM."""

from __future__ import annotations

from pathlib import Path

import pytest

from memex.core.config import MemexSettings, set_settings


@pytest.fixture(autouse=True)
def _settings(tmp_path: Path):  # type: ignore[no-untyped-def]
    set_settings(MemexSettings(vault_path=tmp_path))
    yield
    set_settings(None)


async def _noop_sleep(_seconds: float) -> None:
    return None


class _FakeVllm:
    """Models a restart that fails attempt 1 (slow warmup / contention) and succeeds attempt 2."""

    def __init__(self, *, succeed_after_restarts: int = 2) -> None:
        self.reach_calls = 0
        self.restart_calls = 0
        self.pkill_calls = 0
        self._succeed_after = succeed_after_restarts

    async def reachable(self, _url: str, timeout_s: float = 2.0) -> bool:
        self.reach_calls += 1
        if self.reach_calls == 1:
            return True  # the initial was_running probe → so the CM actually pauses
        return self.restart_calls >= self._succeed_after  # reachable only once enough restarts fired

    async def restart(self, _scripts_dir: Path) -> None:
        self.restart_calls += 1

    async def pkill(self) -> None:
        self.pkill_calls += 1


def _patch_pause(monkeypatch: pytest.MonkeyPatch, fake: _FakeVllm) -> None:
    from memex.parse import pipeline as P

    monkeypatch.setattr(P, "_vllm_reachable", fake.reachable)
    monkeypatch.setattr(P, "_vllm_restart", fake.restart)
    monkeypatch.setattr(P, "_vllm_pkill", fake.pkill)
    monkeypatch.setattr(P, "_detect_scripts_dir", lambda: Path("/tmp"))  # noqa: S108
    monkeypatch.setattr(P.asyncio, "sleep", _noop_sleep)  # instant polls


@pytest.mark.asyncio
async def test_pause_vllm_restart_retries_until_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The load-bearing fix: a 1st restart that doesn't come back is RETRIED (vs the old single attempt
    that left the 4B down + broke the next /ask). The retry re-pkills (the script fallback double-spawns
    otherwise)."""
    from memex.parse.pipeline import pause_vllm_for_gpu

    fake = _FakeVllm(succeed_after_restarts=2)
    _patch_pause(monkeypatch, fake)

    async with pause_vllm_for_gpu():
        pass

    assert fake.restart_calls == 2  # attempt 1 failed → attempt 2 retried
    assert fake.pkill_calls >= 2  # the initial pause pkill + the retry's re-pkill (double-spawn guard)


@pytest.mark.asyncio
async def test_pause_vllm_finally_preserves_body_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """B012: the restart-retry loop must NEVER raise/return out of the `finally` — a parse-body exception
    must still surface even when every restart attempt is exhausted."""
    from memex.parse.pipeline import pause_vllm_for_gpu

    fake = _FakeVllm(succeed_after_restarts=999)  # never comes back → exhausts the retry budget
    _patch_pause(monkeypatch, fake)

    with pytest.raises(ValueError, match="boom"):
        async with pause_vllm_for_gpu():
            raise ValueError("boom")
    assert fake.restart_calls >= 2  # it did exhaust the bounded retries (didn't hang / didn't swallow)


# ── boot pre-flight ──────────────────────────────────────────────────────────────────────────────────
def test_preflight_gpu_noop_off_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from memex.cli import bootstrap as B

    monkeypatch.setattr("memex.core.vram.free_vram_gb", lambda: None)
    monkeypatch.setattr("memex.core.vram.total_vram_gb", lambda: None)
    B._preflight_gpu(MemexSettings(vault_path=Path("/tmp")))  # noqa: S108 — must not raise off-GPU


def test_preflight_gpu_runs_with_a_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    # Low free + a holder present → the warning path; just assert it runs cleanly (report-and-warn, no kill).
    from memex.cli import bootstrap as B

    monkeypatch.setattr("memex.core.vram.free_vram_gb", lambda: 2.0)
    monkeypatch.setattr("memex.core.vram.total_vram_gb", lambda: 12.0)
    monkeypatch.setattr("memex.core.vram.gpu_compute_apps", lambda: ["pid 123: python (2024 MiB)"])
    B._preflight_gpu(MemexSettings(vault_path=Path("/tmp")))  # noqa: S108


# ── registry.unload_all (clean shutdown) ─────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_registry_unload_all_releases_every_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    from memex.core.config import ModelSettings
    from memex.models.registry import ModelRegistry

    reg = ModelRegistry(ModelSettings())
    unloaded: list[str] = []

    async def _fake_unload(name: str) -> None:
        unloaded.append(name)

    monkeypatch.setattr(reg, "unload", _fake_unload)
    await reg.unload_all()
    assert unloaded == ["embedder", "reranker", "vlm", "chart_ocr"]


def test_gpu_compute_apps_none_when_nvidia_smi_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # FileNotFoundError (nvidia-smi not installed) → None, never raises (it runs in the error path).
    import subprocess

    from memex.core import vram

    def _boom(*_a: object, **_k: object) -> object:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert vram.gpu_compute_apps() is None


@pytest.mark.asyncio
async def test_serve_vlm_fails_fast_with_vramexhausted_naming_holder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dynamic VLM-serve util: when even after freeing the retrieval models the GPU can't hold
    weights+KV, raise VRAMExhausted NAMING the holder (vs vLLM's cryptic ValueError) — and it WIDENS
    (unload_retrieval) before giving up. No subprocess is spawned (it fails before the spawn)."""
    from memex.core.errors import VRAMExhausted
    from memex.parse import vlm_backend as V

    # free 7.0 → budget 6.4 < the 8.5 GB floor → fit_serve_util returns None (can't fit) on every probe.
    monkeypatch.setattr("memex.core.vram.free_vram_gb", lambda: 7.0)
    monkeypatch.setattr("memex.core.vram.total_vram_gb", lambda: 12.0)
    monkeypatch.setattr("memex.core.vram.gpu_compute_apps", lambda: ["pid 9: hog (5000 MiB)"])
    monkeypatch.setattr(V.asyncio, "sleep", _noop_sleep)
    unloaded = {"n": 0}

    class _Reg:
        async def unload_retrieval(self) -> None:
            unloaded["n"] += 1

    monkeypatch.setattr(V, "get_registry", lambda: _Reg())

    with pytest.raises(VRAMExhausted) as ei:
        async with V._serve_vlm_vllm("model-x"):
            pass
    assert unloaded["n"] == 1  # it widened (freed the retrieval models) before failing fast
    assert "hog" in str(ei.value.context.get("holders"))  # the error names which process holds the GPU


def test_webui_shutdown_releases_gpu_models() -> None:
    """The webui lifespan calls registry.unload_all() on shutdown (the clean-shutdown half of the dynamic
    VRAM manager) so its VRAM doesn't linger and contend with the next process."""
    from fastapi.testclient import TestClient

    from memex.models import registry as R
    from memex.webui.app import create_app

    called = {"n": 0}

    class _FakeReg:
        async def unload_all(self) -> None:
            called["n"] += 1

    R.set_registry(_FakeReg())  # type: ignore[arg-type]  # duck-typed; only unload_all is exercised
    try:
        with TestClient(create_app()):  # __enter__/__exit__ runs the lifespan (startup + shutdown)
            pass
    finally:
        R.set_registry(None)
    assert called["n"] == 1  # released on shutdown
