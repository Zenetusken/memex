"""Unit tests for the orchestrator serve-env bridge + the 4B VRAM tier.

The bridge (`daemon/supervisor.orchestrator_serve_env`) maps
`settings.models.orchestrator` + `orchestrator_quantization` to the
`MEMEX_VLLM_*` serve env `scripts/serve-vllm.sh` reads, so a config-only
model swap actually changes what the daemon serves — without it the script's
hardcoded 8B default silently 404s the client (which sends
`settings.models.orchestrator`). See Phase 1 of the Qwen3.5-4B swap.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **env: str):  # type: ignore[no-untyped-def]  # returns MemexSettings; lazy import keeps torch out of the import path
    """Build MemexSettings with a tmp vault and the given model env."""
    from memex.core.config import MemexSettings

    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return MemexSettings()  # type: ignore[call-arg]


def test_default_orchestrator_is_the_4b(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Post-ADR-0015 the committed default IS the unified Qwen3.5-4B → it omits
    the quant flag (compressed-tensors auto-detect) and uses auto KV."""
    from memex.daemon.supervisor import orchestrator_serve_env

    env = orchestrator_serve_env(_settings(monkeypatch, tmp_path))
    assert env["MEMEX_VLLM_MODEL"] == "cyankiwi/Qwen3.5-4B-AWQ-4bit"
    assert env["MEMEX_VLLM_QUANTIZATION"] == ""
    assert env["MEMEX_VLLM_KV_CACHE_DTYPE"] == "auto"


def test_8b_awq_kill_switch_serve_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The 8B kill-switch path maps to the proven awq_marlin + fp8_e5m2 serve —
    byte-identical to the pre-swap serve-vllm.sh defaults (the rollback regression guard)."""
    from memex.daemon.supervisor import orchestrator_serve_env

    env = orchestrator_serve_env(
        _settings(
            monkeypatch,
            tmp_path,
            MEMEX_MODELS__ORCHESTRATOR="Qwen/Qwen3-8B-AWQ",
            MEMEX_MODELS__ORCHESTRATOR_QUANTIZATION="AWQ",
        )
    )
    assert env["MEMEX_VLLM_MODEL"] == "Qwen/Qwen3-8B-AWQ"
    assert env["MEMEX_VLLM_QUANTIZATION"] == "awq_marlin"
    assert env["MEMEX_VLLM_KV_CACHE_DTYPE"] == "fp8_e5m2"


def test_compressed_tensors_4b_serve_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The unified Qwen3.5-4B (compressed-tensors) omits the quant flag and uses
    auto KV (it's an fp8 checkpoint that rejects fp8_e5m2 KV)."""
    from memex.daemon.supervisor import orchestrator_serve_env

    env = orchestrator_serve_env(
        _settings(
            monkeypatch,
            tmp_path,
            MEMEX_MODELS__ORCHESTRATOR="cyankiwi/Qwen3.5-4B-AWQ-4bit",
            MEMEX_MODELS__ORCHESTRATOR_QUANTIZATION="compressed_tensors",
        )
    )
    assert env["MEMEX_VLLM_MODEL"] == "cyankiwi/Qwen3.5-4B-AWQ-4bit"
    # CRITICAL: the empty quantization must be PRESENT (set-to-empty), NOT absent.
    # serve-vllm.sh's single-dash `${VAR-default}` omits the flag only for an
    # explicitly-empty var; an UNSET var falls back to awq_marlin → the wrong
    # kernel for compressed-tensors → a startup refusal.
    assert "MEMEX_VLLM_QUANTIZATION" in env
    assert env["MEMEX_VLLM_QUANTIZATION"] == ""
    # fp8 checkpoint -> auto KV (fp8_e5m2 is rejected at cold start).
    assert env["MEMEX_VLLM_KV_CACHE_DTYPE"] == "auto"


def test_gptq_serve_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from memex.daemon.supervisor import orchestrator_serve_env

    env = orchestrator_serve_env(
        _settings(
            monkeypatch, tmp_path, MEMEX_MODELS__ORCHESTRATOR_QUANTIZATION="GPTQ"
        )
    )
    assert env["MEMEX_VLLM_QUANTIZATION"] == "gptq_marlin"
    assert env["MEMEX_VLLM_KV_CACHE_DTYPE"] == "fp8_e5m2"


@pytest.mark.parametrize("mode", ["manual", "fast", "full", "gpu_only"])
def test_estimated_vram_compressed_tensors_no_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    """The atomic partner of the config Literal: `_estimated_vram_gb` must not
    KeyError for the compressed_tensors tier at any co-residence mode (the
    bootstrap.py:128 subscript would brick daemon startup without the table entry)."""
    from memex.cli.bootstrap import _VRAM_GB, _estimated_vram_gb

    settings = _settings(
        monkeypatch,
        tmp_path,
        MEMEX_MODELS__ORCHESTRATOR="cyankiwi/Qwen3.5-4B-AWQ-4bit",
        MEMEX_MODELS__ORCHESTRATOR_QUANTIZATION="compressed_tensors",
        MEMEX_MODELS__CO_RESIDENCE_MODE=mode,
    )
    estimated = _estimated_vram_gb(settings)  # must not raise
    # The orchestrator is always counted; the 4B tier is the floor.
    assert estimated >= _VRAM_GB[("orchestrator", "compressed_tensors")]
