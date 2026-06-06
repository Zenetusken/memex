"""Serve-config helpers shared between the daemon supervisor and the parse-time
vLLM pause/restart.

`orchestrator_serve_env` + the daemon PID-file location are SHARED concerns: the
`daemon/` supervisor uses them on `start()`/`stop()`, and `parse/pause_vllm_for_gpu`'s
restart must inject the SAME serve-env (so a CLI VLM/chart-OCR parse brings the
CONFIGURED orchestrator model back, not the serve-script's hardcoded default — the
ADR-0015 silent-404, previously fixed only on the webui ingest path) and track the
restarted PID in the SAME file. They live HERE in `core/` (no Memex deps beyond
`MemexSettings`) so `parse/` can reuse them WITHOUT a `parse/ → daemon/` import edge
(forbidden by the module-boundary rules); the daemon supervisor imports them from here.
"""

from __future__ import annotations

from pathlib import Path

from memex.core.config import MemexSettings

# --- Orchestrator serve-env bridge (ADR-0015) -----------------------------
# `scripts/serve-vllm.sh` hardcodes the model (Qwen/Qwen3-8B-AWQ), the
# `--quantization` flag (awq_marlin), and the KV dtype (fp8_e5m2) as env
# DEFAULTS, while the client (`models/client.py`) sends
# `settings.models.orchestrator` as the requested model name. Without
# exporting the configured model id into the serve env, a config-only model
# swap silently 404s — the daemon serves the hardcoded 8B while every /ask
# requests the new id. This bridge makes `settings.models.orchestrator` the
# single source of truth the daemon actually serves, and maps the quant tier
# to the matching serve flags.
_ORCH_QUANT_FLAG: dict[str, str] = {
    # config quant tier -> the vLLM `--quantization` flag value. EMPTY means
    # OMIT the flag (compressed-tensors auto-detects; serve-vllm.sh's
    # single-dash `${VAR-default}` treats an explicit '' as "no flag").
    "AWQ": "awq_marlin",
    "GPTQ": "gptq_marlin",
    "compressed_tensors": "",
}
_ORCH_KV_DTYPE: dict[str, str] = {
    # compressed-tensors W4A16 builds carry fp8 components; vLLM rejects
    # `fp8_e5m2` KV on an fp8 checkpoint ("fp8_e5m2 kv-cache is not supported
    # with fp8 checkpoints" — confirmed at 4B cold start), so that tier MUST
    # use `auto` KV (fp16). AWQ/GPTQ keep the proven fp8_e5m2 default.
    "AWQ": "fp8_e5m2",
    "GPTQ": "fp8_e5m2",
    "compressed_tensors": "auto",
}


def orchestrator_serve_env(settings: MemexSettings) -> dict[str, str]:
    """The per-model vLLM serve environment `scripts/serve-vllm.sh` reads.

    Maps `settings.models.orchestrator` + `orchestrator_quantization` to the
    `MEMEX_VLLM_MODEL` / `MEMEX_VLLM_QUANTIZATION` / `MEMEX_VLLM_KV_CACHE_DTYPE`
    the serve script consumes. The daemon `start()` exports these before spawning,
    and `parse/pause_vllm_for_gpu`'s restart merges them into the spawn env, so the
    detached child inherits them. Pure + unit-testable (no os.environ mutation
    here). An unknown quant tier maps to an empty quantization flag (let vLLM
    auto-detect) + `auto` KV (the safe default). The empty string for
    `compressed_tensors` is DELIBERATE — it must be SET-to-empty, never
    unset, so the serve script's `${VAR-default}` omits the flag rather than
    falling back to awq_marlin.
    """
    quant = settings.models.orchestrator_quantization
    return {
        "MEMEX_VLLM_MODEL": settings.models.orchestrator,
        "MEMEX_VLLM_QUANTIZATION": _ORCH_QUANT_FLAG.get(quant, ""),
        "MEMEX_VLLM_KV_CACHE_DTYPE": _ORCH_KV_DTYPE.get(quant, "auto"),
    }


def daemon_dir(settings: MemexSettings) -> Path:
    """The vLLM daemon's state dir (`vault/.memex/daemon`), created 0o700.

    Holds the PID + log files. Mode 0o700 is deliberate (it holds a PID a stop
    signals) — keep it in lockstep with any other creator (the supervisor + the
    parse-restart PID-write both delegate here so the perms can't diverge)."""
    d = settings.vault_path / ".memex" / "daemon"
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    return d


def daemon_pid_file(settings: MemexSettings) -> Path:
    """The vLLM daemon's PID file (`vault/.memex/daemon/vllm.pid`).

    Written by `daemon.supervisor.start()` AND by `parse/pause_vllm_for_gpu`'s
    restart (so a CLI-parse-restarted orchestrator stays supervisor-tracked — a
    later `daemon stop`/`restart` finds + kills the right process group instead of
    leaving a stray that blocks the port)."""
    return daemon_dir(settings) / "vllm.pid"
