"""Live GPU VRAM probe + the dynamic-fit formulas — the seam the dynamic VRAM manager reads.

`core/resources.py` is PURE (config imports it) and must stay torch-free; this module is where the
torch-dependent LIVE probe lives. `resolve_profile`/`fit_serve_util` take the probed numbers as inputs,
so the policy logic stays pure and unit-testable while the placement decisions become VRAM-aware.

Only Memex dep is the leaf `core.types` (pure pydantic, never imports this → no cycle), so this stays
importable anywhere (registry, parse, webui all read it). Off-GPU / torch-absent → the probes return
`None` and every caller falls back to its static default (so a CPU-only or torch-less environment behaves
exactly as before).
"""

from __future__ import annotations

from memex.core.types import GpuProcess

_GB = 1024.0**3


def _mem_get_info() -> tuple[int, int] | None:
    """`(free_bytes, total_bytes)` for the active CUDA device, or `None` if CUDA/torch is unavailable.

    Lazy torch import (heavy optional dep); any failure → `None` (caller uses its static default)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return int(free), int(total)
    except Exception:  # torch absent / driver error / no device — degrade to the static default
        return None


def free_vram_gb() -> float | None:
    """Live FREE VRAM (GB) on the active GPU, or `None` off-GPU. Reads the device-global free, so it
    reflects every resident consumer (the orchestrator vLLM, other processes, the desktop)."""
    info = _mem_get_info()
    return None if info is None else info[0] / _GB


def total_vram_gb() -> float | None:
    """Total VRAM (GB) on the active GPU, or `None` off-GPU."""
    info = _mem_get_info()
    return None if info is None else info[1] / _GB


# Floor for the short-lived VLM vLLM (`parse/vlm_backend._serve_vlm_vllm`): the serve needs its weights
# (~7.4 GB for Qwen3-VL-8B-AWQ) PLUS minimal KV. Calibrated from the live datum: at util 0.70 (8.4 GB
# budget) startup was "0.42 GiB KV needed vs 0.35 available" — 70 MB short — so the real floor is ~8.5 GB.
_VLM_SERVE_FLOOR_GB = 8.5
# Leave this much of `free` unused so vLLM's own init-time "free memory < desired" check passes.
_VLM_SERVE_SAFETY_GB = 0.6
# Never exceed the validated desktop-peak cap (the static default when the GPU is otherwise free).
_VLM_SERVE_UTIL_CAP = 0.80


def fit_serve_util(
    free_gb: float | None,
    total_gb: float | None,
    *,
    floor_gb: float = _VLM_SERVE_FLOOR_GB,
    safety_gb: float = _VLM_SERVE_SAFETY_GB,
    cap: float = _VLM_SERVE_UTIL_CAP,
) -> float | None:
    """The `--gpu-memory-utilization` for the VLM serve that FITS the live free VRAM, or `None` if even the
    max available can't hold weights+KV (→ caller fails fast with an actionable error instead of vLLM's
    cryptic `ValueError`).

    PURE (arithmetic only). `vLLM` reserves `util × total` of the card, but only `free` is actually
    available, so we cap `util` at `(free − safety)/total` to dodge the "Free memory < desired" startup
    error, and require that budget ≥ `floor_gb` to dodge the "No available memory for the cache blocks"
    error. When the probe is unavailable (`None`) or the GPU is wide open, this returns the static `cap`
    (0.80) — no regression on a free GPU."""
    if free_gb is None or total_gb is None or total_gb <= 0:
        return cap
    budget_gb = free_gb - safety_gb
    if budget_gb < floor_gb:
        return None  # can't fit weights + minimal KV even using all available free VRAM
    return min(budget_gb / total_gb, cap)


def gpu_processes() -> list[GpuProcess] | None:
    """Best-effort STRUCTURED list of processes holding GPU memory, via `nvidia-smi`. Returns `None` if
    `nvidia-smi` is absent / errors / reports nothing. NEVER raises. The structured sibling of
    `gpu_compute_apps` (which formats from this) — consumed by the webui `/resources` VRAM panel to show
    the holder breakdown (orchestrator vs this process vs other), and by the `VRAMExhausted` error path."""
    import subprocess

    try:
        out = subprocess.run(
            [  # noqa: S607 — nvidia-smi is a trusted system binary on PATH; fixed argv, no shell
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # nvidia-smi absent / timeout — no holder info
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    procs: list[GpuProcess] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                procs.append(GpuProcess(pid=int(parts[0]), name=parts[2], used_mib=int(parts[1])))
            except ValueError:  # a non-integer pid/used_memory row (e.g. "[N/A]") — skip it
                continue
    return procs or None


def gpu_compute_apps() -> list[str] | None:
    """Best-effort list of `"pid <pid>: <name> (<MiB> MiB)"` for processes holding GPU memory, via
    `nvidia-smi`. Returns `None` if `nvidia-smi` is absent / errors / has no apps. NEVER raises — it runs
    inside the `VRAMExhausted` error path to name WHICH process holds the GPU (the actionable cause).
    A thin human-readable formatting of `gpu_processes` (the single source of truth for the probe)."""
    procs = gpu_processes()
    if procs is None:
        return None
    return [f"pid {p.pid}: {p.name} ({p.used_mib} MiB)" for p in procs] or None
