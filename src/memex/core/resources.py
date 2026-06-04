"""Co-residence resource modes — the policy seam for the VRAM tradeoff.

On a single GPU the answering stack (orchestrator vLLM + embedder + reranker)
competes for VRAM. A *mode* is a named bundle of the one tradeoff that
governs it: ~3 GB of swing VRAM goes EITHER to the orchestrator's KV cache /
context window (`--max-model-len`) OR to keeping the reranker on the GPU.

This module is the **precursor to a dynamic VRAM manager**: a single pure
resolver (`resolve_profile`) maps a mode to a concrete `ResourceProfile`.
Every consumer reads the resolved profile — the registry (device placement),
`cli.bootstrap` (the VRAM estimate), the daemon (orchestrator util +
max-model-len), the CLI, and the webui (display). The future dynamic manager
replaces this resolver's BODY (compute the profile from live free-VRAM /
load) without changing a single caller.

Pure: imports only stdlib + pydantic + `core/errors`. Callers pass settings
PRIMITIVES (the mode + the manual device fields), never a `MemexSettings`
object, so there is no import cycle with `core/config` (config may import
this; this never imports config).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from memex.core.errors import ConfigurationError

CoResidenceMode = Literal["auto", "fast", "full", "gpu_only", "manual"]
Device = Literal["cuda", "cpu"]

# The `auto` mode (the default) keeps the reranker on the GPU while the live FREE VRAM at its load point
# clears this floor, else demotes it to the CPU (graceful fallback instead of OOM). CALIBRATED EMPIRICALLY
# (2026-06-04) by loading the reranker on GPU co-resident with the 4B (which reserves ~7.4 GB at 0.62 util)
# and running a rerank: the reranker's REAL footprint is ~1.49 GB (load ~1.06 + forward ~0.43), and free at
# the decision point measured ~3.77 GB → GPU fits with ~2.28 GB slack. Floor = footprint + 0.5 GB buffer =
# 2.0 GB, leaving ~1.77 GB of ambient-contention tolerance before the fallback to CPU. (Reranker order is
# byte-identical CPU vs GPU — see the reranker-GPU-AB finding — so this is a latency knob, not correctness.)
RERANKER_GPU_FLOOR_GB = 2.0

# The reference rig these curated bundles are calibrated for (RTX 4070 12 GB).
# `resolve_profile(total_vram_gb=...)` accepts a real card size as the seam the
# dynamic manager will compute from; the curated numbers below are the 12 GB
# operating points validated in the co-residence work.
REFERENCE_VRAM_GB = 12.0

# The curated modes, in display order. `manual` is intentionally absent — it
# has no fixed profile (it echoes the user's explicit device knobs).
_CURATED_ORDER: tuple[CoResidenceMode, ...] = ("fast", "full", "gpu_only")

# The historical RAG default (matches the prior hard-coded `MEMEX_RERANK_TOP_K`
# default in the answer node) — `manual` mode echoes it, so the common path is
# byte-unchanged; only the curated `full` mode raises retrieval depth.
_DEFAULT_TOP_K = 5


class ResourceProfile(BaseModel):
    """The concrete resource posture a mode resolves to.

    `orchestrator_gpu_fraction` / `orchestrator_max_model_len` are the values
    the daemon launches vLLM with (`MEMEX_VLLM_GPU_FRACTION` /
    `MEMEX_VLLM_MAX_MODEL_LEN`); they are `None` for `manual` — the app does
    not prescribe the orchestrator's launch there. The string fields are
    human-facing (CLI table + webui panel).
    """

    mode: CoResidenceMode
    label: str
    summary: str
    embedder_device: Device
    reranker_device: Device
    orchestrator_gpu_fraction: float | None
    orchestrator_max_model_len: int | None
    # How many reranked chunks the answering agent grounds against. This is the
    # concrete way a mode LEVERAGES its orchestrator window: the fast 8,192 window
    # holds ~5 truncated chunks, while full's 24,576 grounds against many more
    # (deeper retrieval → more evidence per answer, the refusal gate unchanged).
    # `MEMEX_RERANK_TOP_K`, when set, overrides this (operator escape hatch).
    retrieval_top_k: int
    expected_latency: str
    context_window: str


def _curated(mode: CoResidenceMode) -> ResourceProfile:
    """The curated 12 GB-reference profile for a non-manual mode."""
    if mode == "fast":
        return ResourceProfile(
            mode="fast",
            label="Fast",
            summary=(
                "Low-latency top-k retrieval answers. Retrieval stays on the GPU; the "
                "orchestrator runs at a lower util, so its context window is tighter."
            ),
            embedder_device="cuda",
            reranker_device="cuda",
            # Calibrated for the unified Qwen3.5-4B orchestrator (compressed-tensors,
            # auto KV — it rejects fp8_e5m2). The 4B's validated window is 8,192 (vs
            # the 8B's 6,144 answer-prompt floor); at 0.62 util the daemon footprint
            # is ~7.0 GB leaving ~4.3 GB for the GPU embedder+reranker (fit-tested,
            # desktop-peak slack). The 8B kill-switch fallback runs fine at this posture.
            orchestrator_gpu_fraction=0.62,
            orchestrator_max_model_len=8192,
            retrieval_top_k=5,
            expected_latency="~14 s / answer",
            context_window="8,192 tokens · top-5 chunks",
        )
    if mode == "full":
        return ResourceProfile(
            mode="full",
            label="Full context",
            summary=(
                "Deeper retrieval: the larger orchestrator window lets an answer ground "
                "against many more reranked chunks (~18 vs 5) than the fast window holds — "
                "more evidence per answer, the no-hallucination gate unchanged. The reranker "
                "moves to the CPU to free that VRAM, at the cost of slower (CPU) reranking."
            ),
            embedder_device="cuda",
            reranker_device="cpu",
            orchestrator_gpu_fraction=0.80,
            orchestrator_max_model_len=24576,
            retrieval_top_k=18,
            expected_latency="~30-40 s / answer",
            context_window="24,576 tokens · top-18 chunks",
        )
    if mode == "gpu_only":
        return ResourceProfile(
            mode="gpu_only",
            label="GPU-only",
            summary=(
                "Everything on the GPU at full util — for cards with headroom (>12 GB) or "
                "when the orchestrator is not co-resident with answering."
            ),
            embedder_device="cuda",
            reranker_device="cuda",
            orchestrator_gpu_fraction=0.72,
            orchestrator_max_model_len=8192,
            retrieval_top_k=5,
            expected_latency="~14 s / answer",
            context_window="8,192 tokens · top-5 chunks",
        )
    raise ConfigurationError(
        f"unknown co-residence mode {mode!r}",
        context={"mode": mode, "known": [*_CURATED_ORDER, "manual"]},
    )


def _auto_reranker_device(free_vram_gb: float | None) -> Device:
    """The `auto` mode's reranker placement: GPU while the live free VRAM clears `RERANKER_GPU_FLOOR_GB`,
    else CPU. `None` (probe unavailable / off-GPU) → GPU (optimistic — the pre-dynamic-manager behaviour).
    The embedder always stays on GPU (small + keeps the query embedding bf16 in lockstep with the index)."""
    if free_vram_gb is None:
        return "cuda"
    return "cuda" if free_vram_gb >= RERANKER_GPU_FLOOR_GB else "cpu"


def resolve_profile(
    mode: CoResidenceMode,
    *,
    total_vram_gb: float | None = None,
    free_vram_gb: float | None = None,
    embedder_device: Device = "cuda",
    reranker_device: Device = "cuda",
) -> ResourceProfile:
    """Resolve a mode to a concrete `ResourceProfile`.

    `auto` (the default) reads the LIVE `free_vram_gb` and adapts the reranker placement (GPU when it
    fits, else CPU), at today's `fast` orchestrator posture. `manual` echoes the explicit per-model device
    fields (the orchestrator util / max-model-len are whatever the user launched → `None`). The other
    curated modes are calibrated for the `REFERENCE_VRAM_GB` tier; `total_vram_gb` is accepted for symmetry
    (reserved for a fuller dynamic manager) and does not alter the curated operating points today.
    """
    _ = total_vram_gb  # reserved: the dynamic-VRAM-manager seam (free_vram_gb is the live driver)
    if mode == "auto":
        rr = _auto_reranker_device(free_vram_gb)
        return ResourceProfile(
            mode="auto",
            label="Auto",
            summary=(
                "Auto-tunes to your live GPU memory: retrieval runs on the GPU when it fits, and the "
                "reranker falls back to the CPU under VRAM pressure (slower, but never an out-of-memory "
                "failure). The orchestrator runs at the low-latency posture. No manual configuration."
            ),
            # Embedder always GPU; reranker adapts to live free VRAM. Orchestrator mirrors `fast`.
            embedder_device="cuda",
            reranker_device=rr,
            orchestrator_gpu_fraction=0.62,
            orchestrator_max_model_len=8192,
            retrieval_top_k=5,
            expected_latency="~14 s (GPU rerank) · ~34 s (CPU fallback)",
            context_window="8,192 tokens · top-5 chunks",
        )
    if mode == "manual":
        return ResourceProfile(
            mode="manual",
            label="Manual",
            summary=(
                "Honors the explicit embedder_device / reranker_device knobs; the "
                "orchestrator's util and context window are whatever it was launched with."
            ),
            embedder_device=embedder_device,
            reranker_device=reranker_device,
            orchestrator_gpu_fraction=None,
            orchestrator_max_model_len=None,
            retrieval_top_k=_DEFAULT_TOP_K,
            expected_latency="(depends on placement)",
            context_window="(as launched)",
        )
    return _curated(mode)


def effective_devices(
    mode: CoResidenceMode,
    embedder_device: Device,
    reranker_device: Device,
    *,
    free_vram_gb: float | None = None,
) -> tuple[Device, Device]:
    """The `(embedder, reranker)` device placement a mode dictates.

    `manual` uses the explicit fields; `auto` reads the LIVE `free_vram_gb` (embedder GPU, reranker
    GPU-if-it-fits-else-CPU); every other (curated) mode OVERRIDES with its fixed placement. The registry
    loads each retrieval model on the returned device; for `auto` it passes a fresh probe so the reranker
    decision sees the embedder + orchestrator already resident.
    """
    if mode == "manual":
        return embedder_device, reranker_device
    if mode == "auto":
        return "cuda", _auto_reranker_device(free_vram_gb)
    profile = _curated(mode)
    return profile.embedder_device, profile.reranker_device


def all_modes() -> list[ResourceProfile]:
    """Every curated mode's profile, in display order (for the webui compare
    panel + `memex mode show`). Excludes `manual` (no fixed profile). `auto` is
    surfaced separately once it's wired + selectable (V4)."""
    return [_curated(m) for m in _CURATED_ORDER]
