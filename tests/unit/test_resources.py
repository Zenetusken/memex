"""Unit tests for `core/resources.py` — the co-residence mode resolver (ADR-0007).

The resolver is the policy seam the dynamic VRAM manager will later replace; these
pin the curated 12 GB bundles + the manual-echo + `effective_devices` contract.
"""

from __future__ import annotations

import pytest

from memex.core.errors import ConfigurationError
from memex.core.resources import (
    ResourceProfile,
    all_modes,
    effective_devices,
    resolve_profile,
)


def test_fast_mode_is_gpu_retrieval_low_util() -> None:
    p = resolve_profile("fast")
    assert (p.embedder_device, p.reranker_device) == ("cuda", "cuda")
    # Calibrated for the unified Qwen3.5-4B orchestrator (fit-tested at 0.62/8192).
    assert p.orchestrator_gpu_fraction == 0.62
    assert p.orchestrator_max_model_len == 8192


def test_full_mode_offloads_reranker_for_a_large_window() -> None:
    p = resolve_profile("full")
    # The defining trade: reranker → CPU frees GPU for a big context window.
    assert p.embedder_device == "cuda"
    assert p.reranker_device == "cpu"
    assert p.orchestrator_gpu_fraction == 0.80
    assert p.orchestrator_max_model_len == 24576


def test_retrieval_top_k_scales_with_the_window() -> None:
    """How a mode LEVERAGES its window: full grounds against MANY more reranked
    chunks than fast (its larger orchestrator window holds them). fast/gpu_only/
    manual keep the historical 5 — so only an explicit `full` switch deepens
    retrieval (the common manual path is unchanged)."""
    # 5 is a MEASURED default: the k=8 flip was full-ladder NO-GO'd 2026-06-10
    # (net −3 ANS deterministic; audit-15 M5) — do not re-raise without a new ladder.
    assert resolve_profile("fast").retrieval_top_k == 5
    assert resolve_profile("gpu_only").retrieval_top_k == 5
    assert resolve_profile("manual").retrieval_top_k == 5
    assert resolve_profile("full").retrieval_top_k == 18
    # full's deeper retrieval must fit its window: 18 chunks × 1800-char truncate
    # ≈ 8.1k tokens + scaffold + output, well under 24,576.
    full = resolve_profile("full")
    assert full.retrieval_top_k * 1800 // 4 < full.orchestrator_max_model_len


def test_gpu_only_mode() -> None:
    p = resolve_profile("gpu_only")
    assert (p.embedder_device, p.reranker_device) == ("cuda", "cuda")
    assert p.orchestrator_gpu_fraction == 0.72


def test_manual_echoes_explicit_devices_and_leaves_orchestrator_unset() -> None:
    p = resolve_profile("manual", embedder_device="cuda", reranker_device="cpu")
    assert (p.embedder_device, p.reranker_device) == ("cuda", "cpu")
    # The app does not prescribe the orchestrator launch in manual mode.
    assert p.orchestrator_gpu_fraction is None
    assert p.orchestrator_max_model_len is None


def test_effective_devices_curated_overrides_explicit_manual_honors() -> None:
    # Curated modes OVERRIDE the explicit device knobs.
    assert effective_devices("full", "cuda", "cuda") == ("cuda", "cpu")
    assert effective_devices("fast", "cpu", "cpu") == ("cuda", "cuda")
    # Manual honors them verbatim.
    assert effective_devices("manual", "cuda", "cpu") == ("cuda", "cpu")
    assert effective_devices("manual", "cpu", "cpu") == ("cpu", "cpu")


def test_total_vram_gb_is_accepted_but_inert_for_now() -> None:
    # The dynamic-manager seam: accepted, doesn't change the curated bundle yet.
    a = resolve_profile("full")
    b = resolve_profile("full", total_vram_gb=24.0)
    assert a.model_dump() == b.model_dump()


def test_unknown_mode_raises() -> None:
    with pytest.raises(ConfigurationError):
        resolve_profile("bogus")  # type: ignore[arg-type]  # deliberately invalid


def test_all_modes_leads_with_auto_then_curated_excluding_manual() -> None:
    modes = all_modes()
    assert [m.mode for m in modes] == ["auto", "fast", "full", "gpu_only"]
    assert all(isinstance(m, ResourceProfile) for m in modes)
    # Every selectable profile prescribes an orchestrator posture (auto mirrors fast's 0.62/8192).
    assert all(m.orchestrator_gpu_fraction is not None for m in modes)


def test_all_modes_auto_row_reflects_live_free_vram() -> None:
    # The auto row's reranker placement tracks the supplied live free-VRAM (GPU when it fits, else CPU).
    assert all_modes(free_vram_gb=5.0)[0].reranker_device == "cuda"
    assert all_modes(free_vram_gb=1.0)[0].reranker_device == "cpu"


# ── auto mode (the default; live-VRAM-driven reranker placement) ─────────────────────────────────────
def test_auto_keeps_reranker_on_gpu_when_free_clears_the_floor() -> None:
    # Free at the decision point ≥ RERANKER_GPU_FLOOR_GB (2.0) → GPU rerank (the optimal default).
    p = resolve_profile("auto", free_vram_gb=3.77)  # the measured live operating point
    assert (p.embedder_device, p.reranker_device) == ("cuda", "cuda")
    # Posture mirrors fast (no orchestrator change vs today's default).
    assert (p.orchestrator_gpu_fraction, p.orchestrator_max_model_len, p.retrieval_top_k) == (0.62, 8192, 5)


def test_auto_demotes_reranker_to_cpu_under_vram_pressure() -> None:
    # Free below the floor (a 2nd GPU consumer / desktop spike) → CPU reranker (graceful, never OOM).
    p = resolve_profile("auto", free_vram_gb=1.5)
    assert (p.embedder_device, p.reranker_device) == ("cuda", "cpu")


def test_auto_optimistic_gpu_when_probe_unavailable() -> None:
    # No probe (off-GPU / torch-less) → GPU reranker (the pre-dynamic-manager behaviour); embedder GPU.
    p = resolve_profile("auto", free_vram_gb=None)
    assert (p.embedder_device, p.reranker_device) == ("cuda", "cuda")


def test_effective_devices_auto_reads_free_vram_and_ignores_explicit_knobs() -> None:
    # auto OVERRIDES the explicit device fields (like the curated modes), driven by free_vram_gb.
    assert effective_devices("auto", "cpu", "cpu", free_vram_gb=4.0) == ("cuda", "cuda")
    assert effective_devices("auto", "cuda", "cuda", free_vram_gb=1.0) == ("cuda", "cpu")
    assert effective_devices("auto", "cpu", "cpu", free_vram_gb=None) == ("cuda", "cuda")  # optimistic
