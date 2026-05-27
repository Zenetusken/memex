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
    assert p.orchestrator_gpu_fraction == 0.60
    assert p.orchestrator_max_model_len == 6144


def test_full_mode_offloads_reranker_for_a_large_window() -> None:
    p = resolve_profile("full")
    # The defining trade: reranker → CPU frees GPU for a big context window.
    assert p.embedder_device == "cuda"
    assert p.reranker_device == "cpu"
    assert p.orchestrator_gpu_fraction == 0.80
    assert p.orchestrator_max_model_len == 24576


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


def test_all_modes_lists_curated_in_order_excluding_manual() -> None:
    modes = all_modes()
    assert [m.mode for m in modes] == ["fast", "full", "gpu_only"]
    assert all(isinstance(m, ResourceProfile) for m in modes)
    # Every curated profile prescribes an orchestrator posture.
    assert all(m.orchestrator_gpu_fraction is not None for m in modes)
