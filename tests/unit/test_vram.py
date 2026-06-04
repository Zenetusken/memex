"""Unit tests for the live-VRAM probe + the dynamic VLM-serve util formula (`core/vram.py`)."""

from __future__ import annotations

import pytest

from memex.core import vram


def test_fit_serve_util_free_gpu_uses_cap() -> None:
    # Wide-open card (orchestrator paused) → the static 0.80 cap (no regression).
    assert vram.fit_serve_util(12.0, 12.0) == pytest.approx(0.80)
    assert vram.fit_serve_util(11.4, 12.0) == pytest.approx(0.80)  # budget 10.8 → 0.90 capped to 0.80


def test_fit_serve_util_lowers_under_partial_occupancy() -> None:
    # A 2 GB co-resident consumer (free 10) → util lowered to fit (budget 9.4 / 12).
    u = vram.fit_serve_util(10.0, 12.0)
    assert u is not None and u == pytest.approx((10.0 - 0.6) / 12.0)
    assert u < 0.80


def test_fit_serve_util_returns_none_when_cant_fit() -> None:
    # The live case: free 8.6 → budget 8.0 < 8.5 floor → can't hold weights+KV → None (fail fast).
    assert vram.fit_serve_util(8.6, 12.0) is None
    assert vram.fit_serve_util(9.0, 12.0) is None  # budget 8.4 < 8.5
    # Just over the floor: budget 8.6 ≥ 8.5 → fits.
    assert vram.fit_serve_util(9.2, 12.0) == pytest.approx((9.2 - 0.6) / 12.0)


def test_fit_serve_util_none_probe_falls_back_to_cap() -> None:
    # Probe unavailable (off-GPU / torch-less) → the static cap, never None (caller keeps its default).
    assert vram.fit_serve_util(None, 12.0) == pytest.approx(0.80)
    assert vram.fit_serve_util(12.0, None) == pytest.approx(0.80)
    assert vram.fit_serve_util(None, None) == pytest.approx(0.80)


def test_fit_serve_util_custom_floor_and_cap() -> None:
    # Bounds are parameterized (e.g. a smaller VLM with a lower floor).
    assert vram.fit_serve_util(6.0, 12.0, floor_gb=4.0, safety_gb=0.5, cap=0.70) == pytest.approx(5.5 / 12.0)
    assert vram.fit_serve_util(4.0, 12.0, floor_gb=4.0, safety_gb=0.5) is None  # budget 3.5 < 4.0


def test_probes_return_float_or_none() -> None:
    # Env-dependent: on a CUDA box returns a positive float; torch-less returns None. Both are valid —
    # assert the contract (never crashes, never a negative/NaN).
    for probe in (vram.free_vram_gb, vram.total_vram_gb):
        v = probe()
        assert v is None or (isinstance(v, float) and v > 0)


def test_mem_get_info_none_on_torch_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the lazy probe to fail → the public probes degrade to None (the static-default path).
    monkeypatch.setattr(vram, "_mem_get_info", lambda: None)
    assert vram.free_vram_gb() is None
    assert vram.total_vram_gb() is None
