"""The CPU/GPU device knobs for the retrieval models (embedder + reranker).

`MEMEX_MODELS__{EMBEDDER,RERANKER}_DEVICE=cpu` lets the retrieval stack run on
CPU so the orchestrator can keep its full GPU KV cache on a single 12 GB card.
Two contracts: (1) the registry loaders place the model on the configured
device with the right dtype (bf16 on cuda, fp32 on cpu), and (2) the bootstrap
VRAM-fit estimate excludes a CPU-placed model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memex.models import registry as R


def test_retrieval_dtype_maps_device_to_dtype() -> None:
    # Asserts against the canonical dtype helpers (which own the torch import).
    assert R._retrieval_dtype("cuda") is R._bf16()
    assert R._retrieval_dtype("cpu") is R._float32()


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_load_embedder_honors_device(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    captured: dict[str, object] = {}

    class _FakeST:
        def __init__(self, model_id: str, **kw: object) -> None:
            captured["model_id"] = model_id
            captured.update(kw)

    # The loader does `from sentence_transformers import SentenceTransformer`
    # at call time, so patching the module attribute is enough.
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeST)
    R.ModelRegistry._load_embedder("emb-model", device)
    assert captured["device"] == device
    model_kwargs = captured["model_kwargs"]
    assert isinstance(model_kwargs, dict)
    expected = R._bf16() if device == "cuda" else R._float32()
    assert model_kwargs["torch_dtype"] is expected


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_load_reranker_honors_device(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    captured: dict[str, object] = {}

    class _FakeCE:
        def __init__(self, model_id: str, **kw: object) -> None:
            captured["model_id"] = model_id
            captured.update(kw)

    monkeypatch.setattr("sentence_transformers.CrossEncoder", _FakeCE)
    R.ModelRegistry._load_reranker("rr-model", device)
    assert captured["device"] == device
    model_kwargs = captured["model_kwargs"]
    assert isinstance(model_kwargs, dict)
    expected = R._bf16() if device == "cuda" else R._float32()
    assert model_kwargs["torch_dtype"] is expected


def _settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **env: str):  # type: ignore[no-untyped-def]  # returns MemexSettings; lazy import keeps torch out of the import path
    """Build MemexSettings with a tmp vault and a clean device-env slate.

    Pins `co_residence_mode=manual` so the explicit device knobs this module exercises are HONORED — the
    new `auto` default would override them from live free-VRAM (which is what the auto tests in
    `test_resources.py` cover instead)."""
    from memex.core.config import MemexSettings

    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("MEMEX_MODELS__CO_RESIDENCE_MODE", "manual")
    for var in ("MEMEX_MODELS__EMBEDDER_DEVICE", "MEMEX_MODELS__RERANKER_DEVICE"):
        monkeypatch.delenv(var, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return MemexSettings()  # type: ignore[call-arg]


def test_estimated_vram_excludes_cpu_placed_retrieval_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from memex.cli.bootstrap import _VRAM_GB, _estimated_vram_gb

    all_gpu = _estimated_vram_gb(_settings(monkeypatch, tmp_path))  # both cuda (default)
    rr_cpu = _estimated_vram_gb(
        _settings(monkeypatch, tmp_path, MEMEX_MODELS__RERANKER_DEVICE="cpu")
    )
    both_cpu = _estimated_vram_gb(
        _settings(
            monkeypatch,
            tmp_path,
            MEMEX_MODELS__RERANKER_DEVICE="cpu",
            MEMEX_MODELS__EMBEDDER_DEVICE="cpu",
        )
    )

    reranker_gb = _VRAM_GB[("reranker", "cross_encoder")]
    embedder_gb = _VRAM_GB[("embedder", None)]
    assert rr_cpu == pytest.approx(all_gpu - reranker_gb)
    assert both_cpu == pytest.approx(all_gpu - reranker_gb - embedder_gb)
    # The orchestrator + overhead never leave the GPU estimate.
    assert both_cpu < all_gpu


@pytest.mark.asyncio
async def test_registry_loads_reranker_on_mode_resolved_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `full` co-residence mode (ADR-0007) overrides the explicit device
    knobs: `_do_load` must load the reranker on CPU via `effective_devices`."""
    from memex.core.config import ModelSettings
    from memex.models.registry import ModelRegistry

    captured: dict[str, object] = {}

    def _fake_load_reranker(model_id: str, device: str = "cuda") -> object:
        captured["device"] = device
        return object()  # a stand-in for the loaded CrossEncoder

    monkeypatch.setattr(
        ModelRegistry, "_load_reranker", staticmethod(_fake_load_reranker)
    )
    # co_residence_mode="full" with the explicit knobs still at cuda — the mode
    # must win (→ reranker on cpu).
    reg = ModelRegistry(ModelSettings(co_residence_mode="full"))
    async with reg.use("reranker"):
        pass
    assert captured["device"] == "cpu"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("free_gb", "expected"),
    [(1.0, "cpu"), (5.0, "cuda"), (None, "cuda")],  # low→CPU, high→GPU, no-probe→GPU (optimistic)
)
async def test_registry_auto_mode_places_reranker_from_live_vram(
    monkeypatch: pytest.MonkeyPatch, free_gb: float | None, expected: str
) -> None:
    """`auto` (the default, ADR-0007 P4.4) reads LIVE free-VRAM at the reranker's load point: GPU when it
    clears the floor (2.0 GB), CPU under pressure, GPU optimistically when the probe is unavailable."""
    from memex.core.config import ModelSettings
    from memex.models.registry import ModelRegistry

    captured: dict[str, object] = {}

    def _fake_load_reranker(model_id: str, device: str = "cuda") -> object:
        captured["device"] = device
        return object()

    monkeypatch.setattr(ModelRegistry, "_load_reranker", staticmethod(_fake_load_reranker))
    # The live probe is the auto-mode driver — fake it at the registry's import site.
    monkeypatch.setattr("memex.core.vram.free_vram_gb", lambda: free_gb)

    reg = ModelRegistry(ModelSettings(co_residence_mode="auto"))
    async with reg.use("reranker"):
        pass
    assert captured["device"] == expected
