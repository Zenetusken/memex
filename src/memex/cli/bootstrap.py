"""CLI bootstrap — wire logging, tracing, CUDA, model client, registry, settings.

Every CLI command starts by calling `bootstrap()`. It loads
`MemexSettings`, configures the observability stack, configures the
CUDA runtime (per ADR-0006), points the OpenAI client at vLLM,
constructs the model registry, and installs the process singletons.
Idempotent: calling twice is safe (the second call replays the same
configuration).
"""

from __future__ import annotations

import os

import structlog

from memex.core.bus import EventBus, set_bus
from memex.core.config import MemexSettings, set_settings
from memex.core.errors import InsufficientVRAMError
from memex.models.client import configure_client
from memex.models.registry import ModelRegistry, set_registry
from memex.observability.logging import configure_logging
from memex.observability.tracing import configure_tracing

# `PYTORCH_CUDA_ALLOC_CONF` must be set *before* the CUDA caching allocator
# initialises (which happens on the first CUDA call). Setting it as a
# module-level side effect on import is the safest way to guarantee that —
# if anything down the import chain triggers a CUDA op before `bootstrap()`
# runs, the allocator will already see this config.
#
# `expandable_segments:True` recovers fragmented VRAM under VLM swap-in /
# swap-out workloads (PyTorch #165419). `garbage_collection_threshold:0.8`
# starts proactive GC when reserved-but-unused VRAM crosses 80% of the
# per-process budget.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)


def _configure_cuda(settings: MemexSettings) -> None:
    """Set torch/CUDA flags and verify GPU availability. ADR-0006.

    Per ADR-0001 Memex requires a CUDA-capable NVIDIA GPU; this is the
    fail-fast point. Per ADR-0006 we enable TF32 on matmuls, set
    cudnn's algorithm-cache heuristic on (safe for our stable batch
    shapes), and pin the per-process memory fraction so `gpu_memory_fraction`
    is no longer dead config.
    """
    import torch

    if not torch.cuda.is_available():
        raise InsufficientVRAMError(
            "CUDA is not available; Memex requires an NVIDIA GPU per ADR-0001.",
            context={
                "fix": (
                    "install NVIDIA driver R570+, CUDA 12.9 toolkit, "
                    "and run `uv sync --extra models` to get the cu129 torch wheel"
                ),
            },
        )

    torch.set_float32_matmul_precision("high")  # TF32 on Ada matmuls
    torch.backends.cudnn.benchmark = True  # safe for stable batch shapes
    torch.cuda.set_per_process_memory_fraction(
        settings.hardware.gpu_memory_fraction
    )


# Approximate steady-state VRAM cost per resident model, in GB. The
# orchestrator runs out-of-process in vLLM (ADR-0001), so it does NOT
# count toward in-process resident memory — but it does count toward the
# total GPU budget, so `_verify_vram_fit` adds it explicitly.
_VRAM_GB: dict[tuple[str, str | None], float] = {
    ("orchestrator", "Q4_K_M"): 5.0,
    ("orchestrator", "Q5_K_M"): 6.0,
    ("orchestrator", "Q8_0"): 8.5,
    ("orchestrator", "AWQ"): 5.0,
    ("orchestrator", "GPTQ"): 5.0,
    ("embedder", None): 0.6,
    # bge-reranker-v2-m3 in BF16 is ~2 GB resident (568 M params × 2 B
    # plus the cross-encoder head). The 0.6 GB figure was a smaller
    # reranker; the load test showed this is the number that hits the
    # budget. Qwen3-Reranker-0.6B in BF16 is ~0.7 GB (300 M params × 2 B
    # + embedding tables — P2.1 measurement).
    ("reranker", "cross_encoder"): 2.0,
    ("reranker", "qwen3"): 0.7,
    ("vlm", "awq_int4"): 7.0,
    ("vlm", "bf16"): 16.0,
}
# KV cache + processor + activations headroom. Empirical from the GUIDELINES
# Part III VRAM table.
_OVERHEAD_GB = 2.5


def _verify_vram_fit(settings: MemexSettings) -> None:
    """Compare the configured model mix against the local GPU's VRAM and
    warn if the budget is tight. ADR-0001 promised this; ADR-0006 wires
    it. Does not raise — the OOM circuit breaker is the last line of
    defence, and this is a heads-up at startup.

    `disable_vlm=True` skips the VLM in the estimate so the warning
    reflects what will actually be loaded. The reranker estimate
    uses the BF16 footprint of bge-reranker-v2-m3 (≈ 2 GB), not the
    0.6 GB sentence-transformer default — the audit's load test
    showed the BF16 number is what hits the budget.
    """
    import torch

    log = structlog.get_logger(__name__)
    total_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
    )
    budget_gb = total_gb * settings.hardware.gpu_memory_fraction

    estimated = (
        _VRAM_GB[("orchestrator", settings.models.orchestrator_quantization)]
        + _VRAM_GB[("embedder", None)]
        + _VRAM_GB[("reranker", settings.models.reranker_backend)]
        + _OVERHEAD_GB
    )
    # Only count the VLM if it's actually going to be loaded.
    if not settings.parse.disable_vlm:
        estimated += _VRAM_GB[("vlm", settings.models.vlm_quantization)]

    if estimated > budget_gb:
        log.warning(
            "vram.budget.tight",
            estimated_gb=round(estimated, 1),
            budget_gb=round(budget_gb, 1),
            total_gb=round(total_gb, 1),
            gpu=torch.cuda.get_device_name(0),
            vlm_counted=not settings.parse.disable_vlm,
            fix=(
                "lower hardware.gpu_memory_fraction, switch to a smaller "
                "VLM variant (vlm_quantization=awq_int4), or reduce the "
                "orchestrator quant tier"
            ),
        )
    else:
        log.info(
            "vram.budget.ok",
            estimated_gb=round(estimated, 1),
            budget_gb=round(budget_gb, 1),
            total_gb=round(total_gb, 1),
            gpu=torch.cuda.get_device_name(0),
            vlm_counted=not settings.parse.disable_vlm,
        )


def bootstrap() -> MemexSettings:
    """Read config, configure singletons, return the loaded settings."""
    settings = MemexSettings()  # type: ignore[call-arg]  # vault_path comes from TOML/env
    configure_logging(json_output=settings.observability.log_json)
    log = structlog.get_logger(__name__)
    log.info("bootstrap", vault_path=str(settings.vault_path))

    _configure_cuda(settings)
    _verify_vram_fit(settings)

    configure_tracing(settings.observability)
    configure_client(settings.inference)
    set_registry(ModelRegistry(settings.models))

    # In-process event bus. Persistence under the vault's derived state
    # (ADR-0003) — disposable, regenerable, never the source of truth.
    bus = EventBus(db_path=settings.vault_path / ".memex" / "events.sqlite")
    bus.start()
    set_bus(bus)

    set_settings(settings)
    return settings
