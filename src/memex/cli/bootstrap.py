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
from typing import Any, cast

import structlog

from memex.core.bus import EventBus, set_bus
from memex.core.config import MemexSettings, set_settings
from memex.core.errors import InsufficientVRAMError
from memex.core.resources import effective_devices
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
    # torch.cuda.set_per_process_memory_fraction has an incomplete stub
    # in the shipped torch wheel (partially-unknown signature); the arg
    # is documented as a float fraction.
    torch.cuda.set_per_process_memory_fraction(  # type: ignore[no-untyped-call]  # torch stub gap
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
    # Qwen3.5-4B compressed-tensors W4A16: ~6.3 GB weights+overhead on a
    # 4070 — NOT half the 8B-AWQ. The dense W4A16 saving is offset because
    # the vision tower / Gated-DeltaNet linear-attn / sparse-MoE router /
    # MTP head stay fp16 (vLLM #37080). Unification buys a simpler lifecycle
    # + a reasoning-capable base, NOT VRAM — do not budget freed VRAM.
    ("orchestrator", "compressed_tensors"): 6.3,
    ("embedder", None): 0.6,
    # bge-reranker-v2-m3 in BF16: ~2 GB resident (568 M params × 2 B
    # plus the cross-encoder head). Qwen3-Reranker-0.6B in BF16:
    # ~1.8 GB weights + ~0.3 GB forward-pass workspace at batch=1
    # ≈ 2.1 GB live. The P2.1 verification on RTX 4070 12 GB confirmed
    # both backends sit at the ~2 GB tier — the swap is a quality
    # play (eval-gated, P0), NOT a memory play. The headline 0.6 B vs
    # 568 M parameter delta does not translate to a meaningful VRAM
    # delta once autoregressive forward-pass activations are counted.
    ("reranker", "cross_encoder"): 2.0,
    ("reranker", "qwen3"): 2.1,
    ("vlm", "awq_int4"): 7.0,
    ("vlm", "bf16"): 16.0,
    # Chart-OCR slot: covers Nemotron-Parse-v1.2 (the new default
    # since the 2026-05-23 P3.3-c shootout) at ~3 GB live in BF16,
    # plus the smaller alternatives (DePlot ~2.3 GB, UniChart ~0.8
    # GB, OneChart ~0.6 GB). vLLM is paused (~8.5 GB freed) during
    # chart-OCR, so the budget the manager needs is ~3 GB on top of
    # embedder + reranker. Counted in the estimate only when
    # `disable_chart_ocr=False` (default since 2026-05-23).
    ("chart_ocr", "bf16"): 3.0,
}
# KV cache + processor + activations headroom. Empirical from the GUIDELINES
# Part III VRAM table.
_OVERHEAD_GB = 2.5
# Conservative fallback for an orchestrator quant tier not in `_VRAM_GB`
# (a future Literal addition without a matching table entry) — over-estimate
# so the fit warning errs toward caution rather than crashing.
_DEFAULT_ORCHESTRATOR_GB = 6.5


def _estimated_vram_gb(settings: MemexSettings) -> float:
    """The co-residence GPU VRAM estimate (GB) for the answering stack:
    orchestrator + the GPU-placed retrieval models + overhead, plus the
    in-process VLM and chart-OCR when those apply.

    Pure (no torch) so it's unit-testable. A retrieval model placed on the CPU
    (`embedder_device`/`reranker_device == "cpu"`) is EXCLUDED — it holds RAM,
    not GPU VRAM — mirroring how the vLLM-served VLM is excluded.
    """
    # The co-residence mode (ADR-0007) resolves the effective device of each
    # retrieval model; a CPU-placed model holds RAM, not GPU VRAM.
    emb_device, rr_device = effective_devices(
        settings.models.co_residence_mode,
        settings.models.embedder_device,
        settings.models.reranker_device,
    )
    # `.get` (not a bare subscript) so a future orchestrator quant tier added
    # to the config Literal degrades to a conservative estimate + the
    # vram.budget.tight warning, never a KeyError that bricks daemon startup.
    orch_gb = _VRAM_GB.get(
        ("orchestrator", settings.models.orchestrator_quantization),
        _DEFAULT_ORCHESTRATOR_GB,
    )
    estimated = orch_gb + _OVERHEAD_GB
    if emb_device == "cuda":
        estimated += _VRAM_GB[("embedder", None)]
    if rr_device == "cuda":
        estimated += _VRAM_GB[("reranker", settings.models.reranker_backend)]
    # Only count the VLM if it's loaded IN-PROCESS. With vlm_serving="vllm" it
    # runs as a separate short-lived vLLM process on the GPU freed by
    # pause_vllm_for_gpu — never co-resident with answering.
    if (not settings.parse.disable_vlm) and settings.models.vlm_serving == "transformers":
        estimated += _VRAM_GB[("vlm", settings.models.vlm_quantization)]
    # chart-OCR loads at parse time (vLLM paused), so the orchestrator's ~5 GB
    # doesn't compete for the same window; the parse-time peak is what governs
    # whether the parse pass OOMs.
    if not settings.parse.disable_chart_ocr:
        estimated += _VRAM_GB[("chart_ocr", "bf16")]
    return estimated


def _device_total_memory_bytes() -> int:
    """Total VRAM of CUDA device 0, in bytes.

    Isolated so the torch-stub Unknowns (`get_device_properties` has a
    partially-unknown signature; its `.total_memory` member is typed
    Unknown in the shipped wheel) don't leak into `_verify_vram_fit`'s
    arithmetic. `total_memory` is documented as int bytes.
    """
    import torch

    # `torch.cuda.get_device_properties` has a partially-unknown
    # signature in the shipped torch stub. Route the call through an
    # explicitly-`Any` reference so the Unknown is contained to this
    # function (explicit `Any` is permitted under strict; inferred
    # Unknown is not). `total_memory` is documented as int bytes.
    cuda: Any = torch.cuda
    props: object = cuda.get_device_properties(0)
    return cast(int, getattr(props, "total_memory"))  # noqa: B009


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
    # torch.cuda.get_device_properties has a partially-unknown signature
    # in the shipped torch stub, and its `.total_memory` member is typed
    # Unknown. It is documented to return int bytes; read it through a
    # narrow helper so the Unknown is contained and the GB arithmetic
    # below is fully typed.
    total_memory_bytes = _device_total_memory_bytes()
    total_gb = total_memory_bytes / (1024**3)
    budget_gb = total_gb * settings.hardware.gpu_memory_fraction
    gpu_name = torch.cuda.get_device_name(0)

    estimated = _estimated_vram_gb(settings)
    # Recomputed only for the log fields below (the estimate itself lives in
    # the pure helper). The mode (ADR-0007) resolves the effective devices.
    vlm_in_process = (not settings.parse.disable_vlm) and (
        settings.models.vlm_serving == "transformers"
    )
    emb_device, rr_device = effective_devices(
        settings.models.co_residence_mode,
        settings.models.embedder_device,
        settings.models.reranker_device,
    )

    if estimated > budget_gb:
        log.warning(
            "vram.budget.tight",
            estimated_gb=round(estimated, 1),
            budget_gb=round(budget_gb, 1),
            total_gb=round(total_gb, 1),
            gpu=gpu_name,
            vlm_counted=vlm_in_process,
            co_residence_mode=settings.models.co_residence_mode,
            embedder_device=emb_device,
            reranker_device=rr_device,
            fix=(
                "lower hardware.gpu_memory_fraction, switch to a lighter co-residence "
                "mode (MEMEX_MODELS__CO_RESIDENCE_MODE=full puts the reranker on CPU), "
                "switch to a smaller VLM variant (vlm_quantization=awq_int4), or reduce "
                "the orchestrator quant tier"
            ),
        )
    else:
        log.info(
            "vram.budget.ok",
            estimated_gb=round(estimated, 1),
            budget_gb=round(budget_gb, 1),
            total_gb=round(total_gb, 1),
            gpu=gpu_name,
            vlm_counted=vlm_in_process,
            co_residence_mode=settings.models.co_residence_mode,
            embedder_device=emb_device,
            reranker_device=rr_device,
        )


# Below this much free VRAM at boot WITH an external holder, surface the contention (the auto runtime
# still adapts, but a squeezed start should be visible). Above it = a normal orchestrator-resident card.
_PREFLIGHT_LOW_FREE_GB = 3.0


def _preflight_gpu(settings: MemexSettings) -> None:
    """Boot-time GPU pre-flight (dynamic VRAM manager): report total/free VRAM + the processes holding the
    GPU, and WARN if free VRAM is low so a degraded start is VISIBLE. Report-and-warn only — NOT auto-kill
    (a GPU holder at boot may be a concurrent legit parse, not a dead orphan); the `auto` runtime adapts
    (CPU reranker fallback / dynamic VLM-serve util / VRAMExhausted naming the holder). Best-effort +
    non-fatal — a probe failure leaves startup unchanged."""
    from memex.core import vram

    free = vram.free_vram_gb()
    total = vram.total_vram_gb()
    if free is None or total is None:
        return  # off-GPU / torch-less — nothing to pre-flight
    holders = vram.gpu_compute_apps()
    log = structlog.get_logger(__name__).bind(component="bootstrap.gpu")
    log.info("gpu.preflight", total_gb=round(total, 2), free_gb=round(free, 2), holders=holders or [])
    if free < _PREFLIGHT_LOW_FREE_GB and holders:
        log.warning(
            "gpu.preflight.low_free",
            free_gb=round(free, 2),
            holders=holders,
            note="auto mode places retrieval on CPU under pressure; free a stray GPU process for full "
            "speed (a parse VLM-serve fails fast naming the holder if it still can't fit)",
        )


def bootstrap() -> MemexSettings:
    """Read config, configure singletons, return the loaded settings."""
    settings = MemexSettings()  # type: ignore[call-arg]  # vault_path comes from TOML/env
    configure_logging(json_output=settings.observability.log_json)
    log = structlog.get_logger(__name__)
    log.info("bootstrap", vault_path=str(settings.vault_path))

    _configure_cuda(settings)
    _verify_vram_fit(settings)
    _preflight_gpu(settings)

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
