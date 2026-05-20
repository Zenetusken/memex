"""Local-process model registry — see IMPLEMENTATION-PLAN.md §2.2.

Owns the embedder, reranker, and VLM handles. The orchestrator
(Qwen3-8B) lives in vLLM out-of-process; we never load it here
(ADR-0001).

All three resident models are loaded lazily on first use, gated by
a per-model `asyncio.Lock`. A CUDA OOM circuit breaker (`name=
"model.oom"`) trips after three consecutive load failures.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import structlog
from pydantic import BaseModel

from memex.core.breakers import CircuitBreaker, CircuitBreakerOpen
from memex.core.config import ModelSettings
from memex.core.errors import InsufficientVRAMError, MemexError

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from transformers import PreTrainedModel, ProcessorMixin


class VLMHandle:
    """Bundled (model, processor) pair returned by `use('vlm')`."""

    def __init__(self, model: PreTrainedModel, processor: ProcessorMixin):
        self.model = model
        self.processor = processor

logger = structlog.get_logger(__name__)


def _is_oom(exc: BaseException) -> bool:
    """True if `exc` is a CUDA OOM. We avoid importing torch unless needed."""
    name = type(exc).__name__
    if name == "OutOfMemoryError":
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg

ModelName = Literal["embedder", "reranker", "vlm"]


class ModelHandle(BaseModel):
    name: ModelName
    model_id: str
    loaded_at: datetime
    inference_count: int = 0


class ModelNotConfigured(MemexError):
    """A model was requested but the registry isn't initialised."""


class ModelRegistry:
    """Process-singleton handle owner."""

    def __init__(self, settings: ModelSettings):
        self._settings = settings
        self._locks: dict[ModelName, asyncio.Lock] = {
            "embedder": asyncio.Lock(),
            "reranker": asyncio.Lock(),
            "vlm": asyncio.Lock(),
        }
        self._handles: dict[ModelName, ModelHandle] = {}
        self._models: dict[ModelName, Any] = {}
        # OOM circuit breaker — three consecutive CUDA OOMs on any model
        # trip the registry; subsequent loads fail fast with a clear
        # message until `memex doctor` (or a `reset_breaker()` call) clears
        # the state.
        self._oom_breaker: CircuitBreaker[None] = CircuitBreaker(
            name="model.oom",
            threshold=3,
            reset_after_s=300.0,
        )

    @asynccontextmanager
    async def use(self, name: ModelName) -> AsyncIterator[Any]:
        """Yield a ready-to-use model handle.

        First call loads the model; subsequent calls are no-ops. The
        embedder, reranker, and VLM all live in this process; the
        orchestrator (Qwen3-8B per ADR-0001) is *not* a registry
        resident — it lives in the out-of-process vLLM daemon.
        """
        async with self._locks[name]:
            if name not in self._models:
                await self._load(name)
            self._handles[name].inference_count += 1
            yield self._models[name]

    async def status(self) -> list[ModelHandle]:
        return list(self._handles.values())

    async def _load(self, name: ModelName) -> None:
        log = logger.bind(name=name)
        log.info("model.load.start")

        async def _do_load() -> None:
            if name == "embedder":
                self._models[name] = await asyncio.to_thread(
                    self._load_embedder, self._settings.embedder
                )
            elif name == "reranker":
                self._models[name] = await asyncio.to_thread(
                    self._load_reranker, self._settings.reranker
                )
            elif name == "vlm":
                self._models[name] = await asyncio.to_thread(
                    self._load_vlm, self._settings.vlm
                )
            else:
                raise ModelNotConfigured(
                    f"unknown model {name!r}", context={"name": name}
                )

        try:
            await self._oom_breaker.run(_do_load, is_failure=_is_oom)
        except CircuitBreakerOpen as e:
            self._cleanup_partial(name)
            raise InsufficientVRAMError(
                "VRAM OOM circuit breaker is open — refusing further loads. "
                "Free GPU memory and run `memex doctor` to reset.",
                context={"name": name, "breaker": str(e)},
            ) from e
        except Exception as e:
            self._cleanup_partial(name)
            if _is_oom(e):
                raise InsufficientVRAMError(
                    f"CUDA out of memory loading {name!r}",
                    context={"name": name, "underlying": str(e)},
                ) from e
            raise

        self._handles[name] = ModelHandle(
            name=name,
            model_id=getattr(self._settings, name, "unknown"),
            loaded_at=datetime.now(UTC),
        )
        log.info("model.load.done")

    def _cleanup_partial(self, name: ModelName) -> None:
        """Release a model that failed to fully load.

        `_load_*` is sync wrapped in `asyncio.to_thread`; an OOM raised
        from `from_pretrained` after partial weight materialisation
        leaves orphaned tensors holding VRAM. Drop the registry slot
        and ask CUDA to release any reserved-but-unowned blocks back
        to the driver. Idempotent — safe to call when nothing was
        ever assigned.
        """
        # The assignment in `_do_load` is `self._models[name] = ...`;
        # if `to_thread` raised before the assignment, the key doesn't
        # exist. If it raised after weights moved to GPU but before
        # `to_thread` returned (rare; only possible on driver-side
        # OOMs), the partial state is held by torch's caching
        # allocator and `empty_cache()` reclaims it.
        self._models.pop(name, None)
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            # No torch in this process (tests, headless ingest path).
            pass

    def reset_breaker(self) -> None:
        """For `memex doctor` and tests — clear the OOM trip state."""
        self._oom_breaker.reset()

    @staticmethod
    def _load_embedder(model_id: str) -> SentenceTransformer:
        """Load the dense embedder.

        ADR-0006: BF16 + explicit `device="cuda"`. EmbeddingGemma 300M's
        official model card states its activations DO NOT support FP16
        — must be FP32 or BF16. Memex picks BF16 globally on Ada.

        Imported lazily so the agent layer can be imported without
        torch installed (`[models]` extra brings it in).
        """
        import torch
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            model_id,
            device="cuda",
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

    @staticmethod
    def _load_reranker(model_id: str) -> CrossEncoder:
        """Load the cross-encoder reranker.

        ADR-0006: BF16 + explicit `device="cuda"`.
        """
        import torch
        from sentence_transformers import CrossEncoder

        return CrossEncoder(
            model_id,
            device="cuda",
            automodel_args={"torch_dtype": torch.bfloat16},
        )

    @staticmethod
    def _load_vlm(model_id: str) -> VLMHandle:
        """Load the vision-language model + its processor.

        Decisions baked in per ADR-0006:

        - `AutoModelForImageTextToText` is the 2026 idiom for VLMs;
          `AutoModelForCausalLM` does not expose the vision tower
          correctly for Qwen2.5/3-VL.
        - `attn_implementation="flash_attention_2"` — FA3 is unavailable
          on Ada (sm_89); SDPA is the fallback when flash-attn isn't
          installed (`[parse]` extra brings it on Linux).
        - `torch_dtype=torch.bfloat16` explicit — no implicit `"auto"`.
        - `device_map={"": "cuda:0"}` — deterministic single-GPU
          placement. `"auto"` silently CPU-offloads under VRAM pressure;
          we prefer the OOM circuit breaker to handle real overflow.
        - Qwen-VL processor `min/max_pixels` bound the visual-token
          budget — the rasteriser in `parse/vlm_backend.py` should
          stay below `max_pixels` to avoid an extra resize hop.
        - `trust_remote_code` dropped — Qwen-VL landed in transformers
          proper in 4.49+.
        """
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )
        model.eval()
        return VLMHandle(model=model, processor=processor)


_REGISTRY: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Return the process registry. Configured at startup."""
    if _REGISTRY is None:
        raise ModelNotConfigured(
            "ModelRegistry not initialised; call set_registry() from the "
            "entry point.",
            context={"fix": "cli.bootstrap.bootstrap() does this for you"},
        )
    return _REGISTRY


def set_registry(registry: ModelRegistry | None) -> None:
    """Install or detach the process registry. Tests pass None to detach."""
    global _REGISTRY
    _REGISTRY = registry
