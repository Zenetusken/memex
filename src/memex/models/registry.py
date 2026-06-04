# pyright: reportConstantRedefinition=false
# `_REGISTRY` is an uppercase module-level singleton intentionally
# rebound by `set_registry()` (bootstrap path) and reset by
# `reset_registry()` (teardown / test cleanup).

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
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import structlog
from pydantic import BaseModel

from memex.core import vram
from memex.core.breakers import CircuitBreaker, CircuitBreakerOpen
from memex.core.config import ModelSettings
from memex.core.errors import InsufficientVRAMError, MemexError
from memex.core.resources import effective_devices

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizerBase,
        ProcessorMixin,
    )


def _bf16() -> Any:
    """Return `torch.bfloat16` (the global dtype across the stack on Ada,
    per ADR-0006).

    torch re-exports its dtype singletons from the private `_C` extension
    via a star-import, so pyright's `reportPrivateImportUsage` flags
    `torch.bfloat16` even though it's documented public API. Funnelling
    every reference through this one helper keeps the suppression to a
    single site. The return is `Any` because the dtype only ever flows
    into transformers' / sentence-transformers' loosely-typed
    `torch_dtype=` / `model_kwargs=` kwargs.
    """
    import torch

    return torch.bfloat16  # type: ignore[reportPrivateImportUsage]  # torch star-exports dtypes from _C; stub omits them from __all__


def _float32() -> Any:
    """Return `torch.float32` — the dtype for a CPU-placed retrieval model.

    bf16 is the GPU dtype (ADR-0006), but on CPU bf16 kernels are slow and
    incomplete; fp32 is the correct, well-supported CPU dtype. Used when
    `embedder_device`/`reranker_device` is `"cpu"`. Same `Any`/private-import
    rationale as `_bf16`.
    """
    import torch

    return torch.float32  # type: ignore[reportPrivateImportUsage]  # torch star-exports dtypes from _C; stub omits them from __all__


def _retrieval_dtype(device: str) -> Any:
    """The load dtype for a retrieval model on `device`: bf16 on cuda, fp32 on cpu."""
    return _bf16() if device == "cuda" else _float32()


def _vlm_attn_implementation() -> str:
    """Pick the VLM attention backend, FA2 when available else `sdpa`.

    ADR-0006 mandates FlashAttention-2 for the VLM on Ada (sm_89). FA2 is
    an optional, separately-compiled native dependency (`flash_attn`);
    when it isn't installed transformers raises ImportError at model
    construction, which would kill the entire VLM escalation path. Degrade
    to PyTorch's built-in scaled-dot-product attention (`sdpa` — no extra
    dependency, numerically correct, modestly slower) rather than
    hard-failing. FA2 stays the preferred path and is used whenever the
    package is importable, so a later `pip install flash-attn` transparently
    restores it.
    """
    import importlib.util

    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    structlog.get_logger(__name__).warning(
        "vlm.attn.flash_attn_unavailable",
        fallback="sdpa",
        hint="pip install flash-attn restores the ADR-0006 FA2 path",
    )
    return "sdpa"


def _ensure_awq_import_compat() -> None:
    """Restore the `PytorchGELUTanh` symbol AutoAWQ imports at load time.

    The AWQ VLM (`Qwen2.5-VL-7B-Instruct-AWQ`) is dequantised through
    AutoAWQ, which `from transformers.activations import PytorchGELUTanh`.
    transformers renamed that class to `GELUTanh` (same tanh-approx GELU,
    still registered as `ACT2CLS["gelu_pytorch_tanh"]`), so on current
    transformers AutoAWQ fails to import and the VLM can't load at all.
    Alias the renamed class back under the old name before the load
    triggers the AutoAWQ import. Idempotent, and a no-op once AutoAWQ is
    retired (the durable fix — it's deprecated; see ROADMAP VLM upgrade).
    """
    import transformers.activations as act

    if not hasattr(act, "PytorchGELUTanh"):
        replacement = getattr(act, "GELUTanh", None)
        if replacement is not None:
            # Dynamic back-compat alias injected into a third-party module;
            # pyright can't see the new attribute, ruff dislikes setattr.
            act.PytorchGELUTanh = replacement  # type: ignore[reportAttributeAccessIssue]  # injecting renamed-symbol alias for AutoAWQ


_TPretrained = TypeVar("_TPretrained")


def _from_pretrained(
    auto_cls: type[Any], as_type: type[_TPretrained], /, *args: Any, **kwargs: Any
) -> _TPretrained:
    """Call `auto_cls.from_pretrained(...)` and return the result typed as
    `as_type`.

    transformers' Auto-class `from_pretrained` classmethods are typed
    with `Unknown` params / return in the shipped stubs, so each direct
    call trips `reportUnknownMemberType` on the method access. Funnelling
    every load through this one wrapper keeps that suppression to a single
    site while preserving precise return types at the call sites (the
    `as_type` arg flows out as the static result type).
    """
    return cast(
        "_TPretrained",
        auto_cls.from_pretrained(*args, **kwargs),  # type: ignore[reportUnknownMemberType]  # transformers stubs type Auto.from_pretrained as Unknown
    )


def _default_pad_to_eos(tokenizer: Any) -> None:
    """Set `pad_token = eos_token` when the tokenizer has no pad token.

    `PreTrainedTokenizerBase.pad_token_id` / `.eos_token` are typed as
    `str | list[str] | Unknown | None` in transformers' stubs (the
    `Unknown` arm poisons strict-mode member access), so we operate on
    the tokenizer through an explicit `Any` — a genuinely-dynamic
    transformers boundary.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def _first_token_id(tokenizer: Any, text: str) -> int:
    """Return the first token id for `text` (no special tokens).

    `tokenizer(...)` returns a `BatchEncoding` whose `.input_ids` is
    typed Unknown in transformers' stubs; access it through an explicit
    `Any` and coerce the leading id to `int`.
    """
    return int(tokenizer(text, add_special_tokens=False).input_ids[0])


class VLMHandle:
    """Bundled (model, processor) pair returned by `use('vlm')`."""

    def __init__(self, model: PreTrainedModel, processor: ProcessorMixin):
        self.model = model
        self.processor = processor


class ChartOCRHandle:
    """Bundled (model, processor) pair for the chart-OCR pass over
    Docling figures (P3.3). The model is a Pix2Struct-derivative
    fine-tuned for plot→linearised-table extraction — default
    `google/deplot` per the P3.3 Session 1 verdict.

    Lives in the orchestrator process (NOT the Docling worker) so it
    can re-use the orchestrator's GPU when vLLM is paused for parse.
    `parse/chart_ocr_backend.py::chart_ocr_extract` is the only
    consumer; the handle shape is private to the registry ↔ backend
    contract.
    """

    def __init__(self, model: PreTrainedModel, processor: ProcessorMixin):
        self.model = model
        self.processor = processor


class Qwen3RerankerHandle:
    """Bundled tokenizer + decoder + cached yes/no token ids for the
    Qwen3-Reranker autoregressive scoring backend (P2.1).

    `_score_qwen3` in `retrieve/rerank.py` is the only consumer; the
    handle's shape is private to the registry ↔ rerank contract.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        yes_id: int,
        no_id: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.yes_id = yes_id
        self.no_id = no_id


logger = structlog.get_logger(__name__)


def _is_oom(exc: BaseException) -> bool:
    """True if `exc` is a CUDA OOM. We avoid importing torch unless needed."""
    name = type(exc).__name__
    if name == "OutOfMemoryError":
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


def _empty_cuda_cache() -> None:
    """Reclaim freed CUDA blocks before a fallback retry (best-effort; no-op off-GPU / torch-less)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):  # torch absent / CUDA driver error — nothing to reclaim
        pass


ModelName = Literal["embedder", "reranker", "vlm", "chart_ocr"]


class ModelHandle(BaseModel):
    """Lightweight metadata record for a loaded model — its `name`
    slot in the registry, the HF model_id, when it was loaded, and
    a usage counter. Used by `ModelRegistry.status()` for the
    doctor report; not the underlying torch handle (that lives in
    `_models`)."""

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
            "chart_ocr": asyncio.Lock(),
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
    async def use(self, name: ModelName) -> AsyncGenerator[Any]:
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
        """Snapshot of every currently-loaded model handle. Used by
        `memex doctor` to report resident models + their usage
        counts."""
        return list(self._handles.values())

    async def unload(self, name: ModelName) -> None:
        """Release a loaded model's weights from VRAM.

        Used by the chart-OCR pass (P3.3 v2 Session 2) to free the
        VLM-class chart-OCR model BEFORE vLLM restarts at the end of
        the pause window — otherwise vLLM (~7 GB) + chart-OCR-VLM
        (~5 GB) + embedder + reranker would exceed the 12 GB rig's
        budget. The model reloads lazily on the next `use()` call.

        Acquires the same per-name lock as `use()` so concurrent
        load/unload races are serialized. Idempotent — safe to call
        when nothing is loaded.
        """
        async with self._locks[name]:
            self._models.pop(name, None)
            self._handles.pop(name, None)
            try:
                import torch  # type: ignore[import-not-found]

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    async def unload_retrieval(self) -> None:
        """Release the embedder + reranker from VRAM so a GPU-EXCLUSIVE op (the parse-time VLM serve,
        chart-OCR) gets the freed budget. They reload lazily on the next retrieval at the then-current
        `auto` placement. The pre-flight half of the dynamic VRAM manager (the auto analogue of the manual
        `…DEVICE=cpu` I had to set for keyframe batches). Idempotent."""
        await self.unload("embedder")
        await self.unload("reranker")

    async def _load(self, name: ModelName) -> None:
        log = logger.bind(name=name)
        log.info("model.load.start")

        # The co-residence mode (ADR-0007) resolves the effective device for each retrieval model;
        # `manual` falls back to the explicit knobs. `auto` (the default) reads the LIVE free-VRAM at THIS
        # model's load point — the reranker loads after the embedder + the out-of-process orchestrator are
        # resident, so the probe sees the true remaining budget (the dynamic-VRAM-manager decision).
        emb_device, rr_device = effective_devices(
            self._settings.co_residence_mode,
            self._settings.embedder_device,
            self._settings.reranker_device,
            free_vram_gb=vram.free_vram_gb(),
        )

        async def _load_retrieval(loader: Any, model_id: str, device: str) -> Any:
            """Load a retrieval model, reactively falling back GPU→CPU on a CUDA OOM — the safety net that
            makes `auto` never-OOM even if the calibrated margin is slightly off (a CPU-placed retrieval
            model is correct, just slower; reranker order is device-invariant)."""
            try:
                return await asyncio.to_thread(loader, model_id, device)
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                if device == "cuda" and _is_oom(e):
                    logger.bind(name=name).warning("model.load.oom_fallback_cpu", error=str(e)[:160])
                    _empty_cuda_cache()
                    return await asyncio.to_thread(loader, model_id, "cpu")
                raise

        async def _do_load() -> None:
            if name == "embedder":
                self._models[name] = await _load_retrieval(
                    self._load_embedder, self._settings.embedder, emb_device
                )
            elif name == "reranker":
                loader = (
                    self._load_reranker_qwen3
                    if self._settings.reranker_backend == "qwen3"
                    else self._load_reranker
                )
                self._models[name] = await _load_retrieval(loader, self._settings.reranker, rr_device)
            elif name == "vlm":
                self._models[name] = await asyncio.to_thread(self._load_vlm, self._settings.vlm)
            elif name == "chart_ocr":
                self._models[name] = await asyncio.to_thread(
                    self._load_chart_ocr, self._settings.chart_ocr
                )
            else:
                raise ModelNotConfigured(f"unknown model {name!r}", context={"name": name})

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
    def _load_embedder(model_id: str, device: str = "cuda") -> SentenceTransformer:
        """Load the dense embedder.

        ADR-0006: BF16 on `device="cuda"`. EmbeddingGemma 300M's official
        model card states its activations DO NOT support FP16 — must be FP32
        or BF16. Memex picks BF16 globally on Ada; on a CPU placement
        (`embedder_device="cpu"`) it loads FP32 (`_retrieval_dtype`).

        Imported lazily so the agent layer can be imported without
        torch installed (`[models]` extra brings it in).
        """
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            model_id,
            device=device,
            model_kwargs={"torch_dtype": _retrieval_dtype(device)},
        )

    @staticmethod
    def _load_reranker(model_id: str, device: str = "cuda") -> CrossEncoder:
        """Load the cross-encoder reranker (default backend).

        ADR-0006: BF16 on `device="cuda"`; FP32 on a CPU placement
        (`reranker_device="cpu"`, to free GPU VRAM for the orchestrator).
        """
        from sentence_transformers import CrossEncoder

        return CrossEncoder(
            model_id,
            device=device,
            # dtype passthrough to the underlying AutoModel. The param is
            # `model_kwargs` in sentence-transformers 5.x (matches
            # `_load_embedder` above); an earlier `automodel_args` name
            # would TypeError at load time — a latent bug the 2026-05-23
            # typing pass surfaced (the reranker is faked in tests).
            model_kwargs={"torch_dtype": _retrieval_dtype(device)},
        )

    @staticmethod
    def _load_reranker_qwen3(model_id: str, device: str = "cuda") -> Qwen3RerankerHandle:
        """Load the Qwen3-Reranker autoregressive backend (P2.1).

        Decoder-only LLM fine-tuned to answer "yes"/"no" given a (query,
        document) pair. Scoring lives in `retrieve.rerank._score_qwen3`:
        format → forward pass → last-token logits → softmax over the
        cached yes/no ids.

        ADR-0006 conventions: explicit `torch_dtype` (bf16 on cuda, fp32 on a
        CPU placement) and an explicit single-device `device_map` (no
        `"auto"` — an unrequested CPU offload should never happen silently;
        a CPU placement here is the EXPLICIT `reranker_device="cpu"` choice).
        """
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            PreTrainedModel,
            PreTrainedTokenizerBase,
        )

        # `_from_pretrained` types the loaded objects as the concrete
        # bases we depend on (transformers' Auto-class stubs return
        # Unknown), so member access (`.eval()`) type-checks.
        tokenizer = _from_pretrained(
            AutoTokenizer, PreTrainedTokenizerBase, model_id, padding_side="left"
        )
        _default_pad_to_eos(tokenizer)
        model = _from_pretrained(
            AutoModelForCausalLM,
            PreTrainedModel,
            model_id,
            torch_dtype=_retrieval_dtype(device),
            device_map={"": "cuda:0" if device == "cuda" else "cpu"},
            low_cpu_mem_usage=True,
        )
        model.eval()
        # The Qwen3-Reranker model card uses "yes" / "no" as the answer
        # tokens. Cache the ids once at load — repeated tokenisation
        # per pair would waste a chunk of the per-rerank latency budget.
        yes_id = _first_token_id(tokenizer, "yes")
        no_id = _first_token_id(tokenizer, "no")
        return Qwen3RerankerHandle(
            tokenizer=tokenizer,
            model=model,
            yes_id=yes_id,
            no_id=no_id,
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
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            PreTrainedModel,
            ProcessorMixin,
        )

        _ensure_awq_import_compat()
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        processor = _from_pretrained(
            AutoProcessor,
            ProcessorMixin,
            model_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        model = _from_pretrained(
            AutoModelForImageTextToText,
            PreTrainedModel,
            model_id,
            torch_dtype=_bf16(),
            device_map={"": "cuda:0"},
            attn_implementation=_vlm_attn_implementation(),
            low_cpu_mem_usage=True,
        )
        model.eval()
        return VLMHandle(model=model, processor=processor)

    @staticmethod
    def _load_chart_ocr(model_id: str) -> ChartOCRHandle:
        """Load the chart-OCR model + its processor.

        Three backends supported, dispatched by model_id substring:

        - **DePlot-style** (default, `google/deplot`): Pix2Struct
          derivative fine-tuned for plot→linearised-table. 0.3B / 1.13
          GB on disk / ~2.3 GB live in BF16. Loaded via
          `Pix2StructForConditionalGeneration` + `Pix2StructProcessor`.
          SDPA attention (T5-style, FA2 not applicable).

        - **VLM-style** (`Qwen*/*-VL-*`, etc.): a vision-language model
          used as a chart extractor with an explicit UNREADABLE-escape-
          hatch prompt (P3.3 v2 Session 2; the
          ChartHal-recommended hallucination-resistant approach).
          Loaded via `AutoModelForImageTextToText` + `AutoProcessor`
          with FA2 + BF16, mirroring `_load_vlm`. ~5-6 GB live for the
          Qwen2.5-VL-7B AWQ variant.

        - **OneChart-style** (`kppkkp/OneChart`, P3.3-b): a 0.3B-param
          specialty model with a custom Vary-derived architecture
          + auxiliary `reliable_check` self-consistency token. Loaded
          via `AutoModel` with `trust_remote_code=True` per the
          ADR-0006 amendment (the only carve-out from the general
          `trust_remote_code` prohibition; see ADR §4). Uses a custom
          `.chat()` interface rather than `.generate()`; the backend
          dispatches on handle class name.

        The VLM-style handle is transient: chart_ocr_backend explicitly
        calls `registry.unload("chart_ocr")` after the extraction pass
        so the model doesn't compete with vLLM's KV cache during query
        time. DePlot + OneChart handles are small enough to stay
        resident without strain.
        """
        # Heuristic: model IDs that contain "VL" / "vision" / "VLM"
        # use the VLM loading pattern. "onechart" gets its own
        # trust_remote_code branch (ADR-0006 amendment). Anything else
        # assumes a Pix2Struct-style chart specialist.
        lid = model_id.lower()
        is_onechart = "onechart" in lid
        is_vlm = "-vl-" in lid or lid.endswith("-vl") or "vision" in lid or "vlm" in lid
        from transformers import (
            PreTrainedModel,
            PreTrainedTokenizerBase,
            ProcessorMixin,
        )

        # Each branch loads via `_from_pretrained`, which types the
        # results as the concrete bases the `ChartOCRHandle` contract
        # expects (transformers' Auto-class stubs return Unknown), so
        # `.eval()` and handle construction type-check. `model`/`processor`
        # carry the branch-resolved concrete types into the shared tail.
        model: PreTrainedModel
        processor: ProcessorMixin

        if is_onechart:
            from transformers import AutoModel, AutoTokenizer

            # ADR-0006 amendment carve-out: OneChart needs
            # trust_remote_code=True because its custom Vary-derived
            # architecture lives in the HF repo, not in transformers
            # proper. Acceptable here ONLY because: (a) the model is
            # opt-in via env-var; (b) it's 0.3B + Apache 2.0 + human-
            # auditable; (c) the backend gates output on the model's
            # own reliable_check self-consistency token.
            tokenizer = _from_pretrained(
                AutoTokenizer, ProcessorMixin, model_id, trust_remote_code=True
            )
            model = _from_pretrained(
                AutoModel,
                PreTrainedModel,
                model_id,
                trust_remote_code=True,
                torch_dtype=_bf16(),
                device_map={"": "cuda:0"},
                low_cpu_mem_usage=True,
            )
            model.eval()
            # OneChart's `.chat()` API takes the tokenizer directly
            # (no processor). Store the tokenizer in the `processor`
            # slot so the rest of the backend has a uniform handle
            # shape; the backend's dispatcher detects OneChart by
            # model class name and uses tokenizer-style preprocessing.
            return ChartOCRHandle(model=model, processor=tokenizer)
        elif is_vlm:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            # Match `_load_vlm`'s processor config so chart crops get
            # the same visual-token budget tuning.
            min_pixels = 256 * 28 * 28
            max_pixels = 1280 * 28 * 28
            processor = _from_pretrained(
                AutoProcessor,
                ProcessorMixin,
                model_id,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            # Try FA2 first (matches `_load_vlm`); fall back to SDPA
            # if flash_attn isn't installed. The chart-OCR backend
            # processes individual figure crops (small) rather than
            # full pages, so SDPA's slightly lower throughput is
            # acceptable for the chart-OCR use case.
            try:
                import flash_attn  # noqa: F401  # type: ignore

                attn_impl = "flash_attention_2"
            except ImportError:
                attn_impl = "sdpa"
            model = _from_pretrained(
                AutoModelForImageTextToText,
                PreTrainedModel,
                model_id,
                torch_dtype=_bf16(),
                device_map={"": "cuda:0"},
                attn_implementation=attn_impl,
                low_cpu_mem_usage=True,
            )
        elif "nemotron-parse" in lid:
            # Path C: NVIDIA Nemotron-Parse-v1.2 (P3.3-c follow-up,
            # 2026-05-23). 885M-param VisionEncoderDecoder-style
            # document parser. NVIDIA Nemotron Open License,
            # vLLM-compatible. Uses custom `NemotronParseForConditional
            # Generation` class via `auto_map` → requires
            # `trust_remote_code=True` (covered by ADR-0006
            # amendment 2026-05-23 chart-OCR carve-out).
            from transformers import (
                AutoModel,
                AutoProcessor,
                AutoTokenizer,
            )

            tokenizer = _from_pretrained(
                AutoTokenizer, PreTrainedTokenizerBase, model_id, trust_remote_code=True
            )
            processor = _from_pretrained(
                AutoProcessor, ProcessorMixin, model_id, trust_remote_code=True
            )
            # Store both tokenizer + processor on the handle. Attach
            # the tokenizer as a private attribute on the processor
            # so the backend has a single dispatch point.
            processor._memex_tokenizer = tokenizer  # type: ignore[attr-defined]  # private side-channel for the chart-OCR backend
            model = _from_pretrained(
                AutoModel,
                PreTrainedModel,
                model_id,
                trust_remote_code=True,
                torch_dtype=_bf16(),
                device_map={"": "cuda:0"},
                low_cpu_mem_usage=True,
            )
            model.eval()
            return ChartOCRHandle(model=model, processor=processor)
        elif "chart-to-table" in lid or "unichart" in lid:
            # Path A: UniChart Donut-style VisionEncoderDecoder
            # (P3.3-c follow-up, 2026-05-23). 0.2B params, Apache 2.0,
            # `donut-swin` encoder + `mbart` decoder, max_position_
            # embeddings=1536 (well-bounded; no OneChart-style overflow
            # risk). Loads via stock transformers — no
            # `trust_remote_code` needed. Uses `DonutProcessor`.
            from transformers import DonutProcessor, VisionEncoderDecoderModel

            processor = _from_pretrained(DonutProcessor, ProcessorMixin, model_id)
            model = _from_pretrained(
                VisionEncoderDecoderModel,
                PreTrainedModel,
                model_id,
                torch_dtype=_bf16(),
                device_map={"": "cuda:0"},
                low_cpu_mem_usage=True,
            )
        else:
            from transformers import (
                Pix2StructForConditionalGeneration,
                Pix2StructProcessor,
            )

            processor = _from_pretrained(Pix2StructProcessor, ProcessorMixin, model_id)
            model = _from_pretrained(
                Pix2StructForConditionalGeneration,
                PreTrainedModel,
                model_id,
                torch_dtype=_bf16(),
                device_map={"": "cuda:0"},
                low_cpu_mem_usage=True,
            )
        model.eval()
        return ChartOCRHandle(model=model, processor=processor)


_REGISTRY: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Return the process registry. Configured at startup."""
    if _REGISTRY is None:
        raise ModelNotConfigured(
            "ModelRegistry not initialised; call set_registry() from the entry point.",
            context={"fix": "cli.bootstrap.bootstrap() does this for you"},
        )
    return _REGISTRY


def set_registry(registry: ModelRegistry | None) -> None:
    """Install or detach the process registry. Tests pass None to detach."""
    global _REGISTRY
    _REGISTRY = registry
