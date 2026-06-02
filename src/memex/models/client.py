"""vLLM client wrapper with structured-output decoding.

Per ADR-0001 we never accept free-text JSON from the model. Every call
that expects structured output goes through `complete_structured`, which
constrains the decoder via `response_format={"type": "json_schema"}` —
the OpenAI-standard form (portable across vLLM, SGLang, llama-server),
preferred over vLLM's deprecated `extra_body={"guided_json": ...}`.

The OpenAI client is imported through Langfuse's wrapper
(`from langfuse.openai import openai`), so every call is automatically
captured as a Langfuse generation span under the current trace, with
prompt, completion, token counts, and latency attached. See ADR-0004.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, TypeVar, cast

import structlog
from pydantic import BaseModel, ValidationError

# Runtime import goes through Langfuse's instrumented wrapper so every
# call is captured as a generation span (ADR-0004). `langfuse.openai`
# re-exports the very same `openai.AsyncOpenAI` class object (it patches
# the methods rather than subclassing), but the re-export is dynamic so
# pyright can't see it ("not exported"). For type-checking we therefore
# reference the canonical `openai.AsyncOpenAI` — identical type, fully
# stubbed — while keeping the instrumented symbol at runtime.
if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion
else:
    from langfuse.openai import AsyncOpenAI

from memex.core.config import InferenceSettings
from memex.core.errors import ModelCallError

T = TypeVar("T", bound=BaseModel)

logger = structlog.get_logger(__name__)

_client: AsyncOpenAI | None = None

# Strong refs to fire-and-forget client-close tasks. Without this the
# event loop only holds a weak ref and may GC the task before
# `old.close()` finishes (RUF006). The done-callback drains the set.
_CLEANUP_TASKS: set[asyncio.Task[None]] = set()

# Per-async-context inference routing override (ADR-0010 summarizer swap-in). When a
# summarize spawns a short-lived stronger model on its own port, the WHOLE map-reduce's
# `complete_structured` calls route to that base_url + model for the duration — via a
# ContextVar, NOT by reconfiguring the global `_client`, so a concurrent /ask on the
# orchestrator is unaffected. A tiny per-base_url client cache avoids leaking a
# connection pool per call.
_inference_override: ContextVar[tuple[str, str] | None] = ContextVar(
    "inference_override", default=None
)
_override_clients: dict[str, AsyncOpenAI] = {}


@asynccontextmanager
async def inference_override(base_url: str, model: str) -> AsyncGenerator[None]:
    """Route every `complete_structured` in this async context to `base_url` + `model`
    (a swapped-in summarizer). Resets on exit; never touches the global client."""
    token = _inference_override.set((base_url, model))
    try:
        yield
    finally:
        _inference_override.reset(token)


def _override_client(base_url: str) -> AsyncOpenAI:
    """A cached client for an override base_url (reused across the summarize's calls)."""
    c = _override_clients.get(base_url)
    if c is None:
        c = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
        _override_clients[base_url] = c
    return c


class _Unset:
    """Sentinel for kwargs where `None` is a meaningful value distinct
    from "use the settings default." Used by `complete_structured`'s
    `seed` parameter: `None` means "no seed," `_UNSET` means
    "fall back to `MemexSettings.inference.sampling.seed`."
    """


_UNSET = _Unset()


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline `$defs`/`$ref` so vLLM's xgrammar backend doesn't silently
    fall back to Outlines on nested pydantic schemas.

    Pydantic's `BaseModel.model_json_schema()` factors nested models
    into a `$defs` block with `$ref` pointers (e.g. `Answer.claims:
    list[CitedClaim]` produces a `$defs/CitedClaim` entry). xgrammar
    skips `$ref`-bearing schemas per
    https://github.com/vllm-project/vllm/pull/10935; vLLM then routes
    to Outlines without notice, which is slower and less reliable.
    We resolve the references in-place before sending.

    Cycles (rare in practice for pydantic-derived schemas) are
    handled by leaving the inner `$ref` intact — xgrammar can skip
    that specific node rather than the whole schema.
    """
    schema = dict(schema)
    # `.pop` on a `dict[str, Any]` returns `Any`; pin `defs` to a
    # concrete mapping type so the recursive resolution below is typed.
    defs: dict[str, Any] = schema.pop("$defs", {}) or schema.pop("definitions", {}) or {}
    if not defs:
        return schema

    in_progress: set[str] = set()

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            # `isinstance` narrows to `dict[Unknown, Unknown]`; the JSON
            # schema is `dict[str, Any]`-shaped, so cast to keep member
            # access typed.
            node_dict = cast(dict[str, Any], node)
            if "$ref" in node_dict:
                ref: str = node_dict["$ref"]
                # Only resolve local references under `$defs`/`definitions`;
                # leave external refs untouched (xgrammar treats those
                # consistently across backends).
                local_prefix: str | None = None
                for prefix in ("#/$defs/", "#/definitions/"):
                    if ref.startswith(prefix):
                        local_prefix = prefix
                        break
                if local_prefix is None:
                    return node_dict
                name = ref[len(local_prefix) :]
                if name in in_progress:
                    return node_dict  # cycle — leave the ref alone
                if name not in defs:
                    return node_dict
                in_progress.add(name)
                try:
                    resolved: Any = _resolve(defs[name])
                finally:
                    in_progress.discard(name)
                # If `$ref` is co-located with other keys (the allOf-like
                # pattern), merge so siblings override the resolved body.
                siblings: dict[str, Any] = {
                    k: _resolve(v) for k, v in node_dict.items() if k != "$ref"
                }
                if isinstance(resolved, dict):
                    resolved_dict = cast(dict[str, Any], resolved)
                    return {**resolved_dict, **siblings}
                return resolved
            return {k: _resolve(v) for k, v in node_dict.items()}
        if isinstance(node, list):
            node_list = cast(list[Any], node)
            return [_resolve(item) for item in node_list]
        return node

    return _resolve(schema)


def configure_client(settings: InferenceSettings) -> None:
    """Initialise the singleton OpenAI client pointed at vLLM. Call once
    at startup; safe to call again to swap settings (the prior client's
    httpx connection pool is best-effort closed first).

    Audit item N3: a previous version simply reassigned `_client`,
    leaking the prior httpx connection pool. AsyncOpenAI wraps an
    httpx AsyncClient with persistent keep-alive connections; without
    explicit `.close()` they linger until the GC reaps the orphan, at
    which point httpx emits `ResourceWarning: unclosed transport`. In
    a long-running daemon that re-reads config (e.g., SIGHUP-driven
    reconfig, future orchestrator swap), this slowly exhausts the
    file-descriptor budget. Best-effort sync cleanup: if we're inside
    a running event loop, schedule the close as a fire-and-forget
    task; if not (sync startup, test reconfiguration), log a warning
    and rely on GC + httpx's TCP timeouts.
    """
    global _client
    if _client is not None:
        old = _client
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(old.close())
            _CLEANUP_TASKS.add(task)
            task.add_done_callback(_CLEANUP_TASKS.discard)
        except RuntimeError:
            logger.warning(
                "configure_client.no_loop_for_cleanup",
                note=(
                    "previous OpenAI/httpx client cannot be closed without a "
                    "running event loop; relying on GC + TCP timeouts. Call "
                    "configure_client only at startup or from within an "
                    "active asyncio loop to avoid connection-pool linger."
                ),
            )
    _client = AsyncOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.request_timeout_s,
    )


def get_client() -> AsyncOpenAI:
    """Return the configured client. Tests override this for fakes."""
    if _client is None:
        raise ModelCallError(
            "OpenAI/vLLM client is not configured; call configure_client() first.",
            context={"fix": "configure_client(MemexSettings(...).inference)"},
        )
    return _client


async def complete_structured(
    prompt: str | list[dict[str, str]],
    schema: type[T],
    *,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    seed: int | None | _Unset = _UNSET,
    max_tokens: int | None = None,
    prompt_tag: str | None = None,
) -> tuple[T, int]:
    """Call the orchestrator with grammar-constrained decoding.

    Returns `(parsed instance of schema, total tokens used)`. Generic
    over `schema` so callers get back the exact subclass they asked
    for — `pyright --strict` keeps the chain typed.

    `prompt` accepts either:
    - A single string — wrapped in `[{"role":"user","content":prompt}]`
      (legacy behaviour, same as before this function was generalised).
    - A list of OpenAI-style message dicts — passed through directly.
      Use this when the prompt is split into a system + user pair via
      `render_messages` (see `src/memex/prompts/loader.py`). The
      multi-message form lets vLLM cache the prefix of the system
      block across calls with different user content.

    Sampling defaults come from `MemexSettings.inference.sampling`
    (`SamplingSettings` in `core/config.py`). The values shipped are
    Qwen team's published non-thinking-mode recommendation, scaled
    down for eval determinism. Per-call kwargs override the settings
    defaults; pass `None` (or omit) to use the settings value.

    `seed` uses a sentinel (`_UNSET`) rather than `None` because
    `None` is a meaningful value ("no seed; vLLM picks one"). The
    sentinel means "use the settings default."

    `prompt_tag` (e.g. "answer@v1") is forwarded to Langfuse as the
    span name; if absent, the schema class name is used.
    """
    from memex.core.config import get_settings

    settings = get_settings()
    sampling = settings.inference.sampling

    # Resolve per-call kwargs against the settings defaults.
    temperature = temperature if temperature is not None else sampling.temperature
    top_p = top_p if top_p is not None else sampling.top_p
    presence_penalty = (
        presence_penalty if presence_penalty is not None else sampling.presence_penalty
    )
    seed_val = sampling.seed if isinstance(seed, _Unset) else seed
    max_tokens = max_tokens if max_tokens is not None else sampling.max_tokens

    override = _inference_override.get()
    if override is not None:
        # A swap-in summarizer (or similar) owns this async context — route here, and
        # use its served model id (NOT the orchestrator's).
        ov_base_url, model = override
        client = _override_client(ov_base_url)
    else:
        client = get_client()
        if model is None:
            # vLLM 0.21+ requires the served model name in chat completions
            # ("default" is no longer accepted as a fallback). The orchestrator
            # string in settings is the same id `vllm serve` was launched with.
            model = settings.models.orchestrator
    log = logger.bind(
        prompt_tag=prompt_tag or schema.__name__,
        schema=schema.__name__,
    )
    # Coerce single-string `prompt` to the OpenAI message shape; pass
    # multi-message lists through unchanged.
    messages: list[dict[str, str]] = (
        [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    )

    log.info(
        "model_call.start",
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        seed=seed_val,
        message_count=len(messages),
    )

    # Default the Langfuse span name to the schema class so traces are
    # meaningfully labelled even when the caller doesn't pass a tag.
    # Passing `name=None` to the wrapped SDK either drops the span name
    # (best case) or trips a vendor schema check (worst case).
    span_name = prompt_tag or schema.__name__

    try:
        # `name=` is a Langfuse-wrapper extension (it sets the generation
        # span name) that the upstream openai `create` overloads don't
        # declare, so the call needs `call-overload` suppressed. Cast the
        # result back to the real `ChatCompletion` return type so the
        # `.choices` / `.usage` access below stays fully typed.
        raw_response = await client.chat.completions.create(  # type: ignore[call-overload]  # langfuse `name=` kwarg
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            seed=seed_val,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": _inline_refs(schema.model_json_schema()),
                    "strict": True,
                },
            },
            name=span_name,  # picked up by langfuse.openai as span name
        )
        response = cast("ChatCompletion", raw_response)
    except asyncio.CancelledError:
        # Don't wrap cancellation as a model error; cooperative shutdown
        # must remain observable to callers (graph timeout, agent abort).
        raise
    except Exception as e:
        # Narrow to expected failure types via duck-checking the module:
        # openai/httpx exceptions are real I/O failures; anything else
        # (TypeError, AttributeError, NameError) is a programming bug
        # we should never disguise as a model-call failure.
        mod = type(e).__module__
        if mod.startswith(("openai", "httpx")) or isinstance(e, TimeoutError):
            raise ModelCallError(
                f"vLLM call failed: {e}",
                context={
                    "schema": schema.__name__,
                    "prompt_tag": span_name,
                    "error_type": type(e).__name__,
                },
            ) from e
        raise

    text = response.choices[0].message.content or ""
    try:
        parsed = schema.model_validate_json(text)
    except ValidationError as e:
        raise ModelCallError(
            "Model output did not match the requested schema.",
            context={
                "schema": schema.__name__,
                "prompt_tag": prompt_tag,
                "raw_output": text[:500],
                "validation_errors": e.errors(),
            },
        ) from e

    tokens = response.usage.total_tokens if response.usage else 0
    log.info("model_call.done", tokens=tokens)
    return parsed, tokens


def split_think(text: str) -> tuple[str | None, str]:
    """Split a ``<think>…</think>`` reasoning trace from the answer body, if present.

    DEFENSIVE / forward-compat: on the live 4B checkpoint via vLLM (verified 2026-06-01)
    ``enable_thinking=true`` emits the chain-of-thought INLINE as prose — there is no
    ``<think>`` tag — so this is normally a no-op returning ``(None, text)``. It handles a
    genuinely tagged trace (a future model, or a vLLM ``--reasoning-parser`` config) so the
    expert surface can show the answer prominently and tuck the trace away. A trace truncated
    at ``max_tokens`` (open ``<think>`` with no close) is treated as all-trace. Fail-open:
    any unexpected shape returns ``(None, text)`` so the caller always has a body to show.
    """
    start = text.find("<think>")
    if start == -1:
        return None, text
    end = text.find("</think>", start)
    if end == -1:
        # The close tag never arrived (cut mid-thought) — the whole tail is the trace.
        return text[start + len("<think>") :].strip() or None, ""
    trace = text[start + len("<think>") : end].strip()
    body = (text[:start] + text[end + len("</think>") :]).strip()
    return (trace or None), body


async def complete_reasoning(
    prompt: str | list[dict[str, str]],
    *,
    model: str | None = None,
    enable_thinking: bool = True,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    prompt_tag: str | None = None,
) -> tuple[str, int]:
    """FREE-TEXT completion in the hybrid-reasoning model's THINKING mode — the engine of the
    UNGROUNDED expert surface (Surface B, ADR-0013).

    Deliberately SEPARATE from `complete_structured`: this call passes NO `response_format`
    (no grammar) and sets `enable_thinking` via `chat_template_kwargs`, so the model reasons
    freely. `complete_structured` stays the SOLE emitter of `response_format` and NEVER sets
    `chat_template_kwargs` — so "is this call grounded?" is answerable by WHICH function it is.
    The grounded /ask + chat graph must never import this; its grammar suppresses reasoning by
    construction, this has no grammar. Returns ``(raw text, total_tokens)`` — the caller splits
    any `<think>` trace via `split_think` (a no-op on this checkpoint; the CoT is inline prose).

    Sampling defaults to a reasoning-appropriate posture — a temperature floor of 0.6 and NO
    grounded seed pin — rather than the determinism floor the grounded path uses; pass explicit
    kwargs to override. Routing mirrors `complete_structured` (the summarizer-style
    `_inference_override` swap-in, else the live daemon — defaulting to `models.reasoner` when
    set, else the orchestrator, since the 4B is itself a reasoning model)."""
    from memex.core.config import get_settings

    settings = get_settings()
    sampling = settings.inference.sampling
    temperature = temperature if temperature is not None else max(sampling.temperature, 0.6)
    top_p = top_p if top_p is not None else sampling.top_p
    max_tokens = max_tokens if max_tokens is not None else 1024

    override = _inference_override.get()
    if override is not None:
        ov_base_url, model = override
        client = _override_client(ov_base_url)
    else:
        client = get_client()
        if model is None:
            model = settings.models.reasoner or settings.models.orchestrator

    messages: list[dict[str, str]] = (
        [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    )
    span_name = prompt_tag or "complete_reasoning"
    log = logger.bind(prompt_tag=span_name)
    log.info(
        "reasoning_call.start",
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
        message_count=len(messages),
    )
    try:
        raw_response = await client.chat.completions.create(  # type: ignore[call-overload]  # langfuse `name=` kwarg + chat_template_kwargs extra_body
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
            name=span_name,
        )
        response = cast("ChatCompletion", raw_response)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        mod = type(e).__module__
        if mod.startswith(("openai", "httpx")) or isinstance(e, TimeoutError):
            raise ModelCallError(
                f"vLLM reasoning call failed: {e}",
                context={"prompt_tag": span_name, "error_type": type(e).__name__},
            ) from e
        raise

    text = response.choices[0].message.content or ""
    tokens = response.usage.total_tokens if response.usage else 0
    log.info("reasoning_call.done", tokens=tokens, chars=len(text))
    return text, tokens
