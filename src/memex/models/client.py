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
from typing import Any, TypeVar

import structlog
from langfuse.openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from memex.core.config import InferenceSettings
from memex.core.errors import ModelCallError

T = TypeVar("T", bound=BaseModel)

logger = structlog.get_logger(__name__)

_client: AsyncOpenAI | None = None


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
    defs = schema.pop("$defs", {}) or schema.pop("definitions", {}) or {}
    if not defs:
        return schema

    in_progress: set[str] = set()

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"]
                # Only resolve local references under `$defs`/`definitions`;
                # leave external refs untouched (xgrammar treats those
                # consistently across backends).
                local_prefix = None
                for prefix in ("#/$defs/", "#/definitions/"):
                    if ref.startswith(prefix):
                        local_prefix = prefix
                        break
                if local_prefix is None:
                    return node
                name = ref[len(local_prefix) :]
                if name in in_progress:
                    return node  # cycle — leave the ref alone
                if name not in defs:
                    return node
                in_progress.add(name)
                try:
                    resolved = _resolve(defs[name])
                finally:
                    in_progress.discard(name)
                # If `$ref` is co-located with other keys (the allOf-like
                # pattern), merge so siblings override the resolved body.
                siblings = {
                    k: _resolve(v) for k, v in node.items() if k != "$ref"
                }
                if isinstance(resolved, dict):
                    return {**resolved, **siblings}
                return resolved
            return {k: _resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_resolve(item) for item in node]
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
            loop.create_task(old.close())
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
    prompt: str,
    schema: type[T],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 640,
    prompt_tag: str | None = None,
) -> tuple[T, int]:
    """Call the orchestrator with grammar-constrained decoding.

    Returns `(parsed instance of schema, total tokens used)`. Generic
    over `schema` so callers get back the exact subclass they asked
    for — `pyright --strict` keeps the chain typed.

    `prompt_tag` (e.g. "answer@v1") is forwarded to Langfuse as the
    span name; if absent, the schema class name is used.
    """
    client = get_client()
    if model is None:
        # vLLM 0.21+ requires the served model name in chat completions
        # ("default" is no longer accepted as a fallback). The orchestrator
        # string in settings is the same id `vllm serve` was launched with.
        from memex.core.config import get_settings

        model = get_settings().models.orchestrator
    log = logger.bind(
        prompt_tag=prompt_tag or schema.__name__,
        schema=schema.__name__,
    )
    log.info("model_call.start", model=model, max_tokens=max_tokens)

    # Default the Langfuse span name to the schema class so traces are
    # meaningfully labelled even when the caller doesn't pass a tag.
    # Passing `name=None` to the wrapped SDK either drops the span name
    # (best case) or trips a vendor schema check (worst case).
    span_name = prompt_tag or schema.__name__

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
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
