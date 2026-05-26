"""Reranker dispatcher — bge cross-encoder by default; Qwen3-Reranker behind a flag.

Two backends share the same public API (`rerank`) and the same model-
registry slot. Selection is by `settings.models.reranker_backend`:

  - `cross_encoder` (default) → `sentence_transformers.CrossEncoder`,
    typically `BAAI/bge-reranker-v2-m3`. One forward pass per pair via
    `predict()`.
  - `qwen3` (P2.1) → `transformers.AutoModelForCausalLM`, typically
    `Qwen/Qwen3-Reranker-0.6B`. Autoregressive yes/no judgement; we
    take softmax over the cached yes/no token logits at the last
    non-pad position.

The legacy public name `cross_encoder_rerank` stays exported as an
alias so existing callers + test monkeypatches don't need to change.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Protocol, cast

import structlog

from memex.core.config import get_settings
from memex.core.types import Chunk
from memex.models.registry import Qwen3RerankerHandle, get_registry

logger = structlog.get_logger(__name__)


class _FloatTensor(Protocol):
    """Typing-only view of a 1-D float `torch.Tensor`. torch's own stub
    types `Tensor.tolist()` as a bare `list` (→ list[Unknown] under
    strict); we know `probs[:, 1]` is 1-D float, so we cast to this
    Protocol whose `tolist()` returns `list[float]`."""

    def tolist(self) -> list[float]: ...


_QWEN3_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the "
    "Query and the Instruct provided. Note that the answer can only "
    'be "yes" or "no".'
)
_QWEN3_TASK = "Given a web search query, retrieve relevant passages that answer the query"


def _read_batch_size() -> int:
    # batch_size=8 is the safe default on a 12 GB rig running the bge
    # cross-encoder alongside vLLM-Qwen3-8B-AWQ. Qwen3-Reranker-0.6B
    # fits a larger batch (smaller model, smaller hidden dim per token),
    # but we keep the same default and let users bump it via env.
    try:
        bs = int(os.environ.get("MEMEX_RERANK_BATCH_SIZE", "8"))
    except ValueError:
        bs = 8
    return max(1, bs)


def _is_cuda_oom(e: BaseException) -> bool:
    """True if `e` is a CUDA out-of-memory error — `torch.cuda.OutOfMemoryError`
    (a `RuntimeError` subclass) on the modern path, or a bare `RuntimeError`
    whose message carries 'out of memory'. torch is imported lazily so the
    module stays importable without the [models] extra."""
    try:
        import torch

        if isinstance(e, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def _empty_cuda_cache() -> None:
    """Free the caching allocator's fragmented blocks before an OOM retry."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


async def _score_with_oom_fallback(
    reranker: object, pairs: list[tuple[str, str]], batch_size: int
) -> Any:
    """Score `pairs`, retrying ONCE at batch_size=1 on a CUDA OOM.

    On the 12 GB reference rig the reranker OOMs when an agent (or
    `memex serve web`) is co-resident with vLLM at the default batch 8 — which
    previously forced operators to set `MEMEX_RERANK_BATCH_SIZE=1` by hand or
    eat a crash mid-answer. Catch the OOM, free the fragmented allocations, and
    retry at batch 1 (the documented safe floor) so the answer path degrades
    gracefully. Re-raises if already at batch 1 (can't reduce further) or if the
    error isn't a CUDA OOM (a real bug should surface, not be swallowed)."""

    def _run(bs: int) -> Any:
        if isinstance(reranker, Qwen3RerankerHandle):
            return _score_qwen3(reranker, pairs, bs)
        return _score_cross_encoder(reranker, pairs, bs)

    try:
        return await asyncio.to_thread(_run, batch_size)
    except RuntimeError as e:
        if batch_size <= 1 or not _is_cuda_oom(e):
            raise
        logger.warning("rerank.oom_fallback", original_batch_size=batch_size, error=str(e)[:160])
        _empty_cuda_cache()
        return await asyncio.to_thread(_run, 1)


async def cross_encoder_rerank(
    query: str,
    candidates: list[Chunk],
    top_k: int = 10,
) -> list[Chunk]:
    """Rescore `candidates` against `query` and return the top `top_k`.

    Misnomer: the function dispatches to either the CrossEncoder backend
    OR the Qwen3-Reranker backend based on
    `settings.models.reranker_backend`. The historical name is kept
    because (1) it's the import path used by `agents/answering.py` and
    test monkeypatches, and (2) renaming would collide with the module
    name `memex.retrieve.rerank` and break attribute resolution on the
    package init.

    Returns each surviving chunk with its `rerank_score` field populated.
    """
    if not candidates:
        return []

    settings = get_settings()
    backend = settings.models.reranker_backend
    log = logger.bind(candidates=len(candidates), top_k=top_k, backend=backend)
    log.info("rerank.start")

    batch_size = _read_batch_size()

    registry = get_registry()
    async with registry.use("reranker") as reranker:
        pairs = [(query, c.text) for c in candidates]
        scores = await _score_with_oom_fallback(reranker, pairs, batch_size)

    ranked = sorted(
        zip(candidates, (float(s) for s in scores), strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    out = [c.model_copy(update={"rerank_score": score}) for c, score in ranked[:top_k]]
    log.info("rerank.done", returned=len(out))
    return out


def _score_cross_encoder(reranker: Any, pairs: list[tuple[str, str]], batch_size: int) -> Any:
    """The CrossEncoder backend — one forward pass per pair via
    `sentence_transformers.CrossEncoder.predict`. Returns a numpy
    array of float logits, one per pair.
    """
    return reranker.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def _qwen3_format(query: str, doc: str) -> str:
    """Build the chat-template-formatted prompt for one (query, doc) pair.

    Mirrors the Qwen3-Reranker model card's reference format. The
    `<think></think>` block is left empty: we're scoring, not
    reasoning, and the next-token logits at the assistant header are
    what we read.
    """
    return (
        "<|im_start|>system\n"
        f"{_QWEN3_SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"<Instruct>: {_QWEN3_TASK}\n"
        f"<Query>: {query}\n"
        f"<Document>: {doc}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


def _score_qwen3(
    handle: Qwen3RerankerHandle,
    pairs: list[tuple[str, str]],
    batch_size: int,
) -> list[float]:
    """The Qwen3-Reranker backend.

    For each batch of pairs:
      1. Format via `_qwen3_format` → chat-template strings.
      2. Tokenise with left-padding (so the "last token" of every
         sequence is the assistant-prefix marker after `<think></think>`).
      3. Forward pass; extract logits at the final position.
      4. Score = softmax([no_logit, yes_logit])[1] — the probability
         the model assigns to "yes".

    Returns a flat `list[float]` aligned with `pairs`.
    """
    import torch

    scores: list[float] = []
    device = next(handle.model.parameters()).device

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        prompts = [_qwen3_format(q, d) for q, d in chunk]
        inputs = handle.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        ).to(device)

        with torch.no_grad():
            logits = handle.model(**inputs).logits  # (B, T, V)

        # With left-padding the assistant-prefix is the last token of
        # every sequence, so `[:, -1, :]` gives us the right slice.
        last_logits = logits[:, -1, :]
        yes_logits = last_logits[:, handle.yes_id]
        no_logits = last_logits[:, handle.no_id]
        # softmax over the two-element [no, yes] vector → P(yes).
        # `torch.nn.functional.softmax` is the publicly-exported entry
        # point (`torch.softmax` re-exports the same op from the private
        # `_C._VariableFunctions` namespace, which pyright flags).
        pair_logits = torch.stack([no_logits, yes_logits], dim=-1)
        probs = torch.nn.functional.softmax(pair_logits.float(), dim=-1)
        # `Tensor.tolist()` is typed as a bare `list` in torch's stubs;
        # `probs[:, 1]` is a 1-D float tensor, so cast the receiver to a
        # typed alias whose `tolist()` returns list[float].
        yes_col = cast(_FloatTensor, probs[:, 1])
        scores.extend(float(p) for p in yes_col.tolist())

    return scores
