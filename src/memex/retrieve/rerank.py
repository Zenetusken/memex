"""Cross-encoder reranker — bge-reranker-v2-m3 by default.

Loaded on first call via `ModelRegistry.use("reranker")`. The reranker
runs CPU-or-GPU depending on what `sentence-transformers` chooses; we
don't pin a device here — the registry can configure it later via the
hardware settings.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from memex.core.types import Chunk
from memex.models.registry import get_registry

logger = structlog.get_logger(__name__)


async def cross_encoder_rerank(
    query: str,
    candidates: list[Chunk],
    top_k: int = 10,
) -> list[Chunk]:
    """Rescore `candidates` against `query`, return top `top_k` by rerank score."""
    if not candidates:
        return []

    log = logger.bind(candidates=len(candidates), top_k=top_k)
    log.info("rerank.start")

    registry = get_registry()
    async with registry.use("reranker") as reranker:
        pairs = [(query, c.text) for c in candidates]

        def _predict() -> Any:
            # batch_size=64 tuned for bge-reranker-v2-m3 on Ada per the
            # CUDA audit; sentence-transformers' default of 32 is too low
            # for the typical top-50 candidate pool.
            return reranker.predict(
                pairs,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        scores = await asyncio.to_thread(_predict)

    ranked = sorted(
        zip(candidates, (float(s) for s in scores), strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    out = [
        c.model_copy(update={"rerank_score": score})
        for c, score in ranked[:top_k]
    ]
    log.info("rerank.done", returned=len(out))
    return out
