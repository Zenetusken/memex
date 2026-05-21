"""Cross-encoder reranker — bge-reranker-v2-m3 by default.

Loaded on first call via `ModelRegistry.use("reranker")`. The reranker
runs CPU-or-GPU depending on what `sentence-transformers` chooses; we
don't pin a device here — the registry can configure it later via the
hardware settings.
"""

from __future__ import annotations

import asyncio
import os
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

    # batch_size=8 is the safe default on a 12 GB rig running
    # bge-reranker-v2-m3 alongside vLLM-Qwen3-8B-AWQ. Originally 64
    # (CUDA audit, Docling-chunk baseline); lowered once
    # PyMuPDF-extracted chunks (denser native text per pair) started
    # OOMing the reranker's attention even at top_k=10. Even
    # batch_size=16 OOMs on the reference RTX 4070 with the 8B
    # orchestrator resident — see the empirical run after the chunker
    # tuning commit. Bigger rigs, smaller orchestrators, or the
    # planned Qwen3-Reranker-0.6B swap (P2.1) can push back up via
    # MEMEX_RERANK_BATCH_SIZE for higher rerank throughput.
    try:
        batch_size = int(os.environ.get("MEMEX_RERANK_BATCH_SIZE", "8"))
    except ValueError:
        batch_size = 8
    batch_size = max(1, batch_size)

    registry = get_registry()
    async with registry.use("reranker") as reranker:
        pairs = [(query, c.text) for c in candidates]

        def _predict() -> Any:
            return reranker.predict(
                pairs,
                batch_size=batch_size,
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
