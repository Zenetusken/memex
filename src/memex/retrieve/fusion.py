"""Reciprocal Rank Fusion — combines multiple ranked lists into one.

Each ranked list contributes `1 / (k + rank)` to a chunk's fused score
(where `rank` is 1-indexed). Chunks present in more than one list
naturally bubble up; the `k` constant dampens the contribution of
low-ranked items. We use the well-studied default `k=60`.
"""

from __future__ import annotations

from memex.core.types import Chunk

DEFAULT_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[Chunk]],
    *,
    k: int = DEFAULT_K,
) -> list[Chunk]:
    """Fuse `rankings` into a single descending-score list.

    Chunks are identified by `chunk_id`. The first occurrence's metadata
    (text, document_title, page, etc.) is preserved; only the score is
    re-derived from rank positions.
    """
    fused: dict[str, tuple[Chunk, float]] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            contribution = 1.0 / (k + rank)
            if chunk.chunk_id in fused:
                first, score = fused[chunk.chunk_id]
                fused[chunk.chunk_id] = (first, score + contribution)
            else:
                fused[chunk.chunk_id] = (chunk, contribution)

    ordered = sorted(fused.values(), key=lambda pair: pair[1], reverse=True)
    return [chunk.model_copy(update={"score": score}) for chunk, score in ordered]
