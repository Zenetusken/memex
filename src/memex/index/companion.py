"""Companion-merge alignment core (ADR-0018, spec docs/specs/companion-merge.md §3).

The PURE, I/O-free heart of the merge: given a lecture TRANSCRIPT doc's chunks + their embeddings and
a SLIDE-DECK doc's chunks + their embeddings, align each transcript chunk to its best-matching slide
PAGE by cosine similarity — the MaViLS method's transcript-only modality: argmax + a NULL floor (an
off-slide tangent) + a cheap monotonic tie-break (lectures advance). Deterministic (embeddings in,
`AlignmentBlock`s out) ⇒ unit-testable with no GPU / no stores. The embed seam, the FTSStore read, the
sidecar store, and the CLI are the B2 increment; the eval-gated `/ask` augmentation is B4. None of this
touches the grounding path (the transcript + deck stay first-class grounded docs — HARD-gate-neutral).
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any

import numpy as np
from numpy.typing import NDArray

from memex.core.config import get_settings
from memex.core.errors import MemexError
from memex.core.types import AlignmentBlock, Chunk, CompanionAlignment
from memex.index.embed_prompts import (
    EMBED_QUERY_PROMPT_NAME,
    chunk_title,
    document_input,
    native_prompts_enabled,
)
from memex.index.fts_store import FTSStore
from memex.models.registry import get_registry

# Near-tie band for the monotonic tie-break: two slide pages whose cosine to a transcript chunk are
# within this are "tied", and we prefer the non-decreasing one (a lecture advances). Small by design —
# it NEVER overrides a clear argmax, only breaks genuine near-ties.
_TIE_EPSILON = 0.02


def cosine_matrix(
    t_emb: list[list[float]], p_emb: list[list[float]]
) -> NDArray[np.float32]:
    """The `(n_transcript × n_deck)` cosine-similarity matrix. EmbeddingGemma already L2-normalizes, but
    we re-normalize DEFENSIVELY so a fake/un-normalized test input still yields true cosine (dot of unit
    vectors). Either side empty → a correctly-shaped zero matrix."""
    if not t_emb or not p_emb:
        return np.zeros((len(t_emb), len(p_emb)), dtype=np.float32)
    t = np.asarray(t_emb, dtype=np.float32)
    p = np.asarray(p_emb, dtype=np.float32)
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-12)
    return t @ p.T


def align_blocks(
    transcript_chunks: list[Chunk],
    deck_chunks: list[Chunk],
    t_emb: list[list[float]],
    p_emb: list[list[float]],
    *,
    min_score: float,
    epsilon: float = _TIE_EPSILON,
) -> tuple[list[AlignmentBlock], int]:
    """PURE MaViLS-style alignment. Each transcript chunk → its best deck page by cosine ARGMAX, NULL
    below `min_score` (an off-slide tangent / no good slide), with a cheap MONOTONIC tie-break: among
    pages within `epsilon` of the best, prefer the smallest page `≥` the last assigned page (lectures
    advance) — never overriding a clear argmax, never forbidding a clear backward jump (the asymmetric-
    penalty DP is the §13 fast-follow). Returns `(blocks, null_count)`.

    `transcript_chunks` MUST be in TIME order (the monotonic tie-break assumes it); `FTSStore`
    `chunks_for_document` returns `char_start` order, which IS time order for a transcript. `t_emb` /
    `p_emb` must be parallel to their chunk lists."""
    if len(t_emb) != len(transcript_chunks):
        raise ValueError(f"t_emb ({len(t_emb)}) must parallel transcript_chunks ({len(transcript_chunks)})")
    if len(p_emb) != len(deck_chunks):
        raise ValueError(f"p_emb ({len(p_emb)}) must parallel deck_chunks ({len(deck_chunks)})")

    sim = cosine_matrix(t_emb, p_emb)
    blocks: list[AlignmentBlock] = []
    null_count = 0
    page_prev: int | None = None  # last assigned non-null page (for the monotonic tie-break)

    for i, tc in enumerate(transcript_chunks):
        if not deck_chunks:  # nothing to align to → all NULL
            blocks.append(AlignmentBlock(transcript_chunk_id=tc.chunk_id, time_range=tc.time_range, score=0.0))
            null_count += 1
            continue
        row = sim[i]
        best_j = int(np.argmax(row))
        best_score = float(row[best_j])
        if best_score < min_score:  # NULL — no slide is a good enough match
            blocks.append(
                AlignmentBlock(
                    transcript_chunk_id=tc.chunk_id,
                    time_range=tc.time_range,
                    deck_chunk_id=None,
                    deck_page=None,
                    score=best_score,
                )
            )
            null_count += 1
            continue
        chosen_j = best_j
        if page_prev is not None:  # monotonic tie-break among near-ties
            ceiling = best_score - epsilon
            forward_ties = [
                j
                for j in range(len(deck_chunks))
                if float(row[j]) >= ceiling
                and deck_chunks[j].page is not None
                and (deck_chunks[j].page or 0) >= page_prev
            ]
            if forward_ties:  # prefer the smallest forward page, breaking further ties by higher score
                chosen_j = min(forward_ties, key=lambda j: (deck_chunks[j].page or 0, -float(row[j])))
        dc = deck_chunks[chosen_j]
        blocks.append(
            AlignmentBlock(
                transcript_chunk_id=tc.chunk_id,
                time_range=tc.time_range,
                deck_chunk_id=dc.chunk_id,
                deck_page=dc.page,
                score=float(row[chosen_j]),
            )
        )
        if dc.page is not None:
            page_prev = dc.page

    return blocks, null_count


class CompanionMergeError(MemexError):
    """A companion-merge alignment couldn't be computed — a doc isn't indexed, or the deck lacks the
    per-page attribution the alignment needs. Not recoverable (a user-fix prerequisite)."""


async def _embed(inputs: list[str], *, prompt_name: str | None) -> list[list[float]]:
    """Batch-embed via the LIVE EmbeddingGemma, L2-normalized (so cosine == dot). The monkeypatchable
    GPU seam — tests fake THIS to run the orchestrator with no model. `prompt_name="query"` for the
    transcript narration (query-side); `None` for the deck (its inputs are pre-wrapped `document_input`
    strings, OR raw text when native prompts are off). Empty input → `[]`."""
    if not inputs:
        return []
    registry = get_registry()
    async with registry.use("embedder") as embedder:

        def _run() -> Any:
            return embedder.encode(
                inputs, prompt_name=prompt_name, normalize_embeddings=True, convert_to_numpy=True
            )

        arr = await asyncio.to_thread(_run)
    return [[float(x) for x in row] for row in arr]


async def compute_alignment(transcript_doc: str, deck_doc: str) -> CompanionAlignment:
    """Read both docs' INDEXED chunks (via FTSStore), embed (transcript query-side, deck document-side
    — the §3 committed asymmetry, both flipped together with `native_prompts`), and align
    (`align_blocks`). Raises `CompanionMergeError` if either doc is un-indexed or the deck has no page
    attribution. Deterministic given fixed embeddings ⇒ reproducible; carries the embed-recipe version
    so a recipe bump (or `--refresh`) re-aligns. HARD-gate-neutral: pure read of indexed chunks +
    embedder, no grounding-path touch."""
    settings = get_settings()
    async with AsyncExitStack() as stack:
        fstore = await FTSStore.open(settings.vault_path)
        stack.push_async_callback(fstore.close)
        t_chunks = await fstore.chunks_for_document(transcript_doc)
        d_chunks = await fstore.chunks_for_document(deck_doc)

    if not t_chunks:
        raise CompanionMergeError(
            "transcript has no indexed chunks — ingest + index it first",
            context={"transcript_doc": transcript_doc},
        )
    if not d_chunks:
        raise CompanionMergeError(
            "slide deck has no indexed chunks — ingest + index it first",
            context={"deck_doc": deck_doc},
        )
    if all(c.page is None for c in d_chunks):
        raise CompanionMergeError(
            "slide deck has no per-page attribution (a pre-2026-05-27 manifest); re-parse the deck "
            "(`memex parse <deck>`) so each chunk carries its slide page",
            context={"deck_doc": deck_doc},
        )

    use_prompt = native_prompts_enabled()
    if use_prompt:
        t_emb = await _embed([c.text for c in t_chunks], prompt_name=EMBED_QUERY_PROMPT_NAME)
        d_emb = await _embed(
            [document_input(chunk_title(c), c.text) for c in d_chunks], prompt_name=None
        )
    else:  # native prompts off (the A/B/revert path): both sides bare, same space
        t_emb = await _embed([c.text for c in t_chunks], prompt_name=None)
        d_emb = await _embed([c.text for c in d_chunks], prompt_name=None)

    blocks, null_count = align_blocks(
        t_chunks, d_chunks, t_emb, d_emb, min_score=settings.agents.companion_align_min_score
    )
    return CompanionAlignment(
        transcript_doc=transcript_doc,
        deck_doc=deck_doc,
        # Inlined recipe string (index/pipeline._embed_recipe_version is private — no cross-module
        # private import); MUST stay in lockstep with it.
        embedding_recipe_version="v1-gemma-prompts" if use_prompt else "v0",
        blocks=blocks,
        null_count=null_count,
    )
