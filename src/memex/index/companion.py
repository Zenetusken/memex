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
from collections.abc import Mapping
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

# Title used to embed a keyframe-OCR text DOC-side (ADR-0018 §13), so a frame's slide content sits in
# the same space as the deck chunks (which embed `document_input(heading, text)`). The frame has no
# heading; the OCR text dominates the embedding, so a neutral constant is sufficient.
_KEYFRAME_TITLE = "slide"


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
    keyframe_signal: dict[str, tuple[str, int | None, float]] | None = None,
) -> tuple[list[AlignmentBlock], int]:
    """PURE MaViLS-style alignment. Each transcript chunk → its best deck page by cosine ARGMAX, NULL
    below `min_score` (an off-slide tangent / no good slide), with a cheap MONOTONIC tie-break: among
    pages within `epsilon` of the best, prefer the smallest page `≥` the last assigned page (lectures
    advance) — never overriding a clear argmax, never forbidding a clear backward jump (the asymmetric-
    penalty DP is the §13 fast-follow). Returns `(blocks, null_count)`.

    `keyframe_signal` (ADR-0018 §13, video lectures) maps a transcript `chunk_id` → its
    `(deck_chunk_id, deck_page, score)` derived from the VIDEO FRAME shown during it (OCR → cosine to
    the deck), pre-filtered to entries above the keyframe floor by `compute_alignment`. When a chunk has
    a keyframe entry it is PRIMARY (used verbatim, no null floor — the floor already passed); otherwise
    the chunk falls back to the transcript-text argmax below. The keyframe page still advances
    `page_prev`, so a following transcript-fallback chunk tie-breaks forward from it.

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
        kf = keyframe_signal.get(tc.chunk_id) if keyframe_signal else None
        if kf is not None:  # keyframe-PRIMARY (already above the keyframe floor) → use it verbatim
            kf_deck_chunk_id, kf_deck_page, kf_score = kf
            blocks.append(
                AlignmentBlock(
                    transcript_chunk_id=tc.chunk_id,
                    time_range=tc.time_range,
                    deck_chunk_id=kf_deck_chunk_id,
                    deck_page=kf_deck_page,
                    score=kf_score,
                )
            )
            if kf_deck_page is not None:
                page_prev = kf_deck_page
            continue
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


async def _keyframe_signal_from_texts(
    t_chunks: list[Chunk],
    d_chunks: list[Chunk],
    d_emb: list[list[float]],
    keyframe_texts: Mapping[str, str],
    *,
    use_prompt: bool,
    floor: float,
) -> dict[str, tuple[str, int | None, float]]:
    """Build the keyframe signal (ADR-0018 §13): embed each transcript chunk's FRAME-OCR text DOC-side
    (so it shares the deck's space) and cosine-match it to the deck pages. A chunk's best deck page at
    cosine `≥ floor` becomes its PRIMARY slide `(deck_chunk_id, deck_page, score)`; below the floor
    (a live demo / off-slide frame) the chunk is omitted → it falls back to its transcript-text signal
    in `align_blocks`. Reuses the SAME embedder + `cosine_matrix` as the transcript-text path."""
    kf_ids = [c.chunk_id for c in t_chunks if c.chunk_id in keyframe_texts]
    if not kf_ids:
        return {}
    if use_prompt:
        kf_inputs = [document_input(_KEYFRAME_TITLE, keyframe_texts[cid]) for cid in kf_ids]
        kf_emb = await _embed(kf_inputs, prompt_name=None)
    else:  # native prompts off: bare text, same space as the bare deck embeddings
        kf_emb = await _embed([keyframe_texts[cid] for cid in kf_ids], prompt_name=None)

    sim = cosine_matrix(kf_emb, d_emb)
    signal: dict[str, tuple[str, int | None, float]] = {}
    for k, cid in enumerate(kf_ids):
        row = sim[k]
        best_j = int(np.argmax(row))
        best = float(row[best_j])
        if best >= floor:
            dc = d_chunks[best_j]
            signal[cid] = (dc.chunk_id, dc.page, best)
    return signal


async def compute_alignment(
    transcript_doc: str,
    deck_doc: str,
    *,
    keyframe_texts: Mapping[str, str] | None = None,
) -> CompanionAlignment:
    """Read both docs' INDEXED chunks (via FTSStore), embed (transcript query-side, deck document-side
    — the §3 committed asymmetry, both flipped together with `native_prompts`), and align
    (`align_blocks`). Raises `CompanionMergeError` if either doc is un-indexed or the deck has no page
    attribution. Deterministic given fixed embeddings ⇒ reproducible; carries the embed-recipe version
    so a recipe bump (or `--refresh`) re-aligns. HARD-gate-neutral: pure read of indexed chunks +
    embedder, no grounding-path touch.

    `keyframe_texts` (ADR-0018 §13, `link-slides --use-video`): a `{transcript_chunk_id: frame_ocr_text}`
    map produced by `parse/keyframe_ocr.ocr_frames_for_chunks` (the CLI wires parse→index). When given,
    each covered chunk's slide is computed from its VIDEO FRAME (the stronger MaViLS signal) and is
    PRIMARY above the keyframe floor; uncovered / below-floor chunks fall back to the transcript-text
    cosine. `None` ⇒ the transcript-only path, byte-identical to before."""
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

    keyframe_signal: dict[str, tuple[str, int | None, float]] = {}
    if keyframe_texts:
        keyframe_signal = await _keyframe_signal_from_texts(
            t_chunks,
            d_chunks,
            d_emb,
            keyframe_texts,
            use_prompt=use_prompt,
            floor=settings.agents.companion_keyframe_min_score,
        )

    blocks, null_count = align_blocks(
        t_chunks,
        d_chunks,
        t_emb,
        d_emb,
        min_score=settings.agents.companion_align_min_score,
        keyframe_signal=keyframe_signal or None,
    )
    # Inlined recipe string (index/pipeline._embed_recipe_version is private — no cross-module private
    # import); MUST stay in lockstep with it. A `+keyframe` suffix records that the VIDEO-frame signal
    # produced this alignment (provenance; distinguishes it from a transcript-only re-run of the pair).
    recipe = "v1-gemma-prompts" if use_prompt else "v0"
    if keyframe_signal:
        recipe += "+keyframe"
    return CompanionAlignment(
        transcript_doc=transcript_doc,
        deck_doc=deck_doc,
        embedding_recipe_version=recipe,
        blocks=blocks,
        null_count=null_count,
    )
