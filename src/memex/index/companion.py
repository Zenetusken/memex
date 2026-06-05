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


def cosine_matrix(t_emb: list[list[float]], p_emb: list[list[float]]) -> NDArray[np.float32]:
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
    use_dp: bool = False,
    lambda_jump: float = 0.1,
    time_weight: float = 0.1,
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

    `use_dp` (ADR-0018 §13, OPT-IN, default OFF) swaps the greedy argmax+tie-break for a monotonic
    Viterbi (`_align_dp`): an asymmetric forward/backward jump penalty (`lambda_jump`) + a `start_s`
    time prior (`time_weight`). OFF ⇒ this function is byte-identical to before. NB the DP's
    below-floor→null is a SOFT (global-optimum) property at the shipped small `lambda_jump`, not the
    greedy path's hard per-chunk `best < min_score → null` (a large λ could assign a below-floor chunk
    to preserve context); benign while default-off + small-λ.

    `transcript_chunks` MUST be in TIME order (the monotonic tie-break assumes it); `FTSStore`
    `chunks_for_document` returns `char_start` order, which IS time order for a transcript. `t_emb` /
    `p_emb` must be parallel to their chunk lists."""
    if len(t_emb) != len(transcript_chunks):
        raise ValueError(
            f"t_emb ({len(t_emb)}) must parallel transcript_chunks ({len(transcript_chunks)})"
        )
    if len(p_emb) != len(deck_chunks):
        raise ValueError(f"p_emb ({len(p_emb)}) must parallel deck_chunks ({len(deck_chunks)})")

    sim = cosine_matrix(t_emb, p_emb)
    if use_dp:
        return _align_dp(
            transcript_chunks,
            deck_chunks,
            sim,
            min_score=min_score,
            lambda_jump=lambda_jump,
            time_weight=time_weight,
            keyframe_signal=keyframe_signal,
        )
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
            blocks.append(
                AlignmentBlock(transcript_chunk_id=tc.chunk_id, time_range=tc.time_range, score=0.0)
            )
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
            if (
                forward_ties
            ):  # prefer the smallest forward page, breaking further ties by higher score
                chosen_j = min(
                    forward_ties, key=lambda j: (deck_chunks[j].page or 0, -float(row[j]))
                )
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


def _align_dp(
    transcript_chunks: list[Chunk],
    deck_chunks: list[Chunk],
    sim: NDArray[np.float32],
    *,
    min_score: float,
    lambda_jump: float,
    time_weight: float,
    keyframe_signal: dict[str, tuple[str, int | None, float]] | None,
) -> tuple[list[AlignmentBlock], int]:
    """§13 monotonic Viterbi (ADR-0018) — the principled refinement of the greedy tie-break. PURE.

    The state after chunk `i` is the LAST ASSIGNED slide page (a context that CARRIES through NULL
    tangents), or START before the first assignment. At each chunk the path either ASSIGNS a page
    (emission = the chunk's best cosine to that page MINUS a `time_weight` prior on the gap between
    the chunk's lecture-time fraction `start_s/T` and the page's deck fraction; transition value =
    `-lambda_jump ×` jump units, forward 1× / backward 2× / stay 0 — lectures advance, revisits cost
    more) or is NULL (a flat `min_score` floor; the context is unchanged). A keyframe-PRIMARY chunk is
    a FIXED anchor: it forces the context to its page (when that page is in the deck) and emits its
    keyframe block verbatim. A Viterbi backtrace from the best final context yields the globally
    optimal monotonic-ish path — improving the per-chunk greedy on revisit-heavy / time-coherent runs.
    """
    n = len(transcript_chunks)
    kf = keyframe_signal or {}
    pages = sorted({c.page for c in deck_chunks if c.page is not None})
    if n == 0 or not pages:  # nothing to align to (compute_alignment guards this) → all NULL
        blocks = [
            AlignmentBlock(transcript_chunk_id=tc.chunk_id, time_range=tc.time_range, score=0.0)
            for tc in transcript_chunks
        ]
        return blocks, n

    pidx = {p: k for k, p in enumerate(pages)}
    page_count = len(pages)
    start = page_count  # sentinel "no page yet" context (index page_count)
    neg = float("-inf")

    # Best cosine + the deck chunk achieving it, per (transcript chunk i, page index k).
    page_score = [[neg] * page_count for _ in range(n)]
    page_cid: list[list[str | None]] = [[None] * page_count for _ in range(n)]
    for j, dc in enumerate(deck_chunks):
        if dc.page is None:
            continue
        k = pidx[dc.page]
        for i in range(n):
            s = float(sim[i, j])
            if s > page_score[i][k]:
                page_score[i][k] = s
                page_cid[i][k] = dc.chunk_id

    ends = [tc.time_range[1] for tc in transcript_chunks if tc.time_range is not None]
    t_total = max(ends) if ends else 0.0

    def _time_penalty(i: int, k: int) -> float:
        tr = transcript_chunks[i].time_range
        if tr is None or t_total <= 0.0 or page_count <= 1:
            return 0.0
        expected = (tr[0] / t_total) * (page_count - 1)
        return time_weight * abs(expected - k) / (page_count - 1)

    def _jump(c: int, k: int) -> float:
        """Transition VALUE added (a non-positive cost). START → any page is free."""
        if c == start:
            return 0.0
        pc, pk = pages[c], pages[k]
        if pk == pc:
            return 0.0
        return -lambda_jump * (1.0 if pk > pc else 2.0)

    def _best_prev(prev_v: list[float], k: int) -> tuple[int, float]:
        best_c, best_val = start, neg
        for c in range(page_count + 1):
            if prev_v[c] == neg:
                continue
            val = prev_v[c] + _jump(c, k)
            if val > best_val:
                best_val, best_c = val, c
        return best_c, best_val

    # Viterbi. `bp[i][ctx] = (prev_ctx, (action, k))`; START seeds the pre-first-chunk context.
    prev_v = [neg] * (page_count + 1)
    prev_v[start] = 0.0
    bp: list[list[tuple[int, tuple[str, int]] | None]] = []

    for i, tc in enumerate(transcript_chunks):
        v = [neg] * (page_count + 1)
        row_bp: list[tuple[int, tuple[str, int]] | None] = [None] * (page_count + 1)
        kfe = kf.get(tc.chunk_id)
        anchor_k = (
            pidx[kfe[1]] if kfe is not None and kfe[1] is not None and kfe[1] in pidx else None
        )

        if anchor_k is not None:  # FIXED anchor with a real deck page → force the context
            best_c, best_val = _best_prev(prev_v, anchor_k)
            v[anchor_k] = best_val  # forced emission is a constant → omit
            row_bp[anchor_k] = (best_c, ("anchor", anchor_k))
        elif (
            kfe is not None
        ):  # anchor whose page is unattributed → emit verbatim, carry the context
            for c in range(page_count + 1):
                if prev_v[c] != neg and prev_v[c] > v[c]:
                    v[c] = prev_v[c]
                    row_bp[c] = (c, ("anchor_noctx", -1))
        else:
            for c in range(page_count + 1):  # NULL: carry each context at the flat floor
                if prev_v[c] == neg:
                    continue
                val = prev_v[c] + min_score
                if val > v[c]:
                    v[c] = val
                    row_bp[c] = (c, ("null", -1))
            for k in range(page_count):  # ASSIGN page k
                if page_score[i][k] == neg:
                    continue
                best_c, best_val = _best_prev(prev_v, k)
                total = best_val + page_score[i][k] - _time_penalty(i, k)
                if total > v[k]:
                    v[k] = total
                    row_bp[k] = (best_c, ("assign", k))
        bp.append(row_bp)
        prev_v = v

    last_ctx = max(range(page_count + 1), key=lambda c: prev_v[c])
    actions: list[tuple[str, int]] = []
    ctx = last_ctx
    for i in range(n - 1, -1, -1):
        entry = bp[i][ctx]
        if entry is None:  # unreachable (NULL is always available, so this shouldn't happen)
            actions.append(("null", -1))
            continue
        prev_ctx, action = entry
        actions.append(action)
        ctx = prev_ctx
    actions.reverse()

    blocks: list[AlignmentBlock] = []
    null_count = 0
    for i, tc in enumerate(transcript_chunks):
        kind, k = actions[i]
        if kind == "assign":
            blocks.append(
                AlignmentBlock(
                    transcript_chunk_id=tc.chunk_id,
                    time_range=tc.time_range,
                    deck_chunk_id=page_cid[i][k],
                    deck_page=pages[k],
                    score=page_score[i][k],
                )
            )
        elif kind in ("anchor", "anchor_noctx"):
            kfe = kf[tc.chunk_id]
            blocks.append(
                AlignmentBlock(
                    transcript_chunk_id=tc.chunk_id,
                    time_range=tc.time_range,
                    deck_chunk_id=kfe[0],
                    deck_page=kfe[1],
                    score=kfe[2],
                )
            )
        else:  # null — record the best (insufficient / un-chosen) cosine for provenance
            best = max(page_score[i])
            blocks.append(
                AlignmentBlock(
                    transcript_chunk_id=tc.chunk_id,
                    time_range=tc.time_range,
                    deck_chunk_id=None,
                    deck_page=None,
                    score=best if best != neg else 0.0,
                )
            )
            null_count += 1
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
        use_dp=settings.agents.companion_align_dp_enabled,
        lambda_jump=settings.agents.companion_dp_lambda_jump,
        time_weight=settings.agents.companion_dp_time_weight,
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
