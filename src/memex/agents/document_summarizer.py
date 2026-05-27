"""Doc-type-aware, structured, grounded "Summarize this doc" (ADR-0008).

A standalone async orchestration (like `agents/synthesize.py`, NOT a LangGraph —
map-reduce is linear, grounding is reusable functions). It supersedes the
free-form `synthesize` baseline for the general case.

The governing rule (the "baseline rule"): QUALITY IS NEVER COMPROMISED and is
INDEPENDENT of the co-residence mode. The strategy is chosen by the DOCUMENT
(size/type), never the mode — `fast` and `full` produce the same-quality grounded
summary; the mode only changes speed/context-capacity. Every emitted key-point is
grounded against its source chunk (reusing `answering.verify`'s machinery) or
dropped; a zero-grounded result refuses. The no-hallucination HARD gate, extended
to summaries.

Pipeline: load the doc's stored chunks in reading order (`FTSStore.chunks_for_
document` — boundary-correct: `agents/` may import `index/`, not `vault/`) → group
by heading-section → MAP each section to a `SectionSummary` (bounded structured
output) → GROUND its key-points → REDUCE the section digests to a doc abstract →
compose a `FinalResponse`. Output is `maxItems`-bounded lists of short strings
(xgrammar enforces list bounds, NOT string `maxLength` — the 2026-05-27 baseline),
so it can't run away the way a single free-form summary did.
"""

from __future__ import annotations

from typing import Annotated, Literal

import structlog
import ulid
from pydantic import Field, create_model

from memex.agents.answering import (
    CitedClaim,
    DocAbstract,
    DraftAnswer,
    FinalResponse,
    SectionSummary,
    VerificationResult,
)
from memex.core.config import get_settings
from memex.core.errors import MemexError, ModelCallError
from memex.core.types import Chunk
from memex.core.wikilinks import format_wikilink
from memex.index.fts_store import FTSStore
from memex.models.client import complete_structured
from memex.observability import bind_run_context, clear_run_context
from memex.prompts import render_prompt

logger = structlog.get_logger(__name__)

DocRoute = Literal["short", "long"]

# Tunable length/detail knob (the user's fine-tuning variable). Each level maps to
# the natural-language length guidance threaded into the MAP/REDUCE prompts — the
# abstract (whole-doc overview) and the per-section digest. The schema's
# `maxItems`/`max_length` bounds are the hard safety net; this tunes the REQUESTED
# verbosity within them.
SummaryDetail = Literal["brief", "standard", "detailed"]
_DETAIL_GUIDANCE: dict[SummaryDetail, dict[str, str]] = {
    "brief": {"abstract": "1-2 sentences", "digest": "1 concise sentence"},
    "standard": {"abstract": "2-4 sentences", "digest": "1-3 sentences"},
    "detailed": {
        "abstract": "a thorough paragraph of 5-8 sentences",
        "digest": "2-4 sentences",
    },
}

# A doc whose total chunk text fits comfortably in one window summarizes in a
# single structured pass; larger docs map-reduce. Char-based (no in-process
# tokenizer); ~14k chars ≈ ~4k tokens, well inside even the fast-mode 6,144 window
# alongside the prompt + bounded output.
_SHORT_DOC_CHARS = 14_000
# Cap per-section chunks shown to the MAP call (a pathologically large section is
# an edge case; the prompt also truncates each chunk to 1,800 chars).
_MAX_SECTION_CHUNKS = 16
# The prompt template's per-chunk `truncate(N)` — kept in sync here so the input
# budget below can predict the real prompt size.
_CHUNK_TRUNCATE_CHARS = 1_800
# Hard cap on the chunk-text a SINGLE map/ground call may show. Sized to fit the
# SMALLEST co-residence window (fast = 6,144 tokens): ~12k chars ≈ ~3k tokens,
# leaving room for the prompt scaffold + the bounded output. This is the load-
# bearing guarantee behind the "baseline rule": because the per-call input is
# bounded to fit fast, it is IDENTICAL in fast and full mode, so the summary's
# quality is independent of the mode (the mode only changes speed/capacity, never
# what the summarizer sees). A section larger than this is truncated identically
# in both modes; sub-splitting a huge section across multiple MAP calls (so none
# of its content is dropped) is a deferred refinement (ADR-0008).
_MAX_SECTION_INPUT_CHARS = 12_000
# Doc-level key-point cap (mirrors DocSummary intent); section list cap.
_MAX_DOC_KEY_POINTS = 12
_MAX_SECTIONS = 40
# Cap the digests fed to the single REDUCE call so its prompt also fits the fast
# window (≈24 digests × ~650 chars ≈ 3.9k tokens + the bounded output stays well
# under 6,144). A doc with more sections still abstracts from a representative
# head; its per-section digests remain on `FinalResponse.sections` in full.
_REDUCE_MAX_SECTIONS = 24
# Output reservation for the small GROUND/REDUCE calls (a verification index-list /
# a short abstract) — kept tight so prompt + output clears the fast window.
_VERIFY_MAX_TOKENS = 768
_REDUCE_MAX_TOKENS = 1024


def _group_sections(chunks: list[Chunk]) -> list[tuple[str, list[Chunk]]]:
    """Group chunks (already in reading order) into heading-sections, keyed by the
    deepest `heading_path` entry (or the document title for a flat doc), preserving
    first-seen order."""
    groups: dict[str, list[Chunk]] = {}
    order: list[str] = []
    for c in chunks:
        title = c.heading_path[-1] if c.heading_path else (c.document_title or c.document_id)
        if title not in groups:
            groups[title] = []
            order.append(title)
        groups[title].append(c)
    return [(t, groups[t]) for t in order]


def _classify_route(chunks: list[Chunk], sections: list[tuple[str, list[Chunk]]]) -> DocRoute:
    """v1 doc-type router: `short` (one structured pass) vs `long` (map-reduce).

    Based on total text size + section count. (Tabular / deck / scan specialisations
    are deferred — they route as `long` for now; ADR-0008.)
    """
    total_chars = sum(len(c.text) for c in chunks)
    if total_chars <= _SHORT_DOC_CHARS or len(sections) <= 1:
        return "short"
    return "long"


def _bound_section_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Take a section's chunks up to a fixed char budget (and the chunk-count cap)
    so a single MAP/GROUND call fits even the SMALLEST window (fast, 6,144). This is
    what makes quality independent of the co-residence mode: the per-call input is
    the same in fast and full. Always keeps at least the first chunk (a lone giant
    chunk is truncated by the prompt's per-chunk `truncate`, identically per mode)."""
    out: list[Chunk] = []
    total = 0
    for c in chunks[:_MAX_SECTION_CHUNKS]:
        clen = min(len(c.text), _CHUNK_TRUNCATE_CHARS)  # mirrors the prompt truncate
        if out and total + clen > _MAX_SECTION_INPUT_CHARS:
            break
        out.append(c)
        total += clen
    return out


async def _map_section(
    section_title: str,
    chunks: list[Chunk],
    instruction: str | None,
    max_output_tokens: int,
    digest_guidance: str,
) -> tuple[SectionSummary | None, int]:
    """Summarize ONE section into a `SectionSummary` (digest + cited key-points).
    Returns `(None, tokens)` if the call fails (a single bad section must not sink
    the whole summary)."""
    prompt = render_prompt(
        "summarize_section",
        section_title=section_title,
        chunks=chunks,  # caller has already bounded the set to fit the fast window
        instruction=instruction or "",
        digest_guidance=digest_guidance,
    )
    try:
        section, tokens = await complete_structured(
            prompt=prompt,
            schema=SectionSummary,
            max_tokens=max_output_tokens,
            prompt_tag="summarize_section@v1",
        )
    except ModelCallError as e:
        logger.bind(section=section_title).warning("summarize.section_failed", error=str(e)[:160])
        return None, 0
    return section, tokens


def _bounded_verification(n: int) -> type[VerificationResult]:
    """Per-call `VerificationResult` with the index lists bounded to `n`
    (mirrors `answering.verify`). xgrammar enforces the list bounds."""
    return create_model(
        "VerificationResult",
        __base__=VerificationResult,
        grounded=(Annotated[list[int], Field(max_length=n)], Field(default_factory=list)),
        ungrounded=(Annotated[list[int], Field(max_length=n)], Field(default_factory=list)),
        ungrounded_reasons=(
            Annotated[list[Annotated[str, Field(max_length=250)]], Field(max_length=n)],
            Field(default_factory=list),
        ),
    )


async def _ground_points(
    digest: str, key_points: list[CitedClaim], chunks: list[Chunk]
) -> tuple[list[CitedClaim], int]:
    """Keep only the key-points the verifier confirms are supported by their cited
    chunk — reusing `verify_grounding/v2` exactly (a section's key-points ARE
    `CitedClaim`s, so we wrap them in a `DraftAnswer` and run the same prompt).
    Conservative: a point survives only if explicitly grounded (missing → dropped)."""
    if not key_points:
        return [], 0
    # DraftAnswer.summary is bounded to 300; the digest is only context for the
    # verifier (it grounds the CLAIMS), so truncating is harmless.
    draft = DraftAnswer(summary=digest[:300], claims=key_points)
    chunk_by_id = {c.chunk_id: c for c in chunks}
    prompt = render_prompt("verify_grounding", draft=draft, chunk_by_id=chunk_by_id)
    n = len(key_points)
    try:
        bounded, tokens = await complete_structured(
            prompt=prompt,
            schema=_bounded_verification(n),
            max_tokens=_VERIFY_MAX_TOKENS,
            prompt_tag="verify_grounding@v2",
        )
    except ModelCallError as e:
        # If grounding itself fails, drop the section's points (never ship ungrounded).
        logger.warning("summarize.grounding_failed", error=str(e)[:160])
        return [], 0
    grounded_idx = [i for i in bounded.grounded if 0 <= i < n]
    ungrounded_idx = {i for i in bounded.ungrounded if 0 <= i < n}
    kept = [key_points[i] for i in grounded_idx if i not in ungrounded_idx]
    return kept, tokens


async def _reduce(
    title: str,
    sections: list[SectionSummary],
    instruction: str | None,
    max_output_tokens: int,
    abstract_guidance: str,
) -> tuple[str, int]:
    """Synthesize the per-section digests into a whole-document abstract."""
    prompt = render_prompt(
        "summarize_reduce",
        title=title,
        sections=sections[:_REDUCE_MAX_SECTIONS],  # keep the REDUCE prompt within the fast window
        instruction=instruction or "",
        abstract_guidance=abstract_guidance,
    )
    try:
        overview, tokens = await complete_structured(
            prompt=prompt,
            schema=DocAbstract,
            max_tokens=min(max_output_tokens, _REDUCE_MAX_TOKENS),
            prompt_tag="summarize_reduce@v1",
        )
    except ModelCallError as e:
        logger.warning("summarize.reduce_failed", error=str(e)[:160])
        return "", 0
    return overview.abstract, tokens


async def summarize_document(
    doc_id: str,
    *,
    instruction: str | None = None,
    detail: SummaryDetail = "standard",
    max_output_tokens: int = 2048,
    token_budget: int = 120_000,
) -> FinalResponse:
    """Produce a structured, GROUNDED summary of one document (ADR-0008).

    `instruction` optionally focuses the summary. `detail` tunes length/verbosity
    (`brief` / `standard` / `detailed` — threaded into the MAP/REDUCE prompts via
    `_DETAIL_GUIDANCE`). `max_output_tokens` bounds each call; `token_budget` caps
    the whole map-reduce's total token traffic (a typical doc completes within it,
    bounded ultimately by `_MAX_SECTIONS`; a pathologically long doc stops early —
    a v1 limit, tunable per call). Returns a
    `FinalResponse`: `summary`=the abstract, `claims`=grounded doc-level
    key-points, `sections`=per-section digests, `wikilinks`/`used_chunks` for
    "Sources". A zero-grounded doc refuses.
    """
    correlation_id = str(ulid.ULID())
    clear_run_context()
    bind_run_context(correlation_id, query_preview=f"summarize {doc_id}")
    log = logger.bind(node="summarize", doc_id=doc_id)
    try:
        settings = get_settings()
        store = await FTSStore.open(settings.vault_path)
        try:
            chunks = await store.chunks_for_document(doc_id)
        finally:
            await store.close()

        if not chunks:
            log.info("summarize.no_chunks")
            return FinalResponse(
                answered=False,
                refusal_reason=f"No indexed content for document {doc_id!r}.",
                artifact_scope_doc_ids=[doc_id],
                correlation_id=correlation_id,
                tokens_used=0,
                nodes_traversed=1,
                regenerate_attempts=0,
            )

        sections = _group_sections(chunks)
        route = _classify_route(chunks, sections)
        doc_title = chunks[0].document_title or doc_id
        groups = [(doc_title, chunks)] if route == "short" else sections[:_MAX_SECTIONS]
        guidance = _DETAIL_GUIDANCE[detail]
        # Short route: the single MAP pass IS the whole-doc summary, so its digest
        # carries the (longer) abstract guidance; long route: sections get the
        # tighter digest guidance and the REDUCE produces the abstract.
        map_guidance = guidance["abstract"] if route == "short" else guidance["digest"]
        log.info(
            "summarize.start",
            chunks=len(chunks),
            sections=len(sections),
            route=route,
            detail=detail,
        )

        tokens_total = 0
        section_summaries: list[SectionSummary] = []
        for title, sec_chunks in groups:
            if tokens_total > token_budget:
                log.info("summarize.budget_exhausted", done=len(section_summaries))
                break
            # Bound the section's chunks ONCE to fit the smallest (fast) window, and
            # feed the SAME set to MAP and GROUND — so the input (hence the quality)
            # is identical in fast and full mode (the baseline rule).
            map_chunks = _bound_section_chunks(sec_chunks)
            mapped, t_map = await _map_section(
                title, map_chunks, instruction, max_output_tokens, map_guidance
            )
            tokens_total += t_map
            if mapped is None:
                continue
            grounded, t_g = await _ground_points(mapped.digest, mapped.key_points, map_chunks)
            tokens_total += t_g
            section_summaries.append(
                SectionSummary(
                    section_title=mapped.section_title or title,
                    digest=mapped.digest,
                    key_points=grounded,
                )
            )

        # Doc-level key-points = the grounded section points (reading order, capped).
        doc_points = [kp for ss in section_summaries for kp in ss.key_points][:_MAX_DOC_KEY_POINTS]
        if not doc_points:
            log.info("summarize.zero_grounded")
            return FinalResponse(
                answered=False,
                refusal_reason=(
                    "No grounded key-points could be extracted — refusing to ship an "
                    "ungrounded summary. The document may be too sparse, or too "
                    "large/dense for this route."
                ),
                sections=section_summaries,
                artifact_scope_doc_ids=[doc_id],
                correlation_id=correlation_id,
                tokens_used=tokens_total,
                nodes_traversed=len(section_summaries) + 1,
                regenerate_attempts=0,
            )

        # REDUCE → abstract (short route: the single section digest IS the abstract).
        if route == "short":
            abstract = section_summaries[0].digest if section_summaries else ""
        else:
            abstract, t_r = await _reduce(
                doc_title, section_summaries, instruction, max_output_tokens, guidance["abstract"]
            )
            tokens_total += t_r
        if not abstract:
            # Fallback: synthesize a headline from the grounded points (never empty).
            abstract = " ".join(kp.claim for kp in doc_points[:3])

        used_ids = {kp.source_chunk_id for kp in doc_points}
        used_chunks = [c for c in chunks if c.chunk_id in used_ids]
        wikilinks: list[str] = []
        seen: set[str] = set()
        for c in used_chunks:
            wl = format_wikilink(c.document_id, c.heading_path[-1] if c.heading_path else None)
            if wl not in seen:
                seen.add(wl)
                wikilinks.append(wl)

        log.info(
            "summarize.done",
            sections=len(section_summaries),
            grounded_points=len(doc_points),
            tokens_used=tokens_total,
        )
        return FinalResponse(
            answered=True,
            summary=abstract,
            claims=doc_points,
            used_chunks=used_chunks,
            wikilinks=wikilinks,
            sections=section_summaries,
            artifact_scope_doc_ids=[doc_id],
            correlation_id=correlation_id,
            tokens_used=tokens_total,
            nodes_traversed=len(section_summaries) + 1,
            regenerate_attempts=0,
        )
    except MemexError:
        raise
    finally:
        clear_run_context()
