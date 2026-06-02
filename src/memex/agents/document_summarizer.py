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

import asyncio
import re
import sqlite3
from collections.abc import Callable
from contextlib import AsyncExitStack
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import structlog
import ulid

from memex.agents.answering import (
    CitedClaim,
    DocAbstract,
    FinalResponse,
    ReportConfidence,
    ReportStructure,
    SectionSummary,
    repair_claim_chunk_ids,
)
from memex.agents.grounding import ground_claims
from memex.agents.table_sql import coerce_number
from memex.core.config import get_settings
from memex.core.errors import MemexError, ModelCallError
from memex.core.types import Chunk, StoredTable
from memex.core.wikilinks import format_wikilink
from memex.index.fts_store import FTSStore
from memex.index.table_store import TableStore
from memex.models.client import complete_structured, inference_override
from memex.observability import bind_run_context, clear_run_context
from memex.prompts import render_prompt

logger = structlog.get_logger(__name__)

DocRoute = Literal["short", "long"]

# Tunable length/detail knob (the user's fine-tuning variable). Each level maps to
# the natural-language length guidance threaded into the MAP/REDUCE prompts — the
# abstract (whole-doc overview) and the per-section digest. The schema's
# `maxItems`/`max_length` bounds are the hard safety net; this tunes the REQUESTED
# verbosity within them.
SummaryDetail = Literal["brief", "standard", "detailed", "report"]
_DETAIL_GUIDANCE: dict[SummaryDetail, dict[str, str]] = {
    "brief": {"abstract": "1-2 sentences", "digest": "1 concise sentence"},
    "standard": {"abstract": "2-4 sentences", "digest": "1-3 sentences"},
    "detailed": {
        "abstract": "a thorough 5-8 sentences",
        "digest": "2-4 sentences",
    },
    # `report` is the multi-paragraph route (ADR-0010): the abstract guidance sizes
    # ONE paragraph per batch of sections (a HIERARCHICAL reduce stitches the
    # paragraphs), and the per-section digest is rich (like `detailed`) so each
    # paragraph has substance to synthesize from.
    "report": {
        "abstract": "4-8 sentences",
        "digest": "2-4 sentences",
    },
}

# A doc whose total chunk text fits comfortably in one window summarizes in a
# single structured pass; larger docs map-reduce. Char-based (no in-process
# tokenizer); ~14k chars ≈ ~4k tokens, well inside even the fast-mode 6,144 window
# alongside the prompt + bounded output.
_SHORT_DOC_CHARS = 14_000
# Deck route (ADR-0008), realized as SECTION-PACKING: a slide deck makes MANY TINY
# heading-sections (one per slide), so per-section MAP digests come out thin (echoing
# the slide title). When a doc has many sections AND most are slide-sized, the long
# route PACKS adjacent sections up to the window budget so each MAP call digests a
# substantive span (any VLM-transcribed figure text rides along inline → figure-aware
# for free). This is doc-type-agnostic: it fires on slide decks AND bullet-heavy
# technical docs (both have thin sections), while a paper / standard with substantive
# sections (median well above the tiny threshold) keeps its own per-section digests.
_PACK_MIN_SECTIONS = 12
_PACK_TINY_SECTION_CHARS = 900
_PACK_TINY_FRACTION = 0.6
# Cap per-section chunks shown to the MAP call (a pathologically large section is
# an edge case; the prompt also truncates each chunk to 1,800 chars).
_MAX_SECTION_CHUNKS = 16
# The prompt template's per-chunk `truncate(N)` — kept in sync here so the input
# budget below can predict the real prompt size.
_CHUNK_TRUNCATE_CHARS = 1_800
# Hard cap on the RENDERED prompt size (per-chunk text + wrapper) a SINGLE map/ground
# call may show — the budget the packers/splitter enforce via `_chunk_budget_chars`. The
# MAP call is budget-binding: it requests up to ~2,048 output tokens inside the SMALLEST
# (fast = 6,144) window, so the rendered input must clear ~6,144 − 2,048 = ~4,096 tokens.
# At the densest realistic ~2.7 chars/token, 10k rendered chars ≈ 3.7k tokens, +2,048
# output ≈ 5.8k < 6,144 (margin ~350). CRUCIAL: the budget counts the per-chunk WRAPPER
# (the ~65-char content-addressed chunk_id + the repeated doc title + formatting), NOT
# just the text — a packed deck holds 20-59 tiny slide-chunks, so a text-ONLY budget of
# 9,500 rendered to 18k chars / 6.6k tokens and the section MAP 400'd → the section was
# silently DROPPED (caught by `_map_section`; surfaced by scripts/report_structure_audit.py
# on the CUDA deck). This is the load-bearing guarantee behind the "baseline rule": the
# per-call input is bounded to fit fast, so it is IDENTICAL in fast and full mode →
# summary quality is mode-independent. A section larger than this is sub-split across MAP
# calls (`_split_section_into_batches`, no content dropped); a lone over-budget chunk is
# truncated by the prompt's per-chunk `truncate`, identically per mode.
_MAX_SECTION_INPUT_CHARS = 10_000
# Per-chunk rendered overhead beyond its text: the literal scaffolding in
# `summarize_section`'s chunk loop — `- [{chunk_id}] {document_title} (p.{page})\n  > …\n`.
# The chunk_id + doc-title lengths are added per chunk in `_chunk_budget_chars`; this is
# just the fixed literals (`- []  (p.) \n  > \n` + a short page no.), rounded up.
_CHUNK_WRAPPER_FIXED_CHARS = 24
# Doc-level key-point cap (mirrors DocSummary intent); section list cap.
_MAX_DOC_KEY_POINTS = 12
_MAX_SECTIONS = 40
# Cap the digests fed to the single REDUCE call so its prompt also fits the fast
# window (≈24 digests × ~650 chars ≈ 3.9k tokens + the bounded output stays well
# under 6,144). A doc with more sections still abstracts from a representative
# head; its per-section digests remain on `FinalResponse.sections` in full.
_REDUCE_MAX_SECTIONS = 24
# Report route (ADR-0010): the HIERARCHICAL reduce batches the grounded section
# digests this many at a time → one bounded paragraph per batch. Small enough that
# each batch's REDUCE prompt is well inside the fast window (≤4 digests), so the
# per-call boundedness that makes the single-level reduce safe holds per batch, and
# EVERY section reaches the output (no `_REDUCE_MAX_SECTIONS` truncation).
_REPORT_SECTIONS_PER_BATCH = 4
# Adaptive structure (ADR-0010): the model picks where paragraph breaks fall over the
# ordered sections (boundary-selection — can't drop/duplicate a section). Guardrail: no
# planned paragraph may span more than this many sections (an over-long run is split),
# which bounds each reduce's input and blocks a degenerate "combine everything" plan.
_REPORT_MAX_SECTIONS_PER_PARAGRAPH = 6
# The OTHER guardrail: an 8B tends to OVER-split (observed live: 15 near-singleton breaks
# over 19 sections — worse fragmentation than mechanical, and it expands tiny 1-section
# sources into long paragraphs that drift). So the planner's groups are COALESCED up to
# this "fuller" target — the model can choose WHERE to break, but never fragment below
# ~this many sections per paragraph. Keeps paragraphs full + faithful regardless of the
# model's compliance with the "combine" instruction.
_REPORT_TARGET_SECTIONS_PER_PARAGRAPH = 4
# Cross-paragraph dedup (ADR-0010). The 8B re-covers a topic when its OWN sections overlap a
# prior paragraph's — IGNORING the reduce prompt's strong "do NOT repeat" rule AND the full
# rolling `preceding` context (confirmed live on SRWE-Module-5: root-bridge election written
# in two adjacent paragraphs; two paragraphs sharing a "Module Practice and Quiz" closer).
# Prompt-only anti-repetition is a known dead end on this model (the OPENING-pattern repeat
# needed a STRUCTURAL fix — the branched prompt — not an instruction). So a DETERMINISTIC
# gate drops a sentence whose content tokens are already >this fraction covered by a SINGLE
# earlier kept sentence (overlap-PRECISION, so merely sharing a few common terms survives).
# Lexical → always-on + reproducible (the confidence embedder degrades under VRAM pressure,
# so it can't be the dedup signal). HARD-gate-safe: only REMOVES already-grounded prose,
# never adds — it cannot introduce an ungrounded assertion.
_REPORT_DEDUP_THRESHOLD = 0.7
# Output reservation for the small GROUND/REDUCE calls (a verification index-list /
# a short abstract) — kept tight so prompt + output clears the fast window.
_VERIFY_MAX_TOKENS = 768
_REDUCE_MAX_TOKENS = 1024
# Tabular route (ADR-0008): a doc with at least this many well-formed stored tables
# gets an extra grounded "Key figures" pass over `tables.sqlite`. Table chunks are
# built up to `_MAX_TABLES_FOR_FIGURES` (then `_bound_section_chunks` keeps only the
# leading subset that fits the fast window — a doc-order sample; figure-salience
# selection is a deferred refinement), each rendered with up to `_MAX_TABLE_ROWS`.
_TABULAR_MIN_TABLES = 4
_MAX_TABLES_FOR_FIGURES = 24
_MAX_TABLE_ROWS = 12
# The Key-figures MAP budget is TIGHTER than the prose `_MAX_SECTION_INPUT_CHARS`
# (10k): the `summarize_tabular` prompt has a bigger scaffold than `summarize_section`,
# so the SAME 10k chunk budget rendered to ~4,204 input tokens on the 98-table 10-K and
# the MAP 400'd (4,204 + 2,048 output > the 6,144 fast window) → the Key-figures section
# was silently DROPPED on the most table-rich doc we have (caught by the stress test,
# 2026-05-28). 8k keeps the top-11 SALIENCE-ranked tables at ~3,491 input tokens (+2,048
# = 5,539, ~600-token margin). The tables are ranked, so the dropped tail is the least
# figure-rich anyway.
_TABULAR_MAP_INPUT_CHARS = 8_000
# Figure-salience (ADR-0008 §7): tables are RANKED, not taken in document order, so
# the headline data tables (income statement, segment revenue, balance sheet) win
# over a front-matter fragment on a many-table doc. The score is PURE + deterministic
# (no LLM — the determinism mandate): numeric density dominates, a monetary/percent
# signal and a headline-section keyword break ties. A monetary cell already parses as
# numeric (coerce_number handles $, %, scale words), so `_MONEY_RE` is the extra
# "this is a money/rate figure, not a bare count" weight.
_MONEY_RE = re.compile(r"[$€£%]|\b(?:thousand|million|billion|trillion)\b", re.IGNORECASE)
# Substring match against the (lowercased) table section/caption — financial-leaning
# but only a tie-breaker (weight 0.6 vs numeric density's 2.0), so a non-financial
# tabular doc still ranks correctly by numeric density.
# A table wider than this is treated as a framing-risky time-series/scenario grid.
_WIDE_TABLE_COLS = 6
_SALIENT_SECTION_TERMS: frozenset[str] = frozenset(
    {
        "revenue",
        "income",
        "sales",
        "earnings",
        "profit",
        "margin",
        "expense",
        "cost",
        "asset",
        "liabilit",
        "equity",
        "cash flow",
        "dividend",
        "repurchase",
        "compensation",
        "balance sheet",
        "segment",
        "operations",
        "financial",
        "statement",
        "results",
        "performance",
    }
)


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


def _should_pack_sections(sections: list[tuple[str, list[Chunk]]]) -> bool:
    """True when a doc has many heading-sections, MOST of them slide-sized — its
    per-section MAP digests would be thin (one slide / one short subsection each), so
    the long route packs adjacent sections (`_pack_sections`). Fires on slide decks
    AND bullet-heavy technical docs; a paper/standard with substantive sections (few
    tiny ones) returns False, keeping its own per-section digests."""
    if len(sections) < _PACK_MIN_SECTIONS:
        return False
    tiny = sum(1 for _t, cs in sections if sum(len(c.text) for c in cs) <= _PACK_TINY_SECTION_CHARS)
    return tiny / len(sections) >= _PACK_TINY_FRACTION


def _chunk_budget_chars(c: Chunk) -> int:
    """A chunk's REAL contribution to the rendered MAP/GROUND prompt: its (truncated) text
    PLUS the per-chunk wrapper (`- [{chunk_id}] {doc_title} (p.{page})\\n  > …`). The
    wrapper is dominated by the long content-addressed chunk_id + the repeated doc title, so
    a text-ONLY budget under-counts a packed deck's many tiny slide-chunks by thousands of
    chars → window overflow + dropped sections. All the window-budget packers measure with
    THIS so the cap reflects the rendered size, not just the text."""
    return (
        min(len(c.text), _CHUNK_TRUNCATE_CHARS)
        + len(c.chunk_id)
        + len(c.document_title or "")
        + _CHUNK_WRAPPER_FIXED_CHARS
    )


def _pack_sections(
    sections: list[tuple[str, list[Chunk]]], max_chars: int
) -> list[tuple[str, list[Chunk]]]:
    """Greedily merge CONSECUTIVE sections into groups whose combined chunk-text fits
    `max_chars` (the fast-window budget, measured like `_bound_section_chunks`), so a
    deck's tiny slide-sections become substantive MAP units. Each group is titled by
    its first section (the digest synthesizes the span); reading order is preserved.
    A lone oversized section forms its own group (trimmed later by the window bound)."""
    groups: list[tuple[str, list[Chunk]]] = []
    buf_title = ""
    buf_chunks: list[Chunk] = []
    buf_total = 0
    for sec_title, sec_chunks in sections:
        sec_chars = sum(_chunk_budget_chars(c) for c in sec_chunks)
        if buf_chunks and buf_total + sec_chars > max_chars:
            groups.append((buf_title, buf_chunks))
            buf_title, buf_chunks, buf_total = "", [], 0
        if not buf_chunks:
            buf_title = sec_title
        buf_chunks.extend(sec_chunks)
        buf_total += sec_chars
    if buf_chunks:
        groups.append((buf_title, buf_chunks))
    return groups[:_MAX_SECTIONS]


def _bound_section_chunks(
    chunks: list[Chunk], max_chars: int = _MAX_SECTION_INPUT_CHARS
) -> list[Chunk]:
    """Take a section's chunks up to a RENDERED char budget (and the chunk-count cap)
    so a single MAP/GROUND call fits even the SMALLEST window (fast, 6,144). This is
    what makes quality independent of the co-residence mode: the per-call input is
    the same in fast and full. Always keeps at least the first chunk (a lone giant
    chunk is truncated by the prompt's per-chunk `truncate`, identically per mode).

    `max_chars` defaults to the prose-MAP budget; the tabular Key-figures MAP passes a
    TIGHTER `_TABULAR_MAP_INPUT_CHARS` because its prompt scaffold is bigger (else the
    same budget overflows — see that constant)."""
    out: list[Chunk] = []
    total = 0
    for c in chunks[:_MAX_SECTION_CHUNKS]:
        clen = _chunk_budget_chars(c)  # text + per-chunk wrapper (mirrors the rendered prompt)
        if out and total + clen > max_chars:
            break
        out.append(c)
        total += clen
    return out


def _split_section_into_batches(chunks: list[Chunk], max_chars: int) -> list[list[Chunk]]:
    """Split a section's chunks into consecutive batches that EACH fit `max_chars`
    (the fast-window budget, by the prompt's per-chunk `truncate` measure), so a
    section larger than one window is summarized across multiple MAP calls rather
    than TRUNCATED — every chunk lands in exactly one batch, so no content is
    dropped, and each batch is mode-independent (fits fast). A section that already
    fits is a single batch (unchanged behaviour); a lone oversized chunk forms its
    own batch (the prompt truncates it). This generalizes `_bound_section_chunks`
    (which returns only the first batch)."""
    batches: list[list[Chunk]] = []
    cur: list[Chunk] = []
    total = 0
    for c in chunks:
        clen = _chunk_budget_chars(c)
        if cur and total + clen > max_chars:
            batches.append(cur)
            cur = []
            total = 0
        cur.append(c)
        total += clen
    if cur:
        batches.append(cur)
    return batches


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


def _select_doc_key_points(
    sections: list[SectionSummary], cap: int
) -> list[CitedClaim]:
    """PURE: pick the doc-level headline key-points from the per-section grounded
    points, distributed ROUND-ROBIN across sections so the `cap` spans many
    sections instead of being a reading-order prefix dominated by a front-matter
    section's points (see the call site for the NIST boilerplate motivation).

    Rank 0 takes `key_points[0]` of each section in reading order, then rank 1
    takes `key_points[1]` of each, etc., until `cap` is reached or every point
    is consumed. Deterministic + stable; a single-section doc is unchanged
    (its points come out in order). HARD-gate-neutral (selection only)."""
    out: list[CitedClaim] = []
    rank = 0
    progressed = True
    while len(out) < cap and progressed:
        progressed = False
        for ss in sections:
            if rank < len(ss.key_points):
                out.append(ss.key_points[rank])
                progressed = True
                if len(out) >= cap:
                    break
        rank += 1
    return out


async def _reduce(
    title: str,
    sections: list[SectionSummary],
    instruction: str | None,
    max_output_tokens: int,
    abstract_guidance: str,
    *,
    scope_note: str = "",
    preceding: str = "",
) -> tuple[DocAbstract, int]:
    """Synthesize the per-section digests into a whole-document abstract.

    Returns the `DocAbstract` (not just its joined `.abstract` string) so the report
    route can dedup at the SENTENCE level across paragraphs; the single-level caller
    reads `.abstract` as before.

    `scope_note` (default "") overrides the prompt's scope framing. Empty = the
    whole-document overview (the single-level reduce — byte-identical to before).
    The report route (`_reduce_report`) passes a per-batch note so a batch
    paragraph does NOT claim to summarize the whole document, plus `preceding` =
    the prior paragraph's text so this one opens with a transition and adds only
    NEW material — the paragraphs read as ONE coherent narrative, not disjoint
    blocks (ADR-0010). Both default empty → single-level prompt byte-identical."""
    prompt = render_prompt(
        "summarize_reduce",
        title=title,
        sections=sections[:_REDUCE_MAX_SECTIONS],  # keep the REDUCE prompt within the fast window
        instruction=instruction or "",
        abstract_guidance=abstract_guidance,
        scope_note=scope_note,
        preceding=preceding,
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
        return DocAbstract(sentences=[]), 0
    return overview, tokens


async def _plan_report_structure(
    title: str, sections: list[SectionSummary], instruction: str | None
) -> tuple[list[list[SectionSummary]] | None, int]:
    """Ask the model where paragraph breaks should fall over the ORDERED section digests
    (ADR-0010 adaptive structure) so related sections COMBINE into one coherent paragraph.

    Boundary-selection — the model returns the section indices that START a paragraph, so
    the result is always a complete, order-preserving partition (it cannot drop or
    duplicate a section). Validated + size-capped (`_REPORT_MAX_SECTIONS_PER_PARAGRAPH`).
    Returns `(groups, tokens)`; `None` groups signals the caller to fall back to mechanical
    batching — on a model error, an empty plan, or a TRIVIAL plan (no internal break). Every
    decision is logged (the observability the structural step needs). HARD-gate-neutral:
    only the grouping changes, never grounding."""
    log = logger.bind(node="summarize")
    n = len(sections)
    # Coalesce fullness target — tunable (default `_REPORT_TARGET_SECTIONS_PER_PARAGRAPH`)
    # via `report_coalesce_target` so the report-structure validator can sweep it.
    coalesce_target = get_settings().agents.report_coalesce_target
    # Suggest a paragraph count so the model combines (~target sections each) instead of
    # breaking at every section; the coalesce guardrail below enforces it regardless.
    target_paragraphs = max(2, round(n / coalesce_target))
    # A bounded, indexed digest preview per section (truncated so the plan prompt fits the
    # fast window even for a many-section doc) — the model picks break points over these.
    sections_block = "\n".join(
        f"[{i}] {s.section_title} — {s.digest[:180]}" for i, s in enumerate(sections)
    )
    prompt = render_prompt(
        "summarize_report_plan",
        title=title,
        sections_block=sections_block,
        instruction=instruction or "",
        target_paragraphs=target_paragraphs,
    )
    try:
        plan, tokens = await complete_structured(
            prompt=prompt,
            schema=ReportStructure,
            max_tokens=_REDUCE_MAX_TOKENS,
            prompt_tag="summarize_report_plan@v1",
        )
    except ModelCallError as e:
        log.warning("report.plan_failed", error=str(e)[:160])
        return None, 0
    # In-range, deduplicated, sorted; a paragraph always begins at section 0.
    starts = sorted({s for s in plan.paragraph_starts if 0 <= s < n})
    if not starts or starts[0] != 0:
        starts = sorted({0, *starts})
    if len(starts) <= 1:
        # No internal break chosen → not a real plan; use mechanical batching.
        log.info("report.plan_trivial", sections=n)
        return None, tokens
    # Build the consecutive runs, splitting any run that exceeds the size guardrail.
    bounds = [*starts, n]
    raw_groups: list[list[SectionSummary]] = []
    for a, b in pairwise(bounds):
        run = sections[a:b]
        for i in range(0, len(run), _REPORT_MAX_SECTIONS_PER_PARAGRAPH):
            raw_groups.append(run[i : i + _REPORT_MAX_SECTIONS_PER_PARAGRAPH])
    # COALESCE: merge the planner's runs (in order, so reading order is preserved) up to the
    # "fuller" target, never exceeding the hard max — this is what stops an over-splitting 8B
    # from shattering the doc into near-singletons (the observed failure). The planner still
    # chose WHERE the breaks may fall; coalescing only drops breaks that would leave a
    # paragraph too thin.
    groups: list[list[SectionSummary]] = []
    cur: list[SectionSummary] = []
    for g in raw_groups:
        if cur and (
            len(cur) >= coalesce_target
            or len(cur) + len(g) > _REPORT_MAX_SECTIONS_PER_PARAGRAPH
        ):
            groups.append(cur)
            cur = list(g)
        else:
            cur.extend(g)
    if cur:
        groups.append(cur)
    # FLOOR: a `report` must read as multi-paragraph. A heavily-packed deck can yield few
    # section_summaries that coalesce to ONE paragraph (observed: an 89-slide WAN module
    # packed to 4 sections → 1 paragraph), which is just a `detailed` summary. If the
    # coalesce produced a single group of ≥4 sections, split it back in half. Only the
    # degenerate case is touched — a doc that already yields ≥2 paragraphs is unchanged.
    if len(groups) == 1 and len(groups[0]) >= 4:
        lone = groups[0]
        mid = len(lone) // 2
        groups = [lone[:mid], lone[mid:]]
    log.info(
        "report.plan",
        chosen_starts=len(starts),
        raw_paragraphs=len(raw_groups),
        paragraphs=len(groups),
        sizes=[len(g) for g in groups],
    )
    return groups, tokens


# Faithfulness confidence for `report` summaries (ADR-0010) — a hybrid, INFORM-ONLY
# signal (never the HARD gate; `must_not_assert` stays that). Lexical overlap is the
# deterministic "advanced pattern matching" half (catches fabricated specifics); embedding
# cosine is the semantic half (sees through paraphrase). Each scores a generated paragraph
# against the grounded digests it was built from, closing the loop on the restructuring.
# Weight the SEMANTIC signal higher than lexical: a faithful abstractive paragraph
# paraphrases (low verbatim overlap) but should stay semantically close, so embedding
# is the better faithfulness proxy; lexical is the secondary "did it keep the key terms"
# check. (Equal weighting read misleadingly low for good abstractive prose.)
_CONFIDENCE_EMB_WEIGHT = 0.7
_STOPWORDS = frozenset(
    "the a an of to in and or for on at by is are was were be been being it its this that these "
    "those with as from into than then so such not no but if while which who whom whose their "
    "there here have has had do does did can could should would may might will shall we you they "
    "he she them his her our your".split()
)


def _content_tokens(text: str) -> list[str]:
    """Lowercased alphanumeric content tokens (stopwords + 1-char tokens dropped)."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in _STOPWORDS]


def _lexical_overlap(paragraph: str, source: str) -> float:
    """Fraction of the paragraph's content (unigrams 0.7 + bigrams 0.3) attested in the
    source digests — the deterministic pattern-matching half. 1.0 for an empty paragraph."""
    p = _content_tokens(paragraph)
    if not p:
        return 1.0
    s = _content_tokens(source)
    s_uni = set(s)
    uni = sum(1 for t in p if t in s_uni) / len(p)
    p_bi = list(pairwise(p))
    if not p_bi:
        return uni
    s_bi = set(pairwise(s))
    bi = sum(1 for b in p_bi if b in s_bi) / len(p_bi)
    return 0.7 * uni + 0.3 * bi


def _dedup_sentences(sentences: list[str], kept_token_sets: list[set[str]]) -> list[str]:
    """Return the `sentences` that ADD content, and append each survivor's content-token
    set to `kept_token_sets` (the running ledger across the WHOLE report). A sentence is a
    cross-paragraph REPEAT — dropped — when >`_REPORT_DEDUP_THRESHOLD` of its content tokens
    already appear in a SINGLE earlier kept sentence (overlap-precision against the best
    match, so a sentence merely sharing a few common terms survives). Deterministic +
    HARD-gate-safe: it only removes already-grounded prose, so it can never add an
    ungrounded assertion. The ledger is per-sentence (not a pooled union) so a genuinely
    novel sentence isn't dropped just because its words appear scattered across earlier
    sentences."""
    survivors: list[str] = []
    for sent in sentences:
        toks = set(_content_tokens(sent))
        if toks and any(
            len(toks & kept) / len(toks) > _REPORT_DEDUP_THRESHOLD for kept in kept_token_sets
        ):
            continue
        survivors.append(sent)
        if toks:
            kept_token_sets.append(toks)
    return survivors


async def _embedding_alignment(paragraphs: list[str], sources: list[str]) -> list[float] | None:
    """Per-paragraph semantic cosine vs its source-digest text, via the shared embedder
    (normalized → cosine = dot). BEST-EFFORT: any failure (e.g. the embedder can't load
    co-resident with the orchestrator) returns None so the caller degrades to lexical-only —
    the confidence signal can never break a summary."""
    log = logger.bind(node="summarize")
    try:
        from memex.models.registry import get_registry

        registry = get_registry()
        texts = [*paragraphs, *sources]

        async with registry.use("embedder") as embedder:

            def _encode() -> Any:
                return embedder.encode(
                    texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
                )

            vecs = await asyncio.to_thread(_encode)
    except Exception as e:
        log.warning("report.confidence_embed_unavailable", error=str(e)[:160])
        return None
    k = len(paragraphs)
    out: list[float] = []
    for i in range(k):
        cos = float(vecs[i] @ vecs[k + i])  # normalized rows → dot == cosine
        out.append(max(0.0, min(1.0, cos)))
    return out


async def _score_report_confidence(
    paragraphs: list[str], groups: list[list[SectionSummary]]
) -> ReportConfidence | None:
    """Hybrid faithfulness confidence (ADR-0010, inform-only): each paragraph scored against
    the digests of the sections it was built from — embedding cosine + lexical overlap,
    combined `_CONFIDENCE_EMB_WEIGHT`/remainder. None when no paragraph was produced."""
    if not paragraphs:
        return None
    sources = [" ".join(s.digest for s in g) for g in groups]
    lex = [_lexical_overlap(p, src) for p, src in zip(paragraphs, sources, strict=False)]
    emb = await _embedding_alignment(paragraphs, sources)
    if emb is None:
        emb = lex  # embedding unavailable → combined collapses to the lexical signal
    w = _CONFIDENCE_EMB_WEIGHT
    per_para = [round(w * e + (1 - w) * lx, 3) for e, lx in zip(emb, lex, strict=False)]
    overall = round(sum(per_para) / len(per_para), 3)
    conf = ReportConfidence(
        overall=overall,
        embedding=round(sum(emb) / len(emb), 3),
        lexical=round(sum(lex) / len(lex), 3),
        per_paragraph=per_para,
    )
    logger.bind(node="summarize").info(
        "report.confidence",
        overall=overall,
        embedding=conf.embedding,
        lexical=conf.lexical,
        paragraphs=len(per_para),
        min_paragraph=min(per_para),
    )
    return conf


async def _reduce_report(
    title: str,
    sections: list[SectionSummary],
    instruction: str | None,
    max_output_tokens: int,
    abstract_guidance: str,
    emit: Callable[[str], None],
) -> tuple[str, int, ReportConfidence | None]:
    """HIERARCHICAL reduce → a multi-paragraph report body (ADR-0010).

    `_plan_report_structure` groups the grounded section digests into paragraphs
    (adaptive — combine related sections; falls back to fixed-size batching), then
    EACH group is reduced to one bounded paragraph (reusing `_reduce`, hence the same
    bounded `DocAbstract` primitive) and the paragraphs are stitched in document order
    with blank-line separators. Finally `_score_report_confidence` closes the loop by
    grading each paragraph against its source digests (inform-only). This is the safe
    inverse of the removed free-form
    `synthesize` (ADR-0009): every model call still takes small grounded input →
    bounded output, so none of the three free-form failure modes (regurgitation /
    won't-stop / mid-word force-close) can occur, and EVERY section reaches the
    output (no `_REDUCE_MAX_SECTIONS` truncation). The stitch is DETERMINISTIC — no
    final whole-output model pass, which would reintroduce the free-form trap.

    The batches are reduced in SEQUENCE with ROLLING context: each paragraph after
    the first is given the PREVIOUS paragraph (`preceding`) so it opens with a
    transition and adds only NEW material — the paragraphs read as one coherent
    narrative, not disjoint "The document focuses on…" blocks. `scope_note` makes it
    a proper ARC: the FIRST batch is the OPENING (introduce the subject), the LAST is
    the CLOSING (conclude), the rest CONTINUE (never claiming whole-doc scope). Both
    signals are bounded (one prior paragraph + ≤4 digests), so the prompt stays inside
    the fast window and grounding is untouched (it recombines already-grounded prose)."""
    # ADAPTIVE structure: let the model group related sections into paragraphs; fall
    # back to fixed-size batching on a trivial/failed plan (deterministic, always valid).
    emit("Reducing · planning structure")
    groups, tokens_total = await _plan_report_structure(title, sections, instruction)
    if groups is None:
        groups = [
            sections[i : i + _REPORT_SECTIONS_PER_BATCH]
            for i in range(0, len(sections), _REPORT_SECTIONS_PER_BATCH)
        ]
    paragraphs: list[str] = []
    para_groups: list[list[SectionSummary]] = []  # the source group per emitted paragraph
    kept_tokens: list[set[str]] = []  # content-token set of every KEPT sentence (cross-para dedup)
    sentences_in = sentences_kept = 0  # dedup observability
    n = len(groups)
    for gi, group in enumerate(groups):
        emit(f"Reducing · paragraph {gi + 1} of {n}")
        # Position-aware framing gives the report a proper ARC: the FIRST group opens
        # (introduce the subject), the LAST closes (conclude), the rest continue. With
        # one group it's just a plain whole-doc overview (no arc).
        if n == 1:
            scope_note = ""
        elif gi == 0:
            scope_note = "the OPENING part of a multi-part overview"
        elif gi == n - 1:
            scope_note = "the CLOSING part of a multi-part overview"
        else:
            scope_note = "a CONTINUING part of a multi-part overview"
        # The overview SO FAR = ALL prior paragraphs (not just the last), so the model can
        # avoid REPEATING any earlier point — critical on a topically-homogeneous doc whose
        # sections restate ONE thesis (a paper, a form), where last-paragraph-only context
        # still produced near-duplicate paragraphs (observed live on the GTE paper). Bounded
        # by the report's own (coalesce-capped) length; the trailing cap keeps the prompt
        # inside the fast window in the pathological many-paragraph case.
        running = "\n\n".join(paragraphs) if (paragraphs and gi > 0) else ""
        preceding = running[-4000:]
        da, t = await _reduce(
            title,
            group,
            instruction,
            max_output_tokens,
            abstract_guidance,
            scope_note=scope_note,
            preceding=preceding,
        )
        tokens_total += t
        # DETERMINISTIC cross-paragraph dedup: keep only this paragraph's sentences that add
        # content not already covered (the rolling `preceding` + prompt rule above don't stop
        # the 8B re-covering a topic when its own sections overlap a prior paragraph). A
        # paragraph emptied by dedup is wholly redundant → dropped (so `preceding` for the
        # next group also reflects the deduped text, and `para_groups` stays aligned).
        survivors = _dedup_sentences(da.sentences, kept_tokens)
        sentences_in += len(da.sentences)
        sentences_kept += len(survivors)
        if survivors:
            paragraphs.append(" ".join(survivors))
            para_groups.append(group)
    # Observability (ADR-0004): how much the dedup gate removed (inform-only).
    logger.bind(node="summarize").info(
        "report.dedup",
        sentences_in=sentences_in,
        kept=sentences_kept,
        dropped=sentences_in - sentences_kept,
        paragraphs=len(paragraphs),
    )
    # Close the loop: score each paragraph's faithfulness to the digests it was built
    # from (ADR-0010, inform-only — never gates). `para_groups` stays aligned with the
    # paragraphs actually emitted (a group that produced nothing is skipped from both).
    confidence = await _score_report_confidence(paragraphs, para_groups)
    return "\n\n".join(paragraphs), tokens_total, confidence


def _table_salience(table: StoredTable) -> float:
    """A PURE, deterministic salience score for a stored table — how likely it is to
    hold the document's headline figures. Numeric density dominates (a key-figures
    table is mostly numbers); a monetary/percent signal and a headline-section
    keyword break ties. A degenerate ≤1-row / cell-less table scores 0 (a parse
    fragment is not a figures source). Higher = more salient."""
    cells = [c.strip() for row in table.rows for c in row if c.strip()]
    if len(table.rows) < 2 or len(cells) < 2:
        return 0.0
    numeric = sum(1 for c in cells if coerce_number(c) is not None)
    money = sum(1 for c in cells if _MONEY_RE.search(c))
    density = numeric / len(cells)
    money_density = money / len(cells)
    section = (table.section or "").lower()
    keyword = 1.0 if any(term in section for term in _SALIENT_SECTION_TERMS) else 0.0
    base = 2.0 * density + 1.5 * money_density + 0.6 * keyword
    # Width factor (MULTIPLICATIVE — a subtractive penalty is swamped by the high
    # numeric density of a dense grid): a narrow `metric: value` table (income
    # statement, opex) yields unambiguous figures, while a WIDE grid (a performance
    # graph, a multi-year / multi-scenario projection) is framing-risky — the MAP can
    # attach a value to the wrong column/period. Validated on the 10-K: the wide
    # perf-graph + projection tables produced the noisy figures, so halve them.
    ncols = len(table.header) or (len(table.rows[0]) if table.rows else 0)
    return base * (0.5 if ncols > _WIDE_TABLE_COLS else 1.0)


def _rank_tables(tables: list[StoredTable]) -> list[StoredTable]:
    """Order a doc's tables by `_table_salience` (descending). Python's sort is
    stable, so equal-salience tables keep document order — deterministic + tie-stable.
    The window budget in `_key_figures_section` then keeps the leading (most-salient)
    subset that fits, so the headline tables are summarized even on a many-table doc."""
    return sorted(tables, key=_table_salience, reverse=True)


def _render_table(table: StoredTable) -> str:
    """Render a `StoredTable` as compact, self-contained text for the MAP/GROUND
    calls: a section label + each row as header-paired `col=cell` pairs (verbatim
    cells, the Phase-1 linearization shape), capped at `_MAX_TABLE_ROWS`. Pairing
    the header with each cell lets the model read AND ground a figure with its
    label without seeing the whole grid."""
    lines = [f"Table — {table.section}" if table.section else "Table"]
    if table.header:
        lines.append(" | ".join(table.header))
    for row in table.rows[:_MAX_TABLE_ROWS]:
        if table.header and len(row) == len(table.header):
            lines.append("; ".join(f"{h}={c}" for h, c in zip(table.header, row, strict=True) if c))
        else:
            lines.append(" | ".join(row))
    # The prompt also truncates to ~1700; cap here so a giant table can't dominate.
    return "\n".join(lines)[:_CHUNK_TRUNCATE_CHARS]


def _table_chunks(doc_id: str, tables: list[StoredTable], document_title: str) -> list[Chunk]:
    """Synthetic chunks (one per stored table, capped) carrying the verbatim cells —
    the grounding anchors for the key-figure points. `chunk_id` is `{doc_id}#tblN`
    (short, ≤80 for `CitedClaim.source_chunk_id`); `heading_path` is the table's
    section so the composed wikilink anchors there."""
    out: list[Chunk] = []
    for i, table in enumerate(tables[:_MAX_TABLES_FOR_FIGURES]):
        out.append(
            Chunk(
                chunk_id=f"{doc_id}#tbl{i}",
                document_id=doc_id,
                document_title=document_title,
                text=_render_table(table),
                heading_path=[table.section] if table.section else [],
                char_start=table.char_start,
                char_end=table.char_end,
            )
        )
    return out


async def _load_doc_tables(vault_path: Path, doc_id: str) -> list[StoredTable]:
    """The doc's well-formed stored tables (`tables.sqlite`), or `[]` — fail-open
    so a missing/corrupt table store never breaks summarization (it just skips the
    tabular route). Lazy `TableStore.open` per the documented `agents/ → index/`
    edge (same as `query_tables`)."""
    try:
        store = await TableStore.open(vault_path)
    except (MemexError, OSError, sqlite3.Error) as exc:
        logger.warning("summarize.table_store_unavailable", error=str(exc)[:160])
        return []
    try:
        return await store.tables_for_document(doc_id)
    except (MemexError, OSError, sqlite3.Error) as exc:
        logger.warning("summarize.table_read_failed", error=str(exc)[:160])
        return []
    finally:
        await store.close()


def _figure_number_in_chunk(claim: CitedClaim, chunk_text_by_id: dict[str, str]) -> bool:
    """Deterministic backstop for key figures (ADR-0008): a key figure is a VERBATIM cell
    read, so at least one of the claim's numbers must literally appear (comma-normalized) in
    its cited table-chunk. Catches what the LLM verifier's numeric false-positive lets
    through — it once grounded a fabricated "$16,042M" against a cell holding "16384" (a
    close number, different figure). A claim with NO number is left to the LLM grounding (a
    qualitative key point). Scoped to the verbatim key-figures path ONLY — NOT the answer
    path, whose claims paraphrase numbers (e.g. "about 800 million") where this would
    false-reject."""
    nums = re.findall(r"\d[\d,]*\.?\d*", claim.claim)
    if not nums:
        return True
    norm_chunk = (chunk_text_by_id.get(claim.source_chunk_id) or "").replace(",", "")
    for n in nums:
        norm = n.replace(",", "").rstrip(".")
        if norm and norm in norm_chunk:
            return True
    return False


async def _key_figures_section(
    table_chunks: list[Chunk],
    instruction: str | None,
    max_output_tokens: int,
    digest_guidance: str,
) -> tuple[SectionSummary | None, int]:
    """Extract the document's KEY FIGURES from its tables into a grounded
    `SectionSummary` (ADR-0008 tabular route). MAPs `summarize_tabular/v1` over the
    table chunks (bounded to fit the fast window, like every other call), then
    GROUNDS the figures against those same chunks — a verbatim cell value grounds, a
    computed/fabricated one is dropped (the row-verbatim fabrication boundary,
    enforced by the existing verifier). Returns `(None, tokens)` on a failed map."""
    if not table_chunks:
        return None, 0
    # Tighter budget than the prose MAP — the tabular prompt scaffold is bigger, so the
    # 10k prose budget overflowed the 6,144 window on the 98-table 10-K (the Key-figures
    # section was then silently dropped). See `_TABULAR_MAP_INPUT_CHARS`.
    shown = _bound_section_chunks(table_chunks, _TABULAR_MAP_INPUT_CHARS)
    prompt = render_prompt(
        "summarize_tabular",
        chunks=shown,
        instruction=instruction or "",
        digest_guidance=digest_guidance,
    )
    try:
        mapped, t_map = await complete_structured(
            prompt=prompt,
            schema=SectionSummary,
            max_tokens=max_output_tokens,
            prompt_tag="summarize_tabular@v1",
        )
    except ModelCallError as e:
        logger.warning("summarize.tabular_map_failed", error=str(e)[:160])
        return None, 0
    # Repair MAP-emitted ids (the synthetic `{doc}#tblN`) against the shown table
    # chunks before grounding — same id-transcription fix as the prose path.
    repaired_points, _ = repair_claim_chunk_ids(mapped.key_points, shown)
    grounded, t_g = await ground_claims(
        mapped.digest[:300], repaired_points, shown, max_tokens=_VERIFY_MAX_TOKENS
    )
    # Deterministic backstop: drop any grounded figure whose number is absent from its cited
    # chunk (the LLM verifier false-positives near-numbers; a key figure must be verbatim).
    chunk_text_by_id = {c.chunk_id: c.text for c in shown}
    verified = [kp for kp in grounded if _figure_number_in_chunk(kp, chunk_text_by_id)]
    dropped = [kp.claim for kp in grounded if not _figure_number_in_chunk(kp, chunk_text_by_id)]
    if dropped:
        logger.info(
            "summarize.key_figures_numeric_drop",
            dropped=len(dropped),
            claims=[c[:120] for c in dropped],
        )
    return (
        SectionSummary(section_title="Key figures", digest=mapped.digest, key_points=verified),
        t_map + t_g,
    )


async def summarize_document(
    doc_id: str,
    *,
    instruction: str | None = None,
    detail: SummaryDetail = "standard",
    max_output_tokens: int = 2048,
    token_budget: int = 120_000,
    correlation_id: str | None = None,
    on_phase: Callable[[str], None] | None = None,
) -> FinalResponse:
    """Produce a structured, GROUNDED summary of one document (ADR-0008).

    `instruction` optionally focuses the summary. `detail` tunes length/verbosity
    (`brief` / `standard` / `detailed` — threaded into the MAP/REDUCE prompts via
    `_DETAIL_GUIDANCE`; `report` additionally switches REDUCE to the multi-paragraph
    hierarchical reducer, ADR-0010). `max_output_tokens` bounds each call; `token_budget` caps
    the whole map-reduce's total token traffic (a typical doc completes within it,
    bounded ultimately by `_MAX_SECTIONS`; a pathologically long doc stops early —
    a v1 limit, tunable per call). Returns a
    `FinalResponse`: `summary`=the abstract, `claims`=grounded doc-level
    key-points, `sections`=per-section digests, `wikilinks`/`used_chunks` for
    "Sources". A zero-grounded doc refuses.

    `correlation_id` (optional) keys the run instead of a fresh ULID — the webui's
    progress registry uses it. `on_phase` (optional) is an observe-only sink called
    with each phase label ("Summarizing · section k of N", "Reducing", …) for a live
    progress UI; it never touches the summary. Both default off → CLI/MCP/eval
    unchanged.
    """
    correlation_id = correlation_id or str(ulid.ULID())
    clear_run_context()
    bind_run_context(correlation_id, query_preview=f"summarize {doc_id}")
    log = logger.bind(node="summarize", doc_id=doc_id)

    def _emit(phase: str) -> None:
        # Observe-only progress sink (the webui's live indicator); a failing sink
        # must NEVER abort the summary.
        if on_phase is None:
            return
        try:
            on_phase(phase)
        except Exception:
            log.warning("summarize.on_phase_failed", phase=phase)

    # Summarizer swap-in (ADR-0010): for `report` detail with a configured summarizer,
    # serve a stronger model on the GPU freed by pausing the orchestrator and route the
    # whole map-reduce there (a 12B dedups across paragraphs where the 8B can't). Entered
    # AFTER the no-chunks check so a missing doc never spawns it; `finally` aclose()s the
    # stack (reap the summarizer + restart the orchestrator), so even an early refusal or
    # an error restores the orchestrator. Empty stack (the default path) is a pure no-op.
    swap_stack = AsyncExitStack()
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

        summarizer_model = settings.models.summarizer
        if summarizer_model and detail == "report":
            from memex.agents.summarizer_serve import serve_summarizer_vllm
            from memex.parse.pipeline import pause_vllm_for_gpu

            log.info("summarize.swap_in", model=summarizer_model)
            _emit("Loading summarizer")
            await swap_stack.enter_async_context(pause_vllm_for_gpu())
            base_url = await swap_stack.enter_async_context(serve_summarizer_vllm(summarizer_model))
            await swap_stack.enter_async_context(inference_override(base_url, summarizer_model))

        # Tabular route (ADR-0008): a table-heavy doc also gets a grounded
        # "Key figures" pass over its stored tables (fail-open → [] skips it).
        tables = await _load_doc_tables(settings.vault_path, doc_id)
        is_tabular = len(tables) >= _TABULAR_MIN_TABLES

        sections = _group_sections(chunks)
        route = _classify_route(chunks, sections)
        doc_title = chunks[0].document_title or doc_id
        # Rank tables by salience (not document order) so the headline data tables
        # win the window budget on a many-table doc.
        table_chunks = _table_chunks(doc_id, _rank_tables(tables), doc_title) if is_tabular else []
        # Deck/tiny-sectioned: pack thin slide-sections into substantive MAP units. For
        # `report` detail, pack to the (tunable, default = window) `report_pack_chars` —
        # SMALLER packs a deck into MORE, finer section_summaries, so the report planner has
        # the granularity to build a richer multi-paragraph structure (ADR-0010). Standard
        # detail keeps the full window. Each packed group is still ≤ the window, so the MAP
        # call's mode-independence holds.
        should_pack = route == "long" and _should_pack_sections(sections)
        pack_chars = (
            min(settings.agents.report_pack_chars, _MAX_SECTION_INPUT_CHARS)
            if detail == "report"
            else _MAX_SECTION_INPUT_CHARS
        )
        if route == "short":
            groups = [(doc_title, chunks)]
        elif should_pack:
            groups = _pack_sections(sections, pack_chars)
        else:
            groups = sections[:_MAX_SECTIONS]
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
            is_tabular=is_tabular,
            packed=should_pack,
            tables=len(tables),
            groups=len(groups),
        )

        tokens_total = 0
        section_summaries: list[SectionSummary] = []

        # Tabular: produce the grounded "Key figures" section FIRST — the headline
        # numbers are the point of summarizing a data-heavy doc, so a long doc's
        # prose budget must not starve them. It leads `section_summaries` (hence the
        # doc-level key-points); the prose sections append after. Bounded + grounded
        # like every other call; an empty/ungrounded result is simply skipped.
        if is_tabular:
            _emit("Summarizing · key figures")
            kf, t_kf = await _key_figures_section(
                table_chunks, instruction, max_output_tokens, guidance["digest"]
            )
            tokens_total += t_kf
            if kf is not None and kf.key_points:
                section_summaries.append(kf)
                log.info("summarize.key_figures", figures=len(kf.key_points))

        for gi, (title, sec_chunks) in enumerate(groups):
            if tokens_total > token_budget or len(section_summaries) >= _MAX_SECTIONS:
                log.info("summarize.budget_exhausted", done=len(section_summaries))
                break
            _emit(f"Summarizing · section {gi + 1} of {len(groups)}")
            # Sub-split a section larger than one window into batches that EACH fit
            # the smallest (fast) window — so its content is summarized across multiple
            # MAP calls rather than TRUNCATED to the first window (no content dropped),
            # while every call's input stays mode-independent. A section that fits is
            # one batch (unchanged). Each batch → its own grounded SectionSummary; a
            # multi-batch section is suffixed "(part k)".
            batches = _split_section_into_batches(sec_chunks, _MAX_SECTION_INPUT_CHARS)
            for bi, batch in enumerate(batches):
                if tokens_total > token_budget or len(section_summaries) >= _MAX_SECTIONS:
                    log.info("summarize.budget_exhausted", done=len(section_summaries))
                    break
                mapped, t_map = await _map_section(
                    title, batch, instruction, max_output_tokens, map_guidance
                )
                tokens_total += t_map
                if mapped is None:
                    continue
                # Snap each MAP-emitted `source_chunk_id` back to a real chunk id from
                # the batch the model was shown — the 8B model occasionally mangles the
                # long `docid#hash` it was told to copy verbatim (drops the prefix,
                # flips a char, corrupts the doc-id; observed live: "d646b8885-…",
                # "…gte-281#…"). `_ground_points` grounds by claim CONTENT/index, so a
                # content-supported point SURVIVES with its id still mangled → its
                # source can't resolve in `used_chunks` → the webui shows the raw hash.
                # Reuse the answering agent's deterministic repair (exact → suffix →
                # fuzzy; an unrepairable id is left to dangle); run it BEFORE grounding
                # so verify + the final stored ids both see the corrected id. Same
                # problem + fix as `answer` (`repair_claim_chunk_ids`).
                repaired_points, _ = repair_claim_chunk_ids(mapped.key_points, batch)
                grounded, t_g = await ground_claims(
                    mapped.digest[:300], repaired_points, batch, max_tokens=_VERIFY_MAX_TOKENS
                )
                tokens_total += t_g
                sec_title = mapped.section_title or title
                if len(batches) > 1:
                    sec_title = f"{sec_title} (part {bi + 1})"
                section_summaries.append(
                    SectionSummary(
                        section_title=sec_title,
                        digest=mapped.digest,
                        key_points=grounded,
                    )
                )

        # Doc-level key-points = the grounded section points, distributed
        # ROUND-ROBIN across sections (not a reading-order prefix). A document
        # with heavy front-matter — a government standard's cover + legal/FISMA
        # boilerplate, a paper's title/abstract/keywords — produces many easily
        # grounded but trivial points in its FIRST sections; a flat prefix
        # `[:12]` would fill the whole cap with them before any substantive
        # section contributes (observed live on NIST SP 800-207: all 12 points
        # were cover-page boilerplate). Round-robin takes each section's top
        # point first, then second points, etc. — so the cap spans up to 12
        # distinct sections and the body's substance is represented. Order
        # within the result still follows reading order at each rank, so the
        # abstract/claims read coherently. HARD-gate-neutral: every point is
        # already grounded; this only changes WHICH grounded points are the
        # headline set, never whether one is asserted.
        doc_points = _select_doc_key_points(section_summaries, _MAX_DOC_KEY_POINTS)
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

        # REDUCE → abstract. When there is exactly ONE section summary (a small
        # single-section doc, not split, not tabular) its digest IS the abstract — no
        # second call. The `report` detail builds a MULTI-paragraph body via the
        # hierarchical reduce (ADR-0010) — every section reaches the output. Otherwise
        # (multiple sections, a sub-split section's parts, or the tabular key-figures
        # section) synthesize across the digests into a single abstract.
        report_confidence: ReportConfidence | None = None
        if len(section_summaries) == 1 and not is_tabular:
            abstract = section_summaries[0].digest
        elif detail == "report":
            _emit("Reducing")
            abstract, t_r, report_confidence = await _reduce_report(
                doc_title,
                section_summaries,
                instruction,
                max_output_tokens,
                guidance["abstract"],
                _emit,
            )
            tokens_total += t_r
        else:
            _emit("Reducing")
            da, t_r = await _reduce(
                doc_title, section_summaries, instruction, max_output_tokens, guidance["abstract"]
            )
            tokens_total += t_r
            abstract = da.abstract
        if not abstract:
            # Fallback: synthesize a headline from the grounded points (never empty).
            abstract = " ".join(kp.claim for kp in doc_points[:3])

        # `used_chunks` must resolve EVERY citation the response SURFACES, not just
        # the doc-level headline. The webui renders the doc-level `claims` (doc_points)
        # AND a "By section" breakdown of every `section_summaries[*].key_points`;
        # MCP/CLI consume the per-section points too. Deriving the set from doc_points
        # alone left every NON-headline per-section point without a resolvable chunk,
        # so its source rendered as the raw `docid#hash` (the chunk wasn't in
        # `used_chunks` → no `chunk_ref`). doc_points are SELECTED from the section
        # key-points, so the section key-points ARE the complete set; the tabular
        # "Key figures" section is in `section_summaries`, so its synthetic `#tblN`
        # points are covered via `table_chunks`. Surviving key-points cite REAL chunk
        # ids by construction (a corrupted id cites a chunk absent from the verifier's
        # `chunk_by_id` → it can't ground → `_ground_points` drops it), so every id
        # here resolves against `chunks`/`table_chunks` — no repair, no template
        # fallback needed.
        used_ids = {kp.source_chunk_id for ss in section_summaries for kp in ss.key_points}
        used_chunks = [c for c in (chunks + table_chunks) if c.chunk_id in used_ids]
        wikilinks: list[str] = []
        seen: set[str] = set()
        for c in used_chunks:
            wl = format_wikilink(c.document_id, c.heading_path[-1] if c.heading_path else None)
            if wl not in seen:
                seen.add(wl)
                wikilinks.append(wl)

        _emit("Composing")
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
            report_confidence=report_confidence,
            artifact_scope_doc_ids=[doc_id],
            correlation_id=correlation_id,
            tokens_used=tokens_total,
            nodes_traversed=len(section_summaries) + 1,
            regenerate_attempts=0,
        )
    except MemexError:
        raise
    finally:
        # Tear down the swap-in (reap the summarizer vLLM + restart the orchestrator via
        # pause_vllm_for_gpu's exit + reset the routing override). No-op when not swapping.
        await swap_stack.aclose()
        clear_run_context()
