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

import re
import sqlite3
from collections.abc import Callable
from pathlib import Path
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
    repair_claim_chunk_ids,
)
from memex.agents.table_sql import coerce_number
from memex.core.config import get_settings
from memex.core.errors import MemexError, ModelCallError
from memex.core.types import Chunk, StoredTable
from memex.core.wikilinks import format_wikilink
from memex.index.fts_store import FTSStore
from memex.index.table_store import TableStore
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
# Tabular route (ADR-0008): a doc with at least this many well-formed stored tables
# gets an extra grounded "Key figures" pass over `tables.sqlite`. Table chunks are
# built up to `_MAX_TABLES_FOR_FIGURES` (then `_bound_section_chunks` keeps only the
# leading subset that fits the fast window — a doc-order sample; figure-salience
# selection is a deferred refinement), each rendered with up to `_MAX_TABLE_ROWS`.
_TABULAR_MIN_TABLES = 4
_MAX_TABLES_FOR_FIGURES = 24
_MAX_TABLE_ROWS = 12
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
        sec_chars = sum(min(len(c.text), _CHUNK_TRUNCATE_CHARS) for c in sec_chunks)
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
        clen = min(len(c.text), _CHUNK_TRUNCATE_CHARS)
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
    shown = _bound_section_chunks(table_chunks)
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
    grounded, t_g = await _ground_points(mapped.digest, repaired_points, shown)
    return (
        SectionSummary(section_title="Key figures", digest=mapped.digest, key_points=grounded),
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
    `_DETAIL_GUIDANCE`). `max_output_tokens` bounds each call; `token_budget` caps
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
        # Deck/tiny-sectioned: pack thin slide-sections into substantive MAP units.
        should_pack = route == "long" and _should_pack_sections(sections)
        if route == "short":
            groups = [(doc_title, chunks)]
        elif should_pack:
            groups = _pack_sections(sections, _MAX_SECTION_INPUT_CHARS)
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
                grounded, t_g = await _ground_points(mapped.digest, repaired_points, batch)
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
        # second call. Otherwise (multiple sections, a sub-split section's parts, or
        # the tabular key-figures section) synthesize across the digests so the
        # abstract reflects them all.
        if len(section_summaries) == 1 and not is_tabular:
            abstract = section_summaries[0].digest
        else:
            _emit("Reducing")
            abstract, t_r = await _reduce(
                doc_title, section_summaries, instruction, max_output_tokens, guidance["abstract"]
            )
            tokens_total += t_r
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
