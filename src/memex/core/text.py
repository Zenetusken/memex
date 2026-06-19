"""Cross-module text-manipulation helpers.

Lives in `core/` so both `index/` (chunker, fts_store) and `agents/`
(answer / verify prompt rendering) can import the same helper without
violating the documented module-import direction (`agents/ → core/`,
no `agents/ → index/`).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

    from memex.core.manifest import ChartExtraction

# Matches the bare `<!-- image -->` placeholder AND the enriched
# `<!-- image: kind=line-chart -->` form (docling_worker folds the PictureClassifier
# label into the marker, audit-10 step 2). ONE home for the regex (was parse-local)
# so the parse-time chart extraction and the index-time re-attach share it.
IMAGE_PLACEHOLDER_RE: Final[re.Pattern[str]] = re.compile(
    r"<!--\s*image(?::[^>]*)?\s*-->", re.IGNORECASE
)

# Matches a `[chart-extracted]...[/chart-extracted]` block emitted by
# the P3.3 chart-OCR stitch step (see parse/pipeline.py::_stitch_chart_
# extractions). The dot-all flag lets `.` span newlines so multi-line
# extracted tables match.
_CHART_EXTRACTED_RE = re.compile(
    r"\[chart-extracted\].*?\[/chart-extracted\]",
    flags=re.DOTALL,
)

# Matches an orphan `[chart-extracted]` opener with no matching closer
# until end-of-text. Used by `chart_extracted_spans` as a fall-through:
# if a chunk got truncated mid-block (or a user-edited vault doc dropped
# the close tag), the section-splitting filter still needs to know the
# orphan range so inert chart-figure `# H1` labels don't get treated as
# document headings. The closer-only regex is a paired safety net.
_CHART_EXTRACTED_OPEN_RE = re.compile(r"\[chart-extracted\]", flags=re.DOTALL)
_CHART_EXTRACTED_CLOSE_RE = re.compile(r"\[/chart-extracted\]", flags=re.DOTALL)


def strip_chart_extracted_for_index(text: str) -> str:
    """Remove `[chart-extracted]...[/chart-extracted]` blocks.

    Used at the **index layer** (`index.fts_store::upsert`) only: strips
    the FTS body so chart-extracted dense numerical tokens (years,
    percentages, raw values) don't inflate BM25 term frequency for
    unrelated queries.

    Previously also used at the agent layer (P3.3 v3, v5, v6) to keep
    long chart-block headers from eating the answer/verify prompt's
    `truncate(N)` budget. The P3.3 v7 fix (commit a9e8326) removed the
    agent-layer strips: the v7 chart-OCR backend emits compact markdown
    (~50-150 chars per block) that fits comfortably alongside prose, and
    the strips were actively masking chart-content answering capability.

    The stored chunk text (LanceDB + chunks_meta + vault markdown)
    is preserved unchanged so display / MCP / future trace tooling
    can still surface the chart-OCR content if needed.

    Idempotent — text without chart blocks passes through unchanged.
    """
    return _CHART_EXTRACTED_RE.sub("", text)


def chart_extracted_spans(text: str) -> list[tuple[int, int]]:
    """Return `(start, end)` char offsets of each
    `[chart-extracted]...[/chart-extracted]` block in `text`.

    Used by `index.chunker._split_into_sections` so heading-detection
    can skip `# H1` lines that appear INSIDE chart-extracted blocks.
    Nemotron-Parse emits H1 prefixes for chart-figure-section labels
    (e.g. nvmath-python's 4 design principles as `# Minimal lead-time`,
    `# Inter-operability`, etc.) which the chunker would otherwise
    treat as document-section boundaries — splitting a single chart
    block across multiple chunks. Established by the P3.3 v7 chunker
    fix (2026-05-23) after chart-OCR validation on the nvmath-python
    architecture figure revealed only the last principle was reaching
    the reranker's top-5.

    Truncation-tolerant: if a `[chart-extracted]` opener has no matching
    closer (e.g. mid-chunk truncation, user-edited vault) the orphan
    span extends to end-of-text. A `[/chart-extracted]` closer with no
    opener extends from start-of-text to the closer position. Both
    defensive cases prevent silent regressions of the P3.3 v7 chunker
    fix the original audit (2026-05-23 post-v7) flagged.

    Returns the inclusive-exclusive `(start, end)` of each span in
    `text`. Empty list if no chart-block tags present. Order-stable
    (sorted by start offset).
    """
    spans: list[tuple[int, int]] = list(
        (m.start(), m.end()) for m in _CHART_EXTRACTED_RE.finditer(text)
    )
    consumed_open: set[int] = {start for start, _ in spans}
    consumed_close: set[int] = {end for _, end in spans}

    for m in _CHART_EXTRACTED_OPEN_RE.finditer(text):
        if m.start() not in consumed_open:
            spans.append((m.start(), len(text)))
    for m in _CHART_EXTRACTED_CLOSE_RE.finditer(text):
        if m.end() not in consumed_close:
            spans.append((0, m.end()))

    spans.sort(key=lambda s: s[0])
    return spans


def is_inside_any_span(offset: int, spans: list[tuple[int, int]]) -> bool:
    """True iff `offset` falls strictly within one of `spans` (where each
    span is `(start, end)`, end-exclusive). Used with
    `chart_extracted_spans` to filter heading matches inside chart
    blocks. O(len(spans)); fine for typical doc sizes (<100 chart
    blocks)."""
    return any(start <= offset < end for start, end in spans)


def reattach_chart_extractions(body: str, blocks: Sequence[ChartExtraction]) -> str:
    """Re-insert the chart-OCR `[chart-extracted]` blocks (from the parse manifest sidecar)
    into a CLEAN body at the `<!-- image -->` placeholder positions — the index-time inverse
    of the parse-time extraction (audit-10 follow-on). Mirrors `[table-rows]` re-derivation,
    but sidecar-backed because chart-OCR output isn't re-derivable from the `.md`.

    LOAD-BEARING byte-equality contract: this reproduces the historical stitched body
    BYTE-FOR-BYTE — identical framing (`\\n\\n[chart-extracted]\\n{markdown}\\n[/chart-extracted]`)
    and identical LAST-TO-FIRST splice order as the old parse stitch
    (`parse/pipeline.py::_stitch_chart_extractions`) — so the chunker/embedder/FTS see the same
    bytes and chunk_ids (content-addressed) stay stable. A `placeholder_index` with no matching
    placeholder is skipped (defensive); empty `blocks` → identity (non-chart docs + already-inline
    legacy bodies, so the change is back-compat and safe to land before a re-parse migration)."""
    if not blocks:
        return body
    by_index: dict[int, str] = {b.placeholder_index: b.markdown for b in blocks if b.markdown}
    placeholders = list(IMAGE_PLACEHOLDER_RE.finditer(body))
    new_body = body
    # Last-to-first so each splice doesn't shift the offsets of earlier placeholders.
    for i, placeholder in reversed(list(enumerate(placeholders))):
        md = by_index.get(i)
        if not md:
            continue
        start, end = placeholder.span()
        new_body = (
            new_body[:start]
            + placeholder.group(0)
            + "\n\n[chart-extracted]\n"
            + md
            + "\n[/chart-extracted]"
            + new_body[end:]
        )
    return new_body


# ----- Citation-grade page-boundary marker scaffold (companion arc-3) ---------
#
# `Chunk.page` is recorded from per-page char counts taken BEFORE the body transforms
# (`_finalize_body` at parse; `reattach_chart_extractions` + `linearize_gfm_tables` at index), so on
# figure-heavy decks the chunker locates chunks in a body that has since grown → the displayed page
# drifts (navigation-grade, not citation-grade). The fix is a TRANSIENT marker that RIDES the
# transforms as a ruler: insert a marker block at each page boundary, run the transforms (the markers
# are inert to all of them — distinct from `IMAGE_PLACEHOLDER_RE`, and excluded from the
# `collapse_consecutive_duplicates` adjacency so a cross-page duplicate still collapses), MEASURE the
# marker positions against the exact body the chunker sees, then STRIP them. The stripped body is
# BYTE-IDENTICAL to the same body built without markers (the golden invariant → zero chunk_id churn).
_PAGE_BOUNDARY_FMT: Final[str] = "<!--MEMEX_PAGE_BOUNDARY:{}-->"
PAGE_BOUNDARY_RE: Final[re.Pattern[str]] = re.compile(r"<!--MEMEX_PAGE_BOUNDARY:(-?\d+)-->")


def is_page_boundary_marker(line: str) -> bool:
    """True iff `line` (stripped) is exactly a page-boundary marker — so
    `collapse_consecutive_duplicates` can keep a marker block verbatim but NOT count it toward
    adjacency (else a marker between two cross-page-duplicate blocks would prevent their collapse and
    break the byte-equality the round-trip depends on)."""
    return bool(PAGE_BOUNDARY_RE.fullmatch(line.strip()))


def mark_pages_for_measure(per_page: Sequence[tuple[int, str]], delimiter: str = "\n\n") -> str:
    """PARSE side: join `(page_no, markdown)` segments, each preceded by a page-boundary marker block.
    Pass only NON-empty segments (mirroring the real per-page join's `if markdown` filter). The
    marker positions after `_finalize_body` give each page's true span in the finalized body."""
    parts: list[str] = []
    for page_no, seg in per_page:
        parts.append(_PAGE_BOUNDARY_FMT.format(page_no))
        parts.append(seg)
    return delimiter.join(parts)


def insert_page_markers_at(body: str, boundaries: Sequence[tuple[int, int]]) -> str:
    """INDEX side: insert a `marker\\n\\n` block at each `(page_no, char_start)` offset in `body`
    (offsets ASCENDING, in `body` coordinates) — placing the ruler at the parse-recorded boundaries
    before the index transforms run. Each boundary should sit at a page's content start (just after a
    `\\n\\n` block separator), so the inserted marker becomes its own block."""
    out: list[str] = []
    cursor = 0
    for page_no, start in boundaries:
        out.append(body[cursor:start])
        out.append(_PAGE_BOUNDARY_FMT.format(page_no) + "\n\n")
        cursor = start
    out.append(body[cursor:])
    return "".join(out)


def measure_and_strip_page_markers(marked: str) -> tuple[str, list[tuple[int, int, int]]]:
    """Strip every page-boundary marker block from `marked` and return `(clean, intervals)` where
    `intervals = [(page_no, char_start, char_end)]` is each page's span IN THE CLEAN body. The strip
    drops each marker line + ONE following blank line (the block separator the marker introduced), so
    `clean` is byte-identical to the same body built WITHOUT markers — the golden invariant the
    round-trip relies on for chunk_id stability. Page `i` spans `[start_i, start_{i+1})` (last → end);
    consecutive markers with no content between (an empty page) yield a zero-width interval."""
    lines = marked.split("\n")
    kept: list[str] = []
    starts: list[tuple[int, int]] = []  # (page_no, char_start in the clean body)
    kept_chars = 0  # running Σ len(kept lines), to derive char offsets without re-joining
    i = 0
    while i < len(lines):
        m = PAGE_BOUNDARY_RE.fullmatch(lines[i].strip())
        if m is not None:
            i += 1
            if i < len(lines) and not lines[i].strip():
                i += 1  # drop ONE following blank line (the separator the marker block introduced)
            # the page's content is the line that will be appended NEXT — its offset in the clean
            # body is Σ(kept lengths) + the '\n' joins before it (= len(kept)).
            starts.append((int(m.group(1)), kept_chars + len(kept)))
            continue
        kept.append(lines[i])
        kept_chars += len(lines[i])
        i += 1
    clean = "\n".join(kept)
    intervals: list[tuple[int, int, int]] = []
    for k, (page_no, start) in enumerate(starts):
        end = starts[k + 1][1] if k + 1 < len(starts) else len(clean)
        intervals.append((page_no, start, end))
    return clean, intervals


# ----- Table-RAG linearization helpers (Phase 1) -----------------------------
#
# A `[table-rows]...[/table-rows]` block is the markdown-KV linearization of a
# GFM table emitted by `parse/table_linearize.py::linearize_gfm_tables`. The
# 3-channel contract (see docs/specs/table-rag.md) routes each representation:
#   - `.md` on disk keeps BOTH the raw GFM table AND the `[table-rows]` block;
#   - dense chunks see them as SEPARATE chunks (the block is a distinct
#     paragraph unit);
#   - the FTS/BM25 body sees ONLY the `[table-rows]` block — the raw GFM table
#     it supersedes is stripped (`strip_superseded_gfm_tables`), the inverse of
#     the chart strip, so the KV rows (not the raw table) carry the BM25 signal.

_VALUE_CHARS: tuple[str, ...] = ("$", "%")


def looks_like_value(s: str) -> bool:
    """True if *s* reads as a numeric value (contains a digit or ``$``/``%``).

    Promoted verbatim from `parse/docling_tables._looks_like_value` so the
    table-linearizer (which lives in `parse/`) and the GFM-table strip (which
    lives here in `core/`, called from `index/fts_store`) share ONE heuristic
    without `index/ → parse/` or `core/ → parse/` import edges. The
    `parse/docling_tables` copy operates on `TableCell` flags and stays put;
    this one is the pure-text predicate.
    """
    if any(ch.isdigit() for ch in s):
        return True
    return any(ch in s for ch in _VALUE_CHARS)


# Matches a `[table-rows]...[/table-rows]` block (the table-linearization KV
# payload). Dot-all so the multi-line row body matches.
_TABLE_ROWS_RE = re.compile(
    r"\[table-rows\].*?\[/table-rows\]",
    flags=re.DOTALL,
)
_TABLE_ROWS_OPEN_RE = re.compile(r"\[table-rows\]", flags=re.DOTALL)
_TABLE_ROWS_CLOSE_RE = re.compile(r"\[/table-rows\]", flags=re.DOTALL)


_DENIAL_MARKER_RE = re.compile(
    r"\b(?:do(?:es)?|did) not\s+(?:state|mention|specify|contain|define|provide|describe|include|list)\b"
    r"|\b(?:is|are) not\s+(?:explicitly\s+)?(?:stated|mentioned|specified|defined|provided|described|listed)\b"
    r"|\bne\s+(?:pr[ée]cisen?t?|mentionnen?t?|contienn?en?t?|indiquen?t?)\s+pas\b",
    flags=re.IGNORECASE,
)
_DENIAL_CONTINUATION_RE = re.compile(
    r"\b(?:only that|, only\b|they mention|but (?:do|does|the|it|they)\b|however\b|"
    r"although\b|; (?:the|it|they)\b|instead\b|while\b)",
    flags=re.IGNORECASE,
)


def is_denial_framed_summary(summary: str) -> bool:
    """Whether an EMPTY-CLAIMS draft summary is a DENIAL THAT CONTAINS THE ANSWER —
    the audit-15 M2 false-refusal class: "The chunks do not state which GPUs were used,
    only that training was conducted on up to 8 NVIDIA A100 GPUs" (the answer, framed as
    a denial; verify short-circuits on zero claims → refuse). Trigger = a denial MARKER
    plus a SUBSTANTIVE CONTINUATION; the prompt-canonical counterfactual summary
    ("No literal answer in chunks.") and flat content-free denials do NOT match, so a
    true refusal stays terminal. Used by the answer node's ONE bounded reframe-retry.
    """
    if not summary:
        return False
    return (
        _DENIAL_MARKER_RE.search(summary) is not None
        and _DENIAL_CONTINUATION_RE.search(summary) is not None
    )


_WORLD_KNOWLEDGE_COMPARISON_RE = re.compile(
    r"\b(?:the\s+)?(?:standard|textbook|widely[- ]accepted|commonly[- ]accepted|conventional)\b"
    r"|\binstead of the (?:standard|usual|typical|correct)\b"
    r"|\bthe correct (?:stages?|versions?|process(?:es)?|definitions?|models?|steps?)\b"
    r"|\bdescribed in [A-Za-z+#]+ documentation\b",
    flags=re.IGNORECASE,
)


def relevance_reason_cites_world_knowledge(reason: str) -> bool:
    """Whether a relevance-gate NON-RESPONSIVE reason compares the answer to STANDARD/
    TEXTBOOK knowledge rather than to the asked topic (audit-15 M3). The v2/v3 prompt
    bans judging grounded content against world knowledge, but the 4B's prior overrides
    the rule on strong-prior topics (measured: handwritten-06 rejected for not matching
    "the standard three stages ... described in C++ documentation" 3/3 UNDER the v3 ban).
    The deterministic override is the GUARANTEE the prompt can't give: when the gate's
    own stated reason is a world-knowledge comparison, the verdict is overridden to
    responsive. Relevance is ADVISORY (ADR-0022; runs only on grounded answers), so the
    override can only ship a GROUNDED answer — never a hallucination. Deliberately tight:
    matches comparison-to-external-authority phrasings, not the word "standard" inside a
    quoted topic (a topic-mismatch reason names the asked topic, not "the standard X").
    """
    return _WORLD_KNOWLEDGE_COMPARISON_RE.search(reason) is not None


def strip_table_rows_blocks(text: str) -> str:
    """Remove `[table-rows]...[/table-rows]` blocks (the BM25-side KV duplication of a
    GFM table). Verify-render dedup (audit-15 M1b-i): when a chunk carries BOTH the raw
    GFM table AND its linearization, the doubled numeric noise drowns trailing prose
    from the 4B verifier (measured: a verbatim-present sentence rejected 3/3 with the
    verifier's reason citing only the table). Callers strip ONLY when a GFM pipe-table
    is also present, so a linearization-only chunk (e.g. a split block) keeps its rows.
    """
    return _TABLE_ROWS_RE.sub("", text)


# --- Provenance-scope backstop (audit-18 §9) -------------------------------------
#
# The provenance-class summary-scope breach: a query names its SOURCE ("According to
# the developer guidelines, …") and the shipped answer cites a chunk from a DIFFERENT
# document entirely (tg-13: cited tui/src/log_layer.rs) — the answer asserts a false
# provenance that verify (claims-vs-chunk) cannot see and the 4B relevance judge
# hallucinates past. Audit-18 measured every semantic arm non-separating here; the
# DETERMINISTIC document-identity check below is the lever that works (probed: fires
# on tg-13 under every variant, clears the `sp 800-207`/tg-01 true-provenance cases).
#
# Deliberately TIGHT (the audit-17 v1 lesson — a deterministic refuse amplifies its
# trigger's false-positive rate):
#   - markers: "according to|per|selon|d'après" leading clause ONLY. Bare "In X," is a
#     TOPIC frame in the measured query population ("In the Linux octal permission
#     system, …"), not provenance — excluded wholesale.
#   - X naming a SUB-document artifact (figure/table/module/deck…) is the #256
#     artifact-scope domain; doc-identity cannot adjudicate it → no tokens (fail-open).
#   - generic source nouns ("the course", "the deck", "the documentation") name no
#     specific document; years and <3-char tokens are too unspecific to substring-match
#     ("2026" matches half the vault; "sp" matches "transport") → dropped.
# The CALLER (agents/answering.py) fires only when a usable token matches ≥1 document
# identity in the VAULT (X is adjudicable — it names a real document) while NO CITED
# chunk's identity (doc id + title + heading_path) matches: a vault-named source the
# answer's own support provably does not come from.

_PROVENANCE_SOURCE_RE = re.compile(
    r"^(?:according to|per|selon|d['’]apr[eè]s)\s+(.{3,80}?)\s*,",
    flags=re.IGNORECASE,
)
# Accent-aware (the FR corpora): an ASCII-only class shatters "résumé" into garbage
# fragments ('sum') that can spuriously match vault identities — the independent
# review's B2 finding. Hyphen/underscore stay INSIDE tokens (the "800-207"/"w-9"
# doc-number shape), unlike `atomise` which splits them.
_PROVENANCE_TOKEN_RE = re.compile(r"[a-z0-9àâäçéèêëîïôöùûüÿœæ][a-z0-9àâäçéèêëîïôöùûüÿœæ_-]+")
# EN + FR determiners/function words that carry no source identity.
_PROVENANCE_STOP = frozenset(
    {"the", "a", "an", "this", "that", "its", "of", "in", "on", "for", "and",
     "le", "la", "les", "un", "une", "de", "du", "des", "au", "aux", "ce", "cette"}
)
# Sub-document artifact nouns: X references something INSIDE a document — the #256
# artifact-scope domain, not adjudicable at document level.
_PROVENANCE_ARTIFACT_NOUNS = frozenset(
    {"figure", "figures", "diagram", "diagrams", "diagramme", "diagrammes", "table",
     "tables", "tableau", "tableaux", "chart", "charts", "graph", "graphs", "image",
     "images", "timeline", "example", "examples", "exemple", "exemples", "deck",
     "decks", "slide", "slides", "module", "modules", "chapter", "chapters",
     "chapitre", "section", "sections", "page", "pages", "screenshot", "photo",
     "schéma", "schémas", "graphique", "graphiques", "croquis", "capture"}
)
# Generic source nouns: name no SPECIFIC document, so they cannot adjudicate. The
# colloquial shapes ("the guide", "the report", "the design doc", "per user") were
# probed FALSE-FIRE-CAPABLE against the live vault ("guide" substring-matches the
# `guidelines` doc; "report" the 10-K; "doc" `project_doc.rs`) — and the eval ladder
# cannot see them (no eval query phrases sources colloquially), so the list errs wide.
_PROVENANCE_GENERIC_NOUNS = frozenset(
    {"course", "cours", "document", "documents", "documentation", "text", "texte",
     "manual", "manuel", "paper", "papers", "materials", "material", "corpus",
     "vault", "notes", "book", "books", "livre", "pdf", "file", "files", "source",
     "sources", "transcript", "video", "lecture", "lesson", "doc", "docs", "report",
     "reports", "data", "spec", "specs", "specification", "guide", "guides",
     "standard", "standards", "study", "studies", "readme", "article", "articles",
     "benchmark", "benchmarks", "summary", "résumé", "resume", "user", "users",
     "form", "forms", "content", "information", "info", "étude", "études"}
)
_PROVENANCE_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_IDENT_NORM_RE = re.compile(r"[-_\s./|—–]")
_DIACRITICS_RE = re.compile("[\\u0300-\\u036f]")  # NFD combining marks

# Embedded DOC-NAME reference (audit-19 follow-up, the cross-doc-scope fix). Unlike the leading
# "according to X," clause, a doc-CLASS noun + a digit-bearing IDENTIFIER ("Schedule 8812", "Form
# 1040", "Pub 501", "Form W-4") is UNAMBIGUOUSLY a document reference ANYWHERE in the query — it is
# not a topic frame ("In the Linux permission system…"), so it is safe to detect mid-sentence ("what
# does the 2025 Schedule 8812 list…", "on Form 1040…"). The captured `id` MUST carry a digit (the
# #256 single-token-specificity rule — every real form id has one; a bare "Form A" no-ops). Only the
# IDENTIFIER becomes a usable token (the generic class noun "schedule"/"form" names no specific doc
# and would false-MATCH a cited chunk whose heading mentions another "Schedule N").
_PROVENANCE_DOC_NAME_RE = re.compile(
    # "pub" alone is dropped: it collides with the pub/bar establishment sense ("pub 500 records"),
    # a false-fire surface; "publication" stays (it never reads as a bar).
    r"\b(?:form|schedule|publication|formulaire|annexe)\s+(?:no\.?\s*)?(?P<id>[a-z]{0,3}-?\d[\w-]*)\b",
    flags=re.IGNORECASE,
)
# A comparison query names a doc to CONTRAST, not to scope to — no provenance adjudication.
_COMPARISON_CUE_RE = re.compile(
    r"\b(?:compared?|compares|comparing|differs?|versus|vs\.?|difference between|"
    r"compared to|relative to|as opposed to|par rapport)\b",
    flags=re.IGNORECASE,
)


def _extract_leading_provenance(query: str) -> tuple[str, list[str]] | None:
    """The (raw phrase, usable identity tokens) of a LEADING "according to X," clause, or None.

    The original audit-18 extractor, UNCHANGED — a tight leading-clause match. Returns ``None`` for
    artifact-noun references; an empty token list never escapes; a LONE usable token adjudicates only
    when it carries a digit (the "800-207"/"w-9" doc-number shape, the #256 specificity gate)."""
    m = _PROVENANCE_SOURCE_RE.match(query.strip())
    if not m:
        return None
    phrase = m.group(1).strip()
    toks = [
        t for t in _PROVENANCE_TOKEN_RE.findall(phrase.lower())
        if t not in _PROVENANCE_STOP
    ]
    if not toks or any(t in _PROVENANCE_ARTIFACT_NOUNS for t in toks):
        return None
    usable = [
        t for t in toks
        if t not in _PROVENANCE_GENERIC_NOUNS
        and not _PROVENANCE_YEAR_RE.match(t)
        and len(t) >= 3
    ]
    if not usable:
        return None
    if len(usable) == 1 and not any(ch.isdigit() for ch in usable[0]):
        return None
    return phrase, usable


def _extract_doc_name_reference(query: str) -> tuple[str, list[str]] | None:
    """An EMBEDDED doc-class-noun reference ("Schedule 8812", "Form 1040"), or None.

    A doc-class noun + digit-bearing identifier is unambiguously a document reference anywhere in the
    query (the cross-doc-scope fix). Conservative: >= 2 DISTINCT doc references, or a comparison cue
    ("compare X and Y"), yield None — the query spans docs, not scopes to one. Only the IDENTIFIER is
    a usable token (the class noun is dropped; it would false-match a cited chunk citing another
    "Schedule N"); the identifier must be >= 3 chars (drops the unspecific "Pub 17" short-id surface)
    and must NOT be a bare YEAR — "publication 2024" / "schedule 2024" is a NON-scoping shape and a
    year substring-matches half the vault (the same `_PROVENANCE_YEAR_RE` guard the leading clause applies)."""
    matches = list(_PROVENANCE_DOC_NAME_RE.finditer(query))
    if not matches:
        return None
    if len({_fold_identity(m.group("id")) for m in matches}) >= 2:
        return None  # >= 2 distinct doc references → ambiguous which is the source
    if _COMPARISON_CUE_RE.search(query):
        return None
    m = matches[0]
    usable = [
        t for t in _PROVENANCE_TOKEN_RE.findall(m.group("id").lower())
        if t not in _PROVENANCE_STOP
        and len(t) >= 3
        and any(ch.isdigit() for ch in t)
        and not _PROVENANCE_YEAR_RE.match(t)
    ]
    if not usable:
        return None
    return m.group(0).strip(), usable


def extract_provenance_source(query: str) -> tuple[str, list[str]] | None:
    """The (raw phrase, usable identity tokens) of a named source, or ``None``.

    Two recognizers: (1) a LEADING "according to|per|selon|d'après X," clause (audit-18, unchanged);
    (2) an EMBEDDED doc-class-noun reference ("…the 2025 Schedule 8812…", the cross-doc-scope fix).
    The leading clause takes precedence; the embedded recognizer is a fallback so existing behavior is
    byte-identical wherever (1) matches. The CALLER (`agents/answering.py`) fires the refusal only when
    the named source matches ≥ 1 vault document identity AND no CITED chunk carries it."""
    return _extract_leading_provenance(query) or _extract_doc_name_reference(query)


def _fold_identity(s: str) -> str:
    """Lowercase, strip diacritics (NFD), drop separators — so 'réseau' matches a
    title carrying 'réseau' or 'reseau', and '800-207' matches 'nist-sp-800-207'."""
    import unicodedata

    folded = _DIACRITICS_RE.sub("", unicodedata.normalize("NFD", s.lower()))
    return _IDENT_NORM_RE.sub("", folded)


_CONTENT_HASH_PREFIX_RE = re.compile(r"^[0-9a-f]{8}-")


def strip_content_hash(doc_id: str) -> str:
    """The identity-bearing part of a content-addressed `<sha8>-<stem>` doc_id — drop the random
    8-hex prefix. Without this, a short numeric provenance token ("941") substring-COLLIDES with the
    random hash inside an UNRELATED doc_id (measured: 29% of 3-digit numbers hit ≥1 vault doc's hash,
    e.g. "941" in "2941523b-lib") → a spurious provenance-scope refusal. The stem ("f1040" → "1040")
    and the title carry the REAL identity; the hash never should. A doc_id with no sha8 prefix is
    returned unchanged."""
    return _CONTENT_HASH_PREFIX_RE.sub("", doc_id)


def provenance_tokens_match(tokens: list[str], identity_blob: str) -> bool:
    """Whether ANY usable source token appears in *identity_blob* after accent folding
    and separator normalization. Substring direction: token-in-blob (a doc slug
    concatenates words). Callers should pass `strip_content_hash(doc_id)` (not the raw doc_id)
    so a numeric token can't collide with the random sha8 prefix.

    KNOWN LIMITATION (not cleanly fixable): the folded SUBSTRING match is load-bearing for multi-part
    doc numbers ("800-207" → "nist-sp-800-207"), so a short numeric token still matches a doc whose
    stem/title legitimately CONTAINS that number as a CROSS-REFERENCE — "1040" matches Schedule 8812
    (stem "f1040s8", title "...(Form 1040)") because it really is a 1040 schedule. This is harmless
    AS LONG AS the doc that IS Form 1040 keeps "1040" in its OWN identity (so a Form-1040 answer
    self-protects via the cited-chunk check). RETITLE POLICY: never retitle a doc to a name that
    DROPS its own identifier — that is the only way this cross-reference match becomes a false
    refusal (a token that misses its home doc + hits a cross-referencing sibling)."""
    blob = _fold_identity(identity_blob)
    return any(_fold_identity(t) in blob for t in tokens)


def table_rows_spans(text: str) -> list[tuple[int, int]]:
    """Return `(start, end)` char offsets of each
    `[table-rows]...[/table-rows]` block in *text*.

    Mirrors `chart_extracted_spans` exactly — including the orphan-opener /
    orphan-closer truncation tolerance — because the chunker's char-split
    (`MAX_CHUNK_CHARS`) WILL split an oversized `[table-rows]` block across
    chunks (unlike chart blocks, which are exempt). A split half therefore
    carries only an opener (extends to end-of-text) or only a closer (extends
    from start-of-text), and any consumer that needs to recognise a partial
    block still gets the span. Order-stable (sorted by start offset).
    """
    spans: list[tuple[int, int]] = list((m.start(), m.end()) for m in _TABLE_ROWS_RE.finditer(text))
    consumed_open: set[int] = {start for start, _ in spans}
    consumed_close: set[int] = {end for _, end in spans}

    for m in _TABLE_ROWS_OPEN_RE.finditer(text):
        if m.start() not in consumed_open:
            spans.append((m.start(), len(text)))
    for m in _TABLE_ROWS_CLOSE_RE.finditer(text):
        if m.end() not in consumed_close:
            spans.append((0, m.end()))

    spans.sort(key=lambda s: s[0])
    return spans


# A GFM table-shape line group: a header pipe-row, a delimiter row, then one or
# more data pipe-rows. The delimiter row is the disambiguator (a run of
# pipes / dashes / colons / whitespace with at least one dash), mirroring
# `index.chunker._GFM_DELIM_RE`. Matched line-anchored + multiline so a table
# embedded in prose is found at its own line boundaries.
_GFM_TABLE_RE = re.compile(
    r"^[ \t]*\|.*\|[ \t]*\n"  # header pipe-row
    r"[ \t]*\|?[ \t:|-]*-[ \t:|-]*\|?[ \t]*\n"  # delimiter row (>=1 dash)
    r"(?:[ \t]*\|.*\|[ \t]*\n?)+",  # >=1 data pipe-row
    flags=re.MULTILINE,
)


def strip_superseded_gfm_tables(text: str) -> str:
    """Remove a GFM table that is immediately followed by a `[table-rows]`
    block (the table-linearization payload).

    Used at the **index layer** (`index.fts_store::upsert`) only, composed
    with `strip_chart_extracted_for_index`: the `[table-rows]` KV rows carry
    the BM25 signal, so leaving the raw GFM table in the FTS body would
    double-count every value token (term-frequency inflation — the same class
    of bug `strip_chart_extracted_for_index` fixes for chart blocks). This is
    its inverse: chart-strip drops the derived block and keeps the source;
    table-strip drops the *source* table and keeps the derived `[table-rows]`.

    Only strips a GFM table when a `[table-rows]` block follows it across an
    optional blank line — an un-linearized table (no following block) is left
    intact so its raw value tokens still reach BM25. Idempotent: text with no
    superseded table passes through unchanged.
    """

    def _replace(m: re.Match[str]) -> str:
        rest = text[m.end() :]
        # The linearizer separates the table from its block by a blank line
        # (`\n\n`); the table regex consumes a trailing newline, so tolerate
        # any further leading whitespace/newlines before the opener.
        if _TABLE_ROWS_OPEN_RE.match(rest.lstrip("\n \t") if rest else ""):
            # Drop the raw table but keep the structural newline so the
            # `[table-rows]` block stays a distinct paragraph in the body.
            return ""
        return m.group(0)

    return _GFM_TABLE_RE.sub(_replace, text)


# Matches a Markdown ATX heading line (1-6 `#` followed by space and
# heading text). Same shape as `index.chunker._HEADING_RE`; lifted
# here so `enrich.citations` can use it for section-anchor discovery
# without violating the `enrich/ → index/` import direction.
#
# Scoped to horizontal whitespace (`[ \t]`) rather than `\s` because
# the latter matches `\n` too, which would cause an empty heading like
# `## ` (no text) + blank line + `Prose.` to silently match as `##
# Prose.` (the regex `\s+(.+?)\s*$` swallows the newline and grabs
# the next-line's content). Post-P4.1 audit (2026-05-23 webui test
# pin) confirmed this was an actual hazard. The chunker uses the same
# regex via `index.chunker._HEADING_RE`; the tightening applies
# uniformly so empty-text headings no longer steal subsequent
# paragraphs.
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)


def extract_heading_texts(body: str, *, skip_chart_blocks: bool = True) -> list[str]:
    """Return the heading text (sans `#` prefix) of every Markdown
    heading in `body`, in document order.

    Used by the P4.1 wikilink section-anchor writer
    (`enrich.citations::insert_wikilinks`) to discover which headings
    the target document has, so the citation-insertion step can emit
    `[[doc#section]]` when the citation context references a
    specific section.

    `skip_chart_blocks=True` (default) filters out the inert
    `# H1` labels Nemotron-Parse emits INSIDE `[chart-extracted]`
    blocks (same defense as the P3.3 v7 chunker fix — see
    `chart_extracted_spans` above). Set to `False` only if you
    explicitly want chart-figure labels in the heading list
    (almost no caller should).

    Headings are stripped of leading/trailing whitespace. Empty
    or all-whitespace headings are skipped. Returns headings in
    document order; duplicates ARE preserved (a doc with two
    `## Methods` sections returns `["Methods", "Methods"]`) so
    callers that care about uniqueness can dedupe themselves.
    """
    spans = chart_extracted_spans(body) if skip_chart_blocks else []
    out: list[str] = []
    for m in _MARKDOWN_HEADING_RE.finditer(body):
        if is_inside_any_span(m.start(), spans):
            continue
        text = m.group(2).strip()
        if text:
            out.append(text)
    return out


# A chunk is "name-only" when it carries NO substantive prose — only a heading and/or a bare
# list of short name fragments (e.g. a slide that lists access-control type NAMES:
# `### Contrôle d'accès` + `- Role-Based Access Control (RBAC)` + `- Attribute-Based …`). A claim
# whose cited chunk is name-only is grounded only by the coincidental presence of the entity NAME,
# not by any sentence that establishes the claim — so the `/ask` verify node's name-only backstop
# AND the reason-then-ground bridge DEMOTE such a (behavioural) claim from grounding (ADR-0016,
# via `claim_grounded_only_by_name` below; the bridge's separate present-as-answer guard is the
# narrowed defense-in-depth layer).
_NAME_ONLY_MIN_SENTENCE_WORDS: Final[int] = 8  # ≥ this many words on one line ⇒ a real sentence
_NAME_ONLY_MIN_LIST_LINES: Final[int] = 2  # floor: need ≥ this many short non-heading lines
# A leading list enumerator ("3.", "4)") is structural scaffolding, not a content word — strip it
# before the per-line word count so a numbered SUB-HEADING ("3. Protection du plan de contrôle
# (Control Plane)" = 8 tokens ONLY because "3." counts) reads as the short label it is. Numeric,
# 1-3 digits, a mandatory `.`/`)`, a mandatory trailing space → decimals ("3.14 ") and dotted
# versions ("3.2.1 ") never match.
_LEADING_ENUMERATOR_RE: Final[re.Pattern[str]] = re.compile(r"^\d{1,3}[.)]\s+")


def is_name_only_chunk(text: str) -> bool:
    """True when `text` has no substantive prose sentence — only a heading and/or ≥2 short
    name-like list lines. False when a real descriptive line, a markdown table, or a
    `[chart-extracted]` / `[table-rows]` block is present (structured data is substantive support).

    Deliberately DIVERGES from `parse/*_worker.py::_looks_like_prose_heading` (`>15 words OR
    (≥4 words AND ends in .!?)`): slide bullets lack terminal punctuation, and a real ~12-word
    descriptive line without a period (e.g. "User accounts must be configured locally on each
    device which is not scalable") MUST count as substantive — the terminal-punct rule would
    false-flag such a well-grounded chunk. So the test here is a per-line word count (≥8 words)
    with a ≥2-short-line floor; do NOT "unify" the two heuristics.

    Conservative by construction: it can only cause a present-as-answer claim to be HELD BACK
    (fall back to the labelled analysis), never assert anything. A single terse sentence or a
    heading-only chunk is NOT confidently a name list, so it is kept. Known residual (documented):
    `.split()` is Latin-centric (CJK no-space prose is under-counted); the floor mitigates the
    single-line case. The repo corpora are FR+EN.
    """
    # Structured data is substantive support — never name-only (marker-first short-circuit).
    if "[table-rows]" in text or "[chart-extracted]" in text or _GFM_TABLE_RE.search(text):
        return False
    list_like_short_lines = 0
    has_any_content_line = False
    for raw in text.splitlines():
        line = IMAGE_PLACEHOLDER_RE.sub("", raw).strip()
        if not line:
            continue
        if _MARKDOWN_HEADING_RE.match(line):
            continue  # a heading is a label, not a sentence that supports a claim
        has_any_content_line = True
        stripped = line.lstrip("-*+>").strip().strip("|").strip()
        stripped = _LEADING_ENUMERATOR_RE.sub("", stripped, count=1)
        if len(stripped.split()) >= _NAME_ONLY_MIN_SENTENCE_WORDS:
            return False  # a substantive sentence exists ⇒ NOT name-only
        list_like_short_lines += 1
    # Floor: only name-only when ≥2 short non-heading lines AND no long line. A 1-line terse chunk
    # or a heading-only chunk is NOT confidently a name list ⇒ keep it (the safe direction).
    return has_any_content_line and list_like_short_lines >= _NAME_ONLY_MIN_LIST_LINES


# Membership / existence / definition phrasings — a claim that merely places its subject IN a
# list (or categorises it) IS grounded by a bare name-list, so these KEEP the claim. Checked
# first; bilingual (EN+FR). Substrings matched against a space-padded lowercased claim.
_MEMBERSHIP_MARKERS: frozenset[str] = frozenset(
    {
        "is one of",
        "are one of",
        "is among",
        "are among",
        "is included",
        "are included",
        "is listed",
        "are listed",
        "listed among",
        "is a type of",
        "is a kind of",
        "is a model",
        "is a method",
        "is a mechanism",
        "is an access control",
        "is an example of",
        "such as",
        "includes",
        "include ",
        "including",
        # FR
        "est un de",
        "est une de",
        "fait partie",
        "font partie",
        "figure parmi",
        "est un type de",
        "est une sorte de",
        "inclut",
        "comprend",
        "parmi les",
    }
)

# Behavioral / property / comparative PREDICATE markers — a claim asserting one of these about
# its subject needs a real sentence (or a table row) to ground; a bare name-list cannot support
# it. Curated for PRECISION (fail-open on misses): generic copulas ("is"/"are") are deliberately
# EXCLUDED. Bilingual (EN+FR). Space-padded substring match.
_BEHAVIORAL_MARKERS: frozenset[str] = frozenset(
    {
        # behaviour / action verbs
        " assigns ",
        " assign ",
        " evaluates ",
        " evaluate ",
        " allows ",
        " allow ",
        " enables ",
        " enable ",
        " requires ",
        " require ",
        " provides ",
        " provide ",
        " supports ",
        " support ",
        " grants ",
        " grant ",
        " denies ",
        " deny ",
        " controls ",
        " control ",
        " enforces ",
        " enforce ",
        " processes ",
        " process ",
        " operates ",
        " operate ",
        " functions ",
        " relies ",
        " rely ",
        " uses ",
        " use ",
        " struggles ",
        " lacks ",
        " lack ",
        " introduces ",
        " validates ",
        " restricts ",
        " prevents ",
        " mitigates ",
        " determines ",
        " adapts ",
        " reacts ",
        " considers ",
        # property / behaviour phrases
        " based on ",
        " works by ",
        " used for ",
        " acts as ",
        " responsible for ",
        " capable of ",
        " is static",
        " is dynamic",
        " is rigid",
        " is scalable",
        " is flexible",
        " not scalable",
        " in real-time",
        " real time",
        # comparative
        " superior",
        " better ",
        " worse ",
        " faster",
        " slower",
        " outperforms ",
        " more granular",
        " less granular",
        " compared to",
        " than ",
        # FR action / property / comparative
        " assigne ",
        " assignent ",
        " évalue ",
        " évaluent ",
        " permet ",
        " permettent ",
        " nécessite ",
        " nécessitent ",
        " fournit ",
        " contrôle ",
        " contrôlent ",
        " repose sur ",
        " basé sur ",
        " basée sur ",
        " utilise ",
        " utilisent ",
        " restreint ",
        " empêche ",
        " détermine ",
        " en fonction de ",
        " supérieur",
        " meilleur",
        " plus rapide",
        " moins ",
    }
)


def claim_asserts_behavior(claim_text: str) -> bool:
    """True when the claim asserts a BEHAVIOUR / PROPERTY / COMPARISON about its subject — as
    opposed to a pure MEMBERSHIP / EXISTENCE / definition claim ("X is one of the listed items",
    "X is an access control model").

    Used by the `/ask` `verify` node's NAME-ONLY grounding backstop: a claim cited to an
    `is_name_only_chunk` chunk is demoted ONLY when it asserts behaviour a bare name-list cannot
    support. **FAIL-OPEN: returns False (= keep the claim) on a membership phrasing OR an
    unrecognised one** — so the backstop only ever demotes a claim it is CONFIDENT is behavioural,
    never a membership claim and never an unknown one. The worst case of a marker-set coverage gap
    is the status-quo over-grounding (a behavioural claim slips through), NEVER a new over-refusal.
    Combined with demotion-only, the backstop is over-refusal-safe BY CONSTRUCTION.

    Keys on the PREDICATE CLASS (a curated bilingual marker set), NOT lexical overlap with the
    chunk — robust to the FR-chunk / EN-claim shape common in this vault. Generic copulas
    ("is"/"are") are excluded. Known residual (documented, cf. the numeric backstop): a behavioural
    predicate phrased outside the marker set slips through. Do NOT "unify" the marker sets with
    `is_name_only_chunk`'s line heuristic — they answer different questions.
    """
    low = " " + " ".join(claim_text.lower().split()) + " "
    if any(m in low for m in _MEMBERSHIP_MARKERS):
        return False  # membership / existence / definition → a name-list grounds it
    return any(m in low for m in _BEHAVIORAL_MARKERS)


def claim_grounded_only_by_name(claim_text: str, chunk_text: str) -> bool:
    """The deterministic name-only DEMOTION rule, shared by the `/ask` `verify` node and the
    reason-then-ground bridge: True iff the cited chunk merely NAMES the subject
    (`is_name_only_chunk` — a bare list/heading, no substantive sentence) AND the claim asserts a
    behaviour/property/comparison a bare name-list cannot establish (`claim_asserts_behavior`).

    Membership/existence/definition claims (a name-list DOES support them) and any unrecognised
    predicate → False (KEEP). Pure + fail-open by construction (both operands fail-open), so a
    demoter using it can only ever flag a claim it is CONFIDENT is behavioural-on-a-name-list —
    never a membership claim, never an unknown one. Order matches the historical inline checks
    (chunk shape first via `and` short-circuit, then predicate)."""
    return is_name_only_chunk(chunk_text) and claim_asserts_behavior(claim_text)


# Bilingual (FR+EN) stopwords + a diacritics-/hyphen-aware word tokenizer, used by the
# artifact-scope qualifier resolver (`agents/artifact_scope.py`). Lives in `core/` (rather
# than `agents/`) so any module may share it without importing UP from `agents/` (the
# documented import direction). The token shape mirrors the FTS index tokenizer
# `unicode61 remove_diacritics 2` (folds accents; splits hyphen/apostrophe).
STOPWORDS: frozenset[str] = frozenset(
    {
        # FR articles / connectors
        "de",
        "du",
        "des",
        "d",
        "l",
        "la",
        "le",
        "les",
        "au",
        "aux",
        "un",
        "une",
        "ce",
        "cette",
        "ces",
        "son",
        "sa",
        "ses",
        # EN articles / connectors / filler
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "for",
        "to",
        "and",
        "or",
        "this",
        "that",
        "with",
        "from",
        # generic corpus words that don't disambiguate a document
        "cours",
        "course",
        "paper",
        "example",
        "project",
        "status",
        "shown",
        "above",
        "below",
        "following",
        "main",
        "given",
        "diagram",
        "diagramme",
        "figure",
        "chart",
        "page",
    }
)

# Keep only word chars FTS5 `unicode61 remove_diacritics 2` indexes: ASCII alphanumerics +
# the FR accented letters. Everything else (punctuation, FTS5 operator chars) is dropped.
_ATOM_KEEP_RE = re.compile(r"[^0-9a-zàâäçéèêëîïôöùûüÿœæ]")
_ATOM_SPLIT_RE = re.compile(r"[-'’]")


def atomise(word: str) -> list[str]:
    """Lowercase a surface word and split it on hyphen/apostrophe into atomic tokens
    (``coupe-feu`` → ``coupe``, ``feu``; ``d'attaque`` → ``attaque``). FTS5
    ``unicode61 remove_diacritics 2`` folds these the same way at index time, so atomic
    tokens match the indexed terms regardless of accent/hyphen."""
    out: list[str] = []
    for part in _ATOM_SPLIT_RE.split(word.lower()):
        cleaned = _ATOM_KEEP_RE.sub("", part)
        if cleaned:
            out.append(cleaned)
    return out


# ----- Number coercion (Table-RAG Phase-2 number grammar, spec §2) -----------
# Lives in `core/` so BOTH the index-time table machinery (`core/table_linearize`,
# `index/table_store`) and the agents (`agents/table_sql` re-exports it) share ONE
# parser without inverting the import direction. Moved here from agents/table_sql
# 2026-05-31 for the column-under-split detector, which is core/-only.
_SCALE_WORDS: dict[str, float] = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_SCALE_LETTERS: dict[str, float] = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}
_NUMERIC_BODY_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def coerce_number(cell: str) -> float | None:
    """Parse a table cell's text to a float per the Phase-2 number grammar, or
    None when the cell does not read as a single number.

    The single documented grammar (spec §2), applied in order:
      - strip a leading currency symbol (`$`/`€`/`£`);
      - accounting negatives: a fully-parenthesised body `(1,234)` → `-1234`;
      - a trailing `%` is KEPT as the percent value (`45%` → `45.0`);
      - a trailing scale word (`thousand|million|billion|trillion`) or letter
        (`K|M|B|T`) multiplies by 1e3/1e6/1e9/1e12;
      - thousands separators (`,`) are removed;
      - the remaining body must be a plain (optionally signed/decimal) number,
        else None.
    Pure-sync. Verbatim-text in, float-or-None out — the load-bearing 10-K
    shapes (`$22.5 billion`, `(1,234)`, `45%`, `1,000,000`) are unit-tested.
    """
    s = cell.strip()
    if not s:
        return None

    negative = False
    # Accounting negative: a fully-parenthesised body.
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()

    # Leading currency symbol.
    if s and s[0] in ("$", "€", "£"):
        s = s[1:].strip()
        # A leading sign may sit inside the currency symbol: `$-5`.
        if s.startswith("-"):
            negative = not negative
            s = s[1:].strip()

    if not s:
        return None

    # Leading sign on the bare body.
    if s.startswith("-"):
        negative = not negative
        s = s[1:].strip()
    elif s.startswith("+"):
        s = s[1:].strip()

    percent = False
    if s.endswith("%"):
        percent = True
        s = s[:-1].strip()

    scale = 1.0
    lowered = s.lower()
    matched_scale = False
    for word, mult in _SCALE_WORDS.items():
        if lowered.endswith(word):
            scale = mult
            s = s[: -len(word)].strip()
            matched_scale = True
            break
    if not matched_scale and len(s) >= 2 and s[-1].lower() in _SCALE_LETTERS:
        # Only treat a trailing letter as a scale suffix when a digit
        # precedes it (`2.5B`) — not a bare token like "B".
        if s[-2].isdigit():
            scale = _SCALE_LETTERS[s[-1].lower()]
            s = s[:-1].strip()

    # Remove thousands separators.
    s = s.replace(",", "")

    if not _NUMERIC_BODY_RE.match(s):
        return None

    value = float(s) * scale
    if negative:
        value = -value
    # `percent` is informational — the value is already the percent number.
    _ = percent
    return value


# A cell whose `coerce_number` reading is UNAMBIGUOUS: plain digits OR US-grouped
# thousands (`\d{1,3}(,\d{3})*`) with at most one decimal, after stripping
# currency / sign / parens / a trailing scale-word|letter | `%`.
_CANONICAL_NUMERIC_CORE_RE = re.compile(r"^(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?$")
_CANONICAL_SCALE_SUFFIX_RE = re.compile(
    r"\s*(?:%|thousand|million|billion|trillion|[kmbt])\s*$", flags=re.IGNORECASE
)


def is_canonical_number_cell(cell: str) -> bool:
    """True iff `cell` reads as an UNAMBIGUOUS US-convention number.

    `coerce_number` is deliberately LENIENT — it strips ALL commas, so a malformed
    `1,2,3` becomes `123` and a mixed-separator European `1.234,56` misreads.
    Because the Table-RAG sqlite `__num` column is built by the SAME `coerce_number`
    the aggregate recompute uses, the recompute-AGREEMENT check is structurally
    BLIND to such a misread (both arms agree on the wrong number). The aggregate /
    superlative gate calls this on each contributing cell and REFUSES when a cell
    coerces but is NOT canonical — closing the malformed / mixed-separator wrong-
    value ship. It does NOT resolve LOCALE ('1.000' European decimal-vs-thousands)
    or UNIT ('5m' metres-vs-million) ambiguity — those need context the system
    lacks and do not occur in the US-format corpora; documented as a residual.

    Strips one balanced paren pair, leading currency/sign, and a trailing
    scale/percent, then requires the core be plain digits OR properly 3-grouped
    thousands with at most one decimal point. Pure-sync."""
    s = cell.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    s = re.sub(r"^[\s$€£+\-]*", "", s)  # leading currency + sign, either order
    s = _CANONICAL_SCALE_SUFFIX_RE.sub("", s).strip()
    return bool(s) and _CANONICAL_NUMERIC_CORE_RE.match(s) is not None


# --- ASR transcript normalization (ADR-0017; spec docs/specs/audio-asr-route.md) ---------
#
# Non-lexical FILLER interjections stripped from a single ASR transcript SEGMENT. DELIBERATELY
# CONSERVATIVE: only unambiguous hesitation / non-lexical interjections. Ambiguous discourse
# markers (EN "like"/"you know"/"so"/"right"; FR "ben"/"bah"/"quoi"/"voilà"/"hein") and
# backchannels ("mhm"/"uh-huh") are EXCLUDED — they can be lexical and dropping them would
# change meaning. (`ahem` is excluded too — it is often an INTENTIONAL attention-getter.)
_TRANSCRIPT_FILLERS: Final[frozenset[str]] = frozenset(
    {"um", "umm", "ummm", "uh", "uhh", "uhhh", "uhm", "erm", "euh", "euhh", "heu", "heuh"}
)
# A filler is a WHOLE token bounded by NON-(word-char-or-hyphen) on both sides — so a filler
# inside a word ("umbrella") or a hyphenated backchannel ("uh-huh") is NEVER clipped; `\w` is
# unicode, so "euh" never bites an accented FR word. An immediately-following comma (the
# comma-set-off / sentence-initial form "Um, so" / "I, um, think") is consumed with it.
_FILLER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w-])(?:"
    + "|".join(sorted(_TRANSCRIPT_FILLERS, key=len, reverse=True))
    + r")(?![\w-])[ \t]*,?",
    re.IGNORECASE,
)
# Mop up the punctuation/space RESIDUE a removed filler leaves — these touch ONLY punctuation
# + whitespace, never a content word. Order matters (run top-to-bottom).
_DOUBLED_DELIM_RE: Final[re.Pattern[str]] = re.compile(r"([,;:])(?:\s*[,;:])+")  # ", ," -> ","
_COMMA_BEFORE_TERMINAL_RE: Final[re.Pattern[str]] = re.compile(r",\s*([.!?])")  # "Yes,." -> "Yes."
_DOUBLED_TERMINAL_RE: Final[re.Pattern[str]] = re.compile(r"([.!?])\s*\.")  # "think.." -> "think."
_SPACE_BEFORE_PUNCT_RE: Final[re.Pattern[str]] = re.compile(r"\s+([,.;:!?])")
# A sentence never legitimately LEADS with terminal punctuation, so a leading `.!?` residue
# (from a filler that WAS its own "sentence", "Um. Okay") is an artifact — but a leading COMMA
# is NOT stripped (it can be legitimate continuation: ", and then it works").
_LEADING_TERMINAL_RE: Final[re.Pattern[str]] = re.compile(r"^\s*[.!?]+\s*")
_MULTISPACE_RE: Final[re.Pattern[str]] = re.compile(r"[ \t]{2,}")


def normalize_transcript_text(text: str) -> str:
    """Deterministically clean ASR noise from a SINGLE transcript SEGMENT's text (ADR-0017;
    spec docs/specs/audio-asr-route.md §"Transcript normalization").

    Removes non-lexical filler interjections (`um`/`uh`/`euh`/…, EN+FR) plus the
    whitespace/punctuation residue they leave. **It preserves all LEXICAL content** — a content
    word is never clipped or dropped (the `(?<![\\w-])…(?![\\w-])` boundaries match only whole
    filler tokens), and nothing is ever added — and it is REPRODUCIBLE (a pure function →
    byte-identical output → stable content-addressed `chunk_id`s, invariant to the interpreter
    hash seed). The **verbatim raw transcript stays in the ASR cache** as the faithfulness anchor.

    Two honest limits (bounded by the cached raw + the deferred LLM pass): in the rare case a
    standalone filler token is also a CAPITALISED homograph of a real word/name (`Heu` a surname,
    `UH` shouted) it is removed too — an accepted conservative loss; and casing after a removed
    sentence-initial filler is left as-is. It does NOT collapse content-word stutters
    ("the the cat"), split run-ons, or restructure — those need semantics (the LLM pass, §15).

    **Apply PER-SEGMENT, to a single segment's raw text — NOT to the assembled `## [mm:ss]` body**
    (it collapses blank lines, which would flatten the heading/section structure the chunker
    splits on)."""
    cleaned = _FILLER_RE.sub("", text)
    cleaned = _DOUBLED_DELIM_RE.sub(r"\1", cleaned)
    cleaned = _COMMA_BEFORE_TERMINAL_RE.sub(r"\1", cleaned)
    cleaned = _DOUBLED_TERMINAL_RE.sub(r"\1", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = _MULTISPACE_RE.sub(" ", cleaned)
    # Re-flow per line: strip a leading terminal-punct residue + trim; drop emptied lines.
    lines = [_LEADING_TERMINAL_RE.sub("", ln).strip() for ln in cleaned.split("\n")]
    return "\n".join(ln for ln in lines if ln)


# ----- Faithful-transform guard (ADR-0017 §3/§15 — BANKED primitive) ------------
# A deterministic gate for "did an LLM only re-FORMAT this text, or did it change the
# content?": every candidate is checked against its verbatim baseline. Built for an
# OPTIONAL LLM transcript-structuring pass (paragraphing / sentence-splitting / light
# disfluency smoothing). **That pass was a VALIDATED NEGATIVE (2026-06-04):**
# `large-v3-turbo` already punctuates + sentence-segments, so the 4B returned every real
# block verbatim — no value to add (see the `transcript-structuring-negative-2026-06-04`
# memory + `docs/specs/audio-asr-route.md` §15). The pass was reverted; this guard + its
# fidelity scorers are KEPT as a reusable, adversarially-hardened primitive for any future
# faithful-rewrite feature (or a non-punctuating ASR model). DELIBERATELY NOT filtered by
# STOPWORDS: `STOPWORDS` carries lecture CONTENT words (`project`/`page`/`course`/`figure`)
# for FTS/artifact-scope disambiguation — excluding them would let an edit drop them past
# the gate. Tokenise EVERYTHING via `atomise`.


def content_tokens(text: str) -> list[str]:
    """The ordered content-token sequence of `text` — each whitespace word run through `atomise`
    (lowercase, diacritics/punctuation dropped, hyphen/apostrophe split). The single tokeniser the
    transcript-structuring faithfulness guard AND its fidelity eval share, so both measure the same
    invariant. No stopword filtering (see the module note): every lexical token counts as content."""
    return [tok for word in text.split() for tok in atomise(word)]


# A numeric run = digits with internal separators (`3.14`, `1,000`, `192.168.0.1`, `12:30`). `atomise`
# STRIPS `.`/`,`, so `3.14` and `314` collapse to the SAME content token — a number value/format change
# is invisible to `content_tokens`. Since the structuring pass must never touch a number (changing one
# is a fabrication, the canonical hallucination), numbers are checked VERBATIM and separately.
_NUMBER_RUN_RE: Final[re.Pattern[str]] = re.compile(r"\d[\d.,:/]*\d|\d")


def _number_surfaces(text: str) -> list[str]:
    """Ordered verbatim numeric runs in `text` — the surface forms the structuring pass must leave
    byte-identical (a `3.14`→`314` or IP/port/value edit is content change `content_tokens` can't see)."""
    return _NUMBER_RUN_RE.findall(text)


def _collapse_adjacent(tokens: list[str]) -> list[str]:
    """Collapse runs of an immediately-repeated token to a single occurrence — the ONLY content-token
    deletion the structuring pass may perform (a stutter `the the`→`the`). A NON-adjacent duplicate
    survives, so dropping one is caught as real content loss."""
    out: list[str] = []
    for tok in tokens:
        if not out or out[-1] != tok:
            out.append(tok)
    return out


def structure_block_is_faithful(structured: str, baseline: str) -> bool:
    """Deterministic per-block faithfulness gate for the LLM structuring pass (ADR-0017 §3).

    `structured` may differ from `baseline` ONLY by re-formatting (paragraph breaks, sentence splits,
    punctuation, capitalisation) and by collapsing an IMMEDIATELY-repeated content-word stutter. Two
    independent deterministic checks, both must hold — else the caller falls back to the verbatim
    baseline block, so a hallucinated / dropped / reordered / value-changed word can never reach the
    grounding text (HARD-gate-safe):

    1. **Numbers verbatim** — the ordered VERBATIM numeric runs (`_number_surfaces`) are identical.
       `atomise` strips `.`/`,`, so `3.14` ≡ `314` as a `content_token`; this separate check is what
       actually rejects a number value/format change (the canonical fabrication).
    2. **Content modulo adjacent-stutter** — `content_tokens` with ADJACENT duplicates collapsed
       (`_collapse_adjacent`) must be EQUAL on both sides. Equality (not subsequence) rejects
       ADDITIONS, real unique-content LOSS, and REORDERING; the adjacent-collapse is the sole
       sanctioned deletion — so a dropped NON-adjacent duplicate is correctly rejected too.

    NO stopword filter (see the module note). The raw transcript is also cached upstream as the deeper
    audit anchor; this gate compares against the deterministic baseline the structuring step was handed.

    Known bounded residual: a SYMBOLIC operator change (`5 < 3`→`5 > 3`) is invisible (operators atomise
    to nothing and aren't numeric runs) — extremely rare in speech-transcribed text, and surfaced by the
    human diff review; revisit only if symbol-heavy transcripts enter scope."""
    if _number_surfaces(structured) != _number_surfaces(baseline):
        return False
    return _collapse_adjacent(content_tokens(structured)) == _collapse_adjacent(
        content_tokens(baseline)
    )
