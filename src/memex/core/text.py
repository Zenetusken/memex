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
# not by any sentence that establishes the claim — so the reason-then-ground bridge keeps such
# claims OUT of a PRESENT-AS-ANSWER surface (ADR-0016; presentation-only, never affects grounding).
_NAME_ONLY_MIN_SENTENCE_WORDS: Final[int] = 8  # ≥ this many words on one line ⇒ a real sentence
_NAME_ONLY_MIN_LIST_LINES: Final[int] = 2  # floor: need ≥ this many short non-heading lines


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
        "is one of", "are one of", "is among", "are among", "is included", "are included",
        "is listed", "are listed", "listed among", "is a type of", "is a kind of",
        "is a model", "is a method", "is a mechanism", "is an access control",
        "is an example of", "such as", "includes", "include ", "including",
        # FR
        "est un de", "est une de", "fait partie", "font partie", "figure parmi",
        "est un type de", "est une sorte de", "inclut", "comprend", "parmi les",
    }
)

# Behavioral / property / comparative PREDICATE markers — a claim asserting one of these about
# its subject needs a real sentence (or a table row) to ground; a bare name-list cannot support
# it. Curated for PRECISION (fail-open on misses): generic copulas ("is"/"are") are deliberately
# EXCLUDED. Bilingual (EN+FR). Space-padded substring match.
_BEHAVIORAL_MARKERS: frozenset[str] = frozenset(
    {
        # behaviour / action verbs
        " assigns ", " assign ", " evaluates ", " evaluate ", " allows ", " allow ",
        " enables ", " enable ", " requires ", " require ", " provides ", " provide ",
        " supports ", " support ", " grants ", " grant ", " denies ", " deny ",
        " controls ", " control ", " enforces ", " enforce ", " processes ", " process ",
        " operates ", " operate ", " functions ", " relies ", " rely ", " uses ", " use ",
        " struggles ", " lacks ", " lack ", " introduces ", " validates ", " restricts ",
        " prevents ", " mitigates ", " determines ", " adapts ", " reacts ", " considers ",
        # property / behaviour phrases
        " based on ", " works by ", " used for ", " acts as ", " responsible for ",
        " capable of ", " is static", " is dynamic", " is rigid", " is scalable",
        " is flexible", " not scalable", " in real-time", " real time",
        # comparative
        " superior", " better ", " worse ", " faster", " slower", " outperforms ",
        " more granular", " less granular", " compared to", " than ",
        # FR action / property / comparative
        " assigne ", " assignent ", " évalue ", " évaluent ", " permet ", " permettent ",
        " nécessite ", " nécessitent ", " fournit ", " contrôle ", " contrôlent ",
        " repose sur ", " basé sur ", " basée sur ", " utilise ", " utilisent ",
        " restreint ", " empêche ", " détermine ", " en fonction de ", " supérieur",
        " meilleur", " plus rapide", " moins ",
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
