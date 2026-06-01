"""Table linearization at parse time (Table-RAG Phase 1).

`linearize_gfm_tables(markdown)` scans the parsed markdown for GFM tables and,
after each well-formed one, appends a derived markdown-KV block:

```
[table-rows]
<row-1 linearization>
<row-2 linearization>
[/table-rows]
```

separated from the table by a blank line (`\\n\\n`) so the block is a DISTINCT
paragraph unit — which is what makes the raw table and the `[table-rows]` block
land in SEPARATE chunks and keeps prose chunking byte-identical (the 3-channel
contract in `docs/specs/table-rag.md`).

Why KV rows: markdown-KV beats markdown-table for table retrieval (60.7% vs
51.9%) because each row co-locates its values with their column labels, so a
value cell that gets chunked / truncated still carries the column it belongs to
and embeds/ranks meaningfully. The raw table stays in the `.md` for fidelity;
the FTS body sees only the KV rows (`core/text.strip_superseded_gfm_tables`).

Engine-agnostic: runs on the post-header-recovery markdown both Docling and
PyMuPDF emit, parsing GFM *text* (no `TableCell.column_header` flags), so it
owns its own row-label-column heuristic. Idempotent + deterministic (a re-run
finds the same tables and emits byte-identical blocks; a table that already has
a `[table-rows]` block following it is skipped, so chunk_ids stay stable).

This module is pure-sync (string transforms only) and `core/`-only in deps.
"""

from __future__ import annotations

import re
from collections import Counter

from memex.core.text import coerce_number, looks_like_value

__all__ = [
    "GFM_TABLE_RE",
    "header_all_value_like",
    "header_has_lost_columns",
    "header_has_prose_cell",
    "is_layout_table",
    "linearize_gfm_tables",
    "nearest_heading_text",
    "nearest_table_caption",
    "parse_gfm_table",
    "split_merged_columns",
    "table_cell_lines",
]

# A GFM delimiter row: pipes / dashes / colons / whitespace with >=1 dash.
# Mirrors `index.chunker._GFM_DELIM_RE` so "is this a table delimiter" is
# decided identically across the parse + index layers.
_GFM_DELIM_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")

# A Markdown ATX heading line (1-6 `#` + space + text). Same shape as
# `index.chunker._HEADING_RE` / `core.text._MARKDOWN_HEADING_RE`; lifted here
# so the linearizer can find the nearest preceding heading without an
# `parse/ → index/` import edge. MULTILINE so `^`/`$` anchor each line.
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)

# A footnote marker to strip from cell text before linearization: a trailing
# superscript-style reference such as `Revenue(1)`, `Total[2]`, or a dagger /
# double-dagger / asterisk run. Conservative — only trailing markers, so an
# in-cell `$1,234 (note 3)` value is NOT mangled (the value is kept verbatim;
# only a pure trailing `(1)` / `[2]` / `*` / `†` is dropped).
#
# A parenthesized/bracketed number must follow a word/value char (the `(?<=...)`
# lookbehind) to count as a footnote — so a STANDALONE accounting-negative cell
# (`(45)`, `$(56)` = negative values) survives verbatim, while `Revenue(1)` still
# strips. The standalone `*`/`†`/`‡` rule keeps no lookbehind.
_FOOTNOTE_RE = re.compile(
    r"(?:\s*(?:(?<=[\w\)\]%])\((?:\d{1,3})\)|(?<=[\w\)\]%])\[(?:\d{1,3})\]|[*†‡]+))+\s*$"
)
_WHITESPACE_RE = re.compile(r"\s+")


def _is_delimiter_row(line: str) -> bool:
    return "-" in line and bool(_GFM_DELIM_RE.match(line))


def _split_pipe_row(line: str) -> list[str]:
    """Split a GFM pipe-row into its cell texts (uncleaned).

    Strips one optional leading and one optional trailing `|` then splits on
    the interior `|`. A GFM cell escapes a literal pipe as `\\|`; we do not
    un-escape (values are kept verbatim) but we don't split on the escaped
    form either.
    """
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    # Split on `|` not preceded by a backslash (GFM pipe escape).
    cells = re.split(r"(?<!\\)\|", s)
    return cells


def _clean_cell(text: str) -> str:
    """Collapse whitespace + strip a trailing footnote marker from a cell.

    Values are kept verbatim otherwise — `$22.5 billion` survives unchanged.
    """
    collapsed = _WHITESPACE_RE.sub(" ", text).strip()
    return _FOOTNOTE_RE.sub("", collapsed).strip()


def _is_row_label_column(header: list[str], data_rows: list[list[str]]) -> bool:
    """True iff col-0 is a row-label column.

    GFM-text heuristic (owned here — NOT the `TableCell` predicate in
    `parse/docling_tables`): col-0 is a row-label column iff ``header[0]`` is
    empty/blank AND at least one data-row col-0 cell is non-value-like. An
    empty corner cell with textual row labels below it (``| | A | B |``) is the
    classic row-label layout; a named first column (``| Metric | A | B |``) is
    the plain layout. Pinned both ways by the spec tests.
    """
    if not header or header[0].strip():
        return False
    return any(row and not looks_like_value(row[0]) for row in data_rows)


def _header_all_value_like(header: list[str]) -> bool:
    """Header-sanity gate: True iff EVERY header cell is value-like.

    Skips a table whose header row is all digits / ``$`` / ``%`` — a tell of a
    mis-structured header (e.g. the 10-K segment table, where the column labels
    detached into prose and the serializer left value cells in the header row).
    Linearizing such a header emits garbage KV pairs. A NON-value-like or an
    EMPTY header cell keeps the table: a year-header (`| Metric | 2024 | 2025
    |`) has the non-value-like `Metric`; an empty-corner table (`| | 2024 |
    2025 |`) has the empty col-0 — neither is "all value-like".
    """
    if not header:
        return False
    return all(looks_like_value(cell) for cell in header)


# A real GFM column-header cell is a short label. A cell that ends in sentence
# punctuation, runs many words, or is very long is a pulled-in sentence/heading,
# not a column label — the tell of a table Docling MIS-BOUNDED (it swallowed the
# surrounding heading + intro, e.g. the 10-K segment table whose header became
# `… | Our two reportable segments are '…' and 'Graphics': | Compute & Networking
# | Graphics Total`). Linearizing such a header emits WRONG KV pairs (mapping the
# wrong label to a value — a hallucination risk), so skip the table entirely.
_HEADER_PROSE_TERMINAL = (".", ":", "!", "?")
_HEADER_MAX_WORDS = 10
_HEADER_MAX_CHARS = 70


def _header_has_prose_cell(header: list[str]) -> bool:
    """Header-sanity gate (mis-bounding defense): True iff any header cell reads
    as prose (a pulled-in sentence/heading) rather than a short column label."""
    for cell in header:
        c = cell.strip()
        if not c:
            continue
        if (
            c[-1] in _HEADER_PROSE_TERMINAL
            or len(c.split()) > _HEADER_MAX_WORDS
            or len(c) > _HEADER_MAX_CHARS
        ):
            return True
    return False


def _header_has_lost_columns(header: list[str]) -> bool:
    """Header-sanity gate (W12, flattened merged-header defense): True iff the
    header has ≥2 EMPTY cells — the tell of a Docling-flattened merged-cell
    header whose real per-column labels collapsed into the first DATA row.

    Docling flattens a multi-row (merged-cell) header into ONE GFM header row:
    the group labels span columns, leaving the interior columns EMPTY, while the
    real per-column sub-labels collapse INTO the first data row. The live 10-K
    "components of pay" table is the prototype — its flattened header is
    ``['', 'Fixed Compensation', '', 'At-Risk', 'Compensation', '']`` (group
    labels over interior gaps), with the real sub-labels (``Base Salary``,
    ``Variable Cash``, …) consumed as data row 0. Linearizing it treats those
    real labels as DATA and keys the money by the garbage group header — a value
    mapped to a WRONG column, the hallucination class the header-sanity gates
    exist to avoid. So skip it: the raw GFM stays in the ``.md`` for fidelity
    and no mis-keyed KV row enters retrieval. This merged-header flatten ALWAYS
    leaves ≥2 gaps (a group label spans ≥2 columns), so ≥2 empties is the signal.

    What this does NOT skip (load-bearing):
      - a SINGLE empty col-0 corner (``['', '2024', '2025']``) → the classic
        row-label table; `_is_row_label_column` keys it right.
      - a SINGLE INTERIOR empty (``['Model', 'Params', '', 'LR', …]``, the gte
        hyperparameter table) → a REAL table that lost ONE column's label. The
        linearizer keys that one column POSITIONALLY (`_linearize_row._hdr`
        emits ``col{j+1}`` for an empty header cell — NEVER a wrong adjacent
        label), so its values surface fabrication-safely while the labelled
        columns key correctly. Skipping it would needlessly drop a real table's
        KV; the ≥2-empties threshold keeps it.
      - a fully-named header (``['Name', 'Fees ($)', 'Total ($)']``) → 0 empty
        cells (the 10-K director-compensation gold table).
    Pure-sync.
    """
    if not header:
        return False
    return sum(1 for cell in header if not cell.strip()) >= 2


# Public wrappers over the header-sanity gate so other modules (e.g.
# `index/table_store.py`) can apply the SAME skip predicate without importing
# the private `_header_*` symbols across a module boundary (forbidden by
# `src/memex/CLAUDE.md`). The behaviour is unchanged — these forward verbatim.
def header_all_value_like(header: list[str]) -> bool:
    """Public alias for the all-value-like header-sanity gate.

    See `_header_all_value_like` for the rationale: a header row that is ALL
    value-like (digits/``$``/``%``) is a mis-structured header → skip the table.
    """
    return _header_all_value_like(header)


def header_has_prose_cell(header: list[str]) -> bool:
    """Public alias for the prose-cell (mis-bounding) header-sanity gate.

    See `_header_has_prose_cell`: a header cell that reads as a pulled-in
    sentence/heading is the tell of a mis-bounded table → skip it.
    """
    return _header_has_prose_cell(header)


def header_has_lost_columns(header: list[str]) -> bool:
    """Public alias for the flattened-merged-header (W12) header-sanity gate.

    See `_header_has_lost_columns`: a header with a non-leading empty cell (or
    ≥2 empties) is a Docling-flattened merged-cell header whose interior columns
    lost their labels → linearizing it mis-keys a value to a wrong/positional
    label → skip the table. Used by `index/table_store.py` so the store and the
    linearizer skip the same tables.
    """
    return _header_has_lost_columns(header)


# Minimum STRUCTURAL column count for a block to be a real DATA table. A genuine
# tabular relation is at least 2-dimensional: >=2 column labels AND >=1 data row
# that actually splits into >=2 cells. A block that fails EITHER is a layout
# graphic / infographic / single-column list that Docling (or PyMuPDF) mis-emitted
# as a GFM table (audit-10 W11). The two failure shapes seen in the vault:
#   - a 1-column "table" — a metric LIST (`| Revenue |`), a references list
#     (`| References |`), a symptom list (`| Symptom Description |`), or an
#     INFOGRAPHIC whose bullets each render as a 1-cell row
#     (`| RISK OVERSIGHT AT NVIDIA |` → `| - Business model … |` rows);
#   - a multi-column HEADER over rows that are each a SINGLE cell — a bullet
#     list mis-grouped under column labels (`| AC | CC | NCGC |` over
#     `| - Financial statement integrity |` rows).
# Linearizing/storing either emits NONSENSE KV (`References=S62162 …`,
# `RISK OVERSIGHT AT NVIDIA=- Business model …`) and pollutes the raw `.md`.
_MIN_DATA_TABLE_COLS = 2


def is_layout_table(header: list[str], data_rows: list[list[str]]) -> bool:
    """True iff a parsed GFM block is a LAYOUT graphic / list, NOT a data table.

    Operates on the SAME (header, data_rows) `parse_gfm_table` returns, so the
    cell-split width is structural (the number of pipe-delimited fields per row),
    NOT trailing-empty-trimmed: a real 2-column table Docling UNDER-FILLED
    (`| IPsec Protocol | Choices |` over `| AH | |` rows — width 2, empty 2nd
    cell) keeps its empty cell and stays a data table, while a bullet list
    (`| AC | CC | NCGC |` over `| - … |` rows — width 1) is correctly flagged.

    The 2-D-relation test keys off the DATA rows' structural width, NOT the
    header's. This is load-bearing: a real 2-column table whose HEADER row
    under-split to a single cell (`| Metric |` over `| --- | --- |` + `| Revenue
    | 100 |`) still has 2-cell DATA rows and MUST stay a data table — flagging it
    on `len(header) < 2` would flatten it to bullets and destroy the row→value
    relation (a latent over-removal bug; 0 such blocks in the current vault, but
    a re-parse could produce one). Conversely a bullet list under a multi-column
    header (`| AC | CC | NCGC |` over `| - … |`) has 1-cell data rows and IS a
    layout block — so the data width, not the (wider) header, is authoritative.
    `header` is accepted to match the `parse_gfm_table` tuple the callers unpack.

    A block is a layout table iff it has no data row OR no data row splits into
    >= 2 cells — either way there is no 2-D relation to query, only a single
    column of text. Pure-sync; the caller (the parse finalize) decides what to do
    (re-render as bullets); the index-time linearizer + table store reuse it to
    SKIP such blocks so the two stay coherent with the cleaned `.md`.
    """
    if not data_rows:
        return True
    return max(len(row) for row in data_rows) < _MIN_DATA_TABLE_COLS


def table_cell_lines(block: str) -> list[str]:
    """Every NON-EMPTY cell of a GFM table block as RAW text, in document order.

    The header row's cells first, then each data row's cells, top-to-bottom /
    left-to-right. Each cell is only WHITESPACE-collapsed (multi-space → single,
    trimmed) — it is NOT footnote-stripped the way `parse_gfm_table` cleans cells,
    so inline markdown survives verbatim: ``**Board of Directors**`` stays
    bold-balanced (the footnote rule in `_clean_cell` would strip the trailing
    ``**`` as a dagger/asterisk marker — desirable for KV retrieval, WRONG for a
    content re-render). The delimiter row is skipped. Used by the parse-time W11
    demotion to re-render a layout table's content as faithful bullets. Pure-sync.
    """
    out: list[str] = []
    for line in block.split("\n"):
        if not line.strip() or _is_delimiter_row(line):
            continue
        for cell in _split_pipe_row(line):
            collapsed = _WHITESPACE_RE.sub(" ", cell).strip()
            if collapsed:
                out.append(collapsed)
    return out


def parse_gfm_table(block: str) -> tuple[list[str], list[list[str]]] | None:
    """Parse one GFM table block into its (header, data_rows) cell text, or
    None to skip — the reusable core factored out of `_linearize_table`.

    Returns the cleaned header cells and the cleaned data-row cells (footnote
    markers stripped, whitespace collapsed, values otherwise verbatim — see
    `_clean_cell`). Returns None for the SAME structural skip conditions the
    linearizer has always applied:
      - fewer than 3 non-blank lines (no header + delimiter + >=1 data row);
      - the second line is not a GFM delimiter row;
      - no non-empty data rows survive.

    It does NOT apply the header-sanity gate (`header_all_value_like` /
    `header_has_prose_cell`) — that stays the caller's responsibility, so both
    consumers (the linearizer and the table store) apply it identically while
    `parse_gfm_table` itself returns the raw parse. Pure-sync.
    """
    lines = [ln for ln in block.split("\n") if ln.strip()]
    if len(lines) < 3:
        return None
    header_line = lines[0]
    delim_line = lines[1]
    if not _is_delimiter_row(delim_line):
        return None
    data_lines = lines[2:]

    header = [_clean_cell(c) for c in _split_pipe_row(header_line)]
    data_rows = [[_clean_cell(c) for c in _split_pipe_row(ln)] for ln in data_lines]
    data_rows = [r for r in data_rows if any(c for c in r)]
    if not data_rows:
        return None
    return header, data_rows


# ----- Column UNDER-SPLIT recovery (2026-05-31) ------------------------------
# Docling sometimes MERGES two real columns into one: a header cell carrying two
# bold-label groups (`**Stock Awards ($) (1)** **Total ($)**`) over data cells
# carrying two number-runs (`278,809 342,559`). The merged column is not a clean
# numeric column, so Table-RAG can't run a MIN over it (ar-15) and the synthetic
# chunk renders garbled rows (ar-14). `split_merged_columns` rewrites such a
# column back into K columns when the StoredTable is built (the SQL-store path
# ONLY — the `[table-rows]` linearizer + raw GFM keep the original; see the
# function's NB).
# DOUBLY-GATED — the header must carry >=2 bold-label groups AND a strong
# majority of cells must split into the SAME K>=2 clean number-runs — so it fires
# on genuine financial merges only (validated to split ONLY the 10-K's ~8 merged
# columns, ZERO false positives across the 47-doc vault; a value-run-only signal
# would wrongly split dates / IP ranges / binaries / key-length lists, all of
# which have non-bold or single-bold headers). FABRICATION-SAFE: a ragged or
# ambiguous cell keeps its whole text in the first new column, never mis-slicing.

_BOLD_BOUNDARY_RE = re.compile(r"\*\*\s*\*\*")  # the interior close→open bold boundary
_ITALIC_UNITS_RE = re.compile(r"\s*_\(.*?\)_\s*$")  # trailing `_(In millions)_` unit note
_FOOTNOTE_RUN_RE = re.compile(r"^\(\d{1,3}\)$")  # a run that is purely a `(3)` marker
_RANGE_CONNECTOR_RE = re.compile(r"\d\s*(?:[-–—]|\bto\b)\s*\d", re.IGNORECASE)  # `150 - 600`, `1 to 5`
_DASH_ONLY_RE = re.compile(r"^[-–—‐]+$")
_CURRENCY_TOKENS = ("$", "€", "£")
_SPLIT_MAJORITY = 0.8  # >= this fraction of non-empty cells must be clean-merge at width K


def bold_groups(header_cell: str) -> list[str]:
    """Split a header cell into its bold-label groups (`**A** **B**` → [A, B]),
    stripping asterisks, whitespace, and a trailing italic-unit annotation. The
    labels are COSMETIC (column names); the values are what resolve a query."""
    out: list[str] = []
    for part in _BOLD_BOUNDARY_RE.split(header_cell):
        label = _ITALIC_UNITS_RE.sub("", part.strip()).strip().strip("*").strip("_").strip()
        if label:
            out.append(label)
    return out


def value_runs(cell: str) -> list[str]:
    """Whitespace-separated runs of a cell, currency-aware: a lone `$`/`€`/`£`
    token binds to the following token so `$ 1,813 $ 1,337` reads as TWO runs,
    not four (which is what collapses the `$ 193,479` word-wrap-header trap)."""
    toks = cell.split()
    runs: list[str] = []
    i = 0
    while i < len(toks):
        if toks[i] in _CURRENCY_TOKENS and i + 1 < len(toks):
            runs.append(toks[i] + " " + toks[i + 1])
            i += 2
        else:
            runs.append(toks[i])
            i += 1
    return runs


def _is_dash_only(cell: str) -> bool:
    return bool(_DASH_ONLY_RE.match(cell.strip()))


def _clean_merge_width(cell: str) -> int | None:
    """The number of value-runs a cell cleanly splits into (>=2), or None when it
    is NOT a clean merge: a digit range, <2 runs, any footnote-marker run, or any
    run that does not parse as a number (`coerce_number` — kills alpha tokens like
    a month name, an IP octet group, a binary string)."""
    if _RANGE_CONNECTOR_RE.search(cell):
        return None
    runs = value_runs(cell)
    if len(runs) < 2:
        return None
    if any(_FOOTNOTE_RUN_RE.match(r.strip()) for r in runs):
        return None
    if all(coerce_number(r) is not None for r in runs):
        return len(runs)
    return None


def _undersplit_width(header: list[str], rows: list[list[str]], ci: int) -> int:
    """The merge-width K (>=2) to split column `ci` into, or 0 to leave it. GATED
    on the header carrying >=2 bold-label groups (the discriminator that excludes
    dates/ranges/lists whose headers are non-bold), then requiring >= _SPLIT_MAJORITY
    of the non-empty, non-dash cells to be clean-merge cells of the SAME modal
    width K>=2."""
    if ci >= len(header) or len(bold_groups(header[ci])) < 2:
        return 0
    cells = [(row[ci] if ci < len(row) else "") for row in rows]
    considered = [c for c in cells if c.strip() and not _is_dash_only(c)]
    if len(considered) < 2:
        return 0
    widths = [_clean_merge_width(c) for c in considered]
    clean = [w for w in widths if w is not None]
    if not clean:
        return 0
    k = Counter(clean).most_common(1)[0][0]
    if k < 2:
        return 0
    if sum(1 for w in widths if w == k) / len(considered) < _SPLIT_MAJORITY:
        return 0
    return k


def _split_cell(cell: str, k: int) -> list[str]:
    """Split one cell into K sub-cells. A clean-merge cell of width K splits into
    its runs; an empty/dash cell → K empties; anything ELSE (ragged: wrong run
    count, a footnote run, markup) keeps its whole text in the FIRST sub-cell and
    the rest empty — never mis-slicing a number."""
    if not cell.strip() or _is_dash_only(cell):
        return [""] * k
    if _clean_merge_width(cell) == k:
        return value_runs(cell)
    return [cell] + [""] * (k - 1)


def split_merged_columns(
    header: list[str], rows: list[list[str]]
) -> tuple[list[str], list[list[str]]]:
    """Recover Docling-MERGED columns: a header column with >=2 bold-label groups
    whose cells carry K>=2 clean number-runs is split into K columns (header by
    bold-groups, cells by their runs), preserving the row/header length invariant
    (each grows by K-1 at the split point). Returns the rewritten (header, rows);
    a no-op when nothing fires. Idempotent (a split column's header has no bold
    boundary, so a re-run finds nothing). Pure-sync.

    NB: applied by `index/table_store.extract_tables` ONLY (the Table-RAG SQL
    store). The `[table-rows]` linearizer and the raw-GFM chunk text KEEP the
    original merged structure by design — the split exists to give the SQL query
    path clean columns to aggregate / order over (ar-14 SUM, ar-15 MIN); the BM25
    `[table-rows]` channel still indexes a merged cell's numbers, and the answer
    LLM grounds on the verbatim GFM either way, so neither needs the rewrite. So
    the SQL and retrieval views of a merged-column table differ intentionally."""
    if not header:
        return header, rows
    new_header = list(header)
    new_rows = [list(r) for r in rows]
    ci = 0
    while ci < len(new_header):
        k = _undersplit_width(new_header, new_rows, ci)
        if k >= 2:
            labels = bold_groups(new_header[ci])
            if len(labels) != k:
                labels = [f"col{ci + j}" for j in range(k)]  # fabrication-safe positional fallback
            new_header = new_header[:ci] + labels + new_header[ci + 1 :]
            for ri, row in enumerate(new_rows):
                cell = row[ci] if ci < len(row) else ""
                new_rows[ri] = row[:ci] + _split_cell(cell, k) + row[ci + 1 :]
            ci += k
        else:
            ci += 1
    return new_header, new_rows


def _trim_furniture_columns(
    header: list[str], data_rows: list[list[str]]
) -> tuple[list[str], list[list[str]]]:
    """Drop trailing HEADER-only "furniture" columns before linearization (W12).

    An OCR/layout artifact (a stray ruling line, a cropped marginal glyph) can
    add a trailing column to the HEADER row that NO data row ever fills — the
    live ``which-chart`` table's ``['City', '2000', …, '2010', 'P']`` header has
    a 13th ``'P'`` column over 12-cell data rows. Linearizing it as-is appends a
    dangling ``P=`` (empty value) to every row — benign noise, but noise. This
    trims any RIGHTMOST header column whose index is beyond EVERY data row's
    width, so the ``P=`` key disappears.

    Conservative by construction — it can only DROP a column the data never
    reaches, so no value is ever re-keyed or lost (a furniture column has no
    value to begin with). It does NOT touch a column a single ragged data row
    DOES fill (that stays — the ragged positional fallback still surfaces it),
    nor an interior empty (only contiguous trailing furniture). Pure-sync.
    """
    if not header or not data_rows:
        return header, data_rows
    max_data_width = max((len(row) for row in data_rows), default=0)
    # Only trim header columns to the RIGHT of every data cell. Stop at the
    # first non-empty header cell within data width to keep it contiguous.
    trimmed = list(header)
    while len(trimmed) > max_data_width and len(trimmed) > _MIN_DATA_TABLE_COLS:
        trimmed.pop()
    return trimmed, data_rows


def _linearize_row(
    header: list[str],
    cells: list[str],
    *,
    row_label_column: bool,
    prefix: str,
) -> str:
    """Render one data row as a markdown-KV line.

    Row-label branch: ``{prefix}{cell0}: {header[1]}={cell1}, …`` — the row
    label leads, the value columns follow as ``label=value`` pairs.
    Plain branch: ``{prefix}{header[0]}={cell0}, {header[1]}={cell1}, …``.

    A cell with no matching header column (ragged row wider than the header) is
    keyed by its 1-based position so its value still surfaces. An empty value
    cell is rendered as an empty value (`label=`), keeping the column visible.
    """
    n = max(len(header), len(cells))

    def _hdr(j: int) -> str:
        return header[j] if j < len(header) and header[j] else f"col{j + 1}"

    def _val(j: int) -> str:
        return cells[j] if j < len(cells) else ""

    if row_label_column:
        label = _val(0)
        pairs = [f"{_hdr(j)}={_val(j)}" for j in range(1, n)]
        return f"{prefix}{label}: " + ", ".join(pairs)
    pairs = [f"{_hdr(j)}={_val(j)}" for j in range(n)]
    return f"{prefix}" + ", ".join(pairs)


def _nearest_heading_prefix(markdown: str, table_start: int) -> str:
    """Return ``[<section heading>] `` for the nearest heading at/before
    *table_start*, or ``""`` when none precedes the table.

    Contextual-retrieval lite: prefixing each KV row with the section heading
    means a row chunked in isolation still carries the section it belongs to.
    """
    nearest: str | None = None
    for m in _HEADING_RE.finditer(markdown[:table_start]):
        nearest = m.group(2).strip()
    if not nearest:
        return ""
    return f"[{nearest}] "


def nearest_heading_text(body: str, pos: int) -> str:
    """Return the text of the nearest ATX heading at/before *pos*, or ``""``.

    Public sibling of `_nearest_heading_prefix` (which wraps the same lookup in
    ``[...] ``): used by `index/table_store.py` to populate `StoredTable.section`
    without an `index/ → parse/` private-symbol import.
    """
    nearest = ""
    for m in _HEADING_RE.finditer(body[:pos]):
        nearest = m.group(2).strip()
    return nearest


# A line that is ENTIRELY a bold caption (`**Director Compensation for Fiscal
# 2026**`) — a financial filing's table TITLE sits as a bold line, NOT an ATX
# heading, immediately above the table. More specific than the distant section
# heading, so it disambiguates near-duplicate tables (director vs executive
# "Compensation" tables) for table-selection.
_BOLD_CAPTION_RE = re.compile(r"^\s*\*\*(.+?)\*\*\s*$")
_CAPTION_MAX_GAP_LINES = 3  # blank lines tolerated between a caption and its table


def nearest_table_caption(body: str, pos: int) -> str:
    """The bold-caption line immediately preceding the table at *pos* (its title,
    e.g. `**Director Compensation for Fiscal 2026**`), or ``""`` when the nearest
    non-blank preceding line is prose or an ATX heading rather than a standalone
    bold caption. Used as a MORE-SPECIFIC section label than the distant heading."""
    lines = body[:pos].split("\n")
    blanks = 0
    for line in reversed(lines):
        if not line.strip():
            blanks += 1
            if blanks > _CAPTION_MAX_GAP_LINES:
                return ""
            continue
        caption = _BOLD_CAPTION_RE.match(line)
        return caption.group(1).strip() if caption else ""
    return ""


def _linearize_table(markdown: str, match: re.Match[str]) -> str | None:
    """Build the `[table-rows]` block for one GFM table match, or None to skip.

    Returns None when the header-sanity gate skips the table (all-value-like
    header) or there are no data rows. The returned block does NOT include the
    leading `\\n\\n` separator — the caller inserts it.
    """
    parsed = parse_gfm_table(match.group(0))
    if parsed is None:
        return None
    header, data_rows = parsed

    if (
        _header_all_value_like(header)
        or _header_has_prose_cell(header)
        or _header_has_lost_columns(header)
    ):
        return None
    # A layout graphic / infographic / single-column list mis-detected as a GFM
    # table (audit-10 W11): no 2-D relation to linearize → skip. The parse-time
    # finalize re-renders these as bullets, so a re-parsed `.md` has none left;
    # this guard keeps a not-yet-re-parsed (still-polluted) `.md` coherent — its
    # re-derived index has no nonsense `[table-rows]` KV for the layout block.
    if is_layout_table(header, data_rows):
        return None

    header, data_rows = _trim_furniture_columns(header, data_rows)
    row_label_column = _is_row_label_column(header, data_rows)
    prefix = _nearest_heading_prefix(markdown, match.start())

    rows = [
        _linearize_row(header, cells, row_label_column=row_label_column, prefix=prefix)
        for cells in data_rows
    ]
    return "[table-rows]\n" + "\n".join(rows) + "\n[/table-rows]"


# A GFM table block: a header pipe-row, a delimiter row, then >=1 data pipe-row.
# Line-anchored + multiline so a table embedded in prose is found at its own
# line boundaries. Tolerates a missing trailing newline on the final row.
_GFM_TABLE_RE = re.compile(
    r"^[ \t]*\|.*\|[ \t]*\n"  # header pipe-row
    r"[ \t]*\|?[ \t:|-]*-[ \t:|-]*\|?[ \t]*\n"  # delimiter row (>=1 dash)
    r"(?:[ \t]*\|.*\|[ \t]*(?:\n|$))+",  # >=1 data pipe-row
    flags=re.MULTILINE,
)

# Public alias so other modules (`index/table_store.py`) can scan a body for
# GFM tables with `match.start()/end()` spans without importing the private
# name across a module boundary (forbidden by `src/memex/CLAUDE.md`). Same
# compiled object — no behaviour change.
GFM_TABLE_RE = _GFM_TABLE_RE

# A `[table-rows]` opener — used to detect an already-linearized table so the
# pass is idempotent.
_TABLE_ROWS_OPEN_RE = re.compile(r"\[table-rows\]")


def linearize_gfm_tables(markdown: str) -> str:
    """Append a `[table-rows]` KV block after every well-formed GFM table.

    For each GFM table (header pipe-row + `_GFM_DELIM_RE` delimiter + >=1 data
    pipe-row), appends — separated by a blank line so the block is a distinct
    paragraph unit — a ``[table-rows]…[/table-rows]`` block of markdown-KV
    rows. Header-sanity-gated (skips an all-value-like header). Idempotent (a
    table already followed by a `[table-rows]` block is left alone) and
    deterministic (same input ⇒ same output bytes ⇒ stable chunk_ids).

    Pure-sync; safe to call on any markdown string (no-op when no GFM tables).
    """
    out: list[str] = []
    cursor = 0
    for m in _GFM_TABLE_RE.finditer(markdown):
        out.append(markdown[cursor : m.end()])
        cursor = m.end()
        # Idempotence: if a `[table-rows]` block already follows this table
        # (across the blank-line separator), don't append another.
        rest = markdown[m.end() :]
        if _TABLE_ROWS_OPEN_RE.match(rest.lstrip("\n \t") if rest else ""):
            continue
        block = _linearize_table(markdown, m)
        if block is None:
            continue
        # Insert a BLANK line between the raw table and the block so the block
        # is a distinct paragraph unit. The table match ends either right after
        # the final row's `\n` (the common case) or at EOF with no newline. In
        # both cases we want the emitted text to read `…row\n\n[table-rows]…`:
        # add the missing newline(s) so exactly one blank line separates them.
        emitted = "".join(out)
        if emitted.endswith("\n\n"):
            sep = ""
        elif emitted.endswith("\n"):
            sep = "\n"
        else:
            sep = "\n\n"
        out.append(sep + block)
    out.append(markdown[cursor:])
    return "".join(out)
