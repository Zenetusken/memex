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

from memex.core.text import looks_like_value

__all__ = [
    "GFM_TABLE_RE",
    "header_all_value_like",
    "header_has_prose_cell",
    "linearize_gfm_tables",
    "nearest_heading_text",
    "parse_gfm_table",
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
_FOOTNOTE_RE = re.compile(r"(?:\s*(?:\((?:\d{1,3})\)|\[(?:\d{1,3})\]|[*†‡]+))+\s*$")
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

    if _header_all_value_like(header) or _header_has_prose_cell(header):
        return None

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
