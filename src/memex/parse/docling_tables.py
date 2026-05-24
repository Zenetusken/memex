# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# ^ docling_core ships a *partial* `py.typed`: its public `DoclingDocument`,
#   `TableItem`, `TableData`, `TableCell` and the markdown serializer surface
#   resolve to Unknown under --strict (computed `grid`, dynamic `model_copy`,
#   `iterate_items` yields). This file is the docling-untyped boundary — the
#   same scoped-suppression pattern `parse/pymupdf_worker.py` uses for its
#   pymupdf/pymupdf4llm surface. The suppression is limited to the three
#   Unknown-* rules; every other --strict check (including reportMissingImports
#   and the public function signatures below) stays on. Do NOT widen it.
"""Header-aware Docling table serialization + gated detached-header re-attach.

Docling's `MarkdownTableSerializer` blindly treats `grid[0]` as the GFM
header and **ignores `TableCell.column_header` flags**. Two failure modes
(see `docs/specs/table-header-reattach.md`):

* **Class A — well-captured, mis-rendered.** Docling correctly flags
  *multiple* header rows; the serializer emits only `grid[0]` and pushes the
  remaining header rows into the body. `HeaderAwareTableSerializer` (Part 1)
  fixes this by counting the leading header rows (exactly like
  `TableItem._export_to_dataframe_with_options`) and, when ``num_headers >= 2``,
  collapsing them into one merged GFM header row.

* **Class B — mis-structured/transposed.** Docling tags value cells as headers
  and the real column labels are detached into preceding heading / text items.
  `reattach_detached_table_headers` (Part 3) re-attaches them, but **only when
  the detected label count exactly matches the value-column count** — the
  safety gate that guarantees no table is ever made worse.

`export_markdown_header_aware` (Part 2) reproduces
`DoclingDocument.export_to_markdown`'s `MarkdownParams` construction exactly,
swapping in only the header-aware table serializer, so non-table / single-header
output stays byte-identical to `doc.export_to_markdown()`.

This module is sync (pure CPU transforms on the in-memory document) and is
imported lazily inside the worker so a missing docling install surfaces as the
worker's `docling_unavailable` exit path rather than an import error here.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from docling_core.transforms.serializer.base import SerializationResult
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
    MarkdownTableSerializer,
)
from docling_core.types.doc.document import (
    DEFAULT_CONTENT_LAYERS,
    DOCUMENT_TOKENS_EXPORT_LABELS,
    DoclingDocument,
    SectionHeaderItem,
    TableCell,
    TableData,
    TableItem,
    TextItem,
)

if TYPE_CHECKING:
    from docling_core.transforms.serializer.base import BaseDocSerializer


__all__ = [
    "HeaderAwareTableSerializer",
    "export_markdown_header_aware",
    "reattach_detached_table_headers",
]


# ----- Header counting (mirrors TableItem._export_to_dataframe_with_options) -----


def _count_header_rows(data: TableData) -> int:
    """Count the leading column-header rows of *data*.

    Walks ``data.grid`` top-down: a row is a header row iff **any** cell in it
    has ``column_header=True``; counting stops at the first non-header row. This
    is byte-for-byte the same logic `TableItem._export_to_dataframe_with_options`
    (document.py ~2280) uses to compute ``num_headers``.
    """
    if data.num_rows == 0 or data.num_cols == 0:
        return 0
    num_headers = 0
    for row in data.grid:
        if not row:
            break
        if any(cell.column_header for cell in row):
            num_headers += 1
        else:
            break
    return num_headers


def _merge_header_rows(grid: list[list[TableCell]], num_headers: int, num_cols: int) -> list[str]:
    """Collapse the first *num_headers* rows of *grid* into one label per column.

    Mirrors the dataframe's per-column `.`-join of the header-row texts, but uses
    a single space for readability and collapses immediate repeats (so a header
    cell that spans/duplicates down a column doesn't double up).
    """
    merged: list[str] = []
    for j in range(num_cols):
        parts: list[str] = []
        for i in range(num_headers):
            text = grid[i][j].text if j < len(grid[i]) else ""
            if text and (not parts or parts[-1] != text):
                parts.append(text)
        merged.append(" ".join(parts))
    return merged


class HeaderAwareTableSerializer(MarkdownTableSerializer):
    """`MarkdownTableSerializer` that respects multi-row column headers.

    For ``num_headers <= 1`` (the vast majority of tables, including the
    ``num_headers == 0`` Class-B case handled by the re-attach pass) it defers
    verbatim to ``super().serialize(...)`` — so output is byte-identical to the
    stock serializer. Only for ``num_headers >= 2`` does it intervene: it builds
    a single merged GFM header row and renders the remaining rows as the body,
    delegating to the parent on a synthetic single-header table so all of the
    parent's cell-escaping (newline→space, ``|``→``&#124;``), tabulate
    formatting, column-alignment, caption + annotation handling is reused
    unchanged.
    """

    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize *item*, merging multi-row headers into one GFM header row."""
        # Nested-in-table fast path: behaviour is unrelated to headers; defer.
        if kwargs.get("_nested_in_table"):
            return super().serialize(item=item, doc_serializer=doc_serializer, doc=doc, **kwargs)

        data = item.data
        num_headers = _count_header_rows(data)
        if num_headers <= 1:
            # Single header (current behaviour already correct) or zero headers
            # (Class B — handled by reattach, which makes num_headers >= 1).
            return super().serialize(item=item, doc_serializer=doc_serializer, doc=doc, **kwargs)

        # num_headers >= 2: collapse the header rows into one, keep the body.
        grid = data.grid
        num_cols = data.num_cols
        merged_header = _merge_header_rows(grid, num_headers, num_cols)

        new_cells: list[TableCell] = [
            TableCell(
                text=merged_header[j],
                start_row_offset_idx=0,
                end_row_offset_idx=1,
                start_col_offset_idx=j,
                end_col_offset_idx=j + 1,
                column_header=True,
            )
            for j in range(num_cols)
        ]
        for new_r, old_r in enumerate(range(num_headers, data.num_rows), start=1):
            old_row = grid[old_r]
            for j in range(num_cols):
                old = old_row[j] if j < len(old_row) else None
                new_cells.append(
                    TableCell(
                        text=old.text if old is not None else "",
                        start_row_offset_idx=new_r,
                        end_row_offset_idx=new_r + 1,
                        start_col_offset_idx=j,
                        end_col_offset_idx=j + 1,
                        column_header=False,
                    )
                )
        new_num_rows = 1 + (data.num_rows - num_headers)
        new_data = TableData(table_cells=new_cells, num_rows=new_num_rows, num_cols=num_cols)
        # Clone the item with the rewritten data so captions/annotations/refs
        # are preserved; the parent then renders it as a clean single-header
        # table, reusing all of its escaping + tabulate logic verbatim.
        clone: TableItem = item.model_copy(update={"data": new_data})
        return super().serialize(item=clone, doc_serializer=doc_serializer, doc=doc, **kwargs)


# ----- Part 2: header-aware export (byte-identical except multi-header/re-attached) -----


def export_markdown_header_aware(doc: DoclingDocument) -> str:
    """`doc.export_to_markdown()` with the header-aware table serializer.

    Reconstructs the exact `MarkdownParams` that
    `DoclingDocument.export_to_markdown()` builds at its defaults (document.py
    ~6072) — `labels=DOCUMENT_TOKENS_EXPORT_LABELS`, `layers=DEFAULT_CONTENT_LAYERS`,
    every other field left at its `MarkdownParams` default, matching the
    export_to_markdown default arguments — and swaps in only
    ``table_serializer=HeaderAwareTableSerializer()``.

    REGRESSION INVARIANT: for a document with no multi-header table and no
    re-attached table, this returns the same bytes as ``doc.export_to_markdown()``
    (the header-aware serializer defers to the parent for ``num_headers <= 1``).
    """
    serializer = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams(
            labels=DOCUMENT_TOKENS_EXPORT_LABELS,
            layers=DEFAULT_CONTENT_LAYERS,
        ),
        table_serializer=HeaderAwareTableSerializer(),
    )
    return serializer.serialize().text


# ----- Part 3: gated re-attach of detached headers (Class B, best-effort) -----

_VALUE_CHARS = ("$", "%")


def _looks_like_value(text: str) -> bool:
    """True if *text* reads as a numeric value (contains a digit or ``$``/``%``)."""
    if any(ch.isdigit() for ch in text):
        return True
    return any(ch in text for ch in _VALUE_CHARS)


def _has_row_label_column(grid: list[list[TableCell]], num_rows: int) -> bool:
    """True if column 0 is a row-label column (non-numeric text, not header-flagged).

    A row-label column means the table's first column holds labels (e.g.
    ``Revenue`` / ``Operating Income``) rather than data, so the detached
    column labels should fill columns ``1..num_cols-1`` and column 0 gets an
    empty header cell.
    """
    if num_rows == 0 or not grid or not grid[0]:
        return False
    saw_text = False
    for r in range(num_rows):
        if not grid[r]:
            continue
        cell = grid[r][0]
        if cell.column_header:
            return False
        text = (cell.text or "").strip()
        if not text:
            continue
        saw_text = True
        if _looks_like_value(text):
            return False
    return saw_text


def _is_mis_structured(data: TableData, num_headers: int) -> bool:
    """True if the table looks transposed/mis-flagged (a re-attach candidate).

    Either there are no header rows at all, OR the single flagged header row is
    "data-like": every flagged-header cell beyond column 0 matches a value
    pattern (digit / ``$`` / ``%``) AND the row's column-0 cell is a genuine
    row label — non-empty text that is itself NOT header-flagged.

    The column-0 distinguisher is what separates a mis-flagged data row from a
    legitimate numeric/year-header table. A real header row's leading cell is an
    empty corner cell or is itself ``column_header=True``; a row label (e.g.
    ``Revenue``) sitting inside a "header" row is contradictory, which is the
    tell that Docling mis-flagged a data row as the header. Tables whose flagged
    header row reads as text labels, OR whose corner cell is empty / header-
    flagged, are left alone (they already have a clean header).
    """
    if num_headers == 0:
        return True
    if num_headers != 1:
        return False
    grid = data.grid
    if not grid or not grid[0]:
        return False
    header_row = grid[0]
    # Distinguisher: the flagged header row must have a genuine row label in
    # column 0 — non-empty text that is NOT itself header-flagged. An empty
    # corner cell or a header-flagged col-0 means this is a real header row
    # (e.g. a numeric/year-header table), which must never be touched.
    col0 = header_row[0]
    if col0.column_header or not (col0.text or "").strip():
        return False
    flagged = [cell for cell in header_row if cell.column_header]
    # Only judge the value columns (skip a leading row-label column at col 0).
    value_flagged = [cell for cell in flagged if cell.start_col_offset_idx >= 1]
    if not value_flagged:
        value_flagged = flagged
    if not value_flagged:
        return False
    return all(_looks_like_value(cell.text or "") for cell in value_flagged)


def _wide_split(text: str) -> list[str]:
    """Split *text* on runs of 2+ spaces (the gap between detached labels)."""
    return [t for t in re.split(r" {2,}", text.strip()) if t]


def _tokenize_run(run: list[str], target: int) -> list[str]:
    """Tokenize a reading-order run of text items into *target* label tokens.

    Strategy, prefer-fewer-splits: first treat each item's wide-gap (2+ space)
    split as its labels and concatenate in reading order — this keeps a
    multi-word phrase like "Compute & Networking" as ONE label. Only if the
    wide-gap total falls short of *target* does it fall back to a single-space
    split of the whole run (used for "Graphics Total" → two labels). Returns
    whatever the chosen strategy yields; the caller's exact-count gate decides
    whether to accept it.
    """
    wide: list[str] = []
    for chunk in run:
        wide.extend(_wide_split(chunk))
    if len(wide) >= target:
        return wide
    single: list[str] = []
    for chunk in run:
        single.extend(t for t in chunk.split() if t)
    return single


def _collect_detached_labels(
    doc: DoclingDocument,
    table: TableItem,
    target: int,
) -> list[str]:
    """Collect the run of heading/text items immediately preceding *table*.

    Walks `doc.iterate_items()` (reading order); records the contiguous trailing
    run of `SectionHeaderItem`/`TextItem` ending right before *table*, stopping
    the run at any non-text item or a blank. Tokenizes the run into label tokens
    (wide-gap split preferred, single-space fallback to reach *target*). Returns
    ``[]`` if no text items precede the table.
    """
    run: list[str] = []
    for item, _level in doc.iterate_items():
        if isinstance(item, TableItem):
            if item is table:
                break
            # A different table breaks the contiguous text run.
            run = []
            continue
        if isinstance(item, (SectionHeaderItem, TextItem)):
            text = (item.text or "").strip()
            if text:
                run.append(text)
            else:
                run = []
        else:
            run = []
    if not run:
        return []
    return _tokenize_run(run, target)


def _reattach_one(doc: DoclingDocument, table: TableItem) -> bool:
    """Try to re-attach detached headers onto *table*; return True if mutated.

    Gated: mutates only when the detected label count EXACTLY equals the
    value-column count. On any mismatch the table is left untouched.
    """
    data = table.data
    if data.num_rows == 0 or data.num_cols == 0:
        return False
    num_headers = _count_header_rows(data)
    if not _is_mis_structured(data, num_headers):
        return False

    grid = data.grid
    has_label_col = _has_row_label_column(grid, data.num_rows)
    value_cols = data.num_cols - (1 if has_label_col else 0)
    if value_cols <= 0:
        return False

    labels = _collect_detached_labels(doc, table, value_cols)
    # GATE: exact count match or skip. Never make a table worse.
    if len(labels) != value_cols:
        return False

    # Build the new header row. A leading row-label column (if present) gets an
    # empty header cell; each value column gets one label cell.
    new_header: list[str] = ([""] if has_label_col else []) + labels
    new_cells: list[TableCell] = [
        TableCell(
            text=new_header[j],
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=j,
            end_col_offset_idx=j + 1,
            column_header=True,
        )
        for j in range(data.num_cols)
    ]
    # Shift every existing cell down one row and clear the bogus header flags on
    # the previously mis-flagged value row(s).
    for cell in data.table_cells:
        cell.start_row_offset_idx += 1
        cell.end_row_offset_idx += 1
        cell.column_header = False
    data.table_cells = new_cells + list(data.table_cells)
    data.num_rows += 1
    return True


def reattach_detached_table_headers(doc: DoclingDocument) -> int:
    """Re-attach detached column labels onto mis-structured tables, in place.

    For each `TableItem` in *doc* that looks transposed/mis-flagged (Class B),
    collects the run of heading/text items immediately preceding it in reading
    order and, **only when** the detected label count exactly equals the
    value-column count, prepends a correct ``column_header=True`` row and clears
    the bogus header flags. On any count mismatch the table is left unchanged
    (the safety gate). Returns the number of tables mutated.

    Runs BEFORE `export_markdown_header_aware`; after the re-attach a Class-B
    table has ``num_headers == 1`` and renders with the correct GFM header.
    """
    tables: list[TableItem] = list(getattr(doc, "tables", None) or [])
    count = 0
    for table in tables:
        if _reattach_one(doc, table):
            count += 1
    return count
