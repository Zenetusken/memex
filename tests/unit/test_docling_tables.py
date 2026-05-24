"""Unit tests for `parse/docling_tables.py` — header-aware table serialization
+ gated detached-header re-attach.

Constructs real `docling_core` `TableData`/`TableCell`/`TableItem`/
`DoclingDocument` objects in-memory and exercises actual mutation/serialization
behaviour (not tautologies). The whole module is skipped when docling_core is
unavailable (e.g. a CPU-only CI box without the parse extras).

Covers the 7 spec test cases from `docs/specs/table-header-reattach.md`:
  1. Class A 2-header merge differs from (buggy) base output.
  2. Single-header table == base output (no change).
  3. Re-attach happy path on the segment-table shape.
  4. Re-attach gate skip on count mismatch.
  5. No-detached-labels skip.
  6. Byte-identity on a single-header doc.
  7. Worker-order wiring (reattach then header-aware export).
"""

from __future__ import annotations

import copy
from typing import ClassVar

import pytest

pytest.importorskip("docling_core")

from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownTableSerializer,
)
from docling_core.types.doc.document import (
    DoclingDocument,
    TableCell,
    TableData,
    TableItem,
)
from docling_core.types.doc.labels import DocItemLabel

from memex.parse.docling_tables import (
    HeaderAwareTableSerializer,
    export_markdown_header_aware,
    reattach_detached_table_headers,
)


def _cell(text: str, r: int, c: int, *, header: bool = False) -> TableCell:
    return TableCell(
        text=text,
        start_row_offset_idx=r,
        end_row_offset_idx=r + 1,
        start_col_offset_idx=c,
        end_col_offset_idx=c + 1,
        column_header=header,
    )


def _serialize_table(
    doc: DoclingDocument, table: TableItem, serializer: MarkdownTableSerializer
) -> str:
    doc_ser = MarkdownDocSerializer(doc=doc)
    return serializer.serialize(item=table, doc_serializer=doc_ser, doc=doc).text


# --------------------------------------------------------------------------- #
# 1. Class A — multi-header merge                                             #
# --------------------------------------------------------------------------- #


def test_class_a_two_header_rows_merge_into_one_gfm_header() -> None:
    """A table with two `column_header` rows: the base serializer drops row 1
    into the body; the header-aware serializer merges both into ONE GFM header.
    """
    doc = DoclingDocument(name="classA")
    cells = [
        # row 0: a spanning super-header repeated across both columns
        _cell("Fiscal 2026 Result", 0, 0, header=True),
        _cell("Fiscal 2026 Result", 0, 1, header=True),
        # row 1: the real column labels (also flagged header)
        _cell("Revenue", 1, 0, header=True),
        _cell("Gross Margin", 1, 1, header=True),
        # row 2: data
        _cell("130.5", 2, 0),
        _cell("75%", 2, 1),
    ]
    table = doc.add_table(data=TableData(table_cells=cells, num_rows=3, num_cols=2))

    base = _serialize_table(doc, table, MarkdownTableSerializer())
    aware = _serialize_table(doc, table, HeaderAwareTableSerializer())

    # The buggy base serializer treats only grid[0] as header, so "Revenue" /
    # "Gross Margin" land in the BODY (a data row).
    base_lines = base.splitlines()
    assert base_lines[0].count("Fiscal 2026 Result") == 2
    assert "Revenue" in base and base.index("Revenue") > base.index("---")  # in body

    # The header-aware serializer must differ and merge both header rows.
    assert aware != base
    aware_header = aware.splitlines()[0]
    assert "Fiscal 2026 Result Revenue" in aware_header
    assert "Fiscal 2026 Result Gross Margin" in aware_header
    # Revenue/Gross Margin must NOT appear as a body row anymore: only one data
    # row remains (the "130.5 | 75%" line).
    body_lines = [ln for ln in aware.splitlines()[2:] if ln.strip()]
    assert len(body_lines) == 1
    assert "130.5" in body_lines[0] and "75%" in body_lines[0]


def test_class_a_collapses_repeated_super_header_per_column() -> None:
    """A duplicated super-header cell down a column collapses (no double-up)."""
    doc = DoclingDocument(name="classA-collapse")
    cells = [
        _cell("Group", 0, 0, header=True),
        _cell("Group", 0, 1, header=True),
        _cell("A", 1, 0, header=True),
        _cell("B", 1, 1, header=True),
        _cell("1", 2, 0),
        _cell("2", 2, 1),
    ]
    table = doc.add_table(data=TableData(table_cells=cells, num_rows=3, num_cols=2))
    header = _serialize_table(doc, table, HeaderAwareTableSerializer()).splitlines()[0]
    assert "Group A" in header
    assert "Group B" in header
    assert "Group Group" not in header


# --------------------------------------------------------------------------- #
# 2. Single-header table — header-aware output == base output                 #
# --------------------------------------------------------------------------- #


def test_single_header_table_matches_base_serializer() -> None:
    doc = DoclingDocument(name="single")
    cells = [
        _cell("Name", 0, 0, header=True),
        _cell("Value", 0, 1, header=True),
        _cell("alpha", 1, 0),
        _cell("1", 1, 1),
        _cell("beta", 2, 0),
        _cell("2", 2, 1),
    ]
    table = doc.add_table(data=TableData(table_cells=cells, num_rows=3, num_cols=2))
    base = _serialize_table(doc, table, MarkdownTableSerializer())
    aware = _serialize_table(doc, table, HeaderAwareTableSerializer())
    assert aware == base


def test_zero_header_table_matches_base_serializer() -> None:
    """num_headers == 0 also defers to the parent (Class B re-attach adds one)."""
    doc = DoclingDocument(name="zero")
    cells = [
        _cell("a", 0, 0),
        _cell("b", 0, 1),
        _cell("c", 1, 0),
        _cell("d", 1, 1),
    ]
    table = doc.add_table(data=TableData(table_cells=cells, num_rows=2, num_cols=2))
    base = _serialize_table(doc, table, MarkdownTableSerializer())
    aware = _serialize_table(doc, table, HeaderAwareTableSerializer())
    assert aware == base


def test_header_aware_reuses_parent_cell_escaping() -> None:
    """Newline + pipe escaping in merged-header body cells must come from the
    parent (not reinvented): pipes → `&#124;`, newlines → spaces."""
    doc = DoclingDocument(name="escape")
    cells = [
        _cell("Top", 0, 0, header=True),
        _cell("Top", 0, 1, header=True),
        _cell("X", 1, 0, header=True),
        _cell("Y", 1, 1, header=True),
        _cell("a|b", 2, 0),
        _cell("line\nbreak", 2, 1),
    ]
    table = doc.add_table(data=TableData(table_cells=cells, num_rows=3, num_cols=2))
    aware = _serialize_table(doc, table, HeaderAwareTableSerializer())
    assert "a&#124;b" in aware
    assert "line break" in aware
    assert "line\nbreak" not in aware


# --------------------------------------------------------------------------- #
# 3. Re-attach happy path — segment-table shape                               #
# --------------------------------------------------------------------------- #


def _build_segment_doc() -> tuple[DoclingDocument, TableItem]:
    """The p23 segment table: col0 row-labels, 3 value columns flagged as the
    (bogus) header row, preceded by a `### Compute & Networking` heading + a
    `Graphics  Total` text line carrying the detached labels."""
    doc = DoclingDocument(name="segment")
    doc.add_heading("Compute & Networking", level=2)
    doc.add_text(label=DocItemLabel.TEXT, text="Graphics  Total")
    cells = [
        _cell("Revenue", 0, 0),
        _cell("$193.5B", 0, 1, header=True),
        _cell("$22.5B", 0, 2, header=True),
        _cell("$215.9B", 0, 3, header=True),
        _cell("Operating Income", 1, 0),
        _cell("$130.1B", 1, 1),
        _cell("$5.0B", 1, 2),
        _cell("$135.1B", 1, 3),
    ]
    table = doc.add_table(data=TableData(table_cells=cells, num_rows=2, num_cols=4))
    return doc, table


def test_reattach_happy_path_segment_table() -> None:
    doc, table = _build_segment_doc()
    n = reattach_detached_table_headers(doc)
    assert n == 1

    data = table.data
    # A new header row was prepended: num_rows 2 -> 3.
    assert data.num_rows == 3

    grid = data.grid
    # New header row: empty row-label cell + the three detached labels.
    assert [c.text for c in grid[0]] == ["", "Compute & Networking", "Graphics", "Total"]
    assert all(c.column_header for c in grid[0])

    # The previously mis-flagged value row's header flags are cleared.
    assert not any(c.column_header for c in grid[1])
    assert [c.text for c in grid[1]] == ["Revenue", "$193.5B", "$22.5B", "$215.9B"]
    assert [c.text for c in grid[2]] == ["Operating Income", "$130.1B", "$5.0B", "$135.1B"]

    # Serialized GFM has the labels as the header.
    md = export_markdown_header_aware(doc)
    header_line = [
        ln for ln in md.splitlines() if ln.startswith("|") and "Compute & Networking" in ln
    ]
    assert header_line, md
    assert "Graphics" in header_line[0] and "Total" in header_line[0]


def test_reattach_offsets_roundtrip_into_valid_grid() -> None:
    """After re-attach, every cell offset is consistent: the grid resolves to
    num_rows x num_cols with no stray empty cells in shifted rows."""
    doc, table = _build_segment_doc()
    reattach_detached_table_headers(doc)
    data = table.data
    grid = data.grid
    assert len(grid) == data.num_rows
    for row in grid:
        assert len(row) == data.num_cols
    # No cell offset exceeds the new bounds.
    for cell in data.table_cells:
        assert 0 <= cell.start_row_offset_idx < data.num_rows
        assert cell.end_row_offset_idx <= data.num_rows


# --------------------------------------------------------------------------- #
# 4. Re-attach gate skip — count mismatch                                     #
# --------------------------------------------------------------------------- #


def test_reattach_gate_skips_on_count_mismatch() -> None:
    """Same shape, but the detached labels tokenize to a count != value-column
    count → the gate refuses and leaves the table byte-identical."""
    doc = DoclingDocument(name="mismatch")
    # Single-word heading + a single-spaced 3-word text line. Wide-gap split
    # yields 1 + 1 = 2 < 3, so it falls back to single-space → 4 tokens
    # ("Compute", "Networking", "Graphics", "Total", "Extra") != 3 value cols.
    doc.add_heading("Compute Networking", level=2)
    doc.add_text(label=DocItemLabel.TEXT, text="Graphics Total Extra")
    cells = [
        _cell("Revenue", 0, 0),
        _cell("$193.5B", 0, 1, header=True),
        _cell("$22.5B", 0, 2, header=True),
        _cell("$215.9B", 0, 3, header=True),
        _cell("Operating Income", 1, 0),
        _cell("$130.1B", 1, 1),
        _cell("$5.0B", 1, 2),
        _cell("$135.1B", 1, 3),
    ]
    table = doc.add_table(data=TableData(table_cells=cells, num_rows=2, num_cols=4))
    before = copy.deepcopy(table.data.table_cells)

    n = reattach_detached_table_headers(doc)
    assert n == 0
    assert table.data.num_rows == 2
    after = table.data.table_cells
    assert [(c.text, c.column_header, c.start_row_offset_idx) for c in after] == [
        (c.text, c.column_header, c.start_row_offset_idx) for c in before
    ]


# --------------------------------------------------------------------------- #
# 5. No detached labels — skip                                                #
# --------------------------------------------------------------------------- #


def test_reattach_skips_when_no_preceding_labels() -> None:
    doc = DoclingDocument(name="nolabels")
    cells = [
        _cell("Revenue", 0, 0),
        _cell("$1", 0, 1, header=True),
        _cell("$2", 0, 2, header=True),
        _cell("$3", 0, 3, header=True),
        _cell("OpInc", 1, 0),
        _cell("$4", 1, 1),
        _cell("$5", 1, 2),
        _cell("$6", 1, 3),
    ]
    table = doc.add_table(data=TableData(table_cells=cells, num_rows=2, num_cols=4))
    before = copy.deepcopy(table.data.table_cells)
    n = reattach_detached_table_headers(doc)
    assert n == 0
    assert table.data.num_rows == 2
    assert [(c.text, c.column_header) for c in table.data.table_cells] == [
        (c.text, c.column_header) for c in before
    ]


def test_reattach_skips_clean_text_header_table() -> None:
    """A table whose flagged header row reads as text labels (not value-like) is
    NOT considered mis-structured even with a preceding text line — leave it."""
    doc = DoclingDocument(name="cleanhdr")
    doc.add_text(label=DocItemLabel.TEXT, text="Some preceding prose")
    cells = [
        _cell("Region", 0, 0, header=True),
        _cell("Sales", 0, 1, header=True),
        _cell("EU", 1, 0),
        _cell("100", 1, 1),
    ]
    table = doc.add_table(data=TableData(table_cells=cells, num_rows=2, num_cols=2))
    n = reattach_detached_table_headers(doc)
    assert n == 0
    assert table.data.num_rows == 2


# --------------------------------------------------------------------------- #
# 5b. BLOCKER repro — legitimate numeric/year-header table must be untouched   #
# --------------------------------------------------------------------------- #


def _grid_snapshot(table: TableItem) -> list[tuple[str, bool, int, int]]:
    """Byte-level snapshot of a table's cells for deep before/after compare."""
    return [
        (
            c.text,
            c.column_header,
            c.start_row_offset_idx,
            c.start_col_offset_idx,
        )
        for c in table.data.table_cells
    ]


def test_reattach_skips_legit_year_header_with_empty_corner() -> None:
    """BLOCKER repro (NEGATIVE test).

    A legitimate numeric/year-header table:

        | (empty corner, not header-flagged) | 2024 (hdr) | 2025 (hdr) |
        | Revenue                            | 10         | 20         |
        | Cost                               |  5         |  8         |

    is preceded by a short heading ("Annual Results" = 2 tokens) whose token
    count EQUALS the value-column count (2), so it passes the exact-count GATE.
    The value-like (`2024`/`2025`) flagged header cells previously tripped the
    Class-B `_is_mis_structured` data-like branch, demoting the real `2024/2025`
    header to a data row and substituting `| | Annual Results | ... |` — a
    corruption in exactly the financial-table domain we target.

    The col-0 distinguisher fixes it: the flagged header row's corner cell is
    EMPTY and not header-flagged, so the table is NOT a re-attach candidate.
    Asserts the re-attach is a no-op AND the grid is byte-unchanged.

    Pre-fix this test FAILS (the table gets corrupted: num_rows 3 -> 4); post-fix
    it passes (no-op).
    """
    doc = DoclingDocument(name="yearhdr")
    doc.add_heading("Annual Results", level=2)  # 2 tokens == 2 value columns
    cells = [
        # row 0: empty corner (NOT header-flagged) + two value-like year headers.
        _cell("", 0, 0),
        _cell("2024", 0, 1, header=True),
        _cell("2025", 0, 2, header=True),
        # data rows
        _cell("Revenue", 1, 0),
        _cell("10", 1, 1),
        _cell("20", 1, 2),
        _cell("Cost", 2, 0),
        _cell("5", 2, 1),
        _cell("8", 2, 2),
    ]
    table = doc.add_table(data=TableData(table_cells=cells, num_rows=3, num_cols=3))
    before = _grid_snapshot(table)

    n = reattach_detached_table_headers(doc)

    assert n == 0
    assert table.data.num_rows == 3
    # Byte-unchanged: deep compare every cell's text/flag/offsets.
    assert _grid_snapshot(table) == before


def test_reattach_skips_when_col0_header_is_itself_flagged() -> None:
    """A normal `| Metric | 2024 | 2025 |` header where col-0 (`Metric`) is
    itself `column_header=True`. Even though the value columns are value-like
    (`2024`/`2025`), the col-0 distinguisher sees a header-flagged corner →
    this is a real header row, not a mis-flagged data row → leave it untouched.
    """
    doc = DoclingDocument(name="metrichdr")
    doc.add_heading("Annual Results", level=2)  # would tokenize but must not matter
    cells = [
        _cell("Metric", 0, 0, header=True),
        _cell("2024", 0, 1, header=True),
        _cell("2025", 0, 2, header=True),
        _cell("Revenue", 1, 0),
        _cell("10", 1, 1),
        _cell("20", 1, 2),
    ]
    table = doc.add_table(data=TableData(table_cells=cells, num_rows=2, num_cols=3))
    before = _grid_snapshot(table)

    n = reattach_detached_table_headers(doc)

    assert n == 0
    assert table.data.num_rows == 2
    assert _grid_snapshot(table) == before


# --------------------------------------------------------------------------- #
# 6. Byte-identity on a single-header doc                                      #
# --------------------------------------------------------------------------- #


def test_byte_identity_on_single_header_doc() -> None:
    doc = DoclingDocument(name="bytes")
    doc.add_title("My Document")
    doc.add_heading("Introduction", level=1)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Some prose with under_scores and an <html> tag and a | pipe.",
    )
    cells = [
        _cell("Name", 0, 0, header=True),
        _cell("Value", 0, 1, header=True),
        _cell("alpha", 1, 0),
        _cell("1", 1, 1),
        _cell("beta", 2, 0),
        _cell("2", 2, 1),
    ]
    doc.add_table(data=TableData(table_cells=cells, num_rows=3, num_cols=2))

    assert export_markdown_header_aware(doc) == doc.export_to_markdown()


def test_byte_identity_on_prose_only_doc() -> None:
    doc = DoclingDocument(name="prose")
    doc.add_title("Title")
    doc.add_heading("Section One", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="First paragraph.")
    doc.add_heading("Section Two", level=2)
    doc.add_text(label=DocItemLabel.TEXT, text="Second paragraph with _emphasis_.")
    assert export_markdown_header_aware(doc) == doc.export_to_markdown()


# --------------------------------------------------------------------------- #
# 7. Worker wiring — reattach then header-aware export, in order               #
# --------------------------------------------------------------------------- #


def test_worker_calls_reattach_then_header_aware_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """`docling_worker._convert_to_payload` must call
    `reattach_detached_table_headers` BEFORE `export_markdown_header_aware`,
    and must NOT call the stock `doc.export_to_markdown()` for the body."""
    from memex.parse import docling_worker

    calls: list[str] = []

    class _FakeDoc:
        texts: ClassVar[list[object]] = []
        tables: ClassVar[list[object]] = []
        pages: ClassVar[list[object]] = []
        pictures: ClassVar[list[object]] = []
        equations: ClassVar[list[object]] = []

        def export_to_markdown(self) -> str:  # pragma: no cover - must not be hit for body
            calls.append("stock_export")
            return "STOCK"

    fake_doc = _FakeDoc()

    class _FakeResult:
        document = fake_doc

    class _FakeConverter:
        def __init__(self, *a: object, **k: object) -> None: ...

        def convert(self, source: object) -> _FakeResult:
            return _FakeResult()

    # Stub the heavy docling imports the worker pulls in lazily.
    import sys
    import types

    def _install(modname: str, **attrs: object) -> None:
        mod = types.ModuleType(modname)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, modname, mod)

    class _InputFormat:
        PDF = "pdf"

    class _PdfPipelineOptions:
        def __init__(self) -> None:
            self.do_ocr = False
            self.do_picture_classification = True

    class _PdfFormatOption:
        def __init__(self, *a: object, **k: object) -> None: ...

    _install("docling", __version__="test")
    _install(
        "docling.datamodel",
    )
    _install("docling.datamodel.base_models", InputFormat=_InputFormat)
    _install("docling.datamodel.pipeline_options", PdfPipelineOptions=_PdfPipelineOptions)
    _install(
        "docling.document_converter",
        DocumentConverter=_FakeConverter,
        PdfFormatOption=_PdfFormatOption,
    )

    def _fake_reattach(doc: object) -> int:
        calls.append("reattach")
        return 2

    def _fake_export(doc: object) -> str:
        calls.append("header_aware_export")
        return "# header-aware body"

    monkeypatch.setattr(docling_worker, "_recover_heading_levels", lambda doc: 0)
    monkeypatch.setattr(
        docling_worker, "_demote_misdetected_headers", lambda doc, *, text_item_cls: 0
    )
    # Patch the symbols inside the docling_tables module the worker imports lazily.
    from memex.parse import docling_tables

    monkeypatch.setattr(docling_tables, "reattach_detached_table_headers", _fake_reattach)
    monkeypatch.setattr(docling_tables, "export_markdown_header_aware", _fake_export)

    from pathlib import Path

    payload = docling_worker._convert_to_payload(Path("/nonexistent.pdf"))

    assert "stock_export" not in calls, "worker must not use the stock export for the body"
    assert calls.index("reattach") < calls.index("header_aware_export"), calls
    assert payload["markdown"] == "# header-aware body"
