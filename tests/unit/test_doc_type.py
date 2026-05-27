"""Unit tests for the doc-type router + section grouping (ADR-0008).

Pure functions — no model, no store. They decide the summarization STRATEGY
from the document's shape (size + section count), which is what keeps quality
independent of the co-residence mode (the "baseline rule").
"""

from __future__ import annotations

from memex.agents.document_summarizer import (
    _MAX_SECTION_INPUT_CHARS,
    _MAX_TABLE_ROWS,
    _bound_section_chunks,
    _classify_route,
    _group_sections,
    _render_table,
    _table_chunks,
)
from memex.core.types import Chunk, StoredTable


def _c(cid: str, heading: str | None, text: str = "x") -> Chunk:
    return Chunk(
        chunk_id=cid,
        document_id="docA",
        document_title="Doc A",
        text=text,
        heading_path=[heading] if heading else [],
    )


def test_group_sections_by_deepest_heading_preserves_order() -> None:
    chunks = [_c("docA#1", "Intro"), _c("docA#2", "Intro"), _c("docA#3", "Methods")]
    groups = _group_sections(chunks)
    assert [t for t, _ in groups] == ["Intro", "Methods"]
    assert [len(cs) for _, cs in groups] == [2, 1]


def test_group_sections_flat_chunk_falls_back_to_doc_title() -> None:
    chunks = [_c("docA#1", None), _c("docA#2", None)]
    groups = _group_sections(chunks)
    assert [t for t, _ in groups] == ["Doc A"]  # document_title fallback for a flat doc
    assert len(groups[0][1]) == 2


def test_classify_short_when_small() -> None:
    chunks = [_c("docA#1", "Intro", "short"), _c("docA#2", "Methods", "short")]
    assert _classify_route(chunks, _group_sections(chunks)) == "short"


def test_classify_short_when_single_section_even_if_large() -> None:
    # One section → one MAP pass is the whole summary, so route short regardless of size.
    chunks = [_c("docA#1", "Intro", "x" * 50_000)]
    assert _classify_route(chunks, _group_sections(chunks)) == "short"


def test_classify_long_when_big_and_multi_section() -> None:
    chunks = [_c("docA#1", "Intro", "a" * 9_000), _c("docA#2", "Methods", "b" * 9_000)]
    assert _classify_route(chunks, _group_sections(chunks)) == "long"


def test_bound_section_chunks_caps_input_to_fast_window_budget() -> None:
    # Many large chunks (each truncated to ~1,800 in the prompt) — the helper keeps
    # only enough to stay within the per-call char budget that fits the fast window.
    chunks = [_c(f"docA#{i}", "Sec", "z" * 5_000) for i in range(16)]
    bounded = _bound_section_chunks(chunks)
    assert 0 < len(bounded) < len(chunks)
    # The mode-independence guarantee: the kept text fits the budget.
    assert sum(min(len(c.text), 1_800) for c in bounded) <= _MAX_SECTION_INPUT_CHARS


def test_bound_section_chunks_always_keeps_at_least_one() -> None:
    # A lone giant chunk is kept (the prompt truncates it) — never an empty section.
    bounded = _bound_section_chunks([_c("docA#1", "Sec", "y" * 200_000)])
    assert len(bounded) == 1


def test_bound_section_chunks_keeps_all_when_small() -> None:
    chunks = [_c("docA#1", "Sec", "short"), _c("docA#2", "Sec", "also short")]
    assert _bound_section_chunks(chunks) == chunks


# ── tabular route (ADR-0008) — table rendering + synthetic chunks ──


def _table(section: str, header: list[str], rows: list[list[str]]) -> StoredTable:
    return StoredTable(
        doc_id="docA",
        table_id="t0",
        section=section,
        header=header,
        rows=rows,
        char_start=0,
        char_end=10,
    )


def test_render_table_pairs_header_with_each_cell_verbatim() -> None:
    t = _table(
        "Segments", ["Segment", "Revenue"], [["Gaming", "$16,042M"], ["Data Center", "$30B"]]
    )
    text = _render_table(t)
    assert "Table — Segments" in text
    assert "Segment=Gaming; Revenue=$16,042M" in text  # self-contained, verbatim
    assert "Segment=Data Center; Revenue=$30B" in text


def test_render_table_caps_rows() -> None:
    rows = [[f"r{i}", f"{i}"] for i in range(_MAX_TABLE_ROWS + 20)]
    text = _render_table(_table("Big", ["Name", "N"], rows))
    # header line + label line + at most _MAX_TABLE_ROWS data lines
    assert len(text.splitlines()) <= _MAX_TABLE_ROWS + 2


def test_render_table_falls_back_when_row_width_mismatches_header() -> None:
    # A ragged row (more cells than headers) renders as a plain pipe-joined row.
    t = _table("Ragged", ["A", "B"], [["x", "y", "z"]])
    assert "x | y | z" in _render_table(t)


def test_table_chunks_ids_sections_and_cap() -> None:
    tables = [_table(f"S{i}", ["A"], [["v"]]) for i in range(30)]
    chunks = _table_chunks("docA", tables, "Doc A")
    assert chunks[0].chunk_id == "docA#tbl0"
    assert chunks[0].heading_path == ["S0"]
    assert chunks[0].document_id == "docA"
    assert len(chunks) <= 24  # _MAX_TABLES_FOR_FIGURES
    assert all(c.chunk_id.startswith("docA#tbl") for c in chunks)
