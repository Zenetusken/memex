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
    _rank_tables,
    _render_table,
    _table_chunks,
    _table_salience,
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
    tables = [_table(f"S{i}", ["A"], [["v"], ["w"]]) for i in range(30)]
    chunks = _table_chunks("docA", tables, "Doc A")
    assert chunks[0].chunk_id == "docA#tbl0"
    assert chunks[0].heading_path == ["S0"]
    assert chunks[0].document_id == "docA"
    assert len(chunks) <= 24  # _MAX_TABLES_FOR_FIGURES
    assert all(c.chunk_id.startswith("docA#tbl") for c in chunks)


# ── figure-salience (ADR-0008 §7) — rank tables by figure-richness, not doc order ──

_FINANCIAL = _table(
    "Consolidated Statements of Income",
    ["Item", "FY2026"],
    [["Revenue", "$130,000M"], ["Net income", "$72,500M"], ["Diluted EPS", "$2.94"]],
)
_GLOSSARY = _table(
    "Definitions",
    ["Term", "Meaning"],
    [["VLAN", "a virtual local area network"], ["OSPF", "a link-state routing protocol"]],
)


def test_salience_prefers_numeric_money_table_over_text() -> None:
    assert _table_salience(_FINANCIAL) > _table_salience(_GLOSSARY)


def test_salience_zero_for_single_row_fragment() -> None:
    assert _table_salience(_table("X", ["A", "B"], [["only one row", "$5M"]])) == 0.0


def test_salience_keyword_breaks_ties_between_equally_numeric_tables() -> None:
    # Same numeric content; only the section keyword differs → headline section wins.
    numbers = [["a", "10"], ["b", "20"], ["c", "30"]]
    headline = _table("Revenue by segment", ["k", "v"], numbers)
    bland = _table("Appendix A", ["k", "v"], numbers)
    assert _table_salience(headline) > _table_salience(bland)


def test_salience_penalizes_wide_framing_risky_grids() -> None:
    # A wide, dense time-series grid (a performance graph) is framing-risky → it
    # ranks below a narrow income statement even though it's MORE numeric-dense.
    narrow = _table(
        "Consolidated Statements of Income",
        ["Item", "FY2026"],
        [["Revenue", "$130,000M"], ["Operating income", "$80,000M"], ["Net income", "$72,000M"]],
    )
    wide = _table(
        "Stock Performance Graph",
        ["Company", "2020", "2021", "2022", "2023", "2024", "2025"],
        [
            ["NVIDIA", "$100", "$220", "$180", "$340", "$900", "$1,448"],
            ["S&P 500", "$100", "$120", "$110", "$130", "$150", "$170"],
        ],
    )
    # The wide grid is MORE numeric-dense, so only the multiplicative width factor
    # (not a subtractive penalty) makes the focused income statement rank higher.
    assert _table_salience(narrow) > _table_salience(wide)


def test_rank_tables_orders_by_salience_then_stable_doc_order() -> None:
    # Two equally-salient financial tables keep document order; glossary sinks last.
    a = _table("Income statement", ["k", "v"], [["Revenue", "$1M"], ["Profit", "$0.5M"]])
    b = _table("Cash flow", ["k", "v"], [["Operating", "$2M"], ["Investing", "$1M"]])
    ranked = _rank_tables([_GLOSSARY, a, b])
    assert ranked[-1] is _GLOSSARY  # text table is least salient
    assert ranked.index(a) < ranked.index(b)  # tie → original (input) order preserved
