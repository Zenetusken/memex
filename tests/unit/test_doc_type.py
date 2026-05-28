"""Unit tests for the doc-type router + section grouping (ADR-0008).

Pure functions — no model, no store. They decide the summarization STRATEGY
from the document's shape (size + section count), which is what keeps quality
independent of the co-residence mode (the "baseline rule").
"""

from __future__ import annotations

from memex.agents.answering import CitedClaim, SectionSummary
from memex.agents.document_summarizer import (
    _MAX_SECTION_INPUT_CHARS,
    _MAX_SECTIONS,
    _MAX_TABLE_ROWS,
    _bound_section_chunks,
    _classify_route,
    _dedup_sentences,
    _group_sections,
    _pack_sections,
    _rank_tables,
    _render_table,
    _select_doc_key_points,
    _should_pack_sections,
    _split_section_into_batches,
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


# ── deck route (ADR-0008) — pack tiny slide-sections into substantive MAP units ──


def _sec(i: int, chars: int) -> tuple[str, list[Chunk]]:
    return (f"S{i}", [_c(f"docA#{i}", f"S{i}", "x" * chars)])


def test_should_pack_true_for_many_tiny_sections() -> None:
    # 15 slide-sized sections → thin per-section digests → pack.
    assert _should_pack_sections([_sec(i, 200) for i in range(15)]) is True


def test_should_pack_false_below_min_sections() -> None:
    assert _should_pack_sections([_sec(i, 200) for i in range(8)]) is False  # < _PACK_MIN_SECTIONS


def test_should_pack_false_when_sections_are_substantive() -> None:
    # Enough sections, but they're paper-sized (none tiny) → keep per-section digests.
    assert _should_pack_sections([_sec(i, 3000) for i in range(14)]) is False


def test_pack_sections_merges_to_budget_titled_by_first_in_order() -> None:
    # 6 sections, each measured at 1,800 (truncate cap); budget 5,000 → 2 per group.
    groups = _pack_sections([_sec(i, 4000) for i in range(6)], 5_000)
    assert [t for t, _ in groups] == ["S0", "S2", "S4"]  # titled by first, order kept
    assert all(len(cs) == 2 for _, cs in groups)


def test_pack_sections_caps_at_max_sections() -> None:
    groups = _pack_sections([_sec(i, 4000) for i in range(100)], 1)  # budget 1 → 1 section/group
    assert len(groups) <= _MAX_SECTIONS


# ── section sub-splitting (ADR-0008) — a huge section spans batches, no content dropped ──


def test_split_section_multi_batch_drops_no_content() -> None:
    chunks = [_c(f"docA#{i}", "Big", "x" * 4000) for i in range(10)]  # 10×1800=18000 > the cap
    batches = _split_section_into_batches(chunks, _MAX_SECTION_INPUT_CHARS)
    assert len(batches) > 1  # splits
    # every chunk lands in exactly one batch, in order — nothing truncated away
    assert [c for batch in batches for c in batch] == chunks
    # each batch fits the window budget (mode-independent)
    assert all(
        sum(min(len(c.text), 1_800) for c in batch) <= _MAX_SECTION_INPUT_CHARS for batch in batches
    )


def test_split_section_single_batch_when_it_fits() -> None:
    chunks = [_c(f"docA#{i}", "S", "x" * 1_000) for i in range(3)]  # 3×1000 = 3000 < budget
    assert _split_section_into_batches(chunks, _MAX_SECTION_INPUT_CHARS) == [chunks]


def test_split_section_lone_giant_chunk_is_its_own_batch() -> None:
    chunks = [_c("docA#0", "S", "y" * 50_000)]
    assert _split_section_into_batches(chunks, _MAX_SECTION_INPUT_CHARS) == [chunks]


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


def _kp(claim: str) -> CitedClaim:
    return CitedClaim(claim=claim, source_chunk_id="d#x", confidence="high")


def _ss(title: str, points: list[str]) -> SectionSummary:
    return SectionSummary(section_title=title, digest="d", key_points=[_kp(p) for p in points])


def test_select_doc_key_points_round_robins_across_sections() -> None:
    """A front-matter section with many trivial points must NOT fill the whole
    cap before substantive sections contribute — round-robin takes each
    section's top point first. (The NIST-boilerplate regression.)"""
    sections = [
        _ss("Cover", ["c1", "c2", "c3", "c4", "c5", "c6"]),  # front-matter: 6 points
        _ss("Introduction", ["i1", "i2"]),
        _ss("Zero Trust Tenets", ["t1", "t2"]),
        _ss("Components", ["k1"]),
    ]
    picked = [c.claim for c in _select_doc_key_points(sections, cap=4)]
    # Round-robin rank 0: one from EACH section → all 4 distinct sections represented,
    # NOT 4 cover points.
    assert picked == ["c1", "i1", "t1", "k1"]


def test_select_doc_key_points_takes_second_rank_after_first_pass() -> None:
    sections = [_ss("A", ["a1", "a2"]), _ss("B", ["b1"])]
    # cap 4 but only 3 points exist → a1,b1 (rank0), a2 (rank1), then stop.
    assert [c.claim for c in _select_doc_key_points(sections, cap=4)] == ["a1", "b1", "a2"]


def test_select_doc_key_points_single_section_keeps_order() -> None:
    """A single-section doc is unchanged — points come out in their original order."""
    sections = [_ss("Only", ["p1", "p2", "p3"])]
    assert [c.claim for c in _select_doc_key_points(sections, cap=12)] == ["p1", "p2", "p3"]


# ── cross-paragraph dedup gate (ADR-0010) — drop sentences that repeat earlier content ──


def test_dedup_sentences_drops_near_restatement_keeps_novel() -> None:
    """A later sentence that restates an earlier one (most of its content tokens already
    covered by a SINGLE kept sentence) is dropped; a genuinely new one survives."""
    kept: list[set[str]] = []
    opening = _dedup_sentences(
        ["Spanning Tree Protocol prevents Layer 2 loops by blocking redundant ports."], kept
    )
    assert len(opening) == 1 and kept  # the opening seeds the running ledger
    survivors = _dedup_sentences(
        [
            # near-restatement of the opening → dropped
            "Layer 2 loops are prevented by Spanning Tree Protocol blocking redundant ports.",
            # genuinely new content → kept
            "RSTP introduces backup and alternate port roles for faster convergence.",
        ],
        kept,
    )
    assert survivors == ["RSTP introduces backup and alternate port roles for faster convergence."]


def test_dedup_sentences_keeps_sentence_sharing_only_common_terms() -> None:
    """Sharing a few common terms with an earlier sentence is NOT a repeat (overlap is
    measured as PRECISION against the best single match, not pooled across all priors)."""
    kept: list[set[str]] = []
    _dedup_sentences(["The root bridge is elected using the lowest bridge ID."], kept)
    survivors = _dedup_sentences(
        ["Designated ports forward frames on each network segment toward the root bridge."], kept
    )
    assert len(survivors) == 1  # only "root"/"bridge" shared → below threshold → survives


def test_dedup_sentences_empty_and_exact_duplicate() -> None:
    """An exact duplicate is dropped; the ledger only grows with content-bearing kept ones."""
    kept: list[set[str]] = []
    s = "Switches exchange BPDU frames to elect the root bridge of the topology."
    assert _dedup_sentences([s], kept) == [s]
    assert _dedup_sentences([s], kept) == []  # exact repeat dropped
    assert len(kept) == 1  # ledger unchanged by the dropped repeat
