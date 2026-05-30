"""W12 hardening — flattened-merged-header skip + furniture-column trim."""
from __future__ import annotations

from memex.core.table_linearize import (
    header_has_lost_columns,
    linearize_gfm_tables,
    parse_gfm_table,
)
from memex.index.table_store import extract_tables


def _rows_block(markdown: str) -> str:
    start = markdown.index("[table-rows]") + len("[table-rows]\n")
    end = markdown.index("\n[/table-rows]")
    return markdown[start:end]


# --- header_has_lost_columns predicate (unit) -------------------------------

def test_lost_columns_flags_multi_empty_merged_header() -> None:
    # Live 10-K "components of pay": group labels over interior empties.
    hdr = ["", "Fixed Compensation", "", "At-Risk", "Compensation", ""]
    assert header_has_lost_columns(hdr) is True


def test_lost_columns_allows_single_interior_empty() -> None:
    # A LONE interior empty is a REAL table that lost ONE column's label (the gte
    # hyperparameter table). It is NOT flagged: the linearizer keys that one
    # column POSITIONALLY (col{j+1}, never a wrong label), so it surfaces
    # fabrication-safely. The ≥2-empties threshold targets the genuine
    # merged-header flatten (group label spans ≥2 columns → ≥2 gaps).
    assert header_has_lost_columns(["Year", "", "Total"]) is False
    assert header_has_lost_columns(["Model", "Params", "", "LR", "GPUs"]) is False


def test_lost_columns_flags_two_interior_empties() -> None:
    # ≥2 empties = the merged-header flatten signal → skip.
    assert header_has_lost_columns(["", "Fixed Compensation", "", "At-Risk"]) is True


def test_lost_columns_allows_single_leading_corner() -> None:
    # The classic row-label layout — one empty col-0 — is NOT flagged.
    assert header_has_lost_columns(["", "2024", "2025"]) is False


def test_lost_columns_allows_fully_named_header() -> None:
    # The 10-K director-compensation gold table.
    assert (
        header_has_lost_columns(
            ["Name", "Fees Earned or Paid in Cash ($)", "Stock Awards ($) (1)", "Total ($)"]
        )
        is False
    )


def test_lost_columns_empty_header_is_false() -> None:
    assert header_has_lost_columns([]) is False


# --- linearizer SKIPS the flattened comp table (no mis-keyed KV) ------------

def test_flattened_comp_table_is_skipped() -> None:
    md = (
        "|  | Fixed Compensation |  | At-Risk | Compensation |  |\n"
        "|---|---|---|---|---|---|\n"
        "|  | Base Salary | Variable Cash | SY PSUs | MYPSUs | RSUs (1) |\n"
        "| Form | Cash | Cash | Equity | Equity | Equity |\n"
    )
    out = linearize_gfm_tables(md)
    # No KV at all (the raw table stays in the .md for fidelity).
    assert "[table-rows]" not in out
    # And in particular none of the mis-keyed pairs the old code emitted.
    assert "col3=Variable Cash" not in out
    assert "At-Risk=SY PSUs" not in out


def test_single_empty_corner_row_label_still_linearizes() -> None:
    """Regression: the legitimate single-empty-corner row-label layout MUST
    still produce its correct KV (not collaterally skipped by the W12 gate)."""
    md = "| | 2024 | 2025 |\n|---|---|---|\n| Compute | 10 | 20 |\n"
    out = _rows_block(linearize_gfm_tables(md))
    assert out == "Compute: 2024=10, 2025=20"


def test_named_header_table_still_linearizes() -> None:
    md = (
        "| Name | Fees Earned or Paid in Cash ($) | Total ($) |\n"
        "|---|---|---|\n"
        "| Robert K. Burgess | 63,750 | 263,750 |\n"
    )
    out = _rows_block(linearize_gfm_tables(md))
    assert out == "Name=Robert K. Burgess, Fees Earned or Paid in Cash ($)=63,750, Total ($)=263,750"


def test_single_interior_empty_table_linearizes_positionally() -> None:
    """A REAL table that lost ONE interior column label (the gte hyperparameter
    table) is NOT skipped — it linearizes with the labelled columns keyed
    correctly and the one empty column keyed POSITIONALLY (col{j+1}), never with
    a wrong adjacent label. Fabrication-safe: no value maps to a wrong column."""
    md = "| Model | Params | | LR |\n|---|---|---|---|\n| GTE-base | 110M | 50k | 6e-4 |\n"
    out = _rows_block(linearize_gfm_tables(md))
    # col-2 (empty header) → positional `col3`; the labelled columns key right.
    assert out == "Model=GTE-base, Params=110M, col3=50k, LR=6e-4"


# --- furniture column trim ---------------------------------------------------

def test_trailing_furniture_column_dropped() -> None:
    """A trailing header-only 'P' column over 12-cell rows (the live
    which-chart table) must NOT leave a dangling `P=` on every row."""
    md = (
        "| City | 2000 | 2001 | P |\n"
        "| --- | --- | --- | --- |\n"
        "| New York city | 8,017,608 | 8,059,813 |\n"
        "| Los Angeles | 3,701,062 | 3,733,328 |\n"
    )
    out = _rows_block(linearize_gfm_tables(md))
    lines = out.split("\n")
    assert lines[0] == "City=New York city, 2000=8,017,608, 2001=8,059,813"
    assert "P=" not in out  # the furniture key is gone


def test_furniture_trim_keeps_a_filled_ragged_column() -> None:
    """A trailing column ONE ragged data row DOES fill is kept (positional)."""
    md = (
        "| A | B | C |\n"
        "| --- | --- | --- |\n"
        "| 1 | 2 | 3 |\n"  # full width — C is real
        "| 4 | 5 |\n"
    )
    out = _rows_block(linearize_gfm_tables(md))
    assert "C=3" in out  # not trimmed (a data row reaches col C)


# --- table store coherence ---------------------------------------------------

def test_table_store_skips_flattened_comp_table() -> None:
    body = (
        "## Components of pay\n\n"
        "|  | Fixed Compensation |  | At-Risk | Compensation |  |\n"
        "|---|---|---|---|---|---|\n"
        "|  | Base Salary | Variable Cash | SY PSUs | MYPSUs | RSUs (1) |\n"
        "| Form | Cash | Cash | Equity | Equity | Equity |\n"
    )
    assert extract_tables("doc1", body) == []


def test_table_store_keeps_named_header_table() -> None:
    body = (
        "## Director comp\n\n"
        "| Name | Fees ($) | Total ($) |\n"
        "|---|---|---|\n"
        "| Burgess | 63,750 | 263,750 |\n"
    )
    tables = extract_tables("doc1", body)
    assert len(tables) == 1
    assert tables[0].header == ["Name", "Fees ($)", "Total ($)"]


# --- determinism / idempotence preserved ------------------------------------

def test_skip_is_idempotent_and_deterministic() -> None:
    md = (
        "|  | Fixed Compensation |  | At-Risk |\n"
        "|---|---|---|---|\n"
        "|  | Base | Variable | SY |\n"
    )
    once = linearize_gfm_tables(md)
    assert once == md  # skipped → unchanged
    assert linearize_gfm_tables(once) == once  # idempotent


def test_furniture_trim_is_deterministic() -> None:
    md = "| City | 2000 | P |\n|---|---|---|\n| NYC | 8,017,608 |\n"
    assert linearize_gfm_tables(md) == linearize_gfm_tables(md)


def test_parse_gfm_table_unchanged_by_w12() -> None:
    """W12 lives in the CALLER (gate + trim), not parse_gfm_table — the raw
    parse still returns the flattened header verbatim (table_store/linearizer
    apply the skip)."""
    blk = (
        "|  | Fixed Compensation |  | At-Risk |\n"
        "|---|---|---|---|\n"
        "|  | Base | Variable | SY |\n"
    )
    parsed = parse_gfm_table(blk)
    assert parsed is not None
    header, _ = parsed
    assert header == ["", "Fixed Compensation", "", "At-Risk"]
