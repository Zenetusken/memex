"""Unit tests for the deterministic numeric-grounding backstop (2026-05-31).

The pure helpers behind the `verify` node's 4th demotion filter that closes the
`verify_grounding` aggregate-numeric FALSE-POSITIVE (the summed-table-total the
LLM verifier rubber-stamps via the literal-table-row loophole; the live
`$159,748,343` 10-K fabrication, annual-report-16). Each helper is pure/sync, so
the load-bearing logic is tested in isolation here; the wiring into `verify` is
pinned in `tests/integration/test_answering_with_fakes.py`.
"""

from __future__ import annotations

from memex.agents.answering import (
    _chunk_has_markdown_table,
    _chunk_numbers,
    _claim_scoped_figures,
    _figure_supported_by_chunk,
)

# The fiscal-2026 SY/MY PSU cells from the cited exec Summary Compensation Table
# (#791b185267) — the chunk the fabricated $159,748,343 SUM was grounded against.
_EXEC_CELLS = [
    19166424.0, 18034343.0, 6099993.0, 7350483.0, 5935261.0,
    7151981.0, 6141443.0, 7400430.0, 6141443.0, 7400430.0,
]


# ----- _claim_scoped_figures: scope (>= 1e4 only) ---------------------------


def test_scoped_figures_drops_year_keeps_dollar() -> None:
    # The live failure: the year 2026 (< 1e4) is dropped, the $ aggregate kept.
    claim = "The total value of stock options granted to NVIDIA's directors in fiscal 2026 is $159,748,343."
    assert _claim_scoped_figures(claim) == [159748343.0]


def test_scoped_figures_verbatim_dollar() -> None:
    assert _claim_scoped_figures("The lowest total compensation was $321,309 (Ellen Ochoa).") == [321309.0]


def test_scoped_figures_millions_scale_word() -> None:
    # "$16,042 million" coerces with the scale word baked in.
    assert _claim_scoped_figures("Gaming revenue was $16,042 million in fiscal 2026.") == [16042e6]


def test_scoped_figures_empty_on_year_only() -> None:
    assert _claim_scoped_figures("The policy was adopted in 2026 across all teams.") == []


def test_scoped_figures_empty_on_word_count() -> None:
    assert _claim_scoped_figures("There are seven tenets of zero trust.") == []


def test_scoped_figures_empty_on_small_numeral_count() -> None:
    assert _claim_scoped_figures("The framework defines 7 task categories.") == []


def test_scoped_figures_empty_on_x_suffix() -> None:
    # coerce_number returns None on a trailing `x` (not a scale letter).
    assert _claim_scoped_figures("FP16 matmul costs 0.5x the energy of FP32.") == []


def test_scoped_figures_empty_on_percent_only() -> None:
    assert _claim_scoped_figures("Gross margin was 71.1%.") == []


def test_scoped_figures_empty_on_no_number() -> None:
    assert _claim_scoped_figures("Directors receive RSUs, not stock options.") == []


def test_scoped_figures_returns_all_large() -> None:
    figs = _claim_scoped_figures("Revenue rose from $50,000 to $1,250,000 over the period.")
    assert set(figs) == {50000.0, 1250000.0}


def test_scoped_figures_keeps_rounded_plus_exact() -> None:
    # gte-05 shape: a rounded headline (~800M) AND the exact table total (788M).
    # Both are returned; the any-supported gate keeps the claim on the exact one.
    figs = _claim_scoped_figures("We used ~800M pairs for pre-training (Table 1 totals 788M).")
    assert 8e8 in figs and 7.88e8 in figs


def test_any_supported_keeps_rounded_plus_exact() -> None:
    # The exact 788M is in the table; the rounded 800M is not — the claim is kept
    # because ANY scoped figure is supported (not just the largest).
    figs = _claim_scoped_figures("We used ~800M pairs for pre-training (Table 1 totals 788M).")
    assert any(_figure_supported_by_chunk(f, [33.0, 788e6]) for f in figs) is True


# ----- _figure_supported_by_chunk: the fabrication boundary -----------------


def test_fabricated_sum_not_supported() -> None:
    # $159,748,343 is a SUM of the cells — no single cell at any unit scale.
    assert _figure_supported_by_chunk(159748343.0, _EXEC_CELLS) is False


def test_verbatim_cell_supported() -> None:
    assert _figure_supported_by_chunk(321309.0, [42500.0, 278809.0, 321309.0]) is True


def test_millions_denomination_supported() -> None:
    # "$16,042 million" (1.6042e10) grounds in a "16,042" cell of a $-millions table.
    assert _figure_supported_by_chunk(16042e6, [16042.0]) is True


def test_billions_denomination_supported() -> None:
    # "$22.5 billion" (2.25e10) grounds in a "22,500" $-millions cell.
    assert _figure_supported_by_chunk(2.25e10, [22500.0]) is True


def test_x10_misscale_not_supported() -> None:
    # A x10 shift is a mis-scale, NOT a clean unit denomination — must NOT match.
    # (Guards against a future "add a rounding band" change re-admitting it.)
    assert _figure_supported_by_chunk(74004300.0, [7400430.0]) is False


def test_boundary_span_not_supported() -> None:
    # A digit-substring spanning two adjacent cells is value-distinct → not matched
    # (guards against regressing to a digit-substring matcher).
    assert _figure_supported_by_chunk(14437400.0, [6141443.0, 7400430.0]) is False


def test_tolerance_floor_allows_unit_rounding() -> None:
    # The aggregate-gate tolerance max(1, 1e-6*|b|) absorbs a sub-ppm difference.
    assert _figure_supported_by_chunk(321309.0, [321309.4]) is True


def test_near_miss_cell_not_supported() -> None:
    # A cell off by more than the tolerance does NOT match (159,748,343 vs a
    # hypothetical 159,748 cell x1000 = 159,748,000, diff 343 > tol ~160).
    assert _figure_supported_by_chunk(159748343.0, [159748.0]) is False


def test_empty_chunk_numbers_not_supported() -> None:
    assert _figure_supported_by_chunk(159748343.0, []) is False


# ----- _chunk_has_markdown_table: the table-presence narrowing ---------------


def test_table_presence_gfm_separator() -> None:
    assert _chunk_has_markdown_table("| A | B |\n|---|---|\n| 1 | 2 |") is True


def test_table_presence_gfm_separator_with_alignment() -> None:
    assert _chunk_has_markdown_table("| A | B |\n|:--|--:|\n| 1 | 2 |") is True


def test_table_presence_table_rows_marker() -> None:
    assert _chunk_has_markdown_table("Heading\n\n[table-rows]\n[Seg] Gaming=16,042\n") is True


def test_table_presence_false_on_prose() -> None:
    # A rounded prose headline ("$216 billion") cites a PROSE chunk → exempt.
    assert _chunk_has_markdown_table("Revenue grew 65 percent to $216 billion this year.") is False


def test_table_presence_false_on_pipe_in_prose() -> None:
    # A stray pipe in prose is not a table (no separator row).
    assert _chunk_has_markdown_table("Use the A | B operator carefully.") is False


# ----- _chunk_numbers: extraction over the full chunk ------------------------


def test_chunk_numbers_extracts_gfm_and_linearized() -> None:
    nums = _chunk_numbers("| Gaming | 16,042 | 11,350 |\n[table-rows]\n[Seg] Gaming=16,042")
    assert 16042.0 in nums and 11350.0 in nums


def test_chunk_numbers_handles_currency_and_scale() -> None:
    nums = _chunk_numbers("Revenue was $22.5 billion and margin 71.1%.")
    assert 2.25e10 in nums
