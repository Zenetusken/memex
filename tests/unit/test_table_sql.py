"""Unit tests for the text-to-SQL helper (Table-RAG Phase 2).

Covers `agents/table_sql.py`:
  - `coerce_number` grammar table (load-bearing 10-K shapes).
  - `query_doc_tables`: row-SELECT → kind="rows" (verbatim cells);
    single-column SUM with filter → kind="aggregate" only when the independent
    recompute agrees; wrong-column SQL → recompute disagrees → None;
    group-by / join / derived superlative → None; non-SELECT / injection
    (`;`, DROP, PRAGMA, ATTACH) → None; empty / exec-error → None.

`complete_structured` is patched at `memex.agents.table_sql.complete_structured`
with a `**_kw: object` fake (forward-compatible kwargs).
"""

from __future__ import annotations

import re
import sqlite3

import pytest

from memex.agents import table_sql
from memex.agents.table_sql import coerce_number, query_doc_tables
from memex.core.types import GeneratedSQL, StoredTable

# ======================================================================
# coerce_number grammar
# ======================================================================


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("$22.5 billion", 2.25e10),
        ("(1,234)", -1234.0),
        ("45%", 45.0),
        ("1,000,000", 1e6),
        ("$1,234.56", 1234.56),
        ("2.5B", 2.5e9),
        ("3K", 3000.0),
        ("€1,000", 1000.0),  # `,` is the thousands sep (`.` stays decimal)
        ("€500", 500.0),  # leading euro symbol stripped
        ("-50", -50.0),
        ("$-5", -5.0),
        ("100", 100.0),
        ("N/A", None),
        ("Revenue", None),
        ("", None),
        ("Compute & Networking", None),
        ("B", None),  # bare scale letter is not a number
    ],
)
def test_coerce_number(cell: str, expected: float | None) -> None:
    got = coerce_number(cell)
    if expected is None:
        assert got is None
    else:
        assert got is not None
        assert got == pytest.approx(expected)


# ======================================================================
# query_doc_tables — fixtures + helpers
# ======================================================================


def _segments_table() -> StoredTable:
    return StoredTable(
        doc_id="doc-1",
        table_id="abc1234567",
        section="Reportable Segments",
        header=["Segment", "Revenue"],
        rows=[
            ["Compute & Networking", "$116,193"],
            ["Graphics", "$17,109"],
            ["All Other", "$1,000"],
        ],
        char_start=10,
        char_end=120,
    )


def _patch_sql(monkeypatch: pytest.MonkeyPatch, sql: str, target_table_id: str) -> None:
    async def _fake(*, prompt: object, schema: type, **_kw: object) -> tuple[object, int]:
        assert schema is GeneratedSQL
        return GeneratedSQL(sql=sql, target_table_id=target_table_id), 10

    monkeypatch.setattr("memex.agents.table_sql.complete_structured", _fake)


# ======================================================================
# kind="rows"
# ======================================================================


@pytest.mark.asyncio
async def test_row_select_returns_verbatim_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    t = _segments_table()
    _patch_sql(
        monkeypatch,
        f'SELECT * FROM "{table_sql._sanitize_identifier(t.table_id, fallback="t0")}" '
        "ORDER BY revenue__num DESC LIMIT 1",
        t.table_id,
    )
    result = await query_doc_tables("which segment had the highest revenue?", [t])
    assert result is not None
    assert result.kind == "rows"
    assert result.rows is not None
    # Highest revenue → Compute & Networking (verbatim cells).
    assert result.rows[0] == ["Compute & Networking", "$116,193"]
    assert result.target_table_id == t.table_id
    assert result.char_start == 10 and result.char_end == 120


# ======================================================================
# kind="aggregate" — recompute agrees / disagrees
# ======================================================================


@pytest.mark.asyncio
async def test_aggregate_sum_agrees_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    t = _segments_table()
    sql_name = table_sql._sanitize_identifier(t.table_id, fallback="t0")
    _patch_sql(monkeypatch, f'SELECT SUM(revenue__num) FROM "{sql_name}"', t.table_id)
    result = await query_doc_tables("total revenue across all segments?", [t])
    assert result is not None
    assert result.kind == "aggregate"
    assert result.aggregate_value == pytest.approx(116193 + 17109 + 1000)
    assert len(result.contributing_rows) == 3


@pytest.mark.asyncio
async def test_aggregate_with_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    t = _segments_table()
    sql_name = table_sql._sanitize_identifier(t.table_id, fallback="t0")
    _patch_sql(
        monkeypatch,
        f"SELECT SUM(revenue__num) FROM \"{sql_name}\" WHERE segment = 'Graphics'",
        t.table_id,
    )
    result = await query_doc_tables("total Graphics revenue?", [t])
    assert result is not None
    assert result.kind == "aggregate"
    assert result.aggregate_value == pytest.approx(17109)
    assert result.contributing_rows == [["Graphics", "$17,109"]]


@pytest.mark.asyncio
async def test_aggregate_count(monkeypatch: pytest.MonkeyPatch) -> None:
    t = _segments_table()
    sql_name = table_sql._sanitize_identifier(t.table_id, fallback="t0")
    _patch_sql(monkeypatch, f'SELECT COUNT(*) FROM "{sql_name}"', t.table_id)
    result = await query_doc_tables("how many segments?", [t])
    assert result is not None
    assert result.kind == "aggregate"
    assert result.aggregate_value == pytest.approx(3)


@pytest.mark.asyncio
async def test_aggregate_recompute_disagree_real_no_patch_returns_none() -> None:
    """REAL recompute-disagreement — no monkeypatch on `_recompute_aggregate`.

    Drives a genuine numeric disagreement through the whole live path
    (`_load_tables` → sqlite SELECT → `_AGG_RE` → real `_recompute_aggregate`
    → the `abs() > tol` tolerance check). The text columns are `COLLATE NOCASE`,
    so a pure CASE mismatch now AGREES (the value-linking recall fix); the
    surviving divergence the independent oracle still catches is WHITESPACE —
    the recompute's `_row_matches` strips both sides, sqlite's NOCASE `=` does
    not:

      - filter column: `Graphics` (100), ` graphics ` (999, padded), `Other` (1).
      - the LLM emits `SUM(revenue__num) WHERE segment = 'graphics'`.
      - sqlite NOCASE `=` matches `Graphics` (case-insensitive) but NOT the
        padded ` graphics ` (no strip) → SUM = 100.
      - the independent recompute strips+lowercases → matches BOTH → 100+999=1099.
      - 100 vs 1099 exceeds `max(1, 1e-6*|1099|)` → the gate refuses → None.

    This proves the INDEPENDENT row-selection oracle (not 'sqlite agrees with
    sqlite') still gates a real divergence even after NOCASE."""
    t = StoredTable(
        doc_id="doc-2",
        table_id="def7654321",
        section="Mixed Case",
        header=["Segment", "Revenue"],
        rows=[
            ["Graphics", "100"],
            [" graphics ", "999"],
            ["Other", "1"],
        ],
        char_start=0,
        char_end=50,
    )
    sql_name = table_sql._sanitize_identifier(t.table_id, fallback="t0")

    async def _fake(*, prompt: object, schema: type, **_kw: object) -> tuple[object, int]:
        return (
            GeneratedSQL(
                sql=f"SELECT SUM(revenue__num) FROM \"{sql_name}\" WHERE segment = 'graphics'",
                target_table_id=t.table_id,
            ),
            10,
        )

    # Sanity: confirm sqlite and the real recompute genuinely disagree here
    # (so the test is meaningful, not vacuously passing on some other path).
    import sqlite3

    db = sqlite3.connect(":memory:")
    try:
        loaded = table_sql._load_tables(db, table_sql._compute_schemas([t]))[t.table_id]
        sqlite_rows = table_sql._execute_select(
            db, f"SELECT SUM(revenue__num) FROM \"{sql_name}\" WHERE segment = 'graphics'"
        )
        assert sqlite_rows is not None
        raw = sqlite_rows[0][0]
        assert isinstance(raw, (int, float))
        sqlite_val = float(raw)
        recompute = table_sql._recompute_aggregate(
            t, loaded, "SUM", "revenue__num", "segment = 'graphics'"
        )
    finally:
        db.close()
    assert recompute is not None
    assert sqlite_val == pytest.approx(100)  # NOCASE matches 'Graphics' only (padded row excluded)
    assert recompute == pytest.approx(1099)  # strip+lower → both 'Graphics' and ' graphics '
    assert abs(sqlite_val - recompute) > max(1.0, 1e-6 * abs(recompute))

    # End-to-end: the live gate must refuse injection (→ None).
    import pytest as _pytest

    monkeypatch = _pytest.MonkeyPatch()
    try:
        monkeypatch.setattr("memex.agents.table_sql.complete_structured", _fake)
        result = await query_doc_tables("total graphics revenue?", [t])
    finally:
        monkeypatch.undo()
    assert result is None


@pytest.mark.asyncio
async def test_aggregate_recompute_disagree_via_stub_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boundary check: an injected disagreeing recompute (simulating a
    wrong-column mapping the recompute catches) → the gate refuses → None.

    Complements the real-disagreement test above — that one exercises the
    actual numeric path; this one pins the gate's contract independently of
    how the disagreement arises."""
    t = _segments_table()
    sql_name = table_sql._sanitize_identifier(t.table_id, fallback="t0")
    _patch_sql(monkeypatch, f'SELECT SUM(revenue__num) FROM "{sql_name}"', t.table_id)

    def _wrong(*_a: object, **_k: object) -> float:
        return 999999.0

    monkeypatch.setattr("memex.agents.table_sql._recompute_aggregate", _wrong)
    result = await query_doc_tables("total revenue?", [t])
    assert result is None


@pytest.mark.asyncio
async def test_aggregate_non_finite_value_refused_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIX 6: a degenerate huge-digit cell coerces to `inf`; the SUM is `inf` on
    BOTH the sqlite side and the independent recompute. `abs(inf - inf) = nan`,
    and `nan > tol` is False → the pre-fix gate would PASS and ship
    `Aggregate result = inf`. The finite-guard requires both sides finite, so
    the gate now refuses → None (no injection)."""
    big = "9" * 400  # float("9"*400) == inf
    assert float(big) == float("inf")
    t = StoredTable(
        doc_id="doc-inf",
        table_id="finf000000",
        section="Degenerate",
        header=["Name", "Value"],
        rows=[["a", big], ["b", "1"]],
        char_start=0,
        char_end=50,
    )
    sql_name = table_sql._sanitize_identifier(t.table_id, fallback="t0")
    _patch_sql(monkeypatch, f'SELECT SUM(value__num) FROM "{sql_name}"', t.table_id)

    # Sanity: both sides really are inf (so the old nan-compare would have
    # passed) — confirm the recompute itself returns inf.
    import sqlite3 as _sqlite3

    db = _sqlite3.connect(":memory:")
    try:
        loaded = table_sql._load_tables(db, table_sql._compute_schemas([t]))[t.table_id]
        rc = table_sql._recompute_aggregate(t, loaded, "SUM", "value__num", None)
    finally:
        db.close()
    assert rc == float("inf")

    assert await query_doc_tables("total value?", [t]) is None


@pytest.mark.asyncio
async def test_count_distinct_refused_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """`COUNT(DISTINCT col)` is not recomputable (a DISTINCT off-by-one could
    slip the ±1 tolerance against a plain count) → `_recompute_aggregate`
    refuses → no injection → None."""
    t = _segments_table()
    sql_name = table_sql._sanitize_identifier(t.table_id, fallback="t0")
    _patch_sql(monkeypatch, f'SELECT COUNT(DISTINCT segment) FROM "{sql_name}"', t.table_id)
    assert await query_doc_tables("how many distinct segments?", [t]) is None


def test_recompute_aggregate_refuses_distinct_unit() -> None:
    """Unit pin: `_recompute_aggregate` returns None whenever the aggregate
    arg contains DISTINCT (case-insensitive), regardless of op."""
    import sqlite3

    t = _segments_table()
    db = sqlite3.connect(":memory:")
    try:
        loaded = table_sql._load_tables(db, table_sql._compute_schemas([t]))[t.table_id]
        assert table_sql._recompute_aggregate(t, loaded, "count", "DISTINCT segment", None) is None
        assert table_sql._recompute_aggregate(t, loaded, "count", "distinct segment", None) is None
        assert (
            table_sql._recompute_aggregate(t, loaded, "sum", "DISTINCT revenue__num", None) is None
        )
        # A non-distinct count still computes.
        assert table_sql._recompute_aggregate(t, loaded, "count", "*", None) == pytest.approx(3)
    finally:
        db.close()


# ======================================================================
# unsupported → None
# ======================================================================


@pytest.mark.asyncio
async def test_group_by_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    t = _segments_table()
    sql_name = table_sql._sanitize_identifier(t.table_id, fallback="t0")
    _patch_sql(
        monkeypatch,
        f'SELECT segment, SUM(revenue__num) FROM "{sql_name}" GROUP BY segment',
        t.table_id,
    )
    assert await query_doc_tables("revenue by segment?", [t]) is None


@pytest.mark.asyncio
async def test_computed_scalar_select_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-aggregate SELECT with a computed/aliased scalar is not verbatim
    rows → None (it produces a new number)."""
    t = _segments_table()
    sql_name = table_sql._sanitize_identifier(t.table_id, fallback="t0")
    _patch_sql(
        monkeypatch,
        f'SELECT revenue__num * 2 AS doubled FROM "{sql_name}"',
        t.table_id,
    )
    assert await query_doc_tables("double the revenue?", [t]) is None


# ======================================================================
# FIX 2 — rows path restricted to SELECT * (non-* projection → None)
# ======================================================================


@pytest.mark.asyncio
async def test_rows_column_list_projection_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reordered/subset column-list projection (`SELECT revenue, segment ...`)
    would be rendered positionally against the FULL stored header → misleading
    header↔value attribution. The rows path now accepts ONLY `SELECT *`, so a
    column list → None (no misleading injection)."""
    t = _segments_table()
    sql_name = table_sql._sanitize_identifier(t.table_id, fallback="t0")
    # Reordered projection: revenue first, then segment.
    _patch_sql(monkeypatch, f'SELECT revenue, segment FROM "{sql_name}"', t.table_id)
    assert await query_doc_tables("show revenue and segment", [t]) is None


@pytest.mark.asyncio
async def test_rows_select_star_still_returns_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """`SELECT *` → rows as before (the projection aligns 1:1 with the header)."""
    t = _segments_table()
    sql_name = table_sql._sanitize_identifier(t.table_id, fallback="t0")
    _patch_sql(monkeypatch, f'SELECT * FROM "{sql_name}"', t.table_id)
    result = await query_doc_tables("show all rows", [t])
    assert result is not None
    assert result.kind == "rows"
    assert result.rows is not None
    assert result.rows[0] == ["Compute & Networking", "$116,193"]


def test_classify_is_select_star_unit() -> None:
    """Unit pin for FIX 2: only `SELECT *` passes; column lists / subsets fail."""
    assert table_sql._classify_is_select_star("SELECT * FROM t")
    assert table_sql._classify_is_select_star("select  *  from t WHERE a = 1")
    assert not table_sql._classify_is_select_star("SELECT a, b FROM t")
    assert not table_sql._classify_is_select_star("SELECT a FROM t")
    assert not table_sql._classify_is_select_star('SELECT "rev", "co" FROM t')
    assert not table_sql._classify_is_select_star("not a select")


# ======================================================================
# read-only guard / injection
# ======================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE x",
        "PRAGMA table_info(x)",
        "ATTACH DATABASE 'evil.db' AS evil",
        "SELECT 1; DROP TABLE x",
        "INSERT INTO x VALUES (1)",
        "UPDATE x SET a = 1",
        "DELETE FROM x",
        "CREATE TABLE y (a)",
        "",
    ],
)
async def test_injection_and_non_select_rejected(monkeypatch: pytest.MonkeyPatch, sql: str) -> None:
    t = _segments_table()
    _patch_sql(monkeypatch, sql, t.table_id)
    assert await query_doc_tables("totals?", [t]) is None


@pytest.mark.asyncio
async def test_unknown_target_table_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    t = _segments_table()
    _patch_sql(monkeypatch, "SELECT * FROM t0", "nope9999")
    assert await query_doc_tables("totals?", [t]) is None


@pytest.mark.asyncio
async def test_empty_tables_returns_none() -> None:
    assert await query_doc_tables("totals?", []) is None


@pytest.mark.asyncio
async def test_exec_error_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A SELECT over a non-existent column → sqlite error → None (no crash)."""
    t = _segments_table()
    sql_name = table_sql._sanitize_identifier(t.table_id, fallback="t0")
    _patch_sql(monkeypatch, f'SELECT nonexistent FROM "{sql_name}"', t.table_id)
    assert await query_doc_tables("totals?", [t]) is None


@pytest.mark.asyncio
async def test_empty_result_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    t = _segments_table()
    sql_name = table_sql._sanitize_identifier(t.table_id, fallback="t0")
    _patch_sql(
        monkeypatch,
        f"SELECT * FROM \"{sql_name}\" WHERE segment = 'DoesNotExist'",
        t.table_id,
    )
    assert await query_doc_tables("rows for missing segment?", [t]) is None


# ======================================================================
# read-only guard unit
# ======================================================================


def test_is_read_only_select_unit() -> None:
    assert table_sql._is_read_only_select("SELECT * FROM t")
    assert table_sql._is_read_only_select("SELECT * FROM t;")  # trailing ; tolerated
    assert table_sql._is_read_only_select("WITH x AS (SELECT 1) SELECT * FROM x")
    assert not table_sql._is_read_only_select("SELECT 1; SELECT 2")
    assert not table_sql._is_read_only_select("DROP TABLE t")
    assert not table_sql._is_read_only_select("DELETE FROM t")
    assert not table_sql._is_read_only_select("")


# ======================================================================
# schema/prompt-identifier consistency — GPU-acceptance regression 2026-05-24
# ======================================================================

_SQL_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")


def test_compute_schemas_emits_valid_sql_identifiers_matching_the_db() -> None:
    """The sanitized names shown to the LLM in the prompt MUST be the same
    valid SQL identifiers `_load_tables` creates — else the generated SQL
    references columns that don't exist and every query errors (the bug GPU
    acceptance surfaced: prompt showed `Fees Earned or Paid in Cash ($)` while
    the db column was `fees_earned_or_paid_in_cash`)."""
    t = StoredTable(
        doc_id="d1",
        table_id="0d724996f7",  # leading digit → must get a `c_` prefix
        section="Director Compensation",
        header=["Name", "Fees Earned or Paid in Cash ($)", "Total ($)"],
        rows=[["Ochoa", "42,500", "321,309"], ["Coxe", "85,000", "363,809"]],
        char_start=0,
        char_end=10,
    )
    (schema,) = table_sql._compute_schemas([t])
    # Every identifier the prompt will show is SQL-safe.
    assert _SQL_IDENT.match(schema.sql_name), schema.sql_name
    assert schema.sql_name == "c_0d724996f7"  # leading-digit prefix
    for col in schema.columns:
        assert _SQL_IDENT.match(col), col
    # And the db built from the SAME schema is queryable by those names.
    db = sqlite3.connect(":memory:")
    try:
        table_sql._load_tables(db, [schema])
        # FROM <sql_name> + a numeric companion column resolve.
        num_col = schema.columns[schema.numeric_cols[0]]
        rows = db.execute(f"SELECT {num_col}, {num_col}__num FROM {schema.sql_name}").fetchall()
        assert len(rows) == 2
    finally:
        db.close()


# ======================================================================
# row-superlative recompute gate (Table-RAG Phase 2 follow-up 2026-05-24)
# ======================================================================


def _dircomp_table() -> StoredTable:
    return StoredTable(
        doc_id="d",
        table_id="dircomp01",  # lowercase-alnum → sql_name == table_id
        section="Director Compensation",
        header=["Name", "Total ($)"],
        rows=[["Ochoa", "321,309"], ["Coxe", "363,809"], ["Burgess", "342,559"]],
        char_start=0,
        char_end=50,
    )


@pytest.mark.asyncio
async def test_superlative_asc_limit1_verified_frames_lowest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ORDER BY <num> ASC LIMIT 1` over a table whose min the returned row holds
    → kind="rows" with superlative=(label, "lowest"), returning the min row."""
    t = _dircomp_table()

    async def _fake(*, prompt: object, schema: type, **_kw: object) -> tuple[object, int]:
        return (
            GeneratedSQL(
                sql='SELECT * FROM "dircomp01" ORDER BY total__num ASC LIMIT 1',
                target_table_id=t.table_id,
            ),
            10,
        )

    monkeypatch.setattr(table_sql, "complete_structured", _fake)
    res = await query_doc_tables("which director got the lowest total comp", [t])
    assert res is not None and res.kind == "rows"
    assert res.superlative == ("Total ($)", "lowest")
    assert res.rows == [["Ochoa", "321,309"]]


@pytest.mark.asyncio
async def test_superlative_desc_limit1_verified_frames_highest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = _dircomp_table()

    async def _fake(*, prompt: object, schema: type, **_kw: object) -> tuple[object, int]:
        return (
            GeneratedSQL(
                sql='SELECT * FROM "dircomp01" ORDER BY total__num DESC LIMIT 1',
                target_table_id=t.table_id,
            ),
            10,
        )

    monkeypatch.setattr(table_sql, "complete_structured", _fake)
    res = await query_doc_tables("highest paid director", [t])
    assert res is not None and res.superlative == ("Total ($)", "highest")
    assert res.rows == [["Coxe", "363,809"]]


@pytest.mark.asyncio
async def test_superlative_limit_gt_1_not_framed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A multi-row `LIMIT 3` is a verbatim list, not a single-row superlative →
    no extremum framing (superlative stays None)."""
    t = _dircomp_table()

    async def _fake(*, prompt: object, schema: type, **_kw: object) -> tuple[object, int]:
        return (
            GeneratedSQL(
                sql='SELECT * FROM "dircomp01" ORDER BY total__num ASC LIMIT 3',
                target_table_id=t.table_id,
            ),
            10,
        )

    monkeypatch.setattr(table_sql, "complete_structured", _fake)
    res = await query_doc_tables("list directors by pay", [t])
    assert res is not None and res.kind == "rows"
    assert res.superlative is None


def test_verify_superlative_returned_row_not_extremum_returns_none() -> None:
    """Direct unit: a returned row that is NOT the column extremum → None (the
    safety check — a wrong ORDER BY can't self-certify a superlative claim)."""
    t = _dircomp_table()
    loaded = table_sql._load_tables(
        __import__("sqlite3").connect(":memory:"), table_sql._compute_schemas([t])
    )[t.table_id]
    # ASC asks for the minimum; hand it the MAX row → must not verify.
    assert (
        table_sql._verify_superlative(t, loaded, "total__num", "asc", None, ["Coxe", "363,809"])
        is None
    )
    # The real minimum row DOES verify.
    assert table_sql._verify_superlative(
        t, loaded, "total__num", "asc", None, ["Ochoa", "321,309"]
    ) == ("Total ($)", "lowest")


def test_verify_superlative_non_numeric_order_col_returns_none() -> None:
    """Ordering by a text column is not framed (lexicographic extremum is risky)."""
    t = _dircomp_table()
    loaded = table_sql._load_tables(
        __import__("sqlite3").connect(":memory:"), table_sql._compute_schemas([t])
    )[t.table_id]
    assert (
        table_sql._verify_superlative(t, loaded, "name", "asc", None, ["Burgess", "342,559"])
        is None
    )


def test_describe_aggregate_labels_by_header() -> None:
    """`describe_aggregate` maps the SQL's aggregated column back to its original
    header so the synthetic chunk self-describes the scalar (the fix that makes
    the answer node recognize the aggregate AS the queried quantity, ar-14)."""
    from memex.agents.table_sql import describe_aggregate
    from memex.core.types import TableQueryResult

    def _agg(sql: str, header: list[str], kind: str = "aggregate") -> TableQueryResult:
        return TableQueryResult(
            kind=kind, aggregate_value=956250.0, sql=sql, header=header,
            contributing_rows=[["x", "85,000"]], doc_id="d", target_table_id="t",
            section="Director Compensation", heading_path=["Director Compensation"],
            char_start=0, char_end=10, document_title="10-K",
        )

    r = _agg(
        "SELECT SUM(fees_earned_or_paid_in_cash__num) FROM c_x",
        ["**Name", "**Fees Earned or Paid in Cash ($)", "**Total ($)"],
    )
    assert describe_aggregate(r) == "SUM of Fees Earned or Paid in Cash ($)"
    # COUNT(*) → a sensible label, not a column.
    assert describe_aggregate(_agg("SELECT COUNT(*) FROM c_x", ["**Name", "**Fee"])).startswith("COUNT")
    # A rows-kind result has no aggregate label.
    assert describe_aggregate(_agg("SELECT * FROM c_x", ["A"], kind="rows")) is None
    # An unmappable column → None (caller falls back to the generic framing).
    assert describe_aggregate(_agg("SELECT SUM(unknown_col__num) FROM c_x", ["**Name"])) is None


# ======================================================================
# WHERE-form robustness + safety matrix (2026-05-31)
# Pinned from the adversarial table-sql-robustness workflow. The independent
# Python WHERE oracle (`_parse_where_predicate`) was widened to all the prompt's
# forms; every ADVERSARY / complex-aggregate row MUST stay REFUSED so the no-
# fabrication HARD gate is held by construction, not by the LLM behaving.
# ======================================================================

_MATRIX_HEADER = ["Region", "Cost"]
# Cost: US=10+20=30, EU=30, JP=40 → total 100
_MATRIX_ROWS = [["US", "10"], ["US", "20"], ["EU", "30"], ["JP", "40"]]


async def _probe_one(sql: str, header: list[str], rows: list[list[str]]) -> object | None:
    """Run one LLM-emitted SQL through the live gate; return the TableQueryResult
    (or None = refuse). Deterministic — the SQL is monkeypatched in."""
    st = StoredTable(
        doc_id="d", table_id="t1", section="S", header=header, rows=rows, char_start=0, char_end=1
    )
    name = table_sql._compute_schemas([st])[0].sql_name

    async def _fake(*, prompt: object, schema: type, **_kw: object) -> tuple[object, int]:
        return (GeneratedSQL(sql=sql.replace("__T__", name), target_table_id=name), 1)

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr("memex.agents.table_sql.complete_structured", _fake)
        # The question names the matrix's aggregate column ("cost") — the column-
        # grounding gate (audit-15 ar-16) requires the real flows' invariant that the
        # question asks about the aggregated column; this matrix tests the WHERE oracle.
        return await query_doc_tables("what is the total cost (val) for the matching regions?", [st])
    finally:
        mp.undo()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT SUM(cost__num) FROM __T__ WHERE region = 'US'", 30.0),
        ("SELECT SUM(cost__num) FROM __T__ WHERE region != 'US'", 70.0),
        ("SELECT SUM(cost__num) FROM __T__ WHERE cost__num > 10", 90.0),
        ("SELECT SUM(cost__num) FROM __T__ WHERE region IN ('US','EU')", 60.0),
        ("SELECT SUM(cost__num) FROM __T__ WHERE region NOT IN ('US')", 70.0),
        ("SELECT SUM(cost__num) FROM __T__ WHERE cost__num BETWEEN 20 AND 40", 90.0),
        ("SELECT SUM(cost__num) FROM __T__ WHERE region LIKE '%U%'", 60.0),  # US, EU
        ("SELECT SUM(cost__num) FROM __T__ WHERE region IS NOT NULL", 100.0),
        ("SELECT SUM(cost__num) FROM __T__ WHERE region = 'US' AND cost__num > 10", 20.0),
        ("SELECT SUM(cost__num) FROM __T__ WHERE region = 'US' OR cost__num = 40", 70.0),
        # NOCASE value-linking: a case-normalized literal still matches.
        ("SELECT SUM(cost__num) FROM __T__ WHERE region = 'us'", 30.0),
    ],
)
async def test_where_forms_ship_correct_value(sql: str, expected: float) -> None:
    """Every WHERE form the prompt permits — incl. NOT IN / BETWEEN / LIKE /
    IS NOT NULL / AND / OR and a case-normalized literal — verifies and ships its
    CORRECT aggregate via the independent Python oracle."""
    result = await _probe_one(sql, _MATRIX_HEADER, _MATRIX_ROWS)
    assert result is not None, f"false-refuse: {sql}"
    assert result.kind == "aggregate"
    assert result.aggregate_value == pytest.approx(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sql",
    [
        # ADVERSARY — over-the-wrong-rows / non-reproducible / expression WHEREs.
        "SELECT SUM(cost__num) FROM __T__ WHERE rowid <= 2",
        "SELECT SUM(cost__num) FROM __T__ WHERE (abs(random()) % 2) = 0",
        "SELECT SUM(cost__num) FROM __T__ WHERE cost__num > (SELECT AVG(cost__num) FROM __T__)",
        "SELECT SUM(cost__num) FROM __T__ WHERE cost__num % 20 = 0",
        "SELECT SUM(amt__num) FROM __T__ WHERE cat GLOB 'x*'",
        # Complex / unverifiable aggregates — conservative refuse.
        "SELECT COUNT(DISTINCT region) FROM __T__",
        "SELECT SUM(DISTINCT cost__num) FROM __T__",
        "SELECT SUM(cost__num), AVG(cost__num) FROM __T__",
        "SELECT SUM(cost__num)-AVG(cost__num) FROM __T__",
        "SELECT region, SUM(cost__num) FROM __T__ GROUP BY region",
        "SELECT CAST(SUM(cost__num) AS INTEGER) FROM __T__",
        "SELECT TOTAL(cost__num) FROM __T__",
        # A1 SAFETY — aggregate over a raw TEXT column (coercion-misread surface).
        "SELECT SUM(cost) FROM __T__",
        "SELECT AVG(region) FROM __T__",
    ],
)
async def test_dangerous_or_complex_forms_refuse(sql: str) -> None:
    """The no-fabrication HARD gate: every adversary WHERE (rowid / random /
    subquery / expression / GLOB), every unverifiable aggregate (DISTINCT /
    multi-scalar / arithmetic / GROUP BY / CAST / TOTAL), and every raw-text-
    column aggregate (A1) MUST refuse — independent of LLM behaviour."""
    result = await _probe_one(sql, _MATRIX_HEADER, _MATRIX_ROWS)
    assert result is None, f"MUST refuse but shipped: {sql}"


@pytest.mark.asyncio
async def test_a2_text_column_superlative_not_framed() -> None:
    """A2 SAFETY: ORDER BY a raw TEXT column sorts LEXICALLY ('9' > '10'); the
    extremum tolerance would otherwise frame the WRONG row as 'highest'. The gate
    withholds the superlative framing (requires the `__num` companion)."""
    result = await _probe_one(
        "SELECT * FROM __T__ ORDER BY gpus DESC LIMIT 1",
        ["Model", "GPUs"],
        [["A", "2"], ["B", "10"], ["C", "9"]],
    )
    # Ships rows (verbatim, safe) but with NO verified-extremum framing.
    assert result is not None
    assert result.kind == "rows"
    assert result.superlative is None


@pytest.mark.asyncio
async def test_a2_numeric_companion_superlative_is_framed() -> None:
    """The correct `ORDER BY gpus__num` sorts numerically → the extremum (B=10)
    is the real max → the superlative framing IS applied."""
    result = await _probe_one(
        "SELECT * FROM __T__ ORDER BY gpus__num DESC LIMIT 1",
        ["Model", "GPUs"],
        [["A", "2"], ["B", "10"], ["C", "9"]],
    )
    assert result is not None
    assert result.kind == "rows"
    assert result.superlative == ("GPUs", "highest")
    assert result.rows is not None and result.rows[0][0] == "B"


# ======================================================================
# Coercion-soundness guard (2026-05-31)
# `coerce_number` is lenient (strips ALL commas → '1,2,3'=123) and the sqlite
# __num column shares it, so the recompute-agreement check is blind to a misread.
# The aggregate/superlative gate refuses a contributing cell that coerces but is
# NOT canonical. Closes the malformed / mixed-separator wrong-value ship.
# ======================================================================


@pytest.mark.parametrize(
    ("cell", "canonical"),
    [
        # Canonical US-convention numbers — MUST pass (no false-refuse).
        ("100", True),
        ("1,234", True),
        ("1,000,000", True),
        ("$1,234.56", True),
        ("$22.5 billion", True),
        ("30M", True),
        ("2.5B", True),
        ("€500", True),
        ("(1,234)", True),
        ("71.1%", True),
        ("-50", True),
        ("$-5", True),
        ("342,559", True),
        # Malformed / mixed-separator — coerces leniently but NON-canonical.
        ("1,2,3", False),  # commas not in thousands positions → coerce='123' (wrong)
        ("12,34", False),
        ("1,23", False),
        ("1.234,56", False),  # European mixed separators
    ],
)
def test_is_canonical_number_cell(cell: str, canonical: bool) -> None:
    from memex.core.text import is_canonical_number_cell

    assert table_sql.is_canonical_number_cell(cell) is canonical
    # Sanity: every NON-canonical case here still coerces (the lenient misread the
    # guard exists to catch); every canonical case coerces too.
    assert is_canonical_number_cell(cell) is canonical


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rows", "sql", "expected"),
    [
        # Legit columns ship the correct aggregate.
        ([["a", "10"], ["b", "20"], ["c", "30"]], "SELECT SUM(val__num) FROM __T__", 60.0),
        ([["a", "1,234"], ["b", "766"]], "SELECT SUM(val__num) FROM __T__", 2000.0),
        ([["a", "$22.5 billion"]], "SELECT SUM(val__num) FROM __T__", 2.25e10),
        # A contributing cell that COERCES but is NON-canonical → refuse (None).
        ([["a", "10"], ["b", "1,2,3"]], "SELECT SUM(val__num) FROM __T__", None),
        ([["a", "10"], ["b", "1.234,56"]], "SELECT SUM(val__num) FROM __T__", None),
        ([["a", "10"], ["b", "12,34"]], "SELECT MAX(val__num) FROM __T__", None),
    ],
)
async def test_aggregate_coercion_soundness(
    rows: list[list[str]], sql: str, expected: float | None
) -> None:
    """A malformed / mixed-separator contributing cell makes the aggregate refuse
    (the recompute-agreement check is blind to the shared-parser misread); legit
    US-convention columns still ship the correct value."""
    st = StoredTable(
        doc_id="d", table_id="t1", section="S", header=["Item", "Val"], rows=rows, char_start=0, char_end=1
    )
    name = table_sql._compute_schemas([st])[0].sql_name

    async def _fake(*, prompt: object, schema: type, **_kw: object) -> tuple[object, int]:
        return (GeneratedSQL(sql=sql.replace("__T__", name), target_table_id=name), 1)

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr("memex.agents.table_sql.complete_structured", _fake)
        result = await query_doc_tables("what is the total val of the items?", [st])
    finally:
        mp.undo()
    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert result.aggregate_value == pytest.approx(expected)


# ---- aggregate COLUMN-GROUNDING gate (audit-15, the ar-16 fabrication) ----


from memex.agents.table_sql import (  # noqa: E402 — section-local import, file convention
    _aggregate_column_grounded_in_question as _colg,
)


def test_column_grounding_blocks_the_ar16_substitution() -> None:
    q = "According to the fiscal 2026 director compensation table, what was the total value of stock options granted to NVIDIA's directors?"
    assert _colg(q, "Stock Awards ($) (1)") is False  # 'awards' not in a question about options


def test_column_grounding_passes_the_ar14_exact_match() -> None:
    q = "What was the total amount of fees earned or paid in cash to all of NVIDIA's directors?"
    assert _colg(q, "**Fees Earned or Paid in Cash ($)") is True


def test_column_grounding_plural_fold_and_stopwords() -> None:
    assert _colg("How many training GPUs were used in total across variants?", "Training GPU") is True
    assert _colg("What is the total number of employees?", "Employees") is True
    assert _colg("What is the sum of revenue?", "Total Revenue ($)") is True  # 'total' is a stop


def test_column_grounding_empty_label_is_permissive() -> None:
    assert _colg("anything", "($)") is True  # no content tokens -> no substitutable claim
