"""Text-to-SQL over a single document's structured tables (Table-RAG Phase 2).

`query_doc_tables(question, tables)` loads a document's `StoredTable`s into a
fresh in-memory sqlite db (one SQL table per stored table, with a `<col>__num`
REAL column for every numeric-coercible column), asks the orchestrator for a
single read-only SELECT via `complete_structured`, executes it under a row +
step cap, and classifies the result:

  - `kind="rows"`  — no aggregate function / no computed scalar: the result rows
    are VERBATIM document cells → returned as-is (no new number, safe to ship).
  - `kind="aggregate"` — a single-table single-column SUM/COUNT/AVG/MIN/MAX with
    an optional simple WHERE: the value is recomputed INDEPENDENTLY in Python
    from the original cell text (`coerce_number`, NOT the sqlite `__num` column)
    and returned ONLY if it agrees with the SQL value; else `None`.
  - anything else (joins, group-by, HAVING, derived superlatives, non-
    recomputable scalars) → `None`.

Returning `None` is the no-op fallback the agent node treats as "no table
answer" — the query then proceeds on the normal reranked set, which refuses if
it can't ground. This holds the HARD gate by construction (see
`docs/specs/table-sql.md` §4): every shipped number is either a verbatim cell
or a recomputed-and-agreeing aggregate.

`complete_structured` is imported into THIS module so the test patch seam is
`memex.agents.table_sql.complete_structured`.
"""

from __future__ import annotations

import asyncio
import math
import re
import sqlite3
from collections.abc import Callable

import structlog

from memex.core.text import coerce_number, is_canonical_number_cell
from memex.core.types import GeneratedSQL, StoredTable, TableQueryResult
from memex.models.client import complete_structured
from memex.prompts import prompt_tag_for, render_prompt

logger = structlog.get_logger(__name__)

# Execution caps — module constants (MEMEX_* env only if a cap ever needs
# tuning, per the spec; none does today).
_ROW_CAP = 1000  # max rows fetched from a result
# Rows shown to the SQL-gen LLM per table for VALUE-LINKING — enough that a small
# table's full value set is visible (so it copies WHERE literals exactly / sees a
# filter is needless), bounded so a wide table doesn't blow the prompt budget.
_SAMPLE_ROW_CAP = 8
_STEP_CAP = 100_000  # sqlite VM instruction budget (no wall-clock dependency)

# Statements forbidden in the read-only guard. The SQL must be a single
# SELECT — no `;`-chain, no DDL/DML/PRAGMA/ATTACH. Word-boundary matched so a
# column literally named "updated" doesn't trip the UPDATE check.
_FORBIDDEN_KEYWORDS = (
    "pragma",
    "attach",
    "detach",
    "insert",
    "update",
    "delete",
    "drop",
    "create",
    "alter",
    "replace",
    "vacuum",
    "reindex",
)

# Aggregate functions we can independently recompute.
_AGG_FUNCS = ("sum", "count", "avg", "min", "max")

# `coerce_number` moved to `core/text.py` 2026-05-31 (a pure core utility shared
# with the index-time table machinery) and is re-exported above for the existing
# `from memex.agents.table_sql import coerce_number` callers.


_SANITIZE_RE = re.compile(r"[^0-9a-zA-Z]+")


def _sanitize_identifier(name: str, *, fallback: str) -> str:
    """Map an arbitrary header/table label to a safe SQL identifier.

    Non-alphanumerics collapse to `_`; a leading digit gets a `c_` prefix; an
    empty result uses *fallback*. The caller dedupes collisions.
    """
    cleaned = _SANITIZE_RE.sub("_", name).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"c_{cleaned}"
    return cleaned.lower()


class _LoadedTable:
    """One stored table loaded into the in-memory sqlite db.

    Holds the sanitized SQL table name and the col-index → (sql_col,
    original_header) mapping plus which columns got a `<col>__num` companion.
    """

    def __init__(
        self,
        stored: StoredTable,
        sql_name: str,
        columns: list[str],
        numeric_cols: list[int],
    ) -> None:
        self.stored = stored
        self.sql_name = sql_name
        self.columns = columns  # sanitized text-column names, by header index
        self.numeric_cols = numeric_cols  # header indices with a __num companion


def _compute_schemas(tables: list[StoredTable]) -> list[_LoadedTable]:
    """Per table, derive its sanitized SQL identifiers WITHOUT touching a db.

    This is the single source of truth for the sql table/column names: both
    `_load_tables` (which CREATEs/INSERTs) and the `generate_table_sql` prompt
    (which shows the LLM the names to use) consume the same `_LoadedTable`
    schemas, so the SQL the model emits references identifiers that actually
    exist. (Pre-fix the prompt showed ORIGINAL names while the db used
    sanitized ones → every generated SQL errored. GPU-acceptance fix
    2026-05-24.)
    """
    schemas: list[_LoadedTable] = []
    used_table_names: set[str] = set()
    for ti, stored in enumerate(tables):
        # A header-less table would build `CREATE TABLE "t" ()` (sqlite
        # OperationalError). Not reachable via the well-formed parse pipeline,
        # but skip it defensively so a degenerate table never crashes the load.
        if not stored.header:
            continue
        sql_name = _sanitize_identifier(stored.table_id, fallback=f"t{ti}")
        while sql_name in used_table_names:
            sql_name = f"{sql_name}_x"
        used_table_names.add(sql_name)

        # Sanitize + dedupe column names.
        columns: list[str] = []
        seen_cols: set[str] = set()
        for ci, raw in enumerate(stored.header):
            col = _sanitize_identifier(raw, fallback=f"col{ci + 1}")
            while col in seen_cols or col.endswith("__num"):
                col = f"{col}_c"
            seen_cols.add(col)
            columns.append(col)

        # Which columns are numeric-coercible across their data cells?
        numeric_cols: list[int] = []
        for ci in range(len(columns)):
            vals = [
                coerce_number(row[ci]) for row in stored.rows if ci < len(row) and row[ci].strip()
            ]
            if vals and all(v is not None for v in vals):
                numeric_cols.append(ci)
        schemas.append(_LoadedTable(stored, sql_name, columns, numeric_cols))
    return schemas


def _load_tables(db: sqlite3.Connection, schemas: list[_LoadedTable]) -> dict[str, _LoadedTable]:
    """Create one sqlite table per precomputed schema and return them keyed by
    table_id. Each header column becomes a TEXT column (original cell text);
    a numeric-coercible column additionally gets a `<col>__num REAL` column.
    """
    loaded: dict[str, _LoadedTable] = {}
    for schema in schemas:
        stored = schema.stored
        sql_name = schema.sql_name
        columns = schema.columns
        numeric_cols = schema.numeric_cols

        # COLLATE NOCASE aligns sqlite's text `=`/`IN`/`LIKE` with the
        # independent recompute's case-insensitive matching (`_row_matches`
        # lower-cases both sides). Without it, a case-normalized literal the LLM
        # routinely writes (`'gte small'` vs stored `'GTE Small'`) matches 0 rows
        # in sqlite while the recompute matches → the two arms DISAGREE and the
        # gate false-refuses (the dominant value-linking recall bug). NOCASE makes
        # both arms select the SAME rows → agree → ship the correct value.
        col_defs = [f'"{c}" TEXT COLLATE NOCASE' for c in columns]
        col_defs += [f'"{columns[ci]}__num" REAL' for ci in numeric_cols]
        db.execute(f'CREATE TABLE "{sql_name}" ({", ".join(col_defs)})')

        insert_cols = [f'"{c}"' for c in columns]
        insert_cols += [f'"{columns[ci]}__num"' for ci in numeric_cols]
        placeholders = ", ".join("?" for _ in insert_cols)
        rows_to_insert: list[list[object]] = []
        for row in stored.rows:
            text_vals: list[object] = [
                row[ci] if ci < len(row) else "" for ci in range(len(columns))
            ]
            num_vals: list[object] = [
                coerce_number(row[ci]) if ci < len(row) else None for ci in numeric_cols
            ]
            rows_to_insert.append(text_vals + num_vals)
        db.executemany(
            f'INSERT INTO "{sql_name}" ({", ".join(insert_cols)}) VALUES ({placeholders})',
            rows_to_insert,
        )

        loaded[stored.table_id] = schema
    return loaded


def _is_read_only_select(sql: str) -> bool:
    """True iff *sql* is a single read-only SELECT — the safety guard.

    Rejects `;`-chains (a trailing single `;` is tolerated), any forbidden
    DDL/DML/PRAGMA/ATTACH keyword (word-boundary matched), and anything that
    doesn't start with SELECT or WITH.
    """
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return False
    if ";" in stripped:
        return False
    head = stripped.lower()
    if not (head.startswith("select") or head.startswith("with")):
        return False
    for kw in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", head):
            return False
    return True


# A simple aggregate SELECT: `SELECT FN(col) FROM tbl [WHERE ...]`. We capture
# the function, the column, the table, and the optional WHERE body. The pattern
# anchors at `\s*$` right after the optional WHERE, so a TRAILING `ORDER BY` /
# `LIMIT` on a bare aggregate makes this fail to match → the query is not
# classified as a recomputable aggregate → unsupported → no-op (refuse). Only a
# plain scalar aggregate (with at most a single simple WHERE) is gated.
_AGG_RE = re.compile(
    r"^\s*select\s+(?P<fn>\w+)\s*\(\s*(?P<col>(?:distinct\s+)?[\w\".]+|\*)\s*\)\s*"
    r"(?:as\s+\w+\s*)?"
    r"from\s+(?P<tbl>[\w\"]+)\s*"
    r"(?:where\s+(?P<where>.+?))?\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)

# A single simple WHERE predicate: `col OP literal` (equality or numeric
# comparison on ONE column). The literal may be quoted text or a number.
_WHERE_RE = re.compile(
    r"^\s*(?P<col>[\w\".]+)\s*(?P<op>=|==|!=|<>|<=|>=|<|>)\s*(?P<lit>.+?)\s*$",
    flags=re.IGNORECASE,
)

# A `col IN (lit, lit, ...)` membership predicate — the OTHER simple WHERE the
# prompt permits. Literals may be quoted text or numbers.
_IN_RE = re.compile(
    r"^\s*(?P<col>[\w\".]+)\s+in\s*\(\s*(?P<lits>.+?)\s*\)\s*$",
    flags=re.IGNORECASE,
)
_NOTIN_RE = re.compile(
    r"^\s*(?P<col>[\w\".]+)\s+not\s+in\s*\(\s*(?P<lits>.+?)\s*\)\s*$",
    flags=re.IGNORECASE,
)
_BETWEEN_RE = re.compile(
    r"^\s*(?P<col>[\w\".]+)\s+between\s+(?P<lo>.+?)\s+and\s+(?P<hi>.+?)\s*$",
    flags=re.IGNORECASE,
)
_ISNULL_RE = re.compile(
    r"^\s*(?P<col>[\w\".]+)\s+is\s+(?P<negate>not\s+)?null\s*$",
    flags=re.IGNORECASE,
)
_LIKE_RE = re.compile(
    r"^\s*(?P<col>[\w\".]+)\s+like\s+(?P<pat>.+?)\s*$",
    flags=re.IGNORECASE,
)
# G3 SAFETY: a WHERE the independent oracle must NEVER trust — non-deterministic
# (random), positional (rowid), raw pattern (glob), or a correlated subquery.
# These don't match any safe atom below either, but reject them EXPLICITLY (the
# safety review's over-the-wrong-rows / non-reproducible-scalar attacks).
_DANGEROUS_WHERE_RE = re.compile(r"\b(?:random|rowid|glob)\b|\(\s*select\b", flags=re.IGNORECASE)


def _strip_quotes(ident: str) -> str:
    return ident.strip().strip('"').strip("'")


def describe_aggregate(result: TableQueryResult) -> str | None:
    """A human-readable label for an aggregate result — e.g. `SUM of Fees Earned
    or Paid in Cash ($)` — by parsing the SQL's function + column and mapping the
    sanitized column back to its ORIGINAL header. Used to frame the synthetic
    chunk so the answer node recognizes the scalar AS the queried quantity (a
    bare "Aggregate result = 956250" reads as un-labelled and the literal-presence
    rule refuses it). None when the SQL isn't a recognizable scalar aggregate or
    the column can't be mapped → the caller falls back to the generic framing."""
    if result.kind != "aggregate":
        return None
    m = _AGG_RE.match(result.sql)
    if not m:
        return None
    fn = m.group("fn").upper()
    col = _strip_quotes(m.group("col"))
    if col == "*":
        return f"{fn} over {result.header[0].strip('*').strip()}" if result.header else None
    # The numeric companion column is `<sanitized>__num`; the aggregate runs on it.
    base = col[:-5] if col.lower().endswith("__num") else col
    for h in result.header:
        if _sanitize_identifier(h, fallback="c") == base.lower():
            label = h.strip().strip("*").strip()
            return f"{fn} of {label}" if label else None
    return None


# A single-row superlative SELECT: `SELECT ... FROM tbl [WHERE ...] ORDER BY col
# [ASC|DESC] LIMIT 1`. Captures the order column, direction (default ASC), and
# the optional WHERE body. Scoped to `LIMIT 1` — the "which X has the most/least
# Y" case — so the verification is a single boundary check (a multi-row list is
# already a verbatim list the agent reads; it gets no superlative framing).
_SUPERLATIVE_RE = re.compile(
    r"^\s*select\s+.+?\s+from\s+(?P<tbl>[\w\"]+)\s*"
    r"(?:where\s+(?P<where>.+?)\s+)?"
    r"order\s+by\s+(?P<col>[\w\".]+)\s*(?P<dir>asc|desc)?\s*"
    r"limit\s+1\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)


def _verify_superlative(
    stored: StoredTable,
    loaded: _LoadedTable,
    order_col: str,
    direction: str,
    where: str | None,
    returned_cells: list[str],
) -> tuple[str, str] | None:
    """Independently confirm *returned_cells* holds the extremum of *order_col*
    (what a `LIMIT 1` superlative claims), or None when it can't be verified.

    Like `_recompute_aggregate`, the extremum is recomputed over the ORIGINAL
    cell text (`coerce_number`, not the sqlite `__num` column) so a wrong ORDER
    BY can't self-certify, and the returned row's order-column value is compared
    to that extremum (a VALUE check — tie-safe: a row tied at the extremum still
    verifies). Returns `(column_label, "highest"|"lowest")` for framing only
    when the order column is numeric-coercible and the returned row sits at the
    extremum. The returned-row check makes the framing grounded.
    """
    desc = direction.lower() == "desc"
    col_name = _strip_quotes(order_col).split(".")[-1].lower()
    # SAFETY: the ORDER BY must target the NUMERIC companion `<col>__num` so
    # sqlite sorts NUMERICALLY. Ordering a raw text column sorts LEXICALLY
    # (`'9' > '10'`), and the extremum tolerance below then swallows abs(9-10)=1
    # and frames the WRONG row as the extremum. The prompt already mandates
    # `<col>__num` for ORDER BY; require it. (Adversary-pinned in test_table_sql.)
    if not col_name.endswith("__num"):
        return None
    cidx = _column_index(loaded, col_name)
    if cidx is None or cidx not in loaded.numeric_cols:
        return None  # only numeric ordering is framed (text ordering is risky)

    match_row: Callable[[list[str]], bool] | None = None
    if where is not None:
        match_row = _parse_where_predicate(where, loaded)
        if match_row is None:
            return None

    vals: list[float] = []
    for row in stored.rows:
        if match_row is not None and not match_row(row):
            continue
        if cidx < len(row) and row[cidx].strip():
            n = coerce_number(row[cidx])
            if n is None:
                continue
            # Coercion-soundness (mirror the aggregate gate): a cell that coerces
            # but is NOT canonical is a lenient misread the __num-vs-coerce check
            # is blind to — don't frame an extremum over a mis-ordered column.
            if not is_canonical_number_cell(row[cidx]):
                return None
            vals.append(n)
    if not vals:
        return None
    if cidx >= len(returned_cells):
        return None
    returned = coerce_number(returned_cells[cidx])
    if returned is None:
        return None
    extremum = max(vals) if desc else min(vals)
    # A degenerate huge-digit cell coerces to inf; `abs(inf - inf)` is nan, which is
    # not `> tol`, so the gate would WRONGLY pass. Mirror `_classify_and_build`'s
    # non-finite guard and refuse to frame on a non-finite value.
    if not (math.isfinite(returned) and math.isfinite(extremum)):
        return None
    # The `±1.0` floor mirrors the aggregate gate's tolerance. It's dead slack
    # here, not load-bearing: `_load_tables` builds the sqlite `__num` column via
    # the SAME `coerce_number`, so sqlite's `ORDER BY <col>__num` and this
    # recompute operate on bit-identical floats and can't disagree on the
    # extremum — `returned` is the genuine extremum, compared against itself.
    if abs(returned - extremum) > max(1.0, 1e-6 * abs(extremum)):
        return None  # the returned row is NOT the extremum → don't frame
    label = stored.header[cidx] if cidx < len(stored.header) else col_name
    return (label, "highest" if desc else "lowest")


def _classify_is_select_star(sql: str) -> bool:
    """Gate for `kind="rows"`: the projection must be exactly `SELECT *`.

    The rows path renders the SQL output positionally against the FULL stored
    `header`, so ONLY a `SELECT *` projection keeps header↔value attribution
    honest. A column-list/subset/reordered projection (`SELECT rev, company
    FROM t`) would mis-align cells under the stored header and ship misleading
    evidence to the LLM — so it returns False here → the rows branch no-ops.
    (The `generate_table_sql` prompt already mandates `SELECT *` for row
    queries; the superlative `SELECT * ... ORDER BY ... LIMIT 1` also uses `*`,
    so it still passes.)
    """
    m = re.match(r"^\s*select\s+(?P<cols>.+?)\s+from\b", sql, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return False
    return m.group("cols").strip() == "*"


def _recompute_aggregate(
    stored: StoredTable,
    loaded: _LoadedTable,
    op: str,
    column_sql: str,
    where: str | None,
) -> float | None:
    """Recompute an aggregate independently from the ORIGINAL cell text.

    Re-parses `stored.rows` via `coerce_number` (NOT the sqlite `__num`
    column) so a wrong column-mapping or filter is caught even though the
    number grammar is shared. Returns the scalar, or None when the op/column/
    filter isn't recomputable here (→ caller refuses).
    """
    op = op.lower()
    # DISTINCT is not recomputable here (a `COUNT(DISTINCT col)` off-by-one
    # could otherwise slip through the ±1 tolerance against a plain count).
    # Refuse outright → the aggregate can't be gated → no injection → refuse.
    if re.search(r"\bdistinct\b", column_sql, flags=re.IGNORECASE):
        return None
    col_name = _strip_quotes(column_sql).split(".")[-1].lower()

    # SAFETY: SUM/AVG/MIN/MAX must target the NUMERIC companion `<col>__num`
    # (which the prompt mandates), never a raw text column. sqlite's lenient
    # prefix-coercion of a text column (`SUM('350000.75 deferred')` → 350000.75)
    # can diverge from `coerce_number` (→ 350000) by LESS than the agreement
    # tolerance floor and SHIP a fabricated value. COUNT is exempt (it counts
    # non-empty cells, no coercion). Forcing `__num` makes sqlite aggregate the
    # SAME parsed numbers `coerce_number` re-derives, so a text-column misread
    # can't masquerade as float noise. (Adversary-pinned in test_table_sql.)
    if op in ("sum", "avg", "min", "max") and not col_name.endswith("__num"):
        return None

    # Resolve the WHERE predicate to a row predicate over the original cells.
    match_row: Callable[[list[str]], bool] | None = None
    if where is not None:
        match_row = _parse_where_predicate(where, loaded)
        if match_row is None:
            return None

    # Select the matching rows (after the WHERE filter).
    selected: list[list[str]] = []
    for row in stored.rows:
        if match_row is not None and not match_row(row):
            continue
        selected.append(row)

    if op == "count":
        # COUNT(*) or COUNT(col): count rows (col-non-null for a named column).
        if col_name == "*":
            return float(len(selected))
        cidx = _column_index(loaded, col_name)
        if cidx is None:
            return None
        return float(sum(1 for r in selected if cidx < len(r) and r[cidx].strip()))

    cidx = _column_index(loaded, col_name)
    if cidx is None:
        return None
    # Coercion-soundness: SUM/AVG/MIN/MAX over a column with a cell that COERCES
    # but is NOT a canonical US-convention number (malformed grouping `1,2,3`→123,
    # mixed European separators) — `coerce_number` is lenient AND the sqlite
    # `__num` column shares it, so the recompute-agreement check is blind to the
    # misread. Refuse the aggregate (HARD-gate-safe conservative drop). See
    # `is_canonical_number_cell`; residual locale/unit ambiguity is documented.
    clean: list[float] = []
    for r in selected:
        if cidx >= len(r) or not r[cidx].strip():
            continue
        v = coerce_number(r[cidx])
        if v is None:
            continue  # a non-numeric cell (N/A / footnote) is skipped, not refused
        if not is_canonical_number_cell(r[cidx]):
            return None  # a lenient misread → refuse rather than ship a wrong value
        clean.append(v)
    if not clean:
        return None
    if op == "sum":
        return sum(clean)
    if op == "avg":
        return sum(clean) / len(clean)
    if op == "min":
        return min(clean)
    if op == "max":
        return max(clean)
    return None


def _column_index(loaded: _LoadedTable, sanitized_col: str) -> int | None:
    """Map a sanitized SQL column name back to its header index, or None.

    Tolerates the `__num` companion suffix (an aggregate over `revenue__num`
    maps to the `revenue` header column).
    """
    name = sanitized_col
    if name.endswith("__num"):
        name = name[: -len("__num")]
    for i, c in enumerate(loaded.columns):
        if c == name:
            return i
    return None


def _row_matches(row: list[str], where_filter: tuple[int, str, str]) -> bool:
    """Apply a single simple WHERE predicate to one row's original cell text."""
    idx, op, literal = where_filter
    if idx >= len(row):
        return False
    cell = row[idx]
    lit_num = coerce_number(literal)
    cell_num = coerce_number(cell)
    if op in ("=", "=="):
        if lit_num is not None and cell_num is not None:
            return cell_num == lit_num
        return cell.strip().lower() == literal.strip().lower()
    if op in ("!=", "<>"):
        if lit_num is not None and cell_num is not None:
            return cell_num != lit_num
        return cell.strip().lower() != literal.strip().lower()
    # Numeric comparisons require both sides to be numbers.
    if lit_num is None or cell_num is None:
        return False
    if op == "<":
        return cell_num < lit_num
    if op == "<=":
        return cell_num <= lit_num
    if op == ">":
        return cell_num > lit_num
    if op == ">=":
        return cell_num >= lit_num
    return False


def _split_sql_list(body: str) -> list[str]:
    """Split a SQL value list on TOP-LEVEL commas, respecting single/double
    quotes so a quoted literal containing a comma (`'1,234'`) stays intact."""
    out: list[str] = []
    cur: list[str] = []
    quote: str | None = None
    for ch in body:
        if quote is not None:
            cur.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            cur.append(ch)
        elif ch == ",":
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return [s for s in (p.strip() for p in out) if s]


def _split_top_level(expr: str, keyword: str) -> list[str]:
    """Split `expr` on a top-level (depth-0, unquoted) ` <keyword> ` (AND / OR),
    case-insensitive, respecting parens + single/double quotes. A keyword inside
    parens or a quoted literal is NOT a split point."""
    target = " " + keyword.lower() + " "
    low = expr.lower()
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    start = 0
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if quote is not None:
            if ch == quote:
                quote = None
            i += 1
        elif ch in ("'", '"'):
            quote = ch
            i += 1
        elif ch == "(":
            depth += 1
            i += 1
        elif ch == ")":
            depth -= 1
            i += 1
        elif depth == 0 and low.startswith(target, i):
            parts.append(expr[start:i])
            start = i + len(target)
            i = start
        else:
            i += 1
    parts.append(expr[start:])
    return [p.strip() for p in parts]


def _like_to_regex(pattern: str) -> re.Pattern[str]:
    """Compile a SQL LIKE pattern (`%` = any run, `_` = any char) to a case-
    insensitive anchored regex over the cell text (NOCASE-consistent)."""
    out = ["^"]
    for ch in pattern:
        if ch == "%":
            out.append(".*")
        elif ch == "_":
            out.append(".")
        else:
            out.append(re.escape(ch))
    out.append("$")
    return re.compile("".join(out), flags=re.IGNORECASE | re.DOTALL)


def _column_predicate(
    col_sql: str,
    loaded: _LoadedTable,
    build: Callable[[int], Callable[[list[str]], bool]],
) -> Callable[[list[str]], bool] | None:
    """Resolve a WHERE column to its header index and hand it to `build`, or None
    when the column isn't a real header column (rowid / unknown → refuse)."""
    widx = _column_index(loaded, _strip_quotes(col_sql).split(".")[-1].lower())
    if widx is None:
        return None
    return build(widx)


def _parse_atom(expr: str, loaded: _LoadedTable) -> Callable[[list[str]], bool] | None:
    """Parse ONE simple predicate (no AND/OR) into an independent row predicate
    over the original cell text, or None when unrecomputable here."""
    expr = expr.strip()
    while expr.startswith("(") and expr.endswith(")"):
        expr = expr[1:-1].strip()

    m = _ISNULL_RE.match(expr)
    if m is not None:
        not_null = m.group("negate") is not None  # `IS NOT NULL` → cell non-empty
        return _column_predicate(
            m.group("col"),
            loaded,
            lambda idx: (lambda row: (idx < len(row) and bool(row[idx].strip())) == not_null),
        )
    m = _NOTIN_RE.match(expr)
    if m is not None:
        lits = [_strip_quotes(x) for x in _split_sql_list(m.group("lits"))]
        if not lits:
            return None
        return _column_predicate(
            m.group("col"),
            loaded,
            lambda idx: (lambda row: not any(_row_matches(row, (idx, "=", lit)) for lit in lits)),
        )
    m = _IN_RE.match(expr)
    if m is not None:
        lits = [_strip_quotes(x) for x in _split_sql_list(m.group("lits"))]
        if not lits:
            return None
        return _column_predicate(
            m.group("col"),
            loaded,
            lambda idx: (lambda row: any(_row_matches(row, (idx, "=", lit)) for lit in lits)),
        )
    m = _LIKE_RE.match(expr)
    if m is not None:
        rx = _like_to_regex(_strip_quotes(m.group("pat")))
        return _column_predicate(
            m.group("col"),
            loaded,
            lambda idx: (lambda row: idx < len(row) and rx.match(row[idx].strip()) is not None),
        )
    m = _WHERE_RE.match(expr)
    if m is not None:
        op = m.group("op")
        lit = _strip_quotes(m.group("lit"))
        return _column_predicate(
            m.group("col"),
            loaded,
            lambda idx: (lambda row: _row_matches(row, (idx, op, lit))),
        )
    return None


def _parse_where_predicate(
    where: str, loaded: _LoadedTable
) -> Callable[[list[str]], bool] | None:
    """Parse a WHERE body into an INDEPENDENT Python row predicate over the
    ORIGINAL cell text, or None when it isn't safely recomputable here (→ the
    caller refuses, the conservative default).

    This is the load-bearing SAFETY oracle. The aggregate gate re-selects the
    contributing rows in PURE PYTHON (here), NOT via sqlite, so the agreement
    check `sqlite_value == python_value` compares TWO INDEPENDENT row selections
    — not 'sqlite agrees with sqlite'. Therefore every form added here is safe by
    construction: a parse that diverges from sqlite's WHERE makes the two sums
    DISAGREE → the gate REFUSES; it can NEVER ship a wrong subset. The worst case
    of any bug here is a false-refuse (recall), never a fabrication — which is why
    the oracle (not a SQLite-decomposition) is the right place to widen WHERE
    coverage (a SQLite-owned WHERE would degenerate the check to a tautology and
    admit rowid/random/subquery partial-sums — see ADR/CLAUDE notes).

    Supported (all the prompt permits): `col <op> value`, `col [NOT] IN (...)`,
    `col BETWEEN lo AND hi`, `col IS [NOT] NULL`, `col LIKE pat`, and a SINGLE
    level of AND / OR over those (SQL precedence: OR splits first, then AND). A
    `<col>__num` reference maps to its text column (coerce_number'd) so numeric
    filters work. Anything else — subqueries, rowid, random(), functions, mixed
    nesting, compound BETWEEN — does not match and returns None → REFUSE."""
    expr = where.strip()
    if _DANGEROUS_WHERE_RE.search(expr):
        return None

    # BETWEEN carries its own AND, which would defeat the AND-split below, so
    # handle a STANDALONE BETWEEN first (a BETWEEN composed under AND/OR won't
    # match the anchored regex → falls through → safely refuses).
    bm = _BETWEEN_RE.match(expr)
    if bm is not None:
        lo = coerce_number(_strip_quotes(bm.group("lo")))
        hi = coerce_number(_strip_quotes(bm.group("hi")))
        if lo is None or hi is None:
            return None
        lo_f, hi_f = lo, hi
        return _column_predicate(
            bm.group("col"),
            loaded,
            lambda idx: (
                lambda row: idx < len(row)
                and (cn := coerce_number(row[idx])) is not None
                and lo_f <= cn <= hi_f
            ),
        )

    ors = _split_top_level(expr, "or")
    if len(ors) > 1:
        or_preds = [_parse_where_predicate(o, loaded) for o in ors]
        if any(p is None for p in or_preds):
            return None
        return lambda row: any(p(row) for p in or_preds if p is not None)

    ands = _split_top_level(expr, "and")
    if len(ands) > 1:
        and_preds = [_parse_atom(a, loaded) for a in ands]
        if any(p is None for p in and_preds):
            return None
        return lambda row: all(p(row) for p in and_preds if p is not None)

    return _parse_atom(expr, loaded)


def _execute_select(db: sqlite3.Connection, sql: str) -> list[tuple[object, ...]] | None:
    """Execute a guarded SELECT under a VM-instruction step cap, returning up
    to `_ROW_CAP` rows, or None on any sqlite error.
    """
    aborted = {"flag": False}

    def _progress() -> int:
        aborted["flag"] = True
        return 1  # non-zero aborts the running statement

    db.set_progress_handler(_progress, _STEP_CAP)
    try:
        cur = db.execute(sql)
        rows = cur.fetchmany(_ROW_CAP)
    except sqlite3.Error:
        return None
    finally:
        db.set_progress_handler(None, _STEP_CAP)
    if aborted["flag"]:
        return None
    return rows


async def query_doc_tables(question: str, tables: list[StoredTable]) -> TableQueryResult | None:
    """Answer *question* over one document's `tables` via guarded text-to-SQL.

    Returns a `TableQueryResult` (kind="rows" verbatim cells, or kind=
    "aggregate" recomputed-and-agreeing scalar) or None on every failure /
    unsupported path — the caller treats None as "no table answer". The sqlite
    work runs through `asyncio.to_thread`; the model call is awaited.
    """
    log = logger.bind(node="query_doc_tables")
    if not tables:
        return None

    # Compute the sanitized schemas ONCE and show the LLM those exact
    # identifiers (sql table name + sql column names + their `__num`
    # companions), so the emitted SQL references columns that actually exist
    # in the in-memory db. The original header text rides along as a label for
    # semantics only.
    schemas = _compute_schemas(tables)
    prompt_tables = [
        {
            "table_id": s.sql_name,
            "section": s.stored.section,
            "columns": [
                {
                    "name": s.columns[ci],
                    "label": s.stored.header[ci] if ci < len(s.stored.header) else s.columns[ci],
                    "num": ci in s.numeric_cols,
                }
                for ci in range(len(s.columns))
            ],
            "sample_rows": [list(r) for r in s.stored.rows[:_SAMPLE_ROW_CAP]],
        }
        for s in schemas
    ]
    prompt = render_prompt("generate_table_sql", question=question, tables=prompt_tables)
    generated, _tokens = await complete_structured(
        prompt=prompt,
        schema=GeneratedSQL,
        prompt_tag=prompt_tag_for("generate_table_sql"),
    )

    sql = generated.sql.strip()
    log.info("table_sql.generated", sql=sql[:240], target=generated.target_table_id)
    if not _is_read_only_select(sql):
        log.info("table_sql.rejected_not_select", sql=sql[:120])
        return None

    # The LLM returns the sql_name (what it was shown); map it back to the
    # source table.
    by_sqlname = {s.sql_name: s.stored for s in schemas}
    target = by_sqlname.get(generated.target_table_id)
    if target is None:
        log.info("table_sql.unknown_target", target=generated.target_table_id)
        return None

    def _run() -> TableQueryResult | None:
        db = sqlite3.connect(":memory:")
        try:
            loaded_all = _load_tables(db, schemas)
            loaded = loaded_all.get(target.table_id)
            if loaded is None:
                return None
            rows = _execute_select(db, sql)
            if rows is None or not rows:
                return None
            return _classify_and_build(sql, target, loaded, rows)
        finally:
            db.close()

    result = await asyncio.to_thread(_run)
    if result is None:
        log.info("table_sql.no_result", target=generated.target_table_id)
    else:
        log.info("table_sql.result", kind=result.kind, target=result.target_table_id)
    return result


def _classify_and_build(
    sql: str,
    target: StoredTable,
    loaded: _LoadedTable,
    rows: list[tuple[object, ...]],
) -> TableQueryResult | None:
    """Classify the executed SELECT and build the gated TableQueryResult."""
    agg = _AGG_RE.match(sql)
    is_agg = agg is not None and agg.group("fn").lower() in _AGG_FUNCS

    if not is_agg:
        # kind="rows" — the projection must be exactly `SELECT *` so the
        # positional rows align with the FULL stored header (a reordered/subset
        # column list would mislead the header↔value attribution).
        if not _classify_is_select_star(sql):
            return None
        # The result rows are verbatim cells. We return the ORIGINAL stored
        # rows that the SQL selected — for a `SELECT *` they correspond
        # 1:1, but to keep the shipped values strictly verbatim-from-store we
        # surface them as string rows from the result, dropping any trailing
        # __num columns the projection might have pulled in.
        n_text = len(loaded.columns)
        result_rows: list[list[str]] = []
        for r in rows:
            cells = [("" if v is None else str(v)) for v in r[:n_text]]
            result_rows.append(cells)
        # A `LIMIT 1` superlative (`ORDER BY <numeric col> ASC|DESC LIMIT 1`)
        # gets a verified extremum framing IFF the returned row independently
        # checks out as the column extremum — so the agent can confidently
        # attribute "which X is highest/lowest". Unverified → plain rows (the
        # agent conservatively refuses, which is HARD-gate-safe).
        superlative: tuple[str, str] | None = None
        sm = _SUPERLATIVE_RE.match(sql)
        if sm is not None and result_rows:
            where = sm.group("where")
            superlative = _verify_superlative(
                target,
                loaded,
                sm.group("col"),
                sm.group("dir") or "asc",
                where.strip() if where else None,
                result_rows[0],
            )
        return TableQueryResult(
            kind="rows",
            sql=sql,
            target_table_id=target.table_id,
            rows=result_rows,
            aggregate_value=None,
            contributing_rows=result_rows,
            header=target.header,
            char_start=target.char_start,
            char_end=target.char_end,
            doc_id=target.doc_id,
            document_title="",
            heading_path=[target.section] if target.section else [],
            section=target.section,
            superlative=superlative,
        )

    # kind="aggregate" — single scalar, recompute-gated. `is_agg` already
    # implies `agg is not None`; this re-guard keeps pyright's narrowing
    # without an `assert` (defensive, unreachable in practice).
    if agg is None:
        return None
    if len(rows) != 1 or len(rows[0]) != 1:
        return None
    sql_raw = rows[0][0]
    if sql_raw is None:
        return None
    try:
        sql_value = float(sql_raw)  # type: ignore[arg-type]  # sqlite scalar agg → number
    except (TypeError, ValueError):
        return None

    where = agg.group("where")
    recompute = _recompute_aggregate(
        target,
        loaded,
        agg.group("fn"),
        agg.group("col"),
        where.strip() if where else None,
    )
    if recompute is None:
        return None
    # Finite-guard: a degenerate huge-digit cell coerces to `inf`, and
    # `abs(inf - inf) = nan`, which is NOT `> tol` (any comparison with nan is
    # False) → the agreement check would PASS and ship `Aggregate result = inf`.
    # Require both sides finite before the tolerance check can accept.
    if not (math.isfinite(sql_value) and math.isfinite(recompute)):
        logger.bind(node="query_doc_tables").info(
            "table_sql.non_finite_aggregate",
            sql_value=sql_value,
            recompute=recompute,
            target=target.table_id,
        )
        return None
    if abs(sql_value - recompute) > max(1.0, 1e-6 * abs(recompute)):
        logger.bind(node="query_doc_tables").info(
            "table_sql.recompute_disagree",
            sql_value=sql_value,
            recompute=recompute,
            target=target.table_id,
        )
        return None

    # Contributing rows: the source rows the aggregate drew from (after the
    # WHERE filter) so verify sees the real basis, not a bare scalar.
    contributing = _contributing_rows(target, loaded, where.strip() if where else None)
    return TableQueryResult(
        kind="aggregate",
        sql=sql,
        target_table_id=target.table_id,
        rows=None,
        aggregate_value=sql_value,
        contributing_rows=contributing,
        header=target.header,
        char_start=target.char_start,
        char_end=target.char_end,
        doc_id=target.doc_id,
        document_title="",
        heading_path=[target.section] if target.section else [],
        section=target.section,
    )


def _contributing_rows(
    stored: StoredTable, loaded: _LoadedTable, where: str | None
) -> list[list[str]]:
    """The source rows an aggregate is built from (post-WHERE), verbatim — the
    SAME independent predicate the recompute used, so the synthetic chunk's
    evidence rows match the verified scalar (not a stale all-rows fallback that
    would mislabel which rows a NOT IN / BETWEEN / AND aggregate drew from)."""
    if where is None:
        return [list(r) for r in stored.rows]
    match_row = _parse_where_predicate(where, loaded)
    if match_row is None:
        # Unreachable once the recompute has agreed (it used the same parse), but
        # fail safe to the full rows rather than an empty/misleading basis.
        return [list(r) for r in stored.rows]
    return [list(r) for r in stored.rows if match_row(r)]
