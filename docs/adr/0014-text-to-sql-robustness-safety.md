# ADR-0014: Text-to-SQL Robustness + Safety — Keep the Independent Python WHERE Oracle, Reject SQL-Stack Decomposition

- **Status**: Accepted
- **Date**: 2026-05-31
- **Deciders**: Memex core team
- **Tags**: agents, table-rag, text-to-sql, safety, hard-gate, architecture, rejected-alternative

## Context

Table-RAG Phase 2 (the `query_tables` node + `agents/table_sql.py`, shipped 2026-05-24, ADR-0003-style derived state) answers numeric/aggregate/superlative questions over tables by generating ONE read-only `SELECT` against a fresh in-memory sqlite — one table per `StoredTable`, every numeric column duplicated as a `<col>__num REAL` companion (built by `coerce_number` at load). The result is injected as a synthetic `Chunk` `<doc_id>#sql0001` into `state.reranked`, so the existing assess/answer/verify/compose machinery cites it like any other chunk — **no new grounding path, no LangGraph `ToolNode`**.

The no-hallucination HARD gate (`refusal_cf=1.0`, 0 hallucinations) is held by the **row-vs-aggregate fabrication boundary**: a `kind="rows"` result ships **verbatim document cells** (safe); a `kind="aggregate"` result ships a **NEW number**, so it is injected ONLY when `_recompute_aggregate` — an **INDEPENDENT Python recompute over the original cell text** — agrees within tolerance, else the node no-ops and the agent refuses.

That gate's correctness rests entirely on the recompute being **independent of sqlite**. But the original WHERE coverage was narrow (a single equality / numeric comparison), so legitimate filtered aggregates (`IN`, `BETWEEN`, `AND`/`OR`, …) and several near-twin 10-K tables **false-refused** — a recall gap, not a safety gap.

The 2026-05-31 audit-10 batched re-process converged two pressures. The recall gap was real, AND the 10-K re-parse exposed adjacent structure defects: Docling **MERGED** the Director Compensation `Stock Awards`/`Total` columns (no clean numeric `Total` for the ar-15 MIN-superlative); a filing table's section was its **BOLD caption** not an ATX heading (near-twin Compensation tables indistinguishable for SQL targeting); the synthetic chunk **named no source table** (the 8B couldn't map a bare `Total ($)` → "total compensation", ar-15); and a fabricated total could be re-expressed as an **arithmetic SUM expression** of verbatim cells to slip the per-figure verify check (ar-16).

The contested question was **where to widen WHERE coverage**. The obvious move — let sqlite own the WHERE — is the one that quietly **destroys the safety property**: because the `__num` companion uses the SAME `coerce_number` the recompute uses, a sqlite re-sum and the Python recompute are algebraically equal for every `W`, so the agreement check degenerates to "sqlite agrees with sqlite."

## Decision Drivers

- **The HARD gate is non-negotiable.** A `kind="aggregate"` number is a fabrication unless an INDEPENDENT oracle confirms it.
- **Independence IS the safety property** — not "more SQL is more correct." A divergence between two independent row selections can only REFUSE, never ship a wrong subset.
- **Recall wins come from widening the independent oracle**, not from delegating row-selection to sqlite.
- **Generality over one-document overfit** — structural detectors, default-on + kill-switchable, validated on a 2nd document.
- **Reversibility / containment** — every new behavior is kill-switchable + regression-gated; worst case is a conservative false-refuse.

## Considered Options

1. **Decompose-and-verify via the SQL stack** — run the LLM's aggregate for the scalar, then `SELECT * FROM t WHERE W` to materialize rows and recompute over those. **Rejected** (unsafe — a tautology; evidence below).
2. **Keep the recompute independent but leave WHERE narrow** — safe, but leaves the filtered-aggregate / near-twin recall gap. **Rejected** (under-delivers).
3. **Keep + WIDEN the independent Python WHERE oracle** (`_parse_where_predicate`) + targeted guards (A1 / A2 / NOCASE / coercion-soundness) + the table-structure fixes (merged-column split, caption attribution, synthetic-chunk naming, SUM-expression verify demotion). **Chosen.**

## Decision

We chose **Option 3**: keep the aggregate gate's safety as an INDEPENDENT Python row-selection oracle and WIDEN it; reject the SQL-stack decomposition as unsafe.

`_recompute_aggregate` re-selects the contributing rows in **pure Python** via `_parse_where_predicate` from the ORIGINAL cell text (NOT the sqlite `__num` column), so `sqlite_value == python_value` compares two independent selections. The oracle (`agents/table_sql.py::_parse_where_predicate`) is widened from single-equality to: `col <op> value`, `col [NOT] IN (...)`, `col BETWEEN lo AND hi`, `col IS [NOT] NULL`, `col LIKE pat`, and a SINGLE level of `AND`/`OR` (OR splits first, then AND), each atom independently evaluated. Dangerous forms (`rowid`/`random()`/`glob`/sub-`SELECT`) are rejected by `_DANGEROUS_WHERE_RE` and match no safe atom. **Because the oracle is independent, a parse divergence makes the two sums DISAGREE → REFUSE: a widening bug can only false-refuse (recall), never ship a wrong subset.**

Three companion safety guards close the misread surfaces:

- **A1** — `SUM`/`AVG`/`MIN`/`MAX` must target the `<col>__num` companion. A raw text-column aggregate is a coercion-misread surface (sqlite's lenient prefix-coercion `'350000.75 deferred'`→350000.75 vs `coerce_number`→350000, divergent within the ±1 floor). `COUNT` is exempt.
- **A2** — a superlative `ORDER BY` must target `__num` so sqlite sorts NUMERICALLY. A text `ORDER BY` sorts lexically (`'9' > '10'`) and the ±1 extremum tolerance framed the WRONG row as the extremum.
- **Coercion-soundness** — refuse an aggregate/superlative whose contributing cell coerces but is NOT `core/text.is_canonical_number_cell` (malformed `1,2,3`→123, mixed-separator European `1.234,56`). `coerce_number` is lenient and the `__num` column shares it, so the agreement check is blind to such a misread.

`_load_tables` builds every text column `COLLATE NOCASE`, aligning sqlite `=`/`IN`/`LIKE` with the case-insensitive recompute → the dominant value-linking false-refuse (`'us'` vs stored `'US'`) ships the correct value.

Three table-structure fixes feed the gate clean inputs: `split_merged_columns` recovers a Docling-MERGED column at the SQL-store extract path ONLY (gated `AgentsSettings.table_column_split_enabled`, regression-gated by `scripts/table_split_audit.py`); `nearest_table_caption` attributes a filing table's section to its BOLD caption; `_build_synthetic_chunk` NAMES the source table. And `verify._claim_is_sum_expression` demotes a claim phrased as `a + b + …` (the ar-16 evasion).

### Why the rejected option is unsafe (empirical)

Decompose-and-verify was prototyped and run. Because `__num[i] == coerce_number(text[i])` by construction, `S = SUM(__num over W)` and a re-sum over the `SELECT * WHERE W` rows are **algebraically equal for every `W`**. With a true table of 100, it shipped: `WHERE rowid <= 2` → 30, `WHERE cost__num > (SELECT AVG(cost__num))` → 70, `WHERE cost__num % 20 = 0` → 60 — each an arbitrary partial-sum framed as the total — and `WHERE (abs(random()) % 2) = 0` → a non-reproducible scalar (violating the content-addressed determinism contract). The current oracle REFUSES all of these. **Do not reintroduce the SQL-stack decomposition.**

## Consequences

### Positive
- **Recall wins, safely**: filtered aggregates (`IN`/`BETWEEN`/`LIKE`/`AND`/`OR`), NOCASE value-linking, near-twin tables, and bare-column→question mapping all answer instead of false-refusing — without weakening the gate.
- **The HARD gate is held by construction**: two independent row selections; a wrong filter or widening bug disagrees → refuse; A1/A2/coercion-soundness close the shared-grammar / lexical-sort / lenient-coercion misread surfaces. Pinned by a 42-row adversary matrix in `tests/unit/test_table_sql.py` (every adversary row MUST-refuse).
- **General, not one-document overfit**: the SQL aggregate/superlative/counterfactual paths are now gated by a 2nd document (`scientific-gte-19/20/21`); the merged-column split is a structural detector firing only where merged columns exist, contained to the SQL path.

### Negative / Trade-offs
- **Documented coercion residual (NOT closed)**: locale (`'1.000'` European decimal-vs-thousands) + unit (`'5m'` metres-vs-million) ambiguity needs context the system lacks; coercion-soundness conservatively refuses rather than guess. Absent from the US-format corpora.
- The merged-column split has a **proven-but-currently-absent false-positive class** (coordinates, mean±stddev cells); contained to the SQL path (recompute-gated, kill-switchable, regenerable), regression-gated (0 false-splits across 47 docs).
- **ar-15 remains a BORDERLINE query** (the 8B `Total ($)`→"total compensation" inference flips ~1-in-4); the mechanism is sound, and the answer node is deliberately NOT over-promoted to force it (that would risk the gate).

### Neutral
- No new grounding path, no LangGraph `ToolNode`; the synthetic-chunk seam + the row-vs-aggregate boundary are unchanged. The split applies at the SQL-store extract path ONLY — the `[table-rows]` linearizer + raw GFM keep the original merged structure by design.

## References
- Spec: [`docs/specs/table-sql.md`](../specs/table-sql.md), [`docs/specs/table-rag.md`](../specs/table-rag.md)
- Backend rules: [`src/memex/CLAUDE.md`](../../src/memex/CLAUDE.md) (the text-to-SQL safety architecture)
- Commits: `2d27d12` (centralize `coerce_number`), `8e4d322` (column-split + caption), `3a69dc3` (answer-side 10-K gaps), `689dee5` (text-to-SQL robustness), `4835029` (docs)
- Sibling: [ADR-0009](0009-remove-free-form-synthesis-baseline.md) (precedent for recording a rejected approach)
