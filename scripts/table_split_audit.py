"""Column-split regression audit (Table-RAG `split_merged_columns`, 2026-05-31).

`split_merged_columns` recovers Docling-MERGED numeric columns at table-extract
time (a header cell carrying `**A** **B**` over data cells `278,809 342,559`).
The fix was developed against ONE document (the NVIDIA 10-K, ar-14/15/16) and
the docstring/config claim "Validated 0-false-split across the 47-doc vault" had
NO committed audit — the analogue of `scripts/report_structure_audit.py` this
script supplies.

It runs the EXACT production path — `index/table_store.extract_tables` with
`split_merged=True` vs `False` — over every vault document, diffs the column
count per table (matched by `char_start`, stable under the split), and reports
every FIRE (where the split changed a table's shape) with its before→after
header. The split has a PROVEN-but-currently-absent false-positive class: a
legitimately two-number-per-cell column with an adjacent-bold header (lat/long
coordinates, mean±stddev, value±uncertainty) would be wrongly sliced. So the
audit's job is to SURFACE every fire for human review — a fire on a NON-10-K /
non-financial table is the signal to inspect.

`--check <baseline.json>` turns the census into a REGRESSION GATE: it fails if
the set of fires (keyed by doc_id + char_start, with before/after headers)
diverges from the committed baseline — a NEW fire (reindex churn, a new doc, or
a real false-positive) or a VANISHED fire both demand review.

Usage:
  MEMEX_OBSERVABILITY__LANGFUSE_ENABLED=false \
    uv run python scripts/table_split_audit.py            # census to stdout
  ... scripts/table_split_audit.py --write-baseline tests/eval-data/table-split-baseline.json
  ... scripts/table_split_audit.py --check tests/eval-data/table-split-baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from memex.cli.bootstrap import bootstrap
from memex.core.config import get_settings
from memex.core.text import coerce_number
from memex.index.table_store import extract_tables
from memex.vault.store import read_document

_SPLIT_MAJORITY = 0.8  # mirror table_linearize._SPLIT_MAJORITY for the validity check


def _doc_ids(vault_path: Path) -> list[str]:
    docs_dir = vault_path / "documents"
    return sorted(p.stem for p in docs_dir.glob("*.md"))


def _column_numeric_fraction(rows: list[list[str]], col: int) -> float:
    """Fraction of non-empty cells in `col` that coerce to a number — the split's
    own contract (every recovered column should be numeric-majority)."""
    cells = [r[col] for r in rows if col < len(r) and r[col].strip()]
    if not cells:
        return 0.0
    return sum(1 for c in cells if coerce_number(c) is not None) / len(cells)


async def _audit_doc(vault_path: Path, doc_id: str) -> list[dict[str, object]]:
    """Return one record per table whose column count changed under the split."""
    doc = await read_document(vault_path, doc_id)
    on = extract_tables(doc_id, doc.body, split_merged=True)
    off = extract_tables(doc_id, doc.body, split_merged=False)
    off_by_pos = {t.char_start: t for t in off}
    fires: list[dict[str, object]] = []
    for t in on:
        base = off_by_pos.get(t.char_start)
        if base is None or len(t.header) == len(base.header):
            continue
        # A fire: confirm every NEW column is numeric-majority (the contract) so a
        # garbage split is visible as low validity in the report.
        valid = all(
            _column_numeric_fraction(t.rows, ci) >= _SPLIT_MAJORITY
            for ci in range(len(t.header))
            if ci >= len(base.header) - 1  # the split region (grew at the split point)
        )
        fires.append(
            {
                "doc_id": doc_id,
                "char_start": t.char_start,
                "section": t.section,
                "cols_before": len(base.header),
                "cols_after": len(t.header),
                "header_before": base.header,
                "header_after": t.header,
                "numeric_valid": valid,
            }
        )
    return fires


async def _run() -> dict[str, object]:
    settings = get_settings()
    vault_path = settings.vault_path
    doc_ids = _doc_ids(vault_path)
    all_fires: list[dict[str, object]] = []
    total_tables = 0
    for doc_id in doc_ids:
        doc = await read_document(vault_path, doc_id)
        total_tables += len(extract_tables(doc_id, doc.body, split_merged=False))
        all_fires.extend(await _audit_doc(vault_path, doc_id))
    fire_docs = sorted({f["doc_id"] for f in all_fires})
    return {
        "total_docs": len(doc_ids),
        "total_tables": total_tables,
        "split_fires": len(all_fires),
        "split_fire_docs": fire_docs,
        "invalid_fires": [f for f in all_fires if not f["numeric_valid"]],
        "fires": sorted(all_fires, key=lambda f: (f["doc_id"], f["char_start"])),
    }


def _fire_key(f: dict[str, object]) -> str:
    return f"{f['doc_id']}@{f['char_start']}:{f['cols_before']}->{f['cols_after']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default=None, help="baseline JSON to gate against (exit 1 on divergence)")
    ap.add_argument("--write-baseline", default=None, help="write the current census as a baseline JSON")
    args = ap.parse_args()

    bootstrap()
    out = asyncio.run(_run())

    # Human census to stderr.
    print(
        f"=== column-split audit: {out['split_fires']} fires across "
        f"{len(out['split_fire_docs'])} doc(s) / {out['total_tables']} tables / {out['total_docs']} docs ===",
        file=sys.stderr,
    )
    for f in out["fires"]:
        flag = "" if f["numeric_valid"] else "  ⚠ NON-NUMERIC SPLIT (review!)"
        print(
            f"  {f['doc_id'][:34]:34} {f['cols_before']}->{f['cols_after']}  {f['section'][:40]!r}{flag}",
            file=sys.stderr,
        )
        print(f"      before: {f['header_before']}", file=sys.stderr)
        print(f"      after : {f['header_after']}", file=sys.stderr)
    if out["invalid_fires"]:
        print(f"\n⚠ {len(out['invalid_fires'])} fire(s) produced a NON-numeric column — likely false-positive split.", file=sys.stderr)

    if args.write_baseline:
        keys = sorted(_fire_key(f) for f in out["fires"])
        Path(args.write_baseline).write_text(json.dumps({"fires": keys}, indent=2) + "\n")
        print(f"\nwrote baseline ({len(keys)} fires) -> {args.write_baseline}", file=sys.stderr)

    if args.check:
        baseline = set(json.loads(Path(args.check).read_text())["fires"])
        current = {_fire_key(f) for f in out["fires"]}
        new = sorted(current - baseline)
        gone = sorted(baseline - current)
        invalid = bool(out["invalid_fires"])
        if new or gone or invalid:
            if new:
                print(f"\nFAIL: {len(new)} NEW split fire(s) — review (genuine merge or a false-positive?):", file=sys.stderr)
                for k in new:
                    print(f"  + {k}", file=sys.stderr)
            if gone:
                print(f"\nFAIL: {len(gone)} fire(s) VANISHED vs baseline:", file=sys.stderr)
                for k in gone:
                    print(f"  - {k}", file=sys.stderr)
            if invalid:
                print("\nFAIL: a fire produced a non-numeric column (see ⚠ above).", file=sys.stderr)
            sys.exit(1)
        print(f"\nOK: {len(current)} fires match the baseline; all numeric-valid.", file=sys.stderr)

    # Machine-readable census to stdout.
    print(json.dumps({k: v for k, v in out.items() if k != "fires"}, indent=2))


if __name__ == "__main__":
    main()
