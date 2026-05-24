"""Corpus-extension scaffolding for Memex eval suites.

Streamlines the "drop a PDF, label some queries, A/B test" workflow
used to bootstrap the slide-decks (English) and french-course (French)
eval corpora. The standard playbook is now:

    # 1) Ingest the PDF
    memex ingest /path/to/report.pdf
    # Returns a doc_id like '5795b16a-pdf-cr350-cours-1'

    # 2) Inspect the parsed markdown structure
    uv run python scripts/extend_corpus.py inspect <doc_id>

    # 3) Initialize a draft eval corpus
    uv run python scripts/extend_corpus.py init <corpus-name> \\
        --doc-id <doc_id>

    # 4) Hand-edit tests/eval-data/<corpus-name>/queries.json with real
    #    questions + `_anchor_phrase` per query (a distinctive substring
    #    from the expected chunk text)

    # 5) Resolve chunk_ids from anchor phrases
    uv run python scripts/extend_corpus.py resolve \\
        tests/eval-data/<corpus-name>/queries.json

    # 6) A/B eval: chart-OCR enabled (default) vs disabled
    uv run python scripts/extend_corpus.py ab \\
        tests/eval-data/<corpus-name>/queries.json

This file is intentionally self-contained — it composes existing Memex
APIs (FTSStore, VaultDocument, etc.) and shells out to `memex eval` for
the A/B step. No new infrastructure.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from memex.cli.bootstrap import bootstrap
from memex.core.config import get_settings
from memex.index.fts_store import FTSStore
from memex.vault.store import read_document


# ---------------------------------------------------------------------------
# init — create the directory + template queries.json
# ---------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    """Create a draft `tests/eval-data/<name>/queries.json` with two
    placeholder queries (one ANS, one REF) showing the schema.
    """
    target_dir = Path(f"tests/eval-data/{args.name}")
    target = target_dir / "queries.json"

    if target.exists() and not args.force:
        print(f"  ✗ {target} already exists. Use --force to overwrite.")
        return 1

    target_dir.mkdir(parents=True, exist_ok=True)

    today = dt.date.today().isoformat()
    doc_id = args.doc_id or "<fill-in-doc-id>"

    template: dict[str, Any] = {
        "_description": (
            f"Hand-labelled eval corpus for {args.name}. Created via "
            f"scripts/extend_corpus.py init."
        ),
        "_doc_id": doc_id,
        "_query_count": 2,
        "_mix": {
            "ans": 1,
            "ref": 1,
        },
        "_labelling_rule": (
            "ANS queries: set `_anchor_phrase` to a distinctive substring "
            "from the chunk that contains the expected answer; run "
            "`scripts/extend_corpus.py resolve <this-file>` to populate "
            "`relevant_chunk_ids`. REF queries: set `should_refuse=true` "
            "and leave `relevant_chunk_ids=[]`."
        ),
        "_created": today,
        "_purpose": (
            f"Extension of the eval matrix to a new corpus ({args.name}). "
            f"Used to A/B test chart-OCR + agent behavior on this content."
        ),
        "queries": [
            {
                "qid": f"{args.name}-01",
                "question": "<your question here, e.g. 'What was X in Y?'>",
                "_anchor_phrase": (
                    "<distinctive substring from the chunk with the answer; "
                    "5-12 words, ideally containing the numeric answer>"
                ),
                "_expected_answer": (
                    "<the expected answer string the agent should produce>"
                ),
                "_answer_type": "single_fact",
                "_note": "<optional notes about the query>",
                "should_refuse": False,
                "relevant_chunk_ids": [],
            },
            {
                "qid": f"{args.name}-02",
                "question": (
                    "<a counterfactual question whose answer is NOT in "
                    "the corpus; e.g. asking for data that the doc "
                    "doesn't cover>"
                ),
                "_anchor_phrase": "",
                "_expected_answer": (
                    "REFUSE — the corpus doesn't cover this."
                ),
                "_answer_type": "counterfactual",
                "_note": (
                    "HARD GATE refusal query. Tests that the agent doesn't "
                    "hallucinate on out-of-corpus questions."
                ),
                "should_refuse": True,
                "relevant_chunk_ids": [],
            },
        ],
    }

    target.write_text(json.dumps(template, indent=2) + "\n")
    print(f"  ✓ Wrote {target}")
    print(f"  Next:")
    print(f"    1. Edit {target} with real questions + _anchor_phrase")
    print(f"    2. Run: uv run python scripts/extend_corpus.py resolve {target}")
    print(f"    3. Run: uv run python scripts/extend_corpus.py ab {target}")
    return 0


# ---------------------------------------------------------------------------
# inspect — dump heading map + chart-block locations for a doc
# ---------------------------------------------------------------------------


def _find_all(text: str, needle: str) -> list[int]:
    out: list[int] = []
    pos = 0
    while True:
        idx = text.find(needle, pos)
        if idx < 0:
            return out
        out.append(idx)
        pos = idx + 1


async def cmd_inspect(args: argparse.Namespace) -> int:
    """Print the doc's heading map + chart-block summary to help the
    user pick interesting query targets.
    """
    bootstrap()
    settings = get_settings()

    try:
        doc = await read_document(settings.vault_path, args.doc_id)
    except Exception as e:
        print(f"  ✗ couldn't read doc: {type(e).__name__}: {e}")
        return 1

    print(f"=== Doc: {args.doc_id} ===")
    print(f"  Markdown size: {len(doc.body):,} chars")
    print(f"  Frontmatter title: {doc.frontmatter.title!r}")
    print()

    # Heading map
    print("=== Heading map ===")
    heading_count = 0
    for line in doc.body.splitlines():
        if line.startswith("# ") or line.startswith("## ") or line.startswith("### "):
            depth = len(line) - len(line.lstrip("#"))
            indent = "  " * (depth - 1)
            print(f"  {indent}{line[:120]}")
            heading_count += 1
            if heading_count >= 60:
                print(f"  ... (truncated; {len(doc.body.splitlines())} more lines)")
                break
    if heading_count == 0:
        print("  (no headings detected)")
    print()

    # Chart-extracted block locations
    chart_starts = _find_all(doc.body, "[chart-extracted]")
    print(f"=== Chart-extracted blocks: {len(chart_starts)} ===")
    for i, idx in enumerate(chart_starts[:8]):
        ctx_before_end = idx
        ctx_before_start = max(0, idx - 120)
        ctx_before = doc.body[ctx_before_start:ctx_before_end].split("\n")[-1].strip()
        block_end = doc.body.find("[/chart-extracted]", idx)
        block_text = doc.body[idx : block_end if block_end > 0 else idx + 250]
        block_text = block_text.replace("\n", " ⏎ ")[:200]
        print(f"  [{i + 1}] @ char {idx}")
        print(f"    nearby heading/text: ...{ctx_before[:100]}")
        print(f"    block:               {block_text}...")
    if len(chart_starts) > 8:
        print(f"  ... ({len(chart_starts) - 8} more chart blocks)")
    print()

    # FTS chunk count for this doc + sample
    fts = await FTSStore.open(settings.vault_path)
    try:
        def _query() -> list[tuple[str, str]]:
            cur = fts._db.execute(  # type: ignore[attr-defined]
                "SELECT chunk_id, substr(text, 1, 100) FROM chunks_fts "
                "WHERE document_id = ? LIMIT 10",
                (args.doc_id,),
            )
            return list(cur.fetchall())

        rows = await asyncio.to_thread(_query)
        # Total count
        def _count() -> int:
            cur = fts._db.execute(  # type: ignore[attr-defined]
                "SELECT COUNT(*) FROM chunks_fts WHERE document_id = ?",
                (args.doc_id,),
            )
            return int(cur.fetchone()[0])

        total = await asyncio.to_thread(_count)
        print(f"=== Chunks for {args.doc_id}: {total} ===")
        print(f"  First 10 chunks (chunk_id, first 100 chars):")
        for cid, snippet in rows:
            print(f"    {cid}")
            print(f"      {snippet!r}")
    finally:
        await fts.close()

    return 0


# ---------------------------------------------------------------------------
# resolve — fill in `relevant_chunk_ids` from `_anchor_phrase`
# ---------------------------------------------------------------------------


async def cmd_resolve(args: argparse.Namespace) -> int:
    """Walk a queries.json, search FTS for each query's `_anchor_phrase`,
    and populate `relevant_chunk_ids` with the matching chunk_id(s).

    REF queries (should_refuse=true) are skipped — their
    `relevant_chunk_ids` stays empty.
    """
    bootstrap()
    settings = get_settings()

    queries_path = Path(args.queries_file)
    if not queries_path.exists():
        print(f"  ✗ {queries_path} not found")
        return 1

    data = json.loads(queries_path.read_text())
    queries = data.get("queries", [])
    corpus_doc_id = data.get("_doc_id", "")
    multi_doc = bool(data.get("_multi_doc"))

    # Single-doc corpora must declare a usable corpus-level `_doc_id`.
    # Multi-doc corpora (`_multi_doc: true`) carry a per-query `_doc_id`
    # instead, so the corpus-level one is allowed to be a placeholder.
    if not multi_doc and (not corpus_doc_id or corpus_doc_id.startswith("<")):
        print(f"  ✗ _doc_id not set in {queries_path}; fill it in first")
        return 1

    fts = await FTSStore.open(settings.vault_path)
    try:
        resolved = 0
        skipped = 0
        unresolved: list[str] = []

        for q in queries:
            qid = q.get("qid", "?")

            if q.get("should_refuse"):
                skipped += 1
                print(f"  ↦ {qid}: REF query, skipped")
                continue

            anchor = q.get("_anchor_phrase", "").strip()
            if not anchor or anchor.startswith("<"):
                print(f"  ⚠ {qid}: no _anchor_phrase, skipping")
                unresolved.append(qid)
                continue

            # Per-query `_doc_id` (multi-doc corpora) overrides the
            # corpus-level one; fall back to the corpus value otherwise.
            target_doc = (q.get("_doc_id") or corpus_doc_id or "").strip()
            if not target_doc or target_doc.startswith("<"):
                print(f"  ⚠ {qid}: no _doc_id (query or corpus), skipping")
                unresolved.append(qid)
                continue

            # FTS5 phrase search restricted to this doc
            chunks = await fts.search(anchor, k=5)
            chunks = [c for c in chunks if c.document_id == target_doc]

            if not chunks:
                print(f"  ✗ {qid}: anchor {anchor!r} matched nothing in {target_doc}")
                unresolved.append(qid)
                q["_unresolved"] = True
                continue

            chunk_ids = [c.chunk_id for c in chunks[: args.max_chunks]]
            q["relevant_chunk_ids"] = chunk_ids
            q.pop("_unresolved", None)
            resolved += 1
            print(f"  ✓ {qid}: {len(chunk_ids)} chunk_id(s): {chunk_ids}")
    finally:
        await fts.close()

    # Update query count + mix. `_mix` is usually a dict but the legacy
    # slide-decks corpus uses a free-form string ("17 answerable + 5
    # empty-retrieval refusals + 8 near-miss refusals"). Only update
    # dict-shaped _mix; leave string-shaped untouched to avoid clobbering
    # the categorisation note.
    data["_query_count"] = len(queries)
    if isinstance(data.get("_mix"), dict):
        data["_mix"]["ans"] = sum(1 for q in queries if not q.get("should_refuse"))
        data["_mix"]["ref"] = sum(1 for q in queries if q.get("should_refuse"))
    data["_resolved_at"] = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    queries_path.write_text(json.dumps(data, indent=2) + "\n")
    print()
    print(f"  Resolved: {resolved} / {len(queries) - skipped} ANS queries")
    print(f"  Skipped:  {skipped} REF queries")
    if unresolved:
        print(f"  ⚠ Unresolved: {unresolved}")
        print(f"    Check that the _anchor_phrase tokens exist in the parsed markdown.")
        return 1
    return 0


# ---------------------------------------------------------------------------
# ab — A/B eval: chart-OCR enabled (default) vs disabled
# ---------------------------------------------------------------------------


def cmd_ab(args: argparse.Namespace) -> int:
    """Run `memex eval <queries-file>` twice — once with the current
    chart-OCR config (default: enabled per 2026-05-23 P3.3-c), once
    with `MEMEX_PARSE__DISABLE_CHART_OCR=true`. Report ANS / refusal_cf
    / mcp_ans side-by-side.

    Caveat: the vault must already be parsed in BOTH states to get a
    clean A/B. If the markdown was last produced with chart-OCR on,
    re-running with chart-OCR off doesn't re-parse — it just changes
    the env-var of the eval process. For a proper A/B, the user
    should:
      1. Re-parse with chart-OCR off:
           MEMEX_PARSE__DISABLE_CHART_OCR=true memex parse <doc-id>
           memex index <doc-id>
      2. Run `memex eval` and save the result.
      3. Re-parse with chart-OCR on (default):
           memex parse <doc-id>
           memex index <doc-id>
      4. Run `memex eval` and save the result.
      5. Compare.

    This subcommand assumes the vault is already in the desired state
    and just runs the eval with both env-var settings to show the
    impact of the chart-OCR markdown on retrieval / answering, NOT a
    full pipeline re-parse.
    """
    queries_path = Path(args.queries_file)
    if not queries_path.exists():
        print(f"  ✗ {queries_path} not found")
        return 1

    def _run(label: str, env_override: dict[str, str]) -> dict[str, Any]:
        env = os.environ.copy()
        env.update(env_override)
        print(f"\n=== {label} ===")
        for k, v in env_override.items():
            print(f"    {k}={v}")
        result = subprocess.run(
            ["uv", "run", "memex", "eval", str(queries_path)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        # Find the run_id JSON line
        for line in result.stdout.splitlines():
            if line.startswith('{"run_id"'):
                return json.loads(line)
        print(f"  ✗ eval did not emit a result line. Last 10 stderr lines:")
        for line in result.stderr.splitlines()[-10:]:
            print(f"    {line}")
        return {}

    on = _run(
        "WITH chart-OCR (current default)",
        {},
    )
    off = _run(
        "WITHOUT chart-OCR",
        {"MEMEX_PARSE__DISABLE_CHART_OCR": "true"},
    )

    if not on or not off:
        return 1

    print()
    print("=== A/B summary ===")
    print(f"  {'metric':<22} {'with chart-OCR':<20} {'without':<20} {'delta':<10}")
    print(f"  {'-' * 70}")
    for key in [
        "answered_count",
        "refused_count",
        "refusal_rate_on_counterfactuals",
        "mean_citation_precision",
        "mean_citation_precision_answered_only",
    ]:
        on_val = on.get(key, "n/a")
        off_val = off.get(key, "n/a")
        if isinstance(on_val, (int, float)) and isinstance(off_val, (int, float)):
            delta = on_val - off_val
            delta_str = f"{delta:+.3f}" if isinstance(on_val, float) else f"{delta:+d}"
            print(f"  {key:<22} {on_val:<20} {off_val:<20} {delta_str}")
        else:
            print(f"  {key:<22} {on_val!s:<20} {off_val!s:<20}")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Scaffold a Memex eval corpus from an ingested document. See "
            "the module docstring for the full workflow."
        )
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="create a new eval-data dir + template")
    p_init.add_argument("name", help="corpus name (e.g. 'annual-report')")
    p_init.add_argument("--doc-id", help="optional doc_id to pre-fill")
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=cmd_init, is_async=False)

    p_inspect = sub.add_parser("inspect", help="dump heading map + chart-block summary")
    p_inspect.add_argument("doc_id", help="vault doc_id (from `memex ingest` output)")
    p_inspect.set_defaults(func=cmd_inspect, is_async=True)

    p_resolve = sub.add_parser("resolve", help="fill `relevant_chunk_ids` from `_anchor_phrase`")
    p_resolve.add_argument("queries_file", help="path to queries.json")
    p_resolve.add_argument("--max-chunks", type=int, default=2, help="max chunk_ids per query")
    p_resolve.set_defaults(func=cmd_resolve, is_async=True)

    p_ab = sub.add_parser("ab", help="A/B eval: chart-OCR on vs off")
    p_ab.add_argument("queries_file", help="path to queries.json")
    p_ab.set_defaults(func=cmd_ab, is_async=False)

    args = p.parse_args()

    if args.is_async:
        return asyncio.run(args.func(args))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
