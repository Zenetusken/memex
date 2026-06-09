"""Diagnostic harness for FALSE REFUSALS — answerable queries the agent wrongly refuses.

A false refusal = an ANSWERABLE eval query (`should_refuse=false`) that the agent REFUSES
(`answered=false`). It's HARD-gate-SAFE (a wrong refusal, never a hallucination) but hurts
utility. This script runs each query through the REAL `answer_query` and records the trifecta
that classifies a false refusal:

  1. outcome      — answered? refusal_reason? how many claims?
  2. refusing NODE — regex-attributed from the refusal_reason (the 7 verified templates).
  3. gold-in-pool — were the query's gold `relevant_chunk_ids` in the pool the gates ACTUALLY
                    saw? Read from `FinalResponse.used_chunks` (the refuse node sets it to
                    `state.reranked`), NOT a re-derived search — so it can't diverge from the
                    agent's real pool. Separates a RETRIEVAL miss from a GATE over-refusal.

Multi-run (`--runs N`) separates deterministic over-refusals from borderline greedy flips (the
documented eval non-determinism). Read-only — no writes, no production-path change.

Usage (device-pinned so refusal behaviour doesn't vary with ambient GPU pressure):
  MEMEX_MODELS__CO_RESIDENCE_MODE=manual MEMEX_MODELS__EMBEDDER_DEVICE=cpu \
  MEMEX_MODELS__RERANKER_DEVICE=cpu \
  uv run python scripts/false_refusal_audit.py tests/eval-data/<corpus>/queries.json [--runs 3] \
      [--qid linux-fundamentals-11] [--json]

Exit 0 always (this is a measurement, not a gate).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

# --- refusal_reason → refusing-node attribution (verified against agents/answering.py) ---
# Order matters: most-specific prefixes first; the free-text sufficiency model reason is the
# fall-through (a non-empty reason that matches none of the fixed templates).
_NODE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("assess-empty", re.compile(r"^No relevant content found in the vault")),
    ("assess-failclosed", re.compile(r"^The sufficiency check failed")),
    ("relevance", re.compile(r"^The retrieved material addresses a related topic")),
    ("verify-zero", re.compile(r"^I drafted an answer but couldn't ground")),
    ("compose-zero", re.compile(r"^Verification returned no grounded claims")),
    ("over-budget", re.compile(r"^I exceeded my reasoning budget")),
    ("catch-all", re.compile(r"^I couldn't construct a confident answer")),
]


def attribute_node(refusal_reason: str | None) -> str:
    """Map a refusal_reason to the node that produced it. `sufficiency-model` = the free-text
    `SufficiencyAssessment.reason` (matches none of the fixed templates)."""
    if not refusal_reason:
        return "answered"
    for node, rx in _NODE_PATTERNS:
        if rx.search(refusal_reason):
            return node
    return "sufficiency-model"


def _gold_in_pool(gold_ids: list[str], used_chunks: list[Any]) -> tuple[int, int]:
    """(# gold chunk_ids present in the pool the gates saw, # gold chunk_ids total)."""
    pool = {c.chunk_id for c in used_chunks}
    present = sum(1 for g in gold_ids if g in pool)
    return present, len(gold_ids)


async def _run_one(question: str) -> dict[str, Any]:
    from memex.agents.answering import answer_query

    resp = await answer_query(question)
    return {
        "answered": resp.answered,
        "refusal_reason": resp.refusal_reason,
        "node": attribute_node(resp.refusal_reason if not resp.answered else None),
        "n_claims": len(resp.claims),
        "used_chunks": list(resp.used_chunks),
    }


def _classify(query_should_refuse: bool, runs: list[dict[str, Any]]) -> str:
    """Per-query verdict across runs."""
    answered = [r["answered"] for r in runs]
    if query_should_refuse:
        # Counterfactual: any answered = a HALLUCINATION (HARD-gate breach).
        return "FALSE-ANSWER" if any(answered) else "correct-refusal"
    # Answerable query.
    if all(answered):
        return "correct-answer"
    if not any(answered):
        return "FALSE-REFUSAL-deterministic"
    return "FALSE-REFUSAL-flaky"


async def _audit_corpus(
    query_set: Path, *, runs: int, qid: str | None, emit_json: bool
) -> list[dict[str, Any]]:
    from memex.eval.runner import _load_queries

    queries = _load_queries(query_set)
    if qid:
        queries = [q for q in queries if q.qid == qid]
    corpus = query_set.parent.name
    rows: list[dict[str, Any]] = []
    for q in queries:
        run_results = [await _run_one(q.question) for _ in range(runs)]
        verdict = _classify(q.should_refuse, run_results)
        # Gold-in-pool from the LAST run's pool (the pool the gates saw); refusals carry it.
        present, total = _gold_in_pool(q.relevant_chunk_ids, run_results[-1]["used_chunks"])
        nodes = [r["node"] for r in run_results]
        row = {
            "corpus": corpus,
            "qid": q.qid,
            "should_refuse": q.should_refuse,
            "verdict": verdict,
            "nodes": nodes,
            "answered_runs": sum(r["answered"] for r in run_results),
            "runs": runs,
            "gold_in_pool": f"{present}/{total}",
            "gold_present": present > 0 if total else None,
            "refusal_reason": run_results[-1]["refusal_reason"],
            "question": q.question,
        }
        rows.append(row)
        if emit_json:
            print(json.dumps(row), flush=True)
        else:
            flag = "  <<< FALSE REFUSAL" if verdict.startswith("FALSE-REFUSAL") else ""
            if verdict == "FALSE-ANSWER":
                flag = "  <<< HALLUCINATION (HARD-GATE BREACH)"
            print(
                f"[{corpus[:18]:<18}] [{verdict:<26}] {q.qid:<28} "
                f"gold={present}/{total} nodes={nodes}{flag}",
                flush=True,
            )
    return rows


def _print_summary(rows: list[dict[str, Any]]) -> None:
    false_ref = [r for r in rows if r["verdict"].startswith("FALSE-REFUSAL")]
    halluc = [r for r in rows if r["verdict"] == "FALSE-ANSWER"]
    answerable = sum(1 for r in rows if not r["should_refuse"])
    cf = sum(1 for r in rows if r["should_refuse"])
    print("\n=== OVERALL SUMMARY ===")
    print(f"  queries: {len(rows)}  (answerable: {answerable}, counterfactual: {cf})")
    print(
        f"  FALSE REFUSALS: {len(false_ref)}  "
        f"(deterministic: {sum(1 for r in false_ref if 'deterministic' in r['verdict'])}, "
        f"flaky: {sum(1 for r in false_ref if 'flaky' in r['verdict'])})"
    )
    if halluc:
        print(f"  *** HALLUCINATIONS (HARD-GATE BREACH): {[r['qid'] for r in halluc]} ***")
    else:
        print("  hallucinations (counterfactual answered): 0  ✓ HARD gate intact")
    for r in false_ref:
        cause = "RETRIEVAL-MISS" if r["gold_present"] is False else "GATE-over-refusal"
        # Most-common node across runs (deterministic ones agree).
        node = max(set(r["nodes"]), key=r["nodes"].count)
        print(
            f"    - {r['corpus'][:18]:<18} {r['qid']:<26} {cause:<17} "
            f"node={node:<16} gold={r['gold_in_pool']}"
        )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query_sets", type=Path, nargs="+", help="One or more queries.json paths.")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--qid", default=None, help="Audit a single query id (e.g. for deep N>=3).")
    ap.add_argument("--json", action="store_true", help="Emit per-query JSON lines.")
    args = ap.parse_args()

    from memex.cli.bootstrap import bootstrap

    bootstrap()
    all_rows: list[dict[str, Any]] = []
    for qs in args.query_sets:
        all_rows.extend(
            await _audit_corpus(qs, runs=args.runs, qid=args.qid, emit_json=args.json)
        )
    _print_summary(all_rows)


if __name__ == "__main__":
    asyncio.run(main())
