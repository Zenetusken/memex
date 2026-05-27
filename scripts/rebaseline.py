#!/usr/bin/env python
"""Re-baseline sweep — runs the answer-eval over many corpora, ONE FRESH PROCESS
PER CORPUS, and checks the HARD gates (refusal_rate_on_counterfactuals == 1.0,
hallucinations == 0).

Why a subprocess per corpus: a single long-lived process that loops every
corpus *fragments* the 12 GB GPU and eventually CUDA-OOMs (vLLM ~8.75 GB +
embedder + reranker + desktop ≈ device-full) — observed at ~corpus 9 of 11
during the 2026-05-26 Qwen3-VL re-baseline. A fresh process per corpus resets
the CUDA context each time, so the sweep can't accumulate into an OOM. The
HARD gates held on every corpus that ran; this is infra hardening, not a
correctness fix. See docs/ROADMAP.md (P0 priorities, item 8) +
docs/specs/vlm-vllm-serving.md.

VLM-dependent corpora run N=3 by default (the agent's BF16/AWQ greedy decode
flips borderline counterfactuals run-to-run — measure multi-run); others N=1.

Usage:
    uv run python scripts/rebaseline.py                 # all corpora, default N
    uv run python scripts/rebaseline.py cr350-diagrams  # one corpus
    uv run python scripts/rebaseline.py --list          # plan only, no GPU
    uv run python scripts/rebaseline.py --runs 1        # force N=1 everywhere

Run it with the eval env on a 12 GB rig:
    MEMEX_RERANK_BATCH_SIZE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MEMEX_OBSERVABILITY__LANGFUSE_ENABLED=false uv run python scripts/rebaseline.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DATA = ROOT / "tests" / "eval-data"

# VLM-dependent corpora → N=3 (borderline-cf non-determinism); others → N=1.
MULTI_RUN: dict[str, int] = {
    "cr350-diagrams": 3,
    "cr350-multidoc": 3,
    "ccna-multidoc": 3,
}


def _discover() -> list[str]:
    """Corpus names that have a queries.json, sorted."""
    return sorted(p.parent.name for p in EVAL_DATA.glob("*/queries.json"))


def _run_one(corpus: str) -> int:
    """Run the eval over ONE corpus IN THIS PROCESS; print a HARD-gate line.

    Returns 0 if the HARD gates hold (refusal_cf == 1.0 and 0 hallucinations),
    1 otherwise. Invoked as a subprocess by the parent sweep (one per corpus).
    """
    import asyncio

    from memex.cli.bootstrap import bootstrap
    from memex.eval.runner import run_eval

    bootstrap()
    qpath = EVAL_DATA / corpus / "queries.json"
    labels = {
        q["qid"]: q.get("should_refuse", False) for q in json.loads(qpath.read_text())["queries"]
    }
    report = asyncio.run(run_eval(qpath))
    halluc = [r.qid for r in report.per_query if labels.get(r.qid, False) and r.answered]
    gates_ok = report.refusal_rate_on_counterfactuals == 1.0 and not halluc
    print(
        f"  {corpus:24} ANS={report.answered_count:>3} REF={report.refused_count:>3} "
        f"refusal_cf={report.refusal_rate_on_counterfactuals:.3f} "
        f"halluc={len(halluc)} {halluc} -> {'PASS' if gates_ok else 'FAIL'}",
        flush=True,
    )
    return 0 if gates_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "corpora", nargs="*", help="corpus names (default: all under tests/eval-data/)"
    )
    parser.add_argument(
        "--one", metavar="CORPUS", help="internal: run a single corpus in THIS process"
    )
    parser.add_argument(
        "--list", action="store_true", help="print the plan (corpus × N) and exit — no GPU"
    )
    parser.add_argument("--runs", type=int, default=None, help="force N runs for every corpus")
    args = parser.parse_args()

    if args.one:
        return _run_one(args.one)

    corpora = args.corpora or _discover()
    missing = [c for c in corpora if not (EVAL_DATA / c / "queries.json").exists()]
    if missing:
        print(f"unknown corpora (no queries.json): {missing}", file=sys.stderr)
        return 2
    plan = [(c, args.runs if args.runs is not None else MULTI_RUN.get(c, 1)) for c in corpora]

    if args.list:
        print("re-baseline plan (one fresh process per run):")
        for corpus, n in plan:
            print(f"  {corpus:24} N={n}")
        return 0

    print(f"=== re-baseline: {len(plan)} corpora, one fresh process per run ===", flush=True)
    any_fail = False
    for corpus, n in plan:
        for i in range(n):
            label = f"{corpus} (run {i + 1}/{n})" if n > 1 else corpus
            print(f">>> {label}", flush=True)
            # Fresh subprocess per (corpus, run): resets the CUDA context so a long
            # sweep can't fragment into a co-residence OOM (see module docstring).
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--one", corpus],
                cwd=str(ROOT),
                env={**os.environ},
                check=False,
            )
            if proc.returncode != 0:
                any_fail = True
    print(
        f"=== DONE — HARD GATES: {'FAIL (see above)' if any_fail else 'ALL HELD'} ===", flush=True
    )
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
