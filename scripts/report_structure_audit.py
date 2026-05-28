"""Report-STRUCTURE validator (ADR-0010 deck granularity).

`memex eval-summary` gates FAITHFULNESS (mention-recall + must_not_assert + summarize/
refuse). This standalone validator measures the STRUCTURE/COHERENCE quality a granularity
change targets, so a candidate knob setting can be compared against a baseline and checked
for no-regression. It runs `summarize_document(detail="report")` over a deck corpus and, per
doc, records:

  paragraphs / section_summaries   — granularity (the latter is what the packing controls)
  confidence_overall / _min        — faithfulness (report_confidence; hybrid emb+lexical)
  distinctness                     — 1 - mean pairwise content-Jaccard between paragraphs;
                                     the ANTI-REPETITION metric (low = paragraphs repeat)
  unique_openers                   — distinct first-8-content-words / paragraphs
  violations                       — must_not_assert leaks (the no-hallucination HARD gate)

The active knob settings (report_pack_chars / report_coalesce_target) are captured in the
output so a sweep is self-documenting. Tune via env, e.g.:

  MEMEX_AGENTS__REPORT_PACK_CHARS=6000 MEMEX_AGENTS__REPORT_COALESCE_TARGET=3 \
    MEMEX_MODELS__RERANKER_DEVICE=cpu \
    uv run python scripts/report_structure_audit.py --label A2-pack6k-coal3 --out /tmp/run.json

Usage:
  uv run python scripts/report_structure_audit.py [config.json] [--label NAME] [--out FILE]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from itertools import combinations
from pathlib import Path

from memex.agents.document_summarizer import summarize_document
from memex.cli.bootstrap import bootstrap
from memex.core.config import get_settings

_STOP = frozenset(
    "the a an of to in and or for on at by is are was were be been it its this that these those "
    "with as from into than then so such not no but if which who whose their there here have has "
    "had do does did can could should would may might will we you they it".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in _STOP]


def _distinctness(paras: list[str]) -> float:
    """1 - mean pairwise content-Jaccard. 1.0 = fully distinct; low = paragraphs repeat."""
    toks = [set(_tokens(p)) for p in paras]
    pairs = list(combinations(range(len(toks)), 2))
    if not pairs:
        return 1.0
    sims = []
    for i, j in pairs:
        union = toks[i] | toks[j]
        sims.append((len(toks[i] & toks[j]) / len(union)) if union else 0.0)
    return round(1.0 - sum(sims) / len(sims), 3)


def _unique_openers(paras: list[str], n: int = 8) -> float:
    openers = [" ".join(_tokens(p)[:n]) for p in paras if p.strip()]
    return round(len(set(openers)) / len(openers), 3) if openers else 1.0


async def _audit(case: dict) -> dict:
    try:
        resp = await summarize_document(case["doc_id"], detail="report")
    except Exception as e:  # a validator records failures, never aborts the sweep (CancelledError is BaseException, uncaught)
        return {"doc_id": case["doc_id"], "error": str(e)[:160]}
    summary = resp.summary or ""
    paras = [p for p in summary.split("\n\n") if p.strip()]
    rc = resp.report_confidence
    low = summary.lower()
    violations = [m for m in case.get("must_not_assert", []) if m.lower() in low]
    return {
        "doc_id": case["doc_id"],
        "answered": resp.answered,
        "paragraphs": len(paras),
        "section_summaries": len(resp.sections),
        "confidence_overall": (rc.overall if rc else None),
        "confidence_min": (min(rc.per_paragraph) if rc and rc.per_paragraph else None),
        "distinctness": _distinctness(paras),
        "unique_openers": _unique_openers(paras),
        "violations": violations,
    }


async def _run(config: Path, label: str) -> dict:
    bootstrap()
    cfg = json.loads(config.read_text(encoding="utf-8"))
    rows = [await _audit(c) for c in cfg["cases"]]
    ok = [r for r in rows if r.get("answered") and "error" not in r]

    def _mean(key: str) -> float:
        vals = [r[key] for r in ok if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    s = get_settings().agents
    return {
        "label": label,
        "config": {"report_pack_chars": s.report_pack_chars, "report_coalesce_target": s.report_coalesce_target},
        "cases": rows,
        "aggregate": {
            "docs": len(rows),
            "answered": len(ok),
            "mean_paragraphs": _mean("paragraphs"),
            "mean_section_summaries": _mean("section_summaries"),
            "mean_confidence": _mean("confidence_overall"),
            "mean_distinctness": _mean("distinctness"),
            "mean_unique_openers": _mean("unique_openers"),
            "total_violations": sum(len(r.get("violations", [])) for r in rows),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="tests/eval-data/report-structure/decks.json")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = asyncio.run(_run(Path(args.config), args.label))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["aggregate"], indent=2))
    agg, cfg = out["aggregate"], out["config"]
    print(f"\n=== [{out['label']}]  pack={cfg['report_pack_chars']} coalesce={cfg['report_coalesce_target']} ===", file=sys.stderr)
    for r in out["cases"]:
        if "error" in r:
            print(f"  {r['doc_id'][:30]:30} ERROR {r['error']}", file=sys.stderr)
            continue
        print(
            f"  {r['doc_id'][:30]:30} para={r['paragraphs']:>2} sec={r['section_summaries']:>2} "
            f"conf={r['confidence_overall']} distinct={r['distinctness']} "
            f"openers={r['unique_openers']} viol={r['violations']}",
            file=sys.stderr,
        )
    print(
        f"  AGG para={agg['mean_paragraphs']} sec={agg['mean_section_summaries']} "
        f"conf={agg['mean_confidence']} distinct={agg['mean_distinctness']} "
        f"openers={agg['mean_unique_openers']} violations={agg['total_violations']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
