"""Capture the LIVE grounded answers for an eval query set's ANS queries.

The answer-text-correctness authoring tool (audit-14 `answer_must_mention` →
`answer_mention_recall`). `EvalReport.per_query` records the METRIC but NOT the raw
answer text, so to author `answer_must_mention` slots that the CORRECT answer
actually satisfies (phrasing varies run-to-run on the greedy 4B), author against
reality: this dumps, per ANS query, exactly what `eval/runner.run_eval` grades —
`response.summary` + each `claim.claim` — alongside the human `_expected_answer`.

    MEMEX_MODELS__CO_RESIDENCE_MODE=manual MEMEX_MODELS__RERANKER_DEVICE=cpu \\
        uv run python scripts/capture_eval_answers.py tests/eval-data/nist-zero-trust/queries.json

Pins the device per the CLAUDE.md eval rule (answer TEXT is orchestrator-driven;
reranker order is byte-identical CPU/GPU, so CPU rerank is VRAM-safe + correctness
-neutral). Read-only: it never writes the query set.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


async def _run(query_set: Path) -> int:
    from memex.agents.answering import answer_query
    from memex.cli.bootstrap import bootstrap

    bootstrap()
    data = json.loads(query_set.read_text())
    queries = data.get("queries", [])
    ans = [q for q in queries if not q.get("should_refuse", False)]
    out: list[dict[str, object]] = []
    for q in ans:
        try:
            resp = await answer_query(q["question"])
        except Exception as e:  # per-query resilience (mirrors eval/runner.run_eval):
            # one query's failure (e.g. a context-overflow on a big deck) must not abort
            # the whole corpus's capture. CancelledError/KeyboardInterrupt/SystemExit are
            # BaseException, so this `except Exception` does not swallow cancellation.
            out.append(
                {
                    "qid": q["qid"],
                    "question": q["question"],
                    "_expected_answer": q.get("_expected_answer", ""),
                    "answered": False,
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                    "refusal_reason": "",
                    "summary": "",
                    "claims": [],
                }
            )
            continue
        # Mirror run_eval's graded text exactly: summary + each claim text.
        out.append(
            {
                "qid": q["qid"],
                "question": q["question"],
                "_expected_answer": q.get("_expected_answer", ""),
                "answered": resp.answered,
                "refusal_reason": (resp.refusal_reason or "") if not resp.answered else "",
                "summary": resp.summary or "",
                "claims": [c.claim for c in resp.claims],
            }
        )
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: capture_eval_answers.py <query_set.json>\n")
        return 2
    return asyncio.run(_run(Path(sys.argv[1])))


if __name__ == "__main__":
    raise SystemExit(main())
