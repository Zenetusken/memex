"""Offline pair check for answer@v6 (the aggregate-vs-part scope lever).

Runs the razor cases N times and reports answered + headline:
- ar-12  should_refuse  (Graphics SEGMENT gross margin — only consolidated present) = the TARGET
- ar-05  ANS            (consolidated gross margin 71.1% — razor twin, MUST answer)
- ar-03  ANS            (Data Center revenue $193.737B — per-part value IS stated, MUST answer)
- ar-01  ANS            (total revenue — MUST answer)
- ar-11  should_refuse  (Automotive operating income — segment absent: another whole-vs-part CF)

Run under the mxbai breach env (where ar-12 breaches on v5) AND/OR bge default.
The active answer version is whatever the loader selects (v6) unless pinned.
"""
import asyncio, json, sys
import memex.agents.answering as A
from memex.cli.bootstrap import bootstrap
from memex.prompts.loader import load_prompt_spec

WANT = {
    "annual-report-12": ("REFUSE", "Graphics segment gross margin"),
    "annual-report-05": ("ANSWER", "consolidated gross margin"),
    "annual-report-03": ("ANSWER", "Data Center revenue"),
    "annual-report-01": ("ANSWER", "total revenue"),
    "annual-report-11": ("REFUSE", "Automotive operating income"),
}

async def main():
    bootstrap()
    ver = load_prompt_spec("answer").version
    print(f"=== answer prompt active version: {ver} ===", flush=True)
    qs = {q["qid"]: q["question"]
          for q in json.load(open("tests/eval-data/annual-report/queries.json"))["queries"]}
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    tally = {qid: [] for qid in WANT}
    for run in range(1, runs + 1):
        for qid, (want, label) in WANT.items():
            resp = await A.answer_query(qs[qid])
            got = "ANSWER" if resp.answered else "REFUSE"
            ok = "ok " if got == want else "XX "
            tally[qid].append(got)
            head = (resp.summary or resp.refusal_reason or "")[:150]
            print(f"r{run} {ok}{qid:20} want={want:6} got={got:6} | {head}", flush=True)
        print("---", flush=True)
    print("=== SUMMARY ===")
    allok = True
    for qid, (want, label) in WANT.items():
        gots = tally[qid]
        consistent = all(g == want for g in gots)
        allok &= consistent
        print(f"{qid:20} want={want:6} runs={gots} {'PASS' if consistent else 'FAIL'}  ({label})")
    print("VERDICT:", "PASS" if allok else "FAIL")
    sys.exit(0 if allok else 1)

asyncio.run(main())
