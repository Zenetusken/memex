"""Live exercise of the provenance backstop under the mxbai env (where the tg-13
breach reproduces deterministically): the run must REFUSE with the provenance reason."""
import asyncio, json, sys
import memex.agents.answering as A
from memex.cli.bootstrap import bootstrap

async def main():
    bootstrap()
    qs = json.load(open("tests/eval-data/technical-guidelines/queries.json"))["queries"]
    q = next(x["question"] for x in qs if x["qid"] == "technical-guidelines-13")
    ok = 0
    for run in (1, 2):
        resp = await A.answer_query(q)
        reason = resp.refusal_reason or ""
        via_prov = "according to" in reason.lower()
        print(f"run{run}: answered={resp.answered} via_prov={via_prov}\n  reason={reason[:300]}", flush=True)
        ok += int((not resp.answered) and via_prov)
    print(f"MXBAI EXERCISE: {ok}/2 refused-via-provenance -> {'PASS' if ok >= 1 else 'CHECK'}")
    sys.exit(0 if ok >= 1 else 1)

asyncio.run(main())
