"""v6 over-refusal mini-sweep (audit-17 discipline): run every should_refuse=False
(ANS) query from the value-heavy corpora the whole-vs-part clause could over-refuse,
under the ACTIVE answer prompt, N=2. Emits per-qid answered. Run once pinned v5,
once active v6; diff = new false refusals (v5 answered AND v6 refused).

Usage: uv run python /tmp/v6_minisweep.py <out.json> [runs]
"""
import asyncio, json, sys
import memex.agents.answering as A
from memex.cli.bootstrap import bootstrap
from memex.prompts.loader import load_prompt_spec

CORPORA = ["annual-report", "scientific-gte", "slide-decks", "nist-zero-trust",
           "technical-guidelines"]

async def main():
    bootstrap()
    ver = load_prompt_spec("answer").version
    out_path = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    cases = []
    for c in CORPORA:
        for q in json.load(open(f"tests/eval-data/{c}/queries.json"))["queries"]:
            if not q.get("should_refuse", False):
                cases.append((c, q["qid"], q["question"]))
    print(f"=== mini-sweep: answer@{ver} | {len(cases)} ANS queries x{runs} ===", flush=True)
    result = {"version": ver, "cases": {}}
    for run in range(1, runs + 1):
        for c, qid, question in cases:
            resp = await A.answer_query(question)
            result["cases"].setdefault(qid, {"corpus": c, "answered": []})
            result["cases"][qid]["answered"].append(bool(resp.answered))
            flag = "" if resp.answered else "  <-- REFUSED"
            print(f"r{run} {qid:24} answered={resp.answered}{flag}", flush=True)
        print("---", flush=True)
    # a query is "answered" if answered in EVERY run (conservative)
    answered_all = {qid for qid, d in result["cases"].items() if all(d["answered"])}
    refused_any = {qid for qid, d in result["cases"].items() if not all(d["answered"])}
    result["answered_all"] = sorted(answered_all)
    result["refused_any"] = sorted(refused_any)
    json.dump(result, open(out_path, "w"), indent=1)
    print(f"=== answer@{ver}: {len(answered_all)}/{len(cases)} answered-every-run; "
          f"refused-any-run: {sorted(refused_any)} ===")
    print(f"-> {out_path}")

asyncio.run(main())
