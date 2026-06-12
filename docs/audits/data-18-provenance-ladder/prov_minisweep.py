"""Provenance-backstop L1 mini-sweep (audit-18 §9): the 30 benign qids LIVE under the
bge default with the backstop ON (branch default), answered-set compared per-qid to the
committed prose_sweep baseline; plus tg-13 live N=2 (must REFUSE via provenance_scope).
Gate: 0 new refusals on the 30 AND tg-13 refuses 2/2."""

import asyncio
import json
import sys
import time

import memex.agents.answering as A
from memex.cli.bootstrap import bootstrap

CORPORA = {
    "annual-report": [f"annual-report-0{i}" for i in range(1, 6)],
    "nist-zero-trust": [f"nist-zero-trust-0{i}" for i in range(1, 6)],
    "scientific-gte": [f"scientific-gte-0{i}" for i in range(1, 6)],
    "linux-fundamentals": [f"linux-fundamentals-0{i}" for i in range(1, 6)],
    "cr350-multidoc": [f"cr350-xref-0{i}" for i in range(1, 6)],
    "technical-guidelines": [f"technical-guidelines-0{i}" for i in range(1, 6)],
}
BASE = "docs/audits/data-17-scope-calibration/raw/prose_sweep"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/prov_minisweep.json"  # noqa: S108


async def main() -> None:
    bootstrap()
    baseline: dict[str, bool] = {}
    for corpus in CORPORA:
        rep = json.load(open(f"{BASE}/{corpus}.json"))
        for q in rep["per_query"]:
            baseline[q["qid"]] = bool(q.get("answered"))
    missing = [q for qids in CORPORA.values() for q in qids if q not in baseline]
    if missing:
        print(f"FATAL: baseline missing for {missing}")
        sys.exit(2)

    questions: dict[str, str] = {}
    for corpus, qids in CORPORA.items():
        qs = json.load(open(f"tests/eval-data/{corpus}/queries.json"))["queries"]
        byid = {q["qid"]: q["question"] for q in qs}
        for qid in qids:
            questions[qid] = byid[qid]
    tgqs = json.load(open("tests/eval-data/technical-guidelines/queries.json"))["queries"]
    tg13 = next(q["question"] for q in tgqs if q["qid"] == "technical-guidelines-13")

    rows, new_refusals = [], 0
    for corpus, qids in CORPORA.items():
        for qid in qids:
            t0 = time.monotonic()
            resp = await A.answer_query(questions[qid])
            regress = baseline.get(qid, False) and not resp.answered
            new_refusals += int(regress)
            rows.append(
                {"qid": qid, "answered": bool(resp.answered),
                 "baseline_answered": baseline.get(qid),
                 "refusal": (resp.refusal_reason or "")[:160],
                 "regress": regress, "s": round(time.monotonic() - t0, 1)}
            )
            json.dump(rows, open(OUT, "w"), indent=1)
            print(f"[mini] {qid:24} answered={resp.answered} base={baseline.get(qid)} "
                  f"{'REGRESS!' if regress else 'ok'} ({rows[-1]['s']}s)", flush=True)

    tg13_refused = 0
    for run in (1, 2):
        resp = await A.answer_query(tg13)
        refused = not resp.answered
        prov = "provenance" in (resp.refusal_reason or "").lower() or "according to" in (
            resp.refusal_reason or ""
        )
        tg13_refused += int(refused)
        rows.append({"qid": f"tg-13-run{run}", "answered": bool(resp.answered),
                     "refusal": (resp.refusal_reason or "")[:300], "via_provenance": prov})
        json.dump(rows, open(OUT, "w"), indent=1)
        print(f"[mini] tg-13 run{run}: answered={resp.answered} via_prov={prov} "
              f"reason={(resp.refusal_reason or '')[:140]}", flush=True)

    ok = new_refusals == 0 and tg13_refused == 2
    print(f"\nMINI-SWEEP: {new_refusals} new refusals /30 (gate 0); "
          f"tg-13 refused {tg13_refused}/2 (gate 2) -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


asyncio.run(main())
