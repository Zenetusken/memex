"""Provenance-backstop L2 grader — EVERY gate in ONE script (the audit-17 v1-grader
lesson: a grader that misses a gate prints PASS on a catastrophe).

Gates (exit nonzero on any failure):
  G1 refusal_rate_on_counterfactuals == 1.0 on all 28 runs (incl. technical-guidelines,
     where tg-13 must now refuse — the backstop's kill target).
  G2 error_count == 0 on all 28 runs.
  G3 codex answer_text_correct_count >= 32 both runs.
  G4 eval-summary: 6/6 summarize/refuse correct, 0 must_not_assert leaks.
  G5 per-qid: any qid answered in the prose_sweep baseline but refused in BOTH ON-runs
     is a deterministic regression -> FAIL if its refusal reason is provenance-caused
     ("according to"); otherwise FLAG for manual attribution (borderline flips need
     N>=2 review, not silent acceptance).
  G6 corpus answered_count floor: each corpus r1/r2 >= prose_sweep baseline count - 1
     (the -1 tolerance is for known greedy-decode borderliners; every drop is listed).
"""

import json
import sys
from pathlib import Path

LADDER = Path("/tmp/prov_ladder")
BASE = Path("docs/audits/data-17-scope-calibration/raw/prose_sweep")
CORPORA = [
    "annual-report", "ccna-multidoc", "chart-types", "cr350-diagrams",
    "cr350-multidoc", "forms-w9", "french-course", "handwritten",
    "linux-fundamentals", "nist-zero-trust", "scientific-gte", "slide-decks",
    "technical-guidelines", "codex-rs-find-the-code",
]

failures: list[str] = []
flags: list[str] = []

base_answered: dict[str, bool] = {}
base_counts: dict[str, int] = {}
for c in CORPORA:
    f = BASE / f"{c}.json"
    if not f.exists():
        flags.append(f"no prose_sweep baseline for {c} (G5/G6 skipped there)")
        continue
    rep = json.load(open(f))
    base_counts[c] = rep["answered_count"]
    for q in rep["per_query"]:
        base_answered[q["qid"]] = bool(q["answered"])

print(f"{'corpus':28} {'run':3} {'cf':>5} {'err':>3} {'ans':>3} {'base':>4} {'atc':>4}")
runs: dict[tuple[str, int], dict] = {}
for c in CORPORA:
    for r in (1, 2):
        f = LADDER / f"{c}-r{r}.json"
        if not f.exists():
            failures.append(f"missing run artifact {f}")
            continue
        rep = json.load(open(f))
        runs[(c, r)] = rep
        cf = rep["refusal_rate_on_counterfactuals"]
        err = rep["error_count"]
        ans = rep["answered_count"]
        atc = rep.get("answer_text_correct_count")
        print(f"{c:28} r{r:2} {cf!s:>5} {err:>3} {ans:>3} {base_counts.get(c, '?')!s:>4} {atc!s:>4}")
        if cf is not None and cf < 1.0:
            failures.append(f"G1 cf={cf} on {c} r{r}")
        if err != 0:
            failures.append(f"G2 errors={err} on {c} r{r}")
        if c == "codex-rs-find-the-code" and (atc or 0) < 32:
            failures.append(f"G3 codex answer_text={atc} < 32 on r{r}")
        if c == "codex-rs-find-the-code" and ans < 38:
            failures.append(f"G6 codex answered {ans} < 38 (audit-15 floor; no prose_sweep baseline)")
        if c in base_counts and ans < base_counts[c] - 1:
            failures.append(f"G6 {c} r{r} answered {ans} < baseline {base_counts[c]} - 1")
        elif c in base_counts and ans < base_counts[c]:
            flags.append(f"G6 {c} r{r} answered {ans} < baseline {base_counts[c]} (within tolerance)")

# G5 per-qid deterministic regressions
for c in CORPORA:
    r1, r2 = runs.get((c, 1)), runs.get((c, 2))
    if not (r1 and r2):
        continue
    pq1 = {q["qid"]: q for q in r1["per_query"]}
    pq2 = {q["qid"]: q for q in r2["per_query"]}
    for qid, q1 in pq1.items():
        if not base_answered.get(qid, False):
            continue
        q2 = pq2.get(qid)
        if q2 is None or q1["answered"] or q2["answered"]:
            continue
        if q1.get("refusal_correct") and q2.get("refusal_correct"):
            # A CORRECT refusal of a counterfactual is never a regression — the
            # baseline answering it was the bug (tg-13, the backstop's kill target).
            continue
        reason = (q1.get("refusal_reason") or "") + (q2.get("refusal_reason") or "")
        if "according to" in reason.lower():
            failures.append(f"G5 PROVENANCE-CAUSED regression {c}/{qid}: {reason[:160]}")
        else:
            flags.append(f"G5 deterministic regression (non-provenance) {c}/{qid}: {reason[:160]}")

# G4 eval-summary — REAL field names per SummaryEvalReport (the independent review's
# B1 catch: the earlier .get() guesses missed `hallucination_count` and would have
# printed PASS on a leak catastrophe). Fail LOUD on a missing key, never .get().
sf = LADDER / "summary.json"
if not sf.exists():
    failures.append("missing eval-summary artifact")
else:
    s = json.load(open(sf))
    print(f"\neval-summary: {json.dumps({k: v for k, v in s.items() if not isinstance(v, (list, dict))})[:400]}")
    try:
        leaks = s["hallucination_count"]
        correct = s["summarize_correct_count"]
        cases = s["case_count"]
    except KeyError as e:
        failures.append(f"G4 eval-summary report missing field {e} — schema drift, inspect")
    else:
        if leaks != 0:
            failures.append(f"G4 hallucination_count={leaks} (must be 0)")
        if correct < cases:
            failures.append(f"G4 summarize_correct={correct} < case_count={cases}")

# tg-13 must be refusal_correct now (the kill target)
for r in (1, 2):
    rep = runs.get(("technical-guidelines", r))
    if rep:
        q = next((x for x in rep["per_query"] if x["qid"] == "technical-guidelines-13"), None)
        if q and (q["answered"] or not q.get("refusal_correct", False)):
            failures.append(f"KILL TARGET tg-13 not refused on r{r}: {q}")
        elif q:
            print(f"tg-13 r{r}: refused, reason={str(q.get('refusal_reason'))[:120]}")

print("\n--- FLAGS (manual attribution) ---")
for fl in flags or ["none"]:
    print(f"  {fl}")
print("\n--- FAILURES ---")
for fa in failures or ["none"]:
    print(f"  {fa}")
print(f"\nLADDER {'PASS' if not failures else 'FAIL'}")
sys.exit(0 if not failures else 1)
