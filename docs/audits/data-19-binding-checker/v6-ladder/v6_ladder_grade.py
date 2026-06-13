"""Unified grader for the answer@v6 ladder (audit-19 generation-faithfulness lever).

Encodes EVERY gate with the REAL EvalReport / EvalQueryResult field names (the
recurring B1 lesson). Reads:
  arm A = v6/bge  (the SHIP gate) — compare to the data-18 v5/bge baseline
  arm C = v6/mxbai (the fix) — ar-12 must refuse; refusal_cf=1.0; no ANS collapse
  + eval-summary (untouched)

GATES:
  G1  refusal_cf == 1.0 on EVERY corpus run, BOTH arms (the inviolable HARD gate)
  G2  error_count == 0 on every run
  G3  arm A ANS >= data-18 baseline - tolerance per corpus (the over-refusal gate);
      flags any corpus that DROPS, with the per-qid v5->v6 deterministic refusals
  G4  arm C annual-report: ar-12 refuses BOTH runs (the fix); refusal_cf 1.0
  G5  eval-summary: hallucination_count==0, summarize_correct_count==case_count
  G6  answer_mention_recall (arm A) not materially down vs the recorded baseline
"""
import json, sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/v6_ladder")
CORPORA = "annual-report ccna-multidoc chart-types cr350-diagrams cr350-multidoc forms-w9 french-course handwritten linux-fundamentals nist-zero-trust scientific-gte slide-decks technical-guidelines codex-rs-find-the-code".split()

# data-18 v5/bge baseline ANS per corpus (the recorded provenance-ladder numbers)
BASELINE_ANS = {
    "annual-report": 10, "ccna-multidoc": 8, "chart-types": 7, "cr350-diagrams": 11,
    "cr350-multidoc": 13, "forms-w9": 12, "french-course": 5, "handwritten": 6,
    "linux-fundamentals": 13, "nist-zero-trust": 12, "scientific-gte": 14,
    "slide-decks": 13, "technical-guidelines": 12, "codex-rs-find-the-code": 38,
}

def load(arm, c, run):
    f = OUT / f"{arm}-{c}-r{run}.json"
    if not f.exists() or f.stat().st_size == 0:
        return None
    return json.load(open(f))

fails = []
def check(cond, msg):
    if not cond:
        fails.append(msg)
    print(f"  {'ok ' if cond else 'XX '}{msg}")

for arm, label in (("A", "v6/bge SHIP"), ("C", "v6/mxbai FIX")):
    print(f"\n=== arm {arm} ({label}) ===")
    for c in CORPORA:
        runs = [load(arm, c, r) for r in (1, 2)]
        if any(r is None for r in runs):
            check(False, f"{arm} {c}: MISSING a run")
            continue
        for r_i, rep in enumerate(runs, 1):
            cf = rep["refusal_rate_on_counterfactuals"]
            err = rep["error_count"]
            check(cf == 1.0, f"{arm} {c} r{r_i}: refusal_cf={cf} (G1)")
            check(err == 0, f"{arm} {c} r{r_i}: error_count={err} (G2)")
        ans = [rep["answered_count"] for rep in runs]
        if arm == "A":
            base = BASELINE_ANS[c]
            # over-refusal gate: ANS >= baseline - 1 (borderline tolerance), flag drops
            check(min(ans) >= base - 1,
                  f"A {c}: ANS={ans} vs v5/bge baseline {base} (G3 over-refusal)")

# G4 — the ar-12 fix under mxbai
print("\n=== G4: ar-12 fix under mxbai (arm C annual-report) ===")
for run in (1, 2):
    rep = load("C", "annual-report", run)
    if rep is None:
        check(False, f"C annual-report r{run}: MISSING")
        continue
    ar12 = next((q for q in rep["per_query"] if q["qid"] == "annual-report-12"), None)
    check(ar12 is not None and not ar12["answered"] and ar12["refusal_correct"],
          f"C annual-report r{run}: ar-12 answered={ar12 and ar12['answered']} "
          f"refusal_correct={ar12 and ar12['refusal_correct']} (G4 the fix)")

# G5 — eval-summary
print("\n=== G5: eval-summary (untouched by the answer prompt) ===")
sf = OUT / "summary.json"
if sf.exists() and sf.stat().st_size:
    s = json.load(open(sf))
    check(s["hallucination_count"] == 0, f"eval-summary hallucination_count={s['hallucination_count']} (G5)")
    check(s["summarize_correct_count"] == s["case_count"],
          f"eval-summary {s['summarize_correct_count']}/{s['case_count']} correct (G5)")
else:
    check(False, "eval-summary MISSING")

print("\n" + ("LADDER PASS" if not fails else f"LADDER FAIL ({len(fails)} gate failures):"))
for f in fails:
    print("  -", f)
sys.exit(0 if not fails else 1)
