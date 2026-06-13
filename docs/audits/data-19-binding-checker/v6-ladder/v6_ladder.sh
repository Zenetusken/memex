#!/usr/bin/env bash
# answer@v6 full ladder (audit-19 generation-faithfulness lever):
#   arm A = v6 / bge default  (the SHIP gate — must not regress vs v5)
#   arm B = v5 / bge default  (same-session baseline for the A/B)
#   arm C = v6 / mxbai env     (the fix: ar-12 refuses; + no mxbai regression)
# 14 corpora x N=2 each + eval-summary once (untouched by the answer prompt).
set -u
export MEMEX_MODELS__CO_RESIDENCE_MODE=manual
export MEMEX_MODELS__EMBEDDER_DEVICE=cpu
export MEMEX_MODELS__RERANKER_DEVICE=cpu
OUT=/tmp/v6_ladder
mkdir -p "$OUT"
CORPORA="annual-report ccna-multidoc chart-types cr350-diagrams cr350-multidoc forms-w9 french-course handwritten linux-fundamentals nist-zero-trust scientific-gte slide-decks technical-guidelines codex-rs-find-the-code"

run_arm () {
  arm="$1"; pin="$2"; reranker="$3"
  echo "[arm $arm] pin=$pin reranker=$reranker $(date +%H:%M:%S)"
  for run in 1 2; do
    for c in $CORPORA; do
      f="$OUT/$arm-$c-r$run.json"
      if [ -s "$f" ]; then echo "[skip] $arm $c r$run"; continue; fi
      MEMEX_PROMPTS__PIN__ANSWER="$pin" MEMEX_MODELS__RERANKER="$reranker" \
        uv run memex eval "tests/eval-data/$c/queries.json" \
        > "$f" 2> "$OUT/$arm-$c-r$run.err"
      echo "[done] $arm $c r$run rc=$?"
    done
  done
}

# arm A: v6/bge — the SHIP gate (compare to the data-18 v5/bge baseline; the
# same-session v5-vs-v6 A/B for the 5 value-heavy corpora is the mini-sweep)
run_arm A "v6" "BAAI/bge-reranker-v2-m3"
# arm C: v6/mxbai — the fix (ar-12 refuses) + no mxbai regression
run_arm C "v6" "mixedbread-ai/mxbai-rerank-base-v2"

# eval-summary once (answer-prompt-independent; confirm 6/6)
if [ ! -s "$OUT/summary.json" ]; then
  echo "[ladder] eval-summary $(date +%H:%M:%S)"
  uv run memex eval-summary tests/eval-data/summary/queries.json \
    > "$OUT/summary.json" 2> "$OUT/summary.err"
  echo "[done] eval-summary rc=$?"
fi
echo "V6 LADDER COMPLETE $(date +%H:%M:%S)"
