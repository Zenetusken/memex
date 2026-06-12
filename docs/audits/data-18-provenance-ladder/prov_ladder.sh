#!/usr/bin/env bash
# Provenance-backstop L2 full ladder (audit-18 §9): 14 corpora x N=2 + eval-summary,
# flag ON (branch default), device-pinned CPU embedder/reranker, daemon serving the 4B.
set -u
export MEMEX_MODELS__CO_RESIDENCE_MODE=manual
export MEMEX_MODELS__EMBEDDER_DEVICE=cpu
export MEMEX_MODELS__RERANKER_DEVICE=cpu
OUT=/tmp/prov_ladder
mkdir -p "$OUT"
CORPORA="annual-report ccna-multidoc chart-types cr350-diagrams cr350-multidoc forms-w9 french-course handwritten linux-fundamentals nist-zero-trust scientific-gte slide-decks technical-guidelines codex-rs-find-the-code"
for run in 1 2; do
  for c in $CORPORA; do
    if [ -s "$OUT/$c-r$run.json" ]; then echo "[skip] $c r$run"; continue; fi
    echo "[ladder] $c r$run $(date +%H:%M:%S)"
    uv run memex eval "tests/eval-data/$c/queries.json" \
      > "$OUT/$c-r$run.json" 2> "$OUT/$c-r$run.err"
    echo "[done] $c r$run rc=$?"
  done
done
echo "[ladder] eval-summary $(date +%H:%M:%S)"
uv run memex eval-summary tests/eval-data/summary/queries.json \
  > "$OUT/summary.json" 2> "$OUT/summary.err"
echo "[done] eval-summary rc=$?"
echo "LADDER COMPLETE $(date +%H:%M:%S)"
