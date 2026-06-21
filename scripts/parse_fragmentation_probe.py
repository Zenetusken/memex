"""Empirics-first probe that REFUTED the legal-statutes "parse-fragmentation
false-refusal" hypothesis (a recorded negative; the structured-workflow
empirics gate working).

For each answerable legal-statutes query that the eval REFUSED, it measures —
as OBSERVED FACT — (1) the gold chunk's rank in the k=50 candidate pool, (2) its
rank after rerank (top-5 = what the gate sees), (3) whether the answer value is
present in the gold chunk and at what character offset, vs the answer/compose
per-chunk truncation budget (~1800 chars; assess is 1200). Gold ids are read
from queries.json (never hardcoded — an earlier hardcoded version had an
off-by-one that briefly mis-attributed the cause).

Verdict (see tests/eval-data/legal-statutes/queries.json `_baseline_2026_06_20`):
NOT parse-fragmentation (the MD line/hyphen splits normalize away in the chunk
text). The 4 refusals are 3 gate over-refusals on VISIBLE top-ranked values
(ADR-0022 class) + 1 truncation-horizon (foia-08: value at offset 1854 > 1800).

Run with the eval device pin (matches the run that produced the refusals):
  MEMEX_MODELS__CO_RESIDENCE_MODE=manual MEMEX_MODELS__RERANKER_DEVICE=cpu \
  MEMEX_MODELS__EMBEDDER_DEVICE=cuda MEMEX_OBSERVABILITY__LANGFUSE_ENABLED=false \
  uv run python scripts/parse_fragmentation_probe.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from memex.cli.bootstrap import bootstrap
from memex.core.config import get_settings
from memex.index.fts_store import FTSStore
from memex.retrieve import cross_encoder_rerank, hybrid_search

QUERIES = Path("tests/eval-data/legal-statutes/queries.json")
# The answerable queries the eval refused, + the answer value to locate verbatim.
FALSE_REFUSALS = {
    "foia-05": "by an entity under Government contract",
    "foia-08": "$250",
    "foia-11": "requested 3 or more",
    "pa-12": "10 days",
}
COMPOSE_TRUNCATE = 1800  # per-chunk char budget in the answer/compose window (answering.py:1061)


async def main() -> None:
    bootstrap()
    qmap = {q["qid"]: q for q in json.loads(QUERIES.read_text())["queries"]}
    fts = await FTSStore.open(get_settings().vault_path)

    for qid, value in FALSE_REFUSALS.items():
        q = qmap[qid]
        gold = q["relevant_chunk_ids"][0]  # read from queries.json — no hardcoding
        question = q["question"]
        candidates = await hybrid_search(question, k=50)
        cand_ids = [c.chunk_id for c in candidates]
        cand_rank = cand_ids.index(gold) + 1 if gold in cand_ids else None
        reranked = await cross_encoder_rerank(question, candidates, top_k=5)
        rr_ids = [c.chunk_id for c in reranked]
        rr_rank = rr_ids.index(gold) + 1 if gold in rr_ids else None

        chunk = (await fts.chunks_by_ids([gold]))[0]
        off = chunk.text.lower().find(value.lower())
        print(f"\n===== {qid} =====")
        print(f"  Q: {question}")
        print(f"  gold chunk: {gold.split('#')[1]}  (len={len(chunk.text)})")
        print(f"  RETRIEVAL rank (k=50): {cand_rank}    RERANK rank (top-5): {rr_rank}")
        if off >= 0:
            past = off > COMPOSE_TRUNCATE
            print(f"  value '{value}': present, offset={off}  (past {COMPOSE_TRUNCATE}-char budget? {past})")
        else:
            print(f"  value '{value}': NOT in gold chunk text")


if __name__ == "__main__":
    asyncio.run(main())
