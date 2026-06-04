#!/usr/bin/env python
"""Author + VERIFY the grounded multi-turn chat eval corpus (`tests/eval-data/chat-multiturn/`).

The chat eval (`memex eval-chat`, `eval/runner.py::run_chat_eval`) scores, per case,
`gold_chunk_recall(hybrid_search(rewritten_follow_up)[:k], relevant_chunk_ids, k)` — a
RETRIEVAL-isolated measure of whether the control-layer query rewrite resolves a
follow-up's referents well enough to surface the gold chunks. This tool helps build that
corpus against the LIVE vault, using the SAME `hybrid_search` path the eval uses (NOT
`memex search`, which adds a cross-encoder rerank the eval metric never sees).

Three subcommands:

  search "<query>" [--k 50] [--doc <id-substr>]
      Print the raw `hybrid_search` ranking (rank · score · chunk_id · title › section ·
      snippet). Use it to RESOLVE gold: run the ideal standalone query, read the snippets,
      copy the chunk_id(s) that actually hold the answer.

  recall "<query>" --gold a,b [--k 10,20,30,50]
      For one query, the rank of each gold id + recall@k at several k. Confirms a gold id
      is retrievable for a given query and HOW DEEP (the margin).

  gap --ideal "<ideal standalone query>" --raw "<bare follow-up>" --gold a,b [--k ...]
      THE two-sided verification (the corpus's whole quality bar): a case discriminates
      the rewrite ONLY if, at its chosen k, the IDEAL query hits gold (recall=1.0, with
      margin) while the RAW follow-up MISSES (recall<1). This prints both recalls per k
      and flags the discriminating k's. Pick the tightest k where ideal=1.0 with headroom.

Needs the live embedder (hybrid_search); the daemon need NOT be up (no rewrite here — you
supply the ideal query yourself). Pin the embedder for a stable read, e.g.:
    MEMEX_MODELS__CO_RESIDENCE_MODE=manual uv run python scripts/chat_eval_author.py gap ...
"""

from __future__ import annotations

import argparse
import asyncio

from memex.cli.bootstrap import bootstrap
from memex.core.types import Chunk
from memex.eval.scoring import gold_chunk_recall
from memex.retrieve import hybrid_search


def _parse_ks(spec: str) -> list[int]:
    return sorted({int(x) for x in spec.split(",") if x.strip()})


def _snippet(text: str, n: int = 130) -> str:
    flat = " ".join(text.split())
    return flat[:n] + ("…" if len(flat) > n else "")


def _section(chunk: Chunk) -> str:
    return " › ".join(chunk.heading_path[-2:]) if chunk.heading_path else "—"


def _ranks(ids: list[str], gold: list[str]) -> dict[str, int | None]:
    """1-based rank of each gold id in `ids` (None = not retrieved within `len(ids)`)."""
    pos = {cid: i + 1 for i, cid in enumerate(ids)}
    return {g: pos.get(g) for g in gold}


async def cmd_search(args: argparse.Namespace) -> int:
    bootstrap()
    chunks = await hybrid_search(args.query, k=args.k)
    if args.doc:
        chunks = [c for c in chunks if args.doc in c.document_id]
    print(f"query: {args.query!r}  (k={args.k}; {len(chunks)} shown"
          + (f"; filtered to {args.doc!r}" if args.doc else "") + ")")
    for i, c in enumerate(chunks, 1):
        print(f"  {i:>3}. {c.score:6.3f}  {c.chunk_id}")
        print(f"        [{c.document_title} | {_section(c)}]")
        print(f"        {_snippet(c.text)}")
    return 0


async def cmd_recall(args: argparse.Namespace) -> int:
    bootstrap()
    gold = [g.strip() for g in args.gold.split(",") if g.strip()]
    ks = _parse_ks(args.k)
    ids = [c.chunk_id for c in await hybrid_search(args.query, k=max(ks))]
    print(f"query: {args.query!r}   gold={gold}")
    for g, r in _ranks(ids, gold).items():
        print(f"  gold {g}: rank {r if r is not None else f'MISS (>{max(ks)})'}")
    for k in ks:
        print(f"  recall@{k:>3} = {gold_chunk_recall(ids, gold, k):.3f}")
    return 0


async def cmd_gap(args: argparse.Namespace) -> int:
    bootstrap()
    gold = [g.strip() for g in args.gold.split(",") if g.strip()]
    ks = _parse_ks(args.k)
    maxk = max(ks)
    ideal_ids = [c.chunk_id for c in await hybrid_search(args.ideal, k=maxk)]
    raw_ids = [c.chunk_id for c in await hybrid_search(args.raw, k=maxk)]
    print(f"ideal: {args.ideal!r}")
    print(f"raw  : {args.raw!r}")
    print(f"gold : {gold}")
    print(f"  {'k':>4}  {'ideal':>7}  {'raw':>7}   verdict")
    for k in ks:
        ri = gold_chunk_recall(ideal_ids, gold, k)
        rr = gold_chunk_recall(raw_ids, gold, k)
        if ri < 1.0:
            verdict = "ideal<1 — gold not reachable even rewritten (raise k or re-pick gold)"
        elif rr < 1.0:
            verdict = "DISCRIMINATING (ideal=1.0, raw<1.0)"
        else:
            verdict = "no gap (raw already hits — too lenient at this k)"
        print(f"  {k:>4}  {ri:>7.3f}  {rr:>7.3f}   {verdict}")
    print("  ideal gold ranks:", _ranks(ideal_ids, gold))
    print("  raw   gold ranks:", _ranks(raw_ids, gold))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Author + verify the chat-multiturn eval corpus.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("search", help="raw hybrid_search ranking (resolve gold + eyeball)")
    ps.add_argument("query")
    ps.add_argument("--k", type=int, default=50)
    ps.add_argument("--doc", default="", help="filter to a document_id substring")
    ps.set_defaults(fn=cmd_search)

    pr = sub.add_parser("recall", help="recall@k + gold ranks for one query")
    pr.add_argument("query")
    pr.add_argument("--gold", required=True, help="comma-separated gold chunk_ids")
    pr.add_argument("--k", default="10,20,30,50")
    pr.set_defaults(fn=cmd_recall)

    pg = sub.add_parser("gap", help="two-sided verification: ideal vs raw recall@k")
    pg.add_argument("--ideal", required=True, help="the ideal self-contained standalone query")
    pg.add_argument("--raw", required=True, help="the bare follow-up (unresolved referents)")
    pg.add_argument("--gold", required=True, help="comma-separated gold chunk_ids")
    pg.add_argument("--k", default="10,20,30,50")
    pg.set_defaults(fn=cmd_gap)

    args = p.parse_args()
    return asyncio.run(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
