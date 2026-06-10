"""Parent-context (neighbor-window) headroom probe (MEASURE-FIRST, read-only).

Stage 1 of the Theory-A validation (docs/audits/16-parent-child-retrieval-audit.md):
measures whether expanding the answer window with READING-ORDER ADJACENT chunks of
the rerank winners would put GOLD evidence in front of the answer model that the
production top-k window misses. This is the additive, HARD-gate-safe reshaping of
the "parent-document retriever" idea (the augment_companion pattern, ADR-0018) —
NOT the text-swap variant, which is rejected in the audit. This script changes
NOTHING in production.

Run against the main vault (the eval corpora live in it), retrieval models on CPU:

    MEMEX_MODELS__CO_RESIDENCE_MODE=manual \\
    MEMEX_MODELS__EMBEDDER_DEVICE=cpu MEMEX_MODELS__RERANKER_DEVICE=cpu \\
    uv run python scripts/parent_context_probe.py \\
        tests/eval-data/slide-decks/queries.json \\
        tests/eval-data/annual-report/queries.json \\
        tests/eval-data/linux-fundamentals/queries.json

Per answerable query (should_refuse=false, non-empty relevant_chunk_ids):
  - pool     : production hybrid_search(question, k=POOL_K) (RRF-fused dense+BM25)
  - winners  : cross_encoder_rerank(pool, top_k=TOP_K) — the production answer
               window (or pool[:TOP_K] under --no-rerank)
  - expansion: for the first EXPAND_WINNERS winners, the ±NEIGHBORS adjacent
               chunks of the SAME document in reading order (char_start), capped
               at ADD_CAP total, de-duplicated against the winners

Each query is classified into exactly one bucket:
  gold_in_winners       — production window already carries gold (no headroom)
  HEADROOM              — gold NOT in winners but IS in the neighbor expansion
                          (the only bucket the variant can convert REF→ANS)
  gold_below_cut        — gold in the k-pool but below the rerank cut AND not
                          adjacent to a winner (a rerank-depth issue, NOT a
                          parent-context issue — reported to keep us honest)
  gold_missed           — gold not in pool and not adjacent (retrieval miss)

Secondary (non-exclusive): adjacency_extension — gold already in winners AND the
expansion adds FURTHER gold chunks (multi-chunk labels; answer-completeness, not
REF→ANS headroom).

GO/NO-GO (audit §5): HEADROOM > 10% of answerable queries on any corpus ⇒ GO for
the Stage-2 live `memex eval` A/B behind a default-OFF flag; otherwise record the
negative result and close.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from memex.cli.bootstrap import bootstrap
from memex.core.config import get_settings
from memex.core.types import Chunk
from memex.index.fts_store import FTSStore
from memex.retrieve.hybrid import hybrid_search
from memex.retrieve.rerank import cross_encoder_rerank

POOL_K = 50
TOP_K = 5
EXPAND_WINNERS = 3
NEIGHBORS = 1
ADD_CAP = 4


@dataclass
class QueryOutcome:
    """One answerable query's classification (exactly one primary bucket)."""

    qid: str
    bucket: str
    adjacency_extension: bool
    gold_count: int
    gold_in_winners_count: int


@dataclass
class CorpusReport:
    """Aggregated buckets for one queries.json file."""

    name: str
    outcomes: list[QueryOutcome] = field(default_factory=list)

    def count(self, bucket: str) -> int:
        return sum(1 for o in self.outcomes if o.bucket == bucket)

    @property
    def answerable(self) -> int:
        return len(self.outcomes)

    @property
    def headroom_pct(self) -> float:
        if not self.outcomes:
            return 0.0
        return 100.0 * self.count("HEADROOM") / len(self.outcomes)


def load_answerable_queries(path: Path) -> list[dict[str, object]]:
    """The answerable subset: should_refuse=false AND non-empty relevant_chunk_ids."""
    data = json.loads(path.read_text(encoding="utf-8"))
    queries: list[dict[str, object]] = data.get("queries", [])
    return [
        q
        for q in queries
        if not q.get("should_refuse", False) and q.get("relevant_chunk_ids")
    ]


def neighbor_chunk_ids(
    doc_chunks: list[Chunk],
    winner_id: str,
    *,
    neighbors: int,
) -> list[str]:
    """The chunk_ids of the ±`neighbors` reading-order siblings of `winner_id`
    inside its document's ordered chunk list (char_start ascending — the order
    `FTSStore.chunks_for_document` guarantees). Empty when the winner isn't
    found (defensive: a stale gold label or a just-reindexed doc)."""
    idx = next((i for i, c in enumerate(doc_chunks) if c.chunk_id == winner_id), None)
    if idx is None:
        return []
    lo = max(0, idx - neighbors)
    hi = min(len(doc_chunks), idx + neighbors + 1)
    return [c.chunk_id for i, c in enumerate(doc_chunks[lo:hi], start=lo) if i != idx]


def build_expansion(
    winners: list[Chunk],
    chunks_by_doc: dict[str, list[Chunk]],
    *,
    expand_winners: int,
    neighbors: int,
    add_cap: int,
) -> set[str]:
    """The de-duplicated neighbor-expansion chunk_id set: greedy in rerank order
    over the first `expand_winners` winners, capped at `add_cap` ids total —
    mirroring the bounded-greedy contract of `augment_companion`."""
    winner_ids = {c.chunk_id for c in winners}
    added: set[str] = set()
    for w in winners[:expand_winners]:
        for cid in neighbor_chunk_ids(
            chunks_by_doc.get(w.document_id, []), w.chunk_id, neighbors=neighbors
        ):
            if cid in winner_ids or cid in added:
                continue
            added.add(cid)
            if len(added) >= add_cap:
                return added
    return added


def classify(
    gold: set[str],
    pool_ids: set[str],
    winner_ids: set[str],
    expansion_ids: set[str],
) -> tuple[str, bool]:
    """(primary bucket, adjacency_extension) for one query — see module docstring."""
    in_winners = gold & winner_ids
    if in_winners:
        extension = bool((gold - winner_ids) & expansion_ids)
        return ("gold_in_winners", extension)
    if gold & expansion_ids:
        return ("HEADROOM", False)
    if gold & pool_ids:
        return ("gold_below_cut", False)
    return ("gold_missed", False)


async def run_corpus(
    path: Path,
    fstore: FTSStore,
    *,
    pool_k: int,
    top_k: int,
    expand_winners: int,
    neighbors: int,
    add_cap: int,
    use_rerank: bool,
) -> CorpusReport:
    report = CorpusReport(name=path.parent.name)
    doc_cache: dict[str, list[Chunk]] = {}

    async def chunks_for(doc_id: str) -> list[Chunk]:
        if doc_id not in doc_cache:
            doc_cache[doc_id] = await fstore.chunks_for_document(doc_id)
        return doc_cache[doc_id]

    for q in load_answerable_queries(path):
        qid = str(q["qid"])
        question = str(q["question"])
        gold = {str(g) for g in q["relevant_chunk_ids"]}  # type: ignore[union-attr]

        pool = await hybrid_search(question, k=pool_k)
        if use_rerank:
            winners = await cross_encoder_rerank(question, pool, top_k=top_k)
        else:
            winners = pool[:top_k]

        chunks_by_doc: dict[str, list[Chunk]] = {}
        for w in winners[:expand_winners]:
            chunks_by_doc[w.document_id] = await chunks_for(w.document_id)
        expansion = build_expansion(
            winners,
            chunks_by_doc,
            expand_winners=expand_winners,
            neighbors=neighbors,
            add_cap=add_cap,
        )

        bucket, extension = classify(
            gold,
            {c.chunk_id for c in pool},
            {c.chunk_id for c in winners},
            expansion,
        )
        report.outcomes.append(
            QueryOutcome(
                qid=qid,
                bucket=bucket,
                adjacency_extension=extension,
                gold_count=len(gold),
                gold_in_winners_count=len(gold & {c.chunk_id for c in winners}),
            )
        )
        print(
            f"  {qid:<28} {bucket:<16}"
            f"{' +adjacency-extension' if extension else ''}",
            flush=True,
        )
    return report


def print_report(reports: list[CorpusReport], *, headroom_gate_pct: float) -> None:
    print("\n=== Parent-context headroom (Stage 1) ===")
    header = (
        f"{'corpus':<26}{'answerable':>11}{'in_winners':>12}{'HEADROOM':>10}"
        f"{'below_cut':>11}{'missed':>8}{'headroom%':>11}{'adj_ext':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in reports:
        adj = sum(1 for o in r.outcomes if o.adjacency_extension)
        print(
            f"{r.name:<26}{r.answerable:>11}{r.count('gold_in_winners'):>12}"
            f"{r.count('HEADROOM'):>10}{r.count('gold_below_cut'):>11}"
            f"{r.count('gold_missed'):>8}{r.headroom_pct:>10.1f}%{adj:>9}"
        )
    go = [r.name for r in reports if r.headroom_pct > headroom_gate_pct]
    if go:
        print(
            f"\nVERDICT: GO — headroom > {headroom_gate_pct:.0f}% on: {', '.join(go)}."
            " Proceed to the Stage-2 live `memex eval` A/B (default-OFF flag)."
        )
    else:
        print(
            f"\nVERDICT: NO-GO — no corpus exceeds {headroom_gate_pct:.0f}% headroom."
            " Record the negative result in the audit and close Theory A."
        )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", nargs="+", type=Path, help="queries.json paths")
    parser.add_argument("--pool-k", type=int, default=POOL_K)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--expand-winners", type=int, default=EXPAND_WINNERS)
    parser.add_argument("--neighbors", type=int, default=NEIGHBORS)
    parser.add_argument("--add-cap", type=int, default=ADD_CAP)
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="use the RRF top-k as the window (skips loading the reranker)",
    )
    parser.add_argument("--headroom-gate", type=float, default=10.0)
    args = parser.parse_args()

    bootstrap()
    settings = get_settings()
    print(f"vault: {settings.vault_path}")

    fstore = await FTSStore.open(settings.vault_path)
    try:
        reports: list[CorpusReport] = []
        for path in args.queries:
            if not path.is_file():
                print(f"SKIP (not found): {path}", file=sys.stderr)
                continue
            print(f"\n--- {path.parent.name} ---")
            reports.append(
                await run_corpus(
                    path,
                    fstore,
                    pool_k=args.pool_k,
                    top_k=args.top_k,
                    expand_winners=args.expand_winners,
                    neighbors=args.neighbors,
                    add_cap=args.add_cap,
                    use_rerank=not args.no_rerank,
                )
            )
    finally:
        await fstore.close()

    if not reports:
        print("No corpora processed.", file=sys.stderr)
        return 1
    print_report(reports, headroom_gate_pct=args.headroom_gate)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
