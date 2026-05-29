"""Citation-graph audit — the reproducible measure behind the citation-chain-following
DATA-FIRST experiment (ADR-0011 build-out).

Citation-chain following traverses `CITES` Document->Document edges. As of 2026-05-28 the
live graph had only 6 CITES edges (one syllabus -> 6 lectures, a depth-1 star, ZERO
multi-hop paths) — so chain-following had no data to run on. The blocker is DATA, not code:
`enrich/citations.py`'s academic resolver works, but the corpus has no citation-dense docs
(CITES is Document->Document WITHIN the vault, so a citation only becomes an edge when the
cited work is ALSO ingested — a lone paper yields zero edges; you need a citation-LINKED
cluster).

This script re-measures the CITES subgraph so a cluster ingest can be compared to the
baseline against the pre-registered decision criteria (build chain-following only if real
data yields a genuine subgraph, not another star). Pure graph read — no GPU, no bootstrap.

Usage:
  uv run python scripts/citation_graph_audit.py [--out FILE]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from typing import Any

from memex.core.config import MemexSettings, get_settings, set_settings
from memex.index.graph_store import GraphStore

# Course cross-references ("Cours 4", "Semaine 5", …) vs academic citations ("Smith 2024",
# a title). Mirrors enrich/course_refs.py's ordinal vocabulary.
_COURSE_REF_RE = re.compile(r"^\s*(?:cours|semaine|le[çc]on|module|chapitre)\s+\d", re.IGNORECASE)

# Pre-registered "worth building" bar (see the plan): a real subgraph, not a star.
_MIN_EDGES = 15
_MIN_DOCS_WITH_EDGES = 5
_MIN_MULTIHOP = 1


def _rows(conn: Any, cypher: str, params: dict[str, Any] | None = None) -> list[tuple[Any, ...]]:
    res = conn.execute(cypher, params or {})
    out: list[tuple[Any, ...]] = []
    while res.has_next():
        out.append(tuple(res.get_next()))
    return out


def _audit(conn: Any) -> dict[str, Any]:
    # Per-type counts (ryugraph's dialect has no `type(r)`); the rel tables are the
    # fixed set declared in index/schemas/graph.cypher (constants, not user input).
    rel_histogram: dict[str, int] = {}
    for rel in ("MENTIONS", "CITES", "DEFINES", "RELATES_TO"):
        r = _rows(conn, f"MATCH ()-[x:{rel}]->() RETURN count(x);")
        rel_histogram[rel] = int(r[0][0]) if r else 0
    n_docs = int(_rows(conn, "MATCH (d:Document) RETURN count(d);")[0][0])

    edges = _rows(
        conn,
        "MATCH (a:Document)-[c:CITES]->(b:Document) "
        "RETURN a.doc_id, a.title, b.doc_id, b.title, c.surface_text, c.confidence;",
    )
    edge_list = [
        {
            "from": a_title or a_id,
            "to": b_title or b_id,
            "surface_text": surface,
            "confidence": round(float(conf), 3) if conf is not None else None,
            "kind": "course_ref" if (surface and _COURSE_REF_RE.match(surface)) else "academic",
        }
        for a_id, a_title, b_id, b_title, surface, conf in edges
    ]

    # Per-doc in/out degree, only for docs touching CITES.
    out_deg: dict[str, int] = {}
    in_deg: dict[str, int] = {}
    for a_id, _at, b_id, _bt, _s, _c in edges:
        out_deg[a_id] = out_deg.get(a_id, 0) + 1
        in_deg[b_id] = in_deg.get(b_id, 0) + 1
    docs_with_edges = sorted(set(out_deg) | set(in_deg))

    # Multi-hop chains A -> … -> C (depth >= 2). The whole premise of "chain" following.
    # (No `length(p)` — ryugraph's path-function support is thin; endpoints are enough.)
    multihop = _rows(
        conn,
        "MATCH (a:Document)-[:CITES*2..4]->(c:Document) RETURN DISTINCT a.title, c.title LIMIT 25;",
    )
    chains = [{"from": a or "?", "to": c or "?"} for a, c in multihop]

    course = sum(1 for e in edge_list if e["kind"] == "course_ref")
    return {
        "rel_histogram": rel_histogram,
        "n_docs": n_docs,
        "cites_edges": len(edge_list),
        "docs_with_any_cites": len(docs_with_edges),
        "academic_edges": len(edge_list) - course,
        "course_ref_edges": course,
        "max_out_degree": max(out_deg.values(), default=0),
        "max_in_degree": max(in_deg.values(), default=0),
        "multihop_paths": len(chains),
        "example_chains": chains[:5],
        "edges": edge_list,
    }


def _verdict(stats: dict[str, Any]) -> tuple[bool, str]:
    ok = (
        stats["cites_edges"] >= _MIN_EDGES
        and stats["docs_with_any_cites"] >= _MIN_DOCS_WITH_EDGES
        and stats["multihop_paths"] >= _MIN_MULTIHOP
    )
    reason = (
        f"edges {stats['cites_edges']}>={_MIN_EDGES}, "
        f"docs {stats['docs_with_any_cites']}>={_MIN_DOCS_WITH_EDGES}, "
        f"multihop {stats['multihop_paths']}>={_MIN_MULTIHOP}"
    )
    return ok, reason


async def _run() -> dict[str, Any]:
    set_settings(MemexSettings())  # type: ignore[call-arg]
    graph = await GraphStore.open(get_settings().vault_path)
    try:
        return await asyncio.to_thread(_audit, graph._conn)
    finally:
        await graph.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    stats = asyncio.run(_run())
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(json.dumps(stats, indent=2), encoding="utf-8")

    build, reason = _verdict(stats)
    print(json.dumps({k: v for k, v in stats.items() if k != "edges"}, indent=2))
    print(f"\n=== CITES audit ({stats['n_docs']} docs) ===", file=sys.stderr)
    print(
        f"  CITES={stats['cites_edges']} "
        f"(academic={stats['academic_edges']} course_ref={stats['course_ref_edges']}) "
        f"docs_with_edges={stats['docs_with_any_cites']} "
        f"multihop_paths={stats['multihop_paths']} "
        f"max_out={stats['max_out_degree']} max_in={stats['max_in_degree']}",
        file=sys.stderr,
    )
    for c in stats["example_chains"]:
        print(f"    chain: {c['from'][:34]} -> … -> {c['to'][:34]}", file=sys.stderr)
    print(
        f"  VERDICT: {'BUILD chain-following' if build else 'DEFER (sparse/shallow)'} — {reason}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
