"""DB store-open cost benchmark — the Stage-0 GO/NO-GO gate for the deferred
"SQLite/LanceDB connection reuse" perf item (db_audit_2026_05_28 "highest-leverage").

The DB audit flagged a long-lived per-store handle as its highest-leverage perf
item but DEFERRED it as riskier. Before paying that risk (process-lifetime reuse
reintroduces a silent stale-inode hazard across `reindex --force`), this script
MEASURES where store-opens actually cost — so the build decision is data-gated,
not a guess (the `docling_heading_histogram.py` diagnostic pattern).

It measures, against the WARM vault DBs (no models needed for A/B/C):
  A. each store's `.open()` in isolation — VectorStore (LanceDB) vs FTSStore /
     TableStore / GraphStore (sqlite). The linchpin: reuse only removes open-time,
     and sqlite open is ~1 ms irreducible — so the question is the LanceDB open.
  B. the DB-BOUND retrieval cost (open vs dense+bm25 search) with a random query
     vector — no embedder — so the open FRACTION of the part reuse can remove is
     isolated from the fixed embed/LLM cost.
  C. the graph DISCOVERY surfaces (`entity_overview`, `related_documents`) — the
     non-LLM read paths where a win, if any, lives.
  D. (best-effort, needs the embedder) one REAL `hybrid_search` decomposition
     (open / embed / search) → the realistic open fraction of a retrieval call.

GO bar (from the approved plan): opens are a non-trivial fraction WHERE IT MATTERS
— > ~15-20 % of a discovery-surface / retrieval call OR > ~100 ms absolute on a
hot read path — AND the cost is confirmably LanceDB (not the ~1 ms sqlite open).
Otherwise: a MEASURED DEFER (record the numbers, ship no reuse code).

Usage:
  MEMEX_OBSERVABILITY__LANGFUSE_ENABLED=false \
    uv run python scripts/db_open_bench.py            # full run (A-D)
  ... scripts/db_open_bench.py --no-models            # A-C only (skip the embedder)
  ... scripts/db_open_bench.py --iterations 30 --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from memex.cli.bootstrap import bootstrap
from memex.core.config import get_settings

# Representative read queries (mix EN/FR, matching the vault's CCNA + 10-K + papers).
_QUERIES = [
    "What is the OSI model and its seven layers?",
    "How does role-based access control assign permissions?",
    "Quelle est la plage de VLAN dans le diagramme de configuration?",
    "NVLink C2C interconnect bandwidth between Grace and Hopper",
    "zero trust architecture tenets and assumptions",
    "director compensation total for the fiscal year",
    "spanning tree protocol root bridge election",
    "comment configurer une liste de controle d'acces ACL",
]
_ENTITIES = ["OSI", "VLAN", "Zero Trust", "NVLink"]


def _stats(samples: list[float]) -> dict[str, float]:
    """median / mean / p90 / min / max in milliseconds (samples are seconds)."""
    if not samples:
        return {"n": 0}
    ms = sorted(s * 1000.0 for s in samples)
    n = len(ms)
    return {
        "n": n,
        "median": ms[n // 2],
        "mean": sum(ms) / n,
        "p90": ms[min(n - 1, math.ceil(0.9 * n) - 1)],
        "min": ms[0],
        "max": ms[-1],
    }


def _fmt(label: str, st: dict[str, float]) -> str:
    if not st.get("n"):
        return f"  {label:<34} (no samples)"
    return (
        f"  {label:<34} median={st['median']:7.2f}ms  mean={st['mean']:7.2f}ms  "
        f"p90={st['p90']:7.2f}ms  (n={st['n']})"
    )


async def _time_open(open_coro_factory: Any, close_attr: str, n: int, warmup: int) -> list[float]:
    """Open a store `n` times (after `warmup` discarded opens), timing each open;
    the store is closed between iterations so each open is independent."""
    samples: list[float] = []
    for i in range(n + warmup):
        t0 = perf_counter()
        store = await open_coro_factory()
        dt = perf_counter() - t0
        closer = getattr(store, close_attr, None)
        if closer is not None:
            res = closer()
            if asyncio.iscoroutine(res):
                await res
        if i >= warmup:
            samples.append(dt)
    return samples


async def part_a(vault: Path, n: int, warmup: int) -> dict[str, dict[str, float]]:
    """Isolated open cost per store. LanceDB (VectorStore) vs the sqlite trio."""
    from memex.index.fts_store import FTSStore
    from memex.index.table_store import TableStore
    from memex.index.vector_store import VectorStore

    out: dict[str, dict[str, float]] = {}
    out["VectorStore.open (LanceDB)"] = _stats(
        await _time_open(lambda: VectorStore.open(vault), "close", n, warmup)
    )
    out["FTSStore.open (sqlite)"] = _stats(
        await _time_open(lambda: FTSStore.open(vault), "close", n, warmup)
    )
    out["TableStore.open (sqlite)"] = _stats(
        await _time_open(lambda: TableStore.open(vault), "close", n, warmup)
    )
    # GraphStore is optional (ryugraph) — fail-open.
    try:
        from memex.index.graph_store import GraphStore

        out["GraphStore.open (ryugraph)"] = _stats(
            await _time_open(lambda: GraphStore.open(vault), "close", n, warmup)
        )
    except ImportError:
        out["GraphStore.open (ryugraph)"] = {"n": 0}
    return out


def _probe_unit_vector(dim: int) -> list[float]:
    """A deterministic non-zero unit vector to drive LanceDB's ANN search — the
    DB does the same work regardless of the vector's content, so no RNG is needed
    (and a fixed vector keeps the benchmark reproducible)."""
    v = [math.sin(i * 0.123 + 0.5) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


async def part_b(vault: Path, k: int = 50) -> dict[str, Any]:
    """DB-bound retrieval decomposition WITHOUT an embedder: a random query vector
    drives `vstore.search`; the open vs (dense+bm25) split isolates the open
    fraction of the part reuse can actually remove."""
    from memex.index.fts_store import FTSStore
    from memex.index.vector_store import EMBEDDING_DIM, VectorStore

    vec = _probe_unit_vector(EMBEDDING_DIM)
    open_s: list[float] = []
    search_s: list[float] = []
    for q in _QUERIES:
        t0 = perf_counter()
        vstore = await VectorStore.open(vault)
        fstore = await FTSStore.open(vault)
        open_s.append(perf_counter() - t0)
        t1 = perf_counter()
        dense_task = asyncio.create_task(vstore.search(vec, k=k))
        bm25_task = asyncio.create_task(fstore.search(q, k=k))
        await asyncio.gather(dense_task, bm25_task)
        search_s.append(perf_counter() - t1)
        await fstore.close()
        await vstore.close()
    return {"open": _stats(open_s), "search": _stats(search_s)}


async def part_c(vault: Path) -> dict[str, Any]:
    """Graph discovery surfaces — the non-LLM read paths (no embedder)."""
    from memex.retrieve.entity import entity_overview
    from memex.retrieve.related import related_documents_for_seeds

    eo: list[float] = []
    for name in _ENTITIES:
        t0 = perf_counter()
        await entity_overview(name)
        eo.append(perf_counter() - t0)

    # related_documents: seed from a handful of real doc_ids.
    doc_ids = sorted(p.stem for p in (vault / "documents").glob("*.md"))[:6]
    rd: list[float] = []
    for did in doc_ids:
        t0 = perf_counter()
        await related_documents_for_seeds(vault, [did])
        rd.append(perf_counter() - t0)
    return {"entity_overview (call)": _stats(eo), "related_documents (call)": _stats(rd)}


async def part_d(vault: Path, k: int = 50) -> dict[str, Any]:
    """Real `hybrid_search` decomposition (open / embed / dense+bm25) with the
    live embedder — the realistic open fraction of a retrieval call. Best-effort."""
    from memex.index.fts_store import FTSStore
    from memex.index.vector_store import VectorStore
    from memex.retrieve.hybrid import _embed_query

    # Warm the embedder once (model load is not part of the measurement).
    await _embed_query("warmup")

    open_s: list[float] = []
    embed_s: list[float] = []
    search_s: list[float] = []
    for q in _QUERIES:
        t0 = perf_counter()
        vstore = await VectorStore.open(vault)
        fstore = await FTSStore.open(vault)
        open_s.append(perf_counter() - t0)
        t1 = perf_counter()
        emb = await _embed_query(q)
        embed_s.append(perf_counter() - t1)
        t2 = perf_counter()
        dense_task = asyncio.create_task(vstore.search(emb, k=k))
        bm25_task = asyncio.create_task(fstore.search(q, k=k))
        await asyncio.gather(dense_task, bm25_task)
        search_s.append(perf_counter() - t2)
        await fstore.close()
        await vstore.close()
    return {"open": _stats(open_s), "embed": _stats(embed_s), "search": _stats(search_s)}


def _go_no_go(a: dict[str, Any], b: dict[str, Any], c: dict[str, Any], d: dict[str, Any]) -> str:
    """Apply the approved GO bar and print the verdict + reasoning."""
    lance = a.get("VectorStore.open (LanceDB)", {})
    sqlite = a.get("FTSStore.open (sqlite)", {})
    lance_med = lance.get("median", 0.0)
    sqlite_med = sqlite.get("median", 0.0)

    lines = [f"\n{'=' * 78}\nGO / NO-GO\n{'=' * 78}"]
    lines.append(f"  LanceDB open median = {lance_med:.2f} ms ; sqlite open median = {sqlite_med:.2f} ms")

    # Fraction of the DB-bound retrieval that is opens (Part B — the part reuse removes).
    b_open = b.get("open", {}).get("median", 0.0)
    b_search = b.get("search", {}).get("median", 0.0)
    b_frac = b_open / (b_open + b_search) * 100 if (b_open + b_search) else 0.0
    lines.append(
        f"  DB-bound retrieval (Part B): open={b_open:.2f}ms  search={b_search:.2f}ms"
        f"  → opens = {b_frac:.1f}% of the DB-bound call"
    )

    # Discovery surfaces (Part C) absolute call cost.
    eo = c.get("entity_overview (call)", {}).get("median", 0.0)
    rd = c.get("related_documents (call)", {}).get("median", 0.0)
    lines.append(f"  Discovery: entity_overview={eo:.2f}ms  related_documents={rd:.2f}ms (whole call)")

    if d:
        d_open = d.get("open", {}).get("median", 0.0)
        d_embed = d.get("embed", {}).get("median", 0.0)
        d_search = d.get("search", {}).get("median", 0.0)
        total = d_open + d_embed + d_search
        d_frac = d_open / total * 100 if total else 0.0
        lines.append(
            f"  Real hybrid_search (Part D): open={d_open:.2f}ms  embed={d_embed:.2f}ms"
            f"  search={d_search:.2f}ms  → opens = {d_frac:.1f}% of the retrieval call"
        )

    # The bar: LanceDB-confirmed AND (>15-20% of a discovery/retrieval call OR >100ms absolute).
    lance_dominates = lance_med >= 2.0 * max(sqlite_med, 0.01)
    over_abs = lance_med > 100.0 or b_open > 100.0
    over_frac = b_frac >= 15.0
    go = lance_dominates and (over_abs or over_frac)
    lines.append("")
    lines.append(f"  LanceDB dominates sqlite (>=2x): {lance_dominates}")
    lines.append(f"  open >100ms absolute on a hot path: {over_abs}")
    lines.append(f"  opens >=15% of the DB-bound call: {over_frac}")
    lines.append("")
    lines.append(f"  >>> VERDICT: {'GO — build the narrow LanceDB-only reuse' if go else 'NO-GO — measured defer (record numbers, ship no reuse code)'}")
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> int:
    bootstrap()
    settings = get_settings()
    vault = settings.vault_path
    print(f"vault: {vault}  (embedder_device={settings.models.embedder_device}, "
          f"reranker_device={settings.models.reranker_device}, mode={settings.models.co_residence_mode})")

    a = await part_a(vault, n=args.iterations, warmup=2)
    print(f"\n{'=' * 78}\nA. Isolated store-open cost\n{'=' * 78}")
    for label, st in a.items():
        print(_fmt(label, st))

    b = await part_b(vault)
    print(f"\n{'=' * 78}\nB. DB-bound retrieval (random vector, no embedder)\n{'=' * 78}")
    print(_fmt("open (vstore+fstore)", b["open"]))
    print(_fmt("dense+bm25 search", b["search"]))

    c = await part_c(vault)
    print(f"\n{'=' * 78}\nC. Graph discovery surfaces (no embedder)\n{'=' * 78}")
    for label, st in c.items():
        print(_fmt(label, st))

    d: dict[str, Any] = {}
    if not args.no_models:
        print(f"\n{'=' * 78}\nD. Real hybrid_search decomposition (live embedder)\n{'=' * 78}")
        try:
            d = await part_d(vault)
            print(_fmt("open (vstore+fstore)", d["open"]))
            print(_fmt("embed (EmbeddingGemma)", d["embed"]))
            print(_fmt("dense+bm25 search", d["search"]))
        except Exception as e:  # best-effort; A-C are the decisive parts
            print(f"  (skipped — embedder unavailable: {type(e).__name__}: {e})")

    print(_go_no_go(a, b, c, d))

    if args.json:
        print("\n" + json.dumps({"a": a, "b": b, "c": c, "d": d}, indent=1, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=20, help="measured opens per store (Part A)")
    ap.add_argument("--no-models", action="store_true", help="skip Part D (no embedder load)")
    ap.add_argument("--json", action="store_true", help="also emit the raw stats as JSON")
    args = ap.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
