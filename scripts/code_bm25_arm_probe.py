"""Phase-3 BM25-for-code arm-separation probe (MEASURE-FIRST, read-only).

Measures whether a LEXICAL (BM25) retrieval arm recovers exact-identifier code chunks
that the DENSE EmbeddingGemma arm misses — i.e. whether the prose finding "BM25 recall
⊆ dense" (docs/audits/09-fts-bm25-arm-separation.md) INVERTS for code. The deliverable
is a recall@k table that decides GO/NO-GO; this script changes NOTHING in production.

Run against the dedicated code probe vault (NOT the prose vault):

    MEMEX_VAULT_PATH=/tmp/memex-code-probe-vault \
    MEMEX_MODELS__CO_RESIDENCE_MODE=manual \
    MEMEX_MODELS__EMBEDDER_DEVICE=cpu MEMEX_MODELS__RERANKER_DEVICE=cpu \
    uv run python scripts/code_bm25_arm_probe.py \
        tests/eval-data/codex-rs-find-the-code/queries.json

Arms (per query, all over the SAME chunk pool, PRE-rerank — this is a retrieval question):
  - dense        : VectorStore.search(_embed_query(q), k)              (EmbeddingGemma)
  - bm25-phrase  : FTSStore.search(q, k)  — the LIVE phrase-wrap        (expected ~0 on NL)
  - bm25-term-*  : a self-built OR'd-quoted-atom FTS5 MATCH (3 variants; the CANDIDATE)
  - union        : the TRUE SET union dense ∪ bm25-term-both (the recall CEILING)
  - rrf (diag)   : reciprocal_rank_fusion([dense, bm25-term-both]) — does fusion realize it

The term-builder is UNDERSCORE-AWARE and NEVER case-splits (the unicode61 index keeps
PascalCase whole but splits snake_case on `_`; `core.text.atomise` STRIPS `_` into a mash,
so it is deliberately NOT used). See docs/specs/code-chunking.md §Phase 3.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from memex.cli.bootstrap import bootstrap
from memex.core.config import get_settings
from memex.core.text import STOPWORDS, atomise
from memex.eval.scoring import gold_chunk_recall
from memex.index.fts_store import FTSStore
from memex.index.vector_store import VectorStore
from memex.retrieve.hybrid import _embed_query

# Question / structural / Rust-keyword filler to drop from a code query before lexical
# matching — these are low-signal and (for `struct`/`fn`/`enum`/…) appear in EVERY symbol
# heading, so keeping them would just add noise. The rare IDENTIFIERS are the signal.
_QUESTION_STOP: frozenset[str] = frozenset(
    {
        "which",
        "what",
        "where",
        "when",
        "why",
        "how",
        "who",
        "whom",
        "whose",
        "whether",
        "does",
        "do",
        "did",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "from",
        "into",
        "by",
        "at",
        "as",
        "it",
        "its",
        "and",
        "or",
        "not",
        "no",
        "function",
        "functions",
        "fn",
        "method",
        "methods",
        "struct",
        "structs",
        "enum",
        "enums",
        "trait",
        "traits",
        "impl",
        "implementation",
        "mod",
        "module",
        "modules",
        "const",
        "constant",
        "constants",
        "static",
        "type",
        "types",
        "macro",
        "macros",
        "call",
        "calls",
        "called",
        "calling",
        "use",
        "uses",
        "used",
        "using",
        "define",
        "defined",
        "defines",
        "defining",
        "definition",
        "implement",
        "implemented",
        "implements",
        "work",
        "works",
        "compute",
        "computes",
        "set",
        "sets",
        "value",
        "values",
        "field",
        "fields",
        "variant",
        "variants",
        "own",
        "owns",
        "hold",
        "holds",
        "create",
        "creates",
        "created",
        "invoke",
        "invokes",
        "construct",
        "constructs",
        "constructed",
        "before",
        "after",
        "apply",
        "applying",
        "decide",
        "decides",
        "agent",
        "run",
        "runs",
        "without",
        "asking",
        "approval",
    }
)
_STOPSET = STOPWORDS | _QUESTION_STOP
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_KS = (5, 10, 20, 50)
_DECISION_K = 50


def _query_tokens(query: str) -> list[str]:
    """Surviving lowercased identifier-ish tokens (stopwords + len<2 dropped)."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(query):
        low = raw.lower()
        if len(low) < 2 or low in _STOPSET:
            continue
        out.append(low)
    return out


def build_term_match(query: str, mode: str) -> str:
    """An OR of individually-quoted FTS5 atoms — underscore-aware, NEVER case-split.

    whole : the raw token quoted (snake_case → a contiguous phrase that matches the
            literal identifier; PascalCase → the one whole unicode61 token).
    sub   : `token.split('_')` parts (snake → its parts; a no-`_` token → itself).
    both  : whole ∪ sub (the recommended primary).
    """
    atoms: set[str] = set()
    for tok in _query_tokens(query):
        if mode in ("whole", "both"):
            atoms.add(tok)
        if mode in ("sub", "both"):
            atoms.update(p for p in tok.split("_") if len(p) >= 2)
    # quote each atom; escape any embedded double-quote (identifiers won't have one)
    return " OR ".join(f'"{a}"' for a in sorted(atoms))


async def _bm25_term_ids(fstore: FTSStore, match: str, k: int) -> list[str]:
    """Raw-SQL bm25 ranking over `chunks_fts` with a self-built MATCH (read-only)."""
    if not match:
        return []

    def _run() -> list[str]:
        cur = fstore._db.execute(
            "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY bm25(chunks_fts) LIMIT ?",
            (match, k),
        )
        return [r[0] for r in cur.fetchall()]

    return await asyncio.to_thread(_run)


def _rrf_ids(id_lists: list[list[str]], k0: int = 60) -> list[str]:
    """A local reciprocal-rank fusion on id lists (the diagnostic; mirrors retrieve.fusion)."""
    score: dict[str, float] = {}
    for ids in id_lists:
        for rank, cid in enumerate(ids):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k0 + rank + 1)
    return sorted(score, key=lambda c: score[c], reverse=True)


async def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("queries_file", type=Path)
    ap.add_argument(
        "--pool-k", type=int, default=50, help="per-arm candidate depth (the decision pool)"
    )
    args = ap.parse_args()

    data = json.loads(args.queries_file.read_text())
    queries = data.get("queries", [])
    ans = [q for q in queries if not q.get("should_refuse") and q.get("relevant_chunk_ids")]
    refs = [q for q in queries if q.get("should_refuse")]

    bootstrap()
    settings = get_settings()
    vstore = await VectorStore.open(settings.vault_path)
    fstore = await FTSStore.open(settings.vault_path)

    # ---- Banner / self-checks ---------------------------------------------------------
    print("=" * 100)
    print(f"BM25-for-code arm-separation probe  ·  vault={settings.vault_path}")
    print(f"corpus={args.queries_file}  ·  ANS={len(ans)}  REF={len(refs)}  pool-k={args.pool_k}")
    print(
        f"atomise('fetch_frame_data') = {atomise('fetch_frame_data')}   "
        f"<- STRIPS '_' (a mash); the term-builder does NOT use it raw"
    )
    print(f"stopset size = {len(_STOPSET)}")
    print("=" * 100)

    arms = ("dense", "bm25-phrase", "term-whole", "term-sub", "term-both", "union", "rrf@50(diag)")
    sums: dict[str, dict[int, float]] = {a: {k: 0.0 for k in _KS} for a in arms}
    per_query: list[dict] = []

    for q in ans:
        question = q["question"]
        gold = q["relevant_chunk_ids"]
        emb = await _embed_query(question)
        dense = [c.chunk_id for c in await vstore.search(emb, k=args.pool_k)]
        phrase = [c.chunk_id for c in await fstore.search(question, k=args.pool_k)]
        t_whole = await _bm25_term_ids(fstore, build_term_match(question, "whole"), args.pool_k)
        t_sub = await _bm25_term_ids(fstore, build_term_match(question, "sub"), args.pool_k)
        t_both = await _bm25_term_ids(fstore, build_term_match(question, "both"), args.pool_k)
        rrf = _rrf_ids([dense, t_both])

        row = {"qid": q["qid"], "type": q.get("_answer_type", ""), "gold": gold}
        for k in _KS:
            r = {
                "dense": gold_chunk_recall(dense, gold, k),
                "bm25-phrase": gold_chunk_recall(phrase, gold, k),
                "term-whole": gold_chunk_recall(t_whole, gold, k),
                "term-sub": gold_chunk_recall(t_sub, gold, k),
                "term-both": gold_chunk_recall(t_both, gold, k),
            }
            union_set = set(dense[:k]) | set(t_both[:k])
            r["union"] = len(set(gold) & union_set) / len(gold)
            r["rrf@50(diag)"] = gold_chunk_recall(rrf, gold, k)
            # self-check: a true set-union can never recall below either parent arm
            if r["union"] < max(r["dense"], r["term-both"]) - 1e-9:
                raise RuntimeError(f"union self-check failed at {q['qid']} @{k}")
            for a in arms:
                sums[a][k] += r[a]
            if k == _DECISION_K:
                row["k50"] = r
                row["only_via_term"] = bool(
                    set(gold) & set(t_both[:k]) and not (set(gold) & set(dense[:k]))
                )
        per_query.append(row)

    n = len(ans)

    # ---- Aggregate table --------------------------------------------------------------
    print(f"\nMEAN recall@k over {n} ANS queries (DECISION k={_DECISION_K}):")
    hdr = f"  {'arm':<14}" + "".join(f"@{k:<8}" for k in _KS)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for a in arms:
        print(f"  {a:<14}" + "".join(f"{sums[a][k] / n:<9.3f}" for k in _KS))

    # ---- Per-regime breakdown at k=50 -------------------------------------------------
    print(f"\nPer-regime mean recall@{_DECISION_K}:")
    regimes: dict[str, list[dict]] = {}
    for row in per_query:
        regimes.setdefault(
            "definition" if row["type"].startswith("definition") else "usage_reference", []
        ).append(row)
    print(
        "  {:<18}{:<7}{:<9}{:<11}{:<10}{:<9}".format(
            "regime", "n", "dense", "term-both", "union", "Δ(u-d)"
        )
    )
    for name, rows in sorted(regimes.items()):
        d = sum(r["k50"]["dense"] for r in rows) / len(rows)
        tb = sum(r["k50"]["term-both"] for r in rows) / len(rows)
        u = sum(r["k50"]["union"] for r in rows) / len(rows)
        print(f"  {name:<18}{len(rows):<7}{d:<9.3f}{tb:<11.3f}{u:<10.3f}{u - d:+.3f}")

    # ---- Queries where the gold enters the pool ONLY via bm25-term (the headroom) -----
    only = [r for r in per_query if r.get("only_via_term")]
    print(f"\nGold recovered ONLY by bm25-term (not dense) @k={_DECISION_K}: {len(only)}")
    for r in only:
        print(f"  + {r['qid']:<22} ({r['type']})")

    # ---- Per-query miss detail (dense misses at k=50) ---------------------------------
    misses = [r for r in per_query if r["k50"]["dense"] < 1.0]
    print(f"\nDense MISSES @k={_DECISION_K} ({len(misses)} / {n}) — per-arm recall:")
    print(
        "  {:<22}{:<22}{:<8}{:<8}{:<9}{:<8}".format(
            "qid", "type", "dense", "phrase", "t-both", "union"
        )
    )
    for r in sorted(misses, key=lambda x: x["qid"]):
        k = r["k50"]
        print(
            f"  {r['qid']:<22}{r['type']:<22}{k['dense']:<8.2f}{k['bm25-phrase']:<8.2f}"
            f"{k['term-both']:<9.2f}{k['union']:<8.2f}"
        )

    # ---- Counterfactual lexical true-negative check -----------------------------------
    print("\nCounterfactual bm25-term hits (should be ~0 — a non-existent identifier):")
    for q in refs:
        m = build_term_match(q["question"], "both")
        hits = await _bm25_term_ids(fstore, m, args.pool_k)
        print(f"  {q['qid']:<22} term-both hits={len(hits):<3}  match={m[:70]}")

    # ---- Validity precondition + verdict ----------------------------------------------
    dense50 = sums["dense"][_DECISION_K] / n
    union50 = sums["union"][_DECISION_K] / n
    print("\n" + "=" * 100)
    print(
        f"VALIDITY: dense@{_DECISION_K} = {dense50:.3f}  "
        + (
            "(== 1.0 → VACUOUS: no retrieval pressure, enlarge corpus / harden queries)"
            if dense50 >= 0.999
            else "(< 1.0 → valid: there ARE dense misses to potentially recover)"
        )
    )
    print(
        f"VERDICT INPUT: union@{_DECISION_K} = {union50:.3f}  vs  dense@{_DECISION_K} = {dense50:.3f}  "
        f"→ Δ = {union50 - dense50:+.3f}  ({len(only)} gold recovered only-by-term)"
    )
    print("GO if Δ materially > 0 AND ≥ a couple of only-by-term recoveries; NO-GO if valid & Δ≈0.")
    print("=" * 100)

    await vstore.close()
    await fstore.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
