"""Reranker A/B — the frozen-pool rank-delta probe (stage 1 of the sharper-reranker arc,
audit-17). Deterministic, CPU-only, NO answer-path LLM: for each probe query the k=50
hybrid pool is retrieved ONCE (frozen), then re-scored by the incumbent and each
candidate; the metric is the GOLD chunk's rank per model. Mirrors the audits-09/13
arm-separation methodology.

Two query sets:
  - CASE FILES (fixed list below): the queries whose gold the incumbent buries —
    success = the gold ENTERS the top-5.
  - CONTROLS (auto-selected): queries whose gold the incumbent ranks top-5 today —
    success = the gold STAYS top-5 (escapes counted against the candidate).

Usage:
    MEMEX_MODELS__CO_RESIDENCE_MODE=manual MEMEX_MODELS__EMBEDDER_DEVICE=cpu \\
    MEMEX_MODELS__RERANKER_DEVICE=cpu uv run python scripts/reranker_ab_probe.py \\
        --candidate ce:BAAI/bge-reranker-v2-m3 \\
        --candidate qwen3:Qwen/Qwen3-Reranker-0.6B \\
        [--out /tmp/reranker_ab.json] [--controls-per-corpus 3]

A candidate is `<backend>:<hf_model_id>` where backend ∈ {ce, qwen3}:
  ce    — sentence-transformers CrossEncoder (.predict logits)
  qwen3 — causal-LM yes/no last-token scoring (the production `_score_qwen3` shape)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

from memex.cli.bootstrap import bootstrap
from memex.core.types import Chunk
from memex.retrieve.hybrid import hybrid_search

# (corpus, qid) — the case-file set; bge ranks from audits 14/15 in the comment.
CASE_FILES: list[tuple[str, str]] = [
    ("codex-rs-find-the-code", "ftc-use-isknownsafe"),  # caller at bge #6
    ("codex-rs-find-the-code", "ftc-use-unifieddiff"),  # caller at bge #6
    ("codex-rs-find-the-code", "ftc-use-issafetocall"),  # definition-distraction class
    ("codex-rs-find-the-code", "ftc-use-convhistory"),  # the demotion lever's regression pair —
    ("codex-rs-find-the-code", "ftc-use-applyhunks"),  # a sharper reranker must rank these WELL
    ("codex-rs-find-the-code", "ftc-big-runmain"),  # gold at bge #6 (k=8-conditional)
    ("slide-decks", "slide-decks-17"),  # two tiny golds at bge #15/#18
    ("slide-decks", "slide-decks-25"),  # tiny gold at bge #15, 10-K noise in top-5
    ("slide-decks", "slide-decks-31"),  # gold buried under 'Architecture Diagram' siblings
    ("slide-decks", "slide-decks-03"),  # k=8-conditional (sibling copy grounds)
    # NB nist-05 / gte-04 (the k=8 canaries) are EXCLUDED: their relevant_chunk_ids are
    # not in the k=50 pool (stale anchors — they answer via sibling chunks), so they
    # cannot be rank-measured; their regression risk is covered by stage 2/3 evals.
]
CONTROL_CORPORA = [
    "codex-rs-find-the-code",  # incl. the dense-saturated definition class
    "annual-report",
    "nist-zero-trust",
    "scientific-gte",
    "linux-fundamentals",
    "technical-guidelines",
    "cr350-multidoc",
]


def _load_query(corpus: str, qid: str) -> dict[str, Any]:
    qs = json.load(open(f"tests/eval-data/{corpus}/queries.json"))
    return next(x for x in qs["queries"] if x["qid"] == qid)


def _gold_rank(ranked_ids: list[str], gold: set[str]) -> int | None:
    return next((i + 1 for i, cid in enumerate(ranked_ids) if cid in gold), None)


class _Scorer:
    """One loaded candidate. backend ∈ {ce, qwen3}; CPU, fp32 (the eval-pinned posture)."""

    def __init__(self, backend: str, model_id: str) -> None:
        self.backend = backend
        self.model_id = model_id
        self.label = f"{backend}:{model_id.split('/')[-1]}"
        t0 = time.time()
        if backend == "ce":
            from sentence_transformers import CrossEncoder

            self._m = CrossEncoder(model_id, device="cpu")
        elif backend == "qwen3":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float32, device_map={"": "cpu"}, low_cpu_mem_usage=True
            )
            model.eval()
            yes_id = tok.encode("yes", add_special_tokens=False)[0]
            no_id = tok.encode("no", add_special_tokens=False)[0]
            self._m = (tok, model, yes_id, no_id)
        else:
            raise SystemExit(f"unknown backend {backend!r} (use ce: or qwen3:)")
        self.load_s = time.time() - t0

    def rank(self, query: str, chunks: list[Chunk]) -> tuple[list[str], float]:
        """Score the frozen pool; return (chunk_ids best-first, seconds)."""
        pairs = [(query, c.text) for c in chunks]
        t0 = time.time()
        if self.backend == "ce":
            scores = [float(s) for s in self._m.predict(pairs, batch_size=8)]
        else:
            import torch

            from memex.retrieve.rerank import _qwen3_format  # the production prompt shape

            tok, model, yes_id, no_id = self._m
            scores = []
            with torch.no_grad():
                for i in range(0, len(pairs), 4):
                    batch = [_qwen3_format(q, d) for q, d in pairs[i : i + 4]]
                    enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048)
                    logits = model(**enc).logits[:, -1, :]
                    two = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1)
                    probs = torch.softmax(two.float(), dim=1)[:, 1]
                    scores.extend(float(p) for p in probs)
        dt = time.time() - t0
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        return [chunks[o].chunk_id for o in order], dt


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", action="append", required=True, help="<ce|qwen3>:<hf_model_id>")
    ap.add_argument("--out", default="/tmp/reranker_ab.json")  # noqa: S108 — probe artifact
    ap.add_argument("--controls-per-corpus", type=int, default=3)
    args = ap.parse_args()

    bootstrap()

    # ---- assemble the query sets + FROZEN pools (one hybrid_search per query) ----
    probes: list[dict[str, Any]] = []
    for corpus, qid in CASE_FILES:
        q = _load_query(corpus, qid)
        gold = set(q.get("relevant_chunk_ids") or [])
        if not gold:
            print(f"  (case-file {qid}: no gold ids — skipped)")
            continue
        probes.append({"set": "case", "corpus": corpus, "qid": qid, "q": q["question"], "gold": gold})
    case_qids = {p["qid"] for p in probes}
    for corpus in CONTROL_CORPORA:
        qs = json.load(open(f"tests/eval-data/{corpus}/queries.json"))
        picked = 0
        for x in qs["queries"]:
            if picked >= args.controls_per_corpus:
                break
            if x.get("should_refuse") or not x.get("relevant_chunk_ids") or x["qid"] in case_qids:
                continue
            probes.append(
                {
                    "set": "control",
                    "corpus": corpus,
                    "qid": x["qid"],
                    "q": x["question"],
                    "gold": set(x["relevant_chunk_ids"]),
                }
            )
            picked += 1
    print(f"probe queries: {len(probes)} ({sum(1 for p in probes if p['set'] == 'case')} case-file)")

    for p in probes:
        p["pool"] = await hybrid_search(p["q"], k=50)

    # ---- score each candidate over the frozen pools ----
    results: dict[str, Any] = {"probes": [], "candidates": []}
    scorers = [_Scorer(*c.split(":", 1)) for c in args.candidate]
    for s in scorers:
        results["candidates"].append({"label": s.label, "load_s": round(s.load_s, 1)})
        print(f"loaded {s.label} in {s.load_s:.1f}s")

    for p in probes:
        row: dict[str, Any] = {
            "set": p["set"],
            "qid": p["qid"],
            "corpus": p["corpus"],
            "pool": len(p["pool"]),
            "ranks": {},
        }
        for s in scorers:
            ranked, dt = s.rank(p["q"], p["pool"])
            row["ranks"][s.label] = {"gold_rank": _gold_rank(ranked, p["gold"]), "s": round(dt, 1)}
        results["probes"].append(row)
        json.dump(results, open(args.out, "w"), indent=1)  # noqa: ASYNC230 — incremental flush
        marks = "  ".join(f"{k}:#{v['gold_rank']}" for k, v in row["ranks"].items())
        print(f"[{p['set']:7}] {p['qid']:26} {marks}", flush=True)

    # ---- the gate summary ----
    # Controls are only meaningful where the INCUMBENT (candidate[0]) ranks the gold
    # top-5 today; auto-picked queries with stale/buried golds are excluded from the
    # control denominator (they cannot "escape" what they never held).
    base = scorers[0].label
    valid_controls = {
        r["qid"]
        for r in results["probes"]
        if r["set"] == "control" and (r["ranks"][base]["gold_rank"] or 99) <= 5
    }
    print(f"\nvalid controls (incumbent top-5): {len(valid_controls)}")
    print("\n===== GATE SUMMARY (case-file: gold ENTERS top-5; control: gold STAYS top-5) =====")
    for s in scorers:
        case_in = sum(
            1
            for r in results["probes"]
            if r["set"] == "case" and (r["ranks"][s.label]["gold_rank"] or 99) <= 5
        )
        case_n = sum(1 for r in results["probes"] if r["set"] == "case")
        esc = sum(
            1
            for r in results["probes"]
            if r["qid"] in valid_controls and (r["ranks"][s.label]["gold_rank"] or 99) > 5
        )
        ctrl_n = len(valid_controls)
        mean_s = sum(r["ranks"][s.label]["s"] for r in results["probes"]) / max(1, len(results["probes"]))
        print(f"{s.label:42} case top-5: {case_in}/{case_n}   control escapes: {esc}/{ctrl_n}   ~{mean_s:.1f}s/query")
    print(f"\nartifact -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
