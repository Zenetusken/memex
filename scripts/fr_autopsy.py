"""False-refusal AUTOPSY harness (stage 2 of the false-refusal-zero arc, audit-15).

For each answerable-but-refused query, run the REAL `answer_query` with observe-only
recorders patched over the `answer` / `verify` / `assess_relevance` nodes (module-attr
assignment + `reset_compiled_graph()` — the integration-test `_fresh_graph` pattern), and
capture: the reranked window the LLM saw, the DRAFT (summary + claims), the per-claim
verify verdicts (`ungrounded_reasons` are code-generated and NAME the killing filter:
LLM-rejection vs numeric/name-only backstop vs phantom/missing), the relevance verdict,
and — for every ungrounded claim — the FULL cited-chunk text, so the mechanism
(paraphrase-vs-verbatim, numeric mismatch, hedged-draft, draft-actually-wrong) is
judgeable from the artifact alone. Read-only: no production code, no vault writes.

Usage:
    MEMEX_MODELS__CO_RESIDENCE_MODE=manual MEMEX_MODELS__EMBEDDER_DEVICE=cpu \
    MEMEX_MODELS__RERANKER_DEVICE=cpu uv run python scripts/fr_autopsy.py [out.json]
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import memex.agents.answering as A
from memex.cli.bootstrap import bootstrap

# (corpus, qid) — the live inventory from the stage-1 classifier (2026-06-09).
INVENTORY: list[tuple[str, str]] = [
    # 13 GATE-killed (evidence at rerank #1-3)
    ("chart-types", "chart-types-01"),
    ("chart-types", "chart-types-04"),
    ("chart-types", "chart-types-05"),
    ("cr350-multidoc", "cr350-img-01"),
    ("handwritten", "handwritten-06"),
    ("nist-zero-trust", "nist-zero-trust-10"),
    ("scientific-gte", "scientific-gte-09"),
    ("scientific-gte", "scientific-gte-11"),
    ("slide-decks", "slide-decks-03"),
    ("slide-decks", "slide-decks-04"),
    ("slide-decks", "slide-decks-05"),
    ("slide-decks", "slide-decks-16"),
    ("slide-decks", "slide-decks-21"),
    # 3 RERANK-CUTOFF (record ranks 1-10 for distractor analysis)
    ("slide-decks", "slide-decks-17"),
    ("slide-decks", "slide-decks-25"),
    ("codex-rs-find-the-code", "ftc-big-runmain"),
    # 1 untraceable chart-content (the vault-state hypothesis) + 1 retrieval-miss (record-only)
    ("slide-decks", "slide-decks-31"),
    ("annual-report", "annual-report-14"),
]

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fr_autopsy.json"  # noqa: S108 — diagnostic artifact

# the real nodes, captured before patching
_real_answer = A.answer
_real_verify = A.verify
_real_relevance = A.assess_relevance

# per-query capture buffer (reset between queries; answer_query runs serially here)
CAP: dict[str, Any] = {}


async def _rec_answer(state: A.AnswerState) -> Any:
    update = await _real_answer(state)
    draft = update.get("draft")
    CAP.setdefault("drafts", []).append(
        {
            "summary": getattr(draft, "summary", None),
            "claims": [
                {"claim": c.claim, "source_chunk_id": c.source_chunk_id}
                for c in (getattr(draft, "claims", None) or [])
            ],
        }
    )
    if not CAP.get("window"):
        CAP["window"] = [
            {
                "rank": i + 1,
                "chunk_id": c.chunk_id,
                "heading": (c.heading_path[-1] if c.heading_path else c.document_title)[:60],
                "text_head": (c.text or "")[:200],
            }
            for i, c in enumerate(state.reranked[:10])
        ]
        CAP["chunks_by_id"] = {c.chunk_id: (c.text or "") for c in state.reranked}
    return update


async def _rec_verify(state: A.AnswerState) -> Any:
    update = await _real_verify(state)
    v = update.get("verification")
    claims = [c.claim for c in (state.draft.claims if state.draft else [])]
    CAP.setdefault("verify_calls", []).append(
        {
            "claims": claims,
            "grounded": list(getattr(v, "grounded", []) or []),
            "ungrounded": list(getattr(v, "ungrounded", []) or []),
            "ungrounded_reasons": list(getattr(v, "ungrounded_reasons", []) or []),
        }
    )
    return update


async def _rec_relevance(state: A.AnswerState) -> Any:
    update = await _real_relevance(state)
    r = update.get("relevance")
    CAP.setdefault("relevance_calls", []).append(
        {
            "responsive": getattr(r, "responsive", None),
            "reason": getattr(r, "reason", None),
        }
    )
    return update


async def main() -> None:
    bootstrap()
    # install the recorders + force a fresh graph compile so they take effect
    A.answer = _rec_answer  # type: ignore[assignment]  # observe-only passthrough
    A.verify = _rec_verify  # type: ignore[assignment]
    A.assess_relevance = _rec_relevance  # type: ignore[assignment]
    A.reset_compiled_graph()

    rows: list[dict[str, Any]] = []
    for corpus, qid in INVENTORY:
        qs = json.load(open(f"tests/eval-data/{corpus}/queries.json"))  # noqa: ASYNC230 — one-shot probe
        q = next(x for x in qs["queries"] if x["qid"] == qid)
        CAP.clear()
        resp = await A.answer_query(q["question"])
        # attach the FULL cited-chunk text for every ungrounded claim of the LAST verify call
        ungrounded_detail: list[dict[str, Any]] = []
        vcalls = CAP.get("verify_calls") or []
        drafts = CAP.get("drafts") or []
        if vcalls and drafts:
            last_v = vcalls[-1]
            last_d = drafts[-1]
            for idx in last_v["ungrounded"]:
                if idx < len(last_d["claims"]):
                    cl = last_d["claims"][idx]
                    ungrounded_detail.append(
                        {
                            "claim": cl["claim"],
                            "cited_chunk_id": cl["source_chunk_id"],
                            "cited_chunk_text": (CAP.get("chunks_by_id") or {}).get(
                                cl["source_chunk_id"], "(chunk not in reranked set)"
                            ),
                        }
                    )
        rows.append(
            {
                "corpus": corpus,
                "qid": qid,
                "question": q["question"],
                "expected": q.get("_expected_answer", ""),
                "final_answered": bool(resp.answered),
                "refusal_reason": resp.refusal_reason,
                "window_top10": CAP.get("window"),
                "drafts": drafts,
                "verify_calls": vcalls,
                "relevance_calls": CAP.get("relevance_calls"),
                "ungrounded_detail": ungrounded_detail,
            }
        )
        json.dump(rows, open(OUT, "w"), indent=1)  # noqa: ASYNC230 — incremental flush
        gate = (
            "no-draft"
            if not drafts
            else (
                "relevance"
                if CAP.get("relevance_calls") and not CAP["relevance_calls"][-1]["responsive"]
                else ("verify" if vcalls and not vcalls[-1]["grounded"] else "answered/other")
            )
        )
        print(f"{qid:24} answered={resp.answered} gate~{gate}", flush=True)
    print(f"AUTOPSY DONE -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
