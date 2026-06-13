"""Summary-scope guard KILL-TEST (audit-17 semantic-seam Step 1).

The two-stage guard design (lexical span TRIGGER + bge-reranker-v2-m3 span-vs-cited-chunk
CONFIRMATION) rests on ONE load-bearing unknown: does the cross-encoder separate
"subject genuinely ABSENT from the cited evidence" (the ar-12/tg-13 breaches) from
"subject PARAPHRASED in the cited evidence" (the 20 recorded v3 false-positives), or
does topical halo ('graphics segment' vs any NVIDIA financial chunk) compress the gap?
Relevance is NOT entailment — this probe measures whether it is close enough here.

Two subcommands:

  capture  — LIVE (daemon + device-pinned CPU env): run the FP-class qids through the
             real `answer_query` with observe-only recorders (the fr_autopsy pattern),
             compute the v3 unsupported-subject SPANS from the final draft (query AND
             summary content-bigrams absent from every claim AND every cited-chunk
             text — git fe5a833, reshaped span-extractor), and freeze
             (qid, question, summary, spans, cited chunks) tuples.

             MEMEX_MODELS__CO_RESIDENCE_MODE=manual MEMEX_MODELS__EMBEDDER_DEVICE=cpu \
             MEMEX_MODELS__RERANKER_DEVICE=cpu \
             uv run python scripts/scope_guard_span_probe.py capture /tmp/scope_probe_fp.json

  score    — CPU-only, no daemon: score every (span, cited chunk) pair on BOTH sides —
             breach pairs are built from the frozen audit-17 traces (summaries/claims
             pinned below; cited chunk text fetched from search.sqlite by id suffix) —
             with (a) bge CrossEncoder, span as QUERY vs `{chunk_title} — {text}`
             [the guard's stage-2 form] and bare text [sensitivity arm];
             (b) EmbeddingGemma cosine + the delta variant (cos(span,chunk) −
             cos(full_query,chunk)) [the cheap fallback arm]. Prints both score
             distributions per arm + the separation margin + the kill verdict.

             uv run python scripts/scope_guard_span_probe.py score \
                 /tmp/scope_probe_fp.json /tmp/scope_probe_scores.json

KILL CONDITION (per the audit-17 synthesis): if max(breach per-span max-over-cited
score) is NOT clearly below min(FP per-span max-over-cited score) — overlap or sliver
margin — the bge arm is dead and the design pivots to the entailment-checker arm
(HHEM/MiniCheck) BEFORE any capture/integration spend.

Read-only: no production code, no vault writes. Artifacts → docs/audits/data-17-scope-calibration/.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- v3 span extractor
# Verbatim from the reverted guard (git fe5a833 core/text.py), reshaped to return the
# offending spans instead of a bool. Frozen here so the probe measures the EXACT
# trigger the two-stage design would restore.

_SUBJECT_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]+")
_SUBJECT_STOP = frozenset(
    {"the", "was", "what", "which", "according", "their", "this",
     "that", "does", "from", "with", "have", "has", "are", "is", "for", "and", "of",
     "in", "to", "on", "a", "an", "per", "by", "at", "as", "its", "it", "how", "into"}
)


def _content_bigrams(text: str) -> set[tuple[str, str]]:
    toks = [t for t in _SUBJECT_TOKEN_RE.findall(text.lower()) if t not in _SUBJECT_STOP]
    return set(itertools.pairwise(toks))


def unsupported_spans(
    query: str, summary: str, claims: list[str], evidence_texts: list[str]
) -> list[str]:
    """The v3 trigger as a span extractor: query∩summary content bigrams absent from
    every claim AND every cited-chunk text. Empty list = trigger does not fire."""
    if not claims:
        return []
    qs = _content_bigrams(query) & _content_bigrams(summary)
    if not qs:
        return []
    supported: set[tuple[str, str]] = set()
    for c in claims:
        supported |= _content_bigrams(c)
    for t in evidence_texts:
        supported |= _content_bigrams(t)
    return sorted(" ".join(b) for b in (qs - supported))


# ---------------------------------------------------------------- frozen breach traces
# From /tmp/ar12_full.json + /tmp/tg13_full.json (now docs/audits/data-17-scope-calibration/
# raw/), captured 2026-06-11 under the mxbai env, deterministic 2/2. The cited chunk ids
# are last-12 suffixes; `score` resolves them against search.sqlite.

BREACHES: list[dict[str, Any]] = [
    {
        "qid": "annual-report-12",
        "question": "What was the gross margin of NVIDIA's Graphics segment in fiscal 2026?",
        "summary": (
            "The gross margin for NVIDIA's Graphics segment in fiscal 2026 was 71.1%, "
            "as stated in the company's 2026 Annual Review."
        ),
        "claims": ["Gross margin was 71.1% in fiscal year 2026"],
        "cited_suffixes": ["b#6688b1c5c3"],
    },
    {
        "qid": "technical-guidelines-13",
        "question": (
            "According to the developer guidelines, what is the exact maximum line "
            "length in characters that the coding standards enforce?"
        ),
        "summary": (
            "The developer guidelines specify a maximum line length of 120 characters "
            "for the TUI log layer, enforced by the _DEFAULT_MAX_LEN constant in log_layer.rs."
        ),
        "claims": ["The maximum line length enforced is 120 characters."],
        "cited_suffixes": ["r#a270b0ee35"],
    },
]

# The recorded v3 mini-sweep FALSE-POSITIVE class (audit-17, 2026-06-12): answered-gold
# queries whose trigger fired because the cited evidence PARAPHRASES/recases the subject.
# Re-captured here LIVE under the SHIPPED bge default (the env any guard ships in first).
FP_QIDS: list[tuple[str, str]] = [
    ("annual-report", "annual-report-01"),
    ("annual-report", "annual-report-03"),
    ("annual-report", "annual-report-05"),
    ("nist-zero-trust", "nist-zero-trust-02"),
    ("nist-zero-trust", "nist-zero-trust-03"),
    ("nist-zero-trust", "nist-zero-trust-05"),
    ("scientific-gte", "scientific-gte-01"),
    ("scientific-gte", "scientific-gte-02"),
    ("linux-fundamentals", "linux-fundamentals-01"),
    ("linux-fundamentals", "linux-fundamentals-03"),
    ("cr350-multidoc", "cr350-xref-02"),
    ("technical-guidelines", "technical-guidelines-01"),
]

SQLITE = Path.home() / ".memex" / "vault" / ".memex" / "search.sqlite"
RERANKER_ID = "BAAI/bge-reranker-v2-m3"
EMBEDDER_ID = "google/embeddinggemma-300m"


def _chunk_title(heading_path: list[str], document_title: str) -> str:
    """Mirror of index/embed_prompts.chunk_title over raw fields (probe-local so the
    score subcommand needs no memex bootstrap)."""
    for entry in reversed(heading_path):
        if entry.strip():
            return entry.strip()[:80]
    return (document_title.strip() or "none")[:80]


def fetch_chunks_by_suffix(suffixes: list[str]) -> dict[str, dict[str, Any]]:
    db = sqlite3.connect(str(SQLITE))
    out: dict[str, dict[str, Any]] = {}
    try:
        for suf in suffixes:
            rows = db.execute(
                """
                SELECT m.chunk_id, m.document_title, COALESCE(m.full_text, f.text), m.heading_path
                FROM chunks_meta m JOIN chunks_fts f ON f.chunk_id = m.chunk_id
                WHERE m.chunk_id LIKE ?
                """,
                (f"%{suf}",),
            ).fetchall()
            if len(rows) != 1:
                print(f"[score] WARNING: suffix {suf} matched {len(rows)} chunks", flush=True)
            if rows:
                cid, title, text, hp = rows[0]
                out[suf] = {
                    "chunk_id": cid,
                    "text": text,
                    "title": _chunk_title(hp.split(" > ") if hp else [], title),
                }
    finally:
        db.close()
    return out


# ---------------------------------------------------------------- capture (live, daemon)


async def capture(out_path: str) -> None:
    import memex.agents.answering as A
    from memex.cli.bootstrap import bootstrap

    bootstrap()
    _real_answer = A.answer
    cap: dict[str, Any] = {}

    async def rec_answer(state: Any) -> Any:
        update = await _real_answer(state)
        draft = update.get("draft")
        cap["draft"] = {
            "summary": getattr(draft, "summary", None),
            "claims": [
                {"claim": c.claim, "source_chunk_id": c.source_chunk_id}
                for c in (getattr(draft, "claims", None) or [])
            ],
        }
        cap["window"] = [
            {
                "chunk_id": c.chunk_id,
                "title": _chunk_title(c.heading_path, c.document_title),
                "text": c.text,
            }
            for c in state.reranked
        ]
        return update

    A.answer = rec_answer
    A.reset_compiled_graph()

    questions: dict[str, str] = {}
    for corpus, qid in FP_QIDS:
        qs = json.load(open(f"tests/eval-data/{corpus}/queries.json"))["queries"]  # noqa: ASYNC230 — one-shot probe
        questions[qid] = next(q["question"] for q in qs if q["qid"] == qid)

    rows: list[dict[str, Any]] = []
    for corpus, qid in FP_QIDS:
        cap.clear()
        t0 = time.monotonic()
        resp = await A.answer_query(questions[qid])
        draft = cap.get("draft") or {}
        window = {c["chunk_id"]: c for c in cap.get("window") or []}
        cited: list[dict[str, Any]] = []
        for cl in draft.get("claims") or []:
            ch = window.get(cl["source_chunk_id"])
            cited.append(
                {
                    "chunk_id": cl["source_chunk_id"],
                    "title": ch["title"] if ch else None,
                    "text": ch["text"] if ch else None,  # None = dangling → no-support
                }
            )
        spans = unsupported_spans(
            questions[qid],
            draft.get("summary") or "",
            [c["claim"] for c in draft.get("claims") or []],
            [c["text"] for c in cited if c["text"]],
        )
        rows.append(
            {
                "qid": qid,
                "corpus": corpus,
                "question": questions[qid],
                "answered": bool(resp.answered),
                "refusal": (resp.refusal_reason or "")[:160],
                "summary": draft.get("summary"),
                "claims": [c["claim"] for c in draft.get("claims") or []],
                "cited": cited,
                "spans": spans,
                "elapsed_s": round(time.monotonic() - t0, 1),
            }
        )
        json.dump(rows, open(out_path, "w"), indent=1, ensure_ascii=False)  # noqa: ASYNC230 — incremental flush
        print(
            f"[capture] {qid:28} answered={resp.answered} spans={spans} "
            f"({rows[-1]['elapsed_s']}s)",
            flush=True,
        )
    fired = [r for r in rows if r["answered"] and r["spans"]]
    print(
        f"[capture] DONE: {len(rows)} run, {sum(r['answered'] for r in rows)} answered, "
        f"{len(fired)} trigger-fired (FP pairs available)",
        flush=True,
    )


# ---------------------------------------------------------------- score (CPU, no daemon)


def score(fp_path: str, out_path: str) -> None:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    # ---- assemble pairs: side, qid, span, [cited {title,text}] ----
    cases: list[dict[str, Any]] = []

    by_suffix = fetch_chunks_by_suffix(
        [s for b in BREACHES for s in b["cited_suffixes"]]
    )
    for b in BREACHES:
        cited = [by_suffix[s] for s in b["cited_suffixes"] if s in by_suffix]
        if len(cited) != len(b["cited_suffixes"]):
            print(f"[score] FATAL: breach {b['qid']} cited chunk not resolved", flush=True)
            sys.exit(2)
        spans = unsupported_spans(
            b["question"], b["summary"], b["claims"], [c["text"] for c in cited]
        )
        if not spans:
            print(f"[score] FATAL: v3 trigger does NOT fire on breach {b['qid']}", flush=True)
            sys.exit(2)
        cases.append(
            {"side": "BREACH", "qid": b["qid"], "question": b["question"],
             "spans": spans, "cited": cited}
        )

    fp_rows = json.load(open(fp_path))
    for r in fp_rows:
        if not (r["answered"] and r["spans"]):
            continue
        cited = [c for c in r["cited"] if c["text"]]
        if not cited:
            continue
        cases.append(
            {"side": "FP", "qid": r["qid"], "question": r["question"],
             "spans": r["spans"], "cited": cited}
        )

    n_fp = sum(1 for c in cases if c["side"] == "FP")
    print(f"[score] {len(cases)} cases ({n_fp} FP, {len(cases) - n_fp} breach)", flush=True)
    if n_fp < 4:
        print("[score] FATAL: <4 FP cases — capture more before judging", flush=True)
        sys.exit(2)

    # ---- arm 1: bge CrossEncoder, span as query ----
    t0 = time.monotonic()
    ce = CrossEncoder(RERANKER_ID, device="cpu")
    print(f"[score] loaded {RERANKER_ID} in {time.monotonic() - t0:.1f}s", flush=True)

    results: list[dict[str, Any]] = []
    for case in cases:
        for span in case["spans"]:
            pairs_titled = [(span, f"{c['title']} — {c['text']}") for c in case["cited"]]
            pairs_bare = [(span, c["text"]) for c in case["cited"]]
            s_titled = [float(x) for x in ce.predict(pairs_titled, batch_size=4)]
            s_bare = [float(x) for x in ce.predict(pairs_bare, batch_size=4)]
            results.append(
                {
                    "side": case["side"],
                    "qid": case["qid"],
                    "span": span,
                    "ce_titled_max": max(s_titled),
                    "ce_bare_max": max(s_bare),
                    "ce_titled_all": s_titled,
                    "n_cited": len(case["cited"]),
                }
            )
            print(
                f"[score] {case['side']:6} {case['qid']:24} span={span!r:34} "
                f"ce_titled_max={max(s_titled):7.3f} ce_bare_max={max(s_bare):7.3f}",
                flush=True,
            )

    # ---- arm 2: EmbeddingGemma cosine (+ delta vs full-query) — best-effort ----
    try:
        t0 = time.monotonic()
        emb = SentenceTransformer(EMBEDDER_ID, device="cpu")
        print(f"[score] loaded {EMBEDDER_ID} in {time.monotonic() - t0:.1f}s", flush=True)
        import numpy as np

        def _cos(a: Any, b: Any) -> float:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        for case in cases:
            docs = [f"title: {c['title']} | text: {c['text']}" for c in case["cited"]]
            d_emb = emb.encode(docs)
            q_full = emb.encode([case["question"]], prompt_name="query")[0]
            for span in case["spans"]:
                q_span = emb.encode([span], prompt_name="query")[0]
                cos_span = max(_cos(q_span, d) for d in d_emb)
                cos_full = max(_cos(q_full, d) for d in d_emb)
                for row in results:
                    if row["qid"] == case["qid"] and row["span"] == span:
                        row["emb_cos_max"] = round(cos_span, 4)
                        row["emb_delta"] = round(cos_span - cos_full, 4)
    except Exception as e:  # fallback arm is best-effort by design
        print(f"[score] embedder arm skipped: {e}", flush=True)

    # ---- distributions + margin + verdict ----
    json.dump(results, open(out_path, "w"), indent=1, ensure_ascii=False)

    def verdict(key: str, lower_is_breach: bool = True) -> None:
        br = sorted(r[key] for r in results if r["side"] == "BREACH" and key in r)
        fp = sorted(r[key] for r in results if r["side"] == "FP" and key in r)
        if not br or not fp:
            return
        margin = min(fp) - max(br) if lower_is_breach else min(br) - max(fp)
        print(f"\n=== {key} ===")
        print(f"  BREACH (n={len(br)}): {[round(x, 3) for x in br]}")
        print(f"  FP     (n={len(fp)}): {[round(x, 3) for x in fp]}")
        print(f"  margin (min FP − max BREACH) = {margin:.3f}")
        if margin > 0:
            print(f"  → SEPARATES (every breach span below every FP span by {margin:.3f})")
        else:
            overlap = sum(1 for x in fp if x <= max(br))
            print(f"  → OVERLAP: {overlap}/{len(fp)} FP spans at-or-below the top breach span")

    verdict("ce_titled_max")
    verdict("ce_bare_max")
    verdict("emb_cos_max")
    verdict("emb_delta")
    print(f"\n[score] rows → {out_path}")


# ------------------------------------------------- hhem (CPU, no daemon) — the pivot arm
# The bge span-as-query arm measured DEAD (2026-06-12: margin −0.961, ar-12 spans score
# 0.71–0.96 on topical halo while 9 FP spans score ~0.0 — inversion, not sliver). The
# pre-registered pivot: HHEM-2.1-Open entailment over the SUMMARY VERBATIM (premise =
# all cited chunks). Its own kill condition: an overlap-biased scorer that passes the
# ~2-token-delta breach summaries ("for NVIDIA's Graphics segment" injected into an
# otherwise fully-supported sentence) dies here too.

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý])")


def _checker_cases(fp_path: str) -> list[dict[str, Any]]:
    """The shared 14-case set (2 breach + 12 FP) for summary-level checkers, each with
    raw and TABLE-LINEARIZED premise variants (the audit-18 §3c mitigation cell —
    vanilla NLI false-lowed on GFM-table premises; production verify renders linearized
    via `linearize_gfm_tables`, so the eventual integration point sees that form)."""
    from memex.core.table_linearize import linearize_gfm_tables

    cases: list[dict[str, Any]] = []
    # The breach chunk TEXTS are frozen alongside the FP tuples (a reindex churns
    # chunk_ids — the chart-types 06-01 lesson); live fetch is the loud fallback.
    frozen_path = Path(fp_path).parent / "breach_chunks_frozen.json"
    if frozen_path.exists():
        by_suffix = json.load(open(frozen_path))
    else:
        print(f"[checker] WARNING: {frozen_path} missing — live-fetching breach chunks "
              "from search.sqlite (chunk ids churn on reindex)", flush=True)
        by_suffix = fetch_chunks_by_suffix([s for b in BREACHES for s in b["cited_suffixes"]])
    for b in BREACHES:
        cited = [by_suffix[s] for s in b["cited_suffixes"]]
        cases.append(
            {"side": "BREACH", "qid": b["qid"], "question": b["question"],
             "summary": b["summary"], "cited": cited}
        )
    for r in json.load(open(fp_path)):
        cited = [c for c in r["cited"] if c["text"]]
        if r["answered"] and r["summary"] and cited:
            cases.append(
                {"side": "FP", "qid": r["qid"], "question": r["question"],
                 "summary": r["summary"], "cited": cited}
            )
    for case in cases:
        case["premise_raw"] = "\n\n".join(f"{c['title']} — {c['text']}" for c in case["cited"])
        case["premise_lin"] = "\n\n".join(
            f"{c['title']} — {linearize_gfm_tables(c['text'])}" for c in case["cited"]
        )
        case["sentences"] = [
            s.strip() for s in _SENT_SPLIT_RE.split(case["summary"]) if s.strip()
        ]
    return cases


def _checker_report(results: list[dict[str, Any]], keys: tuple[str, ...], out_path: str,
                    tag: str) -> None:
    json.dump(results, open(out_path, "w"), indent=1, ensure_ascii=False)
    for key in keys:
        br = sorted(r[key] for r in results if r["side"] == "BREACH" and key in r)
        fp = sorted(r[key] for r in results if r["side"] == "FP" and key in r)
        if not br or not fp:
            continue
        margin = min(fp) - max(br)
        print(f"\n=== {key} (low = unsupported) ===")
        print(f"  BREACH (n={len(br)}): {br}")
        print(f"  FP     (n={len(fp)}): {fp}")
        print(f"  margin (min FP − max BREACH) = {margin:.3f}")
        print(
            f"  → {'SEPARATES' if margin > 0 else 'OVERLAP: ' + str(sum(1 for x in fp if x <= max(br))) + f'/{len(fp)} FP at-or-below top breach'}"
        )
    print(f"\n[{tag}] rows → {out_path}")


_ATTRSCORE_PROMPT = (
    "As an Attribution Validator, your task is to verify whether a given reference can "
    "support the given claim. A claim can be either a plain sentence or a question "
    "followed by its answer. Specifically, your response should clearly indicate the "
    "relationship: Attributable, Contradictory or Extrapolatory. A contradictory error "
    "occurs when you can infer that the answer contradicts the fact presented in the "
    "context, while an extrapolatory error means that you cannot infer the correctness "
    "of the answer based on the information provided in the context. \n\n"
    "Claim: {claim} \n Reference: {reference}"
)


def attrscore_arm(fp_path: str, out_path: str) -> None:
    """osunlp/attrscore-flan-t5-large (apache-2.0) — the design-§6 K4 windfall probe:
    the only off-the-shelf checkpoint with attribution-error supervision (3-way
    Attributable/Contradictory/Extrapolatory, Yue et al. 2023). Prompt transcribed
    VERBATIM from the AttrScore repo README. Claim = question + answer (the repo's QA
    form) with a summary-only variant; score = P(Attributable) over the three labels'
    first decoder tokens (single step, the MiniCheck scoring shape); the generated
    label is recorded alongside for honesty."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    ckpt = "osunlp/attrscore-flan-t5-large"
    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
    model.eval()
    label_ids = [tok(lbl, add_special_tokens=False).input_ids[0]
                 for lbl in ("Attributable", "Contradictory", "Extrapolatory")]
    assert len(set(label_ids)) == 3, f"label first-tokens collide: {label_ids}"  # noqa: S101 — probe invariant
    print(f"[attrscore] loaded {ckpt} in {time.monotonic() - t0:.1f}s "
          f"label_ids={label_ids}", flush=True)

    def at_score(reference: str, claim: str) -> tuple[float, str]:
        text = _ATTRSCORE_PROMPT.format(claim=claim, reference=reference)
        enc = tok(text, max_length=2048, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                decoder_input_ids=torch.zeros((1, 1), dtype=torch.long),
            )
        logits = out.logits.squeeze(1)[:, torch.tensor(label_ids)]
        probs = torch.softmax(logits, dim=-1)[0]
        label = ("Attributable", "Contradictory", "Extrapolatory")[int(probs.argmax())]
        return float(probs[0]), label

    results: list[dict[str, Any]] = []
    for case in _checker_cases(fp_path):
        row: dict[str, Any] = {"side": case["side"], "qid": case["qid"]}
        for label, premise in (("raw", case["premise_raw"]), ("lin", case["premise_lin"])):
            p_qa, lbl_qa = at_score(premise, f"{case['question']} {case['summary']}")
            p_noq, lbl_noq = at_score(premise, case["summary"])
            row[f"at_p_attr_qa_{label}"] = round(p_qa, 4)
            row[f"at_label_qa_{label}"] = lbl_qa
            row[f"at_p_attr_noq_{label}"] = round(p_noq, 4)
            row[f"at_label_noq_{label}"] = lbl_noq
        results.append(row)
        print(
            f"[attrscore] {case['side']:6} {case['qid']:24} "
            f"qa_raw={row['at_p_attr_qa_raw']:.3f}({row['at_label_qa_raw'][:6]}) "
            f"noq_raw={row['at_p_attr_noq_raw']:.3f}({row['at_label_noq_raw'][:6]}) "
            f"qa_lin={row['at_p_attr_qa_lin']:.3f}({row['at_label_qa_lin'][:6]})",
            flush=True,
        )
    _checker_report(
        results,
        ("at_p_attr_qa_raw", "at_p_attr_noq_raw", "at_p_attr_qa_lin", "at_p_attr_noq_lin"),
        out_path, "attrscore",
    )


def hhem(fp_path: str, out_path: str) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification

    t0 = time.monotonic()
    model = AutoModelForSequenceClassification.from_pretrained(
        "vectara/hallucination_evaluation_model", trust_remote_code=True
    )
    model.eval()
    print(f"[hhem] loaded HHEM-2.1-Open in {time.monotonic() - t0:.1f}s", flush=True)

    results: list[dict[str, Any]] = []
    for case in _checker_cases(fp_path):
        row: dict[str, Any] = {"side": case["side"], "qid": case["qid"],
                               "n_sentences": len(case["sentences"])}
        for label, premise in (("raw", case["premise_raw"]), ("lin", case["premise_lin"])):
            hyps = [case["summary"], *case["sentences"]]
            with torch.no_grad():
                scores = [float(x) for x in model.predict([(premise, h) for h in hyps])]
            row[f"hhem_whole_{label}"] = round(scores[0], 4)
            row[f"hhem_min_sentence_{label}"] = round(min(scores[1:]), 4)
        results.append(row)
        print(
            f"[hhem] {case['side']:6} {case['qid']:24} "
            f"whole_raw={row['hhem_whole_raw']:.3f} min_raw={row['hhem_min_sentence_raw']:.3f} "
            f"whole_lin={row['hhem_whole_lin']:.3f} min_lin={row['hhem_min_sentence_lin']:.3f}",
            flush=True,
        )
    _checker_report(
        results,
        ("hhem_whole_raw", "hhem_min_sentence_raw", "hhem_whole_lin", "hhem_min_sentence_lin"),
        out_path, "hhem",
    )


def minicheck_arm(fp_path: str, out_path: str) -> None:
    """MiniCheck-Flan-T5-Large, faithful single-chunk reimplementation (the pip package's
    overlay env resolved a broken torch/torchvision combo, so the recipe is transcribed
    VERBATIM from minicheck/inference.py: input = 'predict: ' + doc + '</s>' + claim,
    max_length 2048, one decoder step, P(support) = softmax(logits[[3, 209]])[1], max
    over doc chunks. Chunking is a no-op here — every probe premise is far below the
    1748-word chunk size, so the single-chunk path is exactly the package's behavior."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    ckpt = "lytang/MiniCheck-Flan-T5-Large"
    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
    model.eval()
    print(f"[minicheck] loaded {ckpt} in {time.monotonic() - t0:.1f}s", flush=True)

    def mc_score(premise: str, claim: str) -> float:
        text = "predict: " + tok.eos_token.join([premise, claim])
        enc = tok(text, max_length=2048, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                decoder_input_ids=torch.zeros((1, 1), dtype=torch.long),
            )
        # token id 3 = no-support, 209 = support (the package's hardcoded label ids)
        label_logits = out.logits.squeeze(1)[:, torch.tensor([3, 209])]
        return float(torch.softmax(label_logits, dim=-1)[0, 1])

    results: list[dict[str, Any]] = []
    for case in _checker_cases(fp_path):
        row: dict[str, Any] = {"side": case["side"], "qid": case["qid"],
                               "n_sentences": len(case["sentences"])}
        for label, premise in (("raw", case["premise_raw"]), ("lin", case["premise_lin"])):
            whole = mc_score(premise, case["summary"])
            min_sent = min(mc_score(premise, s) for s in case["sentences"])
            row[f"mc_whole_{label}"] = round(whole, 4)
            row[f"mc_min_sentence_{label}"] = round(min_sent, 4)
        results.append(row)
        print(
            f"[minicheck] {case['side']:6} {case['qid']:24} "
            f"whole_raw={row['mc_whole_raw']:.3f} min_raw={row['mc_min_sentence_raw']:.3f} "
            f"whole_lin={row['mc_whole_lin']:.3f} min_lin={row['mc_min_sentence_lin']:.3f}",
            flush=True,
        )
    _checker_report(
        results,
        ("mc_whole_raw", "mc_min_sentence_raw", "mc_whole_lin", "mc_min_sentence_lin"),
        out_path, "minicheck",
    )


_LD_QA_TEMPLATE = (
    "Briefly answer the following question:\n{question}\n"
    "Bear in mind that your response should be strictly based on the following "
    "{num_passages} passages:\n{context}\n"
    'In case the passages do not contain the necessary information to answer the '
    'question, please reply with: "Unable to answer based on given passages."\n'
    "output:"
)


def lettuce_arm(fp_path: str, out_path: str, model_path: str) -> None:
    """LettuceDetect token-span detector, faithful reimplementation (the pip overlay
    env resolved a broken torch combo; recipe transcribed from
    lettucedetect/detectors/transformer.py + prompts/qa_prompt_en.txt: passages block →
    QA template → tokenizer(prompt, answer, truncation='only_first', max_length=4096)
    → token-classification → class-1 spans in the answer region, confidence = max
    class-1 prob). Two variants: question=REAL (the trained input shape — tests the
    question-echo-benign risk) and question=EMPTY (evidence-only). Score = max span
    confidence (0.0 = no span); INVERTED vs the entailment arms (high = breach)."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()
    print(f"[lettuce] loaded {model_path} in {time.monotonic() - t0:.1f}s", flush=True)

    def ld_score(passages: list[str], question: str, answer: str) -> tuple[float, list[str]]:
        ctx_block = "\n".join(f"passage {i + 1}: {p}" for i, p in enumerate(passages))
        prompt = _LD_QA_TEMPLATE.format(
            question=question, num_passages=len(passages), context=ctx_block
        )
        enc = tok(prompt, answer, truncation="only_first", max_length=4096,
                  return_offsets_mapping=True, return_tensors="pt", add_special_tokens=True)
        offsets = enc.pop("offset_mapping")[0]
        answer_start = tok(prompt, add_special_tokens=True, return_tensors="pt")[
            "input_ids"
        ].shape[1]
        with torch.no_grad():
            logits = model(**enc).logits
        preds = torch.argmax(logits, dim=-1)[0]
        probs = torch.softmax(logits, dim=-1)[0]
        answer_char_offset = (
            offsets[answer_start][0].item() if answer_start < offsets.size(0) else 0
        )
        spans: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for i in range(answer_start, preds.size(0)):
            t_start, t_end = offsets[i].tolist()
            if t_start == t_end:
                continue
            if int(preds[i]) == 1:
                conf = float(probs[i, 1])
                rel_s, rel_e = t_start - answer_char_offset, t_end - answer_char_offset
                if current is None:
                    current = {"start": rel_s, "end": rel_e, "confidence": conf}
                else:
                    current["end"] = rel_e
                    current["confidence"] = max(current["confidence"], conf)
            elif current is not None:
                spans.append(current)
                current = None
        if current is not None:
            spans.append(current)
        texts = [answer[s["start"]: s["end"]][:60] for s in spans]
        return max((s["confidence"] for s in spans), default=0.0), texts

    questions = {b["qid"]: b["question"] for b in BREACHES}
    for r in json.load(open(fp_path)):
        questions[r["qid"]] = r["question"]

    # The Memex answer style appends a provenance tail ("…, as stated in the 2026
    # Annual Review (Form 10-K).") that RAGTruth answers don't have — LD flags it as
    # baseless. The "qs" variant strips the tail BEFORE scoring: if a breach signal
    # survives only in the tail, the catch was style noise, not the binding.
    tail_re = re.compile(r",?\s+as (?:stated|noted|reported) in [^.]*\.\s*$", re.IGNORECASE)

    results: list[dict[str, Any]] = []
    for case in _checker_cases(fp_path):
        passages = [f"{c['title']} — {c['text']}" for c in case["cited"]]
        stripped = tail_re.sub(".", case["summary"])
        conf_q, spans_q = ld_score(passages, questions[case["qid"]], case["summary"])
        conf_noq, spans_noq = ld_score(passages, "", case["summary"])
        conf_qs, spans_qs = ld_score(passages, questions[case["qid"]], stripped)
        row = {
            "side": case["side"], "qid": case["qid"],
            "ld_max_conf_q": round(conf_q, 4), "ld_spans_q": spans_q[:4],
            "ld_max_conf_noq": round(conf_noq, 4), "ld_spans_noq": spans_noq[:4],
            "ld_max_conf_qs": round(conf_qs, 4), "ld_spans_qs": spans_qs[:4],
            "tail_stripped": stripped != case["summary"],
        }
        results.append(row)
        print(
            f"[lettuce] {case['side']:6} {case['qid']:24} "
            f"conf_q={conf_q:.3f} {spans_q[:2]!r} conf_noq={conf_noq:.3f} "
            f"conf_qs={conf_qs:.3f} {spans_qs[:2]!r}",
            flush=True,
        )

    json.dump(results, open(out_path, "w"), indent=1, ensure_ascii=False)
    for key in ("ld_max_conf_q", "ld_max_conf_noq", "ld_max_conf_qs"):
        br = sorted(r[key] for r in results if r["side"] == "BREACH")
        fp = sorted(r[key] for r in results if r["side"] == "FP")
        margin = min(br) - max(fp)  # INVERTED: high = hallucinated
        print(f"\n=== {key} (HIGH = hallucinated span found) ===")
        print(f"  BREACH (n={len(br)}): {br}")
        print(f"  FP     (n={len(fp)}): {fp}")
        print(f"  margin (min BREACH − max FP) = {margin:.3f}")
        print(
            f"  → {'SEPARATES' if margin > 0 else 'OVERLAP: ' + str(sum(1 for x in fp if x >= min(br))) + f'/{len(fp)} FP at-or-above bottom breach'}"
        )
    print(f"\n[lettuce] rows → {out_path}")


# ------------------------------------------------- nli (CPU, no daemon) — entailment arm
# Standard-architecture NLI (no trust_remote_code; HHEM/MiniCheck need it or a pip
# package — user decision pending). Per the audit-17 research recipe for vanilla MNLI:
# premise split per chunk-SENTENCE (+ the title as one unit, covering subject-in-heading)
# with MAX-aggregation; hypothesis = the summary verbatim AND each summary sentence.
# score = max over premise units of P(entailment). Low = unsupported framing.

NLI_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def nli(fp_path: str, out_path: str) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(NLI_ID)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_ID)
    model.eval()
    ent_idx = next(i for i, lbl in model.config.id2label.items() if "entail" in lbl.lower())
    print(f"[nli] loaded {NLI_ID} in {time.monotonic() - t0:.1f}s (ent_idx={ent_idx})", flush=True)

    cases: list[dict[str, Any]] = []
    by_suffix = fetch_chunks_by_suffix([s for b in BREACHES for s in b["cited_suffixes"]])
    for b in BREACHES:
        cases.append(
            {"side": "BREACH", "qid": b["qid"], "summary": b["summary"],
             "cited": [by_suffix[s] for s in b["cited_suffixes"]]}
        )
    for r in json.load(open(fp_path)):
        cited = [c for c in r["cited"] if c["text"]]
        if r["answered"] and r["summary"] and cited:
            cases.append({"side": "FP", "qid": r["qid"], "summary": r["summary"], "cited": cited})

    def p_entail(premises: list[str], hypothesis: str) -> float:
        best = 0.0
        with torch.no_grad():
            for i in range(0, len(premises), 8):
                batch = premises[i : i + 8]
                enc = tok(batch, [hypothesis] * len(batch), return_tensors="pt",
                          truncation=True, max_length=512, padding=True)
                probs = torch.softmax(model(**enc).logits, dim=-1)[:, ent_idx]
                best = max(best, float(probs.max()))
        return best

    results: list[dict[str, Any]] = []
    for case in cases:
        units: list[str] = []
        for c in case["cited"]:
            units.append(c["title"])
            units.extend(s.strip() for s in _SENT_SPLIT_RE.split(c["text"]) if s.strip())
            units.append(c["text"][:1800])  # whole-chunk premise unit (granularity arm)
        sentences = [s.strip() for s in _SENT_SPLIT_RE.split(case["summary"]) if s.strip()]
        whole = p_entail(units, case["summary"])
        min_sent = min(p_entail(units, s) for s in sentences)
        row = {
            "side": case["side"], "qid": case["qid"],
            "nli_whole": round(whole, 4), "nli_min_sentence": round(min_sent, 4),
            "n_premise_units": len(units), "n_sentences": len(sentences),
        }
        results.append(row)
        print(
            f"[nli] {case['side']:6} {case['qid']:24} whole={whole:.3f} "
            f"min_sent={min_sent:.3f} (units={len(units)})",
            flush=True,
        )

    json.dump(results, open(out_path, "w"), indent=1, ensure_ascii=False)
    for key in ("nli_whole", "nli_min_sentence"):
        br = sorted(r[key] for r in results if r["side"] == "BREACH")
        fp = sorted(r[key] for r in results if r["side"] == "FP")
        margin = min(fp) - max(br)
        print(f"\n=== {key} (low = unsupported) ===")
        print(f"  BREACH (n={len(br)}): {br}")
        print(f"  FP     (n={len(fp)}): {fp}")
        print(f"  margin (min FP − max BREACH) = {margin:.3f}")
        print(
            f"  → {'SEPARATES' if margin > 0 else 'OVERLAP: ' + str(sum(1 for x in fp if x <= max(br))) + f'/{len(fp)} FP at-or-below top breach'}"
        )
    print(f"\n[nli] rows → {out_path}")


# ---------------------------------------------- judge (daemon 4B) + doc-identity report
# Arm: the narrow evidence-only 4B judge (sweep2-C) — sees ONLY cited evidence + the
# span, never the summary or the question (removes the relevance gate's hallucination
# channel). Guided bool-only JSON via the daemon's OpenAI endpoint. Pre-registered kill:
# mentioned=true for the ar-12 spans on the consolidated-margin chunk at N>=2.
# Plus the DOC-IDENTITY report the probe data suggested: provenance-shaped spans
# ("according to <source>") name the cited chunk's DOCUMENT, not its body text — match
# span tokens against document_title/document_id (deterministic, report-only here).

VLLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
VLLM_MODEL = "cyankiwi/Qwen3.5-4B-AWQ-4bit"
_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"mentioned": {"type": "boolean"}},
    "required": ["mentioned"],
}


def _judge_call(evidence: str, span: str) -> bool | None:
    import urllib.request

    prompt = (
        f"EVIDENCE:\n{evidence}\n\n"
        f'Does the evidence text mention or refer to "{span}", in any wording, casing, '
        "or paraphrase? A mention of a broader or parent topic (the company in general, "
        "the document's general subject area) does NOT count — the specific subject must "
        "be present. The phrase may be a sentence fragment; judge its content words. "
        "Answer with JSON: {\"mentioned\": true|false}."
    )
    body = json.dumps(
        {
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
            "temperature": 0,
            "guided_json": _JUDGE_SCHEMA,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(  # noqa: S310 — fixed localhost vLLM endpoint
        VLLM_URL, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 — localhost
            out = json.load(resp)["choices"][0]["message"]["content"]
        return bool(json.loads(out)["mentioned"])
    except Exception as e:  # probe records the failure and moves on
        print(f"[judge] call failed: {e}", flush=True)
        return None


def binding_gate(fp_path: str, out_path: str, model_dir: str, runs: int = 2) -> None:
    """The audit-19 §4 ONE-SHOT calibration gate for a trained binding-checker
    candidate. Scores the frozen 14-case set with the candidate via the UNCHANGED
    lettuce_arm machinery (train == gate == wiring input shape), at the threshold
    FROZEN on the minted dev split (model_dir/threshold.json — never tuned here).

    PASS bar (all): both breaches fire at >= t on conf_q AND survive the
    tail-strip variant (conf_qs >= t — a catch living only in the provenance tail
    is style noise, the lesson that killed the pip checkpoint); ZERO of the 12
    FPs fire on conf_q. N=runs determinism asserted byte-stable.
    """
    meta = json.load(open(Path(model_dir) / "threshold.json"))
    t = float(meta["threshold"])
    print(f"[binding] candidate {model_dir} | frozen threshold {t} "
          f"(dev F1 {meta.get('dev_example_f1')}, dev FP {meta.get('dev_fp_rate')})",
          flush=True)
    rows_runs = []
    for _ in range(runs):
        lettuce_arm(fp_path, out_path, model_dir)
        rows_runs.append(json.load(open(out_path)))
    if rows_runs[0] != rows_runs[-1]:
        print("[binding] FATAL: non-deterministic scoring across runs — harness bug",
              flush=True)
        sys.exit(2)
    rows = rows_runs[0]
    br = {r["qid"]: r for r in rows if r["side"] == "BREACH"}
    fp = {r["qid"]: r for r in rows if r["side"] == "FP"}
    missed = {q: r["ld_max_conf_q"] for q, r in br.items() if r["ld_max_conf_q"] < t}
    tail_only = {q: r["ld_max_conf_qs"] for q, r in br.items()
                 if r["ld_max_conf_q"] >= t and r["ld_max_conf_qs"] < t}
    fired = {q: r["ld_max_conf_q"] for q, r in fp.items() if r["ld_max_conf_q"] >= t}
    margin = (min(r["ld_max_conf_q"] for r in br.values())
              - max(r["ld_max_conf_q"] for r in fp.values()))
    print(f"\n[binding] breaches missed (conf_q < t): {missed or 'none'}")
    print(f"[binding] breaches tail-only (conf_qs < t): {tail_only or 'none'}")
    print(f"[binding] FPs fired (conf_q >= t): {fired or 'none'} ({len(fired)}/12)")
    print(f"[binding] margin min(BREACH) - max(FP) on conf_q = {margin:.3f}"
          + (" [KNIFE-EDGE]" if 0 < margin < 0.05 else ""))
    ok = not missed and not tail_only and not fired
    print(f"\n[binding] GATE: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


def _doc_meta(chunk_ids: list[str]) -> dict[str, tuple[str, str]]:
    db = sqlite3.connect(str(SQLITE))
    try:
        out = {}
        for cid in chunk_ids:
            row = db.execute(
                "SELECT document_id, document_title FROM chunks_meta WHERE chunk_id = ?",
                (cid,),
            ).fetchone()
            if row:
                out[cid] = (row[0], row[1])
        return out
    finally:
        db.close()


def judge(fp_path: str, out_path: str, n_runs: int = 2) -> None:
    cases: list[dict[str, Any]] = []
    by_suffix = fetch_chunks_by_suffix([s for b in BREACHES for s in b["cited_suffixes"]])
    for b in BREACHES:
        cited = [by_suffix[s] for s in b["cited_suffixes"]]
        spans = unsupported_spans(b["question"], b["summary"], b["claims"],
                                  [c["text"] for c in cited])
        cases.append({"side": "BREACH", "qid": b["qid"], "spans": spans, "cited": cited})
    for r in json.load(open(fp_path)):
        cited = [c for c in r["cited"] if c["text"]]
        if r["answered"] and r["spans"] and cited:
            cases.append({"side": "FP", "qid": r["qid"], "spans": r["spans"], "cited": cited})

    doc_meta = _doc_meta([c["chunk_id"] for case in cases for c in case["cited"]])
    results: list[dict[str, Any]] = []
    for case in cases:
        evidence = "\n\n".join(f"{c['title']} — {c['text']}" for c in case["cited"])
        doc_blob = " ".join(
            f"{doc_meta.get(c['chunk_id'], ('', ''))[0]} {doc_meta.get(c['chunk_id'], ('', ''))[1]}"
            for c in case["cited"]
        ).lower()
        doc_toks = set(_SUBJECT_TOKEN_RE.findall(doc_blob))
        for span in case["spans"]:
            votes = [_judge_call(evidence, span) for _ in range(n_runs)]
            span_toks = [t for t in _SUBJECT_TOKEN_RE.findall(span) if t not in _SUBJECT_STOP]
            doc_match = bool(span_toks) and all(t in doc_toks for t in span_toks)
            results.append(
                {"side": case["side"], "qid": case["qid"], "span": span,
                 "judge_votes": votes, "doc_identity_match": doc_match}
            )
            print(
                f"[judge] {case['side']:6} {case['qid']:24} span={span!r:34} "
                f"votes={votes} doc_id_match={doc_match}",
                flush=True,
            )

    json.dump(results, open(out_path, "w"), indent=1, ensure_ascii=False)
    br_true = [r for r in results if r["side"] == "BREACH" and all(r["judge_votes"])]
    fp_false = [r for r in results if r["side"] == "FP" and not any(v for v in r["judge_votes"] if v)]
    unstable = [r for r in results if len(set(filter(lambda v: v is not None, r["judge_votes"]))) > 1]
    print("\n=== 4B judge (want: BREACH spans false, FP spans true) ===")
    print(f"  breach spans judged MENTIONED (misses):      {len(br_true)} "
          f"{[(r['qid'], r['span']) for r in br_true]}")
    print(f"  FP spans judged NOT-mentioned (false fires): {len(fp_false)} "
          f"{[(r['qid'], r['span']) for r in fp_false]}")
    print(f"  unstable across N={n_runs}: {len(unstable)} {[(r['qid'], r['span']) for r in unstable]}")
    print(f"\n[judge] rows → {out_path}")


# --------------------------------------------- provenance (offline L0, no daemon/LLM)
# The doc-identity backstop's offline validation (audit-18 §9): replay the PRODUCTION
# decision (memex.core.text helpers, the same order as answering._provenance_scope_
# violation) over (a) the frozen 14-case calibration set — tg-13 must FIRE, the other
# 13 must PASS — and (b) EVERY eval query across all corpora, predicting the verdict
# against each ANS query's GOLD chunks: any predicted fire on an ANS query is a
# false-refusal bug to fix BEFORE a live run. Exit nonzero unless both gates hold.


def _doc_blob_for_chunks(db: sqlite3.Connection, chunk_ids: list[str]) -> list[str]:
    """Per cited chunk: 'doc_id title heading_path' (the production blob shape);
    falls back to the chunk-id's doc prefix when the chunk no longer resolves."""
    blobs = []
    for cid in chunk_ids:
        row = db.execute(
            "SELECT document_id, document_title, heading_path FROM chunks_meta WHERE chunk_id=?",
            (cid,),
        ).fetchone()
        if row:
            heads = row[2].replace(" > ", " ") if row[2] else ""
            blobs.append(f"{row[0]} {row[1] or ''} {heads}")
        else:
            blobs.append(cid.split("#")[0])
    return blobs


def provenance_l0(fp_path: str) -> None:
    from memex.core.text import extract_provenance_source, provenance_tokens_match

    db = sqlite3.connect(str(SQLITE))
    identities = db.execute(
        "SELECT DISTINCT document_id, document_title FROM chunks_meta"
    ).fetchall()

    def verdict(question: str, cited_chunk_ids: list[str]) -> str | None:
        extracted = extract_provenance_source(question)
        if extracted is None or not cited_chunk_ids:
            return None
        phrase, tokens = extracted
        for blob in _doc_blob_for_chunks(db, cited_chunk_ids):
            if provenance_tokens_match(tokens, blob):
                return None
        named = [t or d for d, t in identities if provenance_tokens_match(tokens, f"{d} {t}")]
        if not named:
            return None
        return f"FIRE: '{phrase}' names {named[0]!r}"

    failures = 0

    # ---- leg (a): the frozen 14-case set ----
    print("=== L0a: frozen calibration set ===")
    for b in BREACHES:
        by_suffix = fetch_chunks_by_suffix(b["cited_suffixes"])
        cids = [by_suffix[s]["chunk_id"] for s in b["cited_suffixes"]]
        v = verdict(b["question"], cids)
        want_fire = b["qid"] == "technical-guidelines-13"
        ok = (v is not None) == want_fire
        failures += 0 if ok else 1
        print(f"  [{'OK' if ok else 'FAIL'}] {b['qid']:24} want={'FIRE' if want_fire else 'PASS'} got={v or 'PASS'}")
    for r in json.load(open(fp_path)):
        cids = [c["chunk_id"] for c in r["cited"] if c["text"]]
        v = verdict(r["question"], cids)
        ok = v is None
        failures += 0 if ok else 1
        print(f"  [{'OK' if ok else 'FAIL'}] {r['qid']:24} want=PASS got={v or 'PASS'}")

    # ---- leg (b): every eval query, verdict vs GOLD chunks ----
    print("\n=== L0b: all-queries sweep (predicted verdict vs gold) ===")
    n_q = n_extracted = n_ans_fire = n_cf_fire = n_stale = 0
    for qfile in sorted(Path("tests/eval-data").glob("*/queries.json")):
        data = json.load(open(qfile))
        for q in data.get("queries", []):
            question = q.get("question")
            if not question:
                continue
            n_q += 1
            extracted = extract_provenance_source(question)
            if extracted is None:
                continue
            n_extracted += 1
            gold = list(q.get("relevant_chunk_ids") or [])
            is_cf = q.get("_answer_type") == "counterfactual" or not gold
            resolved = [
                g for g in gold
                if db.execute(
                    "SELECT 1 FROM chunks_meta WHERE chunk_id=?", (g,)
                ).fetchone()
            ]
            if gold and not resolved:
                n_stale += 1
            v = verdict(question, resolved or gold)
            tag = "CF" if is_cf else "ANS"
            if v is not None:
                if is_cf:
                    n_cf_fire += 1
                else:
                    n_ans_fire += 1
                    failures += 1
                print(f"  [{'INFO' if is_cf else 'FAIL'}] {tag} {qfile.parent.name}/{q['qid']}: {v}")
            else:
                print(f"  [ok]   {tag} {qfile.parent.name}/{q['qid']}: extracted={extracted[1]} -> PASS")
    db.close()
    print(
        f"\nL0b: {n_q} queries, {n_extracted} extracted, "
        f"{n_ans_fire} ANS predicted-fires (gate: 0), {n_cf_fire} CF fires (informational), "
        f"{n_stale} stale-gold (verdict on raw ids)"
    )
    print(f"\nL0 {'PASS — PROMOTE TO MINI-SWEEP' if failures == 0 else f'FAIL ({failures})'}")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "capture":
        asyncio.run(capture(sys.argv[2] if len(sys.argv) > 2 else "/tmp/scope_probe_fp.json"))  # noqa: S108 — probe artifact
    elif cmd == "score":
        score(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/scope_probe_fp.json",  # noqa: S108 — probe artifact
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/scope_probe_scores.json",  # noqa: S108 — probe artifact
        )
    elif cmd == "hhem":
        hhem(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/scope_probe_fp.json",  # noqa: S108 — probe artifact
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/scope_probe_hhem.json",  # noqa: S108 — probe artifact
        )
    elif cmd == "minicheck":
        minicheck_arm(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/scope_probe_fp.json",  # noqa: S108 — probe artifact
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/scope_probe_minicheck.json",  # noqa: S108 — probe artifact
        )
    elif cmd == "attrscore":
        attrscore_arm(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/scope_probe_fp.json",  # noqa: S108 — probe artifact
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/scope_probe_attrscore.json",  # noqa: S108 — probe artifact
        )
    elif cmd == "provenance":
        provenance_l0(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/scope_probe_fp.json",  # noqa: S108 — probe artifact
        )
    elif cmd == "lettuce":
        lettuce_arm(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/scope_probe_fp.json",  # noqa: S108 — probe artifact
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/scope_probe_lettuce.json",  # noqa: S108 — probe artifact
            sys.argv[4] if len(sys.argv) > 4 else "KRLabsOrg/lettucedect-base-modernbert-en-v1",
        )
    elif cmd == "binding":
        binding_gate(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/scope_probe_fp.json",  # noqa: S108 — probe artifact
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/scope_probe_binding.json",  # noqa: S108 — probe artifact
            sys.argv[4],
        )
    elif cmd == "nli":
        nli(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/scope_probe_fp.json",  # noqa: S108 — probe artifact
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/scope_probe_nli.json",  # noqa: S108 — probe artifact
        )
    elif cmd == "judge":
        judge(
            sys.argv[2] if len(sys.argv) > 2 else "/tmp/scope_probe_fp.json",  # noqa: S108 — probe artifact
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/scope_probe_judge.json",  # noqa: S108 — probe artifact
        )
    else:
        print(__doc__)
        sys.exit(1)
