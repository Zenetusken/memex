"""Mint presence-preserving binding-fabrication training data from the vault
(audit-19 design §2 — the content-class checker increment).

VERDICT — DO NOT RE-WALK (audit-19 §9 K2): the checker this data trains was
measured a STRUCTURAL negative (question-echo entanglement) and the companion
answer@v6 generation lever was reverted (§10). Re-runnable artifact only, for a
future masked-subject / complete-evidence-entailment design — not a tuning revisit.

Every negative satisfies the core invariant: the swapped-in subject occurs
VERBATIM in the passage, the predicate+value is true of some OTHER subject in
the same passage, and the claim is false ONLY because of the rebind. Within a
matched pair the lexical overlap with the context is near-identical, so the
model can only reduce loss by learning binding (the failure mode all 8 measured
zero-shot arms share — audit-18 §9 + the attrscore row).

Phase T (this module's deterministic core, LLM-free): GFM-table mints —
  A2  positives: verbalize (row label, header, cell) via the template bank
  NEG-row: the cell re-attributed to a sibling row's label (guard: target cell
           differs — the accidental-truth discard F2)
  NEG-col: the cell re-attributed to a sibling column's header (same guard)
  C   hard positives: a TRUE claim about every rebind target (kills the
      inverse-presence shortcut — a subject being a frequent rebind target must
      not itself signal "breach")
  A3  unit-transform positives: %→percent / $NB→billion word rewrites of true
      claims (these ARE the calibration FP modes; they must exist as positives)

Labels are LettuceDetect HallucinationSample-compatible: char spans over the
answer, span = the SWAPPED SLOT (the train script can widen to whole-claim for
the §3 span-target A/B). Prompt format == the lettuce_arm inference shape
(scope_guard_span_probe._LD_QA_TEMPLATE + "passage N: {title} — {text}") so
train == gate == wiring by construction.

Determinism: zero randomness — template rotation and cell sampling key off
sha256 of stable ids; reruns are byte-identical.

Leak rule (design §2.6, hard): docs cited by ANY frozen calibration tuple are
excluded; a token-Jaccard >0.8 claim vs any calibration claim/summary aborts.

Usage:
    uv run python scripts/mint_binding_data.py --phase t --out /tmp/mints_t.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from memex.index.table_store import extract_tables

CAL_DIR = Path("docs/audits/data-17-scope-calibration")
SQLITE = Path.home() / ".memex" / "vault" / ".memex" / "search.sqlite"

# Mirrors scope_guard_span_probe._LD_QA_TEMPLATE — train/gate/wiring parity.
LD_QA_TEMPLATE = (
    "Briefly answer the following question:\n{question}\n"
    "Bear in mind that your response should be strictly based on the following "
    "{num_passages} passages:\n{context}\n"
    'In case the passages do not contain the necessary information to answer the '
    'question, please reply with: "Unable to answer based on given passages."\n'
    "output:"
)

# Metric-flavored templates fit NUMERIC cell values; text-valued cells get the
# neutral bank (a "Long Option reached COMMENT" tell would let the model key on
# template-value mismatch instead of binding).
CLAIM_TEMPLATES = {
    ("en", "num"): [
        "The {header} for {label} was {value}.",
        "{label}'s {header} was {value}.",
        "The {header} of {label} came to {value}.",
        "{label} recorded a {header} of {value}.",
        "For {label}, the {header} reached {value}.",
        "The {header} of {label} stood at {value}.",
        "{label} reported a {header} of {value}.",
        "{value} was the {header} reported for {label}.",
        "In the period shown, {label} had a {header} of {value}.",
        "The table lists a {header} of {value} for {label}.",
    ],
    ("en", "txt"): [
        "The {header} for {label} was {value}.",
        "{label} has a {header} of {value}.",
        "The {header} of {label} is {value}.",
        "For {label}, the {header} is {value}.",
        "The table lists a {header} of {value} for {label}.",
        "{value} is the {header} for {label}.",
    ],
    ("fr", "num"): [
        "Le {header} de {label} était de {value}.",
        "{label} a enregistré un {header} de {value}.",
        "Pour {label}, le {header} a atteint {value}.",
        "Le {header} pour {label} s'élevait à {value}.",
        "{label} affichait un {header} de {value}.",
        "{value} était le {header} de {label}.",
        "Sur la période indiquée, {label} avait un {header} de {value}.",
        "Le tableau indique un {header} de {value} pour {label}.",
    ],
    ("fr", "txt"): [
        "Le {header} de {label} était {value}.",
        "{label} a un {header} de {value}.",
        "Pour {label}, le {header} est {value}.",
        "Le tableau indique un {header} de {value} pour {label}.",
        "{value} est le {header} de {label}.",
        "Le {header} pour {label} est {value}.",
    ],
}

_NUM_VALUE_RE = re.compile(r"^[~≈]?[\$€]?\d[\d.,   ]*\s?(%|B|M|K|Mds|GB|MB|Gbit/s|Mbit/s)?$")


def is_numeric_value(v: str) -> bool:
    return bool(_NUM_VALUE_RE.match(v.strip()))
QUESTION_TEMPLATES = {
    "en": [
        "What was the {header} for {label}?",
        "What is the {header} of {label}?",
        "What {header} did {label} have?",
    ],
    "fr": [
        "Quel était le {header} pour {label} ?",
        "Quelle est la valeur de {header} pour {label} ?",
        "Quel {header} {label} avait-il ?",
    ],
}

_FR_HINTS = re.compile(
    r"\b(le|la|les|des|une|est|dans|pour|avec|qui|sont|été|être|aux|cette)\b", re.IGNORECASE
)
_EN_HINTS = re.compile(r"\b(the|of|and|is|for|with|that|are|was|this|from)\b", re.IGNORECASE)
_MD_NOISE = re.compile(r"[*_`]")
_TOTAL_RE = re.compile(r"^\s*(total|ensemble|somme|sum)\b", re.IGNORECASE)
_PCT_RE = re.compile(r"^(\d+(?:\.\d+)?)\s?%$")
_MONEY_AB_RE = re.compile(r"^\$(\d+(?:\.\d+)?)\s?([BM])$")
_TOKEN_RE = re.compile(r"[a-z0-9àâäçéèêëîïôöùûüÿœæ]+")


def _h(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest(), 16)


def detect_lang(text: str) -> str:
    fr = len(_FR_HINTS.findall(text))
    en = len(_EN_HINTS.findall(text))
    return "fr" if fr > en else "en"


def clean_cell(s: str) -> str:
    return _MD_NOISE.sub("", s).strip()


def render(template: str, slots: dict[str, str]) -> tuple[str, dict[str, tuple[int, int]]]:
    """Format `template` tracking the char span each slot lands on."""
    out: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    pos = 0
    for lit, field, _spec, _conv in __import__("string").Formatter().parse(template):
        out.append(lit)
        pos += len(lit)
        if field is not None:
            val = slots[field]
            spans[field] = (pos, pos + len(val))
            out.append(val)
            pos += len(val)
    return "".join(out), spans


def _tokens(s: str) -> set[str]:
    return set(_TOKEN_RE.findall(s.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def load_calibration() -> tuple[set[str], set[str], list[set[str]]]:
    """(holdout doc ids, calibration chunk ids, calibration claim/summary token sets)."""
    docs: set[str] = set()
    chunk_ids: set[str] = set()
    texts: list[set[str]] = []
    for t in json.load(open(CAL_DIR / "scope_probe_fp.json")):
        texts.append(_tokens(t["summary"]))
        texts.extend(_tokens(c) for c in t.get("claims", []))
        for c in t["cited"]:
            chunk_ids.add(c["chunk_id"])
            docs.add(c["chunk_id"].split("#")[0])
    for c in json.load(open(CAL_DIR / "breach_chunks_frozen.json")).values():
        chunk_ids.add(c["chunk_id"])
        docs.add(c["chunk_id"].split("#")[0])
    # the frozen breach summaries/claims live in the probe module
    sys.path.insert(0, str(Path(__file__).parent))
    from scope_guard_span_probe import BREACHES

    for b in BREACHES:
        texts.append(_tokens(b["summary"]))
        texts.extend(_tokens(c) for c in b["claims"])
    return docs, chunk_ids, texts


def eligible_chunks(holdout_docs: set[str]) -> list[dict[str, str]]:
    db = sqlite3.connect(str(SQLITE))
    try:
        rows = db.execute(
            """
            SELECT m.chunk_id, m.document_id, m.document_title, m.heading_path,
                   COALESCE(m.full_text, f.text)
            FROM chunks_meta m JOIN chunks_fts f ON f.chunk_id = m.chunk_id
            """
        ).fetchall()
    finally:
        db.close()
    out = []
    for cid, did, title, hp, text in rows:
        if did in holdout_docs or not text:
            continue
        heading = next((e.strip() for e in reversed((hp or "").split(" > ")) if e.strip()), "")
        out.append(
            {
                "chunk_id": cid,
                "doc_id": did,
                "title": (heading or title or "none")[:80],
                "doc_title": (title or "the document")[:90],
                "text": text,
            }
        )
    out.sort(key=lambda c: c["chunk_id"])  # determinism
    return out


_MILLIONS_RE = re.compile(r"^\$?\s?(\d{1,3}(?:,\d{3})+)$")  # "193,737" comma-grouped


def unit_transforms(value: str, lang: str, *, millions_table: bool) -> list[str]:
    """ALL supported surface forms of a cell value — these ARE the calibration FP
    modes (ar-03 '$193,737' [millions] → '$193.737 billion'; gte unit words), so they
    must exist as POSITIVES or the checker flags every legit unit rewrite."""
    out: list[str] = []
    m = _PCT_RE.match(value)
    if m:
        out.append(f"{m.group(1)} {'pour cent' if lang == 'fr' else 'percent'}")
    m = _MONEY_AB_RE.match(value)
    if m:
        word = {"B": "billion", "M": "million"}[m.group(2)]
        out.append(f"${m.group(1)} {word}")
    m = _MILLIONS_RE.match(value)
    if m and millions_table:
        n = float(m.group(1).replace(",", ""))
        if n >= 1000:
            out.append(f"${n / 1000:g} billion")  # 193,737 (M) -> $193.737 billion
        out.append(f"${n:,.0f} million")
        out.append(f"${m.group(1)}")  # the dollar-sign surface form
    return out


def _is_millions_table(table: Any) -> bool:
    blob = (table.section + " " + " ".join(table.header)).lower()
    return "million" in blob or "(in millions" in blob or "in millions" in blob


# Abstract skill-teaching positives the VAULT can't supply (the financial
# unit-transform register lives only in the held-out 10-K; the meta-provenance FP
# references doc structure). Synthesized standalone so they don't leak and directly
# target the ar-03 (unit) and gte-01 (meta-provenance) calibration FPs. audit-19 §8.
_SYNTH_METRICS = [
    ("Data Center revenue", "Data Center"), ("total revenue", "total revenue"),
    ("operating income", "operating income"), ("Gaming revenue", "Gaming"),
    ("net income", "net income"), ("Professional Visualization revenue", "ProViz"),
    ("Compute revenue", "Compute segment"), ("the cost of revenue", "cost of revenue"),
]
_SYNTH_STRUCT = {
    "en": ["as stated in the abstract.", "as shown in the table.",
           "as reported in multiple tables.", "as noted in the abstract and tables.",
           "per the summary table.", "as listed in the results section."],
    "fr": ["comme indiqué dans le tableau.", "selon le résumé.",
           "comme présenté dans plusieurs tableaux."],
}


def mint_synthetic_positives(n_unit: int = 400, n_struct: int = 200) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    def synth_chunk(cid: str, text: str) -> dict[str, str]:
        return {"chunk_id": f"synthetic#{cid}", "doc_id": "synthetic",
                "title": "Financial Summary", "doc_title": "the 2026 report", "text": text}

    for k in range(n_unit):
        metric, _short = _SYNTH_METRICS[k % len(_SYNTH_METRICS)]
        millions = 1000 + (_h("unit", k) % 499000)  # 1,000 .. 500,000 (in millions)
        billions = millions / 1000
        split = "dev" if _h("usplit", k) % 100 < 12 else "train"
        # evidence states the comma-grouped millions form; answer the worded billions form
        evidence = (f"The following table summarizes results (in millions): "
                    f"|Item|Amount| |---|---| |{metric}|$ {millions:,}|")
        forms = [f"${billions:g} billion", f"${millions:,} million", f"${millions:,}"]
        ans_value = forms[_h("uform", k) % len(forms)]
        answer = f"The {metric} was {ans_value}."
        samples.append(make_sample(synth_chunk(f"u{k}", evidence), "en",
                                   f"What was the {metric}?", answer, None,
                                   "synth_unit_pos", split))
    for k in range(n_struct):
        metric, _short = _SYNTH_METRICS[k % len(_SYNTH_METRICS)]
        lang = "fr" if k % 5 == 0 else "en"
        val = f"{30 + (_h('s', k) % 60)}.{_h('s2', k) % 10}%"
        split = "dev" if _h("ssplit", k) % 100 < 12 else "train"
        evidence = (f"Abstract — The {metric} reached {val} this year. "
                    f"Details appear in the results table below.")
        struct = _SYNTH_STRUCT[lang][_h("struct", k) % len(_SYNTH_STRUCT[lang])]
        if lang == "fr":
            answer = f"Le {metric} était de {val}, {struct}"
        else:
            answer = f"The {metric} was {val}, {struct}"
        samples.append(make_sample(synth_chunk(f"s{k}", evidence), lang,
                                   f"What was the {metric}?", answer, None,
                                   "synth_struct_pos", split))
    return samples


# When True, the question slot is BLANK in every minted prompt — train == the
# evidence-only (noq) inference the checker must use, because the breach subject is
# DEFINITIONALLY in the question (ar-12 fired at 0.0 with the question, 0.535 without:
# the question-echo licenses the very subject the binding fabricates). audit-19 §8.
QUESTION_BLIND = False


def make_sample(
    chunk: dict[str, str],
    lang: str,
    question: str,
    answer: str,
    breach_span: tuple[int, int] | None,
    kind: str,
    split: str,
) -> dict[str, Any]:
    passage = f"{chunk['title']} — {chunk['text']}"
    prompt = LD_QA_TEMPLATE.format(
        question="" if QUESTION_BLIND else question,
        num_passages=1, context=f"passage 1: {passage}",
    )
    return {
        "prompt": prompt,
        "answer": answer,
        "labels": (
            [{"start": breach_span[0], "end": breach_span[1]}] if breach_span else []
        ),
        "split": split,
        "task_type": "qa",
        "dataset": "memex-binding-mints",
        "language": lang,
        "meta": {
            "kind": kind,
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
        },
    }


def mint_table_phase(
    chunks: list[dict[str, str]],
    per_chunk_cap: int = 6,
    per_doc_cap: int = 400,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    doc_counts: Counter[str] = Counter()
    for chunk in chunks:
        if doc_counts[chunk["doc_id"]] >= per_doc_cap:
            continue
        tables = extract_tables(chunk["doc_id"], chunk["text"])
        lang = detect_lang(chunk["text"])
        split = "dev" if _h("split", chunk["doc_id"]) % 100 < 12 else "train"
        agg = mint_aggregate_bind(chunk, tables, lang, split)
        samples.extend(agg)
        doc_counts[chunk["doc_id"]] += len(agg)
        if not tables:
            continue
        emitted = 0
        for table in tables:
            if emitted >= per_chunk_cap * 2:
                break
            header = [clean_cell(h) for h in table.header]
            rows = [[clean_cell(c) for c in r] for r in table.rows]
            if len(rows) < 2 or len(header) < 2:
                continue
            millions = _is_millions_table(table)
            labels = [r[0] for r in rows]
            if len(set(labels)) != len(labels):  # ambiguous row identity
                continue

            def cell_ok(v: str) -> bool:
                return bool(v) and v not in {"—", "-", "–", "N/A", "n/a"} and len(v) <= 40

            def label_ok(v: str) -> bool:
                return bool(v) and 2 <= len(v) <= 60

            cells = [
                (i, j)
                for i in range(len(rows))
                for j in range(1, len(header))
                if j < len(rows[i]) and cell_ok(rows[i][j]) and label_ok(rows[i][0])
                and header[j]
            ]
            if not cells:
                continue
            rot = _h("cells", chunk["chunk_id"], table.table_id)
            cells = cells[rot % len(cells):] + cells[: rot % len(cells)]
            # numeric cells first AND a deeper cap: the numeric-metric binding shape
            # (the literal ar-12 class) is scarce in the vault — mint it exhaustively.
            cells.sort(key=lambda ij: not is_numeric_value(rows[ij[0]][ij[1]]))
            chunk_cap = (
                per_chunk_cap * 2
                if any(is_numeric_value(rows[i][j]) for i, j in cells)
                else per_chunk_cap
            )

            for i, j in cells:
                if emitted >= chunk_cap or doc_counts[chunk["doc_id"]] >= per_doc_cap:
                    break
                value = rows[i][j]
                vt = "num" if is_numeric_value(value) else "txt"
                bank = CLAIM_TEMPLATES[(lang, vt)]
                t_idx = _h("tpl", chunk["chunk_id"], i, j)
                ct = bank[t_idx % len(bank)]
                qt = QUESTION_TEMPLATES[lang][t_idx % len(QUESTION_TEMPLATES[lang])]

                # A2 positive
                claim, _ = render(ct, {"label": rows[i][0], "header": header[j], "value": value})
                q, _ = render(qt, {"label": rows[i][0], "header": header[j]})
                samples.append(make_sample(chunk, lang, q, claim, None, "table_pos", split))
                emitted += 1
                doc_counts[chunk["doc_id"]] += 1

                # NEG-row: re-attribute the value to a sibling row label
                ks = [
                    k
                    for k in range(len(rows))
                    if k != i and label_ok(rows[k][0])
                    and not _TOTAL_RE.match(rows[k][0])
                    and (j >= len(rows[k]) or rows[k][j] != value)  # F2 accidental truth
                ]
                if ks:
                    k = ks[_h("negrow", chunk["chunk_id"], i, j) % len(ks)]
                    claim, spans = render(
                        ct, {"label": rows[k][0], "header": header[j], "value": value}
                    )
                    q, _ = render(qt, {"label": rows[k][0], "header": header[j]})
                    samples.append(
                        make_sample(chunk, lang, q, claim, spans["label"], "table_neg_row", split)
                    )
                    emitted += 1
                    doc_counts[chunk["doc_id"]] += 1

                    # C hard positive: the rebind target's own true value
                    if j < len(rows[k]) and cell_ok(rows[k][j]):
                        claim, _ = render(
                            ct, {"label": rows[k][0], "header": header[j], "value": rows[k][j]}
                        )
                        samples.append(
                            make_sample(chunk, lang, q, claim, None, "table_hard_pos", split)
                        )
                        emitted += 1
                        doc_counts[chunk["doc_id"]] += 1

                # NEG-col: re-attribute the value to a sibling column header
                js = [
                    j2
                    for j2 in range(1, len(header))
                    if j2 != j and header[j2]
                    and (j2 >= len(rows[i]) or rows[i][j2] != value)  # F2
                ]
                if js and emitted < chunk_cap:
                    j2 = js[_h("negcol", chunk["chunk_id"], i, j) % len(js)]
                    claim, spans = render(
                        ct, {"label": rows[i][0], "header": header[j2], "value": value}
                    )
                    q, _ = render(qt, {"label": rows[i][0], "header": header[j2]})
                    samples.append(
                        make_sample(chunk, lang, q, claim, spans["header"], "table_neg_col", split)
                    )
                    emitted += 1
                    doc_counts[chunk["doc_id"]] += 1

                # A3 unit-transform positives (the calibration FP modes as positives):
                # ALL supported surface forms, incl. the millions-table → billions
                # rewrite that is the literal ar-03 FP shape.
                for tv in unit_transforms(value, lang, millions_table=millions):
                    if emitted >= chunk_cap:
                        break
                    claim, _ = render(
                        ct, {"label": rows[i][0], "header": header[j], "value": tv}
                    )
                    q, _ = render(qt, {"label": rows[i][0], "header": header[j]})
                    samples.append(
                        make_sample(chunk, lang, q, claim, None, "table_unit_pos", split)
                    )
                    emitted += 1
                    doc_counts[chunk["doc_id"]] += 1
    return samples


def enforce_template_cap(samples: list[dict[str, Any]], frac: float = 0.15) -> list[dict[str, Any]]:
    """F7: no claim template may dominate. Templates are recoverable by masking the
    slots — approximate by the first 3 words of the answer."""
    cap = max(1, int(len(samples) * frac))
    seen: Counter[str] = Counter()
    out = []
    for s in samples:
        key = " ".join(s["answer"].split()[:3]).lower()
        if seen[key] >= cap:
            continue
        seen[key] += 1
        out.append(s)
    return out


def leak_assert(samples: list[dict[str, Any]], cal_chunk_ids: set[str],
                cal_texts: list[set[str]]) -> None:
    for s in samples:
        if s["meta"]["chunk_id"] in cal_chunk_ids:
            raise AssertionError(f"LEAK: minted from calibration chunk {s['meta']['chunk_id']}")
        toks = _tokens(s["answer"])
        for ct in cal_texts:
            if _jaccard(toks, ct) > 0.8:
                raise AssertionError(
                    f"LEAK: claim too close to a calibration text: {s['answer'][:80]!r}"
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["t", "p", "all"], default="t")
    ap.add_argument("--out", default="/tmp/binding_mints.json")  # noqa: S108 — mint artifact
    ap.add_argument("--limit-chunks", type=int, default=0)
    ap.add_argument("--max-candidates", type=int, default=1400)
    ap.add_argument("--reuse-p", default="", help="reuse phase-P samples from a prior mint file")
    ap.add_argument("--question-blind", action="store_true",
                    help="blank the question slot in every prompt (train == noq inference)")
    args = ap.parse_args()

    if args.question_blind:
        globals()["QUESTION_BLIND"] = True
        print("[mint] QUESTION_BLIND: prompts emitted with an empty question slot", flush=True)

    holdout_docs, cal_chunk_ids, cal_texts = load_calibration()
    print(f"[mint] holdout docs: {len(holdout_docs)}", flush=True)
    chunks = eligible_chunks(holdout_docs)
    if args.limit_chunks:
        chunks = chunks[: args.limit_chunks]
    print(f"[mint] eligible chunks: {len(chunks)}", flush=True)

    samples: list[dict[str, Any]] = []
    if args.phase in ("t", "all"):
        t_samples = enforce_template_cap(mint_table_phase(chunks))
        print(f"[mint] phase T: {len(t_samples)}", flush=True)
        samples.extend(t_samples)
    if args.reuse_p:
        p_samples = [s for s in json.load(open(args.reuse_p))
                     if s["meta"]["kind"].startswith("prose_")]
        if args.question_blind:  # rebuild the prompt with a blank question slot
            blank_re = re.compile(
                r"(Briefly answer the following question:\n).*?(\nBear in mind)", re.DOTALL
            )
            for s in p_samples:
                s["prompt"] = blank_re.sub(r"\1\2", s["prompt"])
        print(f"[mint] phase P reused from {args.reuse_p}: {len(p_samples)}", flush=True)
        samples.extend(p_samples)
    elif args.phase in ("p", "all"):
        import asyncio as _aio

        p_samples = _aio.run(mint_prose_phase(chunks, max_candidates=args.max_candidates))
        json.dump(p_samples, open(args.out + ".rawp", "w"), indent=1, ensure_ascii=False)
        print(f"[mint] phase P raw (pre-F4) -> {args.out}.rawp", flush=True)
        p_samples = f4_nli_discard(p_samples)
        print(f"[mint] phase P: {len(p_samples)}", flush=True)
        samples.extend(p_samples)

    synth = mint_synthetic_positives()
    if args.question_blind:
        blank_re = re.compile(
            r"(Briefly answer the following question:\n).*?(\nBear in mind)", re.DOTALL
        )
        for s in synth:
            s["prompt"] = blank_re.sub(r"\1\2", s["prompt"])
    print(f"[mint] synthetic skill positives (unit/struct): {len(synth)}", flush=True)
    samples.extend(synth)

    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    add_provenance_tails(samples, chunks_by_id)
    n_tail = sum(1 for s in samples if s["meta"]["kind"].endswith("_tail"))
    print(f"[mint] provenance tails applied: {n_tail}", flush=True)
    leak_assert(samples, cal_chunk_ids, cal_texts)

    kinds = Counter(s["meta"]["kind"] for s in samples)
    langs = Counter(s["language"] for s in samples)
    splits = Counter(s["split"] for s in samples)
    n_breach = sum(1 for s in samples if s["labels"])
    print(f"[mint] {len(samples)} samples | breach {n_breach} / ok {len(samples) - n_breach}")
    print(f"[mint] kinds: {dict(kinds)}")
    print(f"[mint] langs: {dict(langs)} | splits: {dict(splits)}")
    json.dump(samples, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"[mint] -> {args.out}")




# ---------------------------------------------------------------- phase P (prose, 4B)

VLLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
VLLM_MODEL = "cyankiwi/Qwen3.5-4B-AWQ-4bit"

_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "maxLength": 300},
                    "subject": {"type": "string", "maxLength": 80},
                    "value": {"type": "string", "maxLength": 40},
                },
                "required": ["text", "subject", "value"],
            },
        }
    },
    "required": ["claims"],
}
_PARA_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string", "maxLength": 350}},
    "required": ["text"],
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
# a claim must contain a predicate — drops title/heading-shaped extractions
_VERBY = {
    "en": re.compile(
        r"\b(is|are|was|were|has|have|had|includes?|included|uses?|used|provides?|"
        r"supports?|enforces?|requires?|reach(?:ed|es)?|recorded|reported|grew|"
        r"increased|decreased|lists?|states?|contains?|runs?|allows?|enables?)\b",
        re.IGNORECASE),
    "fr": re.compile(
        r"\b(est|sont|était|étaient|a|ont|avait|inclut|incluent|utilise(?:nt)?|"
        r"fournit|prend|atteint|enregistré|déclaré|contient|permet|exige|liste)\b",
        re.IGNORECASE),
}
_CAP_TOKEN = re.compile(r"\b[A-ZÀ-Ý][\w-]{2,}\b")


async def _llm_json(client: Any, prompt: str, schema: dict[str, Any],
                    max_tokens: int) -> dict[str, Any] | None:
    body = {
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        # the OpenAI-standard structured-output form (models/client.py — vLLM's
        # bare `guided_json` is deprecated and silently IGNORED by the daemon)
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "mint", "schema": schema, "strict": True},
        },
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = await client.post(VLLM_URL, json=body, timeout=180)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"] or ""
        return json.loads(content.strip())
    except Exception as e:
        print(f"[mint-p] llm call failed: {type(e).__name__}: {e}", flush=True)
        return None


def _coreferent(a: str, b: str) -> bool:
    """F3: alias/substring/initialism guard — never rebind onto a co-reference."""
    from memex.index.initialism import derive_initialism

    la, lb = a.lower().strip(), b.lower().strip()
    if not la or not lb or la in lb or lb in la:
        return True
    ia, ib = derive_initialism(a), derive_initialism(b)
    return bool((ia and ia.lower() == lb) or (ib and ib.lower() == la))


def _f1_conflict(chunk_text: str, s2: str, value: str) -> bool:
    """F1: some sentence already binds s2 to the value → the rebind may be TRUE."""
    s2l, vl = s2.lower(), value.lower()
    return any(
        s2l in sent.lower() and vl in sent.lower() for sent in _SENT_SPLIT.split(chunk_text)
    )


def _splice(claim: str, subject: str, s2: str) -> tuple[str, tuple[int, int]] | None:
    """EN proper-noun path: replace the single occurrence of `subject` in `claim`."""
    occ = [m.start() for m in re.finditer(re.escape(subject), claim)]
    if len(occ) != 1:
        return None
    i = occ[0]
    return claim[:i] + s2 + claim[i + len(subject):], (i, i + len(s2))


def _verbatim(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()


async def mint_prose_phase(
    chunks: list[dict[str, str]],
    per_doc_cap: int = 16,
    max_candidates: int = 1400,
    concurrency: int = 6,
) -> list[dict[str, Any]]:
    """A1 extraction → presence-preserving rebind negatives (pool = OTTER same-kind
    entities ∪ table row labels = the aggregate-bind class) → hard positives →
    style-decorrelating paraphrase. Labels 100% by construction (the 4B never labels)."""
    import httpx

    from memex.cli.bootstrap import bootstrap
    from memex.core.types import Chunk
    from memex.enrich.ner_otter import extract_chunk_entities

    bootstrap()

    # cheap candidate scoring before any model call: digits + capitalized tokens
    def cand_key(c: dict[str, str]) -> tuple[int, str]:
        text = c["text"]
        caps = len(set(_CAP_TOKEN.findall(text)))
        has_digit = any(ch.isdigit() for ch in text)
        has_table = "|---" in text
        score = caps + (8 if has_digit else 0) + (6 if has_table else 0)
        return (-score, c["chunk_id"])

    prose = [c for c in chunks if len(c["text"]) > 400]
    prose.sort(key=cand_key)
    doc_counts: Counter[str] = Counter()
    cands: list[dict[str, str]] = []
    for c in prose:
        if doc_counts[c["doc_id"]] >= 10:
            continue
        doc_counts[c["doc_id"]] += 1
        cands.append(c)
        if len(cands) >= max_candidates:
            break
    print(f"[mint-p] candidates: {len(cands)} across {len(doc_counts)} docs", flush=True)

    samples: list[dict[str, Any]] = []
    emitted_per_doc: Counter[str] = Counter()
    sem = __import__("asyncio").Semaphore(concurrency)
    asyncio = __import__("asyncio")

    async def process(chunk: dict[str, str]) -> list[dict[str, Any]]:
        if emitted_per_doc[chunk["doc_id"]] >= per_doc_cap:
            return []
        text = chunk["text"]
        lang = detect_lang(text)
        split = "dev" if _h("split", chunk["doc_id"]) % 100 < 12 else "train"

        # typed rebind pool: OTTER entities + table row labels (aggregate-bind)
        ents = await extract_chunk_entities(
            Chunk(chunk_id=chunk["chunk_id"], document_id=chunk["doc_id"],
                  document_title=chunk["title"], text=text)
        )
        by_kind: dict[str, list[str]] = {}
        for e in ents:
            name = e.name.strip()
            if 2 <= len(name) <= 60 and _verbatim(text, name):
                by_kind.setdefault(e.kind, []).append(name)
        table_labels = [
            clean_cell(r[0])
            for t in extract_tables(chunk["doc_id"], text)
            for r in t.rows
            if clean_cell(r[0]) and not _TOTAL_RE.match(clean_cell(r[0]))
        ]

        async with sem:
            async with httpx.AsyncClient() as client:
                allowed = sorted(
                    {n for names in by_kind.values() for n in names} | set(table_labels)
                )[:25]
                if not allowed:
                    return []
                ex = await _llm_json(
                    client,
                    "Extract up to 3 atomic factual claims from this passage. Each claim "
                    "must state ONE fact whose SUBJECT is EXACTLY one of these entities "
                    f"(copy it verbatim): {json.dumps(allowed, ensure_ascii=False)}.\n"
                    "Each claim also needs a VALUE (a number, date, name, or short phrase "
                    "stated verbatim in the passage) that the fact attributes to the "
                    "subject. The claim text must contain both subject and value verbatim. "
                    "Use the passage's language. Skip titles/headings — only real facts.\n\n"
                    f"PASSAGE:\n{text[:6000]}\n\n"
                    'JSON: {"claims": [{"text", "subject", "value"}]}',
                    _EXTRACT_SCHEMA, 500,
                )
                if not ex:
                    return []
                raw_claims = ex.get("claims", []) if isinstance(ex, dict) else ex
                claims = []
                for c in raw_claims if isinstance(raw_claims, list) else []:
                    if not (isinstance(c, dict) and {"subject", "value", "text"} <= c.keys()):
                        continue
                    subj, val, ctext = c["subject"].strip(), c["value"].strip(), c["text"].strip()
                    if (subj and val and ctext and subj in allowed
                            and _verbatim(text, val)
                            and subj in ctext and val in ctext
                            and _VERBY[lang].search(ctext)
                            and not _jaccard(_tokens(ctext), _tokens(subj)) > 0.9):
                        claims.append({"text": ctext, "subject": subj, "value": val})
                if not claims:
                    return []

                subjects_here = {c["subject"] for c in claims}
                out: list[dict[str, Any]] = []

                def q_for(subject: str) -> str:
                    return (f"Que dit le document à propos de {subject} ?" if lang == "fr"
                            else f"What does the document state about {subject}?")

                for ci, c in enumerate(claims):
                    if len(out) >= 6:
                        break
                    # POS
                    out.append(make_sample(chunk, lang, q_for(c["subject"]), c["text"],
                                           None, "prose_pos", split))
                    # NEG: typed pool; prefer rebind targets that have their own
                    # positive claim here (kills the inverse-presence shortcut)
                    kind = next((k for k, names in by_kind.items()
                                 if any(n == c["subject"] for n in names)), None)
                    pool = list(dict.fromkeys(
                        (by_kind.get(kind, []) if kind else [])
                        + table_labels
                    ))
                    pool = [s2 for s2 in pool
                            if not _coreferent(s2, c["subject"])
                            and not _f1_conflict(text, s2, c["value"])]
                    pool.sort(key=lambda s2: (s2 not in subjects_here, s2))
                    if not pool:
                        continue
                    s2 = pool[_h("pool", chunk["chunk_id"], ci) % min(len(pool), 3)]
                    spliced = _splice(c["text"], c["subject"], s2)
                    if spliced is None:
                        continue
                    neg_text, span = spliced
                    out.append(make_sample(chunk, lang, q_for(s2), neg_text,
                                           span, "prose_neg_rebind", split))

                # D: style decorrelation — paraphrase ~50% of BOTH classes
                for idx, s in enumerate(list(out)):
                    if _h("para", chunk["chunk_id"], idx) % 2:
                        continue
                    para = await _llm_json(
                        client,
                        "Rewrite this sentence preserving its EXACT meaning. Keep every "
                        "name, number, unit and quoted phrase VERBATIM. One sentence, "
                        "same language.\n\n"
                        f"SENTENCE: {s['answer']}\n\n"
                        'JSON: {"text": "..."}',
                        _PARA_SCHEMA, 380,
                    )
                    if not para or not para.get("text"):
                        continue
                    new = para["text"].strip()
                    if s["labels"]:  # NEG: the swapped subject must survive exactly once
                        old_span = s["labels"][0]
                        swapped = s["answer"][old_span["start"]:old_span["end"]]
                        occ = [m.start() for m in re.finditer(re.escape(swapped), new)]
                        if len(occ) != 1:
                            continue
                        s2_span = (occ[0], occ[0] + len(swapped))
                        out.append({**s, "answer": new,
                                    "labels": [{"start": s2_span[0], "end": s2_span[1]}],
                                    "meta": {**s["meta"], "kind": s["meta"]["kind"] + "_para"}})
                    else:
                        out.append({**s, "answer": new,
                                    "meta": {**s["meta"], "kind": s["meta"]["kind"] + "_para"}})
                return out

    done = 0
    for batch_start in range(0, len(cands), 40):
        batch = cands[batch_start:batch_start + 40]
        results = await asyncio.gather(*(process(c) for c in batch))
        for c, rows in zip(batch, results, strict=True):
            emitted_per_doc[c["doc_id"]] += len(rows)
            samples.extend(rows)
        done += len(batch)
        if done % 200 == 0 or done == len(cands):
            print(f"[mint-p] {done}/{len(cands)} chunks -> {len(samples)} samples", flush=True)
    return samples


_METRIC_SENT_RE = re.compile(
    r"\b([A-Za-z][a-zA-Z &/'-]{2,40}?)\s+"
    r"(?:was|is|are|reached|totaled|rose to|grew to|increased to|stood at|defaults to|"
    r"s'élevait à|était de|est de|est|a atteint)\s+"
    r"(\$?\d[\d.,]*\s?(?:%|billion|million|trillion|B|M|GB|MB|Mbit/s|Gbit/s|bits?|bytes?|seconds?|ms|days?|hours?|characters?|octets?)?)(?=[\s,.;)])"
)
# key-value lines ("Access Points: 10") — the vault's dominant prose-binding shape
_KV_LINE_RE = re.compile(r"^\s*[-*•]?\s*([A-Za-z][\w &/'()-]{2,40}?)\s*:\s+(\$?\d[\d.,]*\s?[\w%/]{0,8})\b",
                         re.MULTILINE)

_TAILS = {
    "en": [", as stated in the {doc}.", ", as noted in the {doc}.", " ({doc})."],
    "fr": [", selon {doc}.", ", comme indiqué dans {doc}."],
}


def add_provenance_tails(samples: list[dict[str, Any]], chunks_by_id: dict[str, dict[str, str]],
                         rate_pct: int = 35) -> None:
    """Style symmetry for the Memex citation-tail answer shape (the documented
    audit-18 FP mode — cand1 fired on '2026 Annual Review (Form 10-K)' tokens
    because NO training positive ever carried a doc-name tail). Applied at the
    same rate to BOTH classes; breach spans live in the claim body so appending
    a tail never moves them."""
    for s in samples:
        if _h("tail", s["meta"]["chunk_id"], s["answer"][:40]) % 100 >= rate_pct:
            continue
        chunk = chunks_by_id.get(s["meta"]["chunk_id"])
        if chunk is None:
            continue
        lang = s["language"]
        bank = _TAILS.get(lang, _TAILS["en"])
        tail = bank[_h("tailtpl", s["meta"]["chunk_id"], s["answer"][:40]) % len(bank)]
        doc = chunk["doc_title"]
        base = s["answer"][:-1] if s["answer"].endswith(".") else s["answer"]
        s["answer"] = base + tail.format(doc=doc)
        s["meta"]["kind"] += "_tail"


def mint_aggregate_bind(chunk: dict[str, str], tables: list[Any], lang: str,
                        split: str, cap: int = 4) -> list[dict[str, Any]]:
    """The deliberate §2.3 NEG-aggregate-bind class — the literal ar-12 generator:
    a value stated in PROSE (consolidated/document-level) rebound to a co-present
    table row label, plus the true consolidated positive. Deterministic, LLM-free."""
    text = chunk["text"]
    prose = text
    for t in tables:
        prose = prose.replace(text[t.char_start:t.char_end], " ")
    labels = [
        clean_cell(r[0])
        for t in tables
        for r in t.rows
        if clean_cell(r[0]) and 2 <= len(clean_cell(r[0])) <= 60
        and not _TOTAL_RE.match(clean_cell(r[0]))
    ]
    kv = [(clean_cell(k), v.strip()) for k, v in _KV_LINE_RE.findall(prose)]
    kv = [(k, v) for k, v in kv if 2 <= len(k) <= 50]
    out: list[dict[str, Any]] = []

    # KV sibling rebind: the value re-attributed to a co-present sibling key
    if len({k for k, _ in kv}) >= 2:
        for i, (k, v) in enumerate(kv):
            if len(out) >= cap:
                break
            sibs = [k2 for j, (k2, v2) in enumerate(kv)
                    if j != i and v2 != v and not _coreferent(k2, k)
                    and not _f1_conflict(text, k2, v)]
            if not sibs:
                continue
            is_fr = lang == "fr"
            pos_tpl = "Le {header} est {value}." if is_fr else "The {header} is {value}."
            claim, _ = render(pos_tpl, {"header": k, "value": v})
            q = (f"Quelle est la valeur de {k} ?" if is_fr else f"What is the {k}?")
            out.append(make_sample(chunk, lang, q, claim, None, "kv_pos", split))
            k2 = sibs[_h("kv", chunk["chunk_id"], k) % len(sibs)]
            neg, spans = render(pos_tpl, {"header": k2, "value": v})
            qn = (f"Quelle est la valeur de {k2} ?" if is_fr else f"What is the {k2}?")
            out.append(make_sample(chunk, lang, qn, neg, spans["header"], "kv_neg_bind", split))

    if not labels and not kv:
        return out
    seen_metrics: set[str] = set()
    for m in _METRIC_SENT_RE.finditer(prose):
        if len(out) >= cap * 2:
            break
        metric, value = m.group(1).strip(), m.group(2).strip()
        ml = metric.lower()
        if ml in seen_metrics or ml.split()[-1] in _PROSE_METRIC_STOP:
            continue
        seen_metrics.add(ml)
        # the true consolidated positive (verbatim value, metric phrase from prose)
        pos_tpl = ("Le {header} était de {value}." if lang == "fr"
                   else "The {header} was {value}.")
        claim, _ = render(pos_tpl, {"header": metric, "value": value})
        q = (f"Quel était le {metric} ?" if lang == "fr" else f"What was the {metric}?")
        out.append(make_sample(chunk, lang, q, claim, None, "agg_pos", split))
        # the rebind: the prose value attributed to a co-present row label
        pool = [
            lab for lab in labels + [k for k, _ in kv]
            if not _coreferent(lab, metric) and not _f1_conflict(text, lab, value)
        ]
        if not pool:
            continue
        lab = pool[_h("agg", chunk["chunk_id"], metric) % len(pool)]
        neg_bank = (
            ["Le {header} de {label} était de {value}.",
             "Pour {label}, le {header} était de {value}."]
            if lang == "fr"
            else ["The {header} for {label} was {value}.",
                  "{label}'s {header} was {value}.",
                  "The {header} of {label} was {value}."]
        )
        nt = neg_bank[_h("aggneg", chunk["chunk_id"], metric) % len(neg_bank)]
        neg, spans = render(nt, {"header": metric, "label": lab, "value": value})
        qn = (f"Quel était le {metric} de {lab} ?" if lang == "fr"
              else f"What was the {metric} for {lab}?")
        out.append(make_sample(chunk, lang, qn, neg, spans["label"], "agg_neg_bind", split))
    return out


_PROSE_METRIC_STOP = frozenset({"it", "this", "that", "which", "there", "he", "she"})


def f4_nli_discard(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """F4: drop negatives the NLI model says are ENTAILED by their passage
    (accidental truth). Direction-correct per audit-18; used only as a
    high-precision discard, never as a labeler."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    negs = [i for i, s in enumerate(samples) if s["labels"]]
    if not negs:
        return samples
    ckpt = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    tok = AutoTokenizer.from_pretrained(ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        free, _total = torch.cuda.mem_get_info()
        if free < 4.0 * 1024**3:  # the co-resident daemon leaves too little headroom
            device = "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device)
    model.eval()
    print(f"[mint-p] F4 NLI on {device}", flush=True)
    ent_idx = int(model.config.label2id.get("entailment", 0))
    drop: set[int] = set()
    with torch.no_grad():
        for b in range(0, len(negs), 8):
            idxs = negs[b:b + 8]
            # premise = the passage text inside the prompt (between 'passage 1: ' and the closing instruction)
            pairs = []
            for i in idxs:
                prompt = samples[i]["prompt"]
                start = prompt.find("passage 1: ") + len("passage 1: ")
                end = prompt.rfind("\nIn case the passages")
                pairs.append((prompt[start:end][:4000], samples[i]["answer"]))
            enc = tok([p for p, _ in pairs], [h for _, h in pairs], truncation=True,
                      max_length=1024, padding=True, return_tensors="pt").to(device)
            try:
                probs = torch.softmax(model(**enc).logits, dim=-1)[:, ent_idx]
            except torch.OutOfMemoryError:
                # co-residence squeeze mid-pass: finish on CPU (forward-only, exact)
                device = "cpu"
                model = model.to(device)
                enc = {k: v.to(device) for k, v in enc.items()}
                probs = torch.softmax(model(**enc).logits, dim=-1)[:, ent_idx]
                print("[mint-p] F4 OOM -> CPU fallback", flush=True)
            for i, p in zip(idxs, probs.tolist(), strict=True):
                if p > 0.5:
                    drop.add(i)
    print(f"[mint-p] F4 NLI discard: {len(drop)}/{len(negs)} negatives dropped", flush=True)
    return [s for i, s in enumerate(samples) if i not in drop]


if __name__ == "__main__":
    main()
