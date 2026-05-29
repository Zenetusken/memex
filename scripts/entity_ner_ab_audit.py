"""Entity-extraction A/B audit — OTTER NER vs the live Qwen3-8B graph entities.

The gating question behind the (still-scoped, not-built) BERT-NER enrich swap
([[bert-ner-enrich-scope-2026-05-28]], deep-research OQ#3): does a zero-shot span
NER (OTTER, `whoisjones/otter-bi-mmbert`) produce SHARPER, more consistently-typed
entities than the current Qwen3-8B LLM extractor — measured on the real
graph-DISCOVERY surface, not a generic NER benchmark? The graph feeds discovery
(`related_documents`, entity-centric retrieval), so "better" = fewer generic /
noise / mis-typed entities and a higher IDF-specificity signal, especially in the
`method`/`tool`/`concept` buckets that have no gold schema.

This harness is PURE-READ on the graph (no mutations). It:
  * reads set A = the live Qwen3-8B entities already in the graph (per enriched doc);
  * runs set B = OTTER fresh over the SAME docs' chunks (CPU-side, offline);
  * computes the SAME IDF-specificity scoring the live `related_documents` uses
    (it reuses `graph_store._rank_related_documents` + the kind-weight / df-cap
    constants verbatim, so the two sets are scored apples-to-apples);
  * reports granularity / type / specificity / noise deltas + a per-seed
    `related_documents` side-by-side, and an ADVISORY verdict.

NOTE on scope: N (doc count) and every entity df are computed over the AUDITED set P. So
under `--match`/`--subset` the absolute specificity/discovery scores are A/B-RELATIVE, not
identical to production `memex related` numbers (which use whole-corpus df and surface
whole-corpus neighbours). The A-vs-B delta stays fair (both sides use the same P); only the
absolute scale shifts. A no-filter run over the whole vault matches production N.

The DECISION is human review of the per-seed diffs (the scope memo proved no single
structural metric cleanly classifies "noise" in a coherent corpus) — the verdict
bar is a coarse signal, not the call.

OTTER is NOT downloaded by default and its custom inference API (the `AllLabelsCollator`
input shape + the `model.predict` output shape) is UNDOCUMENTED in the model READMEs —
see the `OtterExtractor` banner. Everything OTHER than that one adapter is complete and
can be validated TODAY with `--extractor self` (set B = the graph again → all deltas ~0),
which exercises the entire read/metric/scoring/report pipeline without the model.

Usage:
  # validate the whole engine against the live graph, no model needed:
  MEMEX_MODELS__RERANKER_DEVICE=cpu uv run python scripts/entity_ner_ab_audit.py \
      --extractor self --out /tmp/ab_selfcheck.json

  # the real A/B once OTTER is fetched (one-time ~1.9 GB download, or pre-cache it):
  uv run python scripts/entity_ner_ab_audit.py \
      --match cr350 --match srwe --seeds 6 --out /tmp/ab_otter.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memex.core.config import MemexSettings, get_settings, set_settings
from memex.core.manifest import read_manifest
from memex.index.chunker import chunk_document

# The audit deliberately reuses the LIVE discovery scoring internals so set A and set B
# are scored identically — duplicating the IDF/kind-weight logic here would let it drift
# from production. `scripts/` is dev tooling (cf. `citation_graph_audit.py` reaching into
# `graph._conn`), so the cross-module-private rule for `src/memex/` doesn't bind here.
from memex.index.graph_store import (
    _DEFAULT_KIND_WEIGHT,
    _ENTITY_KIND_WEIGHT,
    _RELATED_GENERIC_ENTITY_DF_FRACTION,
    GraphStore,
    _discovery_noise_filters,
    _rank_related_documents,
    entity_id,
)
from memex.vault.store import list_documents, read_document

# --- OTTER label space (an A/B knob — OQ#3) ----------------------------------------------
# OTTER is zero-shot: we pass these surface labels at inference, then map each returned
# label back to Memex's FULL 7-kind taxonomy (incl. `other`) so the kind-weight applies
# identically to both sets. The label SET and this mapping are exactly the
# `method`/`tool`/`concept` lever the scope memo flags as the mandatory validation focus —
# tune here, not in the scorer. `miscellaneous`→`other` is included so OTTER can express the
# catch-all bucket the live graph uses (else B would be structurally barred from a kind A
# has — an unfair taxonomy-coverage gap), and `entities_for_doc` defaults any unrecognised
# returned label to `other` rather than DROPPING the span (which would skew B's coverage).
_OTTER_LABEL_TO_KIND: dict[str, str] = {
    "person": "person",
    "organization": "org",
    "company": "org",
    "location": "place",
    "place": "place",
    "concept": "concept",
    "method": "method",
    "technique": "method",
    "algorithm": "method",
    "tool": "tool",
    "software": "tool",
    "protocol": "tool",
    "standard": "tool",
    "miscellaneous": "other",
}
_OTTER_LABELS: list[str] = list(dict.fromkeys(_OTTER_LABEL_TO_KIND))

# DOMAIN-AWARE preset (the "fully-leverage" lever): GLiNER-family models are sensitive to
# label PHRASING + specificity, so descriptive, networking/security-aware labels can extract
# sharper spans than the generic single words above. Mapped to the same 7-kind space so the
# scorer is unchanged. A/B these vs "generic" with `--labels` to find OTTER's quality ceiling.
_DOMAIN_LABEL_TO_KIND: dict[str, str] = {
    "person": "person",
    "company or organization": "org",
    "location or place": "place",
    "networking protocol": "tool",
    "network device or hardware": "tool",
    "software or application": "tool",
    "network service": "tool",
    "technical standard or specification": "tool",
    "security attack or threat technique": "method",
    "security control or defense mechanism": "method",
    "cryptographic algorithm or method": "method",
    "vulnerability or weakness": "concept",
    "technical concept or term": "concept",
    "miscellaneous entity": "other",
}

_LABEL_PRESETS: dict[str, dict[str, str]] = {
    "generic": _OTTER_LABEL_TO_KIND,
    "domain": _DOMAIN_LABEL_TO_KIND,
    # best-of-both: all domain phrases + all generic single words (generic wins the lone
    # "person" collision). More type embeddings → more matching options, slower.
    "union": {**_DOMAIN_LABEL_TO_KIND, **_OTTER_LABEL_TO_KIND},
}

_OTTER_DEFAULT_MODEL = "whoisjones/otter-bi-mmbert"

# The live Qwen3 set (A) only ever saw the first ~6000 chars of each chunk: the enrich
# prompt is `{{ passage | truncate(6000) }}` (src/memex/prompts/extract_entities/v2.md).
# Mirror that cap before feeding OTTER so B reads the SAME window A did — else a chunk
# longer than this (chart-extracted blocks are exempt from the chunker's force-split cap)
# would give B a coverage edge A never had. Keep this in sync with that prompt.
_ENRICH_PASSAGE_CHARS = 6000

# Documented offenders the scope memo names (bare ports/numbers, EN+FR connectors, the
# `entity_stopwords` residue). A CONSERVATIVE predicate by design — the memo proved you
# can't separate "administrative connector" from "central concept" structurally, so this
# counts only the unambiguous junk, not a general fragment classifier.
_CONNECTOR_NOISE = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "and",
        "or",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "at",
        "as",
        "is",
        "be",
        "this",
        "that",
        "le",
        "la",
        "les",
        "de",
        "des",
        "du",
        "un",
        "une",
        "et",
        "ou",
        "à",
        "au",
        "aux",
        "en",
        "dans",
        "pour",
        "sur",
        "avec",
    }
)

# Scalar metrics for which a B-minus-A delta is meaningful (kind_hist is excluded).
_NUMERIC_KEYS = [
    "unique_entities",
    "total_doc_entity_pairs",
    "mean_entities_per_doc",
    "mean_name_words",
    "mean_name_chars",
    "single_token_rate",
    "noise_rate",
    "generic_rate",
    "mean_specificity",
    "topical_rate",
    "proper_noun_rate",
]


@dataclass
class EntitySet:
    """One extractor's output over the audited doc set: per-doc, document-level-deduped
    `(name, kind)` pairs (mirroring the enrich `dedupe` key `(lower(name), kind)`)."""

    label: str
    by_doc: dict[str, list[tuple[str, str]]]
    titles: dict[str, str] = field(default_factory=dict)


def _rows(conn: Any, cypher: str, params: dict[str, Any] | None = None) -> list[tuple[Any, ...]]:
    res = conn.execute(cypher, params or {})
    out: list[tuple[Any, ...]] = []
    while res.has_next():
        out.append(tuple(res.get_next()))
    return out


def _dedupe_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Document-level dedupe by `(lower(name), kind)`, first-seen case wins — the exact
    key `enrich.entities.dedupe` uses, so both sets normalize identically."""
    seen: dict[tuple[str, str], tuple[str, str]] = {}
    for name, kind in pairs:
        key = (name.strip().lower(), kind)
        if key not in seen:
            seen[key] = (name.strip(), kind)
    return list(seen.values())


def _looks_like_noise(name: str, stopwords: frozenset[str]) -> bool:
    n = name.strip().lower()
    if not n or len(n) == 1:
        return True
    if n in stopwords or n in _CONNECTOR_NOISE:
        return True
    if n.replace(".", "").replace(":", "").replace("/", "").isdigit():  # bare number / port
        return True
    if not any(c.isalnum() for c in n):  # pure punctuation
        return True
    return False


def _docs_by_entity(eset: EntitySet) -> tuple[dict[str, set[str]], dict[str, tuple[str, str]]]:
    docs_by_eid: dict[str, set[str]] = {}
    meta: dict[str, tuple[str, str]] = {}
    for doc_id, pairs in eset.by_doc.items():
        for name, kind in pairs:
            eid = entity_id(name, kind)
            docs_by_eid.setdefault(eid, set()).add(doc_id)
            meta.setdefault(eid, (name, kind))
    return docs_by_eid, meta


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def _specificity(kind: str, df: int, n_docs: int) -> float:
    """The live `_rank_related_documents` per-entity contribution: `ln(N/df) × kind_weight`,
    0 for a generic (df over the cap) or zero-IDF entity."""
    if df <= 0 or n_docs <= 1 or df > _RELATED_GENERIC_ENTITY_DF_FRACTION * n_docs:
        return 0.0
    contribution = math.log(n_docs / df) * _ENTITY_KIND_WEIGHT.get(kind, _DEFAULT_KIND_WEIGHT)
    return contribution if contribution > 0 else 0.0


def _metrics(eset: EntitySet, *, stopwords: frozenset[str], n_docs: int) -> dict[str, Any]:
    docs_by_eid, meta = _docs_by_entity(eset)
    unique = len(meta)
    # Document-level-deduped (name, kind) pairs summed across docs — NOT raw mention
    # frequency (the graph stores one MENTIONS edge per (doc, entity), no count), so this
    # differs from `unique_entities` only when an entity spans multiple docs.
    total_pairs = sum(len(p) for p in eset.by_doc.values())
    df_cap = _RELATED_GENERIC_ENTITY_DF_FRACTION * n_docs

    name_words: list[float] = []
    name_chars: list[float] = []
    specificities: list[float] = []
    single_token = noise = generic = 0
    kind_hist: dict[str, int] = {}
    for eid, (name, kind) in meta.items():
        df = len(docs_by_eid[eid])
        kind_hist[kind] = kind_hist.get(kind, 0) + 1
        words = name.split()
        name_words.append(len(words))
        name_chars.append(len(name))
        if len(words) <= 1:
            single_token += 1
        if _looks_like_noise(name, stopwords):
            noise += 1
        if df > df_cap:
            generic += 1
        # A TRUE per-entity mean over ALL unique entities (generics contribute 0 via
        # _specificity) — so a set is scored over its whole kind universe, not a
        # survivor-only subpopulation that could reward a noisier extractor.
        specificities.append(_specificity(kind, df, n_docs))

    def _rate(count: int) -> float:
        return round(count / unique, 4) if unique else 0.0

    topical = sum(kind_hist.get(k, 0) for k in ("concept", "method", "tool"))
    proper = sum(kind_hist.get(k, 0) for k in ("person", "place", "org"))
    return {
        "unique_entities": unique,
        "total_doc_entity_pairs": total_pairs,
        "mean_entities_per_doc": round(total_pairs / n_docs, 2) if n_docs else 0.0,
        "mean_name_words": _mean(name_words),
        "mean_name_chars": _mean(name_chars),
        "single_token_rate": _rate(single_token),
        "noise_rate": _rate(noise),
        "generic_rate": _rate(generic),
        "mean_specificity": _mean(specificities),
        "topical_rate": _rate(topical),
        "proper_noun_rate": _rate(proper),
        "kind_hist": dict(sorted(kind_hist.items())),
    }


def _hard_kind_metrics(
    eset: EntitySet, *, stopwords: frozenset[str], n_docs: int
) -> dict[str, Any]:
    """The load-bearing OQ#3 view: noise / generic / specificity restricted to the
    `method`/`tool`/`concept` entities (the buckets with no gold NER schema)."""
    docs_by_eid, meta = _docs_by_entity(eset)
    hard = {eid: nk for eid, nk in meta.items() if nk[1] in ("method", "tool", "concept")}
    unique = len(hard)
    df_cap = _RELATED_GENERIC_ENTITY_DF_FRACTION * n_docs
    noise = generic = 0
    specificities: list[float] = []
    for eid, (name, kind) in hard.items():
        df = len(docs_by_eid[eid])
        if _looks_like_noise(name, stopwords):
            noise += 1
        if df > df_cap:
            generic += 1
        specificities.append(_specificity(kind, df, n_docs))  # 0 for generic; mean over all
    return {
        "unique_entities": unique,
        "noise_rate": round(noise / unique, 4) if unique else 0.0,
        "generic_rate": round(generic / unique, 4) if unique else 0.0,
        "mean_specificity": _mean(specificities),
    }


def _related(
    eset: EntitySet, seeds: list[str], *, stopwords: frozenset[str], n_docs: int
) -> dict[str, Any]:
    """Per-seed `related_documents` from this set, via the verbatim live scorer. Under
    `--match`/`--subset` the df + neighbour set are SUBSET-LOCAL (see the module NOTE), so
    these scores are an A/B-internal artifact, not the production discovery surface."""
    docs_by_eid, _meta = _docs_by_entity(eset)
    per_seed: list[dict[str, Any]] = []
    top_scores: list[float] = []
    counts: list[float] = []
    for seed in seeds:
        rows: list[tuple[str, str, str, str, int]] = []
        for name, kind in eset.by_doc.get(seed, []):
            holders = docs_by_eid.get(entity_id(name, kind), set())
            df = len(holders)
            for other in holders:
                if other != seed:
                    rows.append((other, eset.titles.get(other, other), name, kind, df))
        ranked = _rank_related_documents(rows, n_docs, limit=5, max_entities=6, stopwords=stopwords)
        per_seed.append(
            {
                "seed": seed,
                "related": [
                    {"doc_id": r.doc_id, "score": r.score, "shared_entities": r.shared_entities}
                    for r in ranked
                ],
            }
        )
        top_scores.append(ranked[0].score if ranked else 0.0)
        counts.append(len(ranked))
    return {
        "mean_top_score": _mean(top_scores),
        "mean_related_count": round(sum(counts) / len(counts), 2) if counts else 0.0,
        "per_seed": per_seed,
    }


def _verdict(a: dict[str, Any], b: dict[str, Any]) -> tuple[bool, str]:
    """ADVISORY only. OTTER shows a build-signal if it cuts noise + generic without
    collapsing coverage, while holding/raising IDF-specificity. The real decision is the
    per-seed `related_documents` diff — no structural metric cleanly classifies noise."""
    cov = b["unique_entities"] / a["unique_entities"] if a["unique_entities"] else 0.0
    noise_cut = a["noise_rate"] - b["noise_rate"]
    generic_cut = a["generic_rate"] - b["generic_rate"]
    spec_gain = b["mean_specificity"] - a["mean_specificity"]
    ok = cov >= 0.6 and noise_cut >= 0.02 and generic_cut >= 0.0 and spec_gain >= 0.0
    reason = (
        f"coverage {cov:.2f}>=0.60, noise_cut {noise_cut:+.3f}>=0.02, "
        f"generic_cut {generic_cut:+.3f}>=0, spec_gain {spec_gain:+.3f}>=0"
    )
    return ok, reason


def _to_device(obj: Any, device: str) -> Any:
    """Recursively move tensors in a (possibly nested-dict/list) collated batch to device.
    No-op on CPU; for CUDA it walks `token_encoder_inputs`/`type_encoder_inputs`/`labels`."""
    if device == "cpu":
        return obj
    if hasattr(obj, "to"):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    return obj


# =========================================================================================
# OTTER adapter — THE one model-specific seam.
#
# VERIFIED against the downloaded model code (collate_fn.py / modeling_biencoder.py /
# metrics.py, snapshot 6f4ba31, 2026-05-29) — NOT guessed:
#   * AllLabelsCollator(tok_token, tok_type, label2id); default format='text',
#     max_seq_length=512 (the model truncates each text to 512 subword tokens).
#   * Each example = {"text": <str>, "char_spans": [{start,end,label}]}. char_spans are GOLD
#     (they only fill the loss tensors), so `[]` is correct for inference — the candidate
#     span space (valid_span_mask + span_subword_indices) is gold-independent.
#   * model.predict(batch, threshold) -> list[B] of list[{"start","end","label","confidence"}]
#     where start/end are SUBWORD-TOKEN indices (metrics.compute_span_predictions, greedy
#     non-overlapping, highest-confidence-first) and `label` is already one of our surface
#     labels (the batch's id2label inverts our label2id). The collator POPS offset_mapping,
#     so the surface is recovered by DECODING input_ids[start:end+1] with the token tokenizer.
# Loading executes the model's trust_remote_code (audited clean: torch/transformers/numpy +
# os.PathLike typing + Path file reads only; no net/exec/subprocess). First load pulls the
# mmBERT + bert-base-multilingual-uncased TOKENIZERS from the hub (small).
# =========================================================================================


class OtterAdapterError(RuntimeError):
    """OTTER's custom API did not behave as verified — re-check collate_fn/modeling code."""


class OtterExtractor:
    name = "otter"

    def __init__(
        self,
        model_id: str,
        *,
        device: str,
        threshold: float,
        max_seq_length: int = 512,
        label_to_kind: dict[str, str] | None = None,
    ) -> None:
        self._threshold = threshold
        self._device = device
        self._max_seq_length = max_seq_length
        self._label_to_kind = label_to_kind if label_to_kind is not None else _OTTER_LABEL_TO_KIND
        try:
            import torch
            from transformers import (  # type: ignore[reportMissingTypeStubs]
                AutoConfig,
                AutoModelForTokenClassification,
                AutoTokenizer,
            )
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
        except ImportError as e:  # torch/transformers are core deps; this should not fire
            raise OtterAdapterError(f"torch/transformers unavailable: {e}") from e

        self._torch = torch
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModelForTokenClassification.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float32
        )
        self._model.to(device)
        self._model.eval()
        self._tok_token = AutoTokenizer.from_pretrained(config.token_encoder)
        tok_type = AutoTokenizer.from_pretrained(config.type_encoder)
        # label2id over the SELECTED preset's surface labels; the type encoder embeds the
        # label STRINGS, and the batch's id2label maps a prediction back to our label.
        labels = list(dict.fromkeys(self._label_to_kind))
        label2id = {label: i for i, label in enumerate(labels)}
        try:
            collator_cls = get_class_from_dynamic_module(
                "collate_fn.AllLabelsCollator", model_id, trust_remote_code=True
            )
        except Exception as e:  # surface the dynamic-module load failure verbatim
            raise OtterAdapterError(f"could not load AllLabelsCollator from {model_id}: {e}") from e
        # max_seq_length caps the token encoder's window (mmBERT supports long context);
        # 512 is the collator default. The type encoder only sees the short label strings.
        self._collator = collator_cls(
            self._tok_token, tok_type, label2id, max_seq_length=max_seq_length
        )

    def predict_spans(self, text: str) -> list[tuple[str, str, float]]:
        if not text.strip():
            return []
        # char_spans=[] → inference (no gold); the candidate span space is gold-independent.
        batch = self._collator([{"text": text, "char_spans": []}])
        batch = _to_device(batch, self._device)
        with self._torch.no_grad():
            preds = self._model.predict(batch, threshold=self._threshold)
        if not (isinstance(preds, list) and preds and isinstance(preds[0], list)):
            raise OtterAdapterError(f"unexpected predict() output: {type(preds)!r}")
        input_ids = batch["token_encoder_inputs"]["input_ids"][0]
        text_norm = " ".join(text.lower().split())
        out: list[tuple[str, str, float]] = []
        for sp in preds[0]:
            # start/end are SUBWORD-TOKEN indices → decode the surface from the input_ids.
            start, end = int(sp["start"]), int(sp["end"])
            surface = self._tok_token.decode(
                input_ids[start : end + 1], skip_special_tokens=True
            ).strip()
            # Drop cross-token-boundary decode GARBLE: a real span's decode is a
            # whitespace-normalised substring of the source; artefacts like "NEMOCLAW" on
            # dense/tabular text are not (found in the 10-K place spot-check, 2026-05-29).
            if surface and " ".join(surface.lower().split()) in text_norm:
                out.append((surface, str(sp["label"]), float(sp["confidence"])))
        return out

    def entities_for_doc(self, doc: Any) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for chunk in chunk_document(doc):
            # Mirror the live enrich truncation (see _ENRICH_PASSAGE_CHARS) for parity.
            for surface, label, _score in self.predict_spans(chunk.text[:_ENRICH_PASSAGE_CHARS]):
                # Default an unrecognised label to `other` (not drop) so B's coverage isn't
                # skewed against A's `other` bucket.
                kind = self._label_to_kind.get(label.strip().lower(), "other")
                if surface.strip():
                    pairs.append((surface, kind))
        return _dedupe_pairs(pairs)


def _collect_qwen(
    conn: Any, pset: set[str]
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, str]]:
    rows = _rows(
        conn,
        "MATCH (d:Document)-[:MENTIONS]->(e:Entity) RETURN d.doc_id, d.title, e.name, e.kind;",
    )
    by_doc: dict[str, list[tuple[str, str]]] = {}
    titles: dict[str, str] = {}
    for doc_id, title, name, kind in rows:
        if doc_id not in pset:
            continue
        titles[doc_id] = title or doc_id
        by_doc.setdefault(doc_id, []).append((str(name), str(kind)))
    for doc_id in by_doc:
        by_doc[doc_id] = _dedupe_pairs(by_doc[doc_id])
    return by_doc, titles


async def _enriched_docs(vault: Path, matches: list[str]) -> list[str]:
    out: list[str] = []
    async for ref in list_documents(vault):
        manifest = await read_manifest(vault, ref.doc_id)
        if manifest is None or manifest.enrich is None:
            continue
        if matches and not any(m.lower() in ref.doc_id.lower() for m in matches):
            continue
        out.append(ref.doc_id)
    return sorted(out)


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    set_settings(MemexSettings())  # type: ignore[call-arg]
    vault = get_settings().vault_path
    min_shared, stopwords = _discovery_noise_filters()

    pids = await _enriched_docs(vault, args.match or [])
    if args.subset:
        pids = pids[: args.subset]
    if len(pids) < 2:
        raise SystemExit(
            f"need >=2 fully-enriched docs to score IDF; found {len(pids)} "
            f"(match={args.match}, subset={args.subset})"
        )
    pset = set(pids)

    graph = await GraphStore.open(vault)
    try:
        qwen_by_doc, titles = await asyncio.to_thread(_collect_qwen, graph._conn, pset)
    finally:
        await graph.close()
    for pid in pids:
        titles.setdefault(pid, pid)
    a_set = EntitySet("qwen3-graph", {p: qwen_by_doc.get(p, []) for p in pids}, titles)

    doc_errors: list[dict[str, str]] = []
    if args.extractor == "self":
        # Independent copy (fresh value lists + titles) so B never aliases A's mutable state.
        b_set = EntitySet(
            "qwen3-graph (self-check)",
            {p: list(v) for p, v in a_set.by_doc.items()},
            dict(titles),
        )
        otter_model: str | None = None
    else:
        extractor = OtterExtractor(
            args.otter_model,
            device=args.device,
            threshold=args.threshold,
            max_seq_length=args.max_seq_length,
            label_to_kind=_LABEL_PRESETS[args.labels],
        )
        otter_model = args.otter_model
        b_by_doc: dict[str, list[tuple[str, str]]] = {}
        for pid in pids:
            try:
                doc = await read_document(vault, pid)
                b_by_doc[pid] = await asyncio.to_thread(extractor.entities_for_doc, doc)
            except OtterAdapterError:
                raise  # a universal API-shape mismatch aborts the whole run (fail-loud)
            except Exception as e:  # one-doc data/runtime failure: record + continue, not fatal
                b_by_doc[pid] = []
                doc_errors.append({"doc_id": pid, "error": f"{type(e).__name__}: {e}"[:200]})
        b_set = EntitySet("otter", b_by_doc, titles)

    n_docs = len(pids)
    a_m = _metrics(a_set, stopwords=stopwords, n_docs=n_docs)
    b_m = _metrics(b_set, stopwords=stopwords, n_docs=n_docs)
    # Seeds = the most-connected docs in the live set (most likely to have related docs).
    seeds = sorted(pids, key=lambda p: (-len(a_set.by_doc.get(p, [])), p))[: args.seeds]

    build, reason = _verdict(a_m, b_m)
    return {
        "extractor": args.extractor,
        "otter_model": otter_model,
        "otter_config": {
            "threshold": args.threshold,
            "max_seq_length": args.max_seq_length,
            "labels": args.labels,
        },
        "docs_audited": n_docs,
        "doc_errors": doc_errors,
        "doc_ids": pids,
        "noise_filters": {"min_shared_docs": min_shared, "stopwords": sorted(stopwords)},
        "qwen3": a_m,
        "challenger": b_m,
        "deltas": {k: round(b_m[k] - a_m[k], 4) for k in _NUMERIC_KEYS},
        "hard_kinds": {
            "qwen3": _hard_kind_metrics(a_set, stopwords=stopwords, n_docs=n_docs),
            "challenger": _hard_kind_metrics(b_set, stopwords=stopwords, n_docs=n_docs),
        },
        "discovery": {
            "seeds": seeds,
            "qwen3": _related(a_set, seeds, stopwords=stopwords, n_docs=n_docs),
            "challenger": _related(b_set, seeds, stopwords=stopwords, n_docs=n_docs),
        },
        "verdict_advisory": {"build_signal": build, "reason": reason},
    }


def _print_report(stats: dict[str, Any]) -> None:
    a, b, d = stats["qwen3"], stats["challenger"], stats["deltas"]
    hk = stats["hard_kinds"]
    disc = stats["discovery"]
    print(
        f"\n=== Entity NER A/B ({stats['docs_audited']} docs, "
        f"qwen3-graph vs {stats['extractor']}) ===",
        file=sys.stderr,
    )
    print(f"  {'metric':<24}{'qwen3':>12}{'challenger':>14}{'Δ (B-A)':>12}", file=sys.stderr)
    for k in _NUMERIC_KEYS:
        print(f"  {k:<24}{a[k]:>12}{b[k]:>14}{d[k]:>+12}", file=sys.stderr)
    print(
        f"  hard-kinds(method/tool/concept)  qwen3: noise={hk['qwen3']['noise_rate']} "
        f"generic={hk['qwen3']['generic_rate']} spec={hk['qwen3']['mean_specificity']} | "
        f"challenger: noise={hk['challenger']['noise_rate']} "
        f"generic={hk['challenger']['generic_rate']} spec={hk['challenger']['mean_specificity']}",
        file=sys.stderr,
    )
    print(
        f"  related_documents mean_top_score  qwen3={disc['qwen3']['mean_top_score']} "
        f"challenger={disc['challenger']['mean_top_score']}  "
        f"(mean_count {disc['qwen3']['mean_related_count']} vs "
        f"{disc['challenger']['mean_related_count']})",
        file=sys.stderr,
    )
    v = stats["verdict_advisory"]
    print(
        f"  VERDICT (ADVISORY — review per-seed diffs): "
        f"{'BUILD-signal' if v['build_signal'] else 'no clear gain'} — {v['reason']}",
        file=sys.stderr,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="OTTER-vs-Qwen3 entity A/B audit (read-only).")
    ap.add_argument(
        "--extractor",
        choices=["otter", "self"],
        default="otter",
        help="'otter' = run OTTER; 'self' = set B is the graph again (validates the engine)",
    )
    ap.add_argument("--otter-model", default=_OTTER_DEFAULT_MODEL)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--threshold", type=float, default=0.1, help="OTTER predict threshold")
    ap.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="OTTER token-encoder window (subword tokens); mmBERT supports long context",
    )
    ap.add_argument(
        "--labels",
        choices=sorted(_LABEL_PRESETS),
        default="generic",
        help="OTTER zero-shot label preset: 'generic' (single words) or 'domain' (descriptive)",
    )
    ap.add_argument("--subset", type=int, default=None, help="cap to the first N enriched docs")
    ap.add_argument(
        "--match",
        action="append",
        default=None,
        help="only docs whose id contains SUBSTR (repeatable)",
    )
    ap.add_argument("--seeds", type=int, default=5, help="docs to compare related_documents on")
    ap.add_argument("--out", default=None, help="write the full JSON (incl. per-seed) to FILE")
    args = ap.parse_args()

    stats = asyncio.run(_run(args))
    if args.out:
        Path(args.out).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    # Machine-readable digest to stdout (drop the heavy per-seed detail); full report to stderr.
    digest = {k: v for k, v in stats.items() if k not in ("doc_ids",)}
    digest["discovery"] = {
        "seeds": stats["discovery"]["seeds"],
        "qwen3_mean_top_score": stats["discovery"]["qwen3"]["mean_top_score"],
        "challenger_mean_top_score": stats["discovery"]["challenger"]["mean_top_score"],
    }
    print(json.dumps(digest, indent=2))
    _print_report(stats)


if __name__ == "__main__":
    main()
