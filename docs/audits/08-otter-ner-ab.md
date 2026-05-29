# OTTER vs Qwen3 entity-NER: empirical A/B (2026-05-29)

Settles the enrich entity-extraction backend against the real 47-doc
vault, not a hypothesis. Subject: every document's enrich-stage entity
extraction. Compared: the LLM extractor (Qwen3-8B, the prior default)
vs the OTTER span NER (`whoisjones/otter-bi-mmbert`). Metric: the
`related_documents` discovery yield (the graph's on-mission consumer,
ADR-0011) + entity typing quality. Harness: `scripts/entity_ner_ab_audit.py`
(read-only on the graph; reuses the verbatim live specificity scorer
`_rank_related_documents` + `_ENTITY_KIND_WEIGHT` + the generic-df filter,
so set A and set B are scored apples-to-apples). Rig: RTX 4070, 12 GB;
OTTER on CPU.

## Setup

| | Qwen3-8B (LLM) | OTTER (BERT-NER) |
|---|---|---|
| Backend | `enrich_ner_backend=llm` | `enrich_ner_backend=otter` |
| Threshold | n/a | **0.05** (master knob; the card's 0.1 strangles recall) |
| Labels | n/a | **union** (generic ∪ domain) |
| Device | GPU (vLLM) | CPU (lazy-loaded once) |
| Full-vault enrich | — | ~17 min CPU, 0 doc-errors |

Set A = the live Qwen3-8B entities per enriched doc (`MATCH (d)-[:MENTIONS]->(e)`);
set B = OTTER run fresh over the **same** docs' chunks. Citations stayed on the
LLM in both arms.

## Results

| | Qwen3-8B | OTTER (0.05 + union) | Δ |
|---|---|---|---|
| `related_documents` mean top-score (full vault) | 99.2 | 201.1 | **+103%** |
| — CR350 subset | — | — | +77% |
| — general-English subset | — | — | +124% |
| Entity coverage | 17,087 | 20,661 | **+21%** (more, not fewer) |
| Re-enriched graph `MENTIONS` | ~19.7k | **~27.6k** | +40% |
| Structural noise rate | 0.034 | 0.035 | flat |
| Mean specificity (IDF) | 3.19 | 3.35 | +5% |

**Typing redistribution (the root-cause fix).** The LLM dumped 60–69% of
entities into a generic `concept` bucket; OTTER spreads them across
`tool`/`method`. Live proof on `06d1557e-ensa-module-12`:

| kind | Qwen3-8B | OTTER |
|---|---|---|
| `method` | 15 | **167** (11×) |
| `tool` | 53 | **329** (6×) |
| `place` | 45 | **14** (noise removal) |
| total | 438 | 880 (+101%, a clean REPLACE via `clear_mentions`) |

A `place` spot-check on the NVIDIA 10-K: the LLM's 359 "places" were ~85%
garbage (dates, URLs, phone/zip, mis-typed orgs); OTTER kept 69, mostly real
geography (~80% precision vs ~15%). OTTER also found *more* real people
(477 → 672).

## Why OTTER wins

A span NER **types entities cleanly upstream** (tool/method/place vs the LLM's
generic `concept` bucket) and recovers far more domain spans at threshold 0.05.
The naive first run looked "mixed" (discovery *below* the LLM) purely from
**under-leveraging** — the card's 0.1 threshold + generic-only labels. The
unlock was **0.05 + union labels** (give OTTER both vocabularies, it picks best
per span). Thematically aligned, too: the reranker is already a BERT
cross-encoder, so a BERT NER is the right NLP placement — upstream of the graph,
not in retrieval.

## Scope / HARD-gate safety

OTTER is **enrich-graph-only**. Citation extraction and the assess/answer/verify
answer path stay on the LLM, so the refusal / no-hallucination HARD gates are
untouched (no eval re-baseline needed). The leverage is bounded to **discovery
quality** — `related_documents`, entity-centric retrieval — never core answer
quality.

## Limits found

- Low per-entity confidence on correct entities (`firewall` ≈ 0.11) → must run
  low-threshold; a real precision/recall trade.
- OTTER faithfully extracts SOURCE parse-artifact tokens (`NEMOCLAW`-class,
  verified in-doc) the LLM skipped — a source-quality issue, not a decode bug.
- CPU throughput ≈ 17 min full-vault (one-time; GPU-during-pause ~10× faster).

## Recommendation

Default path: `enrich_ner_backend=otter` @ threshold 0.05 + union labels
(activated in `~/.config/memex/config.toml`; the schema default stays `llm` for
backward-compat). The vault was backed up (`vault-pre-otter-…`) and fully
re-enriched. Decision recorded in [ADR-0012](../adr/0012-otter-bert-ner-enrich-backend.md);
mechanism in [`docs/specs/ner-enrich.md`](../specs/ner-enrich.md).

## Reproducibility

```sh
# Read-only A/B over the live graph (no mutations). CPU-side; CR350/CCNA
# French networking jargon is the right stress target.
MEMEX_MODELS__RERANKER_DEVICE=cpu \
  uv run python scripts/entity_ner_ab_audit.py --labels union --threshold 0.05
```

The per-seed `related_documents` side-by-side diffs (the real verdict — no
single structural metric cleanly classifies noise) print to the report; the
numbers above are the full-vault production-N confirmation.
