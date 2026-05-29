# ADR-0012: A BERT Span-NER (OTTER) Extracts Entities at Enrich, Not the LLM

- **Status**: Accepted
- **Date**: 2026-05-29
- **Deciders**: Memex core team
- **Tags**: architecture, enrich, graph, models, ner

## Context

ADR-0011 settled the entity graph's role as explicit **discovery** (`related_documents`, entity-centric retrieval) and, in its Build-out + Revisit-When, named the residual entity noise — `STP` fragmented to `spanning`, the course code `CR350` mis-typed as a concept, junk ports, bilingual connectors — as an **NER problem upstream of the graph**, not a resolution or ranking one. It even proved the point negatively: a scoped auto-noise-detection helper was falsified because no graph *structure* separates a connector from a central concept; only *semantics/typing* can.

Until now the enrich entity extractor was the orchestrator LLM (Qwen3-8B) prompted to emit a per-chunk entity list (`enrich/entities.py`). Two failure modes degraded the discovery graph at its source: (1) **60–69% of its output collapsed into a generic `concept` bucket** (measured live), so the specificity ranking had little signal to discriminate on; (2) it **fragmented multi-word entities** (`STP` → `spanning`), so the acronym bridge had nothing to bridge to. The graph DATA is healthy; the entity *typing/specificity* is the weak link, and it is fixed at extraction.

The question this ADR settles: should entity extraction stay on the LLM, or move to a dedicated NER engine — and at what blast radius.

## Decision Drivers

- Fix the generic-entity **root cause upstream** of the graph (ADR-0011's named lever), not with more downstream heuristics (the removed `entity_stopwords` band-aid).
- **HARD-gate safety**: the change must not touch the answer/refusal path. Citations stay on the LLM; entities are enrich-graph-only.
- **Local-first / 12 GB**: a CPU-side (or pause-vLLM-window GPU) enrich-time model, with no answer-path co-residence.
- **Evidence over intuition**: an A/B over the live `related_documents` specificity scorer on the real 47-doc vault before flipping the default.
- **Reversibility**: a config flag whose default keeps the LLM extractor.

## Considered Options

1. **Keep the LLM extractor, add more downstream noise heuristics** (per-class regexes, the curated `entity_stopwords` list) — rejected, doesn't generalise.
2. **Add an optional BERT span-NER backend (OTTER), gated behind a config flag** (chosen).
3. **A heavier NER or a fine-tune** — deferred (OTTER fits the CPU/12 GB enrich budget and the discovery-only blast radius doesn't justify more).

## Decision

We added an **optional BERT span-NER backend, OTTER** (`whoisjones/otter-bi-mmbert` — a GLiNER-style multilingual span NER on the mmBERT backbone), for entity extraction **at enrich**, selected by `AgentsSettings.enrich_ner_backend` (`"llm"` default = unchanged | `"otter"`). The live `~/.config/memex/config.toml` activates `"otter"` @ `enrich_ner_threshold=0.05` + `enrich_ner_labels="union"`.

It produces the **same per-chunk `list[Entity]`** the LLM path's `merge_entities` emits, so it flows through the unchanged document-level `dedupe` + graph write — **only the entity SOURCE changes; citations stay on the LLM**. The adapter (`src/memex/enrich/ner_otter.py`) is a lazy process-global loaded **out of `models/registry`** — it's a CPU-side (or pause-window GPU) enrich-time model with no answer-path co-residence, so the registry's OOM-breaker / co-residence machinery doesn't apply (the same out-of-registry precedent as the parse-time VLM and summarizer serves). Its forward is lock-serialized (CPU-bound, not reentrancy-guaranteed). `trust_remote_code=True` is required for OTTER's custom `OtterBiEncoderModel` + `AllLabelsCollator` (verified against the model's shipped code, 2026-05-29). Zero-shot labels map to Memex's 7-kind taxonomy (`person`/`org`/`place`/`concept`/`method`/`tool`/`other`) via the `generic`/`domain`/`union` presets; `union` (both vocabularies) is the A/B winner.

`GraphStore.clear_mentions(doc_id)` (commit `ba8042c`) makes a re-enrich **REPLACE** a doc's `MENTIONS` rather than append — the prior `enrich_document` merge-appended, so a backend switch would otherwise leave stale Qwen entities alongside OTTER's. The whole 47-doc vault was backed up (`vault-pre-otter-…`, the vault is not git-tracked) and re-enriched in one process.

## Consequences

### Positive

- **+103% `related_documents` discovery** on the full vault (mean top-score 99.2 → 201.1), **+21% entity coverage** (17,087 → 20,661 in the A/B — *more*, not fewer), noise flat (0.034 → 0.035), specificity +5%. The re-enriched graph holds **~27.6k `MENTIONS` (+40% over the LLM's ~19.7k)**. Subsets: CR350 +77%, general-English +124%.
- **Far cleaner typing** — the LLM's 60–69% `concept`-dump redistributes to `tool`/`method`. Live proof on `ensa-module-12`: `method` 15 → 167, `tool` 53 → 329, `place`-noise 45 → 14; corpus-wide `place`-noise 1539 → ~400 (an NVIDIA-10-K spot-check put OTTER place-precision ~80% vs the LLM's ~15%), and OTTER finds *more* real people (477 → 672). This is the root-cause fix ADR-0011 named.
- **HARD-gate-neutral by construction**: enrich-graph-only; citations and the assess/answer/verify path are untouched, so no eval re-baseline is needed. The acronym bridge and the kept `cooccurring_min_shared_docs` floor now operate on better-typed entities.

### Negative / Trade-offs

- A new model dependency and a new `trust_remote_code` surface at enrich. Mitigated: out-of-registry, CPU/pause-window only, **never the answer path** — a narrower blast radius than the chart-OCR `trust_remote_code` carve-out (ADR-0006 §4), which is also parse/enrich-side, since OTTER never co-resides with answering.
- **Threshold is a master knob**: the model card's 0.1 strangles recall; 0.05 + union labels was the A/B unlock (a naive 0.1/generic run looked "mixed" purely from under-leveraging). A mis-set threshold quietly under-extracts.
- Re-enrich is now a REPLACE, so an interrupted re-enrich leaves a doc with fewer `MENTIONS` until it completes (acceptable — derived state, ADR-0003).
- CPU enrich is ~17 min full-vault (one-time; GPU-during-pause ~10× faster). Viable.

### Neutral

- The default stays `"llm"` (fully backward-compatible); the swap is config-activated.
- Residual junk that remains is **SOURCE parse-artifacts** (`NEMOCLAW`-class tokens verified in-doc), not OTTER mis-extraction — a parse-quality issue, separately tracked.

## Alternatives in Detail

### Keep the LLM extractor + downstream heuristics

The removed `entity_stopwords` list (ADR-0011 follow-up) was the band-aid; a hand-curated per-corpus name list (one user's `CR350`) doesn't generalise to a local-first app run on arbitrary corpora. The auto-noise-detection helper was scoped and falsified (no structural signal classifies noise). Rejected — fix at the extractor, where typing lives.

### A heavier NER / a fine-tune

Deferred. OTTER's mmBERT backbone (~0.5B, 1.91 GB F32) fits the CPU/12 GB enrich budget zero-shot (no labelled corpus needed), and the discovery-only blast radius doesn't justify a training pipeline or a larger model.

## Revisit When

- Discovery quality is still entity-typing-bound after OTTER — then a sharper NER, a fine-tune, label-set tuning, or OTTER∪LLM fusion (scoped, currently moot — OTTER-alone already wins).
- A future entity consumer touches the **answer path** — then the HARD-gate-neutral "enrich-graph-only" premise must be re-examined.
- OTTER's upstream archives, or a materially better offline multilingual span-NER appears.

## References

- [ADR-0011](0011-entity-graph-from-expansion-to-discovery.md) — the graph's discovery role; named this swap as the deferred root-cause fix (Build-out "Deferred", Revisit-When #3)
- [ADR-0005](0005-ryugraph-replaces-kuzu.md) — the graph store OTTER writes into; [ADR-0006](0006-cuda-dispatch-and-dtype.md) §4 — the `trust_remote_code` posture; [ADR-0003](0003-markdown-vault-as-source-of-truth.md) — the graph is regenerable derived state
- `docs/specs/ner-enrich.md` — the backend mechanism; `docs/specs/graph-discovery.md` — the discovery consumers; `docs/audits/08-otter-ner-ab.md` — the A/B record; `scripts/entity_ner_ab_audit.py` — the (read-only) reproduction harness
- Commits `0583600` (A/B harness), `0891480` (OTTER backend), `ba8042c` (`clear_mentions`), `1b8aaa7` (roadmap note)
- [[bert-ner-enrich-scope-2026-05-28]]
