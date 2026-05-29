# Spec: Entity extraction at enrich — the OTTER BERT-NER backend

Status: **SHIPPED + LIVE 2026-05-29** (config `enrich_ner_backend=otter` @ `enrich_ner_threshold=0.05` + `enrich_ner_labels=union`). Decision record: [ADR-0012](../adr/0012-otter-bert-ner-enrich-backend.md). A/B record: [`docs/audits/08-otter-ner-ab.md`](../audits/08-otter-ner-ab.md). Commits `0583600` (A/B harness) / `0891480` (backend) / `ba8042c` (`clear_mentions`) / `1b8aaa7` (roadmap note).

The pluggable entity-extraction backend at enrich. `AgentsSettings.enrich_ner_backend` selects `"llm"` (the orchestrator Qwen3-8B, default — unchanged) or `"otter"` (a BERT span NER). Only the entity SOURCE changes; **citations always stay on the LLM**, and the surface is enrich-graph-only ⇒ HARD-gate-neutral.

## Problem

The LLM entity extractor (`enrich/entities.py`, Qwen3-8B prompted to emit a per-chunk entity list) is the generic-entity root cause behind weak graph discovery (ADR-0011): **60–69% of its output collapsed into a generic `concept` bucket**, and it **fragmented multi-word entities** (`STP` → `spanning`). The `related_documents` specificity ranking and the acronym bridge can only work as well as the entities they're given; the noise is set at extraction, and no downstream filter (the removed `entity_stopwords` list, the falsified auto-noise-detector) fixes a *typing* problem.

## Model

`whoisjones/otter-bi-mmbert` — a GLiNER-style **multilingual span NER** on the **mmBERT** backbone (covers the FR+EN corpus natively). A bi-encoder ≈ 0.5B params / 1.91 GB F32 (mmBERT-base token encoder + a `bert-base-multilingual-uncased` type encoder). Offline-OK; loaded F32 on CPU by default. `trust_remote_code=True` is required for its custom `OtterBiEncoderModel` + the dynamic `AllLabelsCollator` (audit the bundled modeling code per local-first hygiene; vendor *both* referenced encoder checkpoints + pin a commit hash for a true air-gapped run).

## Design

`enrich/ner_otter.py::extract_chunk_entities(chunk) -> list[Entity]` returns the **same per-chunk `list[Entity]`** the LLM path's `merge_entities` emits, so it flows through the **unchanged** document-level `dedupe` + graph write — the dispatch is a single branch in `enrich/pipeline.py::_extract_chunk` on `otter_backend_enabled()`. Internals:

- **Lazy process-global, out-of-registry.** `_get_handle()` loads once under `_load_lock`; the model is *not* in `models/registry` because it's a CPU-side (or pause-vLLM-window GPU) enrich-time model with no answer-path co-residence — so the registry's GPU OOM-breaker / co-residence machinery doesn't apply. Same out-of-registry precedent as the parse-time VLM serve ([`vlm-vllm-serving.md`](vlm-vllm-serving.md)) and the summarizer swap-in.
- **Lock-serialized forward.** Enrich is CPU-bound here and a single torch CPU model isn't reentrancy-guaranteed, so `_OtterHandle.predict_entities` holds a `threading.Lock`; the async surface is `asyncio.to_thread`.
- **Subword-span decode + garble filter.** `model.predict(batch, threshold)` returns spans as *subword-token* indices; the surface is decoded from `input_ids[start:end+1]`, and a span whose whitespace-normalised decode is *not* a substring of the source is dropped (cross-token-boundary garble).
- **Label → kind taxonomy.** Zero-shot labels map to Memex's 7 kinds (`person`/`org`/`place`/`concept`/`method`/`tool`/`other`) via the `_LABEL_PRESETS`: `generic` (single-word labels), `domain` (descriptive networking/security phrases), and `union = {**domain, **generic}`.
- **`_PASSAGE_CHARS = 6000`** mirrors the LLM path's `extract_entities/v2` `truncate(6000)` — input parity.

## The two master knobs

- **`enrich_ner_threshold` (0.05).** *The* master knob. The model card's 0.1 strangles recall (a naive 0.1/generic run looked "mixed" — discovery *below* the LLM — purely from under-leveraging); 0.05 is the sweet spot; 0.02 maxes recall but dilutes specificity. Low per-entity confidence on correct entities (e.g. `firewall` ≈ 0.11) is a real precision/recall trade — run low-threshold.
- **`enrich_ner_labels` (`union`).** Domain phrases ~10× the confidence on attack entities and fix mis-types, but `domain`-alone hurts general English; `union` wins **both** corpora (give OTTER both vocabularies, it picks best per span).

## `clear_mentions` — REPLACE, not append

`GraphStore.clear_mentions(doc_id)` (commit `ba8042c`) runs before re-linking so a re-enrich **REPLACES** a doc's `MENTIONS` rather than appending. Without it (the prior merge-append behaviour) a backend switch would leave stale Qwen entities alongside OTTER's — a latent bug the swap exposed.

## A/B verdict (the operating point 0.05 + union)

Measured against the live 47-doc graph with the verbatim discovery scorer (`_rank_related_documents` + `_ENTITY_KIND_WEIGHT` + the generic-df filter) via `scripts/entity_ner_ab_audit.py` (read-only):

- **`related_documents` discovery (the on-mission metric): +103% full-vault** (mean top-score 99.2 → 201.1; CR350 +77%, general-English +124%).
- **Coverage +21%** (17,087 → 20,661 entities in the A/B — *more*, not fewer); noise flat (0.034 → 0.035); specificity +5%. The re-enriched graph carries **~27.6k `MENTIONS` (+40%** over the LLM's ~19.7k).
- **Cleaner typing** — the LLM's `concept`-dump redistributes to `tool`/`method`. Live proof on `ensa-module-12`: `method` 15 → 167, `tool` 53 → 329, `place`-noise 45 → 14. Corpus-wide `place`-noise 1539 → ~400 (10-K spot-check: OTTER place-precision ~80% vs the LLM's ~15%); `person` 477 → 672 (*more* real people).
- Full-vault enrich ≈ **17 min CPU** (one-time; GPU-during-pause ~10× faster), 0 doc-errors.

## HARD-gate invariant

OTTER is **enrich-graph-only**. Citation extraction and the assess/answer/verify answer path stay on the LLM, so the no-hallucination / `refusal_cf=1.0` HARD gates are untouched — no eval re-baseline is needed. The leverage is **discovery quality** (`related_documents`, entity-centric retrieval), a deliberate bounded scope, never core answer quality.

## Anti-scope (don't do these)

- **Don't move entity extraction onto the answer path** — that would break the HARD-gate-neutral premise (re-examine ADR-0012 first).
- **Don't reintroduce a curated by-name noise list** (`entity_stopwords`, removed `bf44f43`) — fix entity noise at the extractor, not downstream. The corpus-agnostic `cooccurring_min_shared_docs` floor is kept.
- **Don't raise the threshold to the card's 0.1** without re-measuring — it strangles recall and resurfaces the "mixed" mirage.
- Residual junk that survives is **SOURCE parse-artifacts** (e.g. `NEMOCLAW`, verified in-doc), a parse-quality issue — not an OTTER mis-extraction to tune around.

## Config

| Setting (`MEMEX_AGENTS__…`) | Default | Live | Purpose |
|---|---|---|---|
| `enrich_ner_backend` | `llm` | `otter` | `llm` (Qwen3-8B) or `otter` (the span NER) |
| `enrich_ner_model` | `whoisjones/otter-bi-mmbert` | — | HF id (only when `otter`) |
| `enrich_ner_device` | `cpu` | — | `cpu` or `cuda` (viable in the CLI enrich pause-window) |
| `enrich_ner_threshold` | `0.05` | `0.05` | span-confidence floor (the master knob) |
| `enrich_ner_labels` | `union` | `union` | `generic` / `domain` / `union` (A/B winner) |
| `enrich_ner_max_seq_length` | `512` | — | collator window (chunks < 512 tok) |

Switching backends requires re-running `memex enrich` (or `reindex`) on existing docs (`clear_mentions` makes it a clean REPLACE). The schema default stays `llm` (backward-compatible); the live install activates `otter` in `~/.config/memex/config.toml`.

## Testing

- `tests/unit/test_ner_otter.py` — the adapter (label-preset mapping, subword decode, garble filter, fail-safe-to-LLM).
- `tests/integration/test_entity_profile.py` — the real-ryugraph `test_clear_mentions_replaces_not_appends` (REPLACE semantics, behind `importorskip("ryugraph")`); `tests/integration/test_enrich_and_graph.py` — fake-graph `clear_mentions` parity.
- Hermeticity: the autouse `tests/conftest.py` fixture points `MemexSettings`' TOML source at an empty file so the live `config.toml` (`backend=otter`) can't bleed into tests (caught a real failure where enrich tests loaded OTTER).
