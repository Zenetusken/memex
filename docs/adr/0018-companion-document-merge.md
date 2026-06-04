# ADR-0018: Companion-document merge — lecture transcript ↔ slide deck (grounded)

**Status:** Proposed (2026-06-03)
**Extends:** [ADR-0017](0017-audio-asr-ingestion-route.md) §"Companion-document merge (deferred to Phase 2)"
**Spec:** [`docs/specs/companion-merge.md`](../specs/companion-merge.md)

## Context

A class lecture has two halves already in the vault as separate documents: the **slide deck** (a PDF)
and the **teacher's spoken commentary** (the ADR-0017 transcript). The transcript is ALREADY a
first-class grounded RAG document — same chunking/embedding/FTS/`verify`/citation path as any PDF
(validated live: a grounded ask cited transcript chunks at `confidence:high`). The user's requirement
is that the two be tied as **jointly grounded factual evidence** — "the teacher's lectures and these
documents are considered grounded factual information" — **not** a navigation cross-link bolted on top.

ADR-0017 deferred the merge to Phase 2 and v1 carried the hooks (per-segment GLOBAL `start_s` + char-
spans + `Chunk.time_range`). Both the CR350 video transcripts AND their `Cours N.pdf` / `Semaine N.pdf`
slide decks are now indexed, so the merge is buildable on real data.

A pre-design measurement (2026-06-03) settled an adjacent question — whether transcript chunks need a
relevance/salience classification layer to keep noise out of retrieval. **Measured NO:** the worst-case
(intro) lecture is only 2.7% true noise (social/artifact), 29.5% is legitimately-queried *admin*
content, and dense retrieval leaks ZERO noise into real-query top-10. The embedder's semantic relevance
is already the right "layer." So the merge builds on clean, already-grounded transcripts.

## Decision

Build a **user-paired** companion-merge that aligns each transcript CHUNK to its best slide-deck PAGE
by **EmbeddingGemma cosine** (the MaViLS method), stores the alignment as **derived sidecar state**,
writes a **document-level CITES link**, and — the load-bearing part — makes slide + commentary
**jointly groundable** via a **HARD-gate-safe retrieval AUGMENTATION**, not chunk fusion.

Load-bearing decisions:

1. **Grounded integration via RETRIEVAL AUGMENTATION, not chunk fusion.** When the retriever surfaces
   one side of an aligned pair, a default-OFF `augment_companion` node pulls the aligned counterpart
   chunk(s) into the candidate pool (a deck page → the transcript chunks that explain it, and
   vice-versa). Each chunk stays PURE (own content-addressed `chunk_id`, own source); `verify` grounds
   every claim against its OWN chunk. **HARD-gate-safe by the `expand_graph` construction** — it only
   adds REAL indexed chunks, never fabricates/fuses/mutates, so no hallucination path opens and a
   counterfactual stays refused. **Chunk-fusion (slide+commentary in one chunk) is REJECTED** — it
   churns `chunk_id`s and injects mixed-source text into grounding (a real gate risk).

2. **The alignment is DERIVED + navigation-grade, never on the grounding path itself.** The per-chunk→
   page map is a regenerable sidecar (`vault/.memex/companion_alignments.json`, atomic write); the
   doc-level link is a reciprocal CITES edge (`related_documents` surfaces it). Neither is read by
   `answer_query`/`verify`/the `chunk_id`/the embeddings — the `Chunk.page` + `related_documents` +
   `scope_sets` precedents. Only the §1 augmentation touches retrieval, and it is gated + eval-gated.

3. **Honest accuracy: transcript-text→slide-text cosine is MaViLS's WEAKEST modality (F1 ≈ 0.53),** not
   its multimodal 0.82 — we discard the strong video-FRAME-OCR signal (0.76) per ADR-0017's "visual
   track ignored." v1 is per-chunk argmax cosine + a null floor (`τ_null`, honestly un-calibrated) + a
   cheap monotonic tie-break. The MaViLS asymmetric-jump DP (with our REAL `start_s` time prior) and
   video-keyframe-OCR matching (recovers the discarded strong signal) are scoped FAST-FOLLOWS.

4. **Pairing is USER-SPECIFIED** (`memex link-slides <transcript> <deck>`). Auto-pairing is non-trivial
   (Cours-04 video ↔ *Semaine*-4 deck) and a wrong pair mis-attributes commentary — deferred to a
   SUGGEST-only layer.

## Consequences

**Positive:** slide + the teacher's explanation answer questions together, each grounded to its real
source; the transcript stays first-class; navigation between a `[mm:ss]` point and its slide; reuses
the embedder + graph + sidecar precedents (no new model/runtime).

**Negative / trade-offs:** transcript-only alignment is modest (~0.53 F1) until the DP / keyframe
fast-follows; `τ_null` is un-calibrated without a gold set; the augmentation is HARD-gate-ADJACENT so
it ships default-OFF behind an eval gate; user must name each pair.

**HARD-gate posture:** alignment + link + nav surfaces are neutral by construction; the augmentation is
validated (`refusal_cf=1.0` must hold with it ON over a real pair) before any default-ON flip.

## Alternatives considered

- **Chunk-fusion (one slide+commentary chunk)** — REJECTED: chunk_id churn + mixed-source grounding.
- **Nav-only sidecar (no grounding effect)** — INSUFFICIENT: the user explicitly wants grounded
  integration, "not just logically integrated." Kept as the always-safe fallback if the augmentation
  eval doesn't clear.
- **A transcript salience/classification layer** — MEASURED-AWAY (2.7% noise, 0 retrieval leakage; the
  embedder already filters; admin ≠ noise).
- **Auto-pairing by name/dir** — REJECTED for v1 (silent mis-pairing risk).
- **A new alignment model / learned fusion** — REJECTED: reuse EmbeddingGemma; MaViLS found no fusion
  gain.

## Revisit when

- v1's argmax under-performs on a real pair → ship the asymmetric-jump DP + the `start_s` prior.
- Alignment is still weak → video-keyframe-OCR matching (the principled accuracy lever).
- A gold transcript↔slide alignment set exists → calibrate `τ_null` + the (query,doc)/(doc,doc) A/B.
- The augmentation eval clears with a measured win → consider default-ON.
