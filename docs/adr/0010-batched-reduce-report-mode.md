# ADR-0010: Multi-paragraph report mode via a hierarchical REDUCE

- **Status**: Accepted
- **Date**: 2026-05-28
- **Deciders**: Memex core team
- **Tags**: agents, summarization, extends-0008, relates-0009

## Context

[ADR-0009](0009-remove-free-form-synthesis-baseline.md) removed the free-form
whole-document synthesis baseline and recorded *why* a single free-form string
cannot produce a proper summary on a 12 GB 8B: the model **regurgitates** instead
of abstracting, **won't condense-and-stop**, and a capped string **force-closes
mid-word** (the "policyEn" cut). It also recorded the binding hardware fact — input
and output share the KV cache, so a free-form whole-doc summary can only complete
when `doc_tokens ≲ ½ × max_model_len` (≈12k tokens in `full` mode); past that the
window is the wall.

That left a real gap: users still want a **rich, multi-paragraph prose summary**,
which the removed baseline was *supposed* to provide. [ADR-0008](0008-document-summarization.md)'s
REDUCE produces only **one** abstract paragraph, and it caps the digests it
synthesizes at `_REDUCE_MAX_SECTIONS` (24) — so on a large document the later
sections are silently dropped from the abstract.

The insight (arrived at in discussion): the rich prose can be recovered **safely**
by *composition* rather than by one free-form generation — feed the
already-grounded, already-condensed section digests back to the model **in bounded
batches**, have it write one coherent paragraph per batch, and stitch the
paragraphs deterministically. This is the **inverse** of what ADR-0009 removed:

| Free-form failure mode (ADR-0009) | Why batched-from-digests avoids it |
|---|---|
| Regurgitates raw input | Input is *already abstractive* digests — nothing verbatim to copy |
| Won't condense-and-stop | Each batch output is `maxItems`-bounded → a natural stop |
| Mid-word force-close | The sentence-**list** `DocAbstract` ([ADR-0009] follow-up) ends at natural boundaries |
| Window arithmetic (doc + output) | No batch ever sees the whole doc; each prompt is a handful of short digests |

## Decision Drivers

- **The no-hallucination HARD gate** (`refusal_cf`, grounded-only) is non-negotiable.
- **Boundedness at every node** — the load-bearing lesson of ADR-0009. Every model
  call must be *small grounded input → bounded output*. The instant any node asks
  for one free-form string over the whole thing, the three failure modes return.
- **No section dropped.** A report should reflect the *whole* document, not a
  `_REDUCE_MAX_SECTIONS` head.
- **Reuse the validated primitive.** The bounded `DocAbstract` sentence-list reduce
  is already eval-proven; the report path should compose it, not invent a new shape.

## Decision

Add a **`report`** detail level (`SummaryDetail = brief | standard | detailed |
report`). When `detail == "report"`, REDUCE switches from the single-level
`_reduce` to a **hierarchical** `_reduce_report`:

1. **Batch** the grounded `SectionSummary` digests `_REPORT_SECTIONS_PER_BATCH` (4)
   at a time, in **document order**.
2. **Reduce each batch** with the existing `_reduce` (hence the same bounded
   `DocAbstract` primitive) → one paragraph. A `scope_note` tells the batch it
   covers "one part of a longer multi-section overview" so it does not falsely
   claim whole-document scope (the only `summarize_reduce` prompt change; an empty
   `scope_note` renders byte-identical to the single-level prompt).
3. **Stitch** the paragraphs with blank-line separators — **deterministically, no
   model call**. There is deliberately *no* final whole-output smoothing pass:
   that would reintroduce the free-form trap for a marginal cohesion gain.
   Section-organized prose is how every real report already reads.

`FinalResponse.summary` becomes the `\n\n`-joined body; everything else
(`claims`, per-section `sections`, `used_chunks`, the refusal gate) is unchanged.
Wired through every surface: CLI `--detail report`, MCP `detail="report"`, the
webui detail `<select>`. The webui renders the body by splitting `summary` on
blank lines into multiple `<p>` inside ONE `.ans-answer` block (one blue rule spans
the paragraphs); a single-paragraph summary is exactly one `<p>` (unchanged).

## Consequences

### Positive

- The rich multi-paragraph prose users wanted, delivered **within the HARD gate** —
  every paragraph derives only from already-grounded digests.
- **No section dropped**: every section reaches the output through its batch, fixing
  the single-level `_REDUCE_MAX_SECTIONS` truncation — a correctness win, not just
  verbosity.
- Reuses the eval-proven bounded `DocAbstract` reduce; the batch loop and the stitch
  are the only new logic, both pure/deterministic.
- Mode-independent like the rest of the summarizer: each batch prompt is small, so
  it fits the `fast` window — `report` produces the same output in `fast` or `full`.

### Negative / Trade-offs

- **Grounding-drift surface grows with output length.** REDUCE *recombines* grounded
  digests rather than inventing, and is trusted un-re-verified (as the single-level
  abstract always was). A longer multi-paragraph body has more room for the model to
  slip in un-grounded connective phrasing. v1 mitigates this the cheap way —
  **tight per-batch bounding** (`maxItems`-bounded `DocAbstract`, ≤4 digests in) —
  and the `must_not_assert` eval gate guards it. A per-paragraph **re-ground** pass
  (run the existing verifier over the synthesized paragraphs) is the heavier option,
  deferred until the eval shows drift.
- **Cross-paragraph cohesion isn't guaranteed** (no smoothing pass, by design).
  Accepted: deterministic section-ordered prose is a well-understood report format.
- More model calls than the single-level reduce (one per batch) → higher latency on
  a many-section doc. Bounded by `_MAX_SECTIONS` / `token_budget` like everything else.

### Neutral

- `report` rides the existing `detail` axis, so it threads through CLI/MCP/webui/eval
  with no new parameter.
- Adjacency batching preserves the document's own narrative order; theme/salience
  clustering (which would need another model call) is a possible future refinement.

## Refinements (2026-05-28): coherence, adaptive structure, confidence

The first cut produced *correct but disjoint* paragraphs (every one opened "The
document focuses on…"). Four additive layers, all preserving the bounded/grounded
invariant (small grounded input → bounded output; no free-form whole-output pass):

1. **Rolling context.** Each paragraph after the first is given the *previous*
   paragraph (`_reduce`'s new `preceding` param) and told to open with a transition
   and add only new material — so the body reads as one narrative, not blocks. Bounded
   (one prior paragraph, itself `maxItems`-capped).
2. **Open / middle / close arc.** Position-aware `scope_note`: the FIRST group is the
   OPENING (introduce the subject), the LAST is the CLOSING (conclude — "assert nothing
   beyond the digests"), the rest CONTINUE. Gated prompt blocks; single-paragraph and
   single-level paths are untouched.
3. **Adaptive structure** (`_plan_report_structure`). Instead of a fixed every-N split,
   the model chooses where paragraph breaks fall via **boundary-selection** — it returns
   the section indices that *start* a paragraph (`ReportStructure.paragraph_starts`).
   Boundary-selection over the ordered sequence **cannot drop or duplicate a section**
   (unlike free index assignment), so it's a clean partition by construction. Guardrails:
   in-range/dedup/force-a-start-at-0, **size-cap** each run at
   `_REPORT_MAX_SECTIONS_PER_PARAGRAPH` (blocks a degenerate "combine everything"), and
   **fall back to mechanical batching** on a model error or a trivial (no-break) plan.
   Every decision logs (`report.plan` / `report.plan_trivial` / `report.plan_failed`) —
   the observability the non-deterministic step needs. HARD-gate-neutral: only the
   *grouping* changes, never grounding.
4. **Faithfulness confidence** (`_score_report_confidence`, `ReportConfidence`). Closes
   the loop: each generated paragraph is scored against the digests it was built from,
   **hybrid** = lexical content-overlap (unigram+bigram precision — the deterministic
   "pattern matching" half, catches fabricated specifics) + semantic embedding cosine
   (the shared EmbeddingGemma — sees through paraphrase), combined 0.6/0.4, per-paragraph
   + overall on `FinalResponse.report_confidence`. **INFORM-ONLY** — surfaced (a quiet
   webui `.summary-confidence` line) + logged (`report.confidence`), never a gate; the
   `must_not_assert` eval stays the no-hallucination HARD gate. The embedding half is
   **best-effort**: any failure (e.g. the embedder can't load co-resident) logs
   `report.confidence_embed_unavailable` and degrades to lexical-only — confidence can
   never break a summary.

These supersede the original "adjacency batching" / "no rolling context" notes above.
Validated: full `eval-summary` HARD gate held (report case `violations=[]`), 930 tests,
and a live browser e2e on the NVIDIA 10-K + a CUDA deck.

## Revisit When

- The `must_not_assert` eval shows drift in `report` bodies, or report-mode
  `report_confidence.overall` trends low → add the per-paragraph **re-ground** pass (run
  the existing verifier over the synthesized paragraphs; the confidence score is the
  trigger signal for when it's worth the cost).
- A forcing function appears for an executive-summary-over-the-body layer (a second
  bounded reduce over the batch paragraphs).

## Summarizer swap-in (2026-05-28): the dedup lever, hardware-blocked on 12 GB

The cross-length browser testing confirmed the residual repetition is an **8B
capability limit** (it re-states a doc's thesis in every paragraph and ignores the
dedup instruction even given the full overview-so-far). The fix is a stronger
summarizer served briefly at summarize-time. **The infrastructure is built + gated off**
(`ModelSettings.summarizer=None` by default):

- `models.client.inference_override(base_url, model)` — a ContextVar routing every
  `complete_structured` in an async context to a swapped model, WITHOUT touching the
  global client (a concurrent `/ask` is unaffected).
- `agents/summarizer_serve.py::serve_summarizer_vllm` — a text-only twin of the parse
  VLM lifecycle (gid-capture-at-spawn + group-emptiness reap, copied not shared).
- `summarize_document` (report detail + a configured summarizer): an `AsyncExitStack`
  wraps the map-reduce in `pause_vllm_for_gpu()` → serve → `inference_override`; the
  `finally` reaps + restarts the orchestrator. Pure no-op when off.

**Blocker:** `gaunernst/gemma-3-12b-it-int4-awq` **OOMs on the 12 GB RTX 4070** even with
the orchestrator paused — it's the multimodal build (loads a ~1.5 GB vision tower it
doesn't need) plus Gemma-3's **262k-vocab unquantized lm_head (~1.9 GB)** on top of ~6 GB
AWQ weights → ~11 GB during load. The swap *lifecycle* worked perfectly (paused → 2 spawn
attempts → clean reap → orchestrator restored → graceful `ModelCallError`); only the model
didn't fit. **Re-enable** with `MEMEX_MODELS__SUMMARIZER=<a-model-that-fits>` — a text-only,
smaller-vocab 14B (e.g. Qwen3-14B-AWQ) is the best fit-bet on 12 GB; Gemma-3-12B needs a
bigger card. See the `summarizer-swap-in-2026-05-28` memory.
