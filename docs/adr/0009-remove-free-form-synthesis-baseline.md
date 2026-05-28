# ADR-0009: Remove the free-form synthesis baseline

- **Status**: Accepted
- **Date**: 2026-05-28
- **Deciders**: Memex core team
- **Tags**: agents, summarization, models, cleanup, supersedes-0007

## Context

[ADR-0007](0007-co-residence-resource-modes.md) shipped `full` mode (a
~24,576-token orchestrator window) and, as its first consumer, a **free-form
whole-document synthesis baseline** — `agents/synthesize.py::synthesize_documents`:
assemble whole docs into one prompt and ask the model for a single
`Synthesis.summary` string sized to the window.

[ADR-0008](0008-document-summarization.md) then superseded that baseline *for the
summary use case* with the structured, grounded, map-reduce `summarize_document`,
and recorded the two hard facts that made the free-form approach a dead end:

1. **vLLM guided-JSON enforces list `maxItems` but NOT string `maxLength`.** A
   schema with one big `summary: str(max_length=N)` does not stop the model at N
   — it runs to `max_tokens`, truncates mid-emission, and the JSON fails to parse.
2. **An 8B model regurgitates on a large/dense document** instead of abstracting,
   overrunning the window.

Despite ADR-0008, `synthesize.py` was left in the tree. A 2026-05-28 audit of
`full` mode found it had become **dead weight that actively misrepresented the
mode's capability**:

- **Unwired.** No CLI / MCP / webui surface imports it — only its own tests do.
  Switching to `full` mode therefore exposed a 24,576 window that no shipped,
  gate-safe feature consumed.
- **Broken by construction.** Live runs (NIST SP 800-207 *and* prose
  `GUIDELINES`, ~10.5k tokens, at `max_output_tokens=4096`) both failed: the
  model overran the output budget before closing the unbounded `summary` string
  → `ModelCallError` → a graceful "exceeded output budget" refusal. This is fact
  (1) in practice — there is no parameter tuning that fixes it, because the
  string bound is not enforced.
- **Ungated.** It emits a free-form summary with no grounding / refusal gate —
  contrary to Memex's no-hallucination HARD gate.
- **Redundant.** The reliable form of "a bounded summary" *is* `maxItems`-lists
  of short strings — which is exactly `summarize_document`. Its `short` route
  already does a single grounded pass for a document that fits one window, so the
  one thing free-form synthesis offered (single-pass whole-doc) is covered, with
  grounding, by the structured path.

The `full`-mode copy ("~24,576 tokens (whole document)") implied a working
whole-document synthesis that did not exist.

## Decision Drivers

- **No dead or misleading code.** A mode must not advertise a capability nothing
  delivers.
- **The HARD gate.** Memex does not ship an ungated free-form summarizer.
- **Reliability is structural, not tunable.** Fact (1) means free-form single-shot
  cannot be made reliable on this stack; only `maxItems`-bounded output can.
- **One path, not two near-duplicates.** `summarize_document` already covers the
  whole size range (short single-pass → map-reduce), gated and mode-independent.

## Decision

**Remove the free-form synthesis baseline entirely:**

- delete `src/memex/agents/synthesize.py` (`synthesize_documents`, `SourceDoc`,
  `Synthesis`, `_bounded_synthesis`, `_assemble`),
- delete `src/memex/prompts/synthesize/v1.md`,
- delete `tests/integration/test_synthesize.py`.

**`summarize_document` (ADR-0008) is THE long-document summary path** — reliable
(bounded lists), grounded (the HARD gate), and mode-independent.

**`full` mode's value is re-grounded honestly** (no synthesis dependency):

- the **ask path grounds against more reranked chunks** in `full` mode
  (`ResourceProfile.retrieval_top_k`: 18 vs 5 — the larger window holds them);
  HARD-gate-neutral, since more candidates only add evidence and the refusal gate
  is unchanged. The `/resources` copy now reads "24,576 tokens · top-18 chunks",
  not "(whole document)".
- the larger KV reservation also raises in-flight concurrency for the
  reranker-on-CPU posture.

## Consequences

### Positive

- `full` mode no longer advertises a capability nothing delivers; its copy is
  honest and its leverage (deeper retrieval) actually fires.
- One summary path (`summarize_document`), gated and reliable — no ungated
  free-form surface could ever be wired by mistake.
- Less surface to maintain; the negative result is preserved here (and in
  ADR-0008's context) so the dead end is not re-walked.

### Negative / Trade-offs

- Memex has **no single-pass whole-document free-form summary**. This is
  acceptable: it never worked reliably (fact 1), and `summarize_document`'s
  `short` route is the gated single-pass for docs that fit a window; larger docs
  map-reduce.
- The `agents/synthesize.py` tests (which only ever pinned the broken baseline's
  own shape) are removed with it.

### Neutral

- A future *reliable* long-context single-pass summary, if ever wanted, would be a
  **structured** (`maxItems`-bounded) pass over a bigger window — i.e.
  `summarize_document`'s short route with a `full`-sized budget — not a revived
  free-form string. The structured spine already supports that shape.

## Revisit When

- A local model reliably **condenses** (not regurgitates) a whole large document
  in one pass, AND the inference engine enforces string `maxLength` (or an
  equivalent bounded free-form mechanism lands) — then a single-pass free-form
  summary could return without the fact-(1) failure mode.
- A forcing function appears for whole-document single-pass synthesis that the
  structured short-route + larger window cannot satisfy.
