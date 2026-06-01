# ADR-0013: An Ungrounded Reasoning "Expert" Mode (Inverts the Grounding Contract)

- **Status**: Proposed
- **Date**: 2026-05-29
- **Deciders**: Memex core team

> **Proposed, not built.** This ADR records a decision boundary *before* implementation so the contract inversion can be reviewed up front. The surface does not exist yet; the ROADMAP carries it as the headline next item.

- **Tags**: agents, models, reasoning, ux, architecture, contract-inversion

## Context

Memex's defining HARD gate is **grounded-only answering**: every shipped claim is verified against a retrieved chunk, and a question the vault cannot support is **refused** (`refusal_cf=1.0`, 0 hallucinations) — the contract that `/ask`, `summarize`, and the eval suite all enforce, and the one VISION principle the test suite makes non-aspirational. It is the project's most load-bearing invariant.

But a class of questions goes **beyond a vault lookup** — analytical, synthesis, advisory, threat-modeling, tutoring. There the user wants the **model's expertise** reasoning over (or beyond) the retrieved evidence, not a refusal. Serving that on the `/ask` path would mean relaxing the very gate that makes Memex trustworthy. The Cisco security-LLM scope ([[cisco-security-llm-scope-2026-05-29]]) sharpened this: a *reasoning* model (e.g. `fdtn-ai/Foundation-Sec-8B-Reasoning`) has no useful home on the **grounded** orchestrator path (its `<think>` traces fight guided-JSON, and the grounded corpus failures are retrieval-bound anyway) — its real home is an *ungrounded* surface.

## Decision Drivers

- **Do NOT weaken the grounded `/ask` contract** — the no-hallucination gate is non-negotiable on the gated path.
- **Honesty of provenance** — an ungrounded answer must be unmistakably labelled "model knowledge, not your vault."
- **Reuse, not a parallel stack** — the summarizer swap-in seam (the `inference_override` ContextVar + a serve-at-call-time vLLM lifecycle) already exists for a model swap.
- **Local-first / 12 GB** — a reasoning model (e.g. Foundation-Sec-8B-Reasoning for the security flavour) must fit / self-quantize on the reference card (an inherited prerequisite). **Update (2026-06-01, ADR-0015):** the default orchestrator is now `cyankiwi/Qwen3.5-4B-AWQ-4bit`, a *hybrid-reasoning* model already resident and 12 GB-proven — a general-purpose reasoning candidate for this surface that needs no self-quantize (toggle its `enable_thinking` ON here, where there is no guided-JSON grammar suppressing the CoT). The domain-specialised Foundation-Sec variant remains the security-flavour option behind its own self-quantize prerequisite.

## Considered Options

1. **A ROADMAP item only** — rejected: it inverts the project's headline contract, which warrants an up-front recorded decision.
2. **Relax `/ask`'s grounding gate for "analytical" questions** — rejected: contaminates the trusted path.
3. **A NEW, SEPARATE ungrounded surface (a sibling of `summarize`), kept OFF the gated `/ask` path** (chosen / proposed).

## Decision (Proposed)

Add a **new ungrounded "expert / analysis" surface**, separate from `/ask` (its own CLI/MCP/webui entry, the way `summarize` is separate), where a **reasoning model answers from domain expertise + chain-of-thought**. It is labelled "model knowledge, not your vault" and kept **off** the gated `/ask` path — `/ask`'s grounded HARD gate is unchanged, and this surface is **not** judged by `refusal_cf`/no-hallucination (a different, non-refusal eval discipline applies).

The defensible **v1 is reasoning OVER retrieved evidence**: relax literal-grounding to "supported-by-the-evidence-set" (not verbatim-cited), still anchored to retrieved context, so the inversion is bounded rather than free-form. It **reuses the summarizer swap-in seam** (the model is swapped in at call-time via `inference_override`) and inherits the reasoning-model 12 GB self-quantize prerequisite.

## Consequences

### Positive

- Serves the analytical/advisory questions `/ask` must refuse, without weakening `/ask`.
- Reuses existing infra (the swap-in seam, co-residence modes, the eval runner shape).

### Negative / Trade-offs

- A second answer surface with a **different (relaxed) trust contract** — the labelling and the separation must be airtight so a user never confuses an expert-mode answer for a grounded one.
- Inherits the reasoning-model availability + 12 GB self-quantize prerequisite (no verified AWQ for the security candidate today; see the Cisco scope).
- Needs its own (non-refusal) eval discipline — the grounded `refusal_cf` gate doesn't apply.

### Neutral

- `/ask`, `summarize`, and their HARD gates are untouched.
- VISION's grounding/no-hallucination principle will need an explicit **carve-out** noting this surface when it lands — not before.

## Alternatives in Detail

### Relax `/ask`'s gate

Contaminates the trusted path; the whole point is to keep the inversion off it. A user who asked a vault question must never silently get model-knowledge instead.

### A ROADMAP item only

Underweights a contract inversion of the project's load-bearing invariant. Recording it as **Proposed** lets the boundary (separate surface, off the `/ask` gate, the "model knowledge, not your vault" label, the v1 = reasoning-over-retrieved-evidence bound) be reviewed before code lands.

## Revisit When

- Implementation lands → move Status to **Accepted**, record the realized contract boundary + the eval discipline, and add the VISION carve-out.
- A reasoning model with a clean 12 GB-fitting build appears (unblocks the security variant). **(Partially fired 2026-06-01 — the now-default Qwen3.5-4B is a 12 GB-fitting hybrid-reasoning model usable as the general-purpose expert model with `enable_thinking` ON; the security-specialised variant is still gated on a self-quantize. See ADR-0015.)**
- The relaxed-grounding boundary proves too loose (a user mistakes expert output for grounded) → tighten the labelling / separation.

## References

- **Spec:** [`grounded-agentic-chat.md`](../specs/grounded-agentic-chat.md) — the implementation design; this ADR's ungrounded expert mode is its **Surface B** (fenced), sibling to the grounded multi-turn chat (Surface A) that is the primary build.
- [ADR-0008](0008-document-summarization.md) — the grounded summarizer (the contract this inverts) + the swap-in seam it reuses; [ADR-0007](0007-co-residence-resource-modes.md) — co-residence modes; [ADR-0001](0001-vllm-as-sole-inference-engine.md) — vLLM serve
- [[reasoning-expert-mode-scope-2026-05-29]], [[cisco-security-llm-scope-2026-05-29]] — the Foundation-Sec-8B-Reasoning candidate + its 12 GB self-quantize prerequisite
