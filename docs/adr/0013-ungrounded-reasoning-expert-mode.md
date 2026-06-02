# ADR-0013: An Ungrounded Reasoning "Expert" Mode (Inverts the Grounding Contract)

- **Status**: Accepted (v1 shipped 2026-06-01)
- **Date**: 2026-05-29 (proposed) · 2026-06-01 (accepted, v1 shipped)
- **Deciders**: Memex core team

> **Shipped 2026-06-01.** v1 is live behind `AgentsSettings.expert_mode_enabled` (default OFF) on CLI (`memex expert`) + webui (`/expert`). The realized contract boundary + the load-bearing implementation finding are recorded in [§Realized v1](#realized-v1-2026-06-01) below; the proposal text above it is preserved as the original decision record.

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

## Realized v1 (2026-06-01)

What shipped, and where it diverged from the proposal:

- **Surfaces = CLI `memex expert` + webui `/expert` only — NOT MCP.** MCP is reserved for a separate upstream purpose (a flagship-model fallback *into* Memex), so the new local-reasoning surface stays off it (the same scoping as the grounded chat, Surface A). Off by default behind `AgentsSettings.expert_mode_enabled`; the webui nav link is hidden until enabled (no dead link).
- **The engine is `models/client.py::complete_reasoning`** — a free-text call that passes **no `response_format`** (so there is no guided-JSON grammar to suppress reasoning) and sets `enable_thinking` via `chat_template_kwargs`. It is deliberately a *separate* function from `complete_structured`, which stays the SOLE emitter of `response_format` and never sets `chat_template_kwargs` — so "is this call grounded?" is answerable by *which function* a call uses. The grounded `/ask` + chat graph never imports it.
- **No swap-in subprocess in v1.** Because the default orchestrator (ADR-0015) is *itself* a hybrid-reasoning model, the expert call hits the **live daemon** directly (`models.reasoner = None` → the orchestrator id). The summarizer-style swap-in seam remains the documented hook for a *distinct* specialist (e.g. Foundation-Sec, gated on its self-quantize) via `MEMEX_MODELS__REASONER`, unused in v1.
- **The pipeline is retrieve (hybrid) → rerank → reason** (`agents/expert.py::expert_answer`). The reranked chunks are shown as **Evidence** (context, not grounding cites); the prompt (`prompts/expert_answer/v1`) instructs the model to prefer the evidence for facts about the user's documents and to **say so when it reasons beyond it**. It **never calls `verify` / `assess_relevance`** — there is no grounding gate here, by construction.
- **Load-bearing finding — `enable_thinking` defaults to FALSE.** Verified live on the 4B via vLLM: `enable_thinking=true` emits a **verbose, UNTAGGED "Thinking Process" scratchpad** (no `<think>` tag on this checkpoint) that *consumes the entire token budget before reaching the answer* and can't be cleanly split from it — poor for a reader-facing surface. v1 therefore reasons with `enable_thinking=false` + the reasoning-eliciting prompt, which yields clean analytical prose that is honest about evidence limits. The dual-decode kwarg **and** a defensive `split_think` (strips a `<think>…</think>` trace if one ever appears) are plumbed through as an opt-in for a future model / a vLLM `--reasoning-parser` that emits a *separable* trace.
- **Provenance is stamped on every answer** (`EXPERT_PROVENANCE_NOTE`, a deterministic constant — never model-generated): CLI prints it as a `⚠` caveat; the webui renders an amber "ungrounded" banner above the form *and* the caveat below the answer (colour **and** the explicit label, WCAG 1.4.1; the amber reuses the established `.ans-flash-refused` caution tone).
- **The grounded surfaces are byte-untouched** — the change is purely additive (a new module, a new prompt, two off-by-default config flags, two surfaces); `/ask`, chat, summarize, MCP, and their HARD gates are unchanged.
- **Eval discipline — `eval-expert` SHIPPED 2026-06-01** (spec [`docs/specs/expert-eval.md`](../specs/expert-eval.md); `memex eval-expert`). Designed + adversarially hardened via a multi-agent workflow, it is an **HONESTY + REGRESSION tripwire, NOT a correctness proof** (a coherent, faithful, well-hedged *wrong* recommendation passes every signal green). Two equally-prominent floors: a deterministic **honesty floor** (`hard_gates_pass` = `vault_contradiction` + `fabricated_specific` + `structural` + `ood_doc_attribution` + `advisory_safety`, all 0) and a separate **anti-vagueness usefulness floor** (`usefulness_floor_pass` — a parrot-vague answer that asserts nothing passes every honesty gate but fails here). The load-bearing decisions: **all hard gates are DETERMINISTIC** (the LLM verifier judge is REPORTED only — judge == answerer is circular, the verify-numeric-backstop failure); the fabrication gate is **value-level** (form-invariant via `coerce_number`, not a surface-form-defeated string blocklist); the surface is **multi-run** (temp 0.6 → N=3/N=5-gated); the judge is a model-parameterized **verifier** (`judge_model` accepts the 8B kill-switch for a non-circular cross-check) with an enforced planted-control health-check. The *structural* invariants stay pinned by `tests/integration/test_expert_agent.py`; `eval-expert` adds the honesty/usefulness corpus. Analytical *correctness* remains out of scope except the human-curated `must_not_recommend` gate.

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

- ~~Implementation lands → move Status to **Accepted**, record the realized contract boundary + the eval discipline, and add the VISION carve-out.~~ **DONE 2026-06-01** — see [§Realized v1](#realized-v1-2026-06-01). The VISION carve-out (the grounding/no-hallucination principle now has an explicit *off-the-gated-path, labelled-ungrounded* exception) is noted on the surface; the qualitative eval discipline remains a documented follow-up.
- ~~A **qualitative analytical-quality eval** is designed → wire it as `eval-expert`.~~ **DONE 2026-06-01** — `eval-expert` shipped as an honesty + usefulness tripwire (spec [`expert-eval.md`](../specs/expert-eval.md)). The remaining lever: promote the judged faithfulness dimensions from *reported* to *gating* once a non-circular cross-model judge (the 8B today via `--judge-model`, or the reserved MCP flagship judge) is validated, behind its own governance.
- A reasoning model with a clean 12 GB-fitting build appears (unblocks the security variant). **(Partially fired 2026-06-01 — the now-default Qwen3.5-4B is a 12 GB-fitting hybrid-reasoning model usable as the general-purpose expert model with `enable_thinking` ON; the security-specialised variant is still gated on a self-quantize. See ADR-0015.)**
- The relaxed-grounding boundary proves too loose (a user mistakes expert output for grounded) → tighten the labelling / separation.

## References

- **Spec:** [`grounded-agentic-chat.md`](../specs/grounded-agentic-chat.md) — the implementation design; this ADR's ungrounded expert mode is its **Surface B** (fenced), sibling to the grounded multi-turn chat (Surface A) that is the primary build.
- [ADR-0008](0008-document-summarization.md) — the grounded summarizer (the contract this inverts) + the swap-in seam it reuses; [ADR-0007](0007-co-residence-resource-modes.md) — co-residence modes; [ADR-0001](0001-vllm-as-sole-inference-engine.md) — vLLM serve
- [[reasoning-expert-mode-scope-2026-05-29]], [[cisco-security-llm-scope-2026-05-29]] — the Foundation-Sec-8B-Reasoning candidate + its 12 GB self-quantize prerequisite
