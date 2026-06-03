# ADR-0016: The Reason-Then-Ground Bridge (Joins the Ungrounded Expert Surface to the Grounded Gate)

- **Status**: Accepted (v1 shipped 2026-06-02)
- **Date**: 2026-06-02
- **Deciders**: Memex core team
- **Tags**: agents, reasoning, grounding, ux, architecture, contract

## Context

ADR-0013 gave Memex two answer contracts at opposite ends of a **trust ↔ reach** trade-off. **Surface A** (grounded `/ask` + chat) verifies every shipped claim against a retrieved chunk and **refuses** what the vault can't support (`refusal_cf=1.0`, 0 hallucinations) — maximal trust, bounded reach. **Surface B** (ungrounded expert mode) reasons from the model's own knowledge over retrieved evidence, labelled "model knowledge, not your vault," and **never refuses** — maximal reach, no trust floor.

A large class of questions wants **both**: the *reach* of reasoning **and** the *trust* of grounding, on the same question. Expert mode produces a fluent analysis, but the small default 4B (ADR-0015) is confidently wrong on specifics often enough that "treat facts as claims to verify" is a real caveat (live-demonstrated on a Kerberoasting answer that confidently conflated two attacks). `/ask` refuses rather than reason. Neither surface serves *"reason about this, but tell me which parts are actually backed by my vault."*

The question this ADR settles: **how do we combine reasoning with grounding WITHOUT relaxing the grounded gate that makes Memex trustworthy?**

## Decision Drivers

- **Do NOT weaken the grounded gate.** Anything presented as *cited* must pass the IDENTICAL `verify_grounding/v2` check `/ask` uses — same prompt, same schema, same evidence pool.
- **Honesty of provenance.** The reasoned analysis stays labelled ungrounded; only the verified subset is shown as cited; the two must be unmistakable (ADR-0013 R3).
- **Reuse the UNCHANGED machinery.** Table-RAG (a synthetic chunk fed through the existing cite path, ADR-0014) and the summarizer (free MAP → per-point `verify_grounding`, ADR-0008) are the precedents; no new grounding path.
- **HARD-gate neutrality.** The `/ask` `answer_query` graph must be byte-untouched — the bridge is a sibling, never a branch in it.
- **Consent + separation.** Combining the surfaces must never let a user mistake a reasoned analysis for a grounded answer, and must never silently leave the grounded contract on their behalf.

## Considered Options

1. **Relax `/ask` to "reason, then present"** — rejected: contaminates the trusted path (the exact thing ADR-0013 forbids).
2. **A flag on expert mode (`--ground`)** — viable; reuses all expert plumbing but is a less visually-distinct surface.
3. **A DEDICATED surface that reasons, then grounds each reasoned claim through the UNCHANGED gate** (chosen).

## Decision

We chose **Option 3**: a dedicated **reason-then-ground bridge** — `memex bridge` + the `/bridge` "Analysis" tab (CLI + WebUI, **NOT MCP**; gated behind `agents.expert_mode_enabled`, the same fence as expert mode). The pipeline:

1. **Stage 1 — reason (ungrounded):** retrieve → rerank → one free-text reasoning pass over the evidence (reuses the shared `agents/expert.py::reason_over_evidence` core, extracted from `expert_answer`).
2. **Stage 1.5 — extract:** a structured `complete_structured` call decomposes the free-text analysis into discrete `CitedClaim`s (`prompt_tag="extract_claims@v2"` — see the Amendment on the v2 under-coverage fix).
3. **Stage 2 — ground (deterministic):** `repair_claim_chunk_ids` → a deterministic drop of any claim whose id doesn't resolve to a reranked chunk → the **UNCHANGED** `verify_grounding/v2` gate, via a NEW shared `agents/grounding.py::ground_claims` (hoisted verbatim from the summarizer's per-point grounding, so the summarizer + bridge share ONE primitive and the `/ask` `verify` node is not touched).

Only the survivors are presented as **cited** (`/ask`-grade by construction); everything else stays inside the labelled ungrounded analysis. Output is a `BridgeAnswer` (analysis + grounded-claims subset + sources + an ungrounded provenance banner).

We also add the **consented A→B escalation**: from a Surface-A `/ask` **refusal**, the user is offered an **explicit, never-automatic** path to re-run the same question through the bridge over the **same scope** (a webui "Reason over this instead →" affordance / a CLI stderr hint naming `memex bridge`).

Three load-bearing v1 contract decisions:

- **The STANDALONE bridge is verify-only, NOT `assess_relevance`.** `assess_relevance` judges whole-answer *responsiveness*; the standalone bridge's grounded subset is "which reasoned claims are vault-supported," not "a direct answer," so running it would over-refuse a legitimately-grounded set of supporting claims. (Reserved for the variant where the bridge output IS presented as the direct answer — now realized in the present-as-answer escalation; see the Amendment below.)
- **A two-layer faithfulness guard.** `verify_grounding/v2` is structurally blind to "the analysis never actually made this claim" — a fabricated-but-coincidentally-grounded claim could pass. So the extractor prompt is an *extractor, not a generator* ("emit a claim only if the analysis explicitly asserts it AND it cites a listed chunk"), backed by a **deterministic** drop of any claim whose `source_chunk_id` doesn't resolve to a reranked chunk.
- **Zero-grounded ≠ refuse.** The analysis is useful on its own; a zero-grounded run returns the labelled analysis with an empty grounded subset (no citation chrome) — the bridge has no refuse state.

## Consequences

### Positive

- Delivers reasoning *reach* with a real *trust floor* without touching `/ask`: the grounded subset passed the same gate as a normal answer.
- No new grounding path — `verify_grounding/v2` is reused verbatim; the summarizer + bridge now share one `ground_claims` primitive; the `/ask` graph is byte-identical (pinned by a structural isolation test).
- The risky half (a small model's imprecise synthesis) is precisely the half the gate refuses to vouch for — live-demonstrated: an OSPF analysis grounded 8/8 verifiable claims whole-vault, 0/0 when scoped to a doc that excludes OSPF.

### Negative / Trade-offs

- A **third** answer surface to keep visually + structurally distinct (mitigated: it reuses expert's ungrounded banner + the `/ask` claim chrome; amber, never the grounded-blue).
- The grounded output is **capped at 8 claims** (it reuses `/ask`'s `DraftAnswer` schema, whose bounds are xgrammar-JSON-close-calibrated) — a richer analysis can surface at most 8 grounded claims in v1.
- **No numeric-aggregate backstop in `ground_claims`** (that deterministic demotion lives in the `/ask` `verify` *node*, not the shared helper) → bare computed-table-figure claims are out of v1 scope; the extractor is told to avoid them (the summarizer carries the same gap).
- The ungrounded *analysis* half still inherits the small-model imprecision risk — bounded only by the labelling, not the gate.

### Neutral

- `/ask`, chat, summarize, expert, and their HARD gates are untouched; the change is additive (`agents/grounding.py`, `agents/bridge.py`, one prompt, two surfaces, a `question` field carried on the webui progress entry).
- MCP is deliberately not a surface (reserved for the upstream flagship-fallback layer, as with chat + expert).

## Alternatives in Detail

### A flag on expert mode (`--ground`)

Reuses all expert plumbing with the least new surface area, and the output is identical. Rejected for v1 because the user chose a dedicated surface: a separate command + tab keeps "reason-then-ground" conceptually distinct from plain ungrounded reasoning, and the dedicated nav makes the trust contract legible. The flag remains a trivial future addition if wanted.

### Include `assess_relevance` (as spec §11 originally drafted)

§11 was written for the escalation variant where the bridge output IS the answer to a Surface-A question — there a whole-answer responsiveness gate is appropriate. For the standalone bridge, the grounded subset is a set of *supporting* claims, not "the answer," so `assess_relevance` frequently returns non-responsive and would refuse a perfectly-grounded subset. So `assess_relevance` is **off the standalone path** but **on the present-as-answer escalation** (the Amendment below) — exactly the split this alternative anticipated.

### Automatic A→B escalation on a refusal

Rejected — it violates ADR-0013 R3 (the user must *choose* B; a silent hand-off from a grounded question to ungrounded reasoning is exactly the mistaken-for-grounded failure). The escalation is consented (an explicit click / a typed command), never automatic.

### Widen the extraction schema beyond 8 claims

Rejected — re-opens the xgrammar force-close-mid-emission trap the bounded schemas exist to prevent. The correct way to lift the cap is a MAP loop over the analysis (the summarizer's per-section idiom), not a wider single schema.

## Revisit When

- A genuinely rich analysis regularly **saturates the 8-claim cap** → add MAP-loop extraction (accumulate claims across windowed passes).
- ~~The **consented-escalation-as-presented-answer** variant is built (bridge output offered as the direct answer to a Surface-A question) → add `assess_relevance` there, behind its own flag + governance.~~ **DONE 2026-06-02 — see the Amendment below.**
- A **numeric-heavy** bridge use emerges (computed-aggregate claims) → wire the `/ask` numeric-grounding backstop into `ground_claims` (it would then also harden the summarizer).
- The labelling proves too subtle (a user reads the analysis half as grounded) → tighten the separation (mirrors ADR-0013's R3 trigger).

## Amendment (2026-06-02): the present-as-answer escalation

The consented A→B escalation, which originally re-ran the question through the **standalone** bridge verbatim (verify-only, the labelled-analysis surface), now **advances to "the bridge output IS the answer"** — the variant this ADR's Decision and Revisit-When deferred. This is an evolution of the consent contract, recorded here rather than as a new ADR because it changes no architecture (no new grounding path, no graph change, `answer_query` still never imported).

**What changed.** `reason_then_ground` gains `present_as_answer: bool = False`. When the consented escalation sets it (a hidden `present_as_answer=true` form field — the ONLY discriminator between the two `POST /bridge` callers; the standalone composer omits it) AND the grounded subset is non-empty, the bridge ALSO runs the responsiveness gate `agents/grounding.py::assess_responsiveness`. When the subset is non-empty **AND responsive** (`BridgeAnswer.presented`), the grounded claims are **presented AS a direct grounded answer** (a distinct "Reasoned, then grounded" surface — the claims are the answer, cited, with the ungrounded reasoning fenced in a collapsed `<details>`), meeting the SAME verify + `assess_relevance` bar as a grounded `/ask` answer. Otherwise it falls back to the labelled-analysis surface (no-refuse-state preserved; a non-responsive subset gets a quiet "related question" note). CLI parity: `memex bridge --answer` + the refusal hint names it.

**Why it still honors the decision drivers.**

- **Gate not weakened.** Every presented claim already passed the UNCHANGED `verify_grounding/v2` + the deterministic id-resolution drop; the responsiveness gate is the UNCHANGED `assess_relevance@v1` prompt + `RelevanceAssessment` schema. A presented answer clears a *strictly higher* bar than the standalone bridge (verify + responsiveness vs verify alone).
- **No ungrounded text in the answer.** The presented body is the grounded `CitedClaim`s only; the `assess_responsiveness` input is a **deterministic join** of the grounded claim texts — never the ungrounded `analysis`, never the extractor's free `summary`. The analysis appears only inside the fenced `<details>`.
- **HARD-gate neutrality unchanged.** The `/ask` graph is still byte-untouched; we do **not** refactor the `/ask` `assess_relevance` node to share the helper (that would create an `answering↔grounding` import cycle AND edit a HARD-gate node) — the single source of truth is the prompt + schema, with a small call wrapper in `grounding.py` (the same accepted pattern as `bounded_verification`). `present_as_answer` defaults False, so the standalone path + the existing bridge tests are byte-identical. Fail-CLOSED: a gate `ModelCallError` → not presented (falls back), never an un-gated answer.
- **R3 (consent + separation) holds.** Still consented (an explicit click / a typed command), never automatic. The presented surface is a distinct third label — neither the plain `/ask` "Answer" eyebrow nor the bare "ungrounded" banner — so it is read as exactly what it is: a grounded answer reached via reasoning.

### Amendment (2026-06-02, follow-on): `extract_claims/v2` — fixing Stage-1.5 under-coverage

An instrumented audit of a zero-grounded bridge fallback isolated the cause to **Stage 1.5, NOT the grounding gate**: on a 5,026-char analysis the v1 extractor pulled exactly **one** claim — the un-groundable synthesis thesis ("ABAC is superior"), mis-cited — so the gate (correctly) grounded 0. A control claim taken **verbatim** from a retrieved chunk grounded **1/1**, proving the `verify_grounding/v2` gate + the id-resolution machinery are sound; the weakness was upstream extraction *coverage*. Root cause: the v1 prompt's *"It is correct to return FEW or ZERO claims. Quality over coverage"* line, plus the model collapsing a thesis-essay to its conclusion.

**`extract_claims/v2`** (designed via a generate→judge→synthesize workflow) flips the bias to **comprehensive-but-faithful**: a foregrounded "PRIMARY TARGET" block makes the facts the analysis *attributes to / quotes from* a document the main extraction target (cited to that document), an explicit **DECOMPOSE compound sentences into atomic claims** rule, and an explicit demotion of the synthesis thesis (emit it only if a chunk directly supports it). Every faithfulness rule of the two-layer guard is **preserved verbatim** (extractor-not-generator, EXPLICITLY-asserted-only, VERBATIM `source_chunk_id`, no computed/aggregate figures) and citation accuracy is sharpened (cite the chunk whose CONTENT supports the claim, not one that merely NAMES the entity — mirroring the name-only guard). **No few-shot example** (the key-figures lesson: the 4B copies concrete example content verbatim). This is safe because over-extraction is *contracted downstream* (the deterministic id-drop + the unchanged verify gate + the name-only guard); v2 only stops the extractor from *under*-emitting the genuinely-groundable facts.

**A/B (isolated — same fixed analysis per question, v1 vs v2, N=3):** grounded **1→8** (the failure class), **5→6**, **7→8** across three questions, with `groundable == grounded` (no fabrication — every extra claim was genuinely vault-supported). The loader auto-selects the highest version (v2 is now active); kill-switch `MEMEX_PROMPTS__PIN__EXTRACT_CLAIMS=v1` (zero derived state). Bridge-only / off the HARD-gate path; the bridge mechanics tests are prompt-agnostic and unchanged. The audit's *other* follow-up (the `verify_grounding/v2` name-presence leniency) was independent and is now **FIXED on the HARD-gate path** (2026-06-03) — a deterministic, fail-open, membership-aware NAME-ONLY GROUNDING BACKSTOP in the `/ask` `verify` node (the numeric backstop's sibling; `core/text.claim_asserts_behavior` + `is_name_only_chunk`), NOT a stricter prompt; `refusal_cf=1.0` held + `eval-summary` byte-stable. See `docs/audits/11-bridge-name-only-overgrounding.md` (the Residual → root-cause backstop section) + the verify-backstop bullet in `src/memex/CLAUDE.md`.

### Amendment (2026-06-02, follow-on): the name-only presentation guard

A live UI audit of the present-as-answer escalation (an independent per-claim re-verification fan-out) found that the shared `verify_grounding/v2` gate **intermittently over-grounds**: it can pass a claim whose cited chunk merely **names** the entity — e.g. RBAC/ABAC behavior claims cited to a slide that only *lists* "Role-Based Access Control (RBAC)", "Attribute-Based Access Control (ABAC)", … with no descriptive sentence (3/8 substantively supported in one audited *presented* answer). This is a property of the shared gate (also used by `/ask` + the summarizer), but **present-as-answer amplifies it** by framing those claims as a VERIFIED direct answer.

**Fix (the cheapest, bridge-local option):** a deterministic **presentation-only** guard. Before presenting, `reason_then_ground` filters the grounded subset through `core/text.is_name_only_chunk(chunk.text)` — True when the cited chunk is a bare list/heading (≥2 short non-heading lines, no ≥8-word line, no table/chart block). Held-back claims are kept OUT of `BridgeAnswer.presented_claims` (the presented body) but remain in `grounded_claims` (the labelled fallback + footer counts are unchanged); a held-back note surfaces the delta. The responsiveness gate runs on the *presentable* subset (guarded on it, so an all-filtered escalation skips the gate and falls back to the labelled analysis — no presented answer). **Zero `/ask`/summarizer/`ground_claims` impact** (the guard never touches the gate, only what the bridge *presents*); kill-switch `agents.bridge_name_only_guard_enabled` (default on, fail-open). The detector deliberately diverges from `_looks_like_prose_heading`'s terminal-punctuation rule (slide bullets lack periods; a real descriptive line without a period must still count as substantive). It is a bridge-LOCAL safeguard, distinct from the HARD-gate fix: the underlying name-presence leniency was SUBSEQUENTLY closed on the `/ask` `verify` node (the deterministic NAME-ONLY GROUNDING BACKSTOP, 2026-06-03 — see the paragraph above + `docs/audits/11`). A live 2026-06-03 UI audit then traced the RESIDUAL bridge over-presentation to a `verify_grounding/v2` **BATCH-LENIENCY** effect (the same behavioral claims ground 4/5 BATCHED vs 0/5 ISOLATED) — language-independent (cross-lingual grounding is EN==FR symmetric) and BOUNDED (a clear counterfactual batched among true claims does NOT leak → `refusal_cf`-safe), tracked as a follow-up in `docs/audits/11` + `next_priorities.md`.

## References

- **Spec:** [`grounded-agentic-chat.md`](../specs/grounded-agentic-chat.md) §11 — the implementation design (the bridge + the consented + present-as-answer escalation).
- [ADR-0013](0013-ungrounded-reasoning-expert-mode.md) — the ungrounded expert surface (Surface B) this bridges to the grounded gate; its R3 (mistaken-for-grounded) guard rail governs the escalation's consent + separation.
- [ADR-0008](0008-document-summarization.md) — the grounded summarizer whose per-point `verify_grounding` (`_ground_points`) was hoisted into the shared `ground_claims` primitive.
- [ADR-0014](0014-text-to-sql-robustness-safety.md) / Table-RAG — the synthetic-chunk → unchanged-cite-machinery precedent (the same "no new grounding path" discipline).
- [ADR-0015](0015-qwen35-4b-unified-orchestrator.md) — the unified 4B whose imprecision-on-specifics motivates a trust floor over its reasoning.
