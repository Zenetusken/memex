# Bridge present-as-answer over-grounding: the name-only guard (2026-06-02)

Settles, against a LIVE re-verification (not a hypothesis), whether the
reason-then-ground bridge's **present-as-answer** surface can show a claim that
is "verified" only because its cited chunk contains the entity NAME — not because
the chunk substantively supports the claim. **Verdict: yes, intermittently — the
shared `verify_grounding/v2` gate over-grounds on entity-name presence; shipped a
deterministic, presentation-only `name-only guard` (`3e3d0ba`) as the backstop.**
Rig: RTX 4070, 12 GB; orchestrator `cyankiwi/Qwen3.5-4B-AWQ-4bit`.

## The finding

A live UI audit of the present-as-answer escalation (`/ask` refusal → "Reason &
verify this →") ran an **independent per-claim re-verification** (an 8-agent
fan-out, each agent given one presented claim + its full cited chunk, asked
*strictly* whether the chunk's CONTENT establishes the claim). On one presented
answer ("security trade-offs of the access-control approaches"), the result was:

| verdict | count |
|---|---|
| **supported** | 3 |
| **unsupported** | **5** (4 "name-only", 1 over-reach) |

**Only 3 of 8 presented claims were substantively supported.** Four of the five
failures were the **name-only** pattern: claims about *how* RBAC/ABAC behave,
cited to a CR350 slide that merely **lists** the type names —

```
### Contrôle d'accès
- Role-Based Access Control (RBAC)
- Attribute-Based Access Control (ABAC)
- Mandatory Access Control (MAC)
- ...
```

The `verify_grounding/v2` gate passed them because the claim's entity name is
present in the chunk (its "literal reading / structural adjacency is sufficient"
rule), even though no sentence in the chunk establishes the asserted behaviour.

**It is a property of the SHARED gate** (`/ask`, the summarizer, and the bridge
all use `verify_grounding/v2`), but **present-as-answer amplifies the harm**: it
frames these as a **VERIFIED** direct answer, the reason-first path feeds the gate
claims phrased around entity names (the exact trigger), and it does so for a
question `/ask` *refused*. The over-grounding is **intermittent / phrasing-driven**
— the same gate correctly grounded **0/0** in most probes (clean out-of-vault, a
named-but-undescribed entity, a one-line terse chunk), and the responsiveness gate
(`assess_relevance`) is a partial mitigation (it falls back when the grounded
subset is off-topic) but does NOT catch name-match over-grounding when the claims
are on-topic.

## The fix (presentation-only)

`core/text.is_name_only_chunk(text)` — a deterministic detector: True when a chunk
is a bare list/heading (≥2 short non-heading lines, no ≥8-word line, no markdown
table / `[chart-extracted]` / `[table-rows]` block; reuses the existing heading /
table / image regexes). Before presenting, `agents/bridge.py::reason_then_ground`
**holds back** any grounded claim cited to a name-only chunk into a NEW additive
field `BridgeAnswer.presented_claims` (the presented body); `grounded_claims` stays
the FULL gate output (the labelled-analysis fallback + the footer counts are
unchanged). The responsiveness gate runs on the surviving `presentable` subset
(guarded on it, so an all-filtered escalation skips the gate and **falls back to
the labelled analysis** — no presented answer). A held-back note surfaces the
delta. Kill-switch `agents.bridge_name_only_guard_enabled` (default ON, fail-open).
+16 tests (1380 total). Live-validated on the 4B: presented+held-back (a mix),
held-back→fallback (all name-only), and a good prose-cited case presents 8/8
unchanged (no false-positive).

## Scope / HARD-gate safety

**PRESENTATION-ONLY.** `ground_claims`, the standalone bridge's grounded subset,
the footer counts, and the `/ask` `answer_query` graph are **untouched** (the
bridge is already proven unreachable from `answer_query`/`run_eval` by
`test_bridge_isolated_from_ask_graph`, still green). No new ADR — an **ADR-0016
Amendment**; spec `docs/specs/grounded-agentic-chat.md` §11 + `src/memex/CLAUDE.md`
updated.

## Residual → the `/ask` root-cause backstop (FIXED 2026-06-03)

The presentation guard masks the symptom **for the bridge surface only**; the root
cause is the shared gate over-grounding on entity-name presence on EVERY consumer.
That `/ask`-path fix was deferred to its own session — **now done** (`agents/answering.py::verify`,
the deterministic NAME-ONLY GROUNDING BACKSTOP). It does **not** take the stricter-prompt
route this residual originally anticipated (a `verify_grounding/v3` "require predicate
support" rule) — that risks over-refusing and is re-interpreted by the 4B with the eval
as its sole net (the same trap as the numeric backstop's reverted prompt-floor). Instead
it mirrors the numeric backstop: a **5th demotion filter** in the `verify` node demotes a
still-grounded claim ONLY when `is_name_only_chunk(chunk.text)` AND
`core/text.claim_asserts_behavior(claim)` — a curated **bilingual (EN+FR)** behavioral/
property/comparative marker set that keys on the PREDICATE CLASS and excludes membership/
existence (`is one of` / `est un de` / `includes`, checked first). **Fail-OPEN + demotion-only
⇒ it can never manufacture a NEW over-refusal** (an unknown/membership predicate is KEPT; a
zero-grounded result routes to the existing `refuse`), so it sidesteps the "a stricter gate
risks over-refusing" concern this residual flagged — no counterfactual re-baseline needed by
construction, and the staged eval (cr350-multidoc / nist-zero-trust / annual-report, backstop
ON) confirmed `refusal_cf=1.0` with zero over-refusal. The name-only chunk test EXCLUDES
table / `[table-rows]` / `[chart-extracted]` blocks, so Table-RAG and the numeric backstop are
byte-disjoint; `eval-summary` stayed byte-stable (the summarizer's `ground_claims` is untouched).
Kill-switch `MEMEX_AGENTS__NAME_ONLY_GROUNDING_BACKSTOP_ENABLED=false`. Pinned by
`tests/unit/test_name_only_grounding.py` + `test_answering_with_fakes.py::test_name_only_backstop_*`.
See the verify-backstop bullet in `src/memex/CLAUDE.md`.

**Now-open follow-up (smaller):** consolidate the shared leniency into one place —
`agents/grounding.py::ground_claims` (so the summarizer + standalone bridge inherit the
membership-aware demotion too, not just `/ask`) AND narrow this audit's bridge presentation
GUARD to be membership-aware (it currently holds back ALL name-only-cited claims, including
membership ones a name-list legitimately supports). Both are HARD-gate-neutral refactors;
tracked in `next_priorities.md`.

### 2026-06-03 follow-up audit — the residual root cause is `verify_grounding/v2` BATCH LENIENCY

A live UI audit of the present-as-answer escalation (the RBAC-vs-ABAC question, the same
trigger) reproduced the over-presentation: 5 behavioral RBAC claims presented, cited to a
French access-control slide. Probes settled the root cause — it is **NOT cross-lingual** and
**NOT `is_name_only_chunk` alone**:

- **Cross-lingual grounding is CLEAN / EN==FR symmetric.** An isolated `ground_claims` probe
  (real FR chunks; EN+FR claims; N=3) grounds the SUPPORTED claim 3/3, rejects the COUNTERFACTUAL
  0/3, rejects the BEHAVIORAL over-reach 0/3, grounds MEMBERSHIP 3/3 — **identical in EN and FR**.
  An EN claim grounds/refuses against a FR chunk exactly like a FR claim.
- **The over-grounding is a BATCH-LENIENCY effect in the gate.** The SAME 5 behavioral claims
  against the SAME chunk: **batched in one verify call → grounded 4/5 (4,4,4); each isolated →
  grounded 0/5.** `verify_grounding/v2` rubber-stamps plausible model-knowledge behavioral claims
  when shown a coherent BATCH, but scrutinizes-and-rejects each alone. The bridge amplifies it
  (free reasoning → many coherent claims → batch-grounded).
- **It is BOUNDED — HARD-gate-safe.** A clear COUNTERFACTUAL batched among 4 true claims did NOT
  leak (CF rejected 0/3; the 4 true grounded). So the batch effect over-grounds plausible-but-
  unstated *behavioral* claims (a citation-precision issue on the fenced bridge surface), never a
  clear falsehood — which is why `refusal_cf=1.0` holds and the `/ask` backstop was safe to ship.
- **Secondary — `is_name_only_chunk` 8-word-line gap.** The FR slide's misdetected numbered
  sub-heading "3. Protection du plan de contrôle (Control Plane)" = exactly 8 words defeats the
  "no ≥8-word line" rule (`_NAME_ONLY_MIN_SENTENCE_WORDS=8`), so the chunk reads as not-name-only
  and neither the bridge guard nor the `/ask` backstop fires on it.

**Implications for the follow-up (its own session, needs an /ask counterfactual eval re-baseline):**
the deterministic name-only backstop is the RIGHT pattern (batch-IMMUNE) but its trigger has the
8-word gap and its marker set is fail-open. Levers: (1) consolidate the backstop into `ground_claims`
(batch-immune for the bridge/summarizer); (2) harden `is_name_only_chunk` to treat "N. Title"
numbered lines as inert headings; (3) consider single-claim (isolated) re-verification on the bridge
present path to defeat batch leniency directly. Full record: `next_priorities.md` +
the `ui-audit-xling-batch-leniency-2026-06-03` memory.

A **separate** audit follow-up surfaced alongside this one — the `extract_claims@v1`
extractor *under-covering* the discrete groundable sub-claims (it pulls the
un-groundable conclusion and mis-cites it; an instrumented trace grounded 0/1
despite a verbatim-groundable NIST quote being present, while a control claim taken
verbatim from the chunk grounded 1/1, proving the gate itself is sound). It was
**mitigated, presentation-only,** by the evidence-consulted UI surface (`3d1a127`):
a navigable "Retrieved from your vault" section that lists the reranked chunks the
model was shown (`BridgeAnswer.evidence`) on both bridge surfaces — most valuably on
the zero-grounded fallback — so the user can open the docs the model read regardless
of grounding/extraction coverage. The root-cause extractor improvement was then
**FIXED** (`extract_claims/v2`, 2026-06-02 — comprehensive-but-faithful extraction
designed via a generate→judge→synthesize workflow; isolated A/B grounded **1→8** on
the failure case with no fabrication; kill-switch `MEMEX_PROMPTS__PIN__EXTRACT_CLAIMS=v1`;
ADR-0016 §Amendment), so the evidence surface is now a *complementary* navigation aid
rather than the sole mitigation.

## Reproducibility

Conversational live audit (no scripted harness): drive the present-as-answer
escalation in the web UI on an access-control comparison question, then re-verify
each presented claim against its cited chunk independently. Non-deterministic at
temp 0.6 — the over-grounding appears on a fraction of runs; the guard's effect
(held-back / fallback) is deterministic given the grounded set.
