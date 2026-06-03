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

## Residual (documented follow-up)

The guard masks the symptom **for the bridge presentation surface only**. The
root cause is the shared gate: `verify_grounding/v2` should require *behaviour/
predicate* support, not bare entity-name presence. That fix touches a HARD-gate
node (`/ask` + the summarizer) and is **deferred to its own session** — it needs
a full answer-eval counterfactual re-baseline (a stricter gate risks over-refusing;
measure multi-run, the borderline-counterfactual discipline). The presentation
guard makes the follow-up NON-urgent. Tracked in `next_priorities.md`.

## Reproducibility

Conversational live audit (no scripted harness): drive the present-as-answer
escalation in the web UI on an access-control comparison question, then re-verify
each presented claim against its cited chunk independently. Non-deterministic at
temp 0.6 — the over-grounding appears on a fraction of runs; the guard's effect
(held-back / fallback) is deterministic given the grounded set.
