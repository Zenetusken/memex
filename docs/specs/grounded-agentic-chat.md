# Grounded Multi-Turn Agentic Chat — and the Ungrounded Expert Surface (fenced)

- **Status**: **Surface A v1 SHIPPED + audit-hardened + fully live-validated 2026-06-01 (`9a6b46e`→`009ae92`)** — grounded multi-turn chat on **CLI (`memex chat`) + WebUI (`/chat`)** (NOT MCP — reserved for the upstream flagship-fallback layer). A 3-agent adversarial audit confirmed the 4 HARD grounding invariants hold BY CONSTRUCTION; the two real findings were fixed (the digest-failure running-summary bloat; a resume blank-bubble guard). Live-validated on the rig: in-chat refusal, digest compaction (5-turn), the scope picker (renders+scopes+persists), and **multi-turn determinism (a 3-turn conversation is bit-identical across runs)** + the `eval-chat` baseline `mean_recall=1.0`. Full suite green. **Surface B (the ungrounded expert mode) v1 is now ALSO SHIPPED + live-validated 2026-06-01** — `memex expert` + webui `/expert`, off by default (`agents.expert_mode_enabled`), **ADR-0013 → Accepted**; see the §10 SHIPPED note for the deltas from the design below. Full suite 1293 green.
- **Date**: 2026-06-01
- **Primary surface (the spine): Surface A — a grounded, multi-turn, conversational version of `/ask`** that keeps the no-hallucination HARD gate intact on every turn. Reasoning is confined to a control-layer query-rewrite; every *answer* turn stays guided-JSON + verified.
- **Secondary surface (fenced, §10–§11): Surface B — the ungrounded ADR-0013 "expert/analysis" mode** that *inverts* the grounding contract (model knowledge, not your vault), plus the reason-then-ground bridge that connects A↔B. Recorded here because the two surfaces share the chat chrome, the conversation store, and the long-poll plumbing — but they must **never blur**.

> **The two-surface thesis.** A trades *reach* for *trust*; B trades *trust* for *reach*. Surface A is the spine because it is what the project's identity is — *trustworthy, sourced answers over your own corpus* — made conversational. Surface B is a deliberate, governed departure from that contract, fenced off so an un-sourced B-style claim can never pass as a grounded A answer. **Build A first; keep B optional, labeled, and reachable only by explicit user choice.**

---

## 1. TL;DR

- **Surface A is grounding-safe across turns *by construction*.** The verifier (`src/memex/agents/answering.py:1667`) builds `chunk_by_id = {c.chunk_id: c for c in state.reranked}` — the *current* turn's reranked chunks only. It has **zero access** to conversation history or prior-turn answers. So no matter how we add memory, a claim can never "ground" against a previous turn. Multi-turn chat adds memory + persistence + rendering on top of the unchanged grounded graph; it cannot create a cross-turn grounding leak.
- **The balanced follow-up design** (your chosen criteria — precision + context/VRAM efficiency + multi-turn relevance): **rewrite-then-retrieve-fresh + a bounded prior-chunk carry.** Each turn (a) rewrites the user's message into a standalone query using a *compact* history (a small **structured**, hence deterministic, call), (b) retrieves **fresh** on that query (precision), (c) re-includes only the immediately-prior answered turn's cited chunks (≤5) as extra rerank candidates (relevance without unbounded accumulation), then (d) runs the **unchanged** answer→verify gate. History lives only in the rewrite call's prompt, never co-occupying the answer window (VRAM-frugal).
- **Minimal, default-off change to the core.** The only edit to `answer_query` is one optional param `prior_carry_chunk_ids: list[str] | None = None`; every existing caller (`/ask`, CLI, MCP, summarize) is byte-unchanged. The carry merges at the *same* candidate-pool seam `expand_graph` already uses, so a carried chunk only reaches grounding if it survives rerank against the rewritten query — no staleness logic needed, and `verify` is untouched.
- **Conversation state persists** in a sqlite sidecar (`vault/.memex/conversations.sqlite`, WAL-tuned), treated as **user data** (preserved across `reindex --force`, like `scope_sets.json`). It stores chunk **ids** (never chunk text) + an opaque `FinalResponse` JSON for re-render.
- **The webui chat tab reuses almost everything**: the long-poll `ProgressRegistry`, `_progress.html`, and — the big win — **`_answer.html` as the assistant bubble**, so the refusal UX + the Related-documents "suggestions" panel come for free. Every turn calls the same `answer_query`, so `refusal_cf=1.0` / 0-hallucination apply per turn under the **existing** eval — no new eval discipline for Surface A.
- **Surface B is fenced** (§10): a separate `complete_reasoning` primitive (grammar off, `enable_thinking=true`), a separate agent never reachable from `answer_query`, its own non-refusal eval, and a deterministic "model knowledge, not your vault" provenance label. The reason-then-ground bridge (§11) is the only way to combine reasoning with grounding, and it does so by feeding free-text conclusions back through the *unchanged* grounded gate.

---

## 2. The shared architectural premise (what ADR-0015 shipped)

### 2.1 The unified 4B orchestrator (kept; vision is parse-time only)

The grounded orchestrator is `cyankiwi/Qwen3.5-4B-AWQ-4bit` (compressed-tensors W4A16, a hybrid-reasoning VL model), live since `ecc67af` (2026-06-01, ADR-0015), serving persistently as the vLLM daemon at ~6.3 GB with an **8192-token** window. The full re-baseline held every HARD gate (12/12 corpora N=3, `refusal_cf=1.0`, 0 hallucinations). Two settled decisions bound this spec: the orchestrator **stays** the VL-4B (a text-only swap's live-VRAM win is illusory — vLLM pre-reserves by util fraction), and the parse-time doc-VLM **stays** the dedicated `Qwen3-VL-8B-Instruct`. **Grounding is text-based**: every claim is verified against a retrieved chunk's *text*, and vision is a *parse-time* concern (diagrams are transcribed to text at ingest). Therefore **a grounded chat loop needs no live vision** — fresh mid-conversation pixels are an ingest event through the existing parse pipeline, then grounded as text.

### 2.2 The determinism property: guided-JSON suppresses reasoning by construction

The 4B is hybrid-reasoning — on an *unguided* call it emits a `<think>…</think>` CoT prefix. On the grounded path it never does, for a mechanical reason:

- Every grounded LLM call routes through the single generic entry point `complete_structured(..., schema: type[T])` (`src/memex/models/client.py:219`). There is **no schema-less overload**; pyright `--strict` flags a missing `schema=` at the call site.
- That function **unconditionally** builds `response_format` (`src/memex/models/client.py:327-334`):

  ```python
  response_format={
      "type": "json_schema",
      "json_schema": {
          "name": schema.__name__,
          "schema": _inline_refs(schema.model_json_schema()),
          "strict": True,
      },
  },
  ```

- vLLM routes a `json_schema` `response_format` to **xgrammar**, which compiles the schema into a finite-state grammar and forces token 0 to be `{`. A `<think>` prefix is free text *before* the JSON object — it violates the grammar and is **unreachable**. The CoT is suppressed by the grammar, not a prompt or a sampling knob.
- `_inline_refs` (`client.py:96-164`) resolves `$defs`/`$ref` before the schema reaches vLLM. This is a **HARD-gate dependency**: xgrammar silently downgrades a `$ref`-bearing schema to the weaker Outlines backend, which can permit off-grammar text. Inlining keeps xgrammar on the strictest grammar.

This was validated against the live daemon in the ADR-0015 fit-test (valid JSON-from-token-1, no `<think>` leak, `maxItems` enforced under stress; `docs/adr/0015-qwen35-4b-unified-orchestrator.md:72-73`).

### 2.3 The cross-turn safety invariant (what makes Surface A safe)

> **INVARIANT.** `verify` (`agents/answering.py:1623`+) rebuilds `chunk_by_id` solely from `state.reranked` (`:1667`) — the chunks retrieved and reranked for the **current turn**. It never reads conversation history, prior `FinalResponse` objects, or prior answer text. The grounding check is a strict text-presence test against *that* pool only.

Corollary: a multi-turn surface is grounding-safe **as long as prior-turn *answer text* never enters the grounding pool**. Only real retrieved *chunks* are valid grounding evidence. The Surface A design honors this absolutely (§3–§4): the conversation memory feeds the *rewrite* step (a planner, asserts nothing) and the *prior-chunk carry* re-introduces real chunks (which must re-survive rerank), but the verifier's evidence pool is unchanged.

---

# SURFACE A — Grounded Multi-Turn Agentic Chat (the spine)

## 3. The per-turn grounded loop

New module `src/memex/agents/chat.py`, entrypoint `answer_turn`:

```python
async def answer_turn(
    conversation_id: str,
    user_text: str,
    *,
    scope_doc_ids: list[str] | None = None,
    correlation_id: str | None = None,
    on_node: Callable[[str], None] | None = None,
) -> ChatTurnResult: ...
```

Per-turn algorithm:

1. **Load** the conversation from the store (§6).
2. **Query rewrite** (control layer; only when the conversation has history): one small `complete_structured(schema=StandaloneQuery)` call (§3.2) resolves "that / it / the previous one" from the compact memory (§4) into a self-contained query. If the message is already self-contained → echo it unchanged, `is_followup=false`. **Fail-open** to `user_text` on a `ModelCallError`.
3. **Bounded prior-chunk carry**: `carry_ids = prior_answered_turn.cited_chunk_ids[:_PRIOR_CARRY_MAX]` (`_PRIOR_CARRY_MAX = 5`, **immediately-prior answered turn only**, never accumulated). Empty unless `is_followup`.
4. **Answer** through the unchanged grounded graph: `answer_query(effective_query, scope_doc_ids=…, prior_carry_chunk_ids=carry_ids, correlation_id=…, on_node=…)`.
5. **Persist** the turn (§6) and compact memory if over budget (§4).

```
load convo ─▶ rewrite? ──standalone_query──▶ carry(prior cited≤5) ──▶
  answer_query(retrieve+carry → rerank → assess → answer → verify → compose | refuse)
  ──▶ persist turn ──▶ compact memory if needed
```

### 3.1 Minimal additive change to the answering core (default-off)

Every existing caller stays byte-identical (the param defaults to `None`/empty):

- **`answer_query`** (`answering.py:2251`): **+1 optional param** `prior_carry_chunk_ids: list[str] | None = None`.
- **`AnswerState`** (`answering.py:465`, set at `:2322`): **+1 input field** `prior_carry_chunk_ids: list[str] = []`.
- **`retrieve` node** (`answering.py:559`): after `hybrid_search`, if `state.prior_carry_chunk_ids`, fetch those chunks via `index/fts_store.py:348 chunks_by_ids` (a lazy `FTSStore.open`, the documented `agents → index` lazy-store edge that `expand_graph`/`query_tables`/`resolve_artifact_scope` already use) and **union them into `state.candidates` with dedup by `chunk_id`** — the *same* candidate-merge seam `expand_graph` uses (`answering.py:805`). The reranker then scores the carried chunks against the **rewritten** query alongside fresh hits.

Why this is safe and minimal: a carried chunk only reaches grounding if it **survives rerank against the current rewritten query**; a stale referent that no longer matches is reranked out — no manual staleness logic. `verify`'s `chunk_by_id` (`:1667`) is untouched. Empty carry → byte-identical to today. The one thing to flag in review: `retrieve` gains a guarded lazy `FTSStore.open` (zero cost when the carry is empty).

### 3.2 The query rewrite — control-layer, deterministic

`prompts/rewrite_followup/v1.md` + a bounded schema in `chat.py`:

```python
class StandaloneQuery(BaseModel):
    standalone_query: str = Field(max_length=400)        # the self-contained query
    is_followup: bool                                    # depended on prior turns?
    referents_resolved: list[Annotated[str, Field(max_length=80)]] = Field(
        default_factory=list, max_length=6)              # 'that->X', audit-only
```

It is a **guided-JSON** call → deterministic, reasoning suppressed by construction (§2.2) — *not* a free-text/thinking primitive. Prompt intent: *resolve every pronoun/reference using the conversation summary + recent turns; if already self-contained, return unchanged with `is_followup=false`; do NOT answer, only rewrite; do NOT invent entities absent from history.* A conservative echo-when-nothing-to-resolve default keeps a non-follow-up turn byte-identical to a bare `answer_query`.

**Failure mode is benign by design:** a mis-resolved referent → a grounded-but-non-responsive answer → the **existing relevance gate** refuses (`assess_relevance`, `answering.py:1906`) and the surface shows the resolved `standalone_query` so the user can reformulate. A mis-rewrite degrades to a clean refusal, never a hallucination.

## 4. Conversation memory model

Retained per turn — **chunk ids only, never chunk text, never prior-answer-as-grounding-evidence**:

| Field | Source | Use |
|---|---|---|
| `user_text` | raw message | rewrite input + audit |
| `standalone_query` | rewrite output | what retrieval ran on |
| `answer_summary` | `FinalResponse.summary` (≤600 chars) or the refusal reason | **rewrite context only** — structurally incapable of becoming grounding evidence |
| `cited_chunk_ids` | `[c.chunk_id for c in FinalResponse.used_chunks]` | the bounded-carry seed; re-fetched live via `chunks_by_ids` |
| `answered` | `FinalResponse.answered` | a refused turn carries no carry-able chunks |

**Compaction.** Keep the last **N = 4 turns verbatim** (`user_text` + `answer_summary`, both already short). When older turns exist, fold them into a single `running_summary` (≤ `_RUNNING_SUMMARY_MAX = 1200` chars), regenerated incrementally. Trigger = turn-count primary (>4) OR token-budget secondary (§5). The compaction call mirrors the summarizer's **bounded-list** idiom (`agents/document_summarizer.py:463` `_reduce` + the `DocAbstract`/`max_length`-bounded-list shape) — a `complete_structured(schema=ConversationDigest)` where `ConversationDigest = {sentences: list[Annotated[str, max_length=200]] (max_length=6)}`. It is **not** `summarize_document` (chat history has no chunks to ground against; this is conversation metadata, never an answer claim).

## 5. Context / VRAM budget (the efficiency criterion)

The key insight: **rewrite and answer are two separate `complete_structured` calls with two separate prompt budgets — history never co-occupies the window with the grounding chunks.**

- **Rewrite call** carries the history only: `running_summary` (≤1200 chars ≈ 300 tok) + 4 recent turns (≈ 900 tok) → **`_HISTORY_TOKEN_CAP = 1400 tok`**, output capped at 256 tok. No chunk contention.
- **Answer call** is *unchanged* from single-turn: top-k chunks × `truncate(1800)` (fast/4B: top-5 ≈ 2250 tok; output ≈ 1800 tok). The carry (≤5) **competes for the same top-k slots** — the reranker caps the total at `top_k`, so carry **cannot inflate the answer prompt**.

Truncation order under pressure (all inside the rewrite call): drop `referents_resolved` → truncate `running_summary` → evict oldest recent turns into the running summary → always keep at least the immediately-prior turn verbatim. The history cap is **mode-independent** (same 1400 tok in fast/full, like the summarizer's mode-independence contract); the 8B kill-switch's 6144 window still fits (rewrite ≈ 2000 tok, answer unchanged).

## 6. The sqlite conversation store

New `src/memex/core/conversation_store.py` — a `ConversationStore` mirroring `index/fts_store.py:103`/`index/table_store.py` (autocommit `sqlite3`, `apply_sqlite_pragmas` from `core/sqlite_tuning.py:20` → WAL+NORMAL+cache+mmap, an `asyncio.Lock` for the multi-statement append, `asyncio.to_thread` reads). File: `vault/.memex/conversations.sqlite`. It lives in `core/` (no `agents/` deps).

```sql
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id  TEXT PRIMARY KEY,          -- ULID
    title            TEXT,
    created_at       TEXT NOT NULL,             -- ISO-8601 UTC
    updated_at       TEXT NOT NULL,
    turn_count       INTEGER NOT NULL DEFAULT 0,
    running_summary  TEXT NOT NULL DEFAULT '',  -- compacted history (≤1200 chars)
    scope_doc_ids    TEXT NOT NULL DEFAULT '[]' -- JSON: conversation-level scope pin
);
CREATE TABLE IF NOT EXISTS turns (
    turn_id          TEXT PRIMARY KEY,          -- ULID (chronological)
    conversation_id  TEXT NOT NULL,
    turn_index       INTEGER NOT NULL,
    created_at       TEXT NOT NULL,
    user_text        TEXT NOT NULL,
    standalone_query TEXT NOT NULL,
    is_followup      INTEGER NOT NULL DEFAULT 0,
    answer_summary   TEXT NOT NULL DEFAULT '',
    answered         INTEGER NOT NULL DEFAULT 0,
    cited_chunk_ids  TEXT NOT NULL DEFAULT '[]',-- JSON: chunk_ids, NOT chunk text
    response_json    TEXT,                       -- opaque FinalResponse.model_dump_json() for re-render
    correlation_id   TEXT,                       -- joins the Langfuse trace (ADR-0004)
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);
CREATE INDEX IF NOT EXISTS turns_convo ON turns(conversation_id, turn_index);
```

- **User data, not regenerable derived state** — **exclude it from the `reindex_vault(force=True)` teardown allow-list** (the `scope_sets.json` precedent; a rebuild must not wipe a user's chats). Recommend no automatic TTL by default; provide `delete_conversation(id)` + an optional `cleanup(older_than_days)`.
- **Boundary:** `core/` must not import `agents/FinalResponse`, so the store persists `response_json` **opaque**; the **webui** does `FinalResponse.model_validate_json` to re-render a resumed thread through the existing `_answer.html` context. Stores ids, never text.

## 7. The webui chat tab

Routes mirror the `/ask` (`webui/app.py:618/:652`) and `summarize` (`:931`) patterns:

- `GET /chat` + `GET /chat/{conversation_id}` — landing / resume (full pages extending `base.html`; rehydrate the thread from the store).
- `POST /chat/{conversation_id}/turn` — mirrors `ask` plus **persist-the-user-turn-first** (durable before the task runs): mint `cid`, `progress.new(cid)`, `create_task(_run_chat_turn)`, return the user bubble **and** `_progress.html` into `#conversation-log` via `hx-swap="beforeend"`.
- `GET /chat/{conversation_id}/status?cid=&v=` — mirrors `summarize_status`; on `done`: `progress.evict(cid)` → persist the assistant turn → return the assistant bubble (appended beforeend; the `_progress.html` fragment self-replaces via its own `hx-swap="outerHTML"`, so the bubble lands where "Working…" was). Idempotent — a retried poll hits the swept cid → `_progress_expired.html`, never a double write.
- `POST /chat/{conversation_id}/delete` + a recent-conversations rail (folded into `GET /chat`).

**Reuse (no change):** `progress.py` **verbatim** — `ProgressEntry` already carries `response: FinalResponse | None`; `cid` keys the in-flight request, `conversation_id` is the path param the status handler closes over, `turn_id` is minted at persist time, so **no new `ProgressEntry` fields**. `_progress.html` / `_progress_expired.html`. The biggest reuse: **the assistant bubble IS `_answer.html`** (`{% include "_answer.html" %}` inside a `.chat-assistant` wrapper, the move `_summary.html` already makes) — so the fixed **refusal UX + the Related-documents "suggestions" panel** (`_answer.html:44-84`) come **free**, no new template. Per-conversation scope: render `_scope_picker.html` on turn 1, persist `scope_doc_ids` on the conversation row, show it read-only after, forward it server-side every turn. Add a `Chat` nav link in `base.html`. New: `chat.html`, `_chat_turn.html`; `.chat-*` semantic CSS (zinc + the one action-blue, AA floors), **not** new Tailwind; a transcript-above-a-composer layout (not a floating corner bubble, per the webui aesthetic rules).

## 8. HARD-gate isolation + eval

Every turn flows through the **unchanged** `answer_query` graph (`retrieve → … → verify → assess_relevance → compose | refuse`), so `refusal_cf=1.0` / 0-hallucination apply **per turn**, measured by the **existing** `run_eval` — **no new eval discipline for Surface A**. The surface adds only memory + persistence + rendering + scope (which can only *narrow* retrieval → a stale-scope turn refuses cleanly, the scope-set safety property). Surface A v1 **never imports** a free-text/reasoning primitive and **never sets** `enable_thinking`; `complete_structured`'s strict-JSON grammar (and the §2.2 reasoning suppression) is on every turn. Validation when built: faked-`complete_structured` (rewrite + digest) + faked-`FTSStore` unit tests (carry-reranks-in/out; empty-carry byte-identical); `pyright` 0/0, `ruff`; the existing per-turn HARD gates; a manual multi-turn smoke (a follow-up resolves its referent and grounds; an unsupported follow-up refuses + suggests related docs).

## 9. Phasing (Surface A)

| Phase | Item | Gating |
|---|---|---|
| **A-v1** | `core/conversation_store.py` + schema; `answer_query` additive `prior_carry_chunk_ids` + `retrieve`-merge; `agents/chat.py::answer_turn` + `rewrite_followup`/`conversation_digest` prompts+schemas; the `/chat` routes + templates reusing `_answer.html`/progress. The full balanced loop. | None beyond the additive core edit (default-off). The existing per-turn HARD gates are the bar. |
| **A-1.5** | Query-rewrite tuning — upgrade only if a **measured** follow-up-resolution gap (>~10% of follow-ups unanswerable for missing referents) appears. Optimization defaults to hand-authored few-shot (see **§9.1**). | A measured gap; otherwise the v1 structured rewrite stands. |
| **A-2 (agentic)** | The *agentic* tool-loop: decomposition, multi-retrieval, an on-demand "transcribe-this-page" tool (the proven 8B-VL as a tool → text → grounded). Each tool output still grounds as text. | A-v1 shipped; a workload that single-shot retrieval underserves. |
| **A-2 (UX)** | Token streaming via SSE/`EventSource` (a new transport outside the all-HTTP-200 long-poll model). | Turn-granular shipped; a UX need that justifies the transport. |

### 9.1 Query-rewrite prompt optimization — researched, deferred (no DSPy dependency, no ADR)

A 2026-06-01 research pass evaluated **DSPy** (MIT, Stanford) for optimizing the rewrite prompt. **Verdict: defer; no dependency; no ADR.** (1) Premature by three independent measures — Surface A isn't built, there is **no multi-turn eval corpus** (all 18 current eval sets are single-turn), and the A-1.5 gap is unmeasured. (2) DSPy cannot enter the runtime — its adapter materializes prompts at inference time and static-prompt export is an unsupported workaround (GitHub #8043), so only a strictly-offline compile-then-bake-static pattern is even ADR-0001/guided-JSON-compatible (confirming no runtime dep, **no ADR-0001 amendment**). (3) At that point hand-authored few-shot demos + the existing eval gate are the ~90% solution at a fraction of the cost.

**When A-1.5 is actually reached** (a three-part AND: Surface A shipped + a *measured* >10% follow-up-resolution gap + hand-authored few-shot tried and proven insufficient): default to **hand-authored few-shot demos** in a version-bumped `prompts/rewrite_followup/vN.md`, A/B'd via `MEMEX_PROMPTS__PIN__REWRITE_FOLLOWUP` against the existing eval. Only if that stalls, a **borrowed ~120-line bootstrap loop** (run our *real* `complete_structured` + `hybrid_search`, keep demos where `gold_chunk_recall@50` improves) — with DSPy `BootstrapFewShot` as an optional offline *idea-generator* behind a dev-only `[optimize]` uv extra, **never runtime-imported** (`grep -r 'import dspy' src/memex/` must be empty), **never** applied to the HARD-gate-validated `answer`/`verify`/`assess` prompts (DSPy #9039 leaks demos even zero-shot). The optimizer's `gold_chunk_recall` is only its inner objective; the existing `run_eval` HARD gates (`refusal_cf=1.0`, 0 hallucinations, N=3, 12 corpora) are the sole ship gate; kill-switch = the prompt-version pin (zero derived state); demos are sanitized to placeholders (the key-figures lesson). No ADR unless DSPy is adopted and proves architecturally load-bearing.

**Prerequisite this surfaced (independent of DSPy):** measuring the A-1.5 gap — and eval-gating Surface A's rewrite at all — needs a **multi-turn eval corpus** (`tests/eval-data/chat-multiturn/`: prior turns + follow-up → gold standalone-query + gold chunk_ids) and a recall runner wiring `gold_chunk_recall` (`eval/scoring.py:463`, unit-tested but not yet in `run_eval`) over the *rewritten* query. Author it via the `extend_corpus` playbook when Surface A is built (its own session, like the ar-14/15 Table-RAG item).

---

# SURFACE B — Ungrounded Expert Mode (FENCED — inverts the grounding contract)

> **⚠️ Everything below is OFF the grounded path.** Surface B is a sibling of `summarize`, never reachable from `answer_query`, **not** judged by `refusal_cf`, and clearly labeled "model knowledge, not your vault." It exists so reasoning *reach* is available when the vault falls short — but it is reached only by **explicit user choice** (never an automatic escalation from a Surface A refusal in v1; that consented bridge is a later product decision). It shares the chat chrome and the conversation store with Surface A but must remain visually and structurally unmistakable.

## 10. The ungrounded expert surface (ADR-0013)

> **SHIPPED 2026-06-01 (v1).** Built largely as designed below — `complete_reasoning` + `split_think` (now a public name) in `models/client.py`, the linear `agents/expert.py::expert_answer`, `prompts/expert_answer/v1.md`, off-by-default `agents.expert_mode_enabled`, the GENERAL no-subprocess path on the live 4B. **Three deltas, all recorded in [ADR-0013 §Realized v1](../adr/0013-ungrounded-reasoning-expert-mode.md#realized-v1-2026-06-01):** (1) **surfaces are CLI + webui only — NOT MCP** (reserved for the upstream flagship-fallback, same as Surface A); (2) **`enable_thinking` defaults to FALSE** — verified live, the 4B's thinking mode emits a verbose UNTAGGED scratchpad that eats the whole token budget before the answer and can't be split, so v1 reasons with a clean decode + a reasoning-eliciting prompt (the kwarg + `split_think` stay plumbed for a future separable-trace model); (3) **`run_expert_eval` / `memex eval-expert` is now ALSO SHIPPED** (workflow-designed + adversarially hardened; spec [`expert-eval.md`](expert-eval.md)) — a two-floor honesty + usefulness tripwire with deterministic hard gates and a REPORTED (model-parameterized, circularity-aware) verifier judge; analytical correctness stays out of scope by construction. The reason-then-ground bridge (§11) remains unbuilt (a future A-enhancement).

**Safe-leverage rule (both conjuncts required):** reasoning (`enable_thinking` + emitted CoT) may be enabled IFF **(1) the grammar is dropped** (the call omits `response_format=json_schema`) AND **(2) the surface is off the gated path**. Dropping (1) without (2) = reasoning on `/ask` = the HARD gate inverted (forbidden). Keeping (2) without (1) = a guided call on the expert surface = no CoT (the grammar still suppresses it — pointless). Both together is exactly ADR-0013's surface.

- **`complete_reasoning()` — a NEW sibling to `complete_structured`, never an `enable_thinking=False` kwarg on it.** It mirrors `complete_structured`'s dispatch (the `_inference_override` ContextVar at `client.py:273`, then `settings.models.reasoner` → `orchestrator`) but **omits `response_format` entirely** and **adds** `extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}}`, returning the **raw string** + token count. Keeping it a separate, greppable function is the single highest-leverage guard: `complete_structured` stays the **sole** `response_format` emitter and never passes `chat_template_kwargs`, so "is this call grounded?" is answerable by *which function it is*.
- **`_split_think(text) -> (trace, body)`** — a pure `<think>…</think>` splitter beside `_inline_refs`; keep the trace for Langfuse, surface only the body; **fail-open** (no/nested/truncated tag). Display hygiene on the free-text side only — **never** a determinism mechanism (do not add it to the grounded path).
- **`agents/expert.py::expert_answer`** — a *linear* agent (like `document_summarizer`, not a LangGraph), **never** a branch in `answer_query`: `hybrid_search` → `cross_encoder_rerank` → an `expert/v1.md` prompt placing chunks as **Evidence** → `complete_reasoning(enable_thinking=True)` → `_split_think`. It **never** calls `verify`/`assess_relevance` — the deliberate ADR-0013 v1 contract ("reasoning OVER retrieved evidence": relax literal-grounding to supported-by-the-evidence-set).
- **Two execution paths.** GENERAL (the crown jewel, the ONLY one wired in v1): `reasoner` is `None`/equals the orchestrator → **no subprocess**, the override stays unset, `complete_reasoning` hits the live 4B daemon. SPECIALIST (a distinct id, e.g. a self-quantized Foundation-Sec-8B): the `pause_vllm_for_gpu → serve_summarizer_vllm → inference_override` `AsyncExitStack` (`document_summarizer.py:1072-1079`), session-pinned for a chat. **⚠️ SHIPPED-delta (#395, 2026-06-02): the SPECIALIST swap-in is UNWIRED in v1** — `expert_answer` does NOT build that AsyncExitStack (ADR-0013: the reasoner hook is RESERVED, UNUSED in v1). A set `reasoner` is sent to the live daemon via `complete_reasoning` and must therefore ALREADY be the served model (a non-served id 404s); the `_inference_override` seam is read symmetrically with `complete_structured` so wiring it later (clone `serve_summarizer_vllm`) stays a small follow-on when a 12 GB-fitting specialist lands. Config: `ModelSettings.reasoner: str | None = None`, `AgentsSettings.expert_mode_enabled: bool = False` (off until ADR-0013 → Accepted).
- **A NEW non-refusal eval discipline** (`run_expert_eval`, modeled on `run_summary_eval`): Axis 1 faithfulness (deterministic `absent_assertion_violations` no-leak HARD floor + a *reported, not gated* soft reasoning-soundness judge); Axis 2 **provenance honesty as a STRUCTURAL assertion** (a deterministic, non-LLM `grounded=False`/provenance marker — never ask the LLM to self-label); Axis 3 refusal only on genuinely out-of-domain probes (the *inverse* of `refusal_cf`). A small **local** `tests/eval-data/expert/` corpus; `memex eval-expert`. **Do not** wire the expert surface into `run_eval` — `refusal_cf` would penalize it for doing its job.

## 11. The reason-then-ground bridge (joins A ↔ B)

> **SHIPPED 2026-06-02 (v1) — a DEDICATED surface, `memex bridge` + the `/bridge` "Analysis" tab** (CLI + webui only, fenced behind `agents.expert_mode_enabled`; NOT MCP). `agents/bridge.py::reason_then_ground → BridgeAnswer`. **Stage 1** reuses the shared `agents/expert.py::reason_over_evidence` (extracted from `expert_answer`); **Stage 1.5** extracts discrete `CitedClaim`s from the free-text analysis via `complete_structured(schema=DraftAnswer, prompt_tag="extract_claims@v2")` (v2, 2026-06-02, fixes under-coverage — extract every attributed/decomposed groundable fact, demote the synthesis thesis — while keeping the faithfulness guard; A/B grounded 1→8 on the failure case; pin `MEMEX_PROMPTS__PIN__EXTRACT_CLAIMS=v1` to revert; see ADR-0016 §Amendment); **Stage 2** runs `repair_claim_chunk_ids` → a deterministic drop of any claim whose id doesn't resolve to a reranked chunk → the UNCHANGED `verify_grounding/v2` via the shared `agents/grounding.py::ground_claims` (hoisted from the summarizer's `_ground_points`, so the summarizer + bridge now share ONE grounding primitive; the `/ask` `verify` node is untouched). **Three deltas from the design below:** (1) **the STANDALONE bridge is verify-only — NO `assess_relevance`** (that whole-answer responsiveness gate misfires on "which reasoned claims are supported"; it was reserved for the consented A→B escalation where the bridge output IS the answer — now REALIZED, see the present-as-answer escalation below + ADR-0016); (2) a **two-layer faithfulness guard** — an extractor-not-generator prompt PLUS the deterministic id-resolution drop — closes the "fabricated-but-coincidentally-grounded" hole the verifier is structurally blind to; (3) **zero grounded ≠ refuse** — the labelled analysis returns with an empty grounded subset (no citation chrome), the bridge has no refuse state. KNOWN GAP (matches the summarizer): the numeric-aggregate backstop lives in the `/ask` `verify` NODE, not in `ground_claims`, so bare computed-figure claims are out of v1 scope. HARD-gate-neutral by construction (the `answer_query` graph is never imported; pinned by a structural test). Live-validated: a single OSPF analysis grounded 8/8 extracted claims, each cited by title › section · page.
>
> **The consented A→B escalation (§11's last piece) is ALSO SHIPPED 2026-06-02.** From a Surface-A `/ask` REFUSAL, the user is offered an EXPLICIT, never-automatic path to re-run the same question through the bridge: webui — a "Reason over this instead →" affordance in the refusal panel (gated on `expert_enabled`, refusal-only by template construction, amber not the grounded-blue) that hx-posts the question + the ORIGINAL scope to `POST /bridge`, swapping the bridge result (with its ungrounded banner) into `#answer`; CLI — a refusal-only, expert-gated stderr HINT naming `memex bridge <shlex-quoted-question>` (the user chooses to run it = consent). HARD-gate-neutral (the `answer_query` graph + refusal logic are untouched). The **standalone `/bridge` composer** also carries the `/ask` document **scope-picker** (`add4a49`, the same `_scope_picker.html` — tick docs + saved scope-sets + entity suggestions), so a fresh analysis can be scoped like a question; the result shows the `.ans-scope` "Scoped to …" note (parity with `/ask`, also on the escalation result). Both scope paths now work and show what they scoped to.
>
> **The present-as-answer escalation (ADR-0016) — ALSO SHIPPED 2026-06-02 — realizes "the bridge output IS the answer."** The first escalation (above) re-ran the question through the standalone bridge VERBATIM (verify-only, the LABELLED-analysis surface). It now ADVANCES: the consented escalation sends `present_as_answer=true` (a hidden form field — the ONLY discriminator between the two `POST /bridge` callers; the standalone composer omits it, unchanged), so the bridge ALSO runs the responsiveness gate. When the grounded subset is **non-empty AND responsive** (`BridgeAnswer.presented`), the grounded claims are **presented AS a direct grounded answer** (a distinct "Reasoned, then grounded" surface — the claims are the answer, cited, with the ungrounded reasoning fenced in a collapsed `<details>`), at the SAME verify + responsiveness bar as `/ask`. Otherwise (zero-grounded OR non-responsive) it falls back to the labelled-analysis surface (the bridge keeps its **no-refuse-state** contract; a non-responsive subset gets a quiet "related question" note). The gate is `agents/grounding.py::assess_responsiveness` — the UNCHANGED `assess_relevance@v1` prompt + `RelevanceAssessment` schema, reused OUTSIDE the `/ask` graph (the `/ask` `assess_relevance` NODE is NOT refactored to share it — that would create an `answering↔grounding` import cycle AND edit a HARD-gate node; the single source of truth is the prompt + schema). **No ungrounded text reaches the answer:** the presented body is the grounded `CitedClaim`s, and the `assess_responsiveness` input is a DETERMINISTIC join of the grounded claim texts — never `analysis`, never the extractor's free summary. Fail-CLOSED: a gate `ModelCallError` → not presented (falls back), never an un-gated answer. HARD-gate-neutral by construction (the `answer_query` graph is still never imported; `present_as_answer` defaults False so the standalone path + the existing bridge tests are byte-identical). CLI parity: `memex bridge --answer` + the refusal hint names it. **A live audit (per-claim re-verification) found the shared gate intermittently over-grounds on entity-NAME presence alone (a claim cited to a slide that only lists "RBAC"/"ABAC"/… names), which present-as-answer would frame as VERIFIED — so a deterministic, presentation-only NAME-ONLY GUARD (`core/text.is_name_only_chunk`, kill-switch `agents.bridge_name_only_guard_enabled`, default on) holds such claims out of `presented_claims`; the responsiveness gate runs on the surviving `presentable` subset (an all-filtered escalation falls back to the labelled analysis). It is presentation-only — `ground_claims`, the standalone grounded subset, and the `/ask` path are untouched; the underlying gate leniency is a documented follow-up (it touches the HARD-gate path). See [ADR-0016 §Amendment (name-only guard)](../adr/0016-reason-then-ground-bridge.md).** **The bridge also surfaces its consulted evidence (`3d1a127`): a navigable "Retrieved from your vault" section over the already-existing `BridgeAnswer.evidence` (deduped by `document_id`; `title › section · p.N` → `/documents/{id}?page=N`), on both surfaces — most valuable on the zero-grounded fallback, which previously NAMED real vault docs in prose but linked to none, so the user can open them and check the vault. Labelled "shown as context", NOT grounding cites; presentation-only + HARD-gate-neutral; a UI follow-on to ADR-0016 (no new ADR).**

The only way to combine reasoning with grounding, isomorphic to Table-RAG (synthetic chunk → unchanged cite machinery) and the summarizer (free-ish MAP → reuse `verify_grounding/v2` per point). **Stage 1 (reason, ungrounded):** `complete_reasoning(enable_thinking=True)` over the same retrieved evidence set, producing free-text analysis + candidate conclusions. **Stage 2 (ground, deterministic):** wrap each candidate as a `CitedClaim`, run `repair_claim_chunk_ids`, then the **existing, unchanged** `verify` + `assess_relevance` gate — only survivors are presented as vault-grounded; the rest stay labeled "model reasoning." Reasoning *expands* the candidate set; the unchanged gate *contracts* it to the grounded subset. This is the safe path to give Surface A *some* analytical reach (a future A-enhancement) without relaxing its gate — and the consented A→B escalation, when built, routes through it.

---

## 12. Risks + guard rails (both surfaces)

| # | Failure mode | Guard rail |
|---|---|---|
| **R1** | **Cross-turn grounding leak** (a follow-up "grounds" against a prior answer) | Structural: `verify`'s `chunk_by_id` is built from `state.reranked` only (`:1667`); the memory feeds the *rewrite* (asserts nothing) and the carry re-introduces only *real chunks* that must re-survive rerank. Prior-answer text never enters the grounding pool. |
| **R2** | **CoT contamination of `/ask`** | `complete_reasoning`/`enable_thinking` live only on Surface B; `complete_structured` is the sole `response_format` emitter and never passes `chat_template_kwargs`. A test asserts every `/ask` graph node's call carries a `json_schema` `response_format`. Surface A never imports the reasoning primitive. |
| **R3** | **Mistaken-for-grounded** (B read as A) | Surface separation by construction: distinct routes, a **deterministic** provenance label, a visually-distinct surface, and **no automatic escalation** from a Surface A refusal. The A→B escalation (SHIPPED 2026-06-02, §11) is **consented** — the user explicitly clicks "Reason & verify this" (webui) or runs the hinted `memex bridge --answer` command (CLI); the user *chooses* B. When the escalation **presents-as-answer** (ADR-0016), it is NOT a relaxation: every presented claim passed the SAME verify + `assess_relevance` bar as a grounded `/ask` answer, the presented body contains NO ungrounded text (grounded `CitedClaim`s only; the ungrounded reasoning is fenced in a collapsed, labelled `<details>`), and the surface carries a distinct "Reasoned, then grounded" provenance — so it is read as exactly what it is (a grounded answer reached via reasoning), never as ungrounded prose nor as a plain `/ask` answer. |
| **R4** | **Context/VRAM overflow** | Surface A: the history cap (1400 tok) lives only in the rewrite call; the answer call is unchanged and reuses the existing context-overflow degrade (`answer`, drop lowest-ranked real chunk + retry). Surface B: bound the answer `max_tokens` reserving budget for the `<think>` trace; general path = no second model. |
| **R5** | **Referent mis-resolution** (Surface A rewrite) | Degrades to a grounded-but-non-responsive answer → the existing relevance gate refuses + surfaces `standalone_query` to reformulate. Never a hallucination. |
| **R6** | **`conversations.sqlite` wiped by `reindex --force`** | Exclude it from the teardown allow-list (the `scope_sets.json` precedent). |
| **R7** | **Doc re-ingested mid-conversation** (a carry id vanishes) | `chunks_by_ids` skips missing ids (fail-open); a vanished carry chunk is simply absent. |

## 13. Cross-references

- **[ADR-0013](../adr/0013-ungrounded-reasoning-expert-mode.md)** — *(Accepted, v1 shipped 2026-06-01)* the ungrounded expert mode (Surface B); the realized contract boundary + the `enable_thinking`-defaults-FALSE finding are in its §Realized v1.
- **[ADR-0015](../adr/0015-qwen35-4b-unified-orchestrator.md)** — the unified 4B; the grammar-suppresses-reasoning property (`:72-73`), the `enable_thinking` mechanism + the reverted VLM-unification (`:104-105`), the kill-switch (`:96-97`).
- **[ADR-0008](../adr/0008-document-summarization.md)** — the grounded summarizer: the surface-separation precedent, the bounded-list digest idiom (reused for conversation compaction), and the swap-in seam (reused by Surface B's specialist path).
- **[ADR-0004](../adr/0004-observability-structlog-langfuse.md)** — the `correlation_id` threaded per turn.
- **Specs:** [`scope-sets.md`](scope-sets.md) / [`artifact-scope.md`](artifact-scope.md) (the scope path reused per-conversation, and the user-authored-state-excluded-from-teardown precedent for `conversations.sqlite`); [`document-summarization.md`](document-summarization.md) (the MAP→GROUND→REDUCE pattern the bridge mirrors).
- **Source seams:** `agents/answering.py:2251` (`answer_query` — the additive `prior_carry_chunk_ids`), `:465`/`:2322` (`AnswerState`), `:559`/`:805` (`retrieve` + the `expand_graph` candidate-merge seam to mirror), `:1667` (the `verify` grounding seam — must stay intact), `:1906` (`assess_relevance`); `index/fts_store.py:348` (`chunks_by_ids`) / `:103` (the sqlite connection pattern); `core/sqlite_tuning.py:20`; `agents/document_summarizer.py:463` (`_reduce` idiom) / `:1072-1079` (the swap-in `AsyncExitStack`, Surface B); `models/client.py:219` (`complete_structured`) / `:327-334` (the unconditional `response_format`) / `:59-73`/`:273-285` (`inference_override`); `webui/app.py:618`/`:652`/`:931` (the `/ask` + `summarize` route patterns to mirror); `webui/progress.py:86`/`:163-187` (the long-poll core to reuse); `templates/_answer.html` (the assistant bubble) + `_progress.html`.
- **Memory:** `reasoning_expert_mode_scope_2026_05_29.md`, `qwen35_4b_orchestrator_swap_2026_06_01.md` (the ADR-0015 ship, the keep-VL-4B + keep-8B-VL + reasoning-doesn't-improve-vision findings).
