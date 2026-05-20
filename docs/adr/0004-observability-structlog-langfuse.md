# ADR-0004: Observability — structlog + Langfuse, joined by correlation_id

- **Status**: Accepted
- **Date**: 2026-05-19
- **Deciders**: Memex core team
- **Tags**: stack, observability, agents

## Context

Memex's fourth principle is *Observable at Every Layer*: every parsing decision, every retrieval, every agent step, every model call must be traceable, timestamped, and inspectable. The promise made in the vision is that any answer Memex gives can be replayed from its trace.

That promise has two technical halves:

- **Logs** — a structured record of what each function decided, written to disk for the grep/jq workflow and for post-hoc forensics when Langfuse isn't open.
- **Traces** — a parent/child span tree of model calls, retrievals, and agent transitions, browsable in a UI built for the job, with prompts and completions inline.

ADR-0001 already committed us to vLLM (OpenAI-compatible). ADR-0003 already committed us to a single-machine, local-first architecture. This ADR settles which libraries fulfill the observability promise, how they cooperate, and how a single `correlation_id` flows through both layers without per-callsite ceremony.

## Decision Drivers

- Local-first: no SaaS, no third-party telemetry endpoints, traces never leave the host
- LangGraph-native: our agents are state machines; the tracer should auto-instrument them
- vLLM is OpenAI-compatible; the tracer should wrap the OpenAI client cleanly
- One `correlation_id` per document run / query, end-to-end across logs and traces
- Low ceremony in module code — instrumentation should not be a per-function chore
- The dev loop on a bad answer should be "open one tool and see what happened"

## Considered Options

1. **structlog + Langfuse (self-hosted)** — JSON logs + a trace store with first-class LLM-call semantics
2. **stdlib logging + OpenTelemetry + Jaeger** — vendor-neutral generic spans, no LLM semantics
3. **structlog + LangSmith** — best-in-class LangChain tracing, but hosted SaaS only
4. **structlog only** — logs without a trace UI; debugging via jq

## Decision

**Option 1: structlog + Langfuse (self-hosted).**

- **structlog** writes structured JSON to disk (console renderer in dev). Configuration lives in `memex.observability.logging`.
- **Langfuse** runs in Docker compose on `localhost`. The v4 self-hosted stack is Postgres ≥14 + ClickHouse + Redis/Valkey + MinIO (Postgres-only is no longer sufficient; this is a v3+ requirement). Storage lives under `vault/.memex/traces/`. Configuration lives in `memex.observability.tracing`.
- A **ULID `correlation_id`** is generated at the top of every document run or query. It is bound into structlog's `contextvars` (so every log line carries it) and forwarded to Langfuse via the LangGraph `config.metadata` convention using `langfuse_*` prefixed keys (so every span carries it). The two layers are joined by that single identifier.
- Agent runs are instrumented via LangGraph's `CallbackHandler` from `langfuse.langchain` (v3+ import path; the v2 `langfuse.callback` path is gone). No per-node manual span code.
- Model calls go through `memex.models.client`, which imports `from langfuse.openai import openai`. Every `complete_structured` call automatically becomes a Langfuse generation span attached to the current trace, with prompt, completion, schema, token counts, and latency captured.

## Consequences

### Positive

- Module authors do not write trace code. They bind structlog context once per run and call model helpers — tracing happens at the framework boundaries.
- One `correlation_id` ties one query's logs to its trace to its model outputs. Forensics is fast: copy the ID from a log line, paste it into Langfuse, see the whole tree.
- Langfuse has LLM-aware semantics (prompt, completion, tokens, latency, cost) that generic OpenTelemetry spans don't model natively. The trace UI renders prompts and completions side-by-side, which is the actual debugging task.
- Self-hosting preserves the privacy promise; nothing leaves the machine.
- The Langfuse UI is *the* tool to open when an answer is wrong, which means there is *a single tool to open* — not three.

### Negative / Trade-offs

- Two libraries to install and configure instead of one. Configuration lives in `memex.observability.{logging,tracing}` and runs once at startup.
- Self-hosted Langfuse adds a Docker compose service to the runtime. Users who don't want to run it can omit it; the agent still works (Langfuse SDK retries in the background and logs warnings on sustained outage rather than failing queries).
- LangGraph's `CallbackHandler` is LangChain-ecosystem-flavored. If we ever leave LangGraph, the auto-instrumentation goes with it and we'd need to manually instrument the replacement. Acceptable: the alternative was manual instrumentation from day one.
- We import Langfuse-specific decorators in `memex.observability.tracing` and `memex.models.client`. If we later need vendor neutrality, those two modules are the migration surface — small.

### Neutral

- Sampling is configuration, not code: 100% for agent runs (always), 100% for parse runs by default with a 10% throttle for high-volume mode, aggregated metrics for inner rerank/embed calls. Set on the handler at construction.
- Trace storage lives under `vault/.memex/traces/` by convention, so it follows ADR-0003's rules for derived state — disposable, regenerable, never the source of truth.

## Alternatives in Detail

### OpenTelemetry + Jaeger

Vendor-neutral, broadly supported, semantic conventions for HTTP/DB out of the box. Rejected for v1 because:

- OTel's GenAI semantic conventions for prompts, completions, tokens, and tool calls are still maturing. Langfuse already models these as first-class fields; OTel forces DIY attributes.
- Jaeger's UI is built for distributed-systems debugging, not for inspecting prompts and reading model outputs. A developer opening a trace to debug a bad answer wants prompt + completion + structured output rendered side by side. Langfuse does this natively; Jaeger doesn't.
- Adopting OTel would still leave the LLM-semantics gap; we'd write or import a Langfuse-equivalent layer regardless.

Worth revisiting once OTel GenAI semantic conventions ship stable and a viewer reaches Langfuse parity for the LLM debugging task.

### structlog + LangSmith

LangSmith is the most polished LangChain-ecosystem trace UI. Disqualifying: it is SaaS only, with no self-hosted path that meets the local-first commitment. Recording prompts and completions in a third-party tenant contradicts the entire vision.

### structlog only

Cheapest. No trace UI; everything is JSON in files, queryable with jq and ripgrep. Rejected because "open a trace and see what happened" is the *primary* way we plan to debug bad answers. Forcing that workflow through jq is a self-imposed productivity tax on the most expensive activity in the dev loop.

## Operational Notes

### Module layout

- `src/memex/observability/logging.py` — `configure_logging(json_output)`. Called once from CLI entrypoints and tests.
- `src/memex/observability/tracing.py` — `configure_tracing(settings)`, `callback_handler(correlation_id, name)`, `bind_run_context(correlation_id, **fields)`. Called once at startup; helpers called per run.
- `src/memex/models/client.py` — imports `from langfuse.openai import openai` so every model call is auto-instrumented without per-callsite changes.

### Idioms

- **Bind once per run** at the top of `answer_query`, `parse_document`, etc. The `correlation_id` flows via `contextvars` to every nested log call without being passed explicitly.
- **Inside nodes / helpers**, just call `log = logger.bind(node="<name>"); log.info("event", k=v)`. The `correlation_id` and run-scope fields appear automatically.
- **No `extra={...}`** — that's the stdlib idiom; structlog's processors drop the dict on the floor.
- **No f-strings in event names** — `f"retrieved {n} chunks"` kills field aggregation. Use `log.info("retrieved", count=n)`.

### Langfuse v4 specifics

The v2 `from langfuse.callback import CallbackHandler` import path no longer exists, and the v2-era `CallbackHandler(trace_name=..., metadata=..., tags=...)` constructor kwargs were removed in v3. The v4 pattern Memex commits to:

- Import: `from langfuse.langchain import CallbackHandler`
- Construct bare: `handler = CallbackHandler()`
- Set per-run attributes via the LangGraph `config.metadata` dict using `langfuse_*` prefixed keys: `langfuse_session_id`, `langfuse_tags`, `langfuse_trace_name`.
- For deterministic trace IDs (so a log-line `correlation_id` is directly findable in the Langfuse UI), use `Langfuse.create_trace_id(seed=correlation_id)` and wrap the `ainvoke` in `start_as_current_observation(trace_context={"trace_id": ...})`. This is an optimisation, not a correctness requirement — adopt when there's a running Langfuse server to validate against.
- The `from langfuse.openai import openai` wrapper survives unchanged in v4; the per-model-call instrumentation idiom doesn't change.

### Failure handling

- If Langfuse isn't reachable, the SDK retries in the background and queries continue to succeed. A sustained outage produces structlog warnings; it never fails a user's query. Observability outages and agent outages are independent.
- Misconfiguration (e.g., missing keys when Langfuse is supposed to be enabled) fails at startup via `configure_tracing`, not silently at the first model call.

## Revisit When

- OpenTelemetry GenAI semantic conventions reach stability across agent + framework spans (client spans graduated in semconv 1.40, April 2026; agent/framework still Development). Langfuse v4 is already OTEL-native under the hood, so the migration path to OTel-direct instrumentation is one config flip rather than a rewrite — re-check Q4 2026.
- We adopt a non-LangGraph agent framework (the `CallbackHandler` integration stops applying)
- Self-hosted Langfuse's Docker compose becomes a maintenance burden disproportionate to its value
- Langfuse's licensing changes in a way that breaks the open-source commitment

## References

- ADR-0001: vLLM as the sole inference engine (OpenAI-compatible API enables Langfuse's wrapper)
- ADR-0003: Markdown vault is the source of truth (traces are derived state — disposable, under `.memex/`)
- Memex developer guidelines, §"Observability with structlog and Langfuse"
- Langfuse: https://langfuse.com (self-hosted via Docker compose)
- structlog: https://www.structlog.org
