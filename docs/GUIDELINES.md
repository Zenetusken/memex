# Memex — Developer Guidelines

### Engineering practices for a local-first, fully agentic document system

---

## How to read this document

This is the engineering counterpart to the [Memex vision document](VISION.md). The vision describes *what* we are building and *why*. This document describes *how* — the stack, the architecture, the patterns, the discipline.

Every guideline here traces back to one of the five principles in the vision:

1. **Local-First, By Construction** — the codebase makes a remote call impossible by design
2. **Markdown as Source of Truth** — files on disk are authoritative; everything else is derived
3. **Small Models, Used Well** — discipline beats parameter count
4. **Observable at Every Layer** — every decision is auditable
5. **Composable, Not Captive** — MCP and plain files at every boundary

When in doubt, return to those.

---

## Part I — Foundations

### Stack at a glance

| Concern | Tool | Why |
|---|---|---|
| Language | Python 3.12+ | ML ecosystem; mature async; type system now strong enough |
| Package management | **uv** | Replaces pip + venv + poetry + pip-tools; ~100× faster |
| Lint & format | **ruff** | Single tool, fast, replaces black + isort + flake8 + pyupgrade |
| Type checking | **pyright** in strict mode | Faster than mypy; better inference |
| Data models | **pydantic v2** | Every interface, no raw dicts crossing module boundaries |
| Settings | **pydantic-settings** | TOML + env, validated at startup |
| Web/API framework | **FastAPI** | For the MCP server, the local UI backend, and the health endpoints |
| CLI | **typer + rich** | Type-driven CLI; pretty output for free |
| Agent orchestration | **LangGraph** | Explicit state machines, not free-form ReAct |
| Inference | **vLLM** | Paged attention, OpenAI-compatible API, sustained throughput under agentic load |
| Document parsing | **Docling** | Layout, tables, equations → Markdown; CPU+GPU |
| Vision-language fallback | **Qwen3-VL-8B-AWQ** (parse-time vLLM) | Hard pages, handwriting, dense diagrams, directed flow/state diagrams |
| Embeddings | **EmbeddingGemma 300M** | Small, multilingual, fast |
| Reranker | **bge-reranker-v2-m3** | Second-stage precision |
| Vector store | **LanceDB** | Embedded, columnar, fast |
| Full-text search | **SQLite FTS5** | Already in stdlib; BM25 free |
| Graph store | **RyuGraph** | Embedded property graph; Cypher; columnar. The maintained Kuzu fork — see ADR-0005 |
| Table analytics (Table-RAG) | **stdlib `sqlite3`** | Per-doc in-memory text-to-SQL over extracted tables; read-only single-SELECT guard (`docs/specs/table-sql.md`). DuckDB was scaffolded here for native Parquet/SQLite/LanceDB reads but superseded by sqlite3 (no cross-table joins needed) and pruned as an orphaned dep 2026-06-05 |
| Tracing | **Langfuse** (self-hosted) | Every agent step, traceable and replayable |
| Logging | **structlog** | Structured JSON logs; Langfuse-compatible |
| Testing | **pytest + hypothesis** | Unit, property-based, agent evals |
| Pre-commit | **pre-commit** | Ruff, pyright, schema checks |

There is no model picker, no inference engine picker, no embedding picker. Memex is opinionated. Plugins can extend; the core does not branch.

### Project layout

A single repository, single Python package, with feature modules under `src/memex/`. We are not building a monorepo of independent packages — the original guidelines did, and it created friction without buying separation. One package, clear module boundaries, ship together.

```
memex/
├── pyproject.toml             # uv-managed; single source of dependency truth
├── uv.lock                    # committed; reproducible builds
├── README.md
├── docs/
│   ├── adr/                   # architectural decision records (ADR-0001 ... 0019)
│   ├── specs/                 # build-level designs the ADRs decide (one per feature)
│   ├── audits/                # multi-agent audit reports + per-phase findings
│   ├── deploy/                # systemd + launchd templates, hardware-tiers.md
│   └── *.md                   # ROADMAP, IMPLEMENTATION-PLAN, VISION, this file
├── src/memex/
│   ├── __init__.py
│   ├── cli/                   # typer entry points
│   ├── core/                  # event bus, config, errors, types
│   ├── ingest/                # file watching, validation, queuing
│   ├── parse/                 # Docling + VLM fallback orchestration
│   ├── enrich/                # entity extraction, citation linking
│   ├── index/                 # LanceDB, FTS5, Kuzu writers
│   ├── retrieve/              # hybrid search + reranking
│   ├── agents/                # LangGraph state machines
│   ├── prompts/               # versioned prompt library
│   ├── models/                # model loading, VRAM management
│   ├── observability/         # Langfuse, structlog setup
│   ├── mcp/                   # MCP server exposing the vault
│   ├── webui/                 # FastAPI + HTMX local UI
│   └── vault/                 # Markdown read/write, frontmatter
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── evals/                 # agent quality regressions
│   └── fixtures/              # test documents with ground truth
└── scripts/
    ├── download-models.py
    └── benchmark.py
```

### Coding standards

**Types are not optional.** Every function signature, every dataclass, every pydantic model. `pyright --strict` runs in CI and `src/memex` is kept at **0 errors / 0 warnings**. Untyped `Any` requires a comment explaining why. When a third-party library ships no usable types, give it a minimal `.pyi` stub under `stubs/` (wired via `stubPath`) covering only the surface Memex uses — don't silence at the call site. For libraries that ship a partial `py.typed` (torch, transformers, openai, typer), don't shadow them with a competing stub; fix at the call site with precise annotations, a single typed wrapper, or `typing.cast`. `# type: ignore[rule]` is a last resort and always carries a trailing `# reason`.

**Pydantic models cross module boundaries, dicts don't.** If two modules need to exchange a `ParsedDocument`, it's a pydantic model in `core/types.py`, not `dict[str, Any]`. This rule alone prevents most integration bugs.

**Async-first for I/O, sync for CPU.** Filesystem reads, HTTP, model inference calls, database writes — all async. Pure computation (parsing tokens, formatting) stays sync. Don't mix without a reason.

**Explicit over clever.** No metaclasses unless there's a specific reason. No decorators that mutate behavior in non-obvious ways. The next person reading this code is you in six months — be kind.

**Errors carry context.** Every custom exception inherits from `MemexError` and includes a `context: dict` field. When an exception propagates to the user, the context is what makes the error message useful.

**Configuration is centralized and validated.** All settings live in a single pydantic-settings model in `core/config.py`. Read from `~/.config/memex/config.toml` with environment overrides. Validated at startup; the application refuses to run with invalid config.

**Logs use structlog with bound context.** Bind the run-scope (`correlation_id`, `query` preview, document ID) once via `structlog.contextvars.bind_contextvars(...)` at the entry point; bind node/function scope via `log = logger.bind(node=name)` locally. Then emit `log.info("event", k=v)`. Never `extra={...}` — that's the stdlib idiom and structlog's processors drop the dict. Never f-strings in event names — `f"retrieved {n}"` kills field aggregation in Langfuse and downstream tooling. The architecture rationale is ADR-0004.

---

## Part II — Architecture

### The pipeline

A document flows through Memex in five stages, each emitting events and traces:

```
ingest → parse → enrich → index → (available for retrieval)
```

Each stage is a separate module, communicates only through events and the vault, and can be re-run independently. If enrichment changes its algorithm tomorrow, you can re-enrich an existing vault without re-parsing.

**Ingest** validates the file (magic numbers, size, format), assigns a stable document ID (content hash + namespace), and emits `document.queued`. Files are copied into `vault/documents/{id}/source.{ext}` immediately so the original is preserved.

**Parse** routes the document. Docling handles clean PDFs, DOCX, HTML, and most slide decks. Hard pages — low-confidence layout, scans, handwriting — are escalated to the VLM. Output is structured Markdown with frontmatter and a `manifest.json` recording every decision: which engine ran on which page, with what confidence.

**Enrich** extracts entities, resolves citations against the vault, inserts `[[wikilinks]]` where confidence is high (with a configurable threshold), and populates frontmatter metadata. Every enrichment writes to the manifest with its source — model name, prompt version, confidence — so any decision can be audited or reverted.

**Index** writes derived state: vector embeddings to LanceDB, FTS5 rows to SQLite, graph nodes and edges to Kuzu. This stage is idempotent and fully regenerable from the Markdown.

**Retrieve** (the query path) hits the indexes, reranks, and hands candidates to the answering agent. This is the read side of the pipeline and runs on demand, not on ingest.

### Event-driven communication

Stages communicate through a typed event bus, not direct function calls. The bus is in-process (no Redis, no Kafka — we're a single-machine application), but the discipline of events keeps modules decoupled and makes the system observable.

```python
# src/memex/core/events.py
from typing import Literal
from pydantic import BaseModel
from datetime import datetime
import ulid

EventStage = Literal["ingest", "parse", "enrich", "index", "retrieve"]

class MemexEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(ulid.new()))
    correlation_id: str  # tracks one document/query across stages
    stage: EventStage
    event_type: str       # e.g. "document.parsed"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: dict
    error: ErrorInfo | None = None
```

Events are persisted to SQLite in a rolling 30-day window. This enables three things: **debugging** (replay a problematic document's run), **recovery** (resume an interrupted pipeline from the last successful stage), and **evaluation** (regression-test by replaying a corpus through a new model).

The `correlation_id` is the most important field. It's a ULID generated at ingest (for documents) or at query receipt (for retrievals), and it flows through every event, every log line, every Langfuse trace, and every model call. Given a correlation ID, you can reconstruct everything that happened. This is the operational manifestation of *Observable at Every Layer*.

### Module boundaries

Each module owns its concerns and exposes a small interface. The interface is a Python module with public functions, not a class hierarchy.

A module **may**: read its own data, emit events, call other modules' public interfaces.

A module **may not**: reach into another module's internals, write to another module's data store, or import private symbols (anything prefixed with `_`).

Concretely: `enrich` does not write to LanceDB. It emits `document.enriched`, and `index` listens. This means swapping the vector store later touches one module, not five.

### Configuration

A single `MemexSettings` pydantic-settings model is the source of truth. Loaded once at startup, immutable thereafter, validated for things like "the configured model fits in available VRAM."

```python
# src/memex/core/config.py
class ModelSettings(BaseModel):
    orchestrator: str = "cyankiwi/Qwen3.5-4B-AWQ-4bit"   # ADR-0015; kill-switch: Qwen/Qwen3-8B-AWQ
    orchestrator_quantization: Literal["AWQ", "GPTQ", "compressed_tensors", "Q4_K_M", "Q5_K_M", "Q8_0"] = "compressed_tensors"
    vlm: str = "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"   # doc-VLM stays the dedicated 8B-VL (4B-VL unification reverted)
    vlm_quantization: Literal["awq_int4", "bf16"] = "awq_int4"
    vlm_serving: Literal["transformers", "vllm"] = "vllm"  # Qwen3-VL → parse-time vLLM process
    embedder: str = "google/embeddinggemma-300m"
    reranker: str = "BAAI/bge-reranker-v2-m3"
    chart_ocr: str = "nvidia/NVIDIA-Nemotron-Parse-v1.2"
    reranker_backend: Literal["cross_encoder", "qwen3"] = "cross_encoder"

class HardwareSettings(BaseModel):
    gpu_memory_fraction: float = 0.85   # torch-level cap; vLLM has its own knob
    max_concurrent_documents: int = 2
    cpu_workers: int = Field(default_factory=lambda: max(1, (os.cpu_count() or 2) - 1))

class MemexSettings(BaseSettings):
    vault_path: Path
    models: ModelSettings = ModelSettings()
    hardware: HardwareSettings = HardwareSettings()
    # ... etc
    model_config = SettingsConfigDict(toml_file="~/.config/memex/config.toml")
```

Since 2026-06-01 (ADR-0015) the 12 GB-tier default orchestrator is itself the 4B-class `cyankiwi/Qwen3.5-4B-AWQ-4bit` (compressed-tensors, no `--quantization` flag, `--kv-cache-dtype auto`, 0.62 util / 8192 window). The legacy `Qwen/Qwen3-8B-AWQ` is retained as the documented one-flip kill-switch. Fail loudly at startup, never silently at runtime — and note the serve-env bridge (`daemon/supervisor.orchestrator_serve_env`) is what makes a config-only orchestrator swap actually reach vLLM (see ADR-0015 / `docs/deploy/hardware-tiers.md`).

---

## Part III — AI and models

### The model stack and VRAM budget

The reference target is an RTX 4070 (12GB VRAM) with 32GB system RAM. Every model choice is constrained by this. **bf16 across the stack on Ada** (ADR-0006); the orchestrator and VLM use Int4 (AWQ / compressed-tensors) for weight compression but their activations stay bf16.

| Model | Role | Quantization | VRAM (live) | Resident? |
|---|---|---|---|---|
| Qwen3.5-4B-AWQ | Orchestrator, answerer (default since 2026-06-01, ADR-0015; unified VL + hybrid-reasoning, 8192 window) | compressed-tensors W4A16 (out-of-process via vLLM, auto KV) | ~6.3 GB | vLLM daemon |
| EmbeddingGemma 300M | Embeddings | bf16 | ~0.6 GB | Always (registry) |
| bge-reranker-v2-m3 | Reranking | bf16 | ~1.0 GB | Always (registry) |
| Qwen3-VL-8B-AWQ | Page transcription fallback (diagrams) | AWQ-Int4 (compressed-tensors, vLLM Marlin) | ~7.4 GB | **Parse-time vLLM process** (`vlm_serving="vllm"`); `disable_vlm=True` by default on 12 GB |
| Nemotron-Parse-v1.2 | Chart-OCR over Docling figures (default since 2026-05-23) | bf16 | ~3.0 GB | Lazy + **enabled by default**; opt-out via `MEMEX_PARSE__DISABLE_CHART_OCR=true` |
| KV cache + vLLM overhead | — | — | ~2.0 GB | — |

Steady-state on 12 GB tier (vLLM + embedder + reranker + KV cache): ~9 GB. **VLM is disabled by default; chart-OCR is enabled by default** (per the 2026-05-23 P3.3-c shootout — Nemotron-Parse-v1.2 achieves no prose regression). Both run only during parse with the orchestrator vLLM paused (`pause_vllm_for_gpu`). The VLM (Qwen3-VL) is itself served by a **short-lived vLLM process** started + torn down inside `vlm_backend.convert_pages` (`vlm_serving="vllm"`; in-process transformers can't run its compressed-tensors int4 on 12 GB — it decompresses to dense → OOM), phase-separated from the in-process chart-OCR pass (the two can't co-reside). See `docs/specs/vlm-vllm-serving.md` + ADR-0006 §4, and `docs/audits/chart_ocr_shootout_2026-05-23.md` for the chart-OCR backend verdict. On the 8 GB tier (P4.2), `Qwen3-4B-AWQ + gpu_memory_utilization=0.50` ships as the documented profile in `docs/deploy/hardware-tiers.md`. On the reference 12 GB rig with chart-OCR enabled, `MEMEX_VLLM_GPU_FRACTION=0.68` is recommended.

### VRAM management

VRAM is a managed resource. We do not let arbitrary code allocate.

- A single `ModelRegistry` in `models/registry.py` owns every model handle.
- Modules request a model via context manager: `async with registry.use("vlm") as vlm:`. The registry handles loading, unloading, and swap.
- The registry tracks resident memory and refuses to load a model that won't fit, raising `InsufficientVRAMError` with context.
- For the orchestrator and embedder, "use" is a no-op (they're always resident). The **default Qwen3-VL VLM is NOT a registry resident** — it's served by a parse-time vLLM process (`vlm_serving="vllm"`); `registry.use("vlm")` is the *legacy* in-process AutoAWQ path (Qwen2.5-VL via `vlm_serving="transformers"`), where it triggers a load if absent.
- Swaps are logged to Langfuse with their wall-clock cost. If swap cost dominates total runtime, that's a configuration problem and the observability will surface it.

### Agent design with LangGraph

Agents in Memex are **state machines with budgets**, not free-form ReAct loops. Every agent is a `langgraph.StateGraph` with:

- A typed `State` (pydantic model) carrying everything the graph reasons over.
- Explicit nodes for distinct reasoning steps.
- Explicit edges, including conditional ones for branching.
- A **step budget** (max nodes traversed) and **token budget** (max model tokens spent).
- A terminal `refuse` node. *Refusing to answer is a first-class outcome*, not a failure.

The answering agent looks roughly like:

```
[start] → retrieve → expand_graph → rerank → query_tables → assess_sufficiency
                                                   │
                                          ┌────────┴────────┐
                                          │                 │
                                  [sufficient]       [insufficient]
                                          │                 │
                                          ▼                 ▼
                                       answer            refuse
                                          │                 │
                                          ▼                 ▼
                                       verify           [end]
                                          │
                          ┌───────────────┼───────────────┐
                          │               │               │
                     [grounded]    [some ungrounded]  [empty draft]
                          │               │               │
                          ▼               ▼               ▼
                       compose      regenerate         refuse
                          │               │               │
                          ▼               └───→ answer ───┘
                        [end]            (with budget cap;
                                          empty-draft shortcircuit
                                          bypasses model call)
```

The `verify` node is non-negotiable. Before an answer reaches the user, a second model pass checks that every claim in the answer is grounded in a retrieved chunk. Ungrounded claims get marked or removed. This is the difference between an agent that's useful and an agent that's a liability.

The `query_tables` node (Table-RAG Phase 2) is an optional augmentation: for aggregation/superlative-shaped queries over a doc with structured tables, it runs guarded text-to-SQL and injects the result as a **synthetic chunk** into the reranked set — so `verify` then grounds it like any other chunk (no separate grounding path). It holds the HARD gate by construction: row-returning SQL ships verbatim document cells; an aggregate (a *new* number) is injected only when an independent Python recompute agrees, otherwise the node no-ops and the agent refuses. See `docs/specs/table-sql.md`.

### Structured outputs, always

Local models without guided decoding hallucinate JSON like it's a creative writing exercise. We don't let them.

- Every model call that expects structured output uses **OpenAI-standard `response_format={"type": "json_schema", ...}`** (portable across vLLM, SGLang, llama-server), preferred over vLLM's deprecated `extra_body={"guided_json": ...}` — see ADR-0001 Revisit. The grammar back-end is xgrammar.
- No regex-parsing of free-text model output. Ever. If you find yourself writing `re.search(r"answer: (.+)", response)`, stop and use a schema.
- **Bound every LLM-emit `str` and `list[T]` with `max_length=N`.** xgrammar enforces the bound at the grammar level, so the model cannot emit past it. Unbounded fields can run away on counterfactual or out-of-distribution queries, trip `max_tokens` mid-emission, and crash JSON validation. Established by `SufficiencyAssessment.reason` (v6) and extended to every other LLM-emit schema.

```python
class CitedClaim(BaseModel):
    claim: str = Field(max_length=500)
    source_chunk_id: str
    confidence: Literal["high", "medium", "low"]

class DraftAnswer(BaseModel):
    summary: str = Field(max_length=600)
    claims: list[CitedClaim] = Field(max_length=20)

# Use the project's `complete_structured` helper — wraps vLLM's
# response_format + Langfuse generation span + bounded-schema
# construction in one call.
from memex.models.client import complete_structured
from memex.prompts import prompt_tag_for

draft, tokens_used = await complete_structured(
    prompt=prompt,
    schema=DraftAnswer,
    prompt_tag=prompt_tag_for("answer"),  # NEVER hardcode "answer@v3" — derive it (see Prompt management)
)
```

### Prompt management

Prompts are code. They are versioned, tested, evaluated, and reviewed.

- All prompts live in `src/memex/prompts/` as `.md` files with YAML frontmatter.
- Frontmatter declares: prompt name, version, role (system/user), target model, input schema (pydantic class name), output schema, eval suite.
- A `Prompt` class loads, validates inputs, renders via Jinja2, and returns a tagged string for tracing.
- Every prompt invocation logs `prompt_name@version` to Langfuse. When eval quality regresses, you can pinpoint which prompt change caused it.
- **Never hardcode the `prompt_tag` string.** `render_prompt`/`render_messages` auto-select the highest on-disk version (and honor `MEMEX_PROMPTS__PIN__<NAME>`), so a literal `prompt_tag="answer@v3"` silently lies the moment a higher `vN` ships or a pin is set. Derive it: `prompt_tag=prompt_tag_for("answer")` (from `memex.prompts`; `active_version(name)` returns the bare version). A source-scan guard test (`tests/unit/test_prompt_tag.py`) fails the build if a `prompt_tag="…@vN"` literal is reintroduced.

Treat prompt edits the way you treat schema migrations: they go in PRs, they get reviewed, and they ship with an eval delta.

### Evaluation

Tests come in four layers. All four run in CI, but they have different cadences.

| Layer | What it tests | Cadence |
|---|---|---|
| **Unit** | Pure functions, parsers, formatters | Every commit |
| **Integration** | Module-to-module flows with fixtures | Every commit |
| **Agent evals** | Quality of agent outputs vs ground truth | Every PR touching prompts/models/agents |
| **End-to-end regression** | Full pipeline on a fixed test corpus | Nightly |

Agent evals are the new thing relative to a traditional codebase. They use a fixed corpus in `tests/fixtures/eval-corpus/` with ground-truth question-answer pairs. Each PR run reports: citation precision, refusal rate on out-of-corpus questions, answer faithfulness (LLM-judged with a local judge model, with periodic human spot-checks). Regressions fail the PR.

Use **ragas** for retrieval evals, **a thin in-house framework** for end-to-end. Don't take a dependency on a heavy eval framework you'll only use 10% of.

### Cheap probes before expensive work; record negative results

Before a full HARD-gate eval or a build, run the cheapest decisive measurement that could kill the idea. For a retrieval change that's an **arm-separation / recall probe**: `eval/scoring.py::gold_chunk_recall(retrieved_ids, relevant_ids, k)` isolates retrieval recall@k from rerank + LLM non-determinism, so a single run per kill-switch setting gives a clean before/after (the agent eval, by contrast, scores only the agent's final cited chunks). The 2026-05-29 FTS BM25-on-NL investigation used it to show `union@50 (dense ∪ bm25) == dense@50` on every corpus — the BM25 arm reaches no gold the dense arm misses — so a *real* phrase-wrap bug was proven **benign** and the fix reverted as a dead lever. The same discipline (a GPU-free pre-check) deferred the Cisco security-LLM orchestrator A/B (the failures were retrieval-bound, not orchestrator-bound) and falsified an auto-noise-detection helper.

**Record negative results in the repo, not just memory** — a ROADMAP "tried + reverted" entry, a `docs/audits/NN-*.md` measurement, and/or a do-not-re-fix docstring at the code site (e.g. `index/fts_store.py::search`) — so the next contributor doesn't re-walk the dead path. The contextual-retrieval prefix, the GTE embedder swap, and the BM25 lexical arm are all banked this way.

---

## Part IV — Storage and retrieval

### The vault is the source of truth

Every architectural decision in this layer follows from one rule: *the Markdown files in `vault/documents/` are authoritative, and `vault/.memex/` is regenerable*.

This rule has consequences:

- We never write derived data without also being able to recompute it.
- We never store information *only* in the index. If a fact about a document exists in LanceDB but not in the Markdown, that's a bug.
- The user can delete the entire `.memex/` directory and rebuild. The rebuild script is shipped, supported, and tested.

### Hybrid retrieval

A single retrieval method is never enough. We run:

1. **BM25** over SQLite FTS5 — strong for keyword queries, proper nouns, exact terms.
2. **Dense vector** over LanceDB — strong for semantic queries, paraphrase, multilingual.
3. **Reciprocal rank fusion** to combine the two ranked lists.
4. **bge-reranker-v2-m3** as a second stage on the top 50 fused candidates, returning the top 10.

Reranking is the highest-leverage retrieval improvement available on local hardware. A 0.6B cross-encoder gives precision gains that no amount of embedding-model tuning will match.

### The knowledge graph

RyuGraph (the maintained fork of Kuzu — see ADR-0005) stores nodes (`Document`, `Entity`, `Concept`, `Citation`) and edges (`MENTIONS`, `CITES`, `DEFINES`, `RELATES_TO`). Schema is in `src/memex/index/schemas/graph.cypher` and versioned.

Entity resolution uses a combination of (a) exact-match canonicalization for known entities (paper DOIs, arXiv IDs), (b) embedding-based clustering for novel entities with manual review thresholds, and (c) explicit `[[wikilinks]]` in Markdown, which are always authoritative.

The graph is used in retrieval for **citation traversal** — answering "what does this paper say that contradicts other papers in my corpus?" requires the graph, not the vector store.

### Caching

Three tiers, all content-addressed (hash of input → cache key).

- **Memory** (LRU, ~500MB): model outputs for the current session.
- **On-disk** (`~/.cache/memex/`, configurable cap, default 5GB): parse results, embedding results, expensive enrichment outputs. Persists across runs.
- **Vault** (forever): final Markdown and derived indexes.

Content addressing means cache invalidation is trivial: if the content changed, the hash changed, and a miss is automatic. No TTLs.

---

## Part V — Interfaces

Memex has three interfaces: CLI, MCP server, and a local web UI. They share the same backend; they differ in audience.

### CLI (primary)

Built with `typer` and `rich`. The CLI is the **canonical interface** — everything the web UI does, the CLI does, often more directly.

```
memex ingest path/to/papers/         # Add documents to the vault
memex ask "What does Smith 2024 say about reflexivity?"
memex graph --document smith-2024    # Print citation neighbors
memex serve mcp                      # Run MCP server on stdio or port
memex serve web                      # Run local web UI on localhost:7423
memex reindex                        # Rebuild .memex/ from documents/
memex eval                           # Run the eval suite
```

Output is rich (tables, syntax highlighting, progress bars) when stdout is a TTY, and plain JSON when piped. This is a non-negotiable for docs-as-code workflows where Memex outputs feed into other tools.

### MCP server

The MCP server is **the public API of Memex**. Anything an external agent should be able to do is exposed as an MCP tool. Anything not exposed is internal.

Initial tool surface:

- `search(query, k=10, filters=...)` → list of cited chunks
- `get_document(id)` → full document content + metadata
- `list_documents(filters=...)` → paginated document list
- `get_graph_neighbors(node_id, depth=1)` → graph traversal
- `ask(question)` → grounded answer with citations

The server runs over stdio (for desktop agent integration) or HTTP (for network-local agents). Use the official MCP Python SDK. Tool schemas are pydantic models, validated at boundary.

### Local web UI

A minimal FastAPI app with server-rendered HTML and HTMX for interactivity. No SPA. No build step beyond Tailwind.

Why HTMX and not React/Svelte: this is a local-only application running on the user's machine. Bundle size doesn't matter, but build complexity does. HTMX gives us all the interactivity we need (search, document preview, graph zoom) for less than 50KB of frontend code total, with no build step.

The UI's job is the *visual* parts of the workflow that CLI can't do well: side-by-side preview of source PDF and extracted Markdown for review, the document-connections view (the `/graph` "Bridges" page — server-rendered/ranked since 2026-05-29; Cytoscape was dropped because a 1-hop neighbourhood is a star with no topology to draw), and document-level annotation correction.

**Browser ingestion (ADR-0019, exclusive-GPU mode).** The UI also has an upload→full-pipeline surface (spec `docs/specs/ui-ingestion.md`): drop a file → it becomes searchable + browsable, zero terminal. The module boundary forbids `webui/` importing `parse`/`index`/`enrich`, so the webui drives the existing CLI as **child subprocesses** (`webui/ingest_driver.py` spawns `memex ingest`/`memex enrich`) — boundary-clean (only `asyncio`/`json`/`os` + `core`), with fresh-process CUDA. Ingestion is an **exclusive-GPU mode**: while a doc ingests, the webui releases its own GPU models, the RAG surfaces lock to an honest "answering paused" (the orchestrator is down during the parse-time VLM serve), and the webui actively reconciles the orchestrator afterwards. All of it is GPU-lifecycle / presentation only ⇒ HARD-gate-neutral; an uploaded doc enters the *same* parse/index/enrich/answer path as a CLI-ingested one.

Dark mode is the default. Light mode is supported. The aesthetic is closer to a developer tool (Linear, Raycast) than a consumer notes app.

### Accessibility

Even though the primary user is a developer at a keyboard, accessibility is not optional.

- All web UI flows are keyboard-navigable; visible focus rings; logical tab order.
- WCAG 2.1 AA contrast minimums (4.5:1 normal text, 3:1 large) — verified in CI via axe-core.
- Color is never the sole carrier of information. Confidence indicators use color + icon + text label.
- Screen reader compatibility tested with NVDA and VoiceOver as part of the release checklist.
- The CLI respects `NO_COLOR` and `TERM=dumb`.

---

## Part VI — Operations

### Observability with structlog and Langfuse

ADR-0004 specifies the architecture in full; the summary:

- **structlog** writes structured JSON logs (console renderer in dev) keyed by `correlation_id`.
- **Langfuse** (self-hosted, Docker compose on `localhost`) stores traces and renders the UI you open when an answer is wrong.
- One ULID-shaped `correlation_id` is generated at the top of every run and flows through both layers via `contextvars` + Langfuse trace metadata. Logs and traces are joined by that ID.

#### Setup

Both libraries are configured once at startup in `memex.observability`:

```python
# src/memex/observability/logging.py
import structlog

def configure_logging(*, json_output: bool = True) -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            (structlog.processors.JSONRenderer()
             if json_output
             else structlog.dev.ConsoleRenderer()),
        ],
        cache_logger_on_first_use=True,
    )

# src/memex/observability/tracing.py
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

def configure_tracing(settings: ObservabilitySettings) -> None:
    Langfuse(
        host=settings.langfuse_host,                # http://localhost:3000
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
    )  # SDK is a singleton; first construction registers it process-wide

def callback_handler(correlation_id: str, name: str) -> CallbackHandler:
    return CallbackHandler(
        trace_name=name,
        metadata={"correlation_id": correlation_id},
        tags=[name],
    )

def bind_run_context(correlation_id: str, **fields) -> None:
    """Bind ContextVars so every log line in the run carries these fields."""
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id, **fields)
```

#### Logging idiom

Bind run-scope once at the entry point; bind node/function scope locally; log normally everywhere else:

```python
# Top of answer_query, parse_document, etc.
bind_run_context(state.correlation_id, query=query[:80])
try:
    ...
finally:
    structlog.contextvars.clear_contextvars()

# Inside any nested function (node, helper, ...)
log = logger.bind(node="retrieve")
log.info("start", query_len=len(state.query))
candidates = await hybrid_search(state.query, k=50)
log.info("done", candidate_count=len(candidates))
```

The `correlation_id` set at entry flows automatically through every nested async task — no manual propagation. Every log line in the run is grouped by ID in JSON output, trivially joinable with Langfuse traces.

#### Tracing agents

LangGraph's `CallbackHandler` auto-instruments every node, edge, and state transition. Pass it on `ainvoke`:

```python
handler = callback_handler(state.correlation_id, "answer_query")
final = await graph.ainvoke(
    state,
    config={
        "callbacks": [handler],
        "metadata": {"correlation_id": state.correlation_id},
    },
)
```

No per-node manual span code. Node topology, transitions, and durations all show up in the trace tree.

#### Tracing model calls

Every model call goes through `memex.models.client`, which imports the Langfuse OpenAI wrapper. The wrapper captures prompt, completion, schema, token counts, and latency as a generation span attached to the current trace:

```python
# src/memex/models/client.py
from langfuse.openai import AsyncOpenAI  # drop-in wrapper around the openai SDK

async def complete_structured(*, prompt, schema, prompt_tag=None, ...):
    response = await client.chat.completions.create(
        model=settings.models.orchestrator,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": _inline_refs(schema.model_json_schema()),
                "strict": True,
            },
        },
        name=prompt_tag or schema.__name__,  # Langfuse span name
    )
    return schema.model_validate_json(response.choices[0].message.content), tokens
```

No callsite manually wraps anything. The generation span lands under whichever trace is active in the calling context.

#### Sampling

- **Agent runs**: 100% sampled. These are the unit of user-visible work.
- **Parse runs**: 100% sampled by default; 10% in high-volume mode (handler constructed with `sample_rate=0.1`).
- **Internal model calls** (rerank, embed): aggregated metrics via a counter; full traces only on errors or anomalies.

#### Privacy and failure handling

Self-hosted Langfuse runs on `localhost`; the trace store lives under `vault/.memex/traces/`, following ADR-0003's rules for derived state (regenerable, disposable, never the source of truth). No external endpoint.

If Langfuse is unreachable, the SDK retries in the background and queries continue to succeed — a sustained outage produces warnings in the structlog stream but does not fail a user's query. Misconfiguration (e.g., missing keys when Langfuse is enabled) fails loudly at startup via `configure_tracing`, never silently at the first model call.

The Langfuse UI is where you go when "the agent gave a bad answer." Open the trace by `correlation_id`, see exactly what happened, find the broken step, fix it.

### Error handling

Errors are typed, structured, and contextual.

```python
class MemexError(Exception):
    def __init__(self, message: str, *, context: dict | None = None, recoverable: bool = False):
        super().__init__(message)
        self.context = context or {}
        self.recoverable = recoverable

class InsufficientVRAMError(MemexError):
    """The requested model does not fit alongside currently resident models."""

class ParseConfidenceTooLow(MemexError):
    """Docling and the VLM both produced low-confidence output for this page."""

class RetrievalEmpty(MemexError):
    """No candidates met the relevance threshold for this query."""
```

Errors propagate to the user as **explanations with options**: "I couldn't confidently extract page 47 of `smith-2024.pdf`. Skip, retry with the VLM, or open the source page?"

Recoverable errors mark the pipeline state for resumption. The next `memex ingest` resumes from the last successful stage, not from scratch.

### Circuit breakers (local resources)

The original guidelines specified circuit breakers for external services. Memex has none. The local equivalents:

- **VRAM circuit breaker**: if three consecutive model calls fail with OOM, halt new work and emit `system.degraded`. The user is told and given recovery options.
- **Parser circuit breaker**: if Docling crashes (segfault, hang past timeout) on five documents in a session, mark Docling unhealthy and route everything to the VLM until restart.
- **Disk circuit breaker**: if vault writes start failing with `ENOSPC`, halt ingestion immediately. Don't half-write Markdown.

### Security

Memex is local, but local is not safe by default.

- **File validation beyond extension**: magic number checks for every input. PDFs verified for `%PDF` header. Office documents inspected for macro flags, with macros stripped by default and surfaced as warnings.
- **Sandboxed parsing**: Docling and the VLM run in a separate process with no network access. If you're running on Linux with seccomp/landlock available, use it. The parsing process should not be able to make a socket connection — full stop.
- **No secret management**: there are no API keys to manage because there is no API to call. If a plugin needs a key (e.g., an external citation database lookup), it's the plugin's problem, and the plugin must declare its network requirements.
- **Vault permissions**: the vault directory is created with `0700`. The `.memex/` subdirectory inherits.
- **No telemetry**: no metrics, no error reports, no usage analytics leave the machine. Period. If we want crash reports, they're stored locally and the user can choose to attach them to a GitHub issue manually.

### Performance on the reference rig

Benchmark targets, measured on the reference RTX 4070 + i7 + 32GB RAM, with `scripts/benchmark.py`:

| Operation | Target | Floor |
|---|---|---|
| 100-page text-heavy PDF, end-to-end ingest | < 4 min | < 6 min |
| 100-page scan-heavy PDF, end-to-end ingest | < 12 min | < 20 min |
| First-token latency on query | < 2 s | < 4 s |
| Full grounded answer (500-doc vault) | < 15 s | < 30 s |
| Cold start (load all resident models) | < 30 s | < 60 s |
| Embedding throughput (batched) | > 500 chunks/sec | > 200 chunks/sec |

CI runs the benchmark on a representative subset; any regression > 15% on a target fails the build. Full benchmarks run nightly.

---

## Part VII — Practices

### Documentation as code

- **User docs** live in `docs/user/` as Markdown, rendered by mkdocs-material. Every CLI command has a docs page generated from typer's introspection.
- **Architectural Decision Records** live in `docs/adrs/` numbered sequentially. Format: context, decision, consequences, alternatives. Every non-trivial decision earns an ADR. When you find yourself in a PR conversation longer than 20 messages about why-we-did-this-thing, write the ADR.
- **Inline docs**: every public function has a docstring with at least one example. Internal functions get docstrings when the *why* isn't obvious from the code.

### Testing strategy

The four layers from §3 plus one more for the parse layer:

**Document corpus testing.** A versioned `tests/fixtures/eval-corpus/` contains real documents (in copyright-cleared categories: open-access papers, public-domain books, sample technical docs) with ground-truth Markdown extraction. Parser changes are scored against this corpus on character accuracy, word accuracy, and structural fidelity (tables-as-tables, equations-as-LaTeX, heading hierarchy intact).

Document categories with separate accuracy thresholds:

- Modern printed text: ≥ 98% character, ≥ 95% word
- Scientific papers (equations, tables): ≥ 95% character, ≥ 90% structure
- Historical / degraded scans: ≥ 85% character
- Handwritten: ≥ 75% character (and we tell the user)

Property-based tests via hypothesis for parsers: feed random valid PDFs, assert invariants (output is valid Markdown, all input pages accounted for, no exceptions on malformed-but-valid PDFs).

### Privacy in telemetry

There is no telemetry. We measure Memex's success the slow, expensive way: by talking to users.

If we ever introduce telemetry, it will be:

- **Off by default** with explicit opt-in.
- **Aggregated locally** before any potential transmission.
- **Content-free** — counts, latencies, error types. Never queries, never document content, never filenames.
- **Auditable** — the user can see exactly what would be sent before opting in.

But the default plan is: no telemetry, ever. The privacy guarantee is meaningless with an asterisk.

### Contribution workflow

- Conventional commits (`feat:`, `fix:`, `docs:`, etc.).
- PRs require: passing CI, one review, an updated ADR for non-trivial changes, an eval-delta report for changes to prompts, models, or agents.
- No squash merges for substantive features — preserve the history; rebase if it's messy.
- Performance regressions > 15% require an ADR explaining the trade.

---

## Closing

These guidelines exist to operationalize the vision, not to constrain it. When a guideline gets in the way of building the right thing, propose the change in an ADR. When you discover a pattern that works better than what's documented here, propose the change in an ADR. This document evolves alongside the code.

The hard part of Memex is not any single component. It's making forty open-source components cooperate, on consumer hardware, fast enough to be useful, reliably enough to be trusted, and observably enough to be debugged when things go wrong. The discipline above is the difference between a system that does that and a system that almost does that.

Build accordingly.
