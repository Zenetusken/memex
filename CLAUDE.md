# Memex — Project conventions

Memex is a local-first, fully agentic document-understanding system. Runs entirely offline on a single consumer GPU (reference: RTX 4070, 12 GB VRAM). The vault on disk (`vault/documents/*.md`) is the source of truth; everything else (LanceDB, FTS5, RyuGraph) is derived state.

## Where things live

| Concern | Read |
|---|---|
| Vision + principles | [`docs/VISION.md`](docs/VISION.md) |
| Engineering practices (full) | [`docs/GUIDELINES.md`](docs/GUIDELINES.md) |
| Operational status | [`docs/ROADMAP.md`](docs/ROADMAP.md) |
| Architectural blueprint | [`docs/IMPLEMENTATION-PLAN.md`](docs/IMPLEMENTATION-PLAN.md) |
| ADRs (`0001`–`0006`) | [`docs/adr/`](docs/adr/) |
| Eval corpus design | [`docs/eval-corpus-plan.md`](docs/eval-corpus-plan.md) |
| Python source | [`src/memex/`](src/memex/) — see [`src/memex/CLAUDE.md`](src/memex/CLAUDE.md) for backend rules |
| Web UI | [`src/memex/webui/`](src/memex/webui/) — see [`src/memex/webui/CLAUDE.md`](src/memex/webui/CLAUDE.md) for frontend rules |

## Five principles (from `docs/VISION.md`)

1. **Local-first, by construction** — no remote endpoints; the air-gap test passes.
2. **Markdown as source of truth** — `vault/.memex/` is regenerable derived state (ADR-0003).
3. **Small models, used well** — single 12 GB GPU is the target; discipline beats parameter count.
4. **Observable at every layer** — one ULID `correlation_id` threads structlog + Langfuse (ADR-0004).
5. **Composable, not captive** — MCP server + plain Markdown are the public surfaces.

## Universal rules

- **Pydantic models cross module boundaries, dicts don't.** If two modules share a payload, it's a pydantic model in `core/types.py`. The agent's `dict[str, Any]` returns are a `TypedDict` (`AnswerStateUpdate`).
- **Errors are typed and carry context.** Every raised exception is a `MemexError` subclass with `context: dict`. No bare `RuntimeError`s in production paths.
- **Async-first for I/O, sync for CPU.** Filesystem reads, model calls, database writes — `async`. Token parsing, formatting, string manipulation — `sync`. Don't mix without a reason.
- **Configuration is centralized.** `MemexSettings` (`core/config.py`) is the single source. Loaded once at startup from `~/.config/memex/config.toml` + `MEMEX_*` env vars. Validated immediately.
- **Logs use structlog with bound context.** `logger.bind(node="...").info("event", k=v)`. No `extra={...}`, no f-strings in event names (ADR-0004).
- **vLLM is the sole inference engine for v1** (ADR-0001). No CPU fallback. The CUDA toolkit and dtype dispatch are settled in ADR-0006.

## When in doubt

- Read the relevant ADR first — they explain *why*.
- Check `docs/ROADMAP.md` to see if your idea is already queued or deliberately deferred.
- Memory under `/home/drei/.claude/projects/-home-drei-project-Doc-Flo/memory/` captures time-sensitive context. Active trackers: `MEMORY.md` (the index), `next_priorities.md` (current pickup list), `p33_tracker.md` (P3.3 chart-OCR — fully closed 2026-05-23 with v7 arc: chart-OCR LaTeX→md converter + verify-phantom-filter + table-literal answer-prompt + chunker chart-block-H1 filter; 7 chart-content REF→ANS flips across 3 corpora), `build_status.md` (last phase-wrap snapshot), `table_sql_phase2.md` (Table-RAG Phase 2 text-to-SQL — SHIPPED + GPU-accepted 2026-05-24: structured `tables.sqlite` + `query_tables` node, synthetic-chunk grounding, row-vs-aggregate fabrication boundary + recompute gate, 3 GPU-caught bugs), `table_rag_research_2026_05_24.md` (Phase 1 linearization + the 3-subagent research verdict that scoped both phases), `contextual_retrieval_negative_2026_05_25.md` (NEGATIVE result — Anthropic LLM-context-prefix on the embedding input broke the refusal HARD gate + regressed retrieval on the 300M mean-pooled embedder; built+validated then reverted; do NOT retry — use EmbeddingGemma native prompts instead, which SHIPPED), `gemma4_embedder_research_2026_05_25.md` (NO Gemma-4 embedder exists; embedder swap = low-leverage, roadmapped P2.5). Reference history: `vault_heading_migration_2026_05_24.md` (heading-quality arc — PyMuPDF + Docling level-recovery + mis-detection reclassification + empirically-gated vault migration + the table-chunk enrich-failure fix; ops notes for re-parse/re-enrich + rerank-OOM), `retitle_and_cleanup_2026_05_24.md` (frictionless retitle + ruff 139→0 + pyright-strict 507→0 + CrossEncoder bug fix + bf16-reranker eval re-baseline), `p41_wikilinks.md` (P4.1 + course-ref resolver), `stack_currency_audit.md` (2026-05-19), `cuda_audit.md` (2026-05-20), `audit_2026_05_20.md` (multi-agent bug-hunt), `p42_tracker.md` (closed smaller-orchestrator A/B), `p42_plan.md` (closed), `pymupdf_prefilter_ship.md`.
