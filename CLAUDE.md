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
- Memory under `/home/drei/.claude/projects/-home-drei-project-Doc-Flo/memory/` captures time-sensitive context: stack-currency audit 2026-05-19, CUDA audit 2026-05-20, multi-agent bug-hunt audit 2026-05-20 (`audit_2026_05_20.md`), and current build status (`build_status.md`).
