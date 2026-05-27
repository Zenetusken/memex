# Memex

> Local-first, fully agentic system for turning documents into a queryable Markdown knowledge vault.

Memex parses your documents, builds a Markdown vault as the source of truth, and exposes an agentic question-answering pipeline over them — entirely on your machine, with no network call you didn't make.

## Five principles

1. **Local-first by construction** — no remote endpoints; the air-gap test passes.
2. **Markdown as source of truth** — `vault/.memex/` is regenerable derived state (ADR-0003).
3. **Small models, used well** — single 12 GB GPU is the target.
4. **Observable at every layer** — one ULID `correlation_id` threads structlog + Langfuse (ADR-0004).
5. **Composable, not captive** — MCP server + plain Markdown are the public surfaces.

## Quick start

Requires NVIDIA driver R570+, CUDA 12.8 toolkit, Python 3.12+, and [`uv`](https://docs.astral.sh/uv/). Reference hardware is an RTX 4070 (12 GB VRAM, Ada Lovelace, sm_89). See the README for the full prerequisites.

```sh
uv sync --extra models --extra parse
./scripts/serve-vllm.sh                 # in another shell
uv run memex daemon status              # verify vLLM is reachable
uv run memex ingest path/to/paper.pdf   # ingest + parse + index
uv run memex ask "What does Smith argue about reflexivity?"
```

## Read next

- **[Vision](VISION.md)** — the long-form *why*, the stack, the principles
- **[Engineering guidelines](GUIDELINES.md)** — patterns, conventions, the discipline manual
- **[Roadmap](ROADMAP.md)** — what's shipped (Phases 0–4 ✓), what's queued
- **[Implementation plan](IMPLEMENTATION-PLAN.md)** — module-by-module architecture
- **[Eval corpus plan](eval-corpus-plan.md)** — how quality is measured

## Architecture decisions

The full set of ADRs lives under [`adr/`](adr/0001-vllm-as-sole-inference-engine.md). Each one is a single decision, with context, drivers, alternatives, consequences, and a revisit-when list. Current set:

- **[ADR-0001](adr/0001-vllm-as-sole-inference-engine.md)** — vLLM as the sole inference engine
- **[ADR-0002](adr/0002-single-package-over-monorepo.md)** — Single Python package over a monorepo
- **[ADR-0003](adr/0003-markdown-vault-as-source-of-truth.md)** — Markdown vault is the source of truth
- **[ADR-0004](adr/0004-observability-structlog-langfuse.md)** — structlog + Langfuse, joined by correlation_id
- **[ADR-0005](adr/0005-ryugraph-replaces-kuzu.md)** — RyuGraph replaces Kuzu after upstream archival
- **[ADR-0006](adr/0006-cuda-dispatch-and-dtype.md)** — CUDA toolkit pin, dtype dispatch, attention implementation
- **[ADR-0007](adr/0007-co-residence-resource-modes.md)** — Co-residence resource modes (the VRAM-tradeoff policy layer; dynamic-VRAM-manager precursor)
- **[ADR-0008](adr/0008-document-summarization.md)** — Document summarization (doc-type-aware, structured, grounded map-reduce)

## Building this site locally

```sh
uv sync --extra docs
uv run mkdocs serve   # http://localhost:8000
```

`uv run mkdocs build` produces a static site under `site/` — drop that under any static-file server, an air-gapped reader, or a USB stick. The site is fully self-contained (no Google Fonts, no analytics).

## License

Apache-2.0.
