# Memex

> Local-first, fully agentic system for turning documents into a queryable Markdown knowledge vault.

Memex parses your documents, builds a Markdown vault as the source of truth, and exposes an agentic question-answering pipeline over them — entirely on your machine, with no network call you didn't make.

- **Vision**: [`docs/VISION.md`](docs/VISION.md)
- **Developer guidelines**: [`docs/GUIDELINES.md`](docs/GUIDELINES.md)
- **Architectural decisions**: [`docs/adr/`](docs/adr/)
- **Eval corpus plan**: [`docs/eval-corpus-plan.md`](docs/eval-corpus-plan.md)

## Status

Phases 0–2 shipped: ingest, parse (Docling + VLM fallback), enrich (entity extraction + graph), index (LanceDB + SQLite FTS5 + RyuGraph), retrieve, answering agent with structured outputs + Langfuse traces, file watcher, eval harness. Next: Phase 3 — MCP server + local web UI. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the operational status.

## Prerequisites

Memex requires an NVIDIA GPU per ADR-0001. The reference rig is an RTX 4070 (12 GB VRAM, Ada Lovelace, sm_89) with 32 GB system RAM.

Software floor:

- **NVIDIA driver R570+** (paired with CUDA 12.8 toolkit)
- **Python 3.12+**
- [**uv**](https://docs.astral.sh/uv/) — sole package manager

The `[models]` extra installs the cu128 PyTorch wheel via the explicit `pytorch-cu128` index declared in `pyproject.toml`. The default PyPI wheel is CPU-only on Linux — relying on it produces a silently broken install where every GPU path runs on CPU. ADR-0006 covers the full CUDA dispatch + dtype policy.

The `[parse]` extra adds `flash-attn>=2.6` (Linux only; prebuilt wheels available, building from source needs a matching CUDA toolkit). Flash-Attention 3 is **not** supported on Ada Lovelace — Memex uses FA2.

The `[parse]` extra also adds `pyseccomp>=0.1` on Linux. The Docling worker uses it to install a seccomp-bpf filter that blocks every network syscall before importing docling — so a malicious document can't make Docling phone home. This implies the worker can't download models on demand: pre-fetch with `huggingface-cli download <model_id>` once and the cache reads stay local. Set `parse.docling_sandbox_network=false` if your deployment genuinely needs network during parse. macOS / Windows users get a graceful no-op (seccomp is Linux-only). See GUIDELINES.md Part VI for the full policy.

## Getting started

```sh
# Install with the GPU + parser extras
uv sync --extra models --extra parse

# Smoke-test CUDA resolution
uv run python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"

# Configure vault path + Langfuse keys (or disable Langfuse)
export MEMEX_VAULT_PATH=~/memex-vault
export MEMEX_OBSERVABILITY__LANGFUSE_ENABLED=false  # or set the keys

# Launch the vLLM orchestrator in another shell — flags codified in:
./scripts/serve-vllm.sh

# Then drive Memex
uv run memex --help
uv run memex daemon status        # probes the configured vLLM endpoint
uv run memex ingest path/to/paper.pdf
uv run memex ask "What does Smith 2024 argue?"
```

If `daemon status` reports `reachable: false`, check that `MEMEX_INFERENCE__BASE_URL` points at your running vLLM (default `http://localhost:8000/v1`) and that `serve-vllm.sh` finished loading.

If bootstrap raises `CUDA unavailable; Memex requires an NVIDIA GPU`, the cu128 torch wheel didn't resolve. Re-run `uv sync --extra models` and verify the lock file pinned `torch+cu128`.

## Documentation map

| Where | What |
|---|---|
| [`docs/VISION.md`](docs/VISION.md) | The "why" — the long-form vision |
| [`docs/GUIDELINES.md`](docs/GUIDELINES.md) | The "how" — engineering practices, stack, architecture rules |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | The "what now" — operational status, queued work |
| [`docs/IMPLEMENTATION-PLAN.md`](docs/IMPLEMENTATION-PLAN.md) | The architectural blueprint — module interfaces, build order |
| [`docs/eval-corpus-plan.md`](docs/eval-corpus-plan.md) | Eval corpus design + scoring rubric |
| [`docs/adr/`](docs/adr/) | Architecture Decision Records (`0001` vLLM → `0006` CUDA dispatch) |

Browse the same content as a navigable site with `uv sync --extra docs && uv run mkdocs serve` (mkdocs-material, dark slate palette, no Google Fonts — fully offline).

## Performance regression gates

Pure-Python orchestration benchmarks (chunker throughput, vault write latency, FTS5 query latency, agent state-machine cycle) run on every PR via the `.github/workflows/benchmark.yml` workflow:

```sh
uv run python scripts/benchmark.py                                  # JSON to stdout
uv run python scripts/benchmark.py --output current.json
uv run python scripts/benchmark.py --gate tests/benchmarks/baseline.json --output current.json
```

`--gate` exits non-zero on any metric regressing more than 15% from the committed baseline. `--real` adds GPU-dependent benchmarks (cold start, embedding throughput, first-token latency) — run those on the reference rig. See [`tests/benchmarks/README.md`](tests/benchmarks/README.md) for the workflow.

## License

Apache-2.0.
