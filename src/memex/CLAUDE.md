# `src/memex/` — Backend conventions

Python 3.12+, pyright `--strict` against `src/memex/`, ruff for lint + format. The full engineering manual is [`docs/GUIDELINES.md`](../../docs/GUIDELINES.md); this is the cheat-sheet.

## Module boundaries

`core/` has no Memex deps — everything else may import from it. Import direction otherwise:

```
agents/ → models/ + observability/ + prompts/ + retrieve/ + core/
retrieve/ → index/ + models/ + core/
index/ → models/ + vault/ + core/
parse/ → models/ + vault/ + core/ + observability/
enrich/ → models/ + index/ + prompts/ + vault/ + core/
ingest/ → vault/ + core/
mcp/ → agents/ + retrieve/ + vault/ + core/
webui/ → agents/ + vault/ + core/
cli/ → everything
```

A module **may not** import a private symbol (`_foo`) from another module. If something needs to be shared, it gets a public name.

## Type discipline

- **No untyped `Any`.** If you need it, add a one-line comment explaining why.
- **Pydantic models cross module boundaries.** Dicts and tuples are module-internal only. Shared types live in `core/types.py`.
- **Node returns use `TypedDict`.** See `agents/answering.py:AnswerStateUpdate` — `total=False` catches state-key typos at type-check time.
- **`complete_structured` is generic over schema.** `tuple[T, int]` where `T = TypeVar("T", bound=BaseModel)`. Don't break that.

## Errors

- Every raise uses a `MemexError` subclass from `core/errors.py`. Subclasses today: `InsufficientVRAMError`, `ParseConfidenceTooLow`, `RetrievalEmpty`, `VaultIntegrityError`, `IndexOutOfSync`, `ConfigurationError`, `AnswerStateInvariantError`, `ModelCallError`.
- Every raise carries a `context: dict` with the values a user needs to act on the error.
- Recoverable errors set `recoverable=True` so the pipeline knows to resume on retry.

## Async / sync

- **Async**: filesystem I/O, model calls, database operations, HTTP, MCP transport.
- **Sync**: token parsing, formatting, chunker, pure transformations.
- CPU-heavy sync work called from async code goes through `asyncio.to_thread` (`models/registry.py` does this for sentence-transformers `encode`).

## Logging

structlog. Bind once per run scope at the entry point (`bind_run_context(correlation_id, ...)`), bind node/function scope locally (`log = logger.bind(node="rerank")`), emit `log.info("event", k=v)`. **Never** `extra={...}` (lost), **never** f-strings in event names (kills aggregation in Langfuse).

## CUDA

ADR-0006 settles it. The summary:

- **bf16** is the dtype across the stack on Ada. Embedder, reranker, VLM all explicit `torch_dtype=torch.bfloat16`. (FP16 is forbidden for EmbeddingGemma — its activations don't support it.)
- **FA2** for the VLM (`attn_implementation="flash_attention_2"`). FA3 is unavailable on sm_89.
- **`device_map={"": "cuda:0"}`** — deterministic single-GPU placement, not `"auto"`.
- **`AutoModelForImageTextToText`** for the VLM, not `AutoModelForCausalLM`.
- **`cli/bootstrap.py:_configure_cuda`** runs once at startup: CUDA assert + TF32 + `cudnn.benchmark` + `set_per_process_memory_fraction`. Tests bypass it.

## Testing

- **Unit tests** under `tests/unit/` — pure functions, no models.
- **Integration tests** under `tests/integration/` — real tmp vaults, faked heavy I/O (vLLM, Langfuse, LanceDB, sentence-transformers, Docling). Three regression nets so far: `test_answering_with_fakes.py`, `test_parse_then_index_then_ask.py`, `test_enrich_and_graph.py`, plus Phase 3's `test_mcp_server.py` and `test_webui.py`.
- Tests don't call `bootstrap()` — they construct `MemexSettings()` directly + `set_settings()` + the `monkeypatch` of `MEMEX_OBSERVABILITY__LANGFUSE_ENABLED=false`. CUDA assert is bypassed.

## When adding a new module

1. Check the architectural blueprint — module signatures are already designed in `docs/IMPLEMENTATION-PLAN.md` §1.
2. Update `src/memex/<module>/__init__.py` to re-export the public surface.
3. Add an integration test under `tests/integration/`. Fake every external (`monkeypatch.setattr`).
4. Wire to the CLI under `cli/commands.py` if user-facing.
5. Update `docs/ROADMAP.md` if the module ships a phase deliverable.
