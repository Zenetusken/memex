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
webui/ → agents/ + vault/ + core/    (+ index/graph_store as a test seam)
cli/ → everything
daemon/ → core/                       (vLLM lifecycle is config-driven)
```

A module **may not** import a private symbol (`_foo`) from another module. If something needs to be shared, it gets a public name.

The single deliberate exception is `webui/app.py` re-exporting `GraphStore` from `memex.index.graph_store` at module top — `tests/integration/test_webui.py` monkeypatches `memex.webui.app.GraphStore.open`, which only works if `GraphStore` lives at the module level. The route handler runs the open lazily and falls back gracefully when ryugraph isn't installed.

## Type discipline

- **No untyped `Any`.** If you need it, add a one-line comment explaining why.
- **Pydantic models cross module boundaries.** Dicts and tuples are module-internal only. Shared types live in `core/types.py`.
- **Node returns use `TypedDict`.** See `agents/answering.py:AnswerStateUpdate` — `total=False` catches state-key typos at type-check time.
- **`complete_structured` is generic over schema.** `tuple[T, int]` where `T = TypeVar("T", bound=BaseModel)`. Don't break that.
- **Bound LLM-emit `str` fields with `max_length=N`.** xgrammar enforces the bound at the JSON-schema grammar level, so the model cannot emit a runaway string that would trip `max_tokens` mid-emission and crash JSON validation. Established by the v6 fix to `SufficiencyAssessment.reason` (was unbounded → 2000-char ramble on counterfactual queries → eval crash) and extended to `DraftAnswer.summary`, `CitedClaim.claim`, `VerificationResult.ungrounded_reasons` items, `ExtractedEntity.name`, `ExtractedEntity.span_text`. Also cap `list[T]` LLM-emit fields with `max_length=N` (e.g., `DraftAnswer.claims` at 20). When adding a new structured-output schema, audit every `str` and `list[...]` field for an explicit bound.
- **Chart-OCR output is LLM-readable markdown, not raw LaTeX.** Nemotron-Parse-v1.2 (the default chart-OCR backend) emits `\begin{tabular}{cc} **On Time 22** & **Late 8**\\ \end{tabular}` which the assess/answer LLMs cannot parse as structured data. The v7 fix (`chart_ocr_backend._normalize_latex_tabulars` + `_split_label_number_cells`) post-processes Nemotron output into markdown tables (multi-row) or key:value bullets (single-row chart summaries where each cell matches `<label> <number>`). When extending the chart-OCR backend with new models, keep emission in LLM-native markdown — pipe-tables, bullet lists, prose sentences. Raw LaTeX in chart-extracted blocks is a regression.
- **Chart-extracted blocks are visible to assess/answer/verify** (P3.3 v7, 2026-05-23). Earlier defenses (v3/v5/v6) stripped `[chart-extracted]...[/chart-extracted]` blocks from the chunks before rendering the assess/answer/verify prompts because raw LaTeX (~500+ chars per block) ate the `truncate(1200)/truncate(1800)` budget. With v7's compact markdown emission (~50-150 chars per block), strips are no longer needed and would actively block chart-content answering. Do not re-introduce strips in `agents/answering.py:assess`, `answer`, or `verify` — chart blocks should flow through to the LLM. The FTS-side strip in `index/fts_store.py` (P3.3 v3 defense) is retained: BM25 keyword search on chart-block content adds noise without helping retrieval recall.
- **Chunker treats `# H1` lines INSIDE `[chart-extracted]` blocks as inert** (P3.3 v7 chunker fix, 2026-05-23). Nemotron-Parse emits H1 prefixes for chart-figure-section labels (e.g. `# Minimal lead-time`, `# Inter-operability`) which look like document headings to a naive regex. `core/text.py::chart_extracted_spans` + `is_inside_any_span` are the helpers; `index/chunker.py::_split_into_sections` and `_heading_path_at` filter heading matches inside those spans. The fix recovered slide-decks Q31 ("4 design principles") + Q18 ("NVLink C2C 900 GB/s" via chunker reflow) + annual-report Q03 ("Data Center revenue $193.737B"). When extending chart-OCR or the chunker, preserve this contract — pinned by `tests/unit/test_chunker_chart_aware.py::test_h1_inside_chart_block_does_not_split_section`.
- **The verify node filters phantom claim-indices** (P3.3 v7 verify-fix, 2026-05-23). The verifier LLM can emit indices > n-1 when distracted by chunk content not in the draft (xgrammar `max_length=n` bounds list length but not index values). `agents/answering.py:verify` filters to `[0, n-1]` and conservatively treats indices missing from BOTH `grounded` and `ungrounded` as ungrounded. Pinned by `tests/integration/test_answering_with_fakes.py::test_verify_filters_phantom_indices` + `test_verify_missing_index_treated_as_ungrounded`.
- **The answer prompt treats markdown-table-row reading as literal** (P3.3 v7 Q09 fix, 2026-05-23). The answer/v3 prompt's "Tables and structured data are literal" rule clarifies that reading a row from a markdown table or a key-value bullet inside a `[chart-extracted]` block counts as a literal claim, not an inferential one. This mirrors `verify_grounding/v2`'s pre-existing rule and was added when chart-types Q09 (Gantt assignees) was false-refusing — the answer LLM was over-applying the No-substitute rule to clearly structured tables.

## Errors

- Every raise uses a `MemexError` subclass from `core/errors.py`. Subclasses today: `InsufficientVRAMError`, `ParseConfidenceTooLow`, `RetrievalEmpty`, `VaultIntegrityError`, `IndexOutOfSync`, `ConfigurationError`, `StaleDocumentError`, `AnswerStateInvariantError`, `ModelCallError`. Module-local subclasses are fine as long as they inherit from `MemexError` — current ones: `CircuitBreakerOpen` (core/breakers.py); `ModelNotConfigured` (models/registry.py); `VLMUnavailable`, `PDFRenderError` (parse/vlm_backend.py); `ChartOCRUnavailable`, `PDFFigureRenderError` (parse/chart_ocr_backend.py); `DoclingUnavailable`, `DoclingTimeout`, `DoclingCrashed`, `SandboxLoadFailed` (parse/docling_backend.py); `PyMuPDFUnavailable`, `PyMuPDFTimeout`, `PyMuPDFCrashed` (parse/pymupdf_backend.py); `DaemonAlreadyRunning`, `DaemonStartTimeout` (daemon/supervisor.py).
- Every raise carries a `context: dict` with the values a user needs to act on the error.
- Recoverable errors set `recoverable=True` so the pipeline knows to resume on retry.
- **Narrow `except` clauses.** Never `except BaseException` (would swallow `CancelledError`, `KeyboardInterrupt`, `SystemExit`); never `except Exception` unless you re-raise the cancellation cases above explicitly. When wrapping a third-party SDK exception, check the module via `type(e).__module__.startswith(...)` rather than catching the broad base class.

## Async / sync

- **Async**: filesystem I/O, model calls, database operations, HTTP, MCP transport.
- **Sync**: token parsing, formatting, chunker, pure transformations.
- CPU-heavy sync work called from async code goes through `asyncio.to_thread` (`models/registry.py` does this for sentence-transformers `encode`; `index/fts_store.py` for sync SQLite).
- **Per-resource locking** is `asyncio.Lock` for stores serving async callers (`index/fts_store.py::FTSStore._lock` gates multi-statement writes; `vault/store.py::_DOC_LOCKS` gates per-doc write paths) and `threading.Lock` for state that crosses both sync and async callers (`agents/answering.py::_COMPILED_GRAPH_LOCK` gates the lazy graph compile). Don't mix the two — pick by who the callers are. The `models/registry.py::_locks` per-model dict + the breaker's lock are the canonical patterns to copy.

## Logging

structlog. Bind once per run scope at the entry point (`bind_run_context(correlation_id, ...)`), bind node/function scope locally (`log = logger.bind(node="rerank")`), emit `log.info("event", k=v)`. **Never** `extra={...}` (lost), **never** f-strings in event names (kills aggregation in Langfuse).

## CUDA

ADR-0006 settles it. The summary:

- **bf16** is the dtype across the stack on Ada. Embedder, reranker, VLM, and chart-OCR all use explicit `torch_dtype=torch.bfloat16`. (FP16 is forbidden for EmbeddingGemma — its activations don't support it.)
- **FA2** for the VLM (`attn_implementation="flash_attention_2"`). FA3 is unavailable on sm_89.
- **`device_map={"": "cuda:0"}`** — deterministic single-GPU placement, not `"auto"`.
- **`AutoModelForImageTextToText`** for the VLM, not `AutoModelForCausalLM`.
- **`cli/bootstrap.py:_configure_cuda`** runs once at startup: CUDA assert + TF32 + `cudnn.benchmark` + `set_per_process_memory_fraction`. Tests bypass it.

## Testing

- **Unit tests** under `tests/unit/` — pure functions, no models.
- **Integration tests** under `tests/integration/` — real tmp vaults, faked heavy I/O (vLLM, Langfuse, LanceDB, sentence-transformers, Docling). Regression nets today: `test_answering_with_fakes.py`, `test_parse_then_index_then_ask.py`, `test_enrich_and_graph.py`, `test_mcp_server.py`, `test_webui.py`, `test_v1x_backlog.py`, `test_graph_expansion.py`, `test_partial_reindex.py`, `test_sandbox.py`, `test_vault_cross_process_lock.py`, `test_vector_store_concurrent.py`, `test_fts_store_concurrent.py`, `test_chart_block_retrieval_isolation.py`. **Unit-test additions for P3.3-v7** (2026-05-23): `test_chart_ocr_backend.py` gained 7 tests on `_latex_tabular_to_markdown` + `_normalize_latex_tabulars` + `_split_label_number_cells`; `test_force_docling.py` pins the `--force-docling` flag contract; `test_chunker_chart_aware.py` gained 3 tests pinning the chart-block-H1 filter; `test_answering_with_fakes.py` gained 2 tests pinning the phantom-index filter. **Today: 255 passing** + skips in environments without privileged seccomp / specific test deps.
- Tests don't call `bootstrap()` — they construct `MemexSettings()` directly + `set_settings()` + the `monkeypatch` of `MEMEX_OBSERVABILITY__LANGFUSE_ENABLED=false`. CUDA assert is bypassed.
- **Fakes accept forward-compatible kwargs.** When you write a stand-in for `complete_structured` (or any other function with optional kwargs), terminate the signature with `**_kw: object` so adding a new kwarg upstream doesn't silently break the test. Same for `docling_convert` and other monkeypatch targets — the kwarg-signature bug from 2026-05-20 was the canonical example.
- **Patch via `monkeypatch.setattr`**, never direct module-attribute mutation. The latter leaks into subsequent tests in the same session.

## When adding a new module

1. Check the architectural blueprint — module signatures are already designed in `docs/IMPLEMENTATION-PLAN.md` §1.
2. Update `src/memex/<module>/__init__.py` to re-export the public surface.
3. Add an integration test under `tests/integration/`. Fake every external (`monkeypatch.setattr`).
4. Wire to the CLI under `cli/commands.py` if user-facing.
5. Update `docs/ROADMAP.md` if the module ships a phase deliverable.
