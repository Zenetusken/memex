# Memex Roadmap

**Last updated:** 2026-05-20

The blueprint in [`IMPLEMENTATION-PLAN.md`](IMPLEMENTATION-PLAN.md) is the architectural design — module signatures, cross-cutting concerns, build order. This document is the **operational view**: what is shipped today, what is in flight, what comes next, and which decisions are still open.

---

## Status at a glance

| Phase | Goal | Status | Notable artefacts |
|---|---|---|---|
| 0 | Answering loop end-to-end on a hand-built vault | ✓ **Done** (2026-05-19) | `core/{bus,errors,events,config,types,manifest,breakers}.py`, `observability/`, `agents/answering.py`, `vault/store.py`, `prompts/loader.py` + 3 templates, `models/client.py`, `retrieve/`, CLI `version`/`ingest --skip-parse`/`index`/`reindex`/`ask`, integration test #1 (6 tests) |
| 1 | Real documents in, queryable out | ✓ **Done** (2026-05-19) | `ingest/{validation,pipeline}.py` (magic-numbers + size + macros), `parse/{docling_backend,vlm_backend,pipeline}.py` (Docling + VLM-routing skeleton + crash breaker), OOM circuit breaker in registry, manifest IngestStage/ParseStage/IndexStage, CLI `parse` + `daemon stub`, integration test #2 (6 tests) |
| 2 | The vault gets smarter | ✓ **Done** (2026-05-20) | `enrich/{entities,pipeline}.py` (LLM entity extraction + dedupe + graph upsert), `index/graph_store.py` + RyuGraph schema (ADR-0005), real VLM path (`vlm_backend.py` via pypdfium2 + transformers), `ingest/watcher.py` (watchdog + 500ms debounce + sha-compare), `eval/{runner,scoring}.py` (CER/WER/F1/citation-precision skeleton), `extract_entities/v1.md` prompt, CLI `enrich`/`graph`/`doctor`/`watch`/`eval`, integration test #3 (7 tests) |
| 3 | External interfaces | ✓ **Done** (2026-05-20) | `mcp/server.py` (FastMCP with `search`/`ask`/`get_document`/`list_documents` tools, stdio + streamable-HTTP transports), `webui/app.py` (FastAPI + HTMX, 5 routes, server-rendered Jinja templates), `webui/templates/` (base, index, _answer fragment, documents, document), `webui/static/style.css`, CLI `search`/`serve mcp`/`serve web`, integration tests #4 (6 tests) and #5 (8 tests) |
| 4 | Production hardening | ◐ **Mostly done** | MCP `get_graph_neighbors` tool, web UI graph viz (Cytoscape + grid-paper canvas + inspector), PDF side-by-side document view, annotation correction (`/edit` + `/body` + `/review` with watcher-race-safe manifest sync), `scripts/benchmark.py` with `--gate` regression detection, GitHub Actions workflow, mkdocs-material site with offline-only fonts. Outstanding: full 125-doc eval corpus (separate multi-week roadmap), CI nightly real-mode benchmark run (needs reference rig). |

**Total**: 61 Python files in `src/memex/` + tests, all parse-clean. Five ADRs (0001 vLLM, 0002 single-package, 0003 Markdown-vault, 0004 observability, 0005 RyuGraph). Three integration tests + chunker unit tests.

---

## Phase 0 — Answering loop end-to-end ✓

Goal met: `memex ask "…"` returns a `FinalResponse` against a hand-built markdown vault with a Langfuse trace.

Shipped:

- **`core/bus.py`** — asyncio pubsub + optional SQLite persistence + 30-day prune
- **`core/{errors,events,config,types,manifest,breakers}.py`** — full primitives layer
- **`observability/{logging,tracing}.py`** — structlog + Langfuse v4 SDK (ADR-0004)
- **`agents/answering.py`** — LangGraph state machine with mandatory verify, budgeted regenerate, refuse-first-class
- **`vault/store.py`** — atomic writes, frontmatter round-trip, doc-id derivation, `make_ref` public helper
- **`prompts/loader.py`** + `assess_sufficiency/v1.md`, `answer/v1.md`, `verify_grounding/v1.md` — versioned templates, StrictUndefined-on-render
- **`models/client.py`** — generic over schema (`tuple[T, int]`), `langfuse.openai` wrapper, OpenAI-standard `response_format={"type":"json_schema"}`
- **`models/registry.py`** — `use(name)` async context, lazy embedder/reranker via sentence-transformers
- **`index/{chunker,vector_store,fts_store,pipeline}.py`** — header-aware chunker, LanceDB + SQLite FTS5
- **`retrieve/{hybrid,rerank,fusion}.py`** — RRF over BM25 + dense, on-demand cross-encoder
- **CLI** — `version`, `ingest --skip-parse`, `index`, `reindex`, `ask`

Regression net: `tests/integration/test_answering_with_fakes.py` (6 tests) covering happy path, refuse-on-insufficient, regenerate-then-succeed, empty-retrieval short-circuit, ULID format, python-ulid API.

---

## Phase 1 — Real documents in ✓

Goal met: `memex ingest path/to/papers/ && memex ask "…"` works, with manifests recorded.

Shipped:

- **`ingest/validation.py`** — magic-number check per supported format (PDF `%PDF-`, Office ZIPs refined by inner directory + `vbaProject.bin` for macros, HTML, markdown, plain text). Reject macros unless `ingest.allow_macros=true`.
- **`ingest/pipeline.py`** — `ingest_file(IngestRequest) → IngestResult` (rejections are values, not exceptions), `ingest_directory`, `ingest_markdown_passthrough`. Writes initial `IngestStage` manifest.
- **`parse/docling_backend.py`** — Docling adapter with `asyncio.wait_for(timeout_s)`, lazy import
- **`parse/pipeline.py`** — Markdown passthrough + Docling path under a `CircuitBreaker(name="docling")`; manifest records per-page `engine + confidence + rationale`; crashes are persisted in the manifest (`crashed=True`) so `memex doctor` can surface them
- **OOM circuit breaker in `models/registry.py`** — `_is_oom(exc)` detects CUDA OOM by class name + message; three consecutive trips the registry; `reset_breaker()` for tests/doctor
- **CLI** — real `ingest` chains through ingest → parse → index by default (with `--ingest-only`, `--skip-parse`, `--no-index` overrides); `parse <doc_id>` explicit re-parse; `daemon {start,stop,status}` sub-app (status is real OpenAI-compat probe; start/stop are stubs)

Regression net: `tests/integration/test_parse_then_index_then_ask.py` (6 tests) covering ingest+parse+manifest, full chain, unknown-format rejection, oversize rejection, markdown passthrough, macros rejected.

---

## Phase 2 — The vault gets smarter ✓

Goal met: editing a markdown file triggers re-enrich + re-index via the watcher; `memex graph --document …` prints citation neighbors.

Shipped:

- **`enrich/entities.py`** — `ExtractedEntity`/`EntityList`/`Entity` models, `merge_entities(chunk, raw)` locating span text, `dedupe()` by `(lower(name), kind)` max'ing confidences
- **`enrich/pipeline.py`** — `enrich_document(doc_id)`: re-chunk, extract per-chunk under `asyncio.Semaphore(4)`, dedupe, graph upsert (Document + Entity + MENTIONS), manifest `EnrichStage` write. Wikilink insertion + citation resolution deferred to v1.x (per the plan's "minimum viable for enrich").
- **`index/graph_store.py`** + `schemas/graph.cypher` — RyuGraph wrapper per ADR-0005. Idempotent schema apply, `upsert_document/entity`, `link_mentions/cites`, `delete_document` (DETACH DELETE), `neighbors(doc_id)` for the `graph` CLI. `_open_graph` returns None with a warning if RyuGraph not installed so retrieval still works.
- **`parse/vlm_backend.py`** — real VLM page transcription. `pypdfium2` renders the PDF page to a PIL image at ~200 DPI; `transformers.AutoProcessor` + `AutoModelForCausalLM` (loaded via `ModelRegistry.use("vlm")`) generates clean Markdown under `torch.inference_mode()`.
- **`parse/pipeline.py`** — `_route_and_escalate` replaces the placeholder. Below-threshold pages on PDF sources are escalated to the VLM when `parse.disable_vlm=false`; failed escalations keep Docling's output with a manifest rationale; the document-level markdown is re-stitched from successfully VLM'd pages.
- **`ingest/watcher.py`** — watchdog Observer on `vault/documents/*.md` with per-path 500ms debounce, `_confirm_user_edit` comparing on-disk sha256 to the manifest's `content_sha256` (matching ⇒ Memex's own write, skip), `default_reaction = enrich + index`.
- **`eval/{runner,scoring}.py`** — in-house CER/WER/structural-F1/citation-precision (no ragas/jiwer dep), `run_eval(query_set, quick)` over a JSON corpus, `EvalReport` with mean citation precision + refusal-rate-on-counterfactuals.
- **`prompts/extract_entities/v1.md`** — entity extraction template with `EntityList` output schema.
- **CLI** — `enrich <doc_id>`, `graph --document <doc_id>`, `doctor` (vault integrity + daemon probe + breaker state), `watch` (blocking), `eval <query_set>`.

Regression net: `tests/integration/test_enrich_and_graph.py` (7 tests) covering enrich pipeline, entity dedupe, stable entity_id across case, watcher swallows own-write, watcher detects real edit, eval runner with synthetic queries, CER/WER smoke.

---

## Phase 3 — External interfaces ✓

Goal met: Claude Code (and any MCP-speaking client) can query the vault via the MCP server; a browser at `localhost:7423` does the same via the FastAPI + HTMX UI.

Shipped:

- **`memex.mcp.server`** — FastMCP-based server. Four v1 tools: `search(query, k)`, `ask(question)`, `get_document(doc_id)`, `list_documents()`. Module-level functions are also exported so tests + scripts can call them directly without spinning up a transport. Two transports: `serve_stdio` (desktop) and `serve_http` (network-local, bind localhost-only — no auth model yet, that's Phase 4).
- **`memex.webui.app`** — FastAPI factory returning the configured app. Routes: `GET /`, `POST /ask`, `GET /documents`, `GET /documents/{doc_id}`, `GET /healthz`. HTMX hits `/ask` and swaps the answer fragment in-place; no page reload, no SPA.
- **`memex.webui.templates/`** — Jinja2 base + four pages + the `_answer` HTMX fragment. Refusals render with a "show retrieved chunks" disclosure so the user understands *what the agent saw* (per IMPLEMENTATION-PLAN §5.4).
- **CLI** — `memex search <query>` (hybrid search + rerank, prints chunks), `memex serve mcp [--transport stdio|http]`, `memex serve web [--host H --port N]`. Both `serve` subcommands chain through `bootstrap()` for the full CUDA + Langfuse + registry init.
- **Integration tests** — `test_mcp_server.py` (six tests: search returns serialisable dicts, ask returns grounded response, get_document round-trips, list_documents enumerates all refs, 404 path, registry smoke); `test_webui.py` (eight tests: index renders, healthz, ask answered, ask refused, empty-question 400, document list, document detail, unknown-doc 404).

---

## Phase 4 — Production hardening ◐

Most of the phase shipped 2026-05-20. What landed:

- **MCP `get_graph_neighbors` tool** — fifth tool on the FastMCP server, returns `list[GraphNeighbor]` from the RyuGraph store, graceful fallback to `[]` when ryugraph isn't installed.
- **Web UI graph view** — `GET /graph/{doc_id}` page with a 32 px-grid Cytoscape canvas, faint corner registration marks, concentric layout pinning the center document, dense inspector panel on the right (`dl-row` definition lists, incident-edge enumeration, click-through to neighbors).
- **PDF side-by-side** — `GET /documents/{doc_id}` now detects `vault/documents/{doc_id}/source.*` and renders a `pane-split` two-column layout (source PDF in an iframe + extracted markdown) with independent scrolling. New `GET /documents/{doc_id}/source` serves the file with the correct media-type.
- **Annotation correction** — `GET /documents/{doc_id}/edit` (textarea form), `GET /documents/{doc_id}/body` (cancel target), `POST /documents/{doc_id}/review` (writes via `vault.write_document` + `update_manifest` to keep the watcher's sha-check in sync, returns the view-mode partial with a fading "saved 14:23:01" toast).
- **Benchmark harness with regression gates** — `scripts/benchmark.py` measures chunker throughput, vault write latency, FTS5 query latency, agent state-machine cycle (`--fake`), plus optional cold start / first-token / embedding throughput (`--real`). `--gate BASELINE` exits non-zero on any metric regressing > 15% (configurable), missing from current, or erroring in current. Per-metric warnings between 5% and 15% print to stderr without failing.
- **GitHub Actions workflow** — `.github/workflows/benchmark.yml` runs the fake-mode benchmarks on every PR touching `src/` or the harness, gates against `tests/benchmarks/baseline.json` when present, uploads the report as an artifact.
- **mkdocs-material site** — `mkdocs.yml` with the dark slate palette matching the web UI, `font: false` so the built site is fully offline (local-first applies to docs too), full ADR sub-section nav, `pymdownx.*` extensions. New `docs/index.md` is the landing page. `uv run mkdocs serve` for dev, `uv run mkdocs build` for static output. New `[docs]` extra in pyproject.

Outstanding (separate from Phase 4 because they're multi-week or hardware-dependent):

- **Full eval corpus** — assembling the 125-document, 7-category corpus per `docs/eval-corpus-plan.md` is its own multi-week effort. The `eval/` engine and the scoring rubric are ready; the corpus assembly is gated on a curator.
- **Real-mode benchmark nightly job** — running `--real` against the reference rig requires GPU CI. Workflow template ships; the user adds the runner.
- **Annotation UI hardening** — surface 409 conflict on stale-sha edits (currently last-write-wins), surface validation errors inline.

---

## v1.x backlog — not in any phase

| Item | Status |
|---|---|
| Citation resolution + `[[wikilink]]` insertion in enrich | ✓ **Done** (2026-05-20) |
| `memex daemon start/stop` (real vLLM supervisor) | ✓ **Done** (2026-05-20) |
| Subprocess-sandboxed Docling | ✓ **Done** (2026-05-20) — crash containment; network-egress sandbox (seccomp/landlock) still queued |
| Watcher → event bus integration | ✓ **Done** (2026-05-20) |
| Citation-graph reasoning during answering | ✓ **Done** (2026-05-20) |
| Incremental partial re-indexing | ✓ **Done** (2026-05-20) |
| Network-egress sandbox for Docling (seccomp/landlock) | ✓ **Done** (2026-05-20) |

### Citation resolution + wikilinks (done 2026-05-20)

- New `memex/enrich/citations.py`: `CitationCandidate`/`CitationList` (LLM output), `ResolvedCitation` (post-resolution), `CitationIndex` + `make_signature` (per-doc signatures with title, title-tokens, author-year forms), `resolve_candidate` (three-signal scoring: title substring 0.88–0.95, author-year 0.90, title-token overlap up to 0.78 — composite resolves at ≥ 0.70), `insert_wikilinks` (first-occurrence-only, ≥ 0.85 composite confidence threshold).
- New `prompts/extract_citations/v1.md` template.
- `enrich/pipeline.py` extended: per-chunk LLM call for citations alongside entities (via `asyncio.gather`); citation index built from every *other* vault doc; `MENTIONS` + `CITES` edges both written; wikilinks substituted atomically via `vault.write_document`; manifest's `EnrichStage.citation_count` and `wikilinks_inserted` populated; `content_sha256` recorded in the manifest so the watcher's sha-check doesn't fire on our own write.
- Tests: 6 new cases in `test_enrich_and_graph.py` (author-year resolver, title resolver, below-threshold returns None, wikilink-insertion + confidence-threshold respect, first-occurrence-only enforcement, full enrich pipeline with a real cross-doc citation).

### Daemon supervisor (done 2026-05-20)

- New `memex.daemon` package: `supervisor.start/stop/status` + `DaemonAlreadyRunning`/`DaemonStartTimeout` exceptions + `DaemonStatus` pydantic model.
- `start()` spawns `inference.serve_script` (defaults to `scripts/serve-vllm.sh`) via `subprocess.Popen(..., start_new_session=True)` so the daemon survives the CLI exit; stdout + stderr land in `vault/.memex/daemon/vllm.log`; PID stored in `vault/.memex/daemon/vllm.pid`. Polls the configured `inference.base_url` with `AsyncOpenAI().models.list()` until reachable or `inference.daemon_startup_timeout_s` (default 120 s) fires. On timeout, terminates the child and cleans up the PID file. Refuses to spawn over a live PID (`DaemonAlreadyRunning`); cleans up stale PIDs whose target process is gone.
- `stop()` reads the PID file, `SIGTERM`s the process group, waits up to 10 s for graceful exit, escalates to `SIGKILL`, removes the PID file. Idempotent — no-op when no PID file.
- CLI `memex daemon {start,stop,status}` now real (the Phase-1 stubs are gone). Exit codes: 0 on success, 1 on `DaemonAlreadyRunning` / `DaemonStartTimeout`.
- New `InferenceSettings.serve_script` + `InferenceSettings.daemon_startup_timeout_s` config knobs.

### Watcher → event bus (done 2026-05-20)

- `EventBus` gained sync `start()` / `stop()` methods (the async context manager now delegates to them) so bootstrap can wire the bus without an async context.
- `bootstrap()` constructs `EventBus(db_path=settings.vault_path/".memex"/"events.sqlite")`, calls `start()`, installs via `set_bus()`. The bus's SQLite persistence is derived state per ADR-0003.
- `ingest/watcher.py:_drain_one` now publishes a `vault.edited` `MemexEvent` (with `doc_id`, `markdown_path`, `prev_sha256`, `new_sha256` in the payload) before invoking the user's `on_edit` callback. Best-effort: if the bus isn't configured or `publish` raises, the watcher logs and continues — never blocks the edit flow on observability.

### Citation-graph reasoning during answering (done 2026-05-20)

The agent's LangGraph topology gains a new `expand_graph` node between `retrieve` and `rerank`. Flat retrieval is still primary; graph augmentation is strictly additive.

- **New retrieval primitives** — `FTSStore.search_in_docs(query, doc_ids, *, k)` and `VectorStore.search_in_docs(query_embedding, doc_ids, *, k)` apply a `document_id IN (...)` filter at the storage layer. `retrieve/hybrid.py:hybrid_search_in_docs(query, doc_ids, *, k)` fuses them with the same RRF pipeline as `hybrid_search`.
- **`expand_graph` node** — runs after `retrieve`, before `rerank`. Takes the top-N unique source documents from `state.candidates` (capped by `graph_expansion_budget`, default 3), queries `GraphStore.neighbors(doc_id, limit=graph_expansion_budget)` for one-hop neighbours (the existing entity-shared signal; CITES edges land in `neighbors()` once the schema query is extended in v1.x+), then calls `hybrid_search_in_docs(query, new_doc_ids, k=chunks_per_neighbor * len(new_doc_ids))` to pull additional chunks. New chunks are de-duped against the existing candidate pool by `chunk_id` and `document_id`, then appended to `state.candidates`. The reranker scores the augmented pool on equal footing.
- **Best-effort wrapper** — `expand_graph` wraps `_expand_graph_impl` and swallows *any* exception (missing ryugraph, missing settings in a unit test, transient I/O), logging the skip and returning a no-op state update. The agent must answer correctly when the graph layer is unavailable for any reason; expansion is strictly additive, never blocking.
- **State knobs** — `AnswerState` gained `graph_expansion_enabled: bool = True`, `graph_expansion_budget: int = 3`, `chunks_per_neighbor: int = 2`, and `graph_expanded_doc_ids: list[str] = []` (for trace introspection). `answer_query` exposes all three as keyword arguments alongside `token_budget` and `max_regenerate_attempts`.
- **Tests** — `tests/integration/test_graph_expansion.py` (6 cases, all green in a real venv): graceful skip when the graph store isn't reachable; `graph_expansion_enabled=False` doesn't touch the graph; real expansion pulls neighbour chunks and the agent's answer cites a chunk that flat retrieval missed; deduping when the graph names a neighbour already in the pool; empty-retrieval short-circuit; default state knobs are what we expect. Existing `test_answering_with_fakes.py` (6 cases) still green with the new node inserted.

### Incremental partial re-indexing (done 2026-05-20)

The index pipeline now diffs the new chunker output against the FTS store's existing per-doc chunk IDs and only re-embeds + re-writes the chunks whose content actually changed. The chunker's chunk_id is `sha1(text)[:10]` — content-derived and stable, so an unchanged paragraph keeps its chunk_id across re-indexings.

- **New selective-delete primitives** — `FTSStore.delete_chunks(chunk_ids)` and `VectorStore.delete_chunks(chunk_ids)` plus `FTSStore.chunk_ids_for_document(doc_id)` for the cheap meta-only lookup.
- **`index_document(doc_id, *, force=False)`** — computes the diff (`to_add = new_ids - existing_ids`, `to_delete = existing_ids - new_ids`, `unchanged = existing_ids & new_ids`), embeds only `to_add`, selectively deletes only `to_delete`, leaves the unchanged chunks (and their embeddings) in place. `force=True` skips the diff and re-embeds everything; useful for `reindex --force` and triggered automatically when the configured embedder differs from the one recorded in the prior manifest (existing vectors would otherwise live in the wrong vector space).
- **Manifest gains breakdown fields** — `IndexStage.chunks_added`, `chunks_deleted`, `chunks_unchanged`, `partial: bool` so `memex doctor` and the eval suite can surface the diff-based behaviour. Pydantic defaults keep existing manifests loading.
- **`IndexResult` gains the same fields** so callers (watcher's `default_reaction`, the `memex index` CLI) see the breakdown.
- **Why this matters** — the watcher fires per edit. Without partial re-indexing, every small markdown tweak triggers a full re-embed of every chunk in the document. With it, a one-paragraph edit re-embeds only the 1–3 affected chunks (the modified one + any adjacent due to chunker overlap). For a 50-chunk document that's a ~6% rewrite instead of 100%.
- **Side-property: stable citations** — agent answers cite chunks by `chunk_id`. Because chunk_ids are content-derived, citations to unedited paragraphs remain valid across re-indexings even when other paragraphs in the document change.
- **Tests** — 9 new in `tests/integration/test_partial_reindex.py` (all green in a real venv): first-index writes everything; re-index of an unchanged body re-embeds nothing; appended paragraph adds only the new chunk(s); removed paragraph deletes the gone chunk(s); modified paragraph swaps exactly one chunk; `force=True` re-embeds everything; embedder-model change implicitly triggers force; the manifest's IndexStage records all three counters + the `partial` flag; existing embeddings for unchanged chunks are preserved byte-for-byte (not silently re-embedded).

### Network-egress sandbox for Docling (done 2026-05-20)

The Docling subprocess that ships from the earlier "subprocess-sandboxed Docling" round now also installs a seccomp-bpf filter that blocks every network-creation syscall *before* importing docling. Any attempt — by docling itself, by transitively-imported libs, or by a malicious document trying to exfiltrate via a triggered fetch — fails at the kernel level with `EPERM`.

- **New `memex/parse/sandbox.py`** — `enable_network_block()` builds a `pyseccomp` filter with `defaction=ALLOW` and explicit `ERRNO(EPERM)` rules for `socket`, `socketpair`, `connect`, `bind`, `sendto`, `sendmsg`, `sendmmsg`. Returns `("applied", reason)` / `("skipped", reason)` / `("failed", reason)`. Linux-only (non-Linux skips immediately); when `pyseccomp` isn't installed it also skips rather than failing.
- **Worker calls it before docling import** — gated on `MEMEX_PARSE_SANDBOX_NETWORK=1` (default). If the user requested the sandbox and we couldn't load it, the worker exits with new code 6.
- **`docling_backend.convert(sandbox_network=True)`** — propagates the setting to the child via env. Maps worker exit 6 to a new `SandboxLoadFailed(MemexError)` so the parent can distinguish "you misconfigured the sandbox" from "docling crashed."
- **`SandboxLoadFailed` is excluded from the Docling circuit-breaker filter** in `parse/pipeline.py:_is_docling_failure` — config errors shouldn't trip the breaker on something Docling itself isn't doing wrong.
- **`ParseSettings.docling_sandbox_network: bool = True`** — opt-out for unusual deployments. The pipeline reads it and passes through `docling_convert`.
- **`pyseccomp>=0.1 ; sys_platform == 'linux'`** added to the `[parse]` extra.
- **DX note documented in the README**: a sandboxed worker can't download models. Users pre-fetch with `huggingface-cli download <model_id>` once and the local cache reads stay local. The pattern is recommended for production deployments anyway (model downloads are slow + load-bearing for cold starts).
- **Tests** — 7 new in `tests/integration/test_sandbox.py`, all green in a real Linux venv with `pyseccomp` installed:
  - graceful skip on non-Linux (mocked `sys.platform`)
  - graceful skip when `pyseccomp` isn't importable (mocked `__import__`)
  - worker honours `MEMEX_PARSE_SANDBOX_NETWORK=0`
  - **end-to-end network block proof**: subprocess applies the sandbox, `socket.socket()` raises `PermissionError` — the kernel-level guarantee
  - filesystem I/O is unaffected (seccomp blocks network only)
  - `convert(sandbox_network=False)` sets the env var to `"0"`; the default sets it to `"1"`
  - worker exit 6 surfaces as `SandboxLoadFailed`, not the generic `DoclingCrashed`

### Pre-existing bugs uncovered along the way

While running the full test suite with the new partial-reindex paths exercised, four pre-existing latent bugs surfaced:

- **`ingest/validation.py:_refine_office`** — `"vbaProject.bin" in n.lower()` checked a mixed-case substring against a lowercased haystack: always False, the macro detector never fired. Fixed by lowercasing the substring too.
- **`eval/scoring.py:word_error_rate`** — used character-level Levenshtein on `"\x00".join(words)` as a "cheap trick" for word-level WER. Counted character edits, not word edits, so "the slow brown fox" vs "the quick brown fox" returned 1.25 instead of 0.25. Replaced with a real token-sequence Levenshtein helper.
- **`memex/mcp/__init__.py`** — re-exported `server` (the FastMCP instance), which shadowed the `memex.mcp.server` module path when `monkeypatch.setattr("memex.mcp.server.hybrid_search", ...)` resolved it. Stopped re-exporting `server`; callers import it explicitly.
- **`tests/unit/test_chunker.py:test_target_window_is_respected_for_long_content`** — used `" ".join(["lorem"] * 1500)` as test input. The chunker's sentence splitter looks for `.!?` followed by uppercase, so run-on text never split. Replaced with realistic prose across multiple sections.

### Subprocess-sandboxed Docling (done 2026-05-20)

- New `memex/parse/docling_worker.py` — a module-as-script entry point. Reads source path from argv, runs `DocumentConverter`, prints the serialised `DoclingConversion` payload to stdout. Exit codes: 0 success, 2 usage, 3 missing source, 4 docling_unavailable, 5 unhandled exception.
- `parse/docling_backend.convert` rewritten: spawns `python -m memex.parse.docling_worker <source>` via `asyncio.create_subprocess_exec`, captures stdout + stderr, deserialises the JSON. Timeouts terminate the child (SIGTERM, then SIGKILL after 2 s); non-zero exit → `DoclingCrashed(MemexError)` with `exit_code` + truncated `stderr` in `context`; parse errors → `DoclingCrashed` with the stdout prefix.
- New `DoclingCrashed` exception, exported from `memex.parse` and added to the Docling-circuit-breaker filter in `parse/pipeline.py:_is_docling_failure`.
- Network-egress sandbox (seccomp / landlock on Linux per GUIDELINES Part VI) is still queued — this round delivers crash containment, not perfect egress isolation.

### Pre-existing fix

`vault/store.py:Frontmatter.date` field's annotation `date | None` was a pydantic-eval footgun under `from __future__ import annotations` (the field name `date` shadows the imported `date` type during pydantic's annotation evaluation). Aliased the import to `_date` and updated the annotation. Caught by the smoke-test for this round; the prior test suite presumably masked it through the integration-test fixtures providing the right import order.

---

## CUDA / GPU stack — audit findings (2026-05-20)

A four-agent audit on 2026-05-20 surfaced six correctness gaps and a longer tail of performance leaks. Full findings in [`cuda_audit`](../../../.claude/projects/-home-drei-project-Doc-Flo/memory/cuda_audit.md) memory; architectural decisions are now formalised in [ADR-0006](adr/0006-cuda-dispatch-and-dtype.md).

**Status:** all 10 actions complete (2026-05-20). ADR-0006 is the durable reference.

| # | Action | Status |
|---|---|---|
| 1 | `pyproject.toml` — `pytorch-cu128` uv index + `torch>=2.7`, `transformers>=4.57`, `sentence-transformers>=5.0`, `accelerate>=1.0`, `flash-attn>=2.6` | ✓ Done |
| 2 | `cli/bootstrap.py` — `_configure_cuda()`: assert + alloc-conf + TF32 + cudnn + `set_per_process_memory_fraction`; `_verify_vram_fit()` warning | ✓ Done |
| 3 | `models/registry.py` — `device="cuda"` + bf16 on embedder/reranker; VLM → `AutoModelForImageTextToText` + FA2 + `device_map={"": "cuda:0"}` | ✓ Done |
| 4 | `core/config.py` — `vlm_quantization` knob, AWQ-Int4 default, `orchestrator_quantization` opened to AWQ/GPTQ | ✓ Done |
| 5 | `scripts/serve-vllm.sh` — codified vLLM startup flags | ✓ Done |
| 6 | `models/client.py` — inline `$defs`/`$ref` before send | ✓ Done |
| 7 | `index/pipeline.py` + `retrieve/rerank.py` — `batch_size` tuning | ✓ Done |
| 8 | `parse/pipeline.py` + `parse/vlm_backend.py` — per-document VLM batch; scale 2.0; `max_new_tokens=1024` | ✓ Done |
| 9 | README — CUDA Prerequisites + getting-started + docs map | ✓ Done |
| 10 | ADR-0006 | ✓ Done |

---

## Stack watchlist — eval-gated upgrades

From the 2026-05-19 multi-agent audit ([stack_currency_audit](../../../.claude/projects/-home-drei-project-Doc-Flo/memory/stack_currency_audit.md)). All wait on `tests/evals/` corpus availability before A/B; the stack-currency items themselves are actioned (Kuzu→RyuGraph via ADR-0005, Langfuse v4 in `observability/tracing.py`).

- **Qwen3-VL-8B-Instruct** to replace Qwen2.5-VL-7B (+6–14 OCR points, native 256K context; Q4_K_M ~6.1GB). Action at next VLM cycle.
- **Granite 4.1-8B** A/B vs Qwen3-8B-Instruct (Apache-2.0, native OpenAI-style tool calling baked into the chat template). Needs eval-delta.
- **Qwen3-Reranker-0.6B** A/B vs bge-reranker-v2-m3 (+8 MMTEB-R; autoregressive ⇒ higher per-pair latency). Gate on the <15s answer floor.
- **PyMuPDF4LLM** as a Docling pre-filter for clean-text PDFs (cuts the easy ~60% ingest time at zero licensing cost).
- **vLLM `response_format` migration** — already adopted in `models/client.py:complete_structured`. Done.

---

## Open architectural questions

Snapshot from IMPLEMENTATION-PLAN.md §5; current state.

| # | Question | Recommended default | Status |
|---|---|---|---|
| 5.1 | Chunking strategy | Header-aware ~600-token windows with 100-token overlap | ✓ Shipped in `index/chunker.py` |
| 5.2 | Embedding dimensionality | EmbeddingGemma native 768 (one-way door) | ✓ Pinned in `index/vector_store.py` |
| 5.3 | Prompt versioning storage | Filesystem `prompts/<name>/v<N>.md` + env-var pin | ✓ Shipped in `prompts/loader.py` |
| 5.4 | Surfacing ungrounded refusals in UI | CLI: refusal + collapsed rejected draft; Web: refusal + rejected draft with strikethroughs + "show retrieval" | ⌗ Open until Phase 3 web UI |
| 5.5 | MCP transport: stdio-only or HTTP too? | stdio-only for v1; HTTP after auth design | ⌗ Open until Phase 3 |
| 5.6 | Orchestrator in registry or remote-only? | Remote vLLM only; registry owns embedder/reranker/VLM | ✓ Settled in code |
| 5.7 | Trace retention | 30 days, matching event-bus retention | ⌗ Open until Langfuse self-host wired |

---

## Decisions still owed

- **Wikilink format for in-vault citation links** — `[[doc_id]]` vs `[[doc_id#section]]` vs human-readable `[[Smith 2024]]`. ADR-0003 commits to `[[doc_id]]`; v1.x enrich work will need to choose for the section-anchor case.
- **`memex serve mcp --http` auth model** — bearer token? mTLS? bind to localhost-only? Decision required before Phase 4 starts shipping HTTP MCP.
- **Daemon process model** — `systemd` unit on Linux? `launchd` on macOS? In-process for v1? Owed by Phase 4.
- **GPU mid-tier story** — what happens on 8GB cards? Per ADR-0001 we accept "no first-class CPU fallback"; should there be a documented 8GB-tier with smaller model defaults? Owed once the v1.x model-swap A/B settles.
