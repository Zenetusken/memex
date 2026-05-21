# Bug-Hunt Synthesis (4-agent audit, 2026-05-20)

Total findings across 4 agents: ~70 (some overlap). Deduplicated and prioritised below.

## Security (fix first)

| # | Source | File | Issue |
|---|---|---|---|
| S1 | Error E5 | webui/app.py:84,155,179,195,208,222,280 | Path traversal via crafted doc_id — needs regex constraint on every `{doc_id}` route |
| S2 | Error E1 / Quality N31 | index/vector_store.py:106,128,168 | LanceDB f-string `where` clauses → SQL injection if doc_id ever contains a quote |

## Broken now (user-visible)

| # | Source | File | Issue |
|---|---|---|---|
| B1 | Wiring W1 | cli/commands.py:237 | `memex doctor` never bootstraps → RuntimeError on every invocation |
| B2 | Wiring W2 | cli/commands.py:344 | Private `_docling_breaker` import; ban + couples to internals |
| B3 | Wiring W4 | cli/commands.py:106-116 | `--skip-parse` silently ignored for directories |

## Data-loss / silent corruption

| # | Source | File | Issue |
|---|---|---|---|
| D1 | Resource C1 | webui/app.py:252 | /review writes md, then manifest. Kill in between loops on every restart. |
| D2 | Resource C5 | vault/store.py:97 | Concurrent write_document → last-writer-wins, silent data loss |
| D3 | Resource C4/Q10 | ingest/watcher.py:211 | Fire-and-forget _drain_one tasks silently dropped on shutdown |
| D4 | Resource C3 / Error E15 | daemon/supervisor.py:248,115 | PID race (start) + PID-reuse (stop) can SIGKILL wrong process |
| D5 | Resource C2 / Error E20 | core/bus.py:106-118 | EventBus.publish sync SQLite in async path blocks event loop |
| D6 | Resource I4 | parse/docling_backend.py:102 | CancelledError leaks subprocess + pipe + zombie |
| D7 | Resource I5 | index/pipeline.py:136 | Non-ImportError from _open_graph leaks vstore + fstore |
| D8 | Resource I1 | models/registry.py:108 | Partial-load exception leaks VRAM |
| D9 | Resource I9 | enrich/pipeline.py:108,148 | gather propagates first exception, half-enriches doc |

## Correctness (exception semantics)

| # | Source | File | Issue |
|---|---|---|---|
| C1 | Error E2 | agents/answering.py:236 | expand_graph `except Exception` swallows CancelledError + real bugs |
| C2 | Error E6 | core/breakers.py:95 | CircuitBreaker `except BaseException` includes Ctrl-C, cancel |
| C3 | Error E7 | daemon/supervisor.py:96 | `_reachable` `except Exception` masks misconfig as transient |
| C4 | Error E10 | models/client.py:159 | complete_structured `except Exception` wraps CancelledError + TypeError |
| C5 | Error E4 / Quality Q4 | enrich/pipeline.py:111 | assert isinstance stripped under python -O |
| C6 | Error N18 | cli/commands.py:111 | `assert r.doc_id is not None` stripped under -O |
| C7 | Error E11 | parse/docling_worker.py:131 | json.dump allows NaN → opaque parent parse failure |
| C8 | Error E16 | index/fts_store.py:165,203 | Empty/NUL query → "malformed MATCH expression" |
| C9 | Wiring W8 | parse/pipeline.py:201,217 | Crash path update_manifest without content_sha256 → double-fault |

## Typed errors

| # | Source | File | Issue |
|---|---|---|---|
| T1 | Error E3 + E8 + E9 / Quality Q8 / Wiring (multiple) | core/bus.py:208, core/config.py:181, core/manifest.py:160, parse/pipeline.py:95, daemon/supervisor.py:134,174 | Bare RuntimeError / ValueError / MemexError should be ConfigurationError / VaultIntegrityError |

## Wiring / API gaps

| # | Source | File | Issue |
|---|---|---|---|
| W1 | Wiring W3 | webui/app.py:43 | `from memex.index.graph_store` violates module boundary |
| W2 | Wiring W5 / Quality Q11 | graph_store.py:159 + agents/answering.py:251 | CITES traversal claimed but not implemented |
| W3 | Wiring W6 / Quality N33 | models/client.py:157 + agents/answering.py:364,415,449 | `name=None` to Langfuse; pass real prompt_tag |
| W4 | Wiring W7 | ingest/pipeline.py:201 | passthrough size/sha discrepancy |
| W5 | Wiring W10 | agents/__init__.py:9 | Re-exports Chunk that lives in core.types |
| W6 | Wiring W13 | ingest/watcher.py:218 | `default_reaction` lazily imports enrich + index; breaks layering |

## Hardening

| # | Source | File | Issue |
|---|---|---|---|
| H1 | Error E13 | webui/app.py:126,226 | Form(...) inputs unbounded (max_length missing) |
| H2 | Error E14 | ingest/pipeline.py:80-82 | _hash_file reads entire file into memory |
| H3 | Error E15 | daemon/supervisor.py:115 | PID race via non-atomic check-then-write |
| H4 | Resource I3 | daemon/supervisor.py:196 | start() timeout doesn't escalate to SIGKILL |
| H5 | Resource I2/I6 | daemon/supervisor.py:165,158 | PID file write-before-reachability + Popen-then-pidfile-write race |
| H6 | Resource I7 | ingest/watcher.py:128 | observer.start() outside try block |
| H7 | Error E12 / Quality Q9 | index/pipeline.py:305 + cli/commands.py:122 | reindex stringifies MemexError context away |
| H8 | Resource I8 | core/bus.py:128 | Full subscriber queue logs every publish; rate-limit |
| H9 | Wiring W9 | parse/pipeline.py:398 | Dead `not isinstance(...)` clause in `_is_docling_failure` |

## Docs / drift

| # | Source | File | Issue |
|---|---|---|---|
| X1 | Quality Q6 | Multiple __init__.py + cli/commands.py:1 + models/registry.py:5 + parse/pipeline.py:58 _PARSER_VERSION | Phase-N doc drift across many modules |
| X2 | Quality Q2 / Wiring W12 | mcp/server.py:3 + test_mcp_server.py:195 + mcp/__init__.py:14 | MCP five-tools-vs-four drift |
| X3 | Quality Q14 | webui/templates/base.html:7-12 | CDN Tailwind + HTMX violate air-gap; backdrop-blur violates CLAUDE.md |
| X4 | Quality Q7 | core/config.py:75 | `disable_vlm: bool = True` Phase-1 default; VLM is wired |

## Tests

| # | Source | File | Issue |
|---|---|---|---|
| TS1 | Quality Q1 | test_answering_with_fakes.py:253 | Direct module mutation w/o teardown leaks |
| TS2 | Quality Q5,Q15 | test_v1x_backlog.py:199-239 | "Watcher publish failure" test calls bus.publish directly; doesn't exercise _drain_one |
| TS3 | Quality Q3,Q12 | enrich/pipeline.py:88,144,76 | Untyped chunk/c/vault_path params |
| TS4 | Quality Q13 | test_enrich_and_graph.py:246 | Reaches into `fake_llm._responses` private attr |

## Nits / minor
N23-N35, Resource N1-N8, Wiring W11,W14,W15 — documented in individual audit files.

## Execution plan
**Phase A (security + broken)**: S1, S2, B1, B2, B3 — 5 fixes, all small, must ship now.
**Phase B (data loss)**: D1-D9 — 9 fixes, mid-sized.
**Phase C (correctness)**: C1-C9 — 9 fixes, small each.
**Phase D (typed errors)**: T1 — sweep across 5 sites.
**Phase E (wiring)**: W1-W6 — 6 fixes.
**Phase F (hardening)**: H1-H9 — 9 fixes.
**Phase G (drift)**: X1-X4 — docs sweep.
**Phase H (tests)**: TS1-TS4 — 4 fixes.

Stop after Phase A+B+C if test suite passes. Phases D-H are next-session work.
