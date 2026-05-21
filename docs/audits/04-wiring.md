# Wiring & Signature Mismatches Audit (agent a1e26d6a)

## Critical

- **W1** cli/commands.py:237-243 — `doctor` command never calls `bootstrap()`. `_doctor_report` immediately does `get_settings()` → RuntimeError. Diagnostic-of-last-resort is BROKEN.
- **W2** cli/commands.py:344 — `from memex.parse.pipeline import _docling_breaker` imports a private symbol across module boundary. CLAUDE.md universal ban. Fix: public `get_docling_breaker_state()`.

## Important

- **W3** webui/app.py:43 — `from memex.index.graph_store import GraphStore` violates import direction (`webui/` → agents + vault + core only). MCP server uses lazy import for this reason. Fix: lazy-import inside the route.
- **W4** cli/commands.py:106-116 — `_ingest_path_chain` honours `--skip-parse` for single files (via `_process_one`) but ignores it for directories (always calls `parse_document`). Fix: route directory items through `_process_one`.
- **W5** index/graph_store.py:159-185 + agents/answering.py:245-298 — `expand_graph` docstring says it traverses CITES edges enrich writes; actual Cypher traverses MENTIONS only. Enrich's `link_cites` writes edges no read path uses. (Overlaps Q11.)
- **W6** models/client.py:157 — Three agent nodes (`answer`, `assess`, `verify`) call `complete_structured` without `prompt_tag` → `name=None` arrives at OpenAI/Langfuse. Lost trace span names at best. (Overlaps Q33.) Fix: pass per-node tag or fall back to `schema.__name__`.
- **W7** ingest/pipeline.py:201-208 — `ingest_markdown_passthrough` records body-only `source_size_bytes` but full-doc (post-frontmatter) `content_sha256`. Fields disagree.
- **W8** parse/pipeline.py:201-214,217-230 — Crash path calls `update_manifest` without `content_sha256`. If manifest doesn't exist (user runs `parse` on a sideloaded markdown), update raises ValueError, masking original DoclingCrashed.
- **W9** parse/pipeline.py:398-412 — `_is_docling_failure` has dead `not isinstance(...)` clause for SandboxLoadFailed/ParseConfidenceTooLow (already excluded by first isinstance check). Confusing but inert.
- **W10** agents/__init__.py:9 — Re-exports `Chunk` from agents.answering even though it lives in core.types. Dead re-export + misleads about layering.

## Nits

- **W11** graph_store.close() is no-op `await asyncio.sleep(0)` (Resource N4 also)
- **W12** test_mcp_server.py:195 comment "four tools" but registers five (Q2 also)
- **W13** ingest/__init__.py re-exports `default_reaction` which lazily imports enrich + index, breaking the documented `ingest/ → vault+core` rule
- **W14** webui/app.py:163,188 — `_kind_for(source)` recomputed in document_source route
- **W15** agents/answering.py:274,290 — `graph_expansion_budget` reused as both source-doc cap and per-doc neighbour limit (semantic overload)

## Top 10 (agent ranking)
1. W1 — Bootstrap doctor
2. W2 — Public docling_breaker_state
3. W3 — Lazy GraphStore import in webui
4. W4 — Honour --skip-parse for directories
5. W8 — Manifest-tolerant parse crash path
6. W5 — CITES traversal in neighbors() OR doc correction
7. W6 — Real prompt_tag in agent nodes
8. W7 — Reconcile passthrough size/sha
9. W13 — Move default_reaction out of ingest/
10. W9 — Clean dead isinstance clause

## Overlap matrix
- W2 = Q?? (private symbol pattern)
- W5 = Q11 (CITES traversal)
- W6 = Q33 (name= kwarg)
- W12 = Q2 (MCP five tools)
- W11 = Resource N4 (graph_store.close)
