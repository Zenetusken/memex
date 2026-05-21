# Error Handling & Edge Cases Audit (agent afcbf4a3)

## Critical

- **E1** vector_store.py:106-107,128-129,168-169 — LanceDB `where` predicates built via f-string interpolation of `doc_id`/`chunk_id`. SQL-injection if a quote ever enters. FTS5 paths use `?` placeholders so safe; LanceDB-only. Fix: validate doc_id regex at boundary + escape quotes at interpolation.
- **E2** agents/answering.py:236 — `expand_graph` `except Exception` swallows real bugs + CancelledError under warning log. Fix: narrow catches, re-raise CancelledError, add `GraphExpansionFailed` MemexError.
- **E3** parse/pipeline.py:95-98 — Missing asset dir raises bare `MemexError`, should be `VaultIntegrityError`. Same: `daemon/supervisor.py:134,174`.
- **E4** enrich/pipeline.py:111-112 — `assert isinstance` stripped under `python -O`. Fix: `if not isinstance: raise ModelCallError`.
- **E5** webui/app.py:84,155,179,195,208,222,280 — Path traversal possible via crafted `doc_id` (Starlette decoding nuances). Fix: `Path(... pattern="^[0-9a-f]{8}(-[a-z0-9-]+)?$")` on every route.
- **E6** core/breakers.py:95-98 — `except BaseException` includes `CancelledError`, `KeyboardInterrupt`, `SystemExit`. Ctrl-C 3x trips Docling breaker. Fix: `except Exception` + explicit re-raise of CancelledError.
- **E7** daemon/supervisor.py:96 — `_reachable` catches all Exception, masks bad base_url as "not ready yet" until timeout. Fix: log type/message at debug.

## Important

- **E8** core/bus.py:208 — `RuntimeError` should be `ConfigurationError`. Affects ingest/watcher.py:161 catch.
- **E9** core/manifest.py:160-162 — `ValueError` should be `VaultIntegrityError`.
- **E10** models/client.py:159 — `complete_structured` `except Exception` wraps `CancelledError` + `TypeError`. Fix: narrow to `openai.OpenAIError`, `httpx.RequestError`, `asyncio.TimeoutError`.
- **E11** parse/docling_worker.py:131 — `json.dump` allows NaN/Inf by default; parent's `json.loads` rejects. Fix: `allow_nan=False` + coerce NaN→None upstream.
- **E12** index/pipeline.py:305-307 — `reindex` failures stringify `MemexError.context` away. Fix: structured `list[FailureRecord]`.
- **E13** webui/app.py:126,226 — `Form(...)` no `max_length`. 10 GB POST → OOM. Fix: max_length=settings.ingest.max_bytes.
- **E14** ingest/pipeline.py:80-82 — `_hash_file` reads whole file into memory. Fix: 64 KiB blocks.
- **E15** daemon/supervisor.py:115-130 — PID file check/read/unlink not atomic. Two `start` calls race. Fix: `O_CREAT | O_EXCL` exclusive create.
- **E16** index/fts_store.py:165-201,203-253 — Empty/NUL query → `malformed MATCH expression`. Fix: short-circuit + strip control chars.
- **E17** core/bus.py:106-118 — Same as Resource C2 (sync SQLite in async). Listed by both agents.

## Nits

- **N17** retrieve/fusion.py:30 — Doesn't validate k≥0.
- **N18** cli/commands.py:111 — `assert r.doc_id is not None` strips under -O.
- **N19** cli/commands.py:377 — Broad Exception for daemon probe.
- **N20** vector_store.py:65, fts_store.py:194 — `" > "` separator destructive if heading contains " > ".
- **N21** agents/answering.py:538-542 — `surviving_claims` doesn't bounds-check `grounded[i]`.
- **N22** parse/pipeline.py:162-168 — Trailing-space frontmatter not handled.

## Top 10 (agent's ranking)
1. Validate doc_id regex on every route (#5 + linchpin for #1)
2. Escape/assert quotes at LanceDB interpolation sites (#1)
3. Narrow except clauses: expand_graph, complete_structured, _reachable, CircuitBreaker (#2, #6, #7, #10)
4. Replace assert isinstance with proper raises (#4, #18)
5. Typed errors: get_bus, get_settings, daemon, manifest (#3, #8, #9)
6. max_length on form inputs (#13)
7. allow_nan=False in docling_worker (#11)
8. PID race via O_CREAT|O_EXCL (#15)
9. Short-circuit empty/NUL FTS queries (#16)
10. Chunked _hash_file + bus to_thread (#14, #20)

## Overlap with Resource Audit
- E17 = Resource C2 (bus.publish sync SQLite)
- E15 (PID race start) complements Resource C3 (PID-reuse stop)
