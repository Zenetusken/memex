# Code Quality, Conventions & Test Quality Audit (agent aa945cbe)

## Critical

- **Q1** test_answering_with_fakes.py:253-255 — `ans.complete_structured = fake_call` direct module mutation w/o restoration. Leaks across session. Fix: `monkeypatch.setattr`.
- **Q2** mcp/server.py:3, test_mcp_server.py:195-203, mcp/__init__.py:14 — Docstring + test name say "four tools" but five are registered (`get_graph_neighbors` added in Phase 4). Fix: docstring + test set + re-export decision.
- **Q3** enrich/pipeline.py:88,144 — `_extract_chunk(chunk, title)` and `_run(c)` lack `chunk: Chunk` annotation. Pyright strict will error.
- **Q4** enrich/pipeline.py:111-112 — `assert isinstance(entity_raw, EntityList)` in production (overlaps Error E4).
- **Q5** test_v1x_backlog.py:199-239 — `test_watcher_publish_failure_does_not_break_edit_flow` calls `bus.publish` directly with try/except; never drives `_drain_one`. The test name is a lie.

## Important

- **Q6** Phase-N docstring drift sweep:
  - parse/__init__.py:1
  - enrich/__init__.py:3-5
  - index/__init__.py:7-8
  - ingest/__init__.py:8
  - webui/__init__.py:6
  - cli/__init__.py:9
  - cli/commands.py:1-12 (claims daemon is stub)
  - models/registry.py:5-9 (claims VLM is Phase-2 deferred)
  - parse/pipeline.py:58 `_PARSER_VERSION="memex.parse@phase-1"` written into every manifest
  - index/pipeline.py:253
- **Q7** core/config.py:75 — `disable_vlm: bool = True` default + "Phase 1" comment. VLM is wired. Decision: flip default or document rationale.
- **Q8** core/bus.py:208, core/config.py:181 — `RuntimeError` should be MemexError subclass (overlaps Error E8).
- **Q9** cli/commands.py:122-124 — `failures: list[str]` loses structure across CLI boundary. Should be `list[FailureItem]` pydantic (overlaps Error E12).
- **Q10** ingest/watcher.py:211 — `asyncio.create_task(_drain_one(path))` without held reference; task may be GC'd (overlaps Resource C4).
- **Q11** index/graph_store.py:159 — `neighbors()` only returns shares_entity; agent's `expand_graph` and ROADMAP claim "CITES + shared-entity." Fix: extend Cypher OR correct docs.
- **Q12** enrich/pipeline.py:76 — `vault_path` untyped → `Path`.
- **Q13** test_enrich_and_graph.py:246 — reaches into `fake_llm._responses` private attr. Fix: rename to public or expose method.
- **Q14** webui/templates/base.html:7-9,11-12 — `cdn.tailwindcss.com` + `unpkg.com` HTMX violate air-gap claim AND frontend CLAUDE.md "no third-party at runtime." Also has `backdrop-blur` which CLAUDE.md explicitly forbids. Fix: vendor into static/ + remove blur.
- **Q15** test_v1x_backlog.py:189-196 — `assert callable(run_watcher)` is a no-op; imports never exercised.
- **Q16** mcp/server.py:41,57,77,92,110 — Tools return dict[str,Any] instead of pydantic. Crosses module boundary; should be pydantic per CLAUDE.md.

## Nits

- N23 graph_store.py:25-26, vlm_backend.py:26-27 — Empty `if TYPE_CHECKING:` blocks
- N24 enrich/pipeline.py:258 — Dead `_ = entity_id` line
- N25 test_sandbox.py:20 — unused `import errno`
- N26 webui/app.py:274 — naive `datetime.now()` (no tz)
- N27 retrieve/rerank.py:1-7 — "registry can configure device" docstring stale (now hardcoded cuda per ADR-0006)
- N28 ingest/pipeline.py:147-149 — "Phase 1 ships without live bus on CLI path" stale
- N29 chunker.py:1-12, index/pipeline.py:9 — chunk_id format docs imprecise
- N30 vlm_backend.py:56,89,127 — `handle`/`image` untyped
- N31 vector_store.py:106-128 — Same as Error E1 (SQL injection vector)
- N32 docling_worker.py:120-121 — misleading comment about JSON-to-stdout that's actually to exit-code
- N33 models/client.py:157 — `name=prompt_tag` only works under Langfuse-patched OpenAI SDK
- N34 cli/commands.py multiple — `async def _run()` missing return type annotations
- N35 parse/pipeline.py:171 — `_bootstrap_ref` missing return type

## Verdict
"Genuinely good shape and real discipline... docs and a few integration tests are a sprint behind the implementation."

## Overlap matrix
- Q4 = E4 (assert isinstance)
- Q8 = E8 (bus/config RuntimeError)
- Q9 = E12 (failures string vs structured)
- Q10 = Resource C4 (watcher task tracking)
- Q31 = E1 (LanceDB SQL injection)
