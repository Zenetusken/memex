# Resource & Concurrency Audit (agent ad19b166)

## Critical (data loss / deadlock / silent corruption)

- **C1** webui/app.py:252-257 — `document_review` writes markdown then manifest. Kill in between leaves watcher loop on every restart. Fix: write manifest first with new sha, or staging+atomic-rename.
- **C2** core/bus.py:106-118 — `EventBus.publish` does blocking `sqlite3.execute` inside `async def`. Fix: wrap in `asyncio.to_thread` (matches FTSStore).
- **C3** daemon/supervisor.py:248,271 — `stop()` PID-reuse race can SIGKILL unrelated process. Fix: verify `/proc/PID/cmdline` matches `serve_script` before signalling.
- **C4** ingest/watcher.py:211 — `_drain_one` tasks fire-and-forget, silently cancelled on shutdown mid-debounce. Fix: track in `set[asyncio.Task]`, await in `finally`.
- **C5** vault/store.py:97-112 — `_atomic_write` not safe across concurrent writers to same path; silent last-writer-wins. Fix: per-`doc_id` `asyncio.Lock` registry.

## Important (resource leaks)

- **I1** models/registry.py:108-140 — `_load` leaks partially-loaded model on exception (VRAM held). Fix: try/except cleanup + `torch.cuda.empty_cache()`.
- **I2** daemon/supervisor.py:165-170 — PID file written before reachability confirmed; Ctrl-C in window orphans daemon. Fix: write PID file after reachability OR write `.tmp` and rename on success.
- **I3** daemon/supervisor.py:196-201 — `start()` timeout doesn't escalate to SIGKILL. Fix: symmetric escalation with `stop()` at 269-273.
- **I4** parse/docling_backend.py:102-115 — `convert` leaks stdout/stderr pipes + zombies child if outer coroutine cancelled. Fix: try/finally wrapping spawn block.
- **I5** index/pipeline.py:136-138 — `_open_graph` non-ImportError raise leaks vstore + fstore. Fix: `AsyncExitStack`.
- **I6** daemon/supervisor.py:158-165 — Popen+log_fp.close, then pid_file.write fails → orphaned child. Fix: pid file write before Popen, kill on Popen-onward failure.
- **I7** ingest/watcher.py:128-132 — `observer.start()` outside try block; watchdog start failure leaks thread.
- **I8** core/bus.py:128-135 — full subscriber queue logs warning on every publish, unbounded noise. Fix: rate-limit per-subscriber.
- **I9** enrich/pipeline.py:108-110,148 — `asyncio.gather` propagates first exception, cancels others, half-enriches doc. Fix: `return_exceptions=True` + per-chunk manifest.
- **I10** ingest/watcher.py:135,142-149,208-211 — `_drain_one` and outer share mutable `last_seen` dict; documented thread-safety invariant absent.

## Nits

- N1: LanceDB concurrent search smoke test missing
- N2: FTSStore writes serialise on SQLite mutex; future `transaction()` would need explicit `asyncio.Lock`
- N3: `configure_client` doesn't close prior `_client` (httpx pool linger)
- N4: GraphStore.close() is a no-op; verify ryugraph WAL flush
- N5: `_pid_alive` returns True on EPERM (treated as alive but unkillable)
- N6: `_COMPILED_GRAPH` written without lock on first compile
- N7: pypdfium2 `to_pil` lifetime vs `doc.close()` ordering
- N8: docling breaker `lambda` returns coroutine — verify `await`

## Top 10 priority (agent's ranking)
1. C1 (watcher re-enrich loop)
2. C2 (event-loop blocking)
3. C5 (concurrent vault writes)
4. I4 (docling zombie containment)
5. C4 (watcher shutdown drain)
6. C3 (PID-reuse safety)
7. I3 (daemon start SIGKILL escalation)
8. I5 (AsyncExitStack)
9. I1 (model registry cleanup)
10. I9 (enrich error isolation)
