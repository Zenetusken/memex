"""RyuGraph cross-process read-consistency probe — the empirical root-cause for the
deferred "GraphStore connection reuse" item (db_connection_reuse_nogo_2026_06_07).

The measured-defer left ONE thing unverified: can a long-lived (reused) ryugraph read
connection stay correct when a SEPARATE process writes to the same on-disk graph? An
Explore agent ASSERTED "a held reader sees writes, don't reopen" — but that contradicts
kuzu's documented write-exclusivity and ignores the checkpoint-under-reader CORRUPTION
hazard (a C++ segfault that would kill the webui, uncatchable in Python). This probe
MEASURES the real behavior of ryugraph 25.9.2 instead of guessing.

CRASH-SAFETY: the dangerous probes (a read connection HELD across another process's
write+checkpoint, and a held handle across a teardown) run the holding reader in a CHILD
process, so a segfault kills the child and the harness OBSERVES it as a negative return
code (signal) — the harness itself never dies and always prints its verdicts.

It operates on throwaway temp graph dirs (NEVER the vault) and imports `ryugraph`
directly (characterizes the LIBRARY, below the `GraphStore` wrapper). Verdicts are DATA:
the script exits 0 even when a probe reports FAIL/CRASH (non-zero exit = HARNESS error).

The probes (see the plan for the full mapping verdict → fix world):
  R0   open-cost decomposition: Database(path) vs Connection(db) (ro + rw)
  R0b  N concurrent Connections minted on ONE shared read_only Database
  R1   held-handle write VISIBILITY + checkpoint SURVIVAL across N write cycles (ro + rw)
  R2   reader-vs-writer LOCKING: does writer-B succeed/block while reader-A holds? (ro+rw)
  R3   FRESH-reopen visibility (validates the CURRENT per-request design)
  R4   teardown (reindex --force) survival + change-detection signals
  R5   read_only cold-start exception class (missing DB / missing schema)

Usage:
  MEMEX_OBSERVABILITY__LANGFUSE_ENABLED=false uv run python scripts/ryugraph_consistency_probe.py
  ... scripts/ryugraph_consistency_probe.py --cycles 25 --json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# ── Embedded standalone scripts (written into the temp dir, run as REAL processes) ──────
# ruff/pyright don't analyse string bodies, so the broad excepts here are intentional and
# safe — these run as separate interpreters whose ONLY job is to report what ryugraph did.

_WRITER = '''
import sys, traceback
import ryugraph
try:
    graph_path, action = sys.argv[1], sys.argv[2]
    node_id = sys.argv[3] if len(sys.argv) > 3 else "x"
    # checkpoint_threshold=1 + db.close() maximise checkpoint pressure UNDER any held reader.
    db = ryugraph.Database(graph_path, checkpoint_threshold=1)
    conn = ryugraph.Connection(db)
    conn.execute("CREATE NODE TABLE IF NOT EXISTS Probe(id STRING, PRIMARY KEY(id));")
    if action == "merge":
        conn.execute("MERGE (p:Probe {id: $id});", {"id": node_id})
    conn.close()
    db.close()
    sys.stdout.write("ok\\n"); sys.stdout.flush()
except BaseException:
    traceback.print_exc()
    sys.exit(1)
'''

_READER_CYCLES = '''
import sys, json, subprocess, time
import ryugraph
graph_path, mode = sys.argv[1], sys.argv[2]
n_cycles, writer_script, py = int(sys.argv[3]), sys.argv[4], sys.argv[5]
read_only = (mode == "ro")
db = ryugraph.Database(graph_path, read_only=read_only)
conn = ryugraph.Connection(db)
def count():
    r = conn.execute("MATCH (p:Probe) RETURN count(p) AS c;")
    return int(r.get_next()[0]) if r.has_next() else 0
print(json.dumps({"event": "baseline", "count": count()}), flush=True)
for k in range(n_cycles):
    rec = {"event": "cycle", "k": k}
    t0 = time.perf_counter()
    # A SEPARATE process opens read-WRITE, MERGEs, checkpoints+closes — the cross-process write.
    try:
        w = subprocess.run([py, writer_script, graph_path, "merge", mode + str(k)],
                           capture_output=True, text=True, timeout=30)
        rec["writer_rc"] = w.returncode
        rec["writer_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if w.returncode != 0:
            rec["writer_err"] = (w.stderr or "").strip()[-200:]
    except subprocess.TimeoutExpired:
        rec["writer_rc"] = "timeout"
        rec["writer_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    # Re-query the HELD connection — NO reopen. This is the visibility + survival test.
    try:
        rec["count"] = count()
    except BaseException as e:
        rec["reader_err"] = type(e).__module__ + "." + type(e).__name__ + ": " + str(e)[:200]
        print(json.dumps(rec), flush=True)
        break
    print(json.dumps(rec), flush=True)
try:
    conn.close(); db.close()
except BaseException:
    pass
print(json.dumps({"event": "done"}), flush=True)
'''

_TEARDOWN_READER = '''
import sys, json, time, os
import ryugraph
graph_path, sentinel = sys.argv[1], sys.argv[2]
db = ryugraph.Database(graph_path, read_only=True)
conn = ryugraph.Connection(db)
def count():
    r = conn.execute("MATCH (p:Probe) RETURN count(p) AS c;")
    return int(r.get_next()[0]) if r.has_next() else 0
print(json.dumps({"event": "baseline", "count": count()}), flush=True)
deadline = time.time() + 20
while not os.path.exists(sentinel) and time.time() < deadline:
    time.sleep(0.05)
rec = {"event": "after"}
try:
    rec["count"] = count()  # query the HELD handle after the dir was rmtree'd+recreated
except BaseException as e:
    rec["reader_err"] = type(e).__module__ + "." + type(e).__name__ + ": " + str(e)[:200]
print(json.dumps(rec), flush=True)
try:
    conn.close(); db.close()
except BaseException:
    pass
'''


# ── Harness helpers ─────────────────────────────────────────────────────────────────────


def _stats_ms(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"n": 0}
    ms = sorted(s * 1000.0 for s in samples)
    n = len(ms)
    return {"n": n, "median": ms[n // 2], "mean": sum(ms) / n,
            "p90": ms[min(n - 1, math.ceil(0.9 * n) - 1)], "min": ms[0], "max": ms[-1]}


def _parse_events(stdout: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _count(conn: Any) -> int:
    r = conn.execute("MATCH (p:Probe) RETURN count(p) AS c;")
    return int(r.get_next()[0]) if r.has_next() else 0


def _seed(ryugraph: Any, path: str, n: int) -> None:
    """Create the Probe table + n nodes in-harness (read-write), then close."""
    db = ryugraph.Database(path, checkpoint_threshold=1)
    conn = ryugraph.Connection(db)
    conn.execute("CREATE NODE TABLE IF NOT EXISTS Probe(id STRING, PRIMARY KEY(id));")
    for i in range(n):
        conn.execute("MERGE (p:Probe {id: $id});", {"id": f"seed{i}"})
    conn.close()
    db.close()


# ── Probes ──────────────────────────────────────────────────────────────────────────────


def probe_r0(ryugraph: Any, tmp: Path, iters: int) -> dict[str, Any]:
    """R0 — decompose Database(path) vs Connection(db), read_only + read-write."""
    out: dict[str, Any] = {}
    for mode in ("ro", "rw"):
        path = str(tmp / f"r0_{mode}")
        _seed(ryugraph, path, 3)
        t_db: list[float] = []
        t_conn: list[float] = []
        for _ in range(iters):
            t = time.perf_counter()
            db = ryugraph.Database(path, read_only=(mode == "ro"))
            t_db.append(time.perf_counter() - t)
            t = time.perf_counter()
            conn = ryugraph.Connection(db)
            t_conn.append(time.perf_counter() - t)
            conn.close()
            db.close()
        out[mode] = {"database": _stats_ms(t_db), "connection": _stats_ms(t_conn)}
    return out


def probe_r0b(ryugraph: Any, tmp: Path, workers: int) -> dict[str, Any]:
    """R0b — N concurrent Connections minted on ONE shared read_only Database."""
    path = str(tmp / "r0b")
    _seed(ryugraph, path, 5)
    db = ryugraph.Database(path, read_only=True)

    def worker(_i: int) -> Any:
        try:
            conn = ryugraph.Connection(db)
            c = _count(conn)
            conn.close()
            return c
        except BaseException as e:  # capture, don't crash the pool
            return f"ERR {type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(worker, range(workers)))
    db.close()
    counts = [r for r in results if isinstance(r, int)]
    errs = [r for r in results if not isinstance(r, int)]
    ok = not errs and len(set(counts)) == 1
    return {"workers": workers, "ok": ok, "distinct_counts": sorted(set(counts)),
            "errors": errs[:3]}


def _run_cycles(py: str, tmp: Path, writer: Path, mode: str, cycles: int) -> dict[str, Any]:
    """Run the held-reader-across-N-write-cycles experiment in a CHILD process (crash-safe).
    Returns raw observations + the R1/R2 verdicts for this reader mode."""
    path = str(tmp / f"cyc_{mode}")
    # Seed via a writer subprocess so the table exists before the read_only reader opens.
    subprocess.run([py, str(writer), path, "merge", "seed0"], capture_output=True, text=True,
                   timeout=30, check=False)
    reader = tmp / "reader_cycles.py"
    proc = subprocess.run(
        [py, str(reader), path, mode, str(cycles), str(writer), py],
        capture_output=True, text=True, timeout=180, check=False,
    )
    events = _parse_events(proc.stdout)
    baseline = next((e["count"] for e in events if e.get("event") == "baseline"), None)
    cyc = [e for e in events if e.get("event") == "cycle"]
    final = next((c["count"] for c in reversed(cyc) if "count" in c), baseline)
    reader_err = next((c["reader_err"] for c in cyc if "reader_err" in c), None)
    writer_rcs = [c.get("writer_rc") for c in cyc]
    writer_bad = [rc for rc in writer_rcs if rc not in (0,)]
    writer_errs = [c["writer_err"] for c in cyc if c.get("writer_err")]
    writer_err_sample = writer_errs[0] if writer_errs else None
    writer_ms = [c["writer_ms"] for c in cyc if isinstance(c.get("writer_ms"), (int, float))]
    all_writers_failed = bool(cyc) and not any(rc == 0 for rc in writer_rcs)

    # ── R1 verdict (held-reader visibility + survival) ──
    if proc.returncode < 0:
        r1 = f"CRASH (reader child killed by signal {-proc.returncode} — likely SEGFAULT)"
    elif reader_err is not None:
        r1 = f"CORRUPT ({reader_err})"
    elif all_writers_failed:
        # No write ever committed while the reader held → staleness is UNTESTABLE here; the
        # writer was locked out (see R2). Do NOT report this as FAIL-STALE (a confound).
        r1 = "BLOCKED (every writer-B was locked out while reader-A held — see R2; staleness untestable)"
    elif baseline is None or final is None:
        r1 = f"INCONCLUSIVE (baseline={baseline} final={final}; rc={proc.returncode})"
    elif final > baseline:
        r1 = f"PASS-FRESH (held reader saw {final}, was {baseline} — re-reads cross-proc writes)"
    else:
        r1 = f"FAIL-STALE (held reader stuck at {baseline} despite a committed write — needs reopen)"

    # ── R2 verdict (writer success/block while reader-A holds) ──
    if proc.returncode < 0 or reader_err is not None:
        r2 = "writer-side moot (reader crashed/corrupted — see R1)"
    elif not writer_bad:
        med = sorted(writer_ms)[len(writer_ms) // 2] if writer_ms else float("nan")
        r2 = f"PASS (writer-B succeeded every cycle while reader-A[{mode}] held; median {med:.1f}ms)"
    elif "timeout" in writer_bad:
        r2 = f"BLOCK (writer-B timed out while reader-A[{mode}] held — held reader blocks the writer)"
    else:
        msg = f" [{writer_err_sample.splitlines()[-1][:90]}]" if writer_err_sample else ""
        r2 = f"FAIL (writer-B locked out while reader-A[{mode}] held: rcs={writer_rcs[:5]}){msg}"

    return {"mode": mode, "returncode": proc.returncode, "baseline": baseline, "final": final,
            "cycles_run": len(cyc), "writer_rcs": writer_rcs[:8], "reader_err": reader_err,
            "writer_err_sample": writer_err_sample, "all_writers_failed": all_writers_failed,
            "stderr_tail": (proc.stderr or "").strip()[-300:], "r1": r1, "r2": r2}


def probe_r3(ryugraph: Any, py: str, tmp: Path, writer: Path) -> dict[str, Any]:
    """R3 — a FRESH Database (new object) after a cross-process write sees it? (current design)."""
    out: dict[str, Any] = {}
    for mode in ("ro", "rw"):
        path = str(tmp / f"r3_{mode}")
        _seed(ryugraph, path, 2)
        # open+close a handle (the per-request reader), then a writer subprocess writes
        db = ryugraph.Database(path, read_only=(mode == "ro"))
        conn = ryugraph.Connection(db)
        before = _count(conn)
        conn.close()
        db.close()
        subprocess.run([py, str(writer), path, "merge", "r3new"], capture_output=True, text=True,
                       timeout=30, check=False)
        # FRESH Database object in the SAME process
        db2 = ryugraph.Database(path, read_only=(mode == "ro"))
        conn2 = ryugraph.Connection(db2)
        after = _count(conn2)
        conn2.close()
        db2.close()
        verdict = ("PASS (fresh reopen saw the cross-proc write)" if after > before
                   else f"FAIL-STALE (fresh reopen stuck at {before} — process page-cache bug!)")
        out[mode] = {"before": before, "after": after, "verdict": verdict}
    return out


def probe_r4(ryugraph: Any, py: str, tmp: Path, writer: Path) -> dict[str, Any]:
    """R4 — held handle across a teardown (rmtree + recreate, = reindex --force)."""
    path = str(tmp / "r4")
    _seed(ryugraph, path, 4)
    stat_before = os.stat(path)
    sentinel = tmp / "r4.sentinel"
    reader = tmp / "teardown_reader.py"
    proc = subprocess.Popen([py, str(reader), path, str(sentinel)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.stdout is None:  # PIPE is set above, so this never fires — narrows the type for the read
        raise RuntimeError("teardown reader subprocess has no stdout pipe")
    base_line = proc.stdout.readline()  # block until the reader has opened + counted
    # Teardown: delete the dir and recreate it with fresh content (a different graph).
    shutil.rmtree(path, ignore_errors=True)
    subprocess.run([py, str(writer), path, "merge", "afterTeardown"], capture_output=True,
                   text=True, timeout=30, check=False)
    stat_after = os.stat(path)
    sentinel.write_text("go")
    try:
        rest_out, rest_err = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        rest_out, rest_err = proc.communicate()
    rc = proc.returncode
    events = _parse_events(base_line + rest_out)
    after = next((e for e in events if e.get("event") == "after"), {})
    if rc < 0:
        verdict = f"CRASH (held reader killed by signal {-rc} on teardown)"
    elif "reader_err" in after:
        verdict = f"ERROR-then-alive ({after['reader_err']})"
    else:
        verdict = f"STALE-OR-OK (held reader returned count={after.get('count')} after teardown)"
    return {
        "verdict": verdict, "returncode": rc, "after": after,
        "stderr_tail": (rest_err or "").strip()[-300:],
        "ino_changed": stat_before.st_ino != stat_after.st_ino,
        "mtime_changed": stat_before.st_mtime_ns != stat_after.st_mtime_ns,
        "size_changed": stat_before.st_size != stat_after.st_size,
    }


def probe_r6(ryugraph: Any, tmp: Path, workers: int) -> dict[str, Any]:
    """R6 — IN-PROCESS concurrent Database(path) opens (the concurrent-discovery-read case).
    The current code mints a fresh Database per call via asyncio.to_thread, so two concurrent
    webui reads = two Database(path) objects in one process. PASS here ⇒ the exclusive lock is
    per-PROCESS (not per-object), so the R2 collision is CROSS-process only (webui vs writer)."""
    path = str(tmp / "r6")
    _seed(ryugraph, path, 5)
    errs: list[str] = []

    def worker(_i: int) -> None:
        try:
            db = ryugraph.Database(path)  # read-WRITE, exactly like GraphStore.open today
            conn = ryugraph.Connection(db)
            _count(conn)
            time.sleep(0.05)  # widen the overlap window so opens genuinely coexist
            conn.close()
            db.close()
        except BaseException as e:  # capture the lock error if it fires in-process
            errs.append(f"{type(e).__name__}: {str(e).splitlines()[0][:90]}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(worker, range(workers)))
    return {"workers": workers, "ok": not errs, "errors": errs[:3]}


def probe_r5(ryugraph: Any, tmp: Path) -> dict[str, Any]:
    """R5 — read_only cold-start: missing DB, and existing-but-no-schema dir."""
    out: dict[str, Any] = {}

    def attempt(label: str, path: str, make_empty_dir: bool) -> None:
        if make_empty_dir:
            Path(path).mkdir(parents=True, exist_ok=True)
        try:
            db = ryugraph.Database(path, read_only=True)
            conn = ryugraph.Connection(db)
            _count(conn)  # the failure may surface at query, not open
            conn.close()
            db.close()
            out[label] = "OPENED+QUERIED OK (no error)"
        except BaseException as e:  # we WANT the exception class
            out[label] = f"{type(e).__module__}.{type(e).__name__}: {str(e)[:160]}"

    attempt("missing_db", str(tmp / "r5_missing"), make_empty_dir=False)
    attempt("empty_dir_no_schema", str(tmp / "r5_empty"), make_empty_dir=True)
    return out


# ── Main ────────────────────────────────────────────────────────────────────────────────


def _write_scripts(tmp: Path) -> Path:
    (tmp / "writer.py").write_text(textwrap.dedent(_WRITER).lstrip())
    (tmp / "reader_cycles.py").write_text(textwrap.dedent(_READER_CYCLES).lstrip())
    (tmp / "teardown_reader.py").write_text(textwrap.dedent(_TEARDOWN_READER).lstrip())
    return tmp / "writer.py"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycles", type=int, default=25, help="held-reader write cycles (R1/R2)")
    ap.add_argument("--iters", type=int, default=15, help="open-cost iterations (R0)")
    ap.add_argument("--workers", type=int, default=16, help="concurrent connections (R0b)")
    ap.add_argument("--json", action="store_true", help="also emit the raw observations as JSON")
    args = ap.parse_args()

    try:
        import ryugraph
    except ImportError as e:
        print(f"ryugraph not importable ({e}) — install with `uv sync`. Cannot probe.")
        return 2

    print(f"ryugraph {getattr(ryugraph.Database, 'get_version', lambda: '?')()}  "
          f"cycles={args.cycles} iters={args.iters} workers={args.workers}")
    tmp = Path(tempfile.mkdtemp(prefix="ryugraph_probe_"))
    py = sys.executable
    raw: dict[str, Any] = {}
    try:
        writer = _write_scripts(tmp)

        raw["R0"] = probe_r0(ryugraph, tmp, args.iters)
        raw["R0b"] = probe_r0b(ryugraph, tmp, args.workers)
        raw["R1R2_ro"] = _run_cycles(py, tmp, writer, "ro", args.cycles)
        raw["R1R2_rw"] = _run_cycles(py, tmp, writer, "rw", args.cycles)
        raw["R3"] = probe_r3(ryugraph, py, tmp, writer)
        raw["R4"] = probe_r4(ryugraph, py, tmp, writer)
        raw["R5"] = probe_r5(ryugraph, tmp)
        raw["R6"] = probe_r6(ryugraph, tmp, args.workers)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ── Verdict report ──
    print(f"\n{'=' * 80}\nVERDICTS\n{'=' * 80}")
    r0 = raw["R0"]
    for mode in ("ro", "rw"):
        d, c = r0[mode]["database"], r0[mode]["connection"]
        print(f"[R0  {mode}] Database(path)={d['median']:6.2f}ms   Connection(db)={c['median']:6.2f}ms")
    b = raw["R0b"]
    print(f"[R0b   ] {'PASS' if b['ok'] else 'FAIL'}: {b['workers']} concurrent Connections on one "
          f"read_only Database; distinct_counts={b['distinct_counts']} errors={b['errors']}")
    print(f"[R1 ro ] {raw['R1R2_ro']['r1']}")
    print(f"[R1 rw ] {raw['R1R2_rw']['r1']}")
    print(f"[R2 ro ] {raw['R1R2_ro']['r2']}")
    print(f"[R2 rw ] {raw['R1R2_rw']['r2']}")
    for mode in ("ro", "rw"):
        print(f"[R3 {mode}] {raw['R3'][mode]['verdict']}")
    r4 = raw["R4"]
    print(f"[R4    ] {r4['verdict']}  | ino_changed={r4['ino_changed']} "
          f"mtime_changed={r4['mtime_changed']} size_changed={r4['size_changed']}")
    print(f"[R5    ] missing_db: {raw['R5']['missing_db']}")
    print(f"[R5    ] empty_dir : {raw['R5']['empty_dir_no_schema']}")
    r6 = raw["R6"]
    r6_detail = ("all coexist (lock is per-PROCESS → R2 is cross-process only)" if r6["ok"]
                 else f"errors={r6['errors']}")
    print(f"[R6    ] {'PASS' if r6['ok'] else 'FAIL'}: {r6['workers']} concurrent IN-PROCESS "
          f"Database(path) opens — {r6_detail}")

    print(f"\n{'=' * 80}\nFIX-WORLD READING\n{'=' * 80}")
    _print_world(raw)

    if args.json:
        print("\n" + json.dumps(raw, indent=1, default=str))
    return 0


def _print_world(raw: dict[str, Any]) -> None:
    ro, rw = raw["R1R2_ro"], raw["R1R2_rw"]
    crash_or_corrupt = any(d["r1"].startswith(("CRASH", "CORRUPT")) for d in (ro, rw))
    # The dominant axis: does a HELD reader (either mode) lock out a cross-process writer?
    lock_conflict = any(d["r2"].startswith(("FAIL", "BLOCK")) for d in (ro, rw))
    r3_stale = any(raw["R3"][m]["verdict"].startswith("FAIL") for m in ("ro", "rw"))
    if r3_stale:
        print("  ⚠ CURRENT BUG: R3 shows a fresh reopen serves stale process-cached pages — the "
              "existing per-request design is NOT safe in a long-lived process. Fix unconditionally.")
    if lock_conflict:
        print("  → EXCLUSIVE-LOCK (the reality): ryugraph locks the DB directory on ANY open (even "
              "read_only); a held handle locks out ALL cross-process writers ('Could not set lock "
              "on file'). A process-lifetime cached handle is ARCHITECTURALLY PRECLUDED — it would "
              "break every concurrent enrich/index/reindex. KEEP open-per-request (R3 confirms it's "
              "correct + fresh). The 'staleness' concern was mis-framed: it's LOCKING, not staleness.")
        scope = ("CROSS-process only (concurrent in-process reads coexist — R6 PASS)"
                 if raw.get("R6", {}).get("ok") else "in-process TOO (R6 FAIL — see errors)")
        print(f"  → COLLISION SCOPE: {scope}; i.e. a webui discovery read vs a separate "
              "enrich/index/reindex process.")
        print("  → REAL fix surface: the lock-contention RuntimeError from that brief reader/writer "
              "race is currently UNCAUGHT in the discovery readers AND _open_graph (only ImportError "
              "is caught) → fail-open (or bounded retry) on it, so a concurrent write never 500s a "
              "discovery read (and a read never fails an enrich step).")
    elif crash_or_corrupt:
        print("  → World C FLOOR: a held read_only reader is crashed/corrupted by a writer's "
              "checkpoint. A concurrently-live cache is OFF the table — only hard mutual exclusion.")
    elif ro["r1"].startswith("FAIL-STALE"):
        print("  → World B: held read_only reader is SAFE but STALE — reuse needs generation-gated "
              "close+reopen on graph change.")
    elif ro["r1"].startswith("PASS-FRESH") and ro["r2"].startswith("PASS"):
        print("  → World A: held read_only reader sees writes AND coexists with the writer — reuse "
              "is safe + minimal (reopen only on teardown).")
    else:
        print("  → INCONCLUSIVE — inspect the raw R1/R2 observations (--json).")
    if not raw["R5"]["missing_db"].startswith("OPENED"):
        print(f"  • cold-start catch class (R5): {raw['R5']['missing_db'].split(':')[0]} "
              "→ the discovery readers must ALSO fail-open on this (a never-enriched/empty graph).")


if __name__ == "__main__":
    sys.exit(main())
