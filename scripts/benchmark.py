"""End-to-end benchmark harness — see GUIDELINES.md Part VI "Performance".

Two modes:

  scripts/benchmark.py                     # default: --fake, no GPU required
  scripts/benchmark.py --real              # adds real-model benchmarks
                                           # (requires reachable vLLM + GPU)
  scripts/benchmark.py --gate BASELINE     # compare current run against
                                           # BASELINE (JSON), exit 1 if any
                                           # metric regresses > 15%

Both modes write a JSON report to stdout (or `--output` file). Pair with
a CI workflow that uploads the report and gates merges on `--gate`
exit code. A starter `.github/workflows/benchmark.yml` ships with the
repo; adapt for your CI of choice.

Targets (from GUIDELINES.md):

  - Cold start                  < 30 s   (real)
  - Embedding throughput        > 500 chunks/sec  (real)
  - First-token latency         < 2 s    (real)
  - Full grounded answer        < 15 s   (real)
  - Chunker throughput          > 5000 chunks/sec (fake)
  - Agent state-machine cycle   < 50 ms  (fake)

The fake-mode metrics measure pure-Python orchestration overhead — they
catch regressions in the state machine, atomic writes, FTS queries,
and chunker without needing a GPU. CI on a CPU-only runner gates on
those; the reference rig gates on the real-mode set nightly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

# Ensure `src/memex/` is importable when running this script directly.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


# ----- Result + report types (local; this script doesn't import pydantic) -----


class _Result:
    """Lightweight result holder — avoids a pydantic dep at script scope."""

    __slots__ = ("name", "value", "unit", "lower_is_better", "target", "floor", "ok", "metadata")

    def __init__(
        self,
        name: str,
        value: float,
        unit: str,
        *,
        lower_is_better: bool = True,
        target: float | None = None,
        floor: float | None = None,
        ok: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.value = value
        self.unit = unit
        self.lower_is_better = lower_is_better
        self.target = target
        self.floor = floor
        self.ok = ok
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "lower_is_better": self.lower_is_better,
            "target": self.target,
            "floor": self.floor,
            "ok": self.ok,
            "metadata": self.metadata,
        }


def _env() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": str(os.cpu_count() or 0),
    }


# ----- Fake-mode benchmarks (no GPU; pure-Python orchestration) -----


def _synth_markdown(target_bytes: int) -> str:
    """Build a synthetic markdown body of roughly `target_bytes` size.

    Uses repeating headings + paragraphs so the chunker sees real
    structure (heading-aware splitting, paragraph boundaries).
    """
    para = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna "
        "aliqua. Ut enim ad minim veniam, quis nostrud exercitation. "
    )
    out: list[str] = []
    size = 0
    section = 0
    while size < target_bytes:
        section += 1
        h = f"\n## Section {section}\n\n"
        out.append(h)
        size += len(h)
        for sub in range(3):
            sh = f"### Subsection {section}.{sub + 1}\n\n"
            out.append(sh)
            size += len(sh)
            for _ in range(4):
                out.append(para + "\n\n")
                size += len(para) + 2
    return "".join(out)


def bench_chunker() -> _Result:
    """Chunk a 100 KB synthetic markdown body, measure throughput."""
    from memex.index.chunker import chunk_document
    from memex.vault.store import DocumentRef, Frontmatter, VaultDocument

    body = _synth_markdown(100_000)
    doc = VaultDocument(
        ref=DocumentRef(
            doc_id="bench-chunker-0",
            markdown_path=Path("/tmp/bench.md"),  # noqa: S108 — never opened
            asset_dir=Path("/tmp/bench"),  # noqa: S108
            source_path=None,
            content_sha256="0" * 64,
        ),
        frontmatter=Frontmatter(title="bench"),
        body=body,
        mtime_ns=0,
    )
    # Warm-up (one pass) to amortise import + JIT, then time the real pass.
    chunk_document(doc)
    start = time.monotonic()
    chunks = chunk_document(doc)
    elapsed = time.monotonic() - start
    rate = len(chunks) / elapsed if elapsed > 0 else 0.0
    return _Result(
        "chunker.throughput",
        rate,
        "chunks/sec",
        lower_is_better=False,
        target=5000,
        floor=1000,
        metadata={"body_bytes": len(body), "chunks": len(chunks)},
    )


def bench_vault_write() -> _Result:
    """Measure `create_document` round-trip latency (atomic write + sha256)."""
    import asyncio

    from memex.core.config import MemexSettings, set_settings
    from memex.vault.store import create_document

    body = _synth_markdown(20_000)
    with tempfile.TemporaryDirectory() as td:
        os.environ["MEMEX_VAULT_PATH"] = td
        os.environ["MEMEX_OBSERVABILITY__LANGFUSE_ENABLED"] = "false"
        settings = MemexSettings()  # type: ignore[call-arg]
        set_settings(settings)
        try:
            async def _one(i: int):
                await create_document(
                    settings.vault_path,
                    body=body,
                    source_stem=f"bench-{i}",
                )

            # Warm + measure 20 writes.
            asyncio.run(_one(0))
            start = time.monotonic()
            for i in range(1, 21):
                asyncio.run(_one(i))
            elapsed = time.monotonic() - start
        finally:
            set_settings(None)
    per_write_ms = (elapsed / 20) * 1000
    return _Result(
        "vault.write.latency",
        per_write_ms,
        "ms",
        lower_is_better=True,
        target=20,
        floor=100,
        metadata={"iterations": 20, "body_bytes": len(body)},
    )


def bench_fts_query() -> _Result:
    """Index 500 chunks into SQLite FTS5, then measure single-query latency."""
    import asyncio

    from memex.core.types import Chunk
    from memex.index.fts_store import FTSStore

    chunks = [
        Chunk(
            chunk_id=f"bench#{i}",
            document_id="bench-doc",
            document_title="Bench Doc",
            text=f"Reflexivity in research design segment number {i}. "
            "The data they collect shapes the researcher's interpretation.",
            page=i // 10,
        )
        for i in range(500)
    ]

    async def _run() -> tuple[float, int]:
        with tempfile.TemporaryDirectory() as td:
            vault = Path(td)
            (vault / ".memex").mkdir(parents=True, exist_ok=True, mode=0o700)
            store = await FTSStore.open(vault)
            try:
                await store.upsert(chunks)
                # Warm.
                await store.search("reflexivity", k=10)
                start = time.monotonic()
                for _ in range(50):
                    await store.search("reflexivity research design", k=10)
                elapsed = time.monotonic() - start
            finally:
                await store.close()
            return elapsed, len(chunks)

    elapsed, n = asyncio.run(_run())
    per_query_ms = (elapsed / 50) * 1000
    return _Result(
        "fts.query.latency",
        per_query_ms,
        "ms",
        lower_is_better=True,
        target=5,
        floor=50,
        metadata={"corpus_chunks": n, "iterations": 50},
    )


def bench_agent_cycle() -> _Result:
    """End-to-end answering-agent cycle with every I/O faked.

    Measures the LangGraph state-machine + pydantic-validation +
    structlog binding cost per query. With real models this is dwarfed
    by inference time; the fake-mode measurement catches regressions in
    the orchestration layer.
    """
    import asyncio
    from typing import Any

    from memex.agents.answering import (
        Chunk,
        CitedClaim,
        DraftAnswer,
        SufficiencyAssessment,
        VerificationResult,
        answer_query,
        reset_compiled_graph,
    )

    canned_chunks = [
        Chunk(
            chunk_id="b#a",
            document_id="b",
            document_title="Bench",
            text="A.",
        )
    ]

    async def _fake_hybrid(q: str, k: int = 50) -> list[Chunk]:
        return list(canned_chunks)

    async def _fake_rerank(q: str, candidates: list[Chunk], top_k: int = 10) -> list[Chunk]:
        return list(candidates[:top_k])

    async def _fake_structured(*, prompt: str, schema: type, **_kw: Any) -> tuple[Any, int]:
        if schema is SufficiencyAssessment:
            return SufficiencyAssessment(sufficient=True, reason="ok"), 1
        if schema is DraftAnswer:
            return (
                DraftAnswer(
                    summary="ok",
                    claims=[
                        CitedClaim(
                            claim="bench claim", source_chunk_id="b#a", confidence="high"
                        )
                    ],
                ),
                1,
            )
        if schema is VerificationResult:
            return VerificationResult(grounded=[0], ungrounded=[]), 1
        raise AssertionError(f"unexpected schema {schema}")

    async def _run() -> float:
        import memex.agents.answering as ans

        ans.hybrid_search = _fake_hybrid  # type: ignore[assignment]
        ans.cross_encoder_rerank = _fake_rerank  # type: ignore[assignment]
        ans.render_prompt = lambda name, **_kw: f"[{name}]"  # type: ignore[assignment]
        ans.complete_structured = _fake_structured  # type: ignore[assignment]

        reset_compiled_graph()
        # Warm (build the graph + LangGraph compile).
        await answer_query("warm")
        start = time.monotonic()
        for i in range(20):
            await answer_query(f"benchmark query {i}")
        elapsed = time.monotonic() - start
        return elapsed

    os.environ.setdefault("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    elapsed = asyncio.run(_run())
    per_cycle_ms = (elapsed / 20) * 1000
    return _Result(
        "agent.cycle.latency",
        per_cycle_ms,
        "ms",
        lower_is_better=True,
        target=50,
        floor=250,
        metadata={"iterations": 20, "model_calls_per_cycle": 3},
    )


# ----- Real-mode benchmarks (require a reachable vLLM + GPU) -----


async def _vllm_reachable() -> bool:
    """Probe the configured base_url. Cheap GET on /models."""
    try:
        from memex.core.config import MemexSettings
        from memex.models.client import configure_client, get_client

        settings = MemexSettings()  # type: ignore[call-arg]
        configure_client(settings.inference)
        client = get_client()
        await client.models.list()
        return True
    except Exception:
        return False


async def bench_real_first_token() -> _Result:
    """Time to first token from a small completion against the live vLLM."""
    from openai import AsyncOpenAI

    from memex.core.config import MemexSettings

    settings = MemexSettings()  # type: ignore[call-arg]
    client = AsyncOpenAI(
        base_url=settings.inference.base_url,
        api_key=settings.inference.api_key,
    )

    # Warm.
    await client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=4,
    )

    start = time.monotonic()
    stream = await client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Reply with the word 'ready'."}],
        max_tokens=8,
        stream=True,
    )
    first_token_ms = -1.0
    async for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if choices and (choices[0].delta.content or "").strip():
            first_token_ms = (time.monotonic() - start) * 1000
            break

    return _Result(
        "query.first_token.latency",
        first_token_ms,
        "ms",
        lower_is_better=True,
        target=2000,
        floor=4000,
    )


# ----- Runner + reporting -----


def _collect(mode: Literal["fake", "real"], real_skipped: list[str]) -> list[_Result]:
    results: list[_Result] = []

    # Fake-mode benchmarks always run.
    for fn in (bench_chunker, bench_vault_write, bench_fts_query, bench_agent_cycle):
        try:
            results.append(fn())
        except Exception as e:
            results.append(
                _Result(
                    fn.__name__,
                    -1.0,
                    "n/a",
                    ok=False,
                    metadata={"error": str(e)},
                )
            )

    if mode == "real":
        # Real-mode benchmarks need a reachable vLLM. Probe first.
        if asyncio.run(_vllm_reachable()):
            try:
                results.append(asyncio.run(bench_real_first_token()))
            except Exception as e:
                results.append(
                    _Result(
                        "query.first_token.latency",
                        -1.0,
                        "ms",
                        ok=False,
                        metadata={"error": str(e)},
                    )
                )
        else:
            real_skipped.append("vLLM unreachable at configured base_url")

    return results


def _report(mode: Literal["fake", "real"], results: list[_Result], skipped: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": f"bench-{int(time.time())}",
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": mode,
        "environment": _env(),
        "skipped": skipped,
        "results": [r.to_dict() for r in results],
    }


# ----- Regression gate -----


def _gate(baseline_path: Path, current: dict[str, Any], threshold: float) -> int:
    """Compare `current` against `baseline_path`. Exit 1 when any of:

      - a current metric regressed > `threshold` from baseline
      - a baseline metric is missing from the current run (drop signal)
      - a baseline metric is present in current but `ok=False` (silent
        infrastructure breakage that hid the measurement)
    """
    if not baseline_path.exists():
        print(
            f"benchmark: no baseline at {baseline_path}; skipping gate (informational only)",
            file=sys.stderr,
        )
        return 0
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_by_name = {r["name"]: r for r in baseline["results"]}
    current_by_name = {r["name"]: r for r in current["results"]}

    regressions: list[str] = []
    warnings: list[str] = []
    missing: list[str] = []
    errored: list[str] = []

    for name, b in baseline_by_name.items():
        if not b.get("ok"):
            # Baseline metric was already broken — informational, not a gate trip.
            continue
        c = current_by_name.get(name)
        if c is None:
            missing.append(name)
            continue
        if not c["ok"]:
            errored.append(
                f"{name}: {c.get('metadata', {}).get('error', 'unknown error')}"
            )
            continue
        if b["value"] <= 0:
            continue
        if c["lower_is_better"]:
            delta = (c["value"] - b["value"]) / b["value"]
        else:
            delta = (b["value"] - c["value"]) / b["value"]
        if delta > threshold:
            regressions.append(
                f"{name}: {b['value']:.2f} {b['unit']} → "
                f"{c['value']:.2f} {c['unit']} ({delta * 100:+.1f}%)"
            )
        elif delta > 0.05:
            warnings.append(
                f"{name}: {b['value']:.2f} → {c['value']:.2f} {c['unit']} "
                f"({delta * 100:+.1f}%)"
            )

    if warnings:
        print("benchmark: warnings (>5% regression, below gate):", file=sys.stderr)
        for w in warnings:
            print(f"  {w}", file=sys.stderr)

    if regressions or missing or errored:
        if regressions:
            print(
                f"benchmark: FAILED — {len(regressions)} metric(s) regressed "
                f"> {threshold * 100:.0f}%:",
                file=sys.stderr,
            )
            for r in regressions:
                print(f"  {r}", file=sys.stderr)
        if errored:
            print(
                f"benchmark: FAILED — {len(errored)} metric(s) errored in current run:",
                file=sys.stderr,
            )
            for e in errored:
                print(f"  {e}", file=sys.stderr)
        if missing:
            print(
                f"benchmark: FAILED — {len(missing)} baseline metric(s) missing "
                "from current run:",
                file=sys.stderr,
            )
            for m in missing:
                print(f"  {m}", file=sys.stderr)
        return 1

    print(
        f"benchmark: PASSED — all {len(baseline_by_name)} baseline metrics within "
        f"{threshold * 100:.0f}% of baseline",
        file=sys.stderr,
    )
    return 0


# ----- CLI -----


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Memex benchmark harness — see scripts/benchmark.py docstring."
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Add real-model benchmarks (require a reachable vLLM endpoint).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the JSON report to this file (otherwise stdout).",
    )
    parser.add_argument(
        "--gate",
        type=Path,
        metavar="BASELINE_JSON",
        help="Compare current results against BASELINE_JSON; exit 1 on regression > 15%%.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Regression threshold for --gate (default: 0.15 = 15%%).",
    )
    args = parser.parse_args(argv)

    mode: Literal["fake", "real"] = "real" if args.real else "fake"
    skipped: list[str] = []
    results = _collect(mode, skipped)
    report = _report(mode, results, skipped)
    output_text = json.dumps(report, indent=2, sort_keys=True)

    if args.output:
        args.output.write_text(output_text + "\n", encoding="utf-8")
    else:
        print(output_text)

    if args.gate is not None:
        return _gate(args.gate, report, args.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
