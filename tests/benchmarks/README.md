# `tests/benchmarks/` — performance regression gates

This directory holds the **baseline** JSON report that
`scripts/benchmark.py --gate` compares against. The baseline is
committed; CI fails if any metric regresses more than 15% (per
GUIDELINES.md Part VI).

## First-time setup

Run the benchmark locally, commit the result:

```sh
uv run python scripts/benchmark.py --output tests/benchmarks/baseline.json
git add tests/benchmarks/baseline.json
git commit -m "bench: seed baseline"
```

## On every PR

CI runs the benchmark on a known runner and compares:

```sh
uv run python scripts/benchmark.py --output current.json
uv run python scripts/benchmark.py --gate tests/benchmarks/baseline.json --output current.json
```

A regression > 15% (per metric) fails the build. Warnings between 5%
and 15% print to stderr but don't fail.

## Updating the baseline

If the regression is accepted (e.g. a deliberate quality-over-speed
trade with an ADR), re-seed the baseline:

```sh
uv run python scripts/benchmark.py --output tests/benchmarks/baseline.json
git add tests/benchmarks/baseline.json
git commit -m "bench: update baseline after <reason>"
```

## Modes

- **Default (`--fake`)** — pure-Python orchestration measurements:
  chunker throughput, vault write latency, FTS5 query latency, agent
  state-machine cycle. No GPU required. CI runs this tier on every PR.

- **`--real`** — adds real-model benchmarks (cold start, embedding
  throughput, first-token latency). Requires a reachable vLLM and the
  full `[models]` extras installed. Run nightly on the reference rig.

## What's measured

| Metric | Mode | Target | Floor |
|---|---|---|---|
| `chunker.throughput` | fake | > 5000 chunks/sec | > 1000 chunks/sec |
| `vault.write.latency` | fake | < 20 ms | < 100 ms |
| `fts.query.latency` | fake | < 5 ms | < 50 ms |
| `agent.cycle.latency` | fake | < 50 ms | < 250 ms |
| `query.first_token.latency` | real | < 2000 ms | < 4000 ms |

See `scripts/benchmark.py` for the full list and the measurement code.
