# Audit reports

Standing record of correctness/security/wiring audits run against the
codebase. Each audit is captured here so that the "why" behind a class
of fixes survives long after the commits land.

| Date | Audit | Method |
|---|---|---|
| 2026-05-19 | Stack currency | single agent — see `memory/stack_currency_audit.md` (off-tree, time-sensitive) |
| 2026-05-20 | CUDA dispatch | four parallel agents → ADR-0006 |
| 2026-05-20 | Multi-agent bug-hunt | four parallel agents (resource/concurrency, error/edge, wiring/signatures, quality) — see `00-synthesis.md` through `04-wiring.md` |
| 2026-05-20 | E2E + load test on RTX 4070 | live verification of every audit fix — see `05-e2e-loadtest.md` |
| 2026-05-20 | 8B-AWQ load test on RTX 4070 | post-tuning verification of the production-target orchestrator — see `06-8b-loadtest.md` |
| 2026-05-20 | OCR off vs on (109-page slide deck) | empirical A/B settling the parse-default question on born-digital PDFs — see `07-ocr-ab.md` |

## Pattern

When you run a new audit, follow the 2026-05-20 template:

1. **Fan out** to specialist subagents, each with a focused lens.
2. **Synthesise** their findings into a deduplicated priority list (the `00-synthesis.md` file).
3. **Fix in batches** ordered by severity (security → broken-now → data-loss → correctness → drift).
4. **Verify live** on the reference rig — every fix should have an observable signal that confirms it landed (`05-e2e-loadtest.md` is the model).

The four-agent pattern is the user-approved standard. New patterns are fine; document them here when you try one.

## On what NOT to commit here

- **In-progress findings** (the raw subagent output before synthesis).
- **Audits whose conclusions never made it into code** — file an issue or an ADR if they're worth keeping; this directory is for the record of *what happened*, not what should happen.
- **Time-sensitive observations that decay quickly** (e.g., "PyPI is rate-limiting today"). Those go in memory under `~/.claude/projects/.../memory/`, not in-tree.
