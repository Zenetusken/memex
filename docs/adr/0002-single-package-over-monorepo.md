# ADR-0002: Single Python Package Over Monorepo

- **Status**: Accepted
- **Date**: 2026-05-19
- **Deciders**: Memex core team
- **Tags**: architecture, repo-layout, process

## Context

The original DocuFlow guidelines specified a seven-package monorepo (`@docuflow/core`, `@docuflow/ocr-engines`, `@docuflow/document-parser`, `@docuflow/notion-sync`, `@docuflow/ui-components`, `@docuflow/storage`, `@docuflow/queue`). That structure assumed (a) a team large enough that several engineers would be working in different packages simultaneously, (b) at least one package would have external consumers worth packaging for, and (c) the operational overhead of maintaining inter-package versioning would pay off in stronger boundaries.

None of those assumptions hold for Memex. We are a solo-dev / very-small-team project. There are no external consumers. The boundaries we need are module boundaries, enforceable by import discipline and linting, not by separate package boundaries. The cost of a monorepo — slower refactors, version drift between internal packages, more complex CI, mental overhead for every cross-package change — is real and immediate; the benefit is theoretical and deferred.

## Decision Drivers

- Solo-dev / small-team velocity is the dominant constraint
- Refactor cost across "module boundaries" should be near-zero
- No external consumers exist for v1
- Python's import system gives us cheap module isolation
- CI complexity should scale with the project, not be paid up front

## Considered Options

1. **Single Python package**, modules under `src/memex/`, boundaries enforced by import discipline
2. **uv workspace** with multiple internal packages, single repository
3. **Separate repositories** per concern (rejected as obviously worse for this scale)

## Decision

**One Python package.** All code lives under `src/memex/` with clear module separation. Boundaries are enforced by (a) import discipline (modules use only public symbols of other modules, never private ones), (b) lint rules that flag cross-module access to private symbols, (c) architectural tests that verify the dependency graph at PR time.

## Consequences

### Positive

- Simplest possible CI — one install, one test run, one lint pass
- No internal package version coordination ever
- Refactors that touch "module boundaries" are atomic — rename a function, fix all callers, ship in one PR
- Lower mental overhead for every change
- Faster onboarding — one mental model, not seven
- `uv sync` reproducibility covers the entire codebase from one lockfile

### Negative / Trade-offs

- Module isolation depends on discipline plus tooling, not on a package wall. A motivated developer can reach across boundaries; we rely on linting and code review to catch this.
- Extracting a public library later (if a module matures enough to be worth shipping standalone) requires real work, not a `pyproject.toml` toggle. We accept this cost as deferred.
- "Everyone touches everything" is a real risk as the team grows. Mitigated by import-graph checks in CI and by the revisit trigger below.

### Neutral

- The directory structure still reflects module boundaries, so the layout reads like a monorepo from the outside. The difference is operational, not architectural.

## Alternatives in Detail

### uv workspace with internal packages

The middle-ground option. Each module becomes its own package (`memex-core`, `memex-parse`, `memex-agents`) inside a single repo, with uv's workspace feature managing them together. Benefits: cleaner enforcement of boundaries, possible to publish individual packages later.

Rejected because the v1 cost is real and the v1 benefit is zero. None of these packages have an external consumer; the boundaries don't need to be defended against anyone. If a specific module matures into something the broader ecosystem wants — say, a "fully local agentic document parser" library — that's the trigger to extract it, with the benefit of a year of usage informing the public API. Doing it preemptively bakes in the wrong abstractions.

### Separate repositories

Strictly worse than the workspace option for our scale. No serious consideration.

## Revisit When

- Team grows past 3–4 active contributors with meaningful module specialization
- A specific module reaches maturity AND has demonstrable external interest, justifying extraction
- The dependency graph develops bidirectional cycles that lint rules cannot resolve cleanly
- CI runtime exceeds 10 minutes on a typical machine (unlikely for pure Python at this scale)

## References

- Memex developer guidelines, §"Project layout"
- ADR-0001 (vLLM): the inference layer is the only candidate for early extraction, if anyone wants a "Memex inference helpers" package
