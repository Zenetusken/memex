# ADR-0005: RyuGraph Replaces Kuzu for the Embedded Graph Store

- **Status**: Accepted
- **Date**: 2026-05-19
- **Deciders**: Memex core team
- **Tags**: stack, storage, supersedes-implicit-from-0003

## Context

ADR-0003 named Kuzu as the embedded property-graph store for entity and citation traversal. The multi-agent stack-currency audit on 2026-05-19 surfaced that Kuzu Inc. wound down: final release `v0.11.3` in October 2025, repository archived shortly after, with reported acquisition of the team in February 2026. The library still works on disk, but a dead upstream conflicts with Memex's vision principle "Honestly open source / if we stop maintaining it, you don't lose anything" — that principle is also a constraint on the dependencies we run.

Three credible community forks emerged. **RyuGraph** (Predictable Labs, `v25.9.2`) is the most active, with the same Cypher dialect, the same property-graph model, and a near-drop-in lineage from Kuzu's on-disk format. **Bighorn** (Kineviz) has no formal releases yet; **Ladybug** is a smaller community effort.

Critically, **no Memex code has been written against the graph store yet** — `src/memex/index/graph_store.py` is unimplemented. The swap cost today is updating ADRs, the stack table, and a `pyproject.toml` pin. Doing it before Phase 2 implements `memex.index.graph_store` avoids ever shipping against an archived dependency.

## Decision Drivers

- Local-first commitment requires our dependencies to be maintained, not just open
- Kuzu's API choices (Cypher, embedded, columnar, no server) are still right for Memex
- A drop-in fork is strictly lower-risk than redesigning around a different graph model
- We have not committed code; the migration window is now and zero-cost

## Considered Options

1. **RyuGraph** — most active Kuzu fork, same Cypher, drop-in on-disk lineage
2. **Bighorn** — Kineviz fork, less activity, no formal releases yet
3. **Stay on archived Kuzu** — works today, accumulates risk every month
4. **Move to a different graph model** — NetworkX in-memory (no persistence), RDFLib (different model), Memgraph (server process, violates local-first single-binary preference)

## Decision

**RyuGraph** replaces Kuzu in the stack effective immediately. All references to Kuzu in ADR-0003, GUIDELINES.md, VISION.md, and `pyproject.toml` are updated. The on-disk path in ADR-0003's operational layout changes from `vault/.memex/graph.kuzu` to `vault/.memex/graph.ryu`. The graph-schema file path remains `src/memex/index/schemas/graph.cypher` (the Cypher dialect is unchanged).

## Consequences

### Positive

- A maintained upstream, with a roadmap from Predictable Labs and visible community contribution
- Zero code-migration cost — `memex.index.graph_store` is still unwritten
- Cypher and the property-graph model are unchanged, so `docs/eval-corpus-plan.md` and the GUIDELINES Part IV graph schema (nodes `Document`/`Entity`/`Concept`/`Citation`; edges `MENTIONS`/`CITES`/`DEFINES`/`RELATES_TO`) carry over verbatim
- Avoids a future emergency migration off an archived dep when we have a real corpus to migrate

### Negative / Trade-offs

- Smaller user base than Kuzu had at its peak; less Stack-Overflow-style help when something breaks
- RyuGraph's release cadence and long-term sustainability are unproven — a fork can also die
- One more "this used to be named X" footnote in the project's history; ADR-0003's "Why Markdown, Not Notion" framing now has a sibling footnote for the graph store

### Neutral

- The on-disk file extension changes (`.kuzu` → `.ryu`); affects no user data because none exists yet
- The `pyproject.toml` dependency name and pip name change; verify the actual PyPI distribution name at first `uv sync`

## Alternatives in Detail

### Bighorn (Kineviz)

A serious effort but currently lacks tagged releases. We can't pin a version that doesn't exist. Re-evaluate if Bighorn ships a stable line.

### Stay on archived Kuzu

The path of least immediate effort. Rejected because every Memex commit against `kuzu` deepens the rip-out cost when the inevitable security or compatibility issue forces our hand. The vision principle about open-source as a complete commitment applies to our dependency choices, not just our own code.

### Different graph model

NetworkX has no persistence. RDFLib uses RDF/SPARQL — different model, different query language, much worse fit for the property-graph patterns ADR-0003 commits to. Memgraph is a server process; it violates the local-first single-binary deployment story (the GUIDELINES Part I "no Redis, no Kafka" rule applies equally to a separate graph server).

## Revisit When

- RyuGraph's release cadence stalls for more than six months
- Bighorn ships a tagged stable line and the community converges
- A successor with materially better property-graph + Cypher performance on embedded deployments emerges (unlikely on this time horizon)
- We acquire requirements (graph algorithms, GQL, distributed graph) the embedded model can't serve

## References

- ADR-0003 (Markdown vault is source of truth) — operational layout updated to use `graph.ryu`
- Memex developer guidelines, Part IV "The knowledge graph"
- Memex vision document, "The Stack"
- Stack-currency audit, 2026-05-19 (multi-agent research run)
- RyuGraph: https://github.com/predictable-labs/ryugraph
