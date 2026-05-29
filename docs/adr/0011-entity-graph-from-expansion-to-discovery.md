# ADR-0011: The Entity Graph Serves Discovery, Not Retrieval Augmentation

- **Status**: Accepted
- **Date**: 2026-05-28
- **Deciders**: Memex core team
- **Tags**: architecture, retrieval, graph

## Context

The entity graph (ADR-0005; `Document`/`Entity`/`Concept`/`Citation` nodes, `MENTIONS`/`CITES`/`DEFINES`/`RELATES_TO` edges, written at enrich) had exactly one consumer in the answer path: `agents/answering.py::expand_graph`, a node between `retrieve` and `rerank` that, for the top-3 retrieved documents, pulled their 1-hop "shares-an-entity" neighbour documents' chunks into the candidate pool. The premise was graph-augmented retrieval — recover a relevant document that flat retrieval missed but that is one entity-hop away.

A microscope audit (2026-05-28, the [[db-audit-2026-05-28]] memo) tested whether `expand_graph` earns its per-`/ask` cost (a graph-store open + neighbour query + extra chunks for the ~20 s CPU reranker to re-score). Three layers of evidence:

1. **A/B (ON vs OFF, N=3)** on the two cross-doc-disambiguation corpora — `cr350-multidoc` and `ccna-multidoc`, the exact case 1-hop expansion exists to serve — was **byte-identical** (14/5 and 8/2, refusal_cf=1.0, 0 hallucinations, deterministic across all runs).
2. **Per-query trace:** the node adds 2–12 neighbour chunks every query; the reranker cites **exactly 0** of them (the answer's cited-doc set always equalled the retrieval-doc set).
3. **Rerank-score microscope:** the added chunks ARE scored (not lost to a plumbing bug) — on queries with signal they rank #25–#53 of ~56 (decisively beaten); on a low-signal query they tie in the noise but still go uncited.

Root cause: the neighbours are linked by **generic** shared entities (`IP`, `HTTP`, the course instructor's name; `neighbors()` is unranked), and at this corpus scale (47 docs / ~3.7k chunks) hybrid retrieval at k=50 already has **near-total recall** — so there is no missed document for 1-hop expansion to recover. Graph-augmented retrieval is a large-corpus recall technique, premature here (the same "right at scale, not now" logic as flat-vector-search in IMPLEMENTATION-PLAN §1).

But the graph is healthy (11.6k entities, 12.3k `MENTIONS` edges) — the problem is the *use*, not the data. The question this ADR settles: what, then, is the entity graph FOR?

## Decision Drivers

- Evidence over intuition — the audit measured the actual answer impact, not the intermediate state
- Don't pay a per-request cost (graph open + rerank load on the 20 s CPU bottleneck) for zero answer benefit
- Keep the graph if it has a use that fits the product (NotebookLM-for-your-corpus + composable local-first), not just because it exists
- Reversibility — leave a kill-switch so a large entity-rich vault can opt back in

## Considered Options

1. **Retire the graph entirely** — delete `expand_graph` + the graph store
2. **Keep `expand_graph` on, improve the neighbour selection** — specificity-rank the expansion
3. **Repurpose the graph: expansion default-off; the graph powers explicit DISCOVERY** (chosen)

## Decision

**The entity graph's role is DISCOVERY, not retrieval augmentation.** Concretely:

- **`expand_graph` is default-OFF** (`MEMEX_AGENTS__GRAPH_EXPANSION_ENABLED`, ANDed with the `answer_query` param; commit `a52d5fa`). It does real work for zero answer impact at this scale. The env opt-in remains for a large entity-rich vault where retrieval recall is no longer near-total.
- **The graph store is kept**, and its on-mission consumer is **`GraphStore.related_documents(doc_id)`** — an explicit, user-initiated "explore connections" surface (CLI `memex related`, MCP `related_documents`, webui doc-view "Related documents"; commits `ecd8372` + `9905965`). Unlike `neighbors()`, it ranks neighbours by shared-entity **specificity**: `Σ ln(N/df(e)) × kind_weight(e)` over the shared entities, excluding near-universal entities (df > 60% of the corpus) and down-weighting incidental proper-noun types (person/place ×0.25). So a document sharing one rare *concept* outranks one sharing five generic terms — the meaningful connection wins. Pure scoring core `_rank_related_documents`, spec `docs/specs/graph-discovery.md`.

This is the difference between *passive recall-boosting in the RAG path* (which the audit killed) and *explicit discovery over the corpus* (which the product — a private NotebookLM — actually wants). The specificity ranking is what makes discovery useful; it is deliberately NOT wired into the (opt-in) `expand_graph` path, because at this scale even perfect neighbour selection can't beat near-total retrieval recall — that wiring is a documented large-corpus-only hook.

## Consequences

### Positive

- Every `/ask` reclaims a graph-store open + the neighbour-chunk rerank load (real on the 20 s CPU-rerank path), with zero answer-quality cost — proven on the corpora where expansion could possibly have helped.
- The graph finally has a validated, on-mission consumer (discovery), and a proven, reusable specificity primitive for the build-out: graph-viz on `related_documents`, entity-centric retrieval ("everything about entity X"), citation-chain following (the still-unqueried `CITES` edges), scope-set suggestions, a "Related" panel in `/ask`.
- HARD-gate-neutral by construction: turning expansion off only removes candidates the reranker discarded anyway; `related_documents` is a read-only discovery surface that never touches the answer/refusal path.

### Negative / Trade-offs

- A genuinely multi-hop query (answer in a document retrieval missed, reachable only via a specific shared entity) is no longer auto-served — it needs the explicit opt-in. Acceptable: no eval query exhibits this, and at 47 docs recall is near-total.
- The graph's value now rests on discovery features being built out + used; if discovery stays niche, the enrich-time cost of maintaining the graph is under-amortised (revisit-when below).
- The opt-in `expand_graph` still uses the OLD unranked `neighbors()`, not the specificity ranking — a known gap, deliberately deferred (it only matters at large-corpus scale).

### Neutral

- The enrich write path (entity/citation extraction → graph) is unchanged.
- `expand_graph` short-circuits when a scope is active (artifact-scope / doc-picker), as before; default-off just makes that the universal case.

## Alternatives in Detail

### Retire the graph entirely

Tempting given expansion added nothing. Rejected: the graph DATA is healthy and the *discovery* use (which the audit did not test, because expansion isn't discovery) is on-mission for a NotebookLM-style product. Deleting it would throw away the substrate for entity-centric retrieval and citation chains.

### Keep `expand_graph` on, specificity-rank it

Fixes the neighbour-quality half (generic entities), but not the scale half: at 47 docs retrieval recall is already near-total, so better neighbours still can't add a missed relevant document. The reranker would still discard them. Re-introducing specificity-ranked expansion into RAG only pays off at a corpus large enough that k=50 misses relevant docs — a documented future hook, not a now-decision.

## Revisit When

- The corpus grows large enough (≫ thousands of docs) that hybrid k=50 retrieval demonstrably misses relevant documents — then specificity-ranked `expand_graph` (using `related_documents`, not `neighbors()`) becomes worth an eval-gated re-introduction.
- Discovery features (entity-centric retrieval, citation chains, the `/ask` "Related" panel) ship and get used — confirming the graph earns its enrich-time keep, or not.
- A better entity extractor (the [[bert-ner-enrich-scope-2026-05-28]] BERT-NER) sharpens entity specificity enough to materially change the discovery quality.

## References

- ADR-0005 (RyuGraph replaces Kuzu) — the storage engine; this ADR is about its USE
- `docs/specs/graph-discovery.md` — the `related_documents` contract
- The DB meta-audit + the `expand_graph` worth-it measurement ([[db-audit-2026-05-28]] memory)
- Commits `a52d5fa` (expansion default-off), `ecd8372` (related_documents), `9905965` (entity-type weighting)
- IMPLEMENTATION-PLAN §"Beyond v1" — "Citation graph reasoning during answering" (the original deferral this ADR closes with evidence)
