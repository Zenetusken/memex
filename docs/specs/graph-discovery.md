# Spec: Graph Discovery — `related_documents` ("explore connections")

Status: shipped 2026-05-28. Decision record: [ADR-0011](../adr/0011-entity-graph-from-expansion-to-discovery.md).

The explicit, user-initiated "what else in my corpus relates to this document" surface
over the entity graph — the on-mission successor to the retired passive `expand_graph`
(ADR-0011). A graph read, not a retrieval/answer-path operation.

## Entry point

```python
# index/graph_store.py
async def related_documents(
    self, doc_id: str, *, limit: int = 10, max_entities: int = 8
) -> list[RelatedDocument]
# RelatedDocument { doc_id; title; score; shared_entities: list[str] }
```

## Ranking — the quality differentiator

Where `neighbors()` returns shared-entity neighbours UNRANKED (so generic connectors —
`IP`, `HTTP`, an instructor's name in every doc — dominate and the result is noise, which
is *why* passive expansion added nothing), `related_documents` scores each neighbour by

```
score(N) = Σ over entities e shared by (doc, N):   ln(n_docs / df(e)) × kind_weight(e)
```

- **IDF `ln(n_docs / df(e))`** — `df(e)` = how many documents mention `e`. A RARE shared
  entity (low df → high IDF) is a strong topical link; a near-universal one (df → N → IDF
  → 0) contributes ~nothing.
- **Generic-entity exclusion** — an entity with `df > _RELATED_GENERIC_ENTITY_DF_FRACTION`
  (0.6) of the corpus is dropped outright (a document connected ONLY by such entities
  disappears from the results).
- **Entity-TYPE weight** (`_ENTITY_KIND_WEIGHT`) — IDF is multiplied by the entity's kind
  weight: concept/method/tool = 1.0, org/other = 0.5, **person/place = 0.25**. The
  proper-noun down-weight catches incidental connectors (the course instructor, `person`,
  df≈7/47 — *under* the 60% bar; a country, `place`) that IDF alone rewards. Down-weight,
  not exclude: a proper noun that's genuinely the strongest signal still counts, and it
  sinks in the surfaced "why".

So a document sharing **one specific concept** outranks one sharing **five generic terms**.
The pure scoring core `_rank_related_documents(rows, n_docs, *, limit, max_entities)`
takes `(neighbour_doc_id, title, entity, kind, df)` rows and is unit-tested without a graph
(`tests/unit/test_related_documents.py`). The Cypher returns the per-(neighbour,entity) df
via a nested `MATCH (e)<-[:MENTIONS]-(m:Document)` count; `shared_entities` is
deduped-by-name, most-significant-first, capped at `max_entities`.

## Surfaces

- **CLI** — `memex related -d <doc_id> [--limit N]` (prints each `RelatedDocument`).
- **MCP** — `related_documents(doc_id, limit=10)` tool (fail-open to `[]` if ryugraph is
  absent; mirrors `get_graph_neighbors`).
- **webui** — a "Related documents" section on the document view (`document.html`,
  `.related-*` CSS): each related doc as a title-link + its connecting entities as quiet
  tags (the "why related"). Fail-open: an `ImportError` from `GraphStore.open` omits the
  section, never 500s the doc view.

## What this is NOT

- NOT in the `/ask` retrieval/answer path. The retired `expand_graph` (passive 1-hop
  injection into the candidate pool) is default-OFF (ADR-0011); discovery is explicit.
- NOT using the specificity ranking inside `expand_graph` — that opt-in path still uses
  unranked `neighbors()`. Wiring specificity into expansion only pays off at large-corpus
  scale (where retrieval misses docs); a documented hook, not shipped.

## HARD-gate safety

A read-only discovery surface — it never touches the answer/refusal path, so it cannot
introduce a hallucination or alter a refusal. Independent of the answering agent entirely.

## Build-out (deferred, in leverage order — ADR-0011 / db-audit)

Point the existing `/graph` Cytoscape viz at `related_documents` (specificity edges) →
entity-centric retrieval ("everything about entity X") → citation-chain following (the
still-unqueried `CITES` edges) → scope-set suggestions + a "Related" panel in `/ask`. The
[[bert-ner-enrich-scope-2026-05-28]] NER swap (sharper entities upstream) gates behind
discovery-quality proving the bottleneck.

## Testing

- `tests/unit/test_related_documents.py` — the ranking core: specific-beats-generic,
  generic-df exclusion, proper-noun down-weight + reordering, weighted-IDF sum, limit cap,
  empty/singleton corpus.
- `tests/integration/test_webui.py` — the doc-view "Related documents" section renders +
  survives an unavailable graph (fail-open).
- `tests/integration/test_mcp_server.py` — the `related_documents` tool returns the ranked
  list + is fail-open; the tool-registration smoke includes it.
