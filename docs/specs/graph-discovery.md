# Spec: Graph Discovery — `related_documents` + entity-centric retrieval

Status: shipped 2026-05-28. Decision record: [ADR-0011](../adr/0011-entity-graph-from-expansion-to-discovery.md).

The explicit, user-initiated discovery surfaces over the entity graph — the on-mission
successor to the retired passive `expand_graph` (ADR-0011). Graph reads, not
retrieval/answer-path operations. Two surfaces: **`related_documents`** ("what else in my
corpus relates to THIS document", below) and **entity-centric retrieval** ("everything about
entity X", further down). Both reuse one specificity primitive and are HARD-gate-neutral.

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
deduped-by-name, most-significant-first, capped at `max_entities`. Both `related_documents`
and `related_bridges` (below) draw from one shared `_fetch_shared_entity_rows(doc_id)`.

### Bridges — the entity-grouped lens (`related_bridges` / `_rank_bridges`, 2026-05-29)

`GraphStore.related_bridges(doc_id, *, limit_bridges=24, max_docs_per_bridge=50, max_via=5)`
**inverts the same rows**: instead of grouping shared entities under each related DOC, it
groups related docs under each shared ENTITY — the bridging concept. It answers a subtly
different question than `related_documents` ("which docs are related"): **"which CONCEPTS are
this document's connective tissue, and to what do they connect"**. Powers the `/graph` concept
lens. Returns `DocumentBridge{entity, kind, doc_count, strength, docs: list[BridgeDoc]}` where
`BridgeDoc{doc_id, title, score, via_entities}`; `doc_count` is the literal "bridges N", and
each doc carries its OVERALL relatedness `score` (identical to its `RelatedDocument.score`) plus
its OTHER connecting entities (`via_entities`, the "·via" tags).

**Ranking — `strength = mean(IDF×kind_weight) × ln(1 + doc_count)`.** Per-edge SPECIFICITY
dominates; fan-out enters LOGARITHMICALLY. This is the load-bearing choice: a *linear* × fan-out
(the first cut) made breadth win and surfaced exactly the generic networking terms the view
should bury (on a CCNA module the top "bridges" were `network`/`IP`/`IPv4`); the log damping
sinks a near-generic entity shared by many docs (`IP`, df just under the 0.6 cutoff) below a
specific concept shared by a few, so the top became `OSI`/`ACL`/`routing table`/`ARP`/`SNMP`/
`DHCP`. Same generic-df exclusion + kind weights as `related_documents`. The pure
`_rank_bridges` is unit-tested without a graph (`tests/unit/test_related_bridges.py`, incl. the
log-damping proof). Same fail-open + HARD-gate-neutrality as the rest of this surface.

## Surfaces

- **CLI** — `memex related -d <doc_id> [--limit N]` (prints each `RelatedDocument`).
- **MCP** — `related_documents(doc_id, limit=10)` tool (fail-open to `[]` if ryugraph is
  absent; mirrors `get_graph_neighbors`).
- **webui** — a "Related documents" section on the document view (`document.html`,
  `.related-*` CSS): each related doc as a title-link + its connecting entities as quiet
  tags (the "why related"). Fail-open: an `ImportError` from `GraphStore.open` omits the
  section, never 500s the doc view.
- **webui `/graph`** — the server-rendered **"Bridges" view** (redesigned 2026-05-29, `b48f8b2`,
  replacing the Cytoscape node-link "hairball" — a 1-hop neighbourhood is a STAR with no
  topology to draw, so a node-link diagram encodes nothing and reads as "all equally related").
  Two no-JS lenses toggled by `?group=`: **concept** (default) groups related docs UNDER the
  bridging ENTITY (`related_bridges` / `_rank_bridges`, see below); **document** is the flat
  strength-ranked neighbour list (`related_documents`). Same fail-open. Drops cytoscape from
  this page (air-gap + maintenance win). See `src/memex/webui/CLAUDE.md` § "Connections view".

## What this is NOT

- NOT in the `/ask` retrieval/answer path. The retired `expand_graph` (passive 1-hop
  injection into the candidate pool) is default-OFF (ADR-0011); discovery is explicit.
- NOT using the specificity ranking inside `expand_graph` — that opt-in path still uses
  unranked `neighbors()`. Wiring specificity into expansion only pays off at large-corpus
  scale (where retrieval misses docs); a documented hook, not shipped.

## HARD-gate safety

A read-only discovery surface — it never touches the answer/refusal path, so it cannot
introduce a hallucination or alter a refusal. Independent of the answering agent entirely.

## Build-out (in leverage order — ADR-0011 / db-audit)

- ✅ **`/graph` neighbourhood viz** — shipped on `related_documents` (specificity edges), then
  **redesigned 2026-05-29 (`b48f8b2`) as the server-rendered "Bridges" view** (Cytoscape dropped;
  concept lens = `related_bridges`, document lens = `related_documents`). See "Bridges view" below.
- ✅ **entity-centric retrieval ("everything about entity X across the corpus")** — shipped
  2026-05-28; see "Entity-centric retrieval" below.
  - ✅ **acronym ↔ expansion bridge** (the resolution deepening) — shipped 2026-05-28
    (`f96c797` + `ecb6c8d`); see "Acronym ↔ expansion bridge" below.
  - ✅ **co-occurring noise reduction** (shared-docs floor; the curated `entity_stopwords` list was later REMOVED 2026-05-29 — see below) — shipped
    2026-05-28 (`3d00ae7`); see "Co-occurring noise reduction" below.
- ✅ **CITES 1-hop "References"** (`citations()`; `memex cites` / MCP `document_citations` /
  webui) — shipped 2026-05-29 (`38647e7`); reads the previously write-only CITES edges.
- ✅ **transitive citation-chain following** (`citation_paths()`; `memex cites --depth N
  [--cited-by]` / MCP `citation_paths`) — shipped 2026-06-14 once the data bar cleared (CITES
  6 → 34 after the embedder-lineage cluster ingest); see "Citations" below.
- ✅ **a "Related documents" panel in `/ask`** + ✅ **scope-set suggestions** — shipped
  2026-05-29 (`ffe23fe` + `04ef4e9`); see "/ask Related panel" + "Scope-set suggestions"
  below. This CLOSES the ADR-0011 discovery build-out (the BERT-NER swap is the remaining,
  separately-gated lever).
- ✅ **MCP/CLI parity for the answer's related docs + attested-chunk passages** — shipped
  2026-05-29 (`ac4b1ba` + `8acaad3`, the NER-leverage build-out). `FinalResponse.related_documents`
  (the /ask panel's data on MCP/CLI too, via the shared `retrieve/related.py`); `entity_overview`
  passages now prefer the EXACT attested chunk (the MENTIONS `chunk_id`) over an FTS name-search,
  with graceful FTS fallback. See [[ner-leverage-buildout-2026-05-29]].
- THEN, if discovery-quality is the bottleneck: the [[bert-ner-enrich-scope-2026-05-28]]
  NER swap (sharper, typed entities upstream of the graph) — **SHIPPED (OTTER, ADR-0012)**.

### Measured-and-NOT-pursued levers (NER-leverage audit, 2026-05-29)

A code-grounded audit asked whether the NER/entities are fully leveraged. Three levers were
MEASURED and deliberately NOT shipped (see [[ner-leverage-buildout-2026-05-29]]):
- **Confidence-weighted discovery ranking** — the OTTER MENTIONS confidence varies (mean .256,
  stdev .20) and re-weighting materially reshuffles `related_documents`, but it measures
  extraction-TYPICALITY, not topical-SPECIFICITY; without a labelled should-relate gold set its
  quality valence is unprovable, so it's NOT wired into the (validated IDF×kind) ranker.
- **Entity signal in the `/ask` retrieve/rerank path** — CONCLUSIVE no headroom at 47 docs:
  58/58 ANS gold DOCS are already in dense@50, so a doc-level entity-overlap signal recovers
  nothing (converges with the db-audit's "1-hop expansion cites 0" + the BM25 arm-separation).
  And it would touch the HARD gate. Revisit only at large-corpus scale.
- **`DEFINES`/`RELATES_TO` edges + OTTER∪LLM fusion** — deferred (need a relation-extraction
  stage OTTER isn't / OTTER-alone already wins; graph-only payoff at this scale).

## Entity-centric retrieval — "everything about entity X" (shipped 2026-05-28)

The second discovery surface: given an entity NAME, surface its corpus-wide profile. The
genuinely-graph capability `expand_graph` never was — and read-only, so HARD-gate-neutral
by construction (never touches the answer/refusal path; passages are for browsing, never
injected into a grounding pool).

```python
# index/graph_store.py — the GRAPH half (identity + docs + neighbourhood)
async def entity_profile(self, name, *, max_docs=50, max_cooccurring=15) -> EntityProfile
# EntityProfile { query_name; matched_names: list[str]; kinds: list[str]; doc_count;
#                 mentions: list[EntityMention]; cooccurring: list[CoOccurringEntity]; resolved }

# retrieve/entity.py — the ORCHESTRATOR (graph + FTS)
async def entity_overview(name, *, max_docs=50, max_cooccurring=15, passages_k=10) -> EntityOverview
# EntityOverview { profile: EntityProfile; passages: list[Chunk]; passages_scoped: bool }
```

**The graph-UNIQUE value** (without these it's a search reskin): canonical identity (resolve
a free-text name → its kind(s) + true corpus `doc_count`); the **authoritative MENTIONS doc
set** (which docs discuss X, from the graph not fuzzy retrieval); **co-occurring entities** —
the concept neighbourhood, ranked by the SAME specificity filter as `related_documents`
(`_rank_co_occurring`: `shared_docs × ln(N/df) × kind_weight`, generic-df exclusion +
proper-noun down-weight; the `shared_docs` multiplier — a co-entity in more of the seed's docs
ranks higher — is the new term). **Passages come from FTS** (the `MENTIONS` edge is doc-level
only, no chunk granularity) — scoped to the mentioning docs when resolved.

**Resolution UX:** exact **case-insensitive name match**, aggregating across kinds (one name
can be `concept` in one doc, `tool` in another → `kinds` is a list). Unknown name OR graph
unavailable → `resolved=False` + a **whole-corpus FTS fallback** (`passages_scoped=False`) —
the honest "not a known entity, here's what text search finds." An entity in many docs:
`doc_count` is the true total; `mentions` capped at `max_docs` (honest "mentioned in N,
showing 50"). Fail-open throughout (ImportError → the fallback); a runtime graph error
deliberately surfaces (it's a bug, not the optional-dependency case).

**Surfaces:** CLI `memex entity <name> [--max-docs N] [--cooccurring N] [-k N]`; MCP
`entity_overview(name, …)` tool; webui `GET /entity?name=` (the lookup view — co-occurring
tags are `/entity?name=` links to TRAVERSE the neighbourhood, the doc-view "Related documents"
connecting-entity tags link in too).

### Acronym ↔ expansion bridge (shipped 2026-05-28)

`DNS` and `Domain Name System` are SEPARATE entity_ids — so an exact-match resolver
fragments one concept across two profiles. The bridge connects them, deterministically and
**SUGGEST-only** (never an identity merge — the profile's `doc_count`/`matched_names`/passages
stay EXACTLY the exact-match's; the bridge is a link, matching the project's no-false-identity
ethos). On a resolved name it adds "Also see"; on an unresolved name, "Did you mean?".

- **Mechanism = INITIALISM, not substring CONTAINS.** Pure `index/initialism.py`:
  `derive_initialism` (first unicode-alnum char of each significant word, skipping EN+FR
  connectors; "Domain Name System" → "DNS"), `looks_like_acronym`, `initialism_matches`.
  CONTAINS was rejected — the probe showed it is noisy for popular acronyms (`dns` → 34
  substring hits incl. "DNS Spoofing") and useless for rare ones (`stp` → only "NISTPRIV").
- **Two directions in `entity_profile`:** query is acronym-shaped → a bounded entity-name
  scan for expansions whose initials match; query is multi-word → derive its initialism +
  exact-probe the bare acronym. Both run on the resolved AND unresolved paths.
- **Conservative gate `_gate_suggestions`** (pure, unit-tested): collision-drop (≥2 DISTINCT
  names → ambiguous → no-op, the #256 rule); generic + self exclusion; and a **doc-count
  floor (≥2)** that kills cross-domain initialism FALSE-FRIENDS — live, query `STP` matched
  the 10-K's 1-doc "Short-term portion" (a coincidental initialism), now floored out, while
  every real bridge recurs (DNS/TCP/DHCP/ICMP expansions all ≥2 docs). A clean bridge ⇒ 0–1
  suggestion.

### Co-occurring noise reduction (shipped 2026-05-28)

The `_RELATED_GENERIC_ENTITY_DF_FRACTION` (0.6) gate catches near-universal entities but
misses a term generic WITHIN a doc-family yet below 60% of the whole vault. Probing showed
the noise is broad — **~69% is single-doc numeric junk** (port/PID numbers, sizes at
`shared_docs=1`), with the reported `CR350` only ~3%. Two filters (both on the pure rankers,
from `AgentsSettings` via `_cooccurring_min_shared_docs`, fail-open to the default):
- **Neighbourhood floor** (`cooccurring_min_shared_docs`, default 2; `_rank_co_occurring`
  only): a co-entity sharing < N of the seed's docs is an incidental single-doc co-mention,
  not a recurring neighbour. Kills the bulk; corpus-AGNOSTIC + structural (zero per-corpus
  tuning) → it scales to any vault. (A single-doc entity then shows no co-occurring set —
  correct; it has no cross-doc neighbourhood. Tunable to 1.)

**The curated `entity_stopwords` list was REMOVED 2026-05-29** (with `_normalize_stopwords`,
the `stopwords` param on both rankers, the `NoDecode` import, and the `config.toml` entry). It
was a by-name, per-corpus exclusion (`CR350`, `Réseautique et sécurité`) — a band-aid for the
LLM mis-typing a course code as a `concept`. A hand-curated name list doesn't generalise to a
local-first app run on ANY corpus (one user's `CR350` is a drop in the ocean of the world's
course codes, product names, and orgs), and the [[bert-ner-enrich-scope-2026-05-28]] OTTER NER
backend — now the LIVE enrich entity extractor — types entities cleanly UPSTREAM, removing the
root cause (mechanism: spec [`ner-enrich.md`](ner-enrich.md) + [ADR-0012](../adr/0012-otter-bert-ner-enrich-backend.md); A/B [`08-otter-ner-ab.md`](../audits/08-otter-ner-ab.md)). Fix entity noise at the extractor, not with a curated downstream list.

**Why no automated noise-detection helper either (scoped + rejected 2026-05-28 — the analysis
that pointed at the NER swap).** A "surface stopword candidates" helper was scoped + validated
against the live graph: **no structural signal can auto-classify noise.** Co-occurrence degree,
degree÷df, and the df-band all FAIL because noise and signal have identical statistical
profiles in a topically-coherent corpus (`CR350` ranks between `TCP`/`DNS`/`IP`). Document-title
overlap is the only real discriminator, and it's high-precision / low-recall. The conclusion —
you cannot separate "administrative connector" from "central concept" by graph STRUCTURE, only
by SEMANTICS/TYPING — is exactly what motivated the OTTER swap (typed, clean entities upstream),
now shipped.

**Cypher lesson (caught by the live-graph test — the no-Cypher-in-CI gap ADR-0011 flagged):**
the mentioning-docs query `RETURN DISTINCT d.doc_id AS doc_id, … ORDER BY d.doc_id` raised a
ryugraph binder error — after a `DISTINCT` projection `d` is out of scope, so `ORDER BY` must
reference the projected **alias** (`ORDER BY doc_id`). The "de-risked" Cypher in the plan
wasn't run WITH the `DISTINCT`; `tests/integration/test_entity_profile.py` now would catch it.

## Citations — the 1-hop "References" surface ✅ + chain-following (data-first ⏳)

**1-hop References — SHIPPED 2026-05-29 (`GraphStore.citations(doc_id)` → `DocumentCitations
{cites, cited_by}`).** The `CITES` Document→Document edges (written at enrich from the resolved
in-vault citations) were WRITE-ONLY — only the body `[[wikilinks]]` consumed the resolution, no
query READ the edges. `citations()` reads them: `cites` (what this doc references) + `cited_by`
(what references it), each a `CitationLink` with the edge surface_text. Surfaces: `memex cites -d`,
MCP `document_citations`, webui doc-view "References" (reuses `.related-*`, fail-open). Read-only ⇒
HARD-gate-neutral. Live: 10 CITES edges (the CR350 syllabus `cites`=6 lectures, each lecture
`cited_by`=1). This is the honest 1-hop fallback the chain-following deferral named.

**Transitive chain-following — ✅ SHIPPED 2026-06-14 (the data bar cleared).**
Traversing the `CITES` edges TRANSITIVELY ("what does this cite / what cites this", multi-hop)
was the last ADR-0011 discovery build-out item — gated on DATA, not engineering. The
make-or-break data condition: `CITES` is Document→Document WITHIN the vault
(`enrich/citations.py` resolves a citation surface form against OTHER vault docs → `link_cites`),
so a **lone paper yields ZERO edges** and real depth needs a **citation-LINKED cluster** (a
survey + several papers it cites, all ingested). At the original measurement the live graph was a
**depth-1 star** — 6 CITES edges, all one syllabus → 6 lectures, 0 academic, 0 multi-hop.

**The experiment ran (2026-06-14):** ingested a 6-paper embedder-lineage cluster
(BGE/C-Pack → GTE, E5, Contriever, SimCSE, Sentence-BERT, GTR) into the main vault (backed up
first). `scripts/citation_graph_audit.py` went **6 → 34 CITES edges** (15 genuine academic, the
rest course-refs), **≥5 docs with edges**, and **real multi-hop chains** (BGE→Contriever→SimCSE).
The **pre-registered bar (≥15 edges / ≥5 docs / ≥1 multi-hop) CLEARED → BUILD**.

**The build (`index/graph_store.py::citation_paths`):** `citation_paths(doc_id, *, depth=3,
direction="cites"|"cited_by") -> CitationPaths` over `MATCH p = (a {doc_id})-[:CITES*1..N]->(o)`
(`<-…-` for `cited_by`), mirroring `citations()`/`related_documents`. Returns each reachable
document at its **SHORTEST** hop-distance (`CitationReach{doc_id, title, hops, path}`) with an
example chain — `path` titles always read in CITATION ORDER (citing doc first → cited doc), so
`cited_by` reverses the raw traversal order. `depth` is clamped to `[1, 6]`; the seed is excluded;
results sort by `(hops, title)`. Surfaced **CLI** (`memex cites --document D --depth N [--cited-by]`),
**MCP** (`citation_paths(doc_id, depth, direction)`, fail-open empty via `open_graph_for_read`),
and **webui** (the `/graph/{doc_id}?group=citations&depth=N` lens — a third lens beside
concept/document — rendering both directions as *References* / *Cited by* columns with per-reach
hop badges + multi-hop example chains + a no-JS depth selector; the doc-view References eyebrow
links into it via "full lineage →"; see `src/memex/webui/CLAUDE.md`). Builds on the shipped 1-hop
`citations()` and complements (not duplicates) the body wikilinks. Read-only ⇒ HARD-gate-neutral.
Pinned by `test_entity_profile.py::test_citation_paths_multihop` (live Cypher) +
`test_mcp_server.py::test_citation_paths_tool` + `test_webui.py::test_graph_citations_lens_*`.

## /ask "Related documents" panel (shipped 2026-05-29)

Discovery woven into the answer flow: after an answer renders, the webui `/ask` result shows
documents the entity graph relates to the docs the answer CITED. **Webui-only + HARD-gate-
neutral by construction** — computed in the PRESENTATION layer (`webui/app.py::_related_for_answer`,
wired into `_answer_context`, the single seam feeding `_answer.html` on the long-poll
completion) from the already-returned `FinalResponse` + a read-only graph query; it NEVER
touches the agent / answer / refusal path, and the CLI/MCP `ask` payloads are unchanged.

- **Answered-only** (a refusal's `used_chunks` are retrieved-but-ungrounded, not "used").
- Seeds from the distinct cited `document_id`s (`used_chunks`), expands each via
  `related_documents`, then merges → dedups by doc_id (keeps the higher-score relation) →
  **EXCLUDES the docs the answer itself cited** → re-ranks by score → caps. ImportError
  fail-open → no panel (mirrors the doc-view).
- Renders below Sources (before the scope note/footer) reusing the doc-view `.related-*`
  link/tag styling (each related doc a link + its connecting entities as `/entity?name=`
  traversal tags) + a small `.ans-related` wrapper. Inherits the SHIPPED noise-filtered
  `related_documents` ranking (specificity + shared-docs floor).

Pinned by `tests/integration/test_webui.py` (answered panel renders + excludes cited docs;
graph-unavailable fail-open; refusal → no panel). Live-validated (Chrome e2e): "What is DNS
used for?" → the panel lists the sibling CR350 lectures with entity tags, the cited Cours 3
excluded; a counterfactual → no panel.

The merge/dedup/exclude/re-rank/cap core is the shared `webui/app.py::_related_for_docs(vault_path,
seed_ids)` ("graph neighbours of a SET of docs, seed-excluded"), reused by both the panel
(`_related_for_answer` delegates to it) and the scope-set suggestions below.

## Scope-set suggestions (shipped 2026-05-29)

Discovery woven into the SCOPING flow: the Ask-page scope-picker surfaces documents the graph
relates to the current selection ("Suggested additions"), each a `scope_doc_ids` checkbox to
tick-and-add. Two triggers (both feed `_related_for_docs` over the selection): **auto** on
apply/save (`_scope_picker_context` computes `suggested` from a non-empty `checked_ids` — an
empty selection does no graph query) and an explicit **"Suggest related"** button
(`POST /scope-sets/suggest` posts the ticked docs + a count flash). Each suggestion is a
checkbox + `/entity?name=` why-related tags; once ticked it re-renders as checked in the main
list and drops out of suggestions. Webui-only + HARD-gate-neutral (a suggestion is just
another `scope_doc_ids` tick flowing through the unchanged, dedup-ing `resolve_artifact_scope`
path). Fail-open → no section + a "No related documents found" flash. Pinned by
`test_webui.py` (auto-on-apply excludes the set's docs; the Suggest button + count flash;
empty-selection hint; graph-unavailable fail-open). Live-validated (Chrome e2e): applying a
CR350 set surfaces related CR350 **and** CCNA docs with why-tags.

## Testing

- `tests/unit/test_related_documents.py` — the ranking core: specific-beats-generic,
  generic-df exclusion, proper-noun down-weight + reordering, weighted-IDF sum, limit cap,
  empty/singleton corpus.
- `tests/integration/test_webui.py` — the doc-view "Related documents" section renders +
  survives an unavailable graph (fail-open).
- `tests/integration/test_mcp_server.py` — the `related_documents` tool returns the ranked
  list + is fail-open; the tool-registration smoke includes it (+ `entity_overview`).
- **Entity-centric:** `tests/unit/test_entity_profile.py` (the pure `_rank_co_occurring`:
  specific-beats-generic, shared_docs multiplier, generic-df exclusion, proper-noun
  down-weight, formula, limit/empty); `tests/integration/test_entity_overview.py` (the
  orchestrator with fakes: resolved→scoped passages / unknown→whole-corpus fallback /
  ImportError→fail-open); `tests/integration/test_entity_profile.py` (**opt-in real
  ryugraph** via `importorskip` — seeds a graph, runs the live Cypher: identity +
  cross-kind aggregation + co-occurring rank + unknown→unresolved; the gap that caught the
  `ORDER BY`-after-`DISTINCT` binder bug); `tests/integration/test_webui.py` (the `/entity`
  view: resolved render / unknown fallback / lookup form / related-tag-is-entity-link).
- **Acronym bridge:** `tests/unit/test_initialism.py` (the pure `derive_initialism` /
  `looks_like_acronym` / `initialism_matches` + the `_gate_suggestions` gate — derivation +
  connector/accent/length/alpha-share boundaries, collision-drop, doc-count floor ordering,
  dedup-max, generic/self exclusion); `tests/integration/test_entity_profile.py` (real
  ryugraph: both bridge directions, ambiguous-not-bridged, the STP honest-miss, multi-kind
  dedup, generic-expansion exclusion, whitespace/case, blank query); `test_entity_overview.py`
  (suggestions survive the orchestrator + passages stay exact-scoped; unresolved-with-bridge
  uses the corpus fallback); `test_mcp_server.py` (the suggestion serialises);
  `test_webui.py` ("Also see" / "Did you mean?" render; the honest no-bridge STP view).
