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
deduped-by-name, most-significant-first, capped at `max_entities`.

## Surfaces

- **CLI** — `memex related -d <doc_id> [--limit N]` (prints each `RelatedDocument`).
- **MCP** — `related_documents(doc_id, limit=10)` tool (fail-open to `[]` if ryugraph is
  absent; mirrors `get_graph_neighbors`).
- **webui** — a "Related documents" section on the document view (`document.html`,
  `.related-*` CSS): each related doc as a title-link + its connecting entities as quiet
  tags (the "why related"). Fail-open: an `ImportError` from `GraphStore.open` omits the
  section, never 500s the doc view.
- **webui `/graph`** — the Cytoscape neighbourhood viz consumes `related_documents` (was
  raw unranked `neighbors`): one node + one edge per related doc, the edge labelled with
  the connecting entities (most-specific first). Same fail-open.

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

- ✅ **`/graph` Cytoscape viz on `related_documents`** (specificity edges) — shipped.
- ✅ **entity-centric retrieval ("everything about entity X across the corpus")** — shipped
  2026-05-28; see "Entity-centric retrieval" below.
  - ✅ **acronym ↔ expansion bridge** (the resolution deepening) — shipped 2026-05-28
    (`f96c797` + `ecb6c8d`); see "Acronym ↔ expansion bridge" below.
  - ✅ **co-occurring noise reduction** (shared-docs floor + opt-in stopword list) — shipped
    2026-05-28 (`3d00ae7`); see "Co-occurring noise reduction" below.
- ⏳ citation-chain following (the still-unqueried `CITES` edges) — **DATA-GATED**, scoped as a
  data-first experiment; see "Citation-chain following" below.
- ✅ **a "Related documents" panel in `/ask`** + ✅ **scope-set suggestions** — shipped
  2026-05-29 (`ffe23fe` + `04ef4e9`); see "/ask Related panel" + "Scope-set suggestions"
  below. This CLOSES the ADR-0011 discovery build-out (citation-chain is data-gated; the
  BERT-NER swap is the remaining, separately-gated lever).
- THEN, if discovery-quality is the bottleneck: the [[bert-ner-enrich-scope-2026-05-28]]
  NER swap (sharper, typed entities upstream of the graph).

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
threaded from `AgentsSettings` via `_discovery_noise_filters`, fail-open to defaults):
- **Neighbourhood floor** (`cooccurring_min_shared_docs`, default 2; `_rank_co_occurring`
  only): a co-entity sharing < N of the seed's docs is an incidental single-doc co-mention,
  not a recurring neighbour. Kills the bulk, corpus-agnostic. (A single-doc entity then shows
  no co-occurring set — correct; it has no cross-doc neighbourhood. Tunable to 1.)
- **Curated stopword list** (`entity_stopwords`, default EMPTY; BOTH rankers): by-NAME,
  case-insensitive, kind-agnostic — `CR350` is stored as FOUR kind-nodes the df-gate +
  kind-weight can't sink, so a name match is required. Env-ergonomic via `NoDecode` +
  comma-split: `MEMEX_AGENTS__ENTITY_STOPWORDS='CR350, Réseautique et sécurité'`.

**Residual (NOT built — an NER problem, deferred to the [[bert-ner-enrich-scope-2026-05-28]]
swap):** the original `STP` symptom (enrich stored the concept as the fragment `spanning`,
no `STP`/`Spanning Tree Protocol` entity to bridge TO — `STP` stays the honest FTS fallback);
and the per-class noise (ports/sizes mis-extracted as entities; `CR350` mis-typed `concept`;
FR generic connectors like "adresse IP"/"connexion"). The floor + opt-in list are a pragmatic
pass; better entity extraction upstream is the root-cause fix. A brittle per-class regex set
(ports/hop/bits/institution) was deliberately NOT built.

**Why no automated noise-detection helper (scoped + rejected 2026-05-28).** A "surface
stopword candidates" helper was scoped and validated against the live graph — verdict: don't
build it, **no structural signal can auto-classify noise.** Co-occurrence degree, degree÷df,
and the df-band all FAIL because noise and signal have identical statistical profiles in a
topically-coherent corpus (`CR350` ranks between `TCP`/`DNS`/`IP` — a generic connector and a
central concept both co-occur with everything). Kind-weight already handles `person`/`place`
but `CR350` is mis-typed `concept`. Document-title overlap is the only real discriminator yet
high-precision / low-recall (`CR350` → 8 doc titles vs 0 for TCP/DNS/IP/ARP, but it misses the
course title and the FR connectors). So a helper could only NARROW (→ the ~126 multi-doc
entities) + flag for human judgment, never decide — low leverage once the handful of offenders
is curated. The real auto-fix is the [[bert-ner-enrich-scope-2026-05-28]] BERT-NER swap (typed,
clean entities upstream). **Manual curation recipe** (for the occasional curator, until then):
inspect entities with `df ≥ 2` ranked by co-occurrence degree + doc-title overlap, and add the
administrative names (course codes, instructor, series titles) to `entity_stopwords`.

**Cypher lesson (caught by the live-graph test — the no-Cypher-in-CI gap ADR-0011 flagged):**
the mentioning-docs query `RETURN DISTINCT d.doc_id AS doc_id, … ORDER BY d.doc_id` raised a
ryugraph binder error — after a `DISTINCT` projection `d` is out of scope, so `ORDER BY` must
reference the projected **alias** (`ORDER BY doc_id`). The "de-risked" Cypher in the plan
wasn't run WITH the `DISTINCT`; `tests/integration/test_entity_profile.py` now would catch it.

## Citation-chain following — data-first (⏳ pending data, scoped 2026-05-29)

Traversing the `CITES` Document→Document edges ("what cites this / what does this cite",
transitively) is an ADR-0011 build-out item — but it has **no data to run on yet**, and the
blocker is DATA, not engineering. Measured on the live graph (`scripts/citation_graph_audit.py`):
**6 CITES edges**, all course cross-references from ONE syllabus → 6 lectures — a **depth-1
star** with **zero multi-hop paths** (`CITES*2..4` → 0) and **0 academic citations**. Only 7 of
47 docs touch CITES; `DEFINES`/`RELATES_TO` are unpopulated (0 each). Chain-following is
structurally impossible here, and those 6 refs are already clickable `[[wikilinks]]` in the body.

**The make-or-break data condition:** `CITES` is Document→Document WITHIN the vault
(`enrich/citations.py` resolves a citation surface form against OTHER vault docs → `link_cites`).
So a **lone paper yields ZERO edges** — its references aren't in-vault to resolve against. Real
density+depth needs a **citation-LINKED cluster** (a survey + several papers it cites, all
ingested). The academic resolver already works (pinned by `test_enrich_resolves_citations_against_vault_docs`).

**The experiment** (curator-gated on a user-provided cluster): back up the vault → `memex ingest`
the cluster → re-run `scripts/citation_graph_audit.py` → compare to the baseline. **Pre-registered
decision bar:** build chain-following only if real data yields a genuine subgraph — roughly
**≥15 CITES edges, ≥5 docs with edges, and ≥1 multi-hop chain** — else stay deferred (an honest
1-hop "References" surface is the fallback). **Pre-registered design IF the bar clears:**
`GraphStore.citations(doc_id)` (1-hop cites/cited-by) and `citation_paths(doc_id, depth)` over
`MATCH (d)-[:CITES*1..N]->(o)`, mirroring `related_documents`, surfaced CLI/MCP/webui —
complementing, not duplicating, the body wikilinks.

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
  `related_documents` ranking (specificity + shared-docs floor / stopword).

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
