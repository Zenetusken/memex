"""RyuGraph-backed entity and citation graph — see ADR-0005.

The graph stores nodes (Document, Entity, Concept, Citation) and edges
(MENTIONS, CITES, DEFINES, RELATES_TO). RyuGraph is the maintained
fork of Kuzu after upstream archival; the Cypher dialect and on-disk
format lineage are preserved, so this module's interface is the same
shape an eventual return to upstream Kuzu (or any successor) would
need to honour.

Schema is the cypher file in `index/schemas/graph.cypher`, applied
idempotently on first connection.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import threading
from pathlib import Path
from typing import Any, cast

import structlog
from pydantic import BaseModel

from memex.core.config import get_settings
from memex.core.errors import ConfigurationError
from memex.core.types import BridgeDoc, DocumentBridge, RelatedDocument
from memex.index.initialism import (
    derive_initialism,
    initialism_matches,
    looks_like_acronym,
)

logger = structlog.get_logger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schemas" / "graph.cypher"
_STMT_SPLIT = re.compile(r";\s*\n")

# The graph DDL is immutable at runtime + the `CREATE ... IF NOT EXISTS` statements
# are idempotent, but `GraphStore.open` runs PER `/ask` (the lazy-store-open pattern),
# so re-reading + re-splitting the file and re-executing ~10 DDL statements on every
# open was a fixed per-request tax. Parse the file ONCE at import, and apply the DDL
# only the FIRST time this process opens a given graph path (the on-disk schema is
# persistent, so later opens of the same DB can skip it). The lock makes the
# check-and-apply atomic across the `asyncio.to_thread` workers that call `_connect`.
# NB this addresses the parse + DDL cost only; the embedded-DB cold-open per request
# is separate (would need connection reuse — deferred as a riskier refactor).


def _strip_cypher_comments(schema: str) -> str:
    """Remove `//`-prefixed line comments so the statement splitter
    doesn't see them as part of the first statement.

    The schema file begins with three banner-comment lines; if those
    survive into the first split chunk, the chunk gets dropped by
    the `not stmt.startswith("//")` check and the very first
    `CREATE NODE TABLE Document` never runs. Subsequent `CREATE REL
    TABLE` statements that reference Document then fail with
    "Binder exception: Table Document does not exist."
    """
    lines: list[str] = []
    for line in schema.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _parse_schema_statements() -> list[str]:
    """Read + strip + split the Cypher schema into executable statements ONCE."""
    schema = _strip_cypher_comments(_SCHEMA_PATH.read_text(encoding="utf-8"))
    out: list[str] = []
    for stmt in _STMT_SPLIT.split(schema):
        # A trailing `;` on the final chunk (no newline after it) would produce
        # `;;`, which ryugraph's parser rejects — strip then re-append exactly one.
        cleaned = stmt.strip().rstrip(";").strip()
        if cleaned:
            out.append(cleaned + ";")
    return out


_SCHEMA_STATEMENTS: list[str] = _parse_schema_statements()
_schema_applied_paths: set[str] = set()
_schema_lock = threading.Lock()


class GraphNeighbor(BaseModel):
    """One related document, surfaced by the graph for the `graph` CLI."""

    doc_id: str
    title: str
    relation: str
    via: str | None = None  # e.g. shared entity name


class CitationLink(BaseModel):
    """One end of a `CITES` edge — a document this doc cites, or one that cites it. The
    `surface_text` is the citation surface form recorded on the edge (e.g. the resolved
    wikilink target / "Cours 4")."""

    doc_id: str
    title: str
    surface_text: str


class DocumentCitations(BaseModel):
    """A document's 1-hop `CITES` neighbourhood — the honest "References" discovery surface.
    `cites` = documents THIS doc references (outgoing CITES); `cited_by` = documents that
    reference THIS doc (incoming). Transitive chain-following stays data-gated (the vault has
    too few in-corpus citation edges to traverse — see `docs/specs/graph-discovery.md`)."""

    cites: list[CitationLink] = []
    cited_by: list[CitationLink] = []


# Entity-specificity threshold for `related_documents`: an entity mentioned by MORE than
# this fraction of the corpus is treated as NOISE (a generic connector like 'IP', 'HTTP',
# or an author/instructor name that appears everywhere) and excluded from the "why". IDF
# already down-weights such entities toward zero; the hard exclusion just keeps the
# surfaced connecting-entities list clean. This is the lever that distinguishes the
# `related_documents` discovery feature from the retired passive `expand_graph` (which
# linked on these generic entities, unranked, and added nothing — see db-audit-2026-05-28).
_RELATED_GENERIC_ENTITY_DF_FRACTION = 0.6

# Entity-TYPE weights for `related_documents` (multiply the IDF). The df-exclusion above
# catches near-universal entities, but a PROPER-NOUN connector can be specific (low df)
# yet topically INCIDENTAL — the course instructor ('person', df≈7/47, well under the
# 60% bar) or a country ('place', 'Vietnam') that two docs both happen to mention. So
# topical types (concept/method/tool) keep full weight, orgs/other are halved (sometimes
# meaningful — 'NVIDIA', 'IRS'), and person/place are heavily down-weighted (rarely the
# REASON two docs relate). Down-weight, not exclude: a person/place that's genuinely the
# strongest shared signal still counts, just less — and it sinks in the surfaced "why".
_ENTITY_KIND_WEIGHT: dict[str, float] = {
    "concept": 1.0,
    "method": 1.0,
    "tool": 1.0,
    "org": 0.5,
    "other": 0.5,
    "person": 0.25,
    "place": 0.25,
}
_DEFAULT_KIND_WEIGHT = 0.5


# `RelatedDocument` is defined in `core/types` (it crosses index→retrieve→agents/mcp/cli/webui)
# and imported above; re-exported in `__all__` so existing
# `from memex.index.graph_store import RelatedDocument` call sites keep working.


class EntityMention(BaseModel):
    """A document that mentions the queried entity (from the MENTIONS edge). `chunk_id` is a
    REPRESENTATIVE attested chunk (where the NER found the entity), `None` for edges written
    before the column existed (→ the consumer falls back to an FTS name-search for passages)."""

    doc_id: str
    title: str
    chunk_id: str | None = None


class CoOccurringEntity(BaseModel):
    """An entity that co-occurs with the queried entity in shared documents — the concept
    NEIGHBOURHOOD. Ranked by `shared_docs × IDF × kind_weight` (the same specificity model
    as `RelatedDocument`, so generic connectors like 'IP' / an instructor's name are
    filtered/down-weighted), most-significant first."""

    name: str
    kind: str
    shared_docs: int  # how many of the entity's docs ALSO mention this co-entity
    score: float


class EntitySuggestion(BaseModel):
    """A deterministic acronym ↔ expansion bridge candidate for a queried name —
    surfaced as a traversal LINK ("Also see" / "Did you mean?"), NEVER a forced
    identity merge (the project resolves identity exactly; see ADR-0011 + the
    SUGGEST-over-MERGE decision). `relation` is the bridge direction for the UI
    label: `expansion` = the query was an acronym and this is its multi-word form
    (`DNS` → `Domain Name System`); `acronym` = the reverse."""

    name: str
    kind: str
    doc_count: int  # distinct docs mentioning the suggested entity (for ranking + display)
    relation: str  # "expansion" | "acronym"


class EntityProfile(BaseModel):
    """The canonical graph identity of an entity name across the corpus (entity-centric
    discovery, ADR-0011). `query_name` is what the user asked; `matched_names`/`kinds` are
    what the case-insensitive resolution actually found (a name can be a `concept` in one
    doc and a `tool` in another, hence lists). `resolved=False` ⇒ the name is not a known
    entity in the graph (the surface then falls back to whole-corpus full-text search).
    `suggestions` are acronym ↔ expansion bridge links (deterministic initialism match) —
    populated on BOTH the resolved ("Also see") and unresolved ("Did you mean?") paths, and
    empty when no real bridge exists (e.g. `STP` whose expansion isn't in the graph)."""

    query_name: str
    matched_names: list[str]
    kinds: list[str]
    doc_count: int  # TRUE total of distinct docs mentioning the entity (mentions may be capped)
    mentions: list[EntityMention]
    cooccurring: list[CoOccurringEntity]
    resolved: bool
    suggestions: list[EntitySuggestion] = []


def _rank_co_occurring(
    rows: list[tuple[str, str, int, int]],
    n_docs: int,
    *,
    limit: int,
    min_shared_docs: int = 2,
) -> list[CoOccurringEntity]:
    """PURE scoring core of `entity_profile`'s co-occurring neighbourhood (separated for
    unit testing). `rows` are `(co_name, co_kind, shared_docs, co_df)` — `shared_docs` =
    how many of the seed entity's docs also mention the co-entity, `co_df` = how many docs
    in the whole corpus mention it. Score = `shared_docs × ln(n_docs / co_df) × kind_weight`:
    the `shared_docs` term rewards a co-entity that pervades the seed's docs, the IDF term
    rewards SPECIFIC co-entities, and the kind-weight down-weights incidental proper nouns —
    the same noise filter as `_rank_related_documents`, so a generic connector (high df, or
    person/place) sinks. Excludes near-universal entities (df > the generic fraction).

    `min_shared_docs` is the neighbourhood FLOOR (ADR-0011) — a co-entity sharing < this many
    docs with the seed is an incidental single-doc co-mention (the bulk of the noise:
    ports/sizes), not a recurring neighbour. Corpus-agnostic; tunable to 1. (The curated
    by-name `entity_stopwords` band-aid was removed 2026-05-29 — the OTTER NER backend types
    entities cleanly upstream, so a hand-curated, per-corpus name list no longer earns its
    keep in a general-purpose app.)"""
    if n_docs <= 1:
        return []
    df_cap = _RELATED_GENERIC_ENTITY_DF_FRACTION * n_docs
    out: list[CoOccurringEntity] = []
    for name, kind, shared_docs, df in rows:
        if df <= 0 or df > df_cap or shared_docs < min_shared_docs:
            continue
        weight = _ENTITY_KIND_WEIGHT.get(kind, _DEFAULT_KIND_WEIGHT)
        score = shared_docs * math.log(n_docs / df) * weight
        if score <= 0:
            continue
        out.append(
            CoOccurringEntity(name=name, kind=kind, shared_docs=shared_docs, score=round(score, 4))
        )
    out.sort(key=lambda c: (-c.score, c.name))
    return out[:limit]


_MAX_SUGGESTIONS = 3  # defensive cap; collision-drop already bounds a clean bridge to ≤1
# A suggested bridge must be attested in ≥ this many docs. A 1-doc entity is the weakest
# possible attestation, and that's exactly where cross-domain initialism FALSE-FRIENDS live
# (live: query "STP" → the 10-K's "Short-term portion", 1 doc — a coincidental initialism,
# not the networking concept). Every confirmed-real bridge recurs (DNS/TCP/DHCP/ICMP
# expansions all ≥2 docs), so the floor kills the noise without dropping a true bridge.
_MIN_SUGGESTION_DOC_COUNT = 2


def _gate_suggestions(
    candidates: list[tuple[str, str, str, int, str]],
    n_docs: int,
    *,
    exclude_ids: set[str],
) -> list[EntitySuggestion]:
    """PURE gate for `entity_profile`'s acronym ↔ expansion suggestions (unit-tested,
    mirrors `_rank_co_occurring`). `candidates` are `(entity_id, name, kind, doc_count,
    relation)` rows that already matched the bridge (an initialism match in one
    direction). Conservatively: drop the exact-resolved entity itself (`exclude_ids`),
    drop barely-attested entities (`doc_count < _MIN_SUGGESTION_DOC_COUNT` — the
    cross-domain false-friend floor), drop near-universal generics (`doc_count > the
    generic fraction`), collapse same-name-different-kind into ONE suggestion (the
    highest-doc representative — NOT a collision), then **drop everything if ≥2 DISTINCT
    names survive** (an ambiguous initialism → conservative no-op, the #256 rule — we
    don't guess which expansion the user meant). So a clean bridge yields 0 or 1 suggestion."""
    if n_docs <= 0:
        return []
    df_cap = _RELATED_GENERIC_ENTITY_DF_FRACTION * n_docs
    by_name: dict[str, EntitySuggestion] = {}
    for entity_id_, name, kind, doc_count, relation in candidates:
        if entity_id_ in exclude_ids or doc_count < _MIN_SUGGESTION_DOC_COUNT or doc_count > df_cap:
            continue
        key = name.lower()
        prev = by_name.get(key)
        if prev is None or doc_count > prev.doc_count:
            by_name[key] = EntitySuggestion(
                name=name, kind=kind, doc_count=doc_count, relation=relation
            )
    if len(by_name) > 1:  # ambiguous bridge → conservative no-op
        return []
    return sorted(by_name.values(), key=lambda s: (-s.doc_count, s.name))[:_MAX_SUGGESTIONS]


def _rank_related_documents(
    rows: list[tuple[str, str, str, str, int]],
    n_docs: int,
    *,
    limit: int,
    max_entities: int,
) -> list[RelatedDocument]:
    """PURE scoring core of `GraphStore.related_documents` (separated for unit testing).

    `rows` are `(neighbour_doc_id, neighbour_title, shared_entity, entity_kind, entity_df)`
    tuples — one per (neighbour, shared-entity) pair, where `entity_df` is how many
    documents in the corpus mention that entity. Scores each neighbour by
    Σ `ln(n_docs / df) × kind_weight` over its shared entities: IDF rewards specific
    (rare) entities, the `_ENTITY_KIND_WEIGHT` multiplier down-weights incidental
    proper-noun types (person/place). Near-universal entities (df above
    `_RELATED_GENERIC_ENTITY_DF_FRACTION` of the corpus) are excluded outright — the
    generic-connector noise the retired passive expansion fell for. Returns the top-`limit`
    neighbours, each with its connecting entities most-significant-first (capped at
    `max_entities`)."""
    if n_docs <= 1:
        return []
    df_cap = _RELATED_GENERIC_ENTITY_DF_FRACTION * n_docs
    agg: dict[str, dict[str, Any]] = {}
    for did, title, entity, kind, df in rows:
        if df <= 0 or df > df_cap:  # generic/noise entity → skip
            continue
        weight = _ENTITY_KIND_WEIGHT.get(kind, _DEFAULT_KIND_WEIGHT)
        contribution = math.log(n_docs / df) * weight
        if contribution <= 0:  # a fully-discounted or zero-IDF entity adds nothing
            continue
        rec = agg.setdefault(did, {"title": title or did, "score": 0.0, "ents": []})
        rec["score"] = float(rec["score"]) + contribution
        ents: list[tuple[float, str]] = rec["ents"]
        ents.append((contribution, entity))
    out: list[RelatedDocument] = [
        RelatedDocument(
            doc_id=did,
            title=str(rec["title"]),
            score=round(float(rec["score"]), 4),
            # most-specific first; dedup by name (the same name can have >1 entity node
            # — e.g. different `kind` — and a doubled "why" tag reads as sloppy).
            shared_entities=list(dict.fromkeys(e for _idf, e in sorted(rec["ents"], reverse=True)))[
                :max_entities
            ],
        )
        for did, rec in agg.items()
    ]
    # Rank by score desc; doc_id as a stable tiebreaker for deterministic output.
    out.sort(key=lambda r: (-r.score, r.doc_id))
    return out[:limit]


def _rank_bridges(
    rows: list[tuple[str, str, str, str, int]],
    n_docs: int,
    *,
    limit_bridges: int,
    max_docs_per_bridge: int,
    max_via: int,
) -> list[DocumentBridge]:
    """PURE entity-grouped inversion of `_rank_related_documents`'s rows (separated for unit
    testing) — the scoring core of `GraphStore.related_bridges` and the /graph "Bridges" view.

    Consumes the SAME `(neighbour_doc_id, neighbour_title, shared_entity, entity_kind,
    entity_df)` rows, applies the SAME `ln(n_docs/df) × kind_weight` contribution and the SAME
    generic-df exclusion — but groups per ENTITY instead of per doc. Each bridge's `strength` is
    `mean_contribution × ln(1 + doc_count)`: the per-edge SPECIFICITY (mean IDF×kind_weight)
    times a LOGARITHMICALLY-damped fan-out. Specificity dominates — fan-out enters sub-linearly
    so a near-generic entity shared by many docs (e.g. 'IP' in 28/47 docs, just under the
    generic cutoff) does NOT swamp a genuinely specific concept shared by a few; a linear ×
    fan-out would make breadth win and surface exactly the generic networking terms the view
    should bury. `doc_count` is still the literal "bridges N" (and orders nothing on its own).
    Within a bridge the reached docs carry their OVERALL relatedness score (identical to the doc
    lens, computed in the same first pass) and their OTHER connecting entities as `via_entities`.
    Bridges sorted by strength desc (entity name as the stable tiebreaker); docs within a bridge
    by overall score."""
    if n_docs <= 1:
        return []
    df_cap = _RELATED_GENERIC_ENTITY_DF_FRACTION * n_docs
    # First pass — per-doc overall aggregate (score + ranked entities), IDENTICAL math to
    # `_rank_related_documents`, so a doc's score + via-ordering match the doc lens exactly.
    doc_agg: dict[str, dict[str, Any]] = {}
    # Per-entity aggregate, keyed by lowercased NAME (same-name-different-kind merges into one
    # bridge — they read as one concept; the displayed kind is the strongest-contributing link's).
    ent_agg: dict[str, dict[str, Any]] = {}
    for did, title, entity, kind, df in rows:
        if df <= 0 or df > df_cap:  # generic/noise entity → skip (same gate as the doc lens)
            continue
        weight = _ENTITY_KIND_WEIGHT.get(kind, _DEFAULT_KIND_WEIGHT)
        contribution = math.log(n_docs / df) * weight
        if contribution <= 0:
            continue
        drec = doc_agg.setdefault(did, {"title": title or did, "score": 0.0, "ents": []})
        drec["score"] = float(drec["score"]) + contribution
        cast(list[tuple[float, str]], drec["ents"]).append((contribution, entity))
        key = entity.lower()
        erec = ent_agg.setdefault(
            key, {"name": entity, "kind": kind, "best": 0.0, "strength": 0.0, "docs": {}}
        )
        erec["strength"] = float(erec["strength"]) + contribution
        if contribution > float(erec["best"]):  # display name/kind from the strongest link
            erec["best"], erec["name"], erec["kind"] = contribution, entity, kind
        cast(dict[str, float], erec["docs"])[did] = contribution
    bridges: list[DocumentBridge] = []
    for erec in ent_agg.values():
        ename = str(erec["name"])
        ename_lower = ename.lower()
        doc_ids = cast(dict[str, float], erec["docs"])
        # strength = mean per-edge specificity × log-damped fan-out (see the docstring): the
        # raw accumulator is Σ contribution, so raw/doc_count is the mean per-edge IDF×weight.
        dc = len(doc_ids)
        strength = (float(erec["strength"]) / dc) * math.log1p(dc) if dc else 0.0
        ordered = sorted(doc_ids, key=lambda did: (-float(doc_agg[did]["score"]), did))
        bridge_docs: list[BridgeDoc] = []
        for did in ordered[:max_docs_per_bridge]:
            drec = doc_agg[did]
            ranked = [e for _c, e in sorted(cast(list[tuple[float, str]], drec["ents"]), reverse=True)]
            via = [e for e in dict.fromkeys(ranked) if e.lower() != ename_lower][:max_via]
            bridge_docs.append(
                BridgeDoc(
                    doc_id=did,
                    title=str(drec["title"]),
                    score=round(float(drec["score"]), 4),
                    via_entities=via,
                )
            )
        bridges.append(
            DocumentBridge(
                entity=ename,
                kind=str(erec["kind"]),
                doc_count=dc,
                strength=round(strength, 4),
                docs=bridge_docs,
            )
        )
    bridges.sort(key=lambda b: (-b.strength, b.entity.lower(), b.entity))
    return bridges[:limit_bridges]


def _cooccurring_min_shared_docs() -> int:
    """The co-occurring neighbourhood FLOOR from settings, FAIL-OPEN to the default `2` when
    settings aren't initialised — a `GraphStore` opened outside bootstrap (tests, scripts)
    still ranks. Mirrors the `AgentsSettings.cooccurring_min_shared_docs` / `_rank_co_occurring`
    default. (The curated `entity_stopwords` band-aid was removed 2026-05-29 — the OTTER NER
    backend types entities cleanly upstream, so a per-corpus name list no longer earns its keep.)"""
    try:
        return get_settings().agents.cooccurring_min_shared_docs
    except ConfigurationError:
        return 2


def entity_id(name: str, kind: str) -> str:
    """Stable, content-derived entity id. Same (name, kind) ⇒ same id."""
    norm = f"{kind}::{name.strip().lower()}"
    # Content-addressing, not security (see chunker._stable_chunk_id).
    digest = hashlib.sha1(norm.encode("utf-8"), usedforsecurity=False)
    return "ent_" + digest.hexdigest()[:16]


class GraphStore:
    """Async wrapper around RyuGraph. Use `await GraphStore.open(vault_path)`."""

    def __init__(self, conn: object) -> None:
        # ryugraph is an optional dependency imported lazily inside
        # `open` (the `expand_graph` node falls back gracefully when it
        # is absent), so its `Connection` has no usable static type
        # here. Hold it as explicit `Any` so the `.execute(...)` /
        # result-cursor calls below are a contained dynamic boundary
        # rather than leaking inferred Unknown under strict.
        self._conn: Any = conn

    @classmethod
    async def open(cls, vault_path: Path) -> GraphStore:
        """Open (or create) the RyuGraph database under
        `{vault_path}/.memex/graph.ryu` and apply the Cypher schema in
        `index/schemas/graph.cypher`. Raises `ImportError` if the
        `ryugraph` package isn't installed (the agent's
        `expand_graph` node falls back gracefully)."""
        path = vault_path / ".memex" / "graph.ryu"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        def _connect() -> object:
            # The active fork's distribution is `ryugraph` (ADR-0005).
            # If the import name differs in practice, adjust here.
            import ryugraph  # type: ignore[import-not-found]

            db = ryugraph.Database(str(path))
            conn = ryugraph.Connection(db)
            # Apply the (pre-parsed, idempotent) DDL only the first time this
            # process opens this path — the on-disk schema persists, so later
            # opens skip it. Lock makes check-and-apply atomic across to_thread
            # workers; only contended on the very first open(s) of a path.
            key = str(path)
            with _schema_lock:
                if key not in _schema_applied_paths:
                    for stmt in _SCHEMA_STATEMENTS:
                        conn.execute(stmt)
                    # Migrate graphs created before MENTIONS.chunk_id existed: CREATE IF NOT
                    # EXISTS won't add a column to a pre-existing table, so ALTER it. Guarded
                    # (a fresh graph already has the column from the CREATE above) — RyuGraph
                    # raises "already has property" on the no-op, which we swallow; any other
                    # error re-raises (a real schema problem, not the idempotent case).
                    try:
                        conn.execute("ALTER TABLE MENTIONS ADD chunk_id STRING;")
                    except RuntimeError as e:
                        if "already has property" not in str(e):
                            raise
                    _schema_applied_paths.add(key)
            return conn

        conn = await asyncio.to_thread(_connect)
        return cls(conn)

    async def upsert_document(self, doc_id: str, title: str) -> None:
        """Insert or update the Document node. Called once per
        `index_document` run; the enrich stage adds MENTIONS / CITES
        edges separately."""

        def _run() -> None:
            self._conn.execute(
                "MERGE (d:Document {doc_id: $id}) SET d.title = $title;",
                {"id": doc_id, "title": title},
            )

        await asyncio.to_thread(_run)
        logger.info("graph.upsert_document", doc_id=doc_id)

    async def upsert_entity(self, name: str, kind: str) -> str:
        """Insert or update the entity. Returns the stable entity_id."""
        eid = entity_id(name, kind)

        def _run() -> None:
            self._conn.execute(
                "MERGE (e:Entity {entity_id: $id}) SET e.name = $name, e.kind = $kind;",
                {"id": eid, "name": name, "kind": kind},
            )

        await asyncio.to_thread(_run)
        return eid

    async def link_mentions(
        self, doc_id: str, entity_id_: str, confidence: float, chunk_id: str | None = None
    ) -> None:
        """Insert or update a `(Document)-[MENTIONS]->(Entity)` edge with the extractor's
        confidence score + a REPRESENTATIVE attested `chunk_id` (where the entity was found;
        `None` leaves it unset → callers fall back to FTS). Idempotent on the
        `(doc_id, entity_id)` pair."""

        def _run() -> None:
            self._conn.execute(
                "MATCH (d:Document {doc_id: $doc_id}), "
                "(e:Entity {entity_id: $entity_id}) "
                "MERGE (d)-[r:MENTIONS]->(e) "
                "SET r.confidence = $confidence, r.chunk_id = $chunk_id;",
                {
                    "doc_id": doc_id,
                    "entity_id": entity_id_,
                    "confidence": confidence,
                    "chunk_id": chunk_id,
                },
            )

        await asyncio.to_thread(_run)

    async def clear_mentions(self, doc_id: str) -> None:
        """Delete a document's outgoing MENTIONS edges so a re-enrich REPLACES (not appends)
        its entities. Entity nodes are left in place (an orphan with no MENTIONS is invisible
        to the discovery queries; `memex doctor` prunes them) and other documents' edges are
        untouched. Without this, re-enriching with a changed entity set — notably the OTTER
        NER backend (`enrich_ner_backend="otter"`), whose entities differ wholesale from the
        LLM's — would leave the prior extractor's stale MENTIONS alongside the new ones."""

        def _run() -> None:
            self._conn.execute(
                "MATCH (:Document {doc_id: $id})-[m:MENTIONS]->() DELETE m;",
                {"id": doc_id},
            )

        await asyncio.to_thread(_run)
        logger.info("graph.clear_mentions", doc_id=doc_id)

    async def link_cites(
        self,
        from_doc_id: str,
        to_doc_id: str,
        surface_text: str,
        confidence: float,
    ) -> None:
        """Insert or update a `(Document)-[CITES]->(Document)` edge
        carrying the resolved surface text (e.g., the wikilink target)
        plus the resolver's confidence."""

        def _run() -> None:
            self._conn.execute(
                "MATCH (a:Document {doc_id: $from_id}), "
                "(b:Document {doc_id: $to_id}) "
                "MERGE (a)-[r:CITES]->(b) "
                "SET r.surface_text = $surface, r.confidence = $confidence;",
                {
                    "from_id": from_doc_id,
                    "to_id": to_doc_id,
                    "surface": surface_text,
                    "confidence": confidence,
                },
            )

        await asyncio.to_thread(_run)

    async def delete_document(self, doc_id: str) -> None:
        """Drop the Document node and every edge incident to it.

        Entities, concepts, and citations are not removed (they may
        still be referenced by other documents). `memex doctor` is the
        place that prunes orphaned nodes later.
        """

        def _run() -> None:
            # Detach-delete unbinds and deletes the node in one statement.
            self._conn.execute(
                "MATCH (d:Document {doc_id: $id}) DETACH DELETE d;",
                {"id": doc_id},
            )

        await asyncio.to_thread(_run)
        logger.info("graph.delete_document", doc_id=doc_id)

    async def neighbors(self, doc_id: str, *, limit: int = 50) -> list[GraphNeighbor]:
        """Documents that share entities with `doc_id` (one-hop)."""

        def _run() -> list[GraphNeighbor]:
            # `self._conn` is the dynamic ryugraph connection (Any); each
            # result row is a positional tuple matching the RETURN
            # clause: (doc_id, title, via).
            result = self._conn.execute(
                "MATCH (d:Document {doc_id: $id})-[:MENTIONS]->(e:Entity)"
                "<-[:MENTIONS]-(other:Document) "
                "WHERE other.doc_id <> $id "
                "RETURN DISTINCT other.doc_id AS doc_id, "
                "other.title AS title, e.name AS via "
                "LIMIT $limit;",
                {"id": doc_id, "limit": limit},
            )
            out: list[GraphNeighbor] = []
            while result.has_next():
                row = result.get_next()
                out.append(
                    GraphNeighbor(
                        doc_id=row[0],
                        title=row[1] or row[0],
                        relation="shares_entity",
                        via=row[2],
                    )
                )
            return out

        return await asyncio.to_thread(_run)

    async def related_documents(
        self, doc_id: str, *, limit: int = 10, max_entities: int = 8
    ) -> list[RelatedDocument]:
        """Documents related to `doc_id`, ranked by the SPECIFICITY of their shared
        entities (IDF). This is the "explore connections" discovery surface — the
        on-mission successor to the retired passive `expand_graph`.

        Where `neighbors()` returns shared-entity neighbours UNRANKED (so generic
        entities — 'IP', 'HTTP', an instructor's name that appears in every doc —
        dominate and the result is noise), this scores each neighbour by Σ IDF(e) over
        the entities it shares with `doc_id`, where `IDF(e) = ln(N / df(e))` and `df(e)`
        is how many documents mention `e`. A rare shared entity (low df → high IDF) is a
        strong topical link; a near-universal one (df → N → IDF → 0) contributes ~nothing
        and, above `_RELATED_GENERIC_ENTITY_DF_FRACTION` of the corpus, is excluded from
        the surfaced `shared_entities` entirely. So a doc sharing ONE specific concept
        outranks one sharing five generic terms — the meaningful connection wins."""

        def _run() -> list[RelatedDocument]:
            rows, n_docs = self._fetch_shared_entity_rows(doc_id)
            return _rank_related_documents(rows, n_docs, limit=limit, max_entities=max_entities)

        return await asyncio.to_thread(_run)

    def _fetch_shared_entity_rows(
        self, doc_id: str
    ) -> tuple[list[tuple[str, str, str, str, int]], int]:
        """Sync fetch of the `(neighbour_doc_id, title, shared_entity, kind, df)` rows + the
        corpus doc-count that feed BOTH `related_documents` (doc-grouped) and `related_bridges`
        (entity-grouped). One MATCH per (neighbour, shared-entity) pair, each carrying the
        entity's GLOBAL doc-frequency. Runs on the `to_thread` worker of whichever caller. An
        empty / single-doc corpus returns `([], n_docs)` (both rankers no-op there)."""
        n_res = self._conn.execute("MATCH (d:Document) RETURN count(d) AS n;")
        n_docs = int(n_res.get_next()[0]) if n_res.has_next() else 0
        if n_docs <= 1:
            return [], n_docs
        result = self._conn.execute(
            "MATCH (d:Document {doc_id: $id})-[:MENTIONS]->(e:Entity)"
            "<-[:MENTIONS]-(other:Document) "
            "WHERE other.doc_id <> $id "
            "WITH other, e "
            "MATCH (e)<-[:MENTIONS]-(m:Document) "
            "WITH other, e, count(DISTINCT m) AS df "
            "RETURN other.doc_id AS doc_id, other.title AS title, "
            "e.name AS entity, e.kind AS kind, df;",
            {"id": doc_id},
        )
        rows: list[tuple[str, str, str, str, int]] = []
        while result.has_next():
            row = result.get_next()
            rows.append((row[0], row[1], row[2], row[3] or "other", int(row[4])))
        return rows, n_docs

    async def related_bridges(
        self,
        doc_id: str,
        *,
        limit_bridges: int = 24,
        max_docs_per_bridge: int = 50,
        max_via: int = 5,
    ) -> list[DocumentBridge]:
        """The ENTITY-grouped lens on `doc_id`'s neighbourhood — powers the /graph "Bridges"
        view. Where `related_documents` groups shared entities under each related DOC, this
        inverts the SAME data (same Cypher fetch, same specificity scoring) to group related
        docs under each shared ENTITY — the bridging concept. Bridges are ranked by `strength`
        (Σ IDF×kind_weight over the links they form: a rare entity shared by many docs wins);
        see the pure, unit-tested `_rank_bridges`."""

        def _run() -> list[DocumentBridge]:
            rows, n_docs = self._fetch_shared_entity_rows(doc_id)
            return _rank_bridges(
                rows,
                n_docs,
                limit_bridges=limit_bridges,
                max_docs_per_bridge=max_docs_per_bridge,
                max_via=max_via,
            )

        return await asyncio.to_thread(_run)

    async def citations(self, doc_id: str) -> DocumentCitations:
        """The document's 1-hop `CITES` neighbourhood — what it cites + what cites it.

        `CITES` edges are Document→Document, written at enrich from the resolved IN-VAULT
        citations (academic + course cross-references; `enrich/citations.py`). This is the
        honest "References" discovery surface that finally READS those edges (they were
        previously write-only — only the body `[[wikilinks]]` consumed the resolution).
        1-HOP only: transitive citation-chain following stays data-gated until the vault holds
        a citation-LINKED cluster dense enough to traverse (`docs/specs/graph-discovery.md`).
        Read-only ⇒ HARD-gate-neutral."""

        def _one(direction: str) -> list[CitationLink]:
            # `direction` picks the arrow: outgoing = this doc cites others; incoming = cited-by.
            match = (
                "MATCH (d:Document {doc_id: $id})-[c:CITES]->(o:Document) "
                if direction == "out"
                else "MATCH (o:Document)-[c:CITES]->(d:Document {doc_id: $id}) "
            )
            result = self._conn.execute(
                match
                + "RETURN o.doc_id AS doc_id, o.title AS title, c.surface_text AS surface "
                "ORDER BY title;",
                {"id": doc_id},
            )
            out: list[CitationLink] = []
            while result.has_next():
                row = result.get_next()
                out.append(
                    CitationLink(doc_id=row[0], title=row[1] or row[0], surface_text=row[2] or "")
                )
            return out

        def _run() -> DocumentCitations:
            return DocumentCitations(cites=_one("out"), cited_by=_one("in"))

        return await asyncio.to_thread(_run)

    async def entity_profile(
        self, name: str, *, max_docs: int = 50, max_cooccurring: int = 15
    ) -> EntityProfile:
        """The canonical graph profile of an entity NAME across the corpus (entity-centric
        discovery, ADR-0011). Resolves `name` case-insensitively (aggregating across kinds —
        a name can be a `concept` in one doc, a `tool` in another), then returns its identity
        (matched names, kind(s), true doc_count), the documents that mention it (capped at
        `max_docs`), and its co-occurring entity neighbourhood (ranked by `_rank_co_occurring`,
        same specificity filter as `related_documents`). `resolved=False` (+ empty lists) when
        the name is not a known entity — the caller (`retrieve.entity.entity_overview`) then
        falls back to whole-corpus full-text search. The quoted PASSAGES are NOT here: the
        MENTIONS edge is doc-level only, so passages come from FTS at the orchestrator."""
        query_name = name.strip()
        key = query_name.lower()
        min_shared = _cooccurring_min_shared_docs()

        def _compute_suggestions(exclude_ids: set[str], n_docs: int) -> list[EntitySuggestion]:
            """The deterministic acronym ↔ expansion bridge (ADR-0011). Direction A —
            query IS acronym-shaped: scan entity names, keep those whose derived
            initialism equals the query (the EXPANSION). Direction B — query is
            multi-word: derive its initialism + exact-probe the bare-acronym entity
            (the ACRONYM). The pure `_gate_suggestions` then drops self/generic/zero-doc
            and ambiguous-collision candidates. Empty when no real bridge exists."""
            if n_docs <= 0 or not query_name:
                return []
            candidates: list[tuple[str, str, str, int, str]] = []
            if looks_like_acronym(query_name):
                # Direction A. A bounded full-name scan (initials can't be derived in
                # Cypher); ~entity-count rows, only on an acronym-shaped query, on a
                # human-driven discovery surface — invisible at this scale.
                scan = self._conn.execute(
                    "MATCH (e:Entity) "
                    "OPTIONAL MATCH (e)<-[:MENTIONS]-(d:Document) "
                    "RETURN e.entity_id AS entity_id, e.name AS name, e.kind AS kind, "
                    "count(DISTINCT d) AS doc_count;"
                )
                while scan.has_next():
                    row = scan.get_next()
                    cand_name = row[1] or ""
                    if initialism_matches(query_name, cand_name):
                        candidates.append(
                            (row[0], cand_name, row[2] or "other", int(row[3]), "expansion")
                        )
            else:
                derived = derive_initialism(query_name)
                if derived is not None:
                    probe = self._conn.execute(
                        "MATCH (e:Entity) WHERE lower(e.name) = $acr "
                        "OPTIONAL MATCH (e)<-[:MENTIONS]-(d:Document) "
                        "RETURN e.entity_id AS entity_id, e.name AS name, e.kind AS kind, "
                        "count(DISTINCT d) AS doc_count;",
                        {"acr": derived.lower()},
                    )
                    while probe.has_next():
                        row = probe.get_next()
                        candidates.append(
                            (row[0], row[1] or derived, row[2] or "other", int(row[3]), "acronym")
                        )
            return _gate_suggestions(candidates, n_docs, exclude_ids=exclude_ids)

        def _run() -> EntityProfile:
            if not key:
                return EntityProfile(
                    query_name=query_name,
                    matched_names=[],
                    kinds=[],
                    doc_count=0,
                    mentions=[],
                    cooccurring=[],
                    resolved=False,
                )
            # A — resolve (case-insensitive) + identity. One row per (entity, kind);
            # doc_count is the entity's own mention count (summed across the matched ids).
            res = self._conn.execute(
                "MATCH (e:Entity) WHERE lower(e.name) = $name "
                "OPTIONAL MATCH (e)<-[:MENTIONS]-(d:Document) "
                "RETURN e.entity_id AS entity_id, e.name AS name, e.kind AS kind, "
                "count(DISTINCT d) AS doc_count;",
                {"name": key},
            )
            ids: list[str] = []
            names: list[str] = []
            kinds: list[str] = []
            while res.has_next():
                row = res.get_next()
                ids.append(row[0])
                if row[1] and row[1] not in names:
                    names.append(row[1])
                if row[2] and row[2] not in kinds:
                    kinds.append(row[2])
            n_res = self._conn.execute("MATCH (d:Document) RETURN count(d) AS n;")
            n_docs = int(n_res.get_next()[0]) if n_res.has_next() else 0
            # Suggestions run on BOTH paths (the unresolved "Did you mean?" case is the
            # most valuable); exclude the exact-resolved ids so we never suggest the match.
            suggestions = _compute_suggestions(set(ids), n_docs)

            if not ids:
                return EntityProfile(
                    query_name=query_name,
                    matched_names=[],
                    kinds=[],
                    doc_count=0,
                    mentions=[],
                    cooccurring=[],
                    resolved=False,
                    suggestions=suggestions,
                )

            # B — distinct docs mentioning the entity (TRUE total, then the capped list).
            dc_res = self._conn.execute(
                "MATCH (e:Entity)<-[:MENTIONS]-(d:Document) WHERE e.entity_id IN $ids "
                "RETURN count(DISTINCT d) AS n;",
                {"ids": ids},
            )
            doc_count = int(dc_res.get_next()[0]) if dc_res.has_next() else 0
            m_res = self._conn.execute(
                "MATCH (e:Entity)<-[r:MENTIONS]-(d:Document) WHERE e.entity_id IN $ids "
                # `collect(r.chunk_id)` groups by the non-aggregated keys (doc_id, title) — so
                # it dedups docs (replacing the old DISTINCT) AND gathers the doc's attested
                # chunk_ids; we take the first non-null as the representative. ORDER BY the
                # projected ALIAS (after the implicit GROUP-BY projection ryugraph drops `d`).
                "RETURN d.doc_id AS doc_id, d.title AS title, collect(r.chunk_id) AS chunk_ids "
                "ORDER BY doc_id LIMIT $max_docs;",
                {"ids": ids, "max_docs": max_docs},
            )
            mentions: list[EntityMention] = []
            while m_res.has_next():
                row = m_res.get_next()
                cids = cast("list[str | None]", row[2] or [])  # the dynamic ryugraph LIST
                rep = next((c for c in cids if c), None)  # first non-null attested chunk
                mentions.append(EntityMention(doc_id=row[0], title=row[1] or row[0], chunk_id=rep))

            # C — co-occurring entities (the concept neighbourhood), with each co-entity's
            # global doc-frequency for the specificity rank. Mirrors related_documents' Cypher.
            co_res = self._conn.execute(
                "MATCH (e:Entity)<-[:MENTIONS]-(d:Document)-[:MENTIONS]->(co:Entity) "
                "WHERE e.entity_id IN $ids AND NOT co.entity_id IN $ids "
                "WITH co, count(DISTINCT d) AS shared_docs "
                "MATCH (co)<-[:MENTIONS]-(m:Document) "
                "WITH co, shared_docs, count(DISTINCT m) AS df "
                "RETURN co.name AS name, co.kind AS kind, shared_docs, df;",
                {"ids": ids},
            )
            co_rows: list[tuple[str, str, int, int]] = []
            while co_res.has_next():
                row = co_res.get_next()
                co_rows.append((row[0], row[1] or "other", int(row[2]), int(row[3])))
            cooccurring = _rank_co_occurring(
                co_rows,
                n_docs,
                limit=max_cooccurring,
                min_shared_docs=min_shared,
            )

            return EntityProfile(
                query_name=query_name,
                matched_names=names,
                kinds=kinds,
                doc_count=doc_count,
                mentions=mentions,
                cooccurring=cooccurring,
                resolved=True,
                suggestions=suggestions,
            )

        return await asyncio.to_thread(_run)

    async def close(self) -> None:
        """Release the ryugraph connection.

        N4 (audit 2026-05-20): ryugraph (kuzu fork) doesn't expose an
        explicit `close()` on its Connection / Database objects — they
        rely on RAII: when the last Python reference is dropped, the
        C++ destructor runs and flushes WAL / closes file handles.
        Setting `self._conn = None` drops the only reference we hold,
        which lets the destructor fire deterministically rather than
        waiting for GC to notice. Idempotent: a second `close()` is a
        no-op because the attribute is already `None`.

        Callers that hold a `GraphStore` past close() will fail on the
        next read with a clear AttributeError on the `None` conn —
        better than the silent stale-data behaviour they'd get if we
        kept the reference alive.
        """
        self._conn = None
        # Yield once to keep the async contract — historically callers
        # may have relied on this being awaitable.
        await asyncio.sleep(0)


# ── Cross-process open policy ────────────────────────────────────────────────────────────
# ryugraph (kuzu fork) takes an EXCLUSIVE lock on the DB directory on ANY open (read_only OR
# read-write). Cross-process, a held handle makes a concurrent open in another process raise
# `RuntimeError: IO exception: Could not set lock on file ...`. Empirically root-caused by
# `scripts/ryugraph_consistency_probe.py` (memory `ryugraph-consistency-probe-2026-06-07`):
# this is why connection REUSE is precluded, and why a brief reader/writer race (a webui
# discovery read vs a separate enrich/index/reindex process) needs handling instead of the
# bare `except ImportError` the call sites used to carry. Readers FAIL-OPEN; writers RETRY
# (a writer must not silently skip the graph write — that drops a doc's entities).

_WRITE_LOCK_RETRIES = 10
_WRITE_LOCK_BACKOFF_S = 0.2
_READ_LOCK_RETRIES = 2
_READ_LOCK_BACKOFF_S = 0.1


def is_graph_lock_error(exc: BaseException) -> bool:
    """True iff `exc` is ryugraph's cross-process lock-contention error — another process
    holds the exclusive directory lock. ryugraph wraps it as `RuntimeError: IO exception:
    Could not set lock on file ...`. Matched by MESSAGE (deliberately narrow): a bare
    `RuntimeError` also wraps real corruption / WAL-replay / schema errors, which must NOT be
    swallowed as 'graph busy' (the CLAUDE.md narrow-except rule)."""
    return isinstance(exc, RuntimeError) and "could not set lock" in str(exc).lower()


async def open_graph_for_read(
    vault_path: Path,
    *,
    retries: int = _READ_LOCK_RETRIES,
    backoff_s: float = _READ_LOCK_BACKOFF_S,
) -> GraphStore | None:
    """Open the graph for a READ, FAIL-OPEN to None when it's unavailable — ryugraph not
    installed (ImportError) OR another process holds the exclusive lock (a brief writer
    race). A short bounded retry rides out a brief window; persistent contention (e.g. a
    full `reindex --force`, which holds the lock across the whole rebuild) returns None so
    the discovery surface degrades gracefully (no Related panel) instead of 500ing. Real
    ryugraph errors (corruption / schema) PROPAGATE. Callers treat None like 'no graph'."""
    for attempt in range(retries + 1):
        try:
            return await GraphStore.open(vault_path)
        except ImportError as e:
            logger.warning("graph.open.unavailable", reason=str(e))
            return None
        except RuntimeError as e:
            if not is_graph_lock_error(e):
                raise
            if attempt >= retries:
                logger.warning("graph.read.locked", reason=str(e), attempts=attempt + 1)
                return None
            await asyncio.sleep(backoff_s)
    return None  # unreachable (the loop always returns or raises) — narrows the return type


async def open_graph_for_write(
    vault_path: Path,
    *,
    retries: int = _WRITE_LOCK_RETRIES,
    backoff_s: float = _WRITE_LOCK_BACKOFF_S,
) -> GraphStore | None:
    """Open the graph for a WRITE. Returns None when ryugraph isn't installed (the graph is
    OPTIONAL — the pipeline degrades gracefully without it). On cross-process lock contention
    RETRY a bounded number of times (a reader window is brief) then RE-RAISE — a writer must
    NOT fail-open, because silently skipping the graph write would drop a document's entities
    (an invisible, lossy partial ingest). Real ryugraph errors PROPAGATE."""
    for attempt in range(retries + 1):
        try:
            return await GraphStore.open(vault_path)
        except ImportError as e:
            logger.warning("graph.open.unavailable", reason=str(e), fix="uv sync installs ryugraph")
            return None
        except RuntimeError as e:
            if not is_graph_lock_error(e):
                raise
            if attempt >= retries:
                logger.warning("graph.write.lock_exhausted", reason=str(e), attempts=attempt + 1)
                raise
            await asyncio.sleep(backoff_s)
    raise AssertionError("unreachable")  # the loop always returns or raises
