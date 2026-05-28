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
from typing import Any

import structlog
from pydantic import BaseModel

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


# Entity-specificity threshold for `related_documents`: an entity mentioned by MORE than
# this fraction of the corpus is treated as NOISE (a generic connector like 'IP', 'HTTP',
# or an author/instructor name that appears everywhere) and excluded from the "why". IDF
# already down-weights such entities toward zero; the hard exclusion just keeps the
# surfaced connecting-entities list clean. This is the lever that distinguishes the
# `related_documents` discovery feature from the retired passive `expand_graph` (which
# linked on these generic entities, unranked, and added nothing — see db-audit-2026-05-28).
_RELATED_GENERIC_ENTITY_DF_FRACTION = 0.6


class RelatedDocument(BaseModel):
    """A document related to a seed doc via SHARED ENTITIES, ranked by the SPECIFICITY of
    those entities (IDF — a rare shared entity is a strong topical signal; a near-universal
    one is noise). The on-mission "explore connections" discovery surface, vs the retired
    passive `expand_graph` which linked on generic entities, unranked, and never helped."""

    doc_id: str
    title: str
    score: float  # Σ IDF(entity) over the shared, non-generic entities — higher = stronger
    shared_entities: list[str]  # the connecting entities, most-specific first


def _rank_related_documents(
    rows: list[tuple[str, str, str, int]],
    n_docs: int,
    *,
    limit: int,
    max_entities: int,
) -> list[RelatedDocument]:
    """PURE scoring core of `GraphStore.related_documents` (separated for unit testing).

    `rows` are `(neighbour_doc_id, neighbour_title, shared_entity, entity_df)` tuples —
    one per (neighbour, shared-entity) pair, where `entity_df` is how many documents in
    the corpus mention that entity. Scores each neighbour by Σ `ln(n_docs / df)` (IDF)
    over its shared entities, EXCLUDING near-universal entities (df above
    `_RELATED_GENERIC_ENTITY_DF_FRACTION` of the corpus — the generic-connector noise the
    retired passive expansion fell for). Returns the top-`limit` neighbours, each with its
    connecting entities most-specific-first (capped at `max_entities`)."""
    if n_docs <= 1:
        return []
    df_cap = _RELATED_GENERIC_ENTITY_DF_FRACTION * n_docs
    agg: dict[str, dict[str, Any]] = {}
    for did, title, entity, df in rows:
        if df <= 0 or df > df_cap:  # generic/noise entity → skip
            continue
        idf = math.log(n_docs / df)
        rec = agg.setdefault(did, {"title": title or did, "score": 0.0, "ents": []})
        rec["score"] = float(rec["score"]) + idf
        ents: list[tuple[float, str]] = rec["ents"]
        ents.append((idf, entity))
    out: list[RelatedDocument] = [
        RelatedDocument(
            doc_id=did,
            title=str(rec["title"]),
            score=round(float(rec["score"]), 4),
            # most-specific first; dedup by name (the same name can have >1 entity node
            # — e.g. different `kind` — and a doubled "why" tag reads as sloppy).
            shared_entities=list(
                dict.fromkeys(e for _idf, e in sorted(rec["ents"], reverse=True))
            )[:max_entities],
        )
        for did, rec in agg.items()
    ]
    # Rank by score desc; doc_id as a stable tiebreaker for deterministic output.
    out.sort(key=lambda r: (-r.score, r.doc_id))
    return out[:limit]


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

    async def link_mentions(self, doc_id: str, entity_id_: str, confidence: float) -> None:
        """Insert or update a `(Document)-[MENTIONS]->(Entity)` edge
        with the extractor's confidence score. Idempotent on the
        `(doc_id, entity_id)` pair."""

        def _run() -> None:
            self._conn.execute(
                "MATCH (d:Document {doc_id: $doc_id}), "
                "(e:Entity {entity_id: $entity_id}) "
                "MERGE (d)-[r:MENTIONS]->(e) SET r.confidence = $confidence;",
                {
                    "doc_id": doc_id,
                    "entity_id": entity_id_,
                    "confidence": confidence,
                },
            )

        await asyncio.to_thread(_run)

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
            n_res = self._conn.execute("MATCH (d:Document) RETURN count(d) AS n;")
            n_docs = int(n_res.get_next()[0]) if n_res.has_next() else 0
            if n_docs <= 1:
                return []
            # Per (neighbour, shared-entity) row, with the entity's GLOBAL doc-frequency.
            result = self._conn.execute(
                "MATCH (d:Document {doc_id: $id})-[:MENTIONS]->(e:Entity)"
                "<-[:MENTIONS]-(other:Document) "
                "WHERE other.doc_id <> $id "
                "WITH other, e "
                "MATCH (e)<-[:MENTIONS]-(m:Document) "
                "WITH other, e, count(DISTINCT m) AS df "
                "RETURN other.doc_id AS doc_id, other.title AS title, "
                "e.name AS entity, df;",
                {"id": doc_id},
            )
            rows: list[tuple[str, str, str, int]] = []
            while result.has_next():
                row = result.get_next()
                rows.append((row[0], row[1], row[2], int(row[3])))
            return _rank_related_documents(
                rows, n_docs, limit=limit, max_entities=max_entities
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
