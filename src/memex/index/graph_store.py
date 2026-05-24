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
import re
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schemas" / "graph.cypher"
_STMT_SPLIT = re.compile(r";\s*\n")


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


class GraphNeighbor(BaseModel):
    """One related document, surfaced by the graph for the `graph` CLI."""

    doc_id: str
    title: str
    relation: str
    via: str | None = None  # e.g. shared entity name


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
            schema = _strip_cypher_comments(_SCHEMA_PATH.read_text(encoding="utf-8"))
            for stmt in _STMT_SPLIT.split(schema):
                # If the file doesn't end with a newline after the final
                # `;`, the last split chunk retains its trailing `;` —
                # strip it so the re-append below doesn't produce `;;`
                # (which ryugraph's parser rejects).
                stmt = stmt.strip().rstrip(";").strip()
                if stmt:
                    conn.execute(stmt + ";")
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
