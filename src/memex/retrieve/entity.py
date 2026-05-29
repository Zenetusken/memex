"""Entity-centric retrieval — "everything across my corpus about entity X" (ADR-0011).

The orchestrator that composes the entity GRAPH (canonical identity + the authoritative
mentioning-doc set + the co-occurring concept neighbourhood) with FULL-TEXT SEARCH (the
quoted passages — the MENTIONS edge is doc-level only, so the graph can't supply text).
This is the genuinely-graph discovery surface, not a search reskin: the profile is what
BM25 can't give, the passages are what the graph can't.

Read-only + HARD-gate-neutral: it never touches the answer/refusal path, only graph + BM25
reads. Lives in `retrieve/` (the FTS+vector composition layer; `retrieve/ → index/` is a
legal edge, same as `hybrid.py`).
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel

from memex.core.config import get_settings
from memex.core.types import Chunk
from memex.index.fts_store import FTSStore
from memex.index.graph_store import EntityProfile

logger = structlog.get_logger(__name__)


class EntityOverview(BaseModel):
    """An entity's corpus-wide profile (graph) + representative passages (FTS).

    `profile.resolved` distinguishes a known entity (graph identity + co-occurring
    neighbourhood + passages scoped to the mentioning docs) from an unknown name (the
    honest fallback: `resolved=False` + whole-corpus full-text passages, `passages_scoped`
    False). The two sources are deliberately distinct — be honest in any UI: documents +
    co-occurring concepts come from the entity graph; the quoted passages come from
    full-text search of those documents."""

    profile: EntityProfile
    passages: list[Chunk]
    passages_scoped: bool  # True = scoped to the entity's MENTIONS docs; False = whole-corpus


async def entity_overview(
    name: str,
    *,
    max_docs: int = 50,
    max_cooccurring: int = 15,
    passages_k: int = 10,
) -> EntityOverview:
    """Resolve `name` to its graph profile + fetch representative passages.

    Graph-resolved → passages are BM25-scoped to the entity's mentioning documents (the
    authoritative set). Unknown name OR graph unavailable → an `EntityProfile(resolved=False)`
    and a whole-corpus BM25 fallback (`passages_scoped=False`), so the user still gets text.
    Fail-open throughout (a missing/optional graph never raises)."""
    log = logger.bind(entity=name[:60])
    settings = get_settings()
    vault_path = settings.vault_path

    # GRAPH: the profile (identity + mentioning docs + co-occurring). Optional + fail-open.
    profile: EntityProfile | None = None
    try:
        from memex.index.graph_store import GraphStore

        store = await GraphStore.open(vault_path)
    except ImportError as e:
        log.warning("entity.graph_unavailable", reason=str(e))
    else:
        try:
            profile = await store.entity_profile(
                name, max_docs=max_docs, max_cooccurring=max_cooccurring
            )
        finally:
            await store.close()

    # FTS: the passages. Scoped to the mentioning docs when the entity resolved; otherwise
    # a whole-corpus search (the honest "not a known entity, here's what text search finds").
    scope_doc_ids = (
        [m.doc_id for m in profile.mentions] if (profile and profile.resolved) else []
    )
    fts = await FTSStore.open(vault_path)
    try:
        if scope_doc_ids:
            passages = await fts.search_in_docs(name, doc_ids=scope_doc_ids, k=passages_k)
            scoped = True
        else:
            passages = await fts.search(name, k=passages_k)
            scoped = False
    finally:
        await fts.close()

    if profile is None:  # graph unavailable → a resolved=False stand-in so callers are uniform
        profile = EntityProfile(
            query_name=name.strip(),
            matched_names=[],
            kinds=[],
            doc_count=0,
            mentions=[],
            cooccurring=[],
            resolved=False,
        )
    log.info(
        "entity.overview.done",
        resolved=profile.resolved,
        doc_count=profile.doc_count,
        cooccurring=len(profile.cooccurring),
        passages=len(passages),
        scoped=scoped,
    )
    return EntityOverview(profile=profile, passages=passages, passages_scoped=scoped)
