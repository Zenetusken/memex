"""Entity-centric retrieval — "everything across my corpus about entity X" (ADR-0011).

The orchestrator that composes the entity GRAPH (canonical identity + the authoritative
mentioning-doc set + the co-occurring concept neighbourhood) with the quoted PASSAGES. The
passages come, best-provenance first, from: (1) the ATTESTED chunks — the exact chunks the
NER found the entity in, carried on the MENTIONS edge as a representative `chunk_id` and
fetched by id (2026-05-29); (2) an FTS name-search scoped to the mentioning docs (for edges
written before the chunk_id column, or un-migrated graphs); (3) a whole-corpus FTS search
(unknown name). This is the genuinely-graph discovery surface, not a search reskin: the
profile is what BM25 can't give, and the attested passages point at the exact mention site.

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
from memex.index.graph_store import EntityProfile, open_graph_for_read

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

    # GRAPH: the profile (identity + mentioning docs + co-occurring). Optional + fail-open:
    # `open_graph_for_read` returns None when ryugraph is absent OR a concurrent writer holds
    # the exclusive lock (a brief enrich/index race), so the graph never blocks the entity
    # view — it falls back to the resolved=False stand-in + whole-corpus FTS passages below.
    profile: EntityProfile | None = None
    store = await open_graph_for_read(vault_path)
    if store is not None:
        try:
            profile = await store.entity_profile(
                name, max_docs=max_docs, max_cooccurring=max_cooccurring
            )
        except RuntimeError as e:
            # Honor the docstring's "fail-open throughout (never raises)" contract: the open
            # is already fail-open (open_graph_for_read), but a fault DURING the query —
            # ryugraph wraps a Cypher/schema/corruption error as RuntimeError — would propagate
            # and 500 the `/entity` route (which deliberately doesn't catch, trusting this).
            # Degrade to the whole-corpus FTS fallback (profile stays None). Programming bugs
            # (AttributeError/TypeError) still propagate — only ryugraph's RuntimeError class
            # is swallowed.
            log.warning("entity.profile_failed", reason=str(e)[:200])
        finally:
            await store.close()

    # PASSAGES — three tiers, best-provenance first:
    #  (1) the ATTESTED chunks: the exact chunks the NER found the entity in (the MENTIONS
    #      edges' representative chunk_ids) — highest provenance, no text search needed;
    #  (2) FTS name-search SCOPED to the mentioning docs (resolved, but no attested chunk —
    #      a graph enriched before the chunk_id column, or un-migrated edges);
    #  (3) a whole-corpus FTS search (unknown name / no graph) — the honest fallback.
    attested = (
        [m.chunk_id for m in profile.mentions if m.chunk_id]
        if (profile and profile.resolved)
        else []
    )
    scope_doc_ids = (
        [m.doc_id for m in profile.mentions] if (profile and profile.resolved) else []
    )
    fts = await FTSStore.open(vault_path)
    try:
        # Decide on the FETCHED passages, not just on `attested` being non-empty:
        # a re-indexed-but-not-re-enriched doc keeps STALE chunk_ids on MENTIONS, so
        # chunks_by_ids can miss every id → fall through to the scoped name-search
        # rather than return empty (scoped) passages.
        passages = (await fts.chunks_by_ids(attested))[:passages_k] if attested else []
        if not passages and scope_doc_ids:
            passages = await fts.search_in_docs(name, doc_ids=scope_doc_ids, k=passages_k)
        elif not passages:  # no scope docs (unknown name / no graph) → whole-corpus
            passages = await fts.search(name, k=passages_k)
        scoped = bool(scope_doc_ids)
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
