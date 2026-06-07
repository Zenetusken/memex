"""Related-documents aggregation — the shared "explore connections" core behind the /ask
Related panel + scope-set suggestions (webui) AND the MCP/CLI `ask` payloads.

Graph reads only, off the answer/refusal path ⇒ HARD-gate-neutral. Lives in `retrieve/`
(the FTS+vector composition layer; `retrieve/ → index/` is a legal edge, like `hybrid.py` /
`entity.py`) so every surface can reach it: `mcp/ → retrieve`, `cli/ → *`, `webui/ → retrieve`.
The agent NEVER populates these (it would add a graph open to every answer + the eval path);
the surfaces enrich the returned `FinalResponse.related_documents` post-hoc. Takes raw doc_ids
rather than a `FinalResponse` because `retrieve/ → agents/` is not a legal edge — each surface
extracts the cited doc_ids from its own response and calls in.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from memex.core.types import RelatedDocument

logger = structlog.get_logger(__name__)


async def related_documents_for_seeds(
    vault_path: Path,
    seed_ids: list[str],
    *,
    seed_limit: int = 5,
    per_seed: int = 8,
    out_limit: int = 6,
) -> list[RelatedDocument]:
    """Graph neighbours of a SET of seed docs: merged, deduped (higher-score relation wins),
    seed-EXCLUDED, re-ranked by score, capped. Reuses the noise-filtered `related_documents`
    specificity ranking, so callers inherit the IDF × kind-weight + shared-docs floor. Expands
    the first `seed_limit` seeds (bounds graph calls) but excludes the FULL seed set from the
    output. ImportError (ryugraph absent) → fail-open `[]`."""
    if not seed_ids:
        return []
    exclude = set(seed_ids)  # never suggest a doc already in the seed / citation set
    # Fail-open: ryugraph absent OR a concurrent writer holds the exclusive lock (a brief
    # enrich/index race) → None → no suggestions, never a crash. See open_graph_for_read.
    from memex.index.graph_store import open_graph_for_read

    store = await open_graph_for_read(vault_path)
    if store is None:
        return []
    merged: dict[str, RelatedDocument] = {}
    try:
        for doc_id in seed_ids[:seed_limit]:
            for r in await store.related_documents(doc_id, limit=per_seed):
                if r.doc_id in exclude:
                    continue
                prev = merged.get(r.doc_id)
                if prev is None or r.score > prev.score:  # dedup keeps the higher-score relation
                    merged[r.doc_id] = r
    finally:
        await store.close()
    return sorted(merged.values(), key=lambda r: (-r.score, r.doc_id))[:out_limit]


async def related_documents_for_answer(
    vault_path: Path,
    used_chunk_doc_ids: list[str],
    *,
    answered: bool,
) -> list[RelatedDocument]:
    """Graph neighbours of the docs an answer CITED — answered-only, seeded from the distinct
    `used_chunks` doc_ids, EXCLUDING those cited docs (handled by `related_documents_for_seeds`'s
    seed-exclusion). A refusal cited nothing groundable → `[]`. The surface passes
    `[c.document_id for c in response.used_chunks]` + `response.answered`."""
    if not answered or not used_chunk_doc_ids:
        return []
    seeds: list[str] = []
    for doc_id in used_chunk_doc_ids:
        if doc_id not in seeds:
            seeds.append(doc_id)
    return await related_documents_for_seeds(vault_path, seeds)
