"""Opt-in REAL-ryugraph test for `GraphStore.entity_profile` — entity-centric
discovery (ADR-0011). Seeds an actual graph and runs the three live Cypher
queries (resolve + identity, mentioning docs, co-occurring traversal), closing
the no-live-Cypher-in-CI gap that the faked orchestrator test leaves open.

Skipped wholesale when ryugraph isn't installed (the optional graph backend).
The pure ranking is covered exhaustively by tests/unit/test_entity_profile.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ryugraph")  # optional backend — skip the whole module without it

from memex.index.graph_store import GraphStore

# Six docs so the generic-entity df exclusion (df > 0.6·N = 3.6) keeps the
# meaningful co-entity ARP (df=3) while the IDF math stays legible.
_SEED: list[tuple[str, list[tuple[str, str]]]] = [
    ("doc_a", [("STP", "concept"), ("ARP", "concept"), ("Charles Tounou", "person")]),
    ("doc_b", [("STP", "concept"), ("ARP", "concept")]),
    ("doc_c", [("ARP", "concept"), ("VLAN", "concept")]),
    ("doc_d", [("VLAN", "concept"), ("firewall", "concept")]),
    ("doc_e", [("firewall", "concept"), ("TLS", "concept")]),
    ("doc_f", [("routing", "concept"), ("TLS", "tool")]),  # TLS = two kinds, two docs
]


async def _seed(graph: GraphStore) -> None:
    for doc_id, ents in _SEED:
        await graph.upsert_document(doc_id, doc_id.replace("_", " ").title())
        for name, kind in ents:
            eid = await graph.upsert_entity(name, kind)
            await graph.link_mentions(doc_id, eid, 0.9)


@pytest.mark.asyncio
async def test_entity_profile_resolves_identity_docs_and_cooccurring(tmp_path: Path) -> None:
    graph = await GraphStore.open(tmp_path)
    try:
        await _seed(graph)
        # Case-insensitive resolve ("stp" → "STP").
        profile = await graph.entity_profile("stp")

        assert profile.resolved is True
        assert profile.matched_names == ["STP"]
        assert profile.kinds == ["concept"]
        assert profile.doc_count == 2  # doc_a + doc_b
        assert {m.doc_id for m in profile.mentions} == {"doc_a", "doc_b"}

        # Co-occurring: ARP (specific concept, shared in both STP docs) must lead;
        # Charles Tounou (proper noun, single shared doc) ranks strictly below it.
        names = [c.name for c in profile.cooccurring]
        assert names, "expected a non-empty co-occurring neighbourhood"
        assert names[0] == "ARP"
        arp = profile.cooccurring[0]
        assert arp.shared_docs == 2
        assert arp.kind == "concept"
        if "Charles Tounou" in names:
            charles = next(c for c in profile.cooccurring if c.name == "Charles Tounou")
            assert arp.score > charles.score  # specificity + kind-weight rank
        # VLAN co-occurs with ARP, not with STP → must NOT appear here.
        assert "VLAN" not in names
    finally:
        await graph.close()


@pytest.mark.asyncio
async def test_entity_profile_aggregates_across_kinds(tmp_path: Path) -> None:
    """A name that is a `concept` in one doc and a `tool` in another resolves to
    BOTH kinds with the true union doc_count (the cross-kind aggregation contract)."""
    graph = await GraphStore.open(tmp_path)
    try:
        await _seed(graph)
        profile = await graph.entity_profile("TLS")
        assert profile.resolved is True
        assert profile.matched_names == ["TLS"]
        assert set(profile.kinds) == {"concept", "tool"}
        assert profile.doc_count == 2  # doc_e (concept) + doc_f (tool)
        assert {m.doc_id for m in profile.mentions} == {"doc_e", "doc_f"}
    finally:
        await graph.close()


@pytest.mark.asyncio
async def test_entity_profile_unknown_name_is_unresolved(tmp_path: Path) -> None:
    graph = await GraphStore.open(tmp_path)
    try:
        await _seed(graph)
        profile = await graph.entity_profile("definitely-not-an-entity")
        assert profile.resolved is False
        assert profile.matched_names == []
        assert profile.kinds == []
        assert profile.doc_count == 0
        assert profile.mentions == []
        assert profile.cooccurring == []
    finally:
        await graph.close()
