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

from memex.core.config import MemexSettings, set_settings
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
        assert profile.suggestions == []  # no bridge for an arbitrary string
    finally:
        await graph.close()


# ── Acronym ↔ expansion bridge (ADR-0011 follow-up) ──────────────────────────


async def _seed_spec(graph: GraphStore, spec: list[tuple[str, list[tuple[str, str]]]]) -> None:
    """Seed an explicit (doc_id, [(name, kind), …]) spec — for the bridge cases that
    need acronym + expansion as DISTINCT entities, separate from the shared _SEED."""
    for doc_id, ents in spec:
        await graph.upsert_document(doc_id, doc_id.replace("_", " ").title())
        for name, kind in ents:
            eid = await graph.upsert_entity(name, kind)
            await graph.link_mentions(doc_id, eid, 0.9)


@pytest.mark.asyncio
async def test_acronym_bridges_to_expansion_both_directions(tmp_path: Path) -> None:
    """DNS and 'Domain Name System' are SEPARATE entities → each resolves exactly AND
    suggests the other (the initialism bridge), with the right `relation` label."""
    graph = await GraphStore.open(tmp_path)
    try:
        await _seed_spec(
            graph,
            [
                ("d1", [("DNS", "concept"), ("TCP", "concept")]),
                ("d2", [("DNS", "concept")]),
                ("d3", [("Domain Name System", "concept"), ("TCP", "concept")]),
                ("d4", [("Domain Name System", "concept")]),
                ("d5", [("routing", "concept")]),
                ("d6", [("switching", "concept")]),
            ],
        )
        # Acronym → expansion: DNS resolves EXACTLY (identity unchanged), suggests the expansion.
        dns = await graph.entity_profile("DNS")
        assert dns.resolved is True
        assert dns.matched_names == ["DNS"]
        assert dns.doc_count == 2  # d1 + d2 — NOT merged with the expansion's docs
        sugg = [s for s in dns.suggestions if s.name == "Domain Name System"]
        assert len(sugg) == 1
        assert sugg[0].relation == "expansion"
        assert sugg[0].doc_count == 2  # d3 + d4

        # Expansion → acronym: the multi-word query suggests the bare acronym.
        expn = await graph.entity_profile("Domain Name System")
        assert expn.resolved is True
        assert expn.matched_names == ["Domain Name System"]
        assert expn.doc_count == 2  # d3 + d4, NOT merged
        back = [s for s in expn.suggestions if s.name == "DNS"]
        assert len(back) == 1
        assert back[0].relation == "acronym"
    finally:
        await graph.close()


@pytest.mark.asyncio
async def test_ambiguous_initialism_not_bridged(tmp_path: Path) -> None:
    """Two DISTINCT expansions deriving the same initialism → ambiguous → NO suggestion
    (the conservative no-op gate)."""
    graph = await GraphStore.open(tmp_path)
    try:
        await _seed_spec(
            graph,
            [
                # Each colliding expansion in 2 docs so it CLEARS the doc-count floor —
                # the COLLISION gate (not the floor) must be what drops the pair.
                ("d1", [("Network Time Protocol", "concept")]),
                ("d2", [("Network Time Protocol", "concept")]),
                ("d3", [("Nested Transaction Processor", "tool")]),
                ("d4", [("Nested Transaction Processor", "tool")]),
                ("d5", [("firewall", "concept")]),
                ("d6", [("routing", "concept")]),
            ],
        )
        # "NTP" matches BOTH expansions' initials → ambiguous → dropped entirely.
        ntp = await graph.entity_profile("NTP")
        assert ntp.resolved is False  # no bare "NTP" entity
        assert ntp.suggestions == []
    finally:
        await graph.close()


@pytest.mark.asyncio
async def test_stp_like_honest_miss_no_fabricated_bridge(tmp_path: Path) -> None:
    """The real STP situation: only a fragment 'spanning' exists — no 'STP', no
    'Spanning Tree Protocol'. entity_profile('STP') must NOT fabricate a bridge."""
    graph = await GraphStore.open(tmp_path)
    try:
        await _seed_spec(
            graph,
            [
                ("d1", [("spanning", "concept"), ("VLAN", "concept")]),
                ("d2", [("VLAN", "concept"), ("routing", "concept")]),
            ],
        )
        stp = await graph.entity_profile("STP")
        assert stp.resolved is False
        assert stp.matched_names == []
        assert stp.suggestions == []  # the design does NOT pretend to resolve STP
    finally:
        await graph.close()


@pytest.mark.asyncio
async def test_bridge_handles_whitespace_and_case(tmp_path: Path) -> None:
    """A padded, lowercase query ('  dns  ') still resolves exactly AND bridges —
    `query_name` is stripped and resolution is case-insensitive."""
    graph = await GraphStore.open(tmp_path)
    try:
        await _seed_spec(
            graph,
            [
                ("d1", [("DNS", "concept")]),
                ("d2", [("DNS", "concept")]),
                ("d3", [("Domain Name System", "concept")]),
                ("d4", [("Domain Name System", "concept")]),
                ("d5", [("routing", "concept")]),
            ],
        )
        prof = await graph.entity_profile("  dns  ")
        assert prof.resolved is True
        assert prof.matched_names == ["DNS"]
        assert [s.name for s in prof.suggestions] == ["Domain Name System"]
    finally:
        await graph.close()


@pytest.mark.asyncio
async def test_bridge_dedupes_multi_kind_expansion(tmp_path: Path) -> None:
    """An expansion that exists under TWO kinds (concept + tool) yields ONE suggestion
    (the highest-doc representative), not a duplicate per kind."""
    graph = await GraphStore.open(tmp_path)
    try:
        # Each kind of the expansion gets 2 docs so BOTH clear the doc-count floor —
        # then the dedup (not the floor) is what collapses them to one suggestion.
        await _seed_spec(
            graph,
            [
                ("d1", [("TLS", "concept")]),
                ("d2", [("TLS", "concept")]),
                ("d3", [("Transport Layer Security", "concept")]),
                ("d4", [("Transport Layer Security", "concept")]),
                ("d5", [("Transport Layer Security", "tool")]),  # same name, 2nd kind
                ("d6", [("Transport Layer Security", "tool")]),
                ("d7", [("routing", "concept")]),
            ],
        )
        prof = await graph.entity_profile("TLS")
        assert prof.resolved is True
        exp = [s for s in prof.suggestions if s.name == "Transport Layer Security"]
        assert len(exp) == 1  # deduped across kinds, not two entries
    finally:
        await graph.close()


@pytest.mark.asyncio
async def test_generic_expansion_excluded_from_suggestions(tmp_path: Path) -> None:
    """An initialism-matching expansion that is near-UNIVERSAL (df > 60% of the corpus)
    is a generic connector → excluded from suggestions, same filter as co-occurring."""
    graph = await GraphStore.open(tmp_path)
    try:
        # 4 docs → df_cap = 2.4. "Alpha Beta Cee" appears in 3 docs (generic); "ABC" in 1.
        await _seed_spec(
            graph,
            [
                ("d1", [("ABC", "concept"), ("Alpha Beta Cee", "concept")]),
                ("d2", [("Alpha Beta Cee", "concept")]),
                ("d3", [("Alpha Beta Cee", "concept")]),
                ("d4", [("routing", "concept")]),
            ],
        )
        prof = await graph.entity_profile("ABC")
        assert prof.resolved is True  # ABC itself resolves
        assert prof.suggestions == []  # the generic expansion is filtered out
    finally:
        await graph.close()


@pytest.mark.asyncio
async def test_resolved_acronym_with_no_expansion_has_no_suggestions(tmp_path: Path) -> None:
    """The COMMON case: an acronym that resolves but whose expansion isn't in the graph
    gets a clean empty suggestion list (no fabricated bridge)."""
    graph = await GraphStore.open(tmp_path)
    try:
        await _seed_spec(
            graph,
            [
                ("d1", [("ARP", "concept"), ("DNS", "concept")]),
                ("d2", [("ARP", "concept")]),
                ("d3", [("routing", "concept")]),
            ],
        )
        prof = await graph.entity_profile("ARP")  # no "Address Resolution Protocol" seeded
        assert prof.resolved is True
        assert prof.suggestions == []
    finally:
        await graph.close()


@pytest.mark.asyncio
async def test_entity_stopword_excludes_cooccurring_on_real_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end on a real graph: a curated `entity_stopwords` (via env → MemexSettings →
    the ranker) removes a multi-kind `CR350`-shaped connector from the co-occurring set —
    across ALL its kind-nodes — while the real neighbour (TCP) stays. Without it, CR350
    surfaces (it's under the df-gate and clears the floor)."""
    graph = await GraphStore.open(tmp_path)
    try:
        await _seed_spec(
            graph,
            [
                (
                    "d1",
                    [
                        ("DNS", "concept"),
                        ("CR350", "concept"),
                        ("CR350", "org"),
                        ("TCP", "concept"),
                    ],
                ),
                (
                    "d2",
                    [
                        ("DNS", "concept"),
                        ("CR350", "concept"),
                        ("CR350", "org"),
                        ("TCP", "concept"),
                    ],
                ),
                ("d3", [("DNS", "concept"), ("CR350", "concept")]),
                ("d4", [("TCP", "concept")]),
                ("d5", [("routing", "concept")]),
                ("d6", [("switching", "concept")]),
            ],
        )
        # Fail-open default (no stopword) → CR350 surfaces (shared_docs ≥ 2, under df-gate).
        before = await graph.entity_profile("DNS")
        assert "CR350" in {c.name for c in before.cooccurring}

        # Configure the stopword via env → the ranker drops every CR350 kind-node.
        monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("MEMEX_AGENTS__ENTITY_STOPWORDS", "CR350")
        set_settings(MemexSettings())  # type: ignore[call-arg]
        try:
            after = await graph.entity_profile("DNS")
        finally:
            set_settings(None)
        names = {c.name for c in after.cooccurring}
        assert "CR350" not in names
        assert "TCP" in names  # the real neighbour is untouched
    finally:
        await graph.close()


@pytest.mark.asyncio
async def test_cooccurring_floor_drops_single_shared_doc_on_real_graph(tmp_path: Path) -> None:
    """The default neighbourhood floor (≥2 shared docs) drops an incidental single-doc
    co-mention on a real graph, while a co-entity sharing ≥2 of the seed's docs survives."""
    graph = await GraphStore.open(tmp_path)
    try:
        await _seed_spec(
            graph,
            [
                ("d1", [("SEEDX", "concept"), ("PERSISTENT", "concept"), ("ONEOFF", "concept")]),
                ("d2", [("SEEDX", "concept"), ("PERSISTENT", "concept")]),
                ("d3", [("filler1", "concept")]),
                ("d4", [("filler2", "concept")]),
                ("d5", [("filler3", "concept")]),
                ("d6", [("filler4", "concept")]),
            ],
        )
        prof = await graph.entity_profile("SEEDX")  # fail-open → default floor 2
        names = {c.name for c in prof.cooccurring}
        assert "PERSISTENT" in names  # shared in d1 + d2 → survives
        assert "ONEOFF" not in names  # shared in d1 only → floored
    finally:
        await graph.close()


@pytest.mark.asyncio
async def test_blank_query_is_safe(tmp_path: Path) -> None:
    """A blank / whitespace-only query resolves to nothing, with no suggestions and no
    crash (the `if not key` short-circuit, before any scan)."""
    graph = await GraphStore.open(tmp_path)
    try:
        await _seed_spec(graph, [("d1", [("DNS", "concept")]), ("d2", [("DNS", "concept")])])
        prof = await graph.entity_profile("   ")
        assert prof.resolved is False
        assert prof.matched_names == []
        assert prof.suggestions == []
    finally:
        await graph.close()


@pytest.mark.asyncio
async def test_clear_mentions_replaces_not_appends(tmp_path: Path) -> None:
    """`clear_mentions` removes a doc's outgoing MENTIONS (so a re-enrich REPLACES its
    entities, not appends) while leaving the entity NODES and OTHER docs' edges intact —
    the load-bearing precondition for the OTTER NER backend switch."""
    graph = await GraphStore.open(tmp_path)
    try:
        await _seed_spec(
            graph,
            [
                ("doc_a", [("STP", "concept"), ("ARP", "concept")]),
                ("doc_b", [("STP", "concept")]),
            ],
        )
        assert (await graph.entity_profile("STP")).doc_count == 2  # doc_a + doc_b

        await graph.clear_mentions("doc_a")

        after = await graph.entity_profile("STP")
        assert after.resolved is True  # node survives (still mentioned by doc_b)
        assert {m.doc_id for m in after.mentions} == {"doc_b"}  # doc_a's STP edge cleared
        arp = await graph.entity_profile("ARP")
        assert "doc_a" not in {m.doc_id for m in arp.mentions}  # doc_a's ARP edge cleared too

        # Re-link doc_a with a DIFFERENT entity → replace semantics (the OTTER swap shape).
        eid = await graph.upsert_entity("BGP", "concept")
        await graph.link_mentions("doc_a", eid, 0.9)
        assert {m.doc_id for m in (await graph.entity_profile("BGP")).mentions} == {"doc_a"}
    finally:
        await graph.close()
