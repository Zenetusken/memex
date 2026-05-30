"""Unit tests for `_rank_bridges` — the ENTITY-grouped scoring core of the /graph
"Bridges" view (the redesign that retired the Cytoscape hairball). Pure function: the same
`(neighbour_doc_id, title, shared_entity, kind, df)` rows that feed `_rank_related_documents`,
inverted into entity-keyed bridges. Rows in, ranked DocumentBridges out, no graph.
"""

from __future__ import annotations

import math

from memex.index.graph_store import _rank_bridges

# rows: (neighbour_doc_id, neighbour_title, shared_entity, entity_kind, entity_doc_frequency)

_KW = {"limit_bridges": 24, "max_docs_per_bridge": 50, "max_via": 5}


def test_inverts_rows_into_entity_keyed_bridges() -> None:
    """A shared entity that connects two docs becomes ONE bridge with doc_count=2; an entity
    shared by one doc becomes a doc_count=1 bridge. The whole inversion in one assertion."""
    rows = [
        ("a", "Doc A", "IPSec", "concept", 4),
        ("b", "Doc B", "IPSec", "concept", 4),
        ("a", "Doc A", "GRE", "concept", 2),
    ]
    out = _rank_bridges(rows, n_docs=50, **_KW)
    by_name = {b.entity: b for b in out}
    assert set(by_name) == {"IPSec", "GRE"}
    assert by_name["IPSec"].doc_count == 2
    assert {d.doc_id for d in by_name["IPSec"].docs} == {"a", "b"}
    assert by_name["GRE"].doc_count == 1
    assert [d.doc_id for d in by_name["GRE"].docs] == ["a"]


def test_strength_is_specificity_times_log_damped_fanout() -> None:
    """strength = mean per-edge IDF×weight × ln(1 + doc_count). A rare entity (high IDF)
    shared by 2 docs outranks a less-rare entity shared by 1 — specificity leads, fan-out
    helps sub-linearly — and the bridges sort by strength desc."""
    rows = [
        # rare (df=2 → high IDF), connects 2 docs
        ("a", "A", "stateful inspection", "concept", 2),
        ("b", "B", "stateful inspection", "concept", 2),
        # less rare (df=10), connects only 1 doc
        ("c", "C", "routing", "concept", 10),
    ]
    out = _rank_bridges(rows, n_docs=100, **_KW)
    assert [b.entity for b in out] == ["stateful inspection", "routing"]
    assert out[0].strength > out[1].strength
    # top bridge: mean IDF = ln(100/2), fan-out 2 → × ln(1+2)
    assert math.isclose(out[0].strength, round(math.log(100 / 2) * math.log1p(2), 4), abs_tol=1e-4)


def test_log_damped_fanout_buries_near_generic_high_fanout_entity() -> None:
    """The whole point of the log damping: a near-generic entity in MANY docs (just under the
    generic-df cutoff) must NOT outrank a specific concept shared by a few. A linear × fan-out
    would invert this — and on a homogeneous corpus surface exactly the generic terms to bury."""
    # 'split horizon' (df=3) shared by 3 neighbours — specific, modest fan-out.
    rows = [(f"s{i}", f"S{i}", "split horizon", "concept", 3) for i in range(3)]
    # 'IP' in 55/100 docs (just under 0.6) shared by 50 neighbours — high fan-out, low IDF.
    rows += [(f"n{i}", f"N{i}", "IP", "concept", 55) for i in range(50)]
    out = _rank_bridges(rows, n_docs=100, **_KW)
    rank = {b.entity: i for i, b in enumerate(out)}
    assert rank["split horizon"] < rank["IP"]  # specific concept ranks ABOVE the generic one
    assert out[rank["IP"]].doc_count == 50  # ...despite 'IP' having far higher fan-out


def test_via_entities_exclude_the_bridge_entity_and_are_specificity_ordered() -> None:
    """Under a bridge, each doc shows its OTHER connecting entities (`via`), most-specific
    first, and NEVER re-lists the bridge's own entity."""
    rows = [
        ("a", "A", "IPSec", "concept", 6),  # the bridge we'll inspect
        ("a", "A", "IKE", "concept", 2),  # rarer → should lead the via list
        ("a", "A", "tunnel", "concept", 9),  # less rare → trails
        ("b", "B", "IPSec", "concept", 6),  # makes IPSec a 2-doc bridge
    ]
    out = _rank_bridges(rows, n_docs=100, **_KW)
    ipsec = next(b for b in out if b.entity == "IPSec")
    doc_a = next(d for d in ipsec.docs if d.doc_id == "a")
    assert "IPSec" not in doc_a.via_entities  # never re-lists the bridge entity
    assert doc_a.via_entities == ["IKE", "tunnel"]  # most-specific first


def test_generic_entity_above_df_fraction_is_excluded() -> None:
    """An entity mentioned by > 60% of the corpus is noise — it never becomes a bridge (same
    gate as the doc lens)."""
    rows = [
        ("a", "A", "the instructor", "concept", 90),  # >0.6 of 100 → dropped
        ("b", "B", "the instructor", "concept", 90),
        ("a", "A", "DNS spoofing", "concept", 3),
    ]
    out = _rank_bridges(rows, n_docs=100, **_KW)
    assert [b.entity for b in out] == ["DNS spoofing"]


def test_docs_within_a_bridge_order_by_overall_relatedness() -> None:
    """The docs under a bridge are ordered by each doc's OVERALL relatedness score (Σ over
    ALL its shared entities), not by this one edge — so the strongest neighbour leads."""
    rows = [
        # both share "VLAN"; doc 'rich' ALSO shares a rare entity → higher overall score
        ("rich", "Rich", "VLAN", "concept", 8),
        ("rich", "Rich", "private VLAN edge", "concept", 2),
        ("thin", "Thin", "VLAN", "concept", 8),
    ]
    out = _rank_bridges(rows, n_docs=100, **_KW)
    vlan = next(b for b in out if b.entity == "VLAN")
    assert [d.doc_id for d in vlan.docs] == ["rich", "thin"]
    assert vlan.docs[0].score > vlan.docs[1].score


def test_same_name_different_kind_merges_into_one_bridge() -> None:
    """Two entity nodes with the same NAME but different kinds read as ONE concept — they
    merge into a single bridge (strength accumulated across both), keeping the kind of the
    strongest-contributing link."""
    rows = [
        ("a", "A", "Cisco", "org", 4),  # kind weight 0.5
        ("b", "B", "Cisco", "concept", 4),  # kind weight 1.0 → the stronger link
    ]
    out = _rank_bridges(rows, n_docs=100, **_KW)
    cisco = [b for b in out if b.entity.lower() == "cisco"]
    assert len(cisco) == 1
    assert cisco[0].doc_count == 2
    assert cisco[0].kind == "concept"  # the strongest link's kind


def test_max_via_caps_secondary_tags() -> None:
    rows = [("a", "A", "BRIDGE", "concept", 5), ("b", "B", "BRIDGE", "concept", 5)]
    rows += [("a", "A", f"e{i}", "concept", 3) for i in range(8)]  # 8 other entities on doc a
    out = _rank_bridges(rows, n_docs=100, limit_bridges=24, max_docs_per_bridge=50, max_via=5)
    bridge = next(b for b in out if b.entity == "BRIDGE")
    doc_a = next(d for d in bridge.docs if d.doc_id == "a")
    assert len(doc_a.via_entities) == 5  # capped


def test_empty_and_single_doc_corpus_return_nothing() -> None:
    assert _rank_bridges([], n_docs=0, **_KW) == []
    assert _rank_bridges([("a", "A", "x", "concept", 1)], n_docs=1, **_KW) == []


def test_deterministic_ordering() -> None:
    """Same rows in any iteration order → identical bridge list (strength desc, entity name
    as the stable tiebreaker)."""
    rows = [
        ("a", "A", "alpha", "concept", 5),
        ("b", "B", "alpha", "concept", 5),
        ("a", "A", "beta", "concept", 5),
        ("c", "C", "beta", "concept", 5),
    ]
    first = _rank_bridges(rows, n_docs=100, **_KW)
    second = _rank_bridges(list(reversed(rows)), n_docs=100, **_KW)
    assert [(b.entity, b.strength, b.doc_count) for b in first] == [
        (b.entity, b.strength, b.doc_count) for b in second
    ]
