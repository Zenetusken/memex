"""Unit tests for `_rank_co_occurring` — the co-occurring-entity scoring core of
entity-centric retrieval (ADR-0011). Pure function: rows in, ranked CoOccurringEntities
out, no graph. Mirrors test_related_documents.py.
"""

from __future__ import annotations

import math

from memex.index.graph_store import _rank_co_occurring

# rows: (co_name, co_kind, shared_docs, co_doc_frequency)


def test_specific_co_entity_outranks_generic_at_equal_shared_docs() -> None:
    """At equal shared-doc count, a RARE co-entity (high IDF) beats a near-universal one."""
    rows = [
        ("DNS spoofing", "concept", 3, 2),  # rare → high IDF
        ("network", "concept", 3, 40),  # common → low IDF
    ]
    out = _rank_co_occurring(rows, n_docs=100, limit=10)
    assert [c.name for c in out] == ["DNS spoofing", "network"]
    assert out[0].score > out[1].score


def test_shared_docs_multiplier_rewards_pervasive_co_entity() -> None:
    """At equal df, a co-entity appearing in MORE of the seed's docs ranks higher —
    the shared_docs term (the neighbourhood-strength signal)."""
    rows = [
        ("ARP", "concept", 5, 4),
        ("telnet", "concept", 1, 4),  # same df, fewer shared docs
    ]
    out = _rank_co_occurring(rows, n_docs=100, limit=10)
    assert [c.name for c in out] == ["ARP", "telnet"]
    # 5× shared docs → ~5× score at equal df (ratio with tolerance — each score is 4dp-rounded).
    assert abs(out[0].score / out[1].score - 5.0) < 0.01


def test_generic_co_entity_above_df_fraction_excluded() -> None:
    rows = [
        ("the instructor", "concept", 4, 70),  # 70/100 > 0.6 → excluded
        ("stateful inspection", "concept", 2, 3),
    ]
    out = _rank_co_occurring(rows, n_docs=100, limit=10)
    assert [c.name for c in out] == ["stateful inspection"]


def test_proper_noun_type_down_weighted() -> None:
    """A 'person'/'place' co-entity is quartered vs an equally-rare, equally-shared
    'concept' (incidental proper nouns rarely define the neighbourhood)."""
    rows = [
        ("stateful inspection", "concept", 3, 3),
        ("Charles Tounou", "person", 3, 3),  # same shared_docs + df, proper noun
    ]
    out = _rank_co_occurring(rows, n_docs=100, limit=10)
    assert [c.name for c in out] == ["stateful inspection", "Charles Tounou"]
    assert abs(out[0].score / out[1].score - 4.0) < 0.01  # concept 1.0 vs person 0.25


def test_score_formula() -> None:
    rows = [("a", "concept", 2, 4)]  # shared_docs=2, kind_weight=1.0
    out = _rank_co_occurring(rows, n_docs=100, limit=10)
    assert out[0].score == round(2 * math.log(100 / 4) * 1.0, 4)
    assert out[0].shared_docs == 2


def test_empty_singleton_and_limit() -> None:
    assert _rank_co_occurring([], n_docs=0, limit=10) == []
    assert _rank_co_occurring([("x", "concept", 1, 1)], n_docs=1, limit=10) == []
    rows = [(f"e{i}", "concept", 2, 2) for i in range(20)]
    assert len(_rank_co_occurring(rows, n_docs=100, limit=5)) == 5


def test_zero_shared_docs_skipped() -> None:
    assert _rank_co_occurring([("x", "concept", 0, 2)], n_docs=100, limit=10) == []
