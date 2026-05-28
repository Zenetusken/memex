"""Unit tests for `_rank_related_documents` — the IDF-specificity scoring core of the
"explore connections" feature (the on-mission successor to the retired passive
expand_graph). Pure function: rows in, ranked RelatedDocuments out, no graph.
"""

from __future__ import annotations

from memex.index.graph_store import _rank_related_documents

# rows: (neighbour_doc_id, neighbour_title, shared_entity, entity_doc_frequency)


def test_one_specific_entity_outranks_several_generic_ones() -> None:
    """A doc sharing ONE rare entity beats a doc sharing several mid-frequency ones —
    the whole point (vs the unranked neighbors() that let generic entities dominate)."""
    rows = [
        ("specific", "Specific Doc", "stateful inspection firewall", 2),  # rare → high IDF
        ("broad", "Broad Doc", "network", 40),
        ("broad", "Broad Doc", "protocol", 40),
        ("broad", "Broad Doc", "system", 40),
    ]
    out = _rank_related_documents(rows, n_docs=100, limit=10, max_entities=8)
    assert [r.doc_id for r in out] == ["specific", "broad"]
    assert out[0].score > out[1].score


def test_generic_entity_above_df_fraction_is_excluded() -> None:
    """An entity mentioned by > 60% of the corpus is noise — excluded from scoring AND
    from the surfaced shared_entities. A doc connected ONLY by such entities disappears."""
    rows = [
        ("only_generic", "Generic", "the course instructor", 90),  # 90/100 > 0.6 → dropped
        ("real", "Real", "DNS spoofing", 3),
    ]
    out = _rank_related_documents(rows, n_docs=100, limit=10, max_entities=8)
    assert [r.doc_id for r in out] == ["real"]  # the generic-only doc is gone
    assert out[0].shared_entities == ["DNS spoofing"]


def test_shared_entities_most_specific_first_and_capped() -> None:
    rows = [
        ("d", "D", "common", 50),  # lower IDF
        ("d", "D", "rare", 2),  # higher IDF → should sort first
        ("d", "D", "mid", 10),
    ]
    out = _rank_related_documents(rows, n_docs=100, limit=10, max_entities=2)
    assert out[0].shared_entities == ["rare", "mid"]  # most-specific first, capped at 2


def test_score_is_sum_of_idf() -> None:
    import math

    rows = [("d", "D", "a", 2), ("d", "D", "b", 4)]
    out = _rank_related_documents(rows, n_docs=100, limit=10, max_entities=8)
    expected = round(math.log(100 / 2) + math.log(100 / 4), 4)
    assert out[0].score == expected


def test_empty_and_singleton_corpus() -> None:
    assert _rank_related_documents([], n_docs=0, limit=10, max_entities=8) == []
    assert _rank_related_documents([("x", "X", "e", 1)], n_docs=1, limit=10, max_entities=8) == []


def test_limit_caps_results() -> None:
    rows = [(f"d{i}", f"D{i}", f"e{i}", 2) for i in range(20)]
    out = _rank_related_documents(rows, n_docs=100, limit=5, max_entities=8)
    assert len(out) == 5
