"""Unit tests for `_rank_related_documents` — the IDF-specificity scoring core of the
"explore connections" feature (the on-mission successor to the retired passive
expand_graph). Pure function: rows in, ranked RelatedDocuments out, no graph.
"""

from __future__ import annotations

from memex.index.graph_store import _rank_related_documents

# rows: (neighbour_doc_id, neighbour_title, shared_entity, entity_kind, entity_doc_frequency)


def test_one_specific_entity_outranks_several_generic_ones() -> None:
    """A doc sharing ONE rare entity beats a doc sharing several mid-frequency ones —
    the whole point (vs the unranked neighbors() that let generic entities dominate)."""
    rows = [
        ("specific", "Specific Doc", "stateful inspection firewall", "concept", 2),  # rare
        ("broad", "Broad Doc", "network", "concept", 40),
        ("broad", "Broad Doc", "protocol", "concept", 40),
        ("broad", "Broad Doc", "system", "concept", 40),
    ]
    out = _rank_related_documents(rows, n_docs=100, limit=10, max_entities=8)
    assert [r.doc_id for r in out] == ["specific", "broad"]
    assert out[0].score > out[1].score


def test_generic_entity_above_df_fraction_is_excluded() -> None:
    """An entity mentioned by > 60% of the corpus is noise — excluded from scoring AND
    from the surfaced shared_entities. A doc connected ONLY by such entities disappears."""
    rows = [
        ("only_generic", "Generic", "the course instructor", "concept", 90),  # >0.6 → dropped
        ("real", "Real", "DNS spoofing", "concept", 3),
    ]
    out = _rank_related_documents(rows, n_docs=100, limit=10, max_entities=8)
    assert [r.doc_id for r in out] == ["real"]  # the generic-only doc is gone
    assert out[0].shared_entities == ["DNS spoofing"]


def test_proper_noun_type_is_down_weighted() -> None:
    """A 'person'/'place' connector (incidental proper noun — the instructor, a country)
    is heavily down-weighted vs an equally-rare 'concept', even at the same df."""
    rows = [
        ("concept_doc", "Concept", "stateful inspection", "concept", 3),
        ("person_doc", "Person", "Charles Tounou", "person", 3),  # same df, proper noun
    ]
    out = _rank_related_documents(rows, n_docs=100, limit=10, max_entities=8)
    assert [r.doc_id for r in out] == ["concept_doc", "person_doc"]
    # concept full weight (1.0) vs person 0.25 → ~4x the score at equal df
    # (each score independently rounded to 4dp, so compare the ratio with tolerance).
    assert abs(out[0].score / out[1].score - 4.0) < 0.01


def test_kind_weight_reorders_shared_entities() -> None:
    """Within a doc, a high-IDF proper-noun can rank BELOW a lower-IDF concept once the
    type weight applies — so the surfaced 'why' leads with the meaningful concept."""
    rows = [
        ("d", "D", "Vietnam", "place", 2),  # rare but place → 0.25 * ln(50)
        ("d", "D", "DNS spoofing", "concept", 5),  # less rare but concept → 1.0 * ln(20)
    ]
    out = _rank_related_documents(rows, n_docs=100, limit=10, max_entities=8)
    assert out[0].shared_entities[0] == "DNS spoofing"  # concept leads despite higher-df


def test_score_is_sum_of_weighted_idf() -> None:
    import math

    rows = [("d", "D", "a", "concept", 2), ("d", "D", "b", "org", 4)]  # 1.0 and 0.5 weights
    out = _rank_related_documents(rows, n_docs=100, limit=10, max_entities=8)
    expected = round(math.log(100 / 2) * 1.0 + math.log(100 / 4) * 0.5, 4)
    assert out[0].score == expected


def test_empty_and_singleton_corpus() -> None:
    assert _rank_related_documents([], n_docs=0, limit=10, max_entities=8) == []
    assert (
        _rank_related_documents([("x", "X", "e", "concept", 1)], n_docs=1, limit=10, max_entities=8)
        == []
    )


def test_limit_caps_results() -> None:
    rows = [(f"d{i}", f"D{i}", f"e{i}", "concept", 2) for i in range(20)]
    out = _rank_related_documents(rows, n_docs=100, limit=5, max_entities=8)
    assert len(out) == 5
