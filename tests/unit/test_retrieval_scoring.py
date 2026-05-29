"""Unit tests for retrieval-eval scorers in `eval/scoring.py`.

`gold_chunk_recall` is the deterministic recall@k scorer used by the ad-hoc
before/after retrieval harnesses (e.g. the dense-vs-bm25 arm-separation probe that
proved the BM25 lexical arm adds no recall over the dense embedder — see the
`fts-bm25-nl-scope` memory). Kept as reusable infra independent of that result.
"""

from __future__ import annotations

from memex.eval.scoring import gold_chunk_recall


def test_gold_chunk_recall() -> None:
    assert gold_chunk_recall(["a", "b", "c"], ["b"], k=2) == 1.0  # in top-2
    assert gold_chunk_recall(["a", "b", "c"], ["c"], k=2) == 0.0  # below top-2
    assert gold_chunk_recall(["x", "y"], ["x", "z"], k=2) == 0.5  # one of two
    assert gold_chunk_recall(["a"], [], k=5) == 1.0  # no gold ⇒ vacuous
