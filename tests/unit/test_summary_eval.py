"""Unit tests for the summary-eval scorers (ADR-0008).

`mention_recall` is the soft coverage signal; `absent_assertion_violations` is the
no-leak HARD gate (any returned term is a leaked assertion the doc doesn't support).
Both match normalized substrings, so they're case/whitespace-insensitive.
"""

from __future__ import annotations

from memex.eval.scoring import absent_assertion_violations, mention_recall


def test_mention_recall_fraction_present() -> None:
    assert mention_recall("The cat sat on the mat", ["cat", "dog"]) == 0.5


def test_mention_recall_empty_requirement_is_one() -> None:
    assert mention_recall("anything", []) == 1.0


def test_mention_recall_case_and_whitespace_insensitive() -> None:
    assert mention_recall("Multi-Stage   CONTRASTIVE learning", ["multi-stage contrastive"]) == 1.0


def test_absent_violations_reports_only_present_terms() -> None:
    # "dog" is correctly absent; "cat" leaked → returned as a violation.
    assert absent_assertion_violations("the cat sat", ["dog", "cat"]) == ["cat"]


def test_absent_violations_empty_when_all_absent() -> None:
    assert absent_assertion_violations("the cat sat", ["dog", "bird"]) == []
