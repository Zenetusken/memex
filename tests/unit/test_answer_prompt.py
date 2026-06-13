"""The answer (drafting) node prompt. The answer node drafts {summary, claims[]} from the
reranked chunks; the downstream `verify` gate + backstops are the real anti-hallucination HARD
gate (refusal_cf=1.0). Version history relevant to false refusals (audit 12, 2026-06-08):

- v3: a ~3:1 refuse-biased prompt (3 "return empty" examples + "refusal is the correct
  outcome" framing) → on a 4B it emitted EMPTY/HEDGING drafts even when the answer was plainly
  in the chunks (linux-11 heading, forms-w9-07 form, annual-report figure).
- v4 (REJECTED, not shipped): a broad "default to ANSWER" rebalance recovered +10 false
  refusals but DETERMINISTICALLY breached the HARD gate — french-course-07 fabricated "the
  course says to configure a Cisco PIX firewall by…" from GENERIC firewall content (a
  named-entity substitution verify grounds at the claim level + relevance missed). 3/3 answered.
- **v5 (shipped): the SUBJECT-PRESENCE reframe.** Lead with a subject test — is the specific
  subject the query names (product/model/vendor/dataset/value-type) actually present in the
  chunks? If only a GENERIC category or a DIFFERENT/neighbouring item is present → empty (the
  Cisco-PIX/FP4/FP128/Hopper-Blackwell traps). If the subject IS present, the answer counts as
  stated in ANY form — heading, table/figure cell, brief clause, or different wording → draft
  (the heading/form/figure recoveries). Validated: refusal_cf=1.0 across 70 counterfactuals ×
  N=3 (0 breaches, incl. french-course-07 back to 0/3) + recovers the heading/form/figure cases.

Pins the subject test, the named-entity + numeric traps, the recovery rules, and truncate(1800).
Kill-switch: MEMEX_PROMPTS__PIN__ANSWER=v3 restores the pre-rebalance prompt.
"""

from __future__ import annotations

import re

import pytest

from memex.prompts.loader import load_prompt_spec


def _body() -> str:
    load_prompt_spec.cache_clear()
    return load_prompt_spec("answer").template


def _norm(text: str) -> str:
    """Collapse whitespace so multi-word phrase checks are immune to line wrapping."""
    return re.sub(r"\s+", " ", text)


def test_v5_is_the_active_version() -> None:
    load_prompt_spec.cache_clear()
    assert load_prompt_spec("answer").version == "v5"


def test_leads_with_the_subject_presence_test() -> None:
    """The load-bearing reframe: decide on whether the query's SPECIFIC subject is present,
    not a broad 'default to ANSWER' (which caused the named-entity substitution breach)."""
    b = _body()
    assert "subject test" in b
    # the over-broad v4 lead must NOT be present (it enabled the conflation breach)
    assert "default is to ANSWER" not in b


def test_keeps_the_named_entity_substitution_trap() -> None:
    """The french-course-07 breach must stay blocked: a query naming a specific entity absent
    from the chunks → empty, even if the generic category is present (Cisco PIX example)."""
    b = _norm(_body())
    assert "Cisco PIX" in b
    assert "generic" in b.lower()
    assert "NEVER attribute generic or neighbouring content" in b


def test_keeps_the_numeric_substitution_traps() -> None:
    b = _body()
    assert "FP128" in b and "FP4" in b and "Hopper-Blackwell" in b
    assert "No-substitute" in b
    assert "Identity, not wording" in b


def test_keeps_the_recovery_rules() -> None:
    """The false-refusal recoveries: a heading that names the answer is literal; a value stated
    briefly still counts; headings/tables/figures are literal."""
    b = _norm(_body())
    assert "Hard Links and Symbolic Links" in b  # the heading-stated recovery
    assert "20 to 30 employee records" in b  # the brief-prose recovery
    assert "heading" in b.lower()


def test_truncate_aligned_at_1800() -> None:
    b = _body()
    assert "truncate(1800)" in b


def test_kill_switch_pins_v3(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reverting to v3 restores the pre-rebalance answer prompt (the documented kill-switch)."""
    monkeypatch.setenv("MEMEX_PROMPTS__PIN__ANSWER", "v3")
    load_prompt_spec.cache_clear()
    spec = load_prompt_spec("answer")
    assert spec.version == "v3"
    assert "subject test" not in spec.template
    load_prompt_spec.cache_clear()
