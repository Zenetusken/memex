"""The assess (sufficiency) gate prompt — its job is a PRE-FILTER, not the safety gate
(verify/grounding is). Version history:

- v2 (2026-06-06): + a CITATION FLOOR (a chunk that states the fact is sufficient even if
  uncited), fixing the "acknowledged the answer then refused for lack of citations" over-refusal.
- v3 (2026-06-08): + truncate(1200)→truncate(1800), the TRUNCATION-MISMATCH fix (a fact at char
  1200-1800 of a long chunk was invisible to the gate but visible to answer/verify).
- **v4 (2026-06-08): the LIGHT-PRE-FILTER reframe.** Audit 12 (`docs/audits/12-false-refusals.md`)
  found the gate over-refusing when the answer was plainly in the retrieved chunks (a false
  refusal is as unreliable as a hallucination). v4 reframes the gate's ROLE: strongly default to
  `sufficient: true`; refuse ONLY when the chunks are essentially OFF-TOPIC or empty; explicitly
  defer "is the answer fully/precisely supported" to the downstream grounding step. **HARD-gate-
  safe by construction:** assess is a pre-filter; loosening it CANNOT add a hallucination (a
  counterfactual that slips through still hits the answer node's empty-draft rule + verify), and
  it CANNOT regress a correct-answer (v4 passes a superset of what v3 passed → previously-answered
  queries reach drafting identically). refusal_cf validated multi-run.

Pins: active version, the v4 pre-filter framing, the candidate-C absence, the truncate alignment.
Kill-switches: MEMEX_PROMPTS__PIN__ASSESS_SUFFICIENCY=v3 (truncate-only), =v2 (1200), =v1 (orig).
"""

from __future__ import annotations

import pytest

from memex.prompts import render_prompt
from memex.prompts.loader import load_prompt_spec


def test_v4_is_the_active_version() -> None:
    load_prompt_spec.cache_clear()
    assert load_prompt_spec("assess_sufficiency").version == "v4"


def test_v4_is_a_light_pre_filter() -> None:
    """v4 reframes the gate as a pre-filter biased toward letting queries through — refuse only
    on OFF-TOPIC/empty retrieval; the grounding step (not assess) judges claim support."""
    load_prompt_spec.cache_clear()
    body = load_prompt_spec("assess_sufficiency").template
    assert "PRE-FILTER" in body
    assert "DEFAULT to `sufficient: true`" in body
    assert "OFF-TOPIC" in body
    # it explicitly hands the real judgment to the downstream grounding/verify step
    assert "VERIFIES" in body or "grounding step" in body


def test_v4_truncate_aligned_with_answer_node() -> None:
    """The gate must render chunks at the SAME budget as the answer + verify nodes (1800) — else a
    fact past the gate's smaller view is invisible to the pre-filter. Pins no drift (the v3 fix)."""
    load_prompt_spec.cache_clear()
    assess = load_prompt_spec("assess_sufficiency").template
    answer = load_prompt_spec("answer").template
    verify = load_prompt_spec("verify_grounding").template
    assert "truncate(1800)" in assess
    assert "truncate(1200)" not in assess
    assert "truncate(1800)" in answer and "truncate(1800)" in verify


def test_v4_does_not_reintroduce_candidate_c_overtightening() -> None:
    """The rejected candidate-C tightening (cost 4 answerables) must never reappear — v4 goes the
    OPPOSITE direction (looser), so guard the C phrases stay absent."""
    load_prompt_spec.cache_clear()
    body = load_prompt_spec("assess_sufficiency").template
    assert "Topic overlap is not an answer" not in body
    assert "EXPLICITLY STATES" not in body
    assert "wrong instance, period, or entity" not in body


def test_target_model_is_the_live_4b() -> None:
    load_prompt_spec.cache_clear()
    assert load_prompt_spec("assess_sufficiency").target_model == "cyankiwi/Qwen3.5-4B-AWQ-4bit"


def test_kill_switch_pins_v3_truncate_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reverting to v3 restores the truncate-only fix without the light-pre-filter reframe."""
    monkeypatch.setenv("MEMEX_PROMPTS__PIN__ASSESS_SUFFICIENCY", "v3")
    load_prompt_spec.cache_clear()
    spec = load_prompt_spec("assess_sufficiency")
    assert spec.version == "v3"
    assert "PRE-FILTER" not in spec.template and "truncate(1800)" in spec.template
    load_prompt_spec.cache_clear()


def test_kill_switch_pins_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMEX_PROMPTS__PIN__ASSESS_SUFFICIENCY", "v1")
    load_prompt_spec.cache_clear()
    spec = load_prompt_spec("assess_sufficiency")
    assert spec.version == "v1"
    assert "with citations" in spec.template  # v1 is the citation-demanding original
    load_prompt_spec.cache_clear()


def test_renders_query_and_chunk_loop() -> None:
    load_prompt_spec.cache_clear()
    chunk = {"chunk_id": "doc1#abc", "document_title": "Net Notes", "page": 7,
             "text": "Fa0/21 is configured as a trunk."}
    out = render_prompt("assess_sufficiency", query="what port is the trunk?", chunks=[chunk])
    assert "what port is the trunk?" in out
    assert "doc1#abc" in out and "Net Notes" in out and "p.7" in out
    assert "Fa0/21 is configured as a trunk." in out
