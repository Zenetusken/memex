"""The assess (sufficiency) gate prompt. v2 (2026-06-06) FIXES the over-refusal where
the 4B gate ACKNOWLEDGED the answer was present then refused for "lack of citations as
requested" (handwritten-04: 5/5 refuse end-to-end). v2 = v1 + a CITATION FLOOR (a chunk
that states the asked fact is sufficient even if brief or uncited), with NO other change.

Validated multi-run vs v1: +2 ANS, 0 regressions, refusal_cf=1.0 across two full 12-corpus
passes, 0 borderline flips. The earlier candidate C — which ALSO added a "must EXPLICITLY
STATE / topic overlap is not an answer" tightening — was REJECTED (net -3: it over-refused
4 legitimate answerables). These tests pin the citation-floor and guard against C creeping
back. Pinnable to v1 via MEMEX_PROMPTS__PIN__ASSESS_SUFFICIENCY=v1.
"""

from __future__ import annotations

import pytest

from memex.prompts import render_prompt
from memex.prompts.loader import load_prompt_spec


def test_v2_is_the_active_version() -> None:
    load_prompt_spec.cache_clear()
    assert load_prompt_spec("assess_sufficiency").version == "v2"


def test_v2_has_the_citation_floor() -> None:
    """The load-bearing fix: a chunk that states the fact is sufficient even if uncited."""
    load_prompt_spec.cache_clear()
    body = load_prompt_spec("assess_sufficiency").template
    assert "states the asked fact is sufficient" in body
    assert "not for citations or" in body  # "...the user asked a question, not for citations or polish"
    assert "Do NOT set `sufficient: false` merely because the chunks lack citations" in body
    # the v1 citation-DEMANDING phrasing ("answer ... with citations") must be GONE
    assert "with citations" not in body


def test_v2_does_not_reintroduce_candidate_c_overtightening() -> None:
    """Candidate C's reframing cost 4 answerables (net -3) and was rejected for the
    citation-floor-only D. Guard against it being re-added under the v2 name."""
    load_prompt_spec.cache_clear()
    body = load_prompt_spec("assess_sufficiency").template
    assert "Topic overlap is not an answer" not in body
    assert "EXPLICITLY STATES" not in body
    assert "wrong instance, period, or entity" not in body


def test_v2_target_model_is_the_live_4b() -> None:
    """Frontmatter sweep: the prompt runs on the live orchestrator, not the pre-swap 8B."""
    load_prompt_spec.cache_clear()
    assert load_prompt_spec("assess_sufficiency").target_model == "cyankiwi/Qwen3.5-4B-AWQ-4bit"


def test_kill_switch_pins_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMEX_PROMPTS__PIN__ASSESS_SUFFICIENCY", "v1")
    load_prompt_spec.cache_clear()
    spec = load_prompt_spec("assess_sufficiency")
    assert spec.version == "v1"
    assert "with citations" in spec.template  # v1 is the citation-demanding original
    load_prompt_spec.cache_clear()  # don't leak the pin into other tests


def test_v2_renders_query_and_chunk_loop() -> None:
    load_prompt_spec.cache_clear()
    chunk = {"chunk_id": "doc1#abc", "document_title": "Net Notes", "page": 7,
             "text": "Fa0/21 is configured as a trunk."}
    out = render_prompt("assess_sufficiency", query="what port is the trunk?", chunks=[chunk])
    assert "what port is the trunk?" in out
    # the chunk loop renders id / title / page / text
    assert "doc1#abc" in out and "Net Notes" in out and "p.7" in out
    assert "Fa0/21 is configured as a trunk." in out
