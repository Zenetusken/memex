"""The assess_relevance (responsiveness) gate prompt. It runs LAST, only on an already-grounded
answer, and judges whether the answer addresses the asked TOPIC (the conflation guard).

**v2 (2026-06-08, audit 12) fixes a WORLD-KNOWLEDGE-INJECTION over-refusal.** v1 refused grounded
answers about NVIDIA's *fiscal 2026* figures with reasons like "fiscal 2026 does not exist yet /
has not yet occurred / making the claim factually impossible" — the 4B applied its training-cutoff
belief that 2026 is the future to OVERRIDE a grounded answer from the user's own document. That is
a grounding violation (the corpus is authoritative, not the model's world knowledge) and would
break ANY future-dated corpus. It refused annual-report-01/02/03/05/13 and caused the answer/v5
annual-report-02 regression. v2 = v1 + an explicit rule: the cited sources are AUTHORITATIVE; do
NOT apply outside/world knowledge about dates, what "exists yet," or what is "factually possible";
judge ONLY topic match. Validated: annual-report figure questions recover 3/3 (incl. the
regression); refusal_cf stays 1.0 (relevance only acts on grounded answers — HARD-gate-safe by
construction). Kill-switch: MEMEX_PROMPTS__PIN__ASSESS_RELEVANCE=v1.
"""

from __future__ import annotations

import re

import pytest

from memex.prompts import render_prompt
from memex.prompts.loader import load_prompt_spec


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def test_v2_is_the_active_version() -> None:
    load_prompt_spec.cache_clear()
    assert load_prompt_spec("assess_relevance").version == "v2"


def test_v2_forbids_world_knowledge_override() -> None:
    """The load-bearing fix: the gate must treat the cited sources as authoritative and must NOT
    reject a grounded answer because it believes the date/period/value isn't real."""
    body = _norm(load_prompt_spec("assess_relevance").template)
    load_prompt_spec.cache_clear()
    assert "AUTHORITATIVE" in body
    assert "Do NOT apply outside or world knowledge" in body
    assert "fiscal 2026" in body  # the exact future-date case it must now accept
    # it still judges topic, not facts
    assert "topic match, NOT fact-checking" in body or "topic match" in body


def test_v2_keeps_the_default_responsive_conflation_guard() -> None:
    """v2 keeps v1's narrow conflation guard: default responsive, only a CLEAR topic mismatch."""
    body = _norm(load_prompt_spec("assess_relevance").template)
    load_prompt_spec.cache_clear()
    assert "Default to responsive" in body
    assert "clear TOPIC mismatch" in body or "CLEAR question/answer topic mismatch" in body


def test_v2_target_model_is_the_live_4b() -> None:
    load_prompt_spec.cache_clear()
    assert load_prompt_spec("assess_relevance").target_model == "cyankiwi/Qwen3.5-4B-AWQ-4bit"


def test_kill_switch_pins_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMEX_PROMPTS__PIN__ASSESS_RELEVANCE", "v1")
    load_prompt_spec.cache_clear()
    spec = load_prompt_spec("assess_relevance")
    assert spec.version == "v1"
    assert "Do NOT apply outside or world knowledge" not in spec.template
    load_prompt_spec.cache_clear()


def test_renders_query_summary_and_claims() -> None:
    load_prompt_spec.cache_clear()
    out = render_prompt(
        "assess_relevance",
        query="What was the revenue?",
        summary="Revenue was $216 billion.",
        claims=["Revenue grew 65% to $216B."],
    )
    assert "What was the revenue?" in out
    assert "Revenue was $216 billion." in out
    assert "Revenue grew 65% to $216B." in out
