"""The bridge claim-extractor prompt (`extract_claims`). v2 (2026-06-02) FIXES the
under-coverage that left a multi-fact analysis grounding 0/1 — it flips the bias from
"few or zero is correct" to comprehensive-but-faithful (extract every attributed/
decomposed groundable fact) WITHOUT weakening the extractor-not-generator faithfulness
guard. Pinnable back to v1 via `MEMEX_PROMPTS__PIN__EXTRACT_CLAIMS=v1`."""

from __future__ import annotations

import pytest

from memex.prompts import render_messages
from memex.prompts.loader import load_prompt_spec


def _system_block() -> str:
    msgs = render_messages(
        "extract_claims",
        question="Q",
        analysis="A",
        evidence=[{"chunk_id": "d#a", "title": "T", "section": "S", "snippet": "x"}],
    )
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "d#a" in msgs[1]["content"]  # the evidence loop renders the chunk id
    return msgs[0]["content"]


def test_extract_claims_v2_is_the_active_version() -> None:
    load_prompt_spec.cache_clear()
    assert load_prompt_spec("extract_claims").version == "v2"


def test_v2_flips_the_bias_to_comprehensive_coverage() -> None:
    system = _system_block()
    # The fix: comprehensive coverage + attribution-first + decomposition.
    assert "PRIMARY TARGET" in system
    assert "EVERY discrete" in system
    assert "DECOMPOSE" in system
    # The v1 under-extraction bias is GONE.
    assert "FEW or ZERO" not in system


def test_v2_preserves_the_faithfulness_guard() -> None:
    system = _system_block()
    # Extractor-not-generator, verbatim id, no computed figures — the ADR-0016 guard,
    # untouched (over-extraction is contracted downstream; fabrication must be blocked here).
    assert "EXTRACTOR, not a generator" in system
    assert "EXPLICITLY asserts" in system
    assert "VERBATIM" in system
    assert "computed or aggregated figure" in system
    # Citation accuracy: cite the chunk whose CONTENT supports it, not one that only NAMES it
    # (mirrors the downstream name-only guard).
    assert "NOT merely one that NAMES" in system


def test_pin_kill_switch_reverts_to_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMEX_PROMPTS__PIN__EXTRACT_CLAIMS", "v1")
    load_prompt_spec.cache_clear()
    try:
        assert load_prompt_spec("extract_claims").version == "v1"
    finally:
        monkeypatch.delenv("MEMEX_PROMPTS__PIN__EXTRACT_CLAIMS", raising=False)
        load_prompt_spec.cache_clear()  # restore v2 for the rest of the suite
