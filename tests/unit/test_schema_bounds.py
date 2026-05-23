"""Audit 2026-05-22 follow-up to P3.3 v6 — pin `max_length` bounds on
LLM-emit string schemas.

v6 surfaced one schema-validation crash on Qwen3-4B-AWQ: the
`SufficiencyAssessment.reason` field was unbounded, and on a
counterfactual query the model rambled for ~2000 chars in `reason`,
exceeding `max_tokens=640` mid-string. JSON validation failed; the
whole eval aborted. The fix bounded `reason` to `max_length=500`.

This same pathology can fire on any unbounded LLM-emit string field.
The follow-up audit identified five candidates:

1. `DraftAnswer.summary` — answer node, "one or two sentences" emit
2. `CitedClaim.claim` — answer node, per-claim text
3. `VerificationResult.ungrounded_reasons` items — verify node,
   per-claim explanation
4. `ExtractedEntity.name` — enrich entity extraction
5. `ExtractedEntity.span_text` — enrich entity extraction

These tests pin the bound on each field so a future refactor doesn't
silently drop the constraint. xgrammar's strict-JSON mode enforces
these `max_length` values at the grammar level, so the LLM literally
cannot emit a longer string — the bound is both a defence and a
performance optimisation (the model stops cold instead of running
into max_tokens).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from memex.agents.answering import (
    CitedClaim,
    DraftAnswer,
    SufficiencyAssessment,
)
from memex.enrich.entities import ExtractedEntity


def test_sufficiency_assessment_reason_rejects_over_max_length() -> None:
    """v6's original fix — pin the bound that fixed the eval crash."""
    SufficiencyAssessment(sufficient=True, reason="ok"[:500])
    with pytest.raises(ValidationError) as exc:
        SufficiencyAssessment(sufficient=False, reason="x" * 501)
    assert "at most 500" in str(exc.value).lower() or "max_length" in str(exc.value)


def test_cited_claim_claim_rejects_over_max_length() -> None:
    """Per-claim runaway: each claim is "one factual statement" — 300
    chars is plenty (Path C tightening fits worst-case schema output
    within max_tokens=1024). Bound prevents per-claim ramble
    exhausting max_tokens before the claims list closes."""
    CitedClaim(
        claim="x" * 300, source_chunk_id="abc", confidence="high"
    )
    with pytest.raises(ValidationError):
        CitedClaim(
            claim="x" * 301, source_chunk_id="abc", confidence="high"
        )


def test_cited_claim_source_chunk_id_rejects_over_max_length() -> None:
    """Path C: source_chunk_id is bounded so a hallucinated chunk_id
    (e.g. model dumping chunk text into the id field) can't run away.
    Real chunk_ids are ~40 chars (sha1[:10] + slug); 80 is headroom."""
    CitedClaim(
        claim="ok", source_chunk_id="x" * 80, confidence="high"
    )
    with pytest.raises(ValidationError):
        CitedClaim(
            claim="ok", source_chunk_id="x" * 81, confidence="high"
        )


def test_draft_answer_summary_rejects_over_max_length() -> None:
    """Answer-node summary is "one or two sentences" — 300 chars
    headroom (Path C tightening). Bound prevents runaway emission
    that previously could eat max_tokens before the claims list
    opened."""
    DraftAnswer(summary="x" * 300, claims=[])
    with pytest.raises(ValidationError):
        DraftAnswer(summary="x" * 301, claims=[])


def test_draft_answer_claims_list_bounded() -> None:
    """Top-k retrieval surfaces 5 chunks; the answer model legitimately
    emits ~5 claims (one per chunk). Cap at 8 (Path C tightening)
    keeps worst-case schema output under max_tokens=1024."""
    DraftAnswer(
        summary="ok",
        claims=[
            CitedClaim(claim=f"c{i}", source_chunk_id=f"id{i}", confidence="high")
            for i in range(8)
        ],
    )
    with pytest.raises(ValidationError):
        DraftAnswer(
            summary="ok",
            claims=[
                CitedClaim(claim=f"c{i}", source_chunk_id=f"id{i}", confidence="high")
                for i in range(9)
            ],
        )


def test_extracted_entity_name_bounds() -> None:
    """Entity name legitimately runs 3-50 chars; 120 covers long
    org names or multi-word concepts. min_length=1 (existing
    constraint) + max_length=120 (new bound)."""
    ExtractedEntity(name="x", kind="person", confidence="high")
    ExtractedEntity(name="x" * 120, kind="person", confidence="high")
    with pytest.raises(ValidationError):
        ExtractedEntity(name="", kind="person", confidence="high")
    with pytest.raises(ValidationError):
        ExtractedEntity(name="x" * 121, kind="person", confidence="high")


def test_extracted_entity_span_text_bounds() -> None:
    """Quoted phrase from passage — 200 chars covers a long
    sentence fragment without enabling whole-paragraph emission."""
    ExtractedEntity(
        name="Alice", kind="person", confidence="high", span_text="x" * 200
    )
    with pytest.raises(ValidationError):
        ExtractedEntity(
            name="Alice",
            kind="person",
            confidence="high",
            span_text="x" * 201,
        )


def test_extracted_entity_span_text_default_empty_is_ok() -> None:
    """The default empty `span_text` is valid (some entities don't
    have an addressable span in the passage)."""
    e = ExtractedEntity(name="Alice", kind="person", confidence="high")
    assert e.span_text == ""
