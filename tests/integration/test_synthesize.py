"""Tests for the full-document synthesis path (`agents/synthesize.py`, ADR-0007).

Fakes `complete_structured` so no model is needed — asserts the doc assembly,
the max_tokens override, scope surfacing, truncation, and the empty refusal.
"""

from __future__ import annotations

from typing import Any

import pytest

from memex.agents.synthesize import SourceDoc, Synthesis, synthesize_documents


def _fake_complete(captured: dict[str, Any]):  # type: ignore[no-untyped-def]  # returns a stand-in coroutine fn
    async def _complete(prompt: str, schema: type[Any], **kw: Any) -> tuple[Any, int]:
        captured["prompt"] = prompt
        captured["schema"] = schema
        captured["max_tokens"] = kw.get("max_tokens")
        return Synthesis(summary="A comprehensive summary of the document."), 1234

    return _complete


@pytest.mark.asyncio
async def test_synthesize_assembles_docs_and_returns_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr("memex.agents.synthesize.complete_structured", _fake_complete(captured))

    sources = [
        SourceDoc(doc_id="d1", title="Doc One", text="The body of document one."),
        SourceDoc(doc_id="d2", title="Doc Two", text="The body of document two."),
    ]
    r = await synthesize_documents(
        sources, instruction="focus on the controls", max_output_tokens=2048
    )

    assert r.answered is True
    assert r.summary == "A comprehensive summary of the document."
    assert r.artifact_scope_doc_ids == ["d1", "d2"]
    assert r.tokens_used == 1234
    assert r.correlation_id  # populated
    # The whole document text + titles + instruction reached the prompt.
    assert "The body of document one." in captured["prompt"]
    assert "The body of document two." in captured["prompt"]
    assert "Doc One" in captured["prompt"]
    assert "focus on the controls" in captured["prompt"]
    # The output budget is honored (raised well above the RAG default).
    assert captured["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_synthesize_truncates_to_input_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr("memex.agents.synthesize.complete_structured", _fake_complete(captured))

    big = SourceDoc(doc_id="d", title="Big", text="x" * 100_000)
    await synthesize_documents([big], max_input_chars=5_000)

    assert "truncated to fit the context window" in captured["prompt"]
    # The body is bounded near the budget (allow slack for the title + marker).
    assert captured["prompt"].count("x") <= 5_200


@pytest.mark.asyncio
async def test_synthesize_empty_refuses_cleanly() -> None:
    r = await synthesize_documents([])
    assert r.answered is False
    assert r.refusal_reason
    assert r.summary is None


@pytest.mark.asyncio
async def test_synthesize_degrades_gracefully_on_output_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truncated/invalid output (the single-shot long-summary failure mode)
    becomes a clean refusal, not a crash."""
    from memex.core.errors import ModelCallError

    async def _boom(prompt: str, schema: type[Any], **kw: Any) -> tuple[Any, int]:
        raise ModelCallError("Model output did not match the requested schema.", context={})

    monkeypatch.setattr("memex.agents.synthesize.complete_structured", _boom)
    r = await synthesize_documents(
        [SourceDoc(doc_id="d", title="T", text="body")], max_output_tokens=4096
    )
    assert r.answered is False
    assert r.refusal_reason is not None
    assert "single-shot" in r.refusal_reason
    assert r.summary is None
