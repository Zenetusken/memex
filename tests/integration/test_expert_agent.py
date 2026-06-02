"""Ungrounded expert mode (`agents/expert.py`, Surface B / ADR-0013).

No GPU: retrieval (`hybrid_search` / `hybrid_search_in_docs`), the reranker, and the
free-text `complete_reasoning` call are all faked. These pin the CONTRACT INVERSION —
the surface reasons over evidence WITHOUT a grounding gate, always carries the provenance
caveat, honours an explicit scope, and defaults to a clean (non-thinking) decode.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from memex.agents.expert import EXPERT_PROVENANCE_NOTE, expert_answer
from memex.core.config import MemexSettings, set_settings
from memex.core.types import Chunk
from memex.models.client import split_think


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemexSettings]:
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    s = MemexSettings()
    set_settings(s)
    yield s
    set_settings(None)


def _chunk(cid: str, title: str, text: str, section: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=cid,
        document_id=cid.split("#")[0],
        document_title=title,
        text=text,
        heading_path=[section] if section else [],
    )


class _FakeReason:
    """Stand-in for `complete_reasoning` — records the kwargs it was called with."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, prompt: object, **kw: Any) -> tuple[str, int]:
        self.calls.append({"prompt": prompt, **kw})
        return self.text, 123


def _patch_retrieval(
    monkeypatch: pytest.MonkeyPatch, chunks: list[Chunk], *, scope_sink: list[Any] | None = None
) -> None:
    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return chunks

    async def _hybrid_in_docs(query: str, doc_ids: list[str], k: int = 50) -> list[Chunk]:
        if scope_sink is not None:
            scope_sink.append(list(doc_ids))
        return chunks

    async def _rerank(query: str, candidates: list[Chunk], top_k: int = 10) -> list[Chunk]:
        return candidates[:top_k]

    monkeypatch.setattr("memex.agents.expert.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.expert.hybrid_search_in_docs", _hybrid_in_docs)
    monkeypatch.setattr("memex.agents.expert.cross_encoder_rerank", _rerank)


@pytest.mark.asyncio
async def test_expert_answer_reasons_over_evidence(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = [
        _chunk("d1#a", "STP Guide", "STP uses timers.", "Convergence"),
        _chunk("d2#b", "RSTP Guide", "RSTP is faster.", "RSTP"),
    ]
    _patch_retrieval(monkeypatch, chunks)
    reason = _FakeReason("RSTP converges faster because of its proposal/agreement handshake.")
    monkeypatch.setattr("memex.agents.expert.complete_reasoning", reason)

    phases: list[str] = []
    ans = await expert_answer("Compare STP and RSTP.", on_phase=phases.append)

    assert ans.answer == "RSTP converges faster because of its proposal/agreement handshake."
    assert ans.reasoning is None  # untagged inline reasoning → no separable trace
    assert [e.title for e in ans.evidence] == ["STP Guide", "RSTP Guide"]
    assert ans.evidence[0].section == "Convergence"
    assert ans.provenance_note == EXPERT_PROVENANCE_NOTE
    assert ans.model == settings.models.orchestrator  # reasoner unset → the live daemon
    assert ans.tokens == 123
    assert phases == ["Retrieving evidence", "Reasoning"]


@pytest.mark.asyncio
async def test_expert_defaults_to_clean_decode(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_retrieval(monkeypatch, [_chunk("d1#a", "Doc", "text")])
    reason = _FakeReason("answer")
    monkeypatch.setattr("memex.agents.expert.complete_reasoning", reason)

    await expert_answer("Why?")
    # v1 default: thinking mode OFF (verified verbose/unseparable on this checkpoint).
    assert reason.calls[0]["enable_thinking"] is False


@pytest.mark.asyncio
async def test_expert_honours_explicit_scope(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope_seen: list[Any] = []
    _patch_retrieval(monkeypatch, [_chunk("d1#a", "Doc", "text")], scope_sink=scope_seen)
    monkeypatch.setattr("memex.agents.expert.complete_reasoning", _FakeReason("scoped answer"))

    ans = await expert_answer("Q?", scope_doc_ids=["d1", " ", "d1", "d2"])
    # Blank-stripped + deduped, routed through the in-docs retrieval.
    assert scope_seen == [["d1", "d2"]]
    assert ans.scope_doc_ids == ["d1", "d2"]


@pytest.mark.asyncio
async def test_expert_answers_with_no_evidence(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_retrieval(monkeypatch, [])  # empty vault retrieval
    reason = _FakeReason("From general expertise: ...")
    monkeypatch.setattr("memex.agents.expert.complete_reasoning", reason)

    ans = await expert_answer("An out-of-corpus analytical question.")
    assert ans.evidence == []
    assert ans.answer == "From general expertise: ..."
    # Still reasoned (the contract: it goes beyond the vault), still caveated.
    assert len(reason.calls) == 1
    assert ans.provenance_note == EXPERT_PROVENANCE_NOTE


@pytest.mark.asyncio
async def test_expert_splits_a_tagged_trace_when_present(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_retrieval(monkeypatch, [_chunk("d1#a", "Doc", "text")])
    monkeypatch.setattr(
        "memex.agents.expert.complete_reasoning",
        _FakeReason("<think>weighing options</think>The recommendation is X."),
    )
    ans = await expert_answer("Advise me.")
    assert ans.reasoning == "weighing options"
    assert ans.answer == "The recommendation is X."


def test_split_think_untagged_is_noop() -> None:
    assert split_think("plain inline reasoning, no tag") == (None, "plain inline reasoning, no tag")


def test_split_think_extracts_tagged_trace() -> None:
    assert split_think("<think>a b c</think>final answer") == ("a b c", "final answer")


def test_split_think_truncated_trace_is_all_trace() -> None:
    # max_tokens cut the close tag → treat the whole tail as the trace, empty body.
    assert split_think("<think>unfinished reasoning") == ("unfinished reasoning", "")


def test_cli_render_expert_answer() -> None:
    from memex.agents.expert import ExpertAnswer, ExpertEvidence
    from memex.cli.commands import _render_expert_answer

    ans = ExpertAnswer(
        question="Q?",
        answer="The reasoned analysis.",
        evidence=[
            ExpertEvidence(chunk_id="d1#a", document_id="d1", title="Guide A", snippet="…"),
            ExpertEvidence(chunk_id="d2#b", document_id="d2", title="Guide B", snippet="…"),
        ],
        model="m",
        tokens=10,
    )
    out = _render_expert_answer(ans)
    assert "The reasoned analysis." in out
    assert "evidence consulted: Guide A, Guide B" in out
    assert EXPERT_PROVENANCE_NOTE in out
