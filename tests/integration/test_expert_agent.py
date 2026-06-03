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


@pytest.mark.asyncio
async def test_expert_uses_reasoner_id_when_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#395 (reserved-hook contract): when `models.reasoner` is set, expert RESOLVES + STAMPS
    that id onto `ExpertAnswer.model` with NO serve subprocess. (The actual daemon-send is
    `complete_reasoning`'s pre-existing `reasoner or orchestrator` routing — faked here; this
    pins that expert prefers the reasoner id, the observable half of the de-promised semantics.)"""
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("MEMEX_MODELS__REASONER", "sentinel-reasoner-id")
    s = MemexSettings()
    set_settings(s)
    try:
        _patch_retrieval(monkeypatch, [_chunk("d1#a", "Doc", "text")])
        monkeypatch.setattr("memex.agents.expert.complete_reasoning", _FakeReason("answer"))
        ans = await expert_answer("Why?")
        assert ans.model == "sentinel-reasoner-id"
    finally:
        set_settings(None)


def test_expert_module_has_no_swap_in_imports() -> None:
    """#395 canary: the reasoner swap-in lifecycle is deliberately UNWIRED in v1 (ADR-0013).
    If a future change WIRES it (clones `serve_summarizer_vllm`), update this test on purpose."""
    import memex.agents.expert as expert_mod

    src = Path(expert_mod.__file__).read_text()
    assert "serve_summarizer_vllm" not in src
    assert "serve_reasoner" not in src
    assert "pause_vllm_for_gpu" not in src
    assert "AsyncExitStack" not in src


@pytest.mark.asyncio
async def test_expert_clears_run_context_on_pre_try_failure(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#396(c): a failure in the PRE-TRY setup window must not leak a bound correlation_id.
    The bind now lives INSIDE the try (covered by `finally: clear_run_context()`); pre-fix it
    bound BEFORE the try, so a raise in the setup leaked context. The discriminating injection
    is therefore in the pre-try window (`get_settings()`) — a raise INSIDE the try clears in
    BOTH orderings and would NOT catch a revert of the reorder."""
    import structlog

    def _boom_settings() -> MemexSettings:
        raise RuntimeError("settings down")

    monkeypatch.setattr("memex.agents.expert.get_settings", _boom_settings)
    with pytest.raises(RuntimeError):
        await expert_answer("Why?")
    # Pre-fix (bind before try) this raise left correlation_id bound; the reorder makes it empty.
    assert structlog.contextvars.get_contextvars() == {}


def test_bridge_escalation_hint_on_refusal_when_enabled() -> None:
    """§11 consented escalation: a CLI `ask` REFUSAL with expert mode on names `memex bridge`
    with the verbatim question (a hint the user chooses to run — never auto-executed)."""
    from memex.cli.commands import _bridge_escalation_hint

    hint = _bridge_escalation_hint(False, "why is the sky blue?", expert_enabled=True)
    assert hint is not None
    assert "memex bridge" in hint
    assert "why is the sky blue?" in hint  # shlex-quoted, but the question text is preserved


def test_bridge_escalation_hint_shell_quotes_the_query() -> None:
    """A question containing a double-quote must produce a copy-paste-safe command."""
    import shlex

    from memex.cli.commands import _bridge_escalation_hint

    hint = _bridge_escalation_hint(False, 'say "hi" now', expert_enabled=True)
    assert hint is not None
    assert shlex.quote('say "hi" now') in hint


def test_bridge_escalation_hint_none_when_answered() -> None:
    from memex.cli.commands import _bridge_escalation_hint

    # Refusal-only: a grounded answer prints no escalation hint.
    assert _bridge_escalation_hint(True, "q", expert_enabled=True) is None


def test_bridge_escalation_hint_none_when_expert_disabled() -> None:
    from memex.cli.commands import _bridge_escalation_hint

    # Gated on the ungrounded surface being enabled.
    assert _bridge_escalation_hint(False, "q", expert_enabled=False) is None


def test_bridge_escalation_hint_uses_answer_flag() -> None:
    """ADR-0016: the hint points at `memex bridge --answer` so the CLI escalation also presents
    the grounded subset AS the answer when responsive."""
    from memex.cli.commands import _bridge_escalation_hint

    hint = _bridge_escalation_hint(False, "why is the sky blue?", expert_enabled=True)
    assert hint is not None
    assert "memex bridge --answer" in hint


def _bridge_answer(*, presented: bool):
    from memex.agents.answering import CitedClaim
    from memex.agents.bridge import BRIDGE_PROVENANCE_NOTE, BridgeAnswer
    from memex.agents.expert import ExpertEvidence
    from memex.core.types import Chunk

    claim = CitedClaim(claim="OSPF is link-state.", source_chunk_id="d#a", confidence="high")
    return BridgeAnswer(
        question="Q?",
        analysis="Some ungrounded reasoning.",
        grounded_claims=[claim],
        grounded_sources=[Chunk(chunk_id="d#a", document_id="d", document_title="OSPF Guide",
                                text="…", heading_path=["Intro"])],
        evidence=[ExpertEvidence(chunk_id="d#a", document_id="d", title="OSPF Guide",
                                 section="Intro", snippet="…")],
        provenance_note=BRIDGE_PROVENANCE_NOTE,
        n_extracted=1,
        n_grounded=1,
        present_as_answer=presented,
        responsive=(True if presented else None),
        answer_headline=("OSPF is link-state." if presented else ""),
        presented_claims=([claim] if presented else []),
    )


def test_cli_render_bridge_surfaces_evidence_consulted() -> None:
    """Both bridge renders name the vault documents reasoned over (parity with `expert`), so the
    user can open them and see what the vault actually says even when nothing grounded."""
    from memex.cli.commands import _render_bridge_answer

    for presented in (True, False):
        out = _render_bridge_answer(_bridge_answer(presented=presented))
        assert "Retrieved from your vault (open to see what it says): OSPF Guide" in out


def test_cli_render_bridge_presented_leads_with_answer() -> None:
    """A presented bridge result leads with the grounded ANSWER; the ungrounded reasoning follows,
    fenced and labelled (no ungrounded text in the answer block)."""
    from memex.cli.commands import _render_bridge_answer

    out = _render_bridge_answer(_bridge_answer(presented=True))
    assert out.index("ANSWER (grounded") < out.index("OSPF is link-state.")
    assert "REASONING (ungrounded" in out
    assert out.index("OSPF is link-state.") < out.index("Some ungrounded reasoning.")


def test_cli_render_bridge_labelled_leads_with_analysis() -> None:
    """The standalone / non-presented result keeps the analysis-first labelled rendering."""
    from memex.cli.commands import _render_bridge_answer

    out = _render_bridge_answer(_bridge_answer(presented=False))
    assert out.startswith("ANALYSIS (ungrounded")
    assert "GROUNDED CLAIMS" in out
    assert "ANSWER (grounded" not in out


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
