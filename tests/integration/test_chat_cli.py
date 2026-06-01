"""The CLI grounded-chat REPL (`memex chat` → `run_chat_repl`).

The per-turn answer is faked, so the test drives the loop with an input iterator and
captures the emitted lines — no GPU, no model.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from memex.agents.answering import CitedClaim, FinalResponse
from memex.agents.chat import ChatTurnResult
from memex.cli.commands import run_chat_repl
from memex.core.config import MemexSettings, set_settings
from memex.core.types import Chunk, ConversationTurn


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemexSettings]:
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    s = MemexSettings()
    set_settings(s)
    yield s
    set_settings(None)


async def test_repl_drives_two_turns_then_exits(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def fake_answer_turn(
        conversation_id: str, user_text: str, **_kw: object
    ) -> ChatTurnResult:
        calls.append(user_text)
        used = [Chunk(chunk_id="d#a", document_id="d", document_title="D", text="x")]
        resp = FinalResponse(
            answered=True,
            summary=f"answer to {user_text}",
            claims=[CitedClaim(claim="a claim", source_chunk_id="d#a", confidence="high")],
            used_chunks=used,
            correlation_id="cid",
            tokens_used=1,
            nodes_traversed=1,
            regenerate_attempts=0,
        )
        turn = ConversationTurn(
            turn_id="t",
            conversation_id=conversation_id,
            turn_index=len(calls) - 1,
            user_text=user_text,
            standalone_query=user_text,
        )
        return ChatTurnResult(
            response=resp, turn=turn, standalone_query=user_text, is_followup=False
        )

    monkeypatch.setattr("memex.cli.commands.answer_turn", fake_answer_turn)

    lines = iter(["What is X?", "", "and Y?", "/exit", "never reached"])
    emitted: list[str] = []

    def _read() -> str | None:
        return next(lines, None)

    def _emit(text: str) -> None:
        emitted.append(text)

    conversation_id = await run_chat_repl(_read, _emit)

    # Blank line skipped; /exit ends before "never reached".
    assert calls == ["What is X?", "and Y?"]
    assert conversation_id  # a conversation was created
    joined = "\n".join(emitted)
    assert "answer to What is X?" in joined
    assert "a claim" in joined


async def test_repl_renders_refusal_with_suggestions(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from memex.core.types import RelatedDocument

    async def fake_answer_turn(
        conversation_id: str, user_text: str, **_kw: object
    ) -> ChatTurnResult:
        resp = FinalResponse(
            answered=False,
            refusal_reason="The vault doesn't cover that.",
            related_documents=[
                RelatedDocument(doc_id="d2", title="Closest Doc", score=1.0, shared_entities=[])
            ],
            correlation_id="cid",
            tokens_used=1,
            nodes_traversed=1,
            regenerate_attempts=0,
        )
        turn = ConversationTurn(
            turn_id="t",
            conversation_id=conversation_id,
            turn_index=0,
            user_text=user_text,
            standalone_query=user_text,
        )
        return ChatTurnResult(
            response=resp, turn=turn, standalone_query=user_text, is_followup=False
        )

    monkeypatch.setattr("memex.cli.commands.answer_turn", fake_answer_turn)

    lines = iter(["something obscure", "/exit"])
    emitted: list[str] = []
    await run_chat_repl(lambda: next(lines, None), emitted.append)

    joined = "\n".join(emitted)
    assert "The vault doesn't cover that." in joined
    assert "Closest Doc" in joined
