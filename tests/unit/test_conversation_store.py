"""ConversationStore CRUD — the grounded-chat sqlite sidecar (Surface A)."""

from __future__ import annotations

from pathlib import Path

import pytest

from memex.core.conversation_store import ConversationStore
from memex.core.errors import VaultIntegrityError


async def test_create_append_load_roundtrip(tmp_path: Path) -> None:
    store = await ConversationStore.open(tmp_path)
    try:
        convo = await store.create_conversation()
        assert convo.turn_count == 0

        t0 = await store.append_turn(
            convo.conversation_id,
            user_text="What is X?",
            standalone_query="What is X?",
            is_followup=False,
            answered=True,
            answer_summary="X is a thing.",
            cited_chunk_ids=["doc#a", "doc#b"],
            response_json='{"answered": true}',
            correlation_id="c0",
        )
        assert t0.turn_index == 0

        t1 = await store.append_turn(
            convo.conversation_id,
            user_text="and its latency?",
            standalone_query="what is the latency of X",
            is_followup=True,
            answered=False,
            answer_summary="I couldn't ground that.",
            cited_chunk_ids=[],
            response_json=None,
            correlation_id="c1",
        )
        assert t1.turn_index == 1

        loaded = await store.load(convo.conversation_id)
        assert loaded is not None
        assert loaded.turn_count == 2
        # Title auto-derived from the first message.
        assert loaded.title == "What is X?"
        assert [t.user_text for t in loaded.turns] == ["What is X?", "and its latency?"]
        # Chunk IDs round-trip (ids only — never text).
        assert loaded.turns[0].cited_chunk_ids == ["doc#a", "doc#b"]
        assert loaded.turns[1].is_followup is True
        assert loaded.turns[1].answered is False
        assert loaded.turns[1].response_json is None
    finally:
        await store.close()


async def test_set_scope_and_running_summary(tmp_path: Path) -> None:
    store = await ConversationStore.open(tmp_path)
    try:
        convo = await store.create_conversation()
        await store.set_scope(convo.conversation_id, [" doc-1 ", "doc-2", "doc-1", ""])
        await store.update_running_summary(convo.conversation_id, "compacted history")
        loaded = await store.load(convo.conversation_id)
        assert loaded is not None
        # Deduped + blank-stripped, order preserved.
        assert loaded.scope_doc_ids == ["doc-1", "doc-2"]
        assert loaded.running_summary == "compacted history"
    finally:
        await store.close()


async def test_list_and_delete(tmp_path: Path) -> None:
    store = await ConversationStore.open(tmp_path)
    try:
        a = await store.create_conversation()
        b = await store.create_conversation()
        await store.append_turn(
            b.conversation_id,
            user_text="hi",
            standalone_query="hi",
            is_followup=False,
            answered=True,
            answer_summary="hello",
            cited_chunk_ids=[],
            response_json=None,
            correlation_id=None,
        )
        listed = await store.list_conversations()
        assert {c.conversation_id for c in listed} == {a.conversation_id, b.conversation_id}

        await store.delete_conversation(a.conversation_id)
        listed2 = await store.list_conversations()
        assert {c.conversation_id for c in listed2} == {b.conversation_id}
        assert await store.load(a.conversation_id) is None
    finally:
        await store.close()


async def test_append_turn_missing_conversation_raises_integrity(tmp_path: Path) -> None:
    """#396: a TOCTOU append onto a concurrently-deleted conversation raises a TYPED
    `VaultIntegrityError` carrying the id in `context` — not a bare `KeyError` (Errors rule)."""
    store = await ConversationStore.open(tmp_path)
    try:
        convo = await store.create_conversation()
        await store.delete_conversation(convo.conversation_id)
        with pytest.raises(VaultIntegrityError) as ei:
            await store.append_turn(
                convo.conversation_id,
                user_text="orphaned turn",
                standalone_query="orphaned turn",
                is_followup=False,
                answered=True,
                answer_summary="...",
                cited_chunk_ids=[],
                response_json=None,
                correlation_id="c0",
            )
        assert ei.value.context["conversation_id"] == convo.conversation_id
    finally:
        await store.close()
