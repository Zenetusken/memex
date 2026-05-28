"""Unit tests for the live-progress plumbing (webui long-poll indicator):

- `agents.answering._NodeProgressHandler` — the LangGraph callback discriminator
  that turns node starts into phase updates (pins the verified langgraph 1.2.0
  field combo so an upgrade that changes it fails loudly).
- `webui.progress.ProgressRegistry` — CRUD, the monotonic version, the
  event-driven `wait_for_change`, lazy TTL/cap cleanup.
- `agents.answering.answer_query` — the opt-in `correlation_id` + `on_node`
  threading (additive; CLI/MCP unaffected).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.callbacks.base import BaseCallbackHandler

import memex.agents.answering as ans
from memex.agents.answering import FinalResponse, _NodeProgressHandler, answer_query
from memex.webui.progress import (
    PHASES,
    SUMMARY_PHASES,
    ProgressEntry,
    ProgressRegistry,
    phase_for,
    summary_phase_view,
)

# --- _NodeProgressHandler discriminator (verified langgraph 1.2.0 fields) -------


def _start(
    handler: _NodeProgressHandler, *, node: str | None, tags: list[str], name: object
) -> None:
    handler.on_chain_start(None, {}, tags=tags, metadata={"langgraph_node": node}, name=name)


def test_handler_emits_on_top_level_node() -> None:
    seen: list[str] = []
    _start(_NodeProgressHandler(seen.append), node="rerank", tags=["graph:step:3"], name="rerank")
    assert seen == ["rerank"]


def test_handler_skips_nested_runnable() -> None:
    seen: list[str] = []
    # A nested runnable inside a node: langgraph_node set, but a seq:step tag.
    _start(
        _NodeProgressHandler(seen.append),
        node="rerank",
        tags=["seq:step:1"],
        name="RunnableLambda",
    )
    assert seen == []


def test_handler_skips_name_mismatch_and_unknown_and_root() -> None:
    seen: list[str] = []
    h = _NodeProgressHandler(seen.append)
    _start(h, node="rerank", tags=["graph:step:2"], name="MISMATCH")  # nested-ish, name≠node
    _start(h, node="not_a_node", tags=["graph:step:1"], name="not_a_node")  # not a real node
    h.on_chain_start(None, {}, tags=[], metadata=None, name="LangGraph")  # the root event
    assert seen == []


def test_handler_swallows_sink_errors() -> None:
    def boom(_node: str) -> None:
        raise RuntimeError("sink failed")

    # A misbehaving progress sink must never propagate into the answering run.
    _start(_NodeProgressHandler(boom), node="retrieve", tags=["graph:step:0"], name="retrieve")


# --- ProgressRegistry -----------------------------------------------------------


def _fr(cid: str = "c") -> FinalResponse:
    return FinalResponse(
        answered=True,
        summary="x",
        claims=[],
        correlation_id=cid,
        tokens_used=0,
        nodes_traversed=0,
        regenerate_attempts=0,
    )


def test_registry_new_set_phase_finish_evict() -> None:
    reg = ProgressRegistry()
    entry = reg.new("c1", scope_doc_ids=["d1"], scope_source="selected")
    assert entry.phase == PHASES[0] and entry.version == 0
    assert entry.scope_doc_ids == ["d1"] and entry.scope_source == "selected"

    reg.set_phase("c1", "Grounding")
    got = reg.get("c1")
    assert got is not None and got.phase == "Grounding" and got.version == 1
    reg.set_phase("c1", "Grounding")  # same phase → no bump
    got = reg.get("c1")
    assert got is not None and got.version == 1

    fr = _fr("c1")
    reg.finish("c1", response=fr)
    done = reg.get("c1")
    assert done is not None and done.done and done.response is fr and done.version == 2

    reg.evict("c1")
    assert reg.get("c1") is None


def test_registry_missing_cid_is_noop() -> None:
    reg = ProgressRegistry()
    reg.set_phase("nope", "Grounding")  # no raise — entry already gone (swept/evicted)
    reg.finish("nope", error="x")  # no raise
    assert reg.get("nope") is None


def test_phase_for_and_active_index() -> None:
    assert phase_for("rerank") == "Reranking"
    assert phase_for("verify") == "Grounding"
    assert phase_for("compose") == "Composing"
    assert phase_for("unknown_node") == ""  # keep current phase upstream
    assert ProgressEntry(
        scope_doc_ids=[], scope_source="named", phase="Grounding"
    ).active_index() == (PHASES.index("Grounding"))
    assert ProgressEntry(scope_doc_ids=[], scope_source="named", phase="weird").active_index() == 0


def test_summary_phase_view_parses_base_and_detail() -> None:
    assert summary_phase_view("Summarizing · section 3 of 9") == (
        SUMMARY_PHASES.index("Summarizing"),
        "section 3 of 9",
    )
    assert summary_phase_view("Summarizing · key figures") == (1, "key figures")
    assert summary_phase_view("Reducing") == (SUMMARY_PHASES.index("Reducing"), "")
    assert summary_phase_view("Reading") == (0, "")
    assert summary_phase_view("Mystery") == (0, "")  # unknown base → step 0


def test_registry_sweep_evicts_stale_on_new() -> None:
    reg = ProgressRegistry()
    stale = reg.new("old", scope_doc_ids=[], scope_source="named")
    stale.started_at -= 10_000  # force it past the TTL
    reg.new("fresh", scope_doc_ids=[], scope_source="named")  # new() sweeps first
    assert reg.get("old") is None
    assert reg.get("fresh") is not None


@pytest.mark.asyncio
async def test_wait_for_change_wakes_on_bump() -> None:
    reg = ProgressRegistry()
    reg.new("c", scope_doc_ids=[], scope_source="named")

    async def _bump() -> None:
        await asyncio.sleep(0.01)
        reg.set_phase("c", "Grounding")

    task = asyncio.create_task(_bump())
    entry = await reg.wait_for_change("c", 0, keepalive=2.0)
    assert entry is not None and entry.phase == "Grounding" and entry.version == 1
    await task


@pytest.mark.asyncio
async def test_wait_for_change_returns_immediately_when_already_advanced() -> None:
    reg = ProgressRegistry()
    reg.new("c", scope_doc_ids=[], scope_source="named")
    reg.set_phase("c", "Grounding")  # version → 1, before we wait
    entry = await reg.wait_for_change("c", 0, keepalive=5.0)  # 1 > 0 → no block
    assert entry is not None and entry.version == 1


@pytest.mark.asyncio
async def test_wait_for_change_keepalive_timeout() -> None:
    reg = ProgressRegistry()
    reg.new("c", scope_doc_ids=[], scope_source="named")
    entry = await reg.wait_for_change("c", 0, keepalive=0.02)  # no bump → heartbeat return
    assert entry is not None and entry.version == 0


@pytest.mark.asyncio
async def test_wait_for_change_unknown_cid_returns_none() -> None:
    reg = ProgressRegistry()
    assert await reg.wait_for_change("nope", 0) is None


# --- answer_query threading (stubbed graph; no models) --------------------------


class _DummyHandler(BaseCallbackHandler):
    """Stand-in for the Langfuse handler so the test needs no observability setup."""


@pytest.mark.asyncio
async def test_answer_query_threads_cid_and_appends_progress_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _StubGraph:
        async def ainvoke(self, initial: Any, config: dict[str, Any]) -> dict[str, Any]:
            captured["callbacks"] = list(config["callbacks"])
            captured["cid"] = initial.correlation_id
            return {"final": _fr(initial.correlation_id)}

    monkeypatch.setattr(ans, "get_compiled_graph", lambda: _StubGraph())
    monkeypatch.setattr(ans, "callback_handler", _DummyHandler)

    seen: list[str] = []
    resp = await answer_query("q", correlation_id="fixed-cid", on_node=seen.append)
    assert resp.correlation_id == "fixed-cid"
    assert captured["cid"] == "fixed-cid"
    assert any(isinstance(c, _NodeProgressHandler) for c in captured["callbacks"])

    # No correlation_id / on_node → a fresh ULID + no progress handler (CLI/MCP path).
    resp2 = await answer_query("q")
    assert resp2.correlation_id and resp2.correlation_id != "fixed-cid"
    assert not any(isinstance(c, _NodeProgressHandler) for c in captured["callbacks"])
