"""MCP server tool surface — regression net.

The MCP transport layer (stdio / streamable HTTP) is exercised by
upstream `mcp` SDK tests. Memex's responsibility is the *tool surface*
— the five functions registered on `server`, their typed return
shapes, and the contract that pydantic models cross this boundary
(not raw dicts).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from memex.agents.answering import (
    Chunk,
    CitedClaim,
    FinalResponse,
    reset_compiled_graph,
)
from memex.core.config import MemexSettings, set_settings
from memex.ingest.pipeline import ingest_markdown_passthrough
from memex.mcp.server import (
    ask,
    get_document,
    list_documents_tool,
    list_scope_sets_tool,
    search,
)


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    (tmp_path / "documents").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def settings(tmp_vault: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemexSettings]:
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_vault))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    s = MemexSettings()  # type: ignore[call-arg]
    set_settings(s)
    yield s
    set_settings(None)


@pytest.fixture(autouse=True)
def _fresh_graph() -> Iterator[None]:
    reset_compiled_graph()
    yield
    reset_compiled_graph()


@pytest.fixture
def fake_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="d1#a",
            document_id="d1",
            document_title="Smith 2024 on reflexivity",
            text="Reflexivity in research design means the researcher's "
            "presence shapes the data they collect.",
            page=3,
            score=0.91,
        ),
        Chunk(
            chunk_id="d1#b",
            document_id="d1",
            document_title="Smith 2024 on reflexivity",
            text="Smith argues that reflexivity is constitutive.",
            page=4,
            score=0.85,
        ),
    ]


@pytest.fixture
def patch_retrieve(monkeypatch: pytest.MonkeyPatch, fake_chunks: list[Chunk]) -> None:
    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return list(fake_chunks)

    async def _rerank(query: str, candidates: list[Chunk], top_k: int = 10) -> list[Chunk]:
        # Return rerank scores so the MCP tool's payload includes them.
        return [
            c.model_copy(update={"rerank_score": 0.95 - 0.1 * i})
            for i, c in enumerate(candidates[:top_k])
        ]

    monkeypatch.setattr("memex.mcp.server.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.mcp.server.cross_encoder_rerank", _rerank)


@pytest.fixture
def patch_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake(question: str, **_kw: Any) -> FinalResponse:
        return FinalResponse(
            answered=True,
            summary="Smith treats reflexivity as constitutive.",
            claims=[
                CitedClaim(
                    claim="Reflexivity shapes the data.",
                    source_chunk_id="d1#a",
                    confidence="high",
                )
            ],
            wikilinks=["[[d1#Reflexivity]]"],
            correlation_id="01HZTESTASKCORRELATIONID00",
            tokens_used=42,
            nodes_traversed=5,
            regenerate_attempts=0,
        )

    monkeypatch.setattr("memex.mcp.server.answer_query", _fake)


# ----- Tests -----


@pytest.mark.asyncio
async def test_search_returns_typed_chunks(settings: MemexSettings, patch_retrieve: None) -> None:
    result = await search("reflexivity", k=2)
    assert isinstance(result, list)
    assert len(result) == 2
    # Each entry is a pydantic Chunk; FastMCP serialises it to JSON
    # on the wire while Python callers get the typed object.
    for entry in result:
        assert isinstance(entry, Chunk)
        assert entry.chunk_id
        assert entry.document_id
        assert entry.text
        assert entry.rerank_score is not None


@pytest.mark.asyncio
async def test_ask_returns_grounded_response(settings: MemexSettings, patch_agent: None) -> None:
    result = await ask("What does Smith say about reflexivity?")
    assert isinstance(result, FinalResponse)
    assert result.answered is True
    assert result.summary
    assert len(result.claims) == 1
    assert result.correlation_id == "01HZTESTASKCORRELATIONID00"
    # P4.1: wikilinks ride the full FinalResponse over MCP (auto-serialized).
    assert result.wikilinks == ["[[d1#Reflexivity]]"]
    assert "wikilinks" in result.model_dump()


@pytest.mark.asyncio
async def test_ask_forwards_scope_doc_ids(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MCP `ask` tool forwards an explicit doc-scope to the agent + echoes the
    applied scope on the FinalResponse — the composable surface for the picker."""
    captured: dict[str, Any] = {}

    async def _fake(question: str, **kw: Any) -> FinalResponse:
        captured["scope_doc_ids"] = kw.get("scope_doc_ids")
        return FinalResponse(
            answered=False,
            refusal_reason="Not in the selected document.",
            artifact_scope_doc_ids=["d1"],
            correlation_id="01HZTESTMCPSCOPE0000000000",
            tokens_used=8,
            nodes_traversed=6,
            regenerate_attempts=0,
        )

    monkeypatch.setattr("memex.mcp.server.answer_query", _fake)

    result = await ask("What does d2 say?", scope_doc_ids=["d1"])
    assert captured["scope_doc_ids"] == ["d1"]
    assert result.artifact_scope_doc_ids == ["d1"]


@pytest.mark.asyncio
async def test_ask_with_scope_set_unions_doc_ids(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ask(scope_set=NAME)` resolves the saved set and unions its docs with any
    explicit `scope_doc_ids` (ordered de-dup)."""
    from memex.core.scope_sets import save_scope_set

    await save_scope_set(settings.vault_path, "net", ["d1", "d2"])
    captured: dict[str, Any] = {}

    async def _fake(question: str, **kw: Any) -> FinalResponse:
        captured["scope_doc_ids"] = kw.get("scope_doc_ids")
        return FinalResponse(
            answered=True,
            summary="ok",
            correlation_id="01HZTESTMCPSET00000000000000",
            tokens_used=8,
            nodes_traversed=6,
            regenerate_attempts=0,
        )

    monkeypatch.setattr("memex.mcp.server.answer_query", _fake)

    await ask("q", scope_doc_ids=["d3"], scope_set="NET")  # case-insensitive
    assert captured["scope_doc_ids"] == ["d3", "d1", "d2"]


@pytest.mark.asyncio
async def test_ask_unknown_scope_set_raises(settings: MemexSettings) -> None:
    from memex.core.errors import ConfigurationError

    with pytest.raises(ConfigurationError):
        await ask("q", scope_set="does-not-exist")


@pytest.mark.asyncio
async def test_list_scope_sets_tool_returns_saved_sets(settings: MemexSettings) -> None:
    from memex.core.scope_sets import save_scope_set

    await save_scope_set(settings.vault_path, "Alpha", ["d1"])
    await save_scope_set(settings.vault_path, "Beta", ["d2", "d3"])
    sets = await list_scope_sets_tool()
    by_name = {s.name: s for s in sets}
    assert set(by_name) == {"Alpha", "Beta"}
    assert by_name["Beta"].doc_ids == ["d2", "d3"]


@pytest.mark.asyncio
async def test_get_document_returns_canonical_markdown(
    settings: MemexSettings,
) -> None:
    from memex.vault.store import VaultDocument

    body = "# Test doc\n\nA short paragraph for the MCP test.\n"
    ref = await ingest_markdown_passthrough(body, source_stem="mcp_test")

    result = await get_document(ref.doc_id)
    assert isinstance(result, VaultDocument)
    assert "Test doc" in result.body or "short paragraph" in result.body
    assert result.frontmatter.title == "mcp_test"
    assert result.ref.doc_id == ref.doc_id


@pytest.mark.asyncio
async def test_list_documents_returns_all_refs(
    settings: MemexSettings,
) -> None:
    from memex.vault.store import DocumentRef

    a = await ingest_markdown_passthrough("# A\n\nFirst.\n", source_stem="a")
    b = await ingest_markdown_passthrough("# B\n\nSecond.\n", source_stem="b")

    result = await list_documents_tool()
    assert isinstance(result, list)
    for entry in result:
        assert isinstance(entry, DocumentRef)
    ids = {entry.doc_id for entry in result}
    assert a.doc_id in ids
    assert b.doc_id in ids


@pytest.mark.asyncio
async def test_get_document_404_on_unknown_doc(
    settings: MemexSettings,
) -> None:
    from memex.core.errors import VaultIntegrityError

    with pytest.raises(VaultIntegrityError):
        await get_document("nonexistent-doc")


def test_server_has_registered_the_core_tools() -> None:
    """Smoke: the FastMCP server has the tools we expect registered."""
    from memex.mcp.server import server

    # FastMCP exposes registered tools via `_tool_manager`; if the API
    # shape shifts in a future SDK version we'll catch it here.
    tools = server._tool_manager.list_tools()  # type: ignore[attr-defined]
    names = {t.name for t in tools}
    assert {
        "search",
        "ask",
        "get_document",
        "list_documents",
        "get_graph_neighbors",
        "related_documents",
        "entity_overview",
    } <= names


@pytest.mark.asyncio
async def test_related_documents_tool_returns_ranked_list(
    settings: MemexSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `related_documents` MCP tool surfaces GraphStore.related_documents
    (entity-specificity-ranked) and is fail-open when the graph is unavailable."""
    from memex.index.graph_store import RelatedDocument
    from memex.mcp.server import related_documents

    class _FakeStore:
        @classmethod
        async def open(cls, vault_path):
            return cls()

        async def related_documents(self, doc_id, *, limit=10, max_entities=8):
            return [
                RelatedDocument(
                    doc_id="sib", title="Sibling", score=3.9, shared_entities=["DNS spoofing"]
                )
            ]

        async def close(self):
            return None

    monkeypatch.setattr("memex.index.graph_store.GraphStore.open", staticmethod(_FakeStore.open))
    out = await related_documents("some-doc", limit=5)
    assert [r.doc_id for r in out] == ["sib"]
    assert out[0].shared_entities == ["DNS spoofing"]

    # Fail-open: ryugraph absent → ImportError → empty list, never a crash.
    def _boom(vault_path):
        raise ImportError("ryugraph not installed")

    monkeypatch.setattr("memex.index.graph_store.GraphStore.open", staticmethod(_boom))
    assert await related_documents("some-doc") == []


@pytest.mark.asyncio
async def test_entity_overview_tool_returns_profile_and_passages(
    settings: MemexSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `entity_overview` MCP tool surfaces the orchestrator's EntityOverview
    (graph profile + FTS passages) as a typed pydantic model across the boundary."""
    from memex.index.graph_store import CoOccurringEntity, EntityMention, EntityProfile
    from memex.mcp import server as srv
    from memex.retrieve import EntityOverview

    overview = EntityOverview(
        profile=EntityProfile(
            query_name="STP",
            matched_names=["STP"],
            kinds=["concept"],
            doc_count=2,
            mentions=[EntityMention(doc_id="d1", title="Doc 1")],
            cooccurring=[
                CoOccurringEntity(name="ARP", kind="concept", shared_docs=2, score=1.39)
            ],
            resolved=True,
        ),
        passages=[Chunk(chunk_id="d1#a", document_id="d1", document_title="Doc 1", text="STP …")],
        passages_scoped=True,
    )

    async def _fake(name: str, **_kw: object) -> EntityOverview:
        assert name == "STP"
        return overview

    monkeypatch.setattr(srv, "_entity_overview", _fake)
    out = await srv.entity_overview("STP")
    assert isinstance(out, EntityOverview)
    assert out.profile.resolved is True
    assert out.passages_scoped is True
    assert [c.name for c in out.profile.cooccurring] == ["ARP"]
    assert [p.chunk_id for p in out.passages] == ["d1#a"]


# ----- HTTP transport: bind validation + auth wiring -----


@pytest.mark.asyncio
async def test_serve_http_refuses_non_loopback_without_token(
    settings: MemexSettings,
) -> None:
    """Non-loopback bind + no auth_token → ConfigurationError before
    any server starts. The default `settings` fixture leaves auth_token
    unset, so `validate_bind` should trip immediately.
    """
    from memex.core.errors import ConfigurationError
    from memex.mcp.server import serve_http

    with pytest.raises(ConfigurationError) as exc:
        await serve_http(host="0.0.0.0", port=7424)
    assert exc.value.context["host"] == "0.0.0.0"
    assert "MEMEX_MCP__AUTH_TOKEN" in cast("str", exc.value.context["fix"])


@pytest.mark.asyncio
async def test_serve_http_wraps_app_with_bearer_middleware_when_token_set(
    tmp_vault: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `auth_token` is configured, `serve_http` must wrap the
    FastMCP ASGI app with `BearerAuthMiddleware` before handing it to
    uvicorn. We verify by intercepting the uvicorn.Server.serve call:
    the captured app's middleware stack must contain BearerAuthMiddleware
    and respond 401 on an unauthenticated request via TestClient.
    """
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_vault))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("MEMEX_MCP__AUTH_TOKEN", "verify-this-token")
    s = MemexSettings()  # type: ignore[call-arg]
    set_settings(s)
    try:
        # Capture the uvicorn.Server.serve invocation without binding a
        # socket. The app is reachable via the captured Config.
        captured: dict[str, Any] = {}

        class _FakeServer:
            def __init__(self, config: Any) -> None:
                captured["config"] = config

            async def serve(self) -> None:
                # Don't actually run; serve_http awaits this — return
                # immediately so the test completes.
                return None

        monkeypatch.setattr("uvicorn.Server", _FakeServer)

        from memex.mcp.server import serve_http

        await serve_http(host="127.0.0.1", port=18001)

        config = captured["config"]
        app = config.app
        # The middleware should be in the stack. Starlette stores
        # user-added middleware on `app.user_middleware`.
        from memex.mcp.auth import BearerAuthMiddleware

        middleware_classes = [m.cls for m in app.user_middleware]
        assert BearerAuthMiddleware in middleware_classes

        # Mount the (real) app in a TestClient and verify the auth gate
        # actually fires. `streamable_http_app` exposes routes under
        # `/mcp`; an unauthenticated GET on any path under there
        # should return 401.
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            resp = client.get("/mcp/")
            assert resp.status_code == 401
            assert resp.headers["WWW-Authenticate"].startswith("Bearer")
    finally:
        set_settings(None)
