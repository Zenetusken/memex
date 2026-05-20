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
from typing import Any

import pytest

from memex.agents.answering import (
    Chunk,
    CitedClaim,
    DraftAnswer,
    FinalResponse,
    SufficiencyAssessment,
    VerificationResult,
    reset_compiled_graph,
)
from memex.core.config import MemexSettings, set_settings
from memex.ingest.pipeline import ingest_markdown_passthrough
from memex.mcp.server import (
    ask,
    get_document,
    list_documents_tool,
    search,
)


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    (tmp_path / "documents").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def settings(
    tmp_vault: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[MemexSettings]:
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
def patch_retrieve(
    monkeypatch: pytest.MonkeyPatch, fake_chunks: list[Chunk]
) -> None:
    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return list(fake_chunks)

    async def _rerank(
        query: str, candidates: list[Chunk], top_k: int = 10
    ) -> list[Chunk]:
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
            correlation_id="01HZTESTASKCORRELATIONID00",
            tokens_used=42,
            nodes_traversed=5,
            regenerate_attempts=0,
        )

    monkeypatch.setattr("memex.mcp.server.answer_query", _fake)


# ----- Tests -----


@pytest.mark.asyncio
async def test_search_returns_typed_chunks(
    settings: MemexSettings, patch_retrieve: None
) -> None:
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
async def test_ask_returns_grounded_response(
    settings: MemexSettings, patch_agent: None
) -> None:
    result = await ask("What does Smith say about reflexivity?")
    assert isinstance(result, FinalResponse)
    assert result.answered is True
    assert result.summary
    assert len(result.claims) == 1
    assert result.correlation_id == "01HZTESTASKCORRELATIONID00"


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


def test_server_has_registered_the_five_tools() -> None:
    """Smoke: the FastMCP server has the five tools we expect."""
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
    } <= names
