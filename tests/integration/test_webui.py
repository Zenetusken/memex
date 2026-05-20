"""FastAPI web UI — Phase 3 regression net.

Drives the routes via Starlette's `TestClient`. The answering agent is
faked; the rest of the stack is real (Jinja templates render, the vault
is a real tmp dir, the document list iterates real files).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from memex.agents.answering import (
    CitedClaim,
    FinalResponse,
    reset_compiled_graph,
)
from memex.core.config import MemexSettings, set_settings
from memex.ingest.pipeline import ingest_markdown_passthrough
from memex.webui.app import create_app


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
def client(settings: MemexSettings) -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def fake_answered(monkeypatch: pytest.MonkeyPatch) -> None:
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
            correlation_id="01HZTESTWEBUI0000000000000",
            tokens_used=42,
            nodes_traversed=5,
            regenerate_attempts=0,
        )

    monkeypatch.setattr("memex.webui.app.answer_query", _fake)


@pytest.fixture
def fake_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake(question: str, **_kw: Any) -> FinalResponse:
        return FinalResponse(
            answered=False,
            refusal_reason="The vault doesn't contain material to answer this.",
            correlation_id="01HZTESTREFUSE000000000000",
            tokens_used=12,
            nodes_traversed=3,
            regenerate_attempts=0,
        )

    monkeypatch.setattr("memex.webui.app.answer_query", _fake)


# ----- Tests -----


def test_index_renders_ask_form(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "Ask the vault" in r.text
    assert 'name="question"' in r.text
    assert 'hx-post="/ask"' in r.text


def test_healthz_reports_vault_path(
    settings: MemexSettings, client: TestClient
) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "ok"
    assert payload["vault_path"] == str(settings.vault_path)


def test_ask_returns_answer_fragment(
    client: TestClient, fake_answered: None
) -> None:
    r = client.post("/ask", data={"question": "What does Smith argue?"})
    assert r.status_code == 200
    # Fragment, not full document — no <html> tag.
    assert "<html" not in r.text
    assert "Smith treats reflexivity" in r.text
    assert "Reflexivity shapes the data" in r.text
    assert "01HZTESTWEBUI" in r.text  # correlation id printed


def test_ask_renders_refusal(
    client: TestClient, fake_refused: None
) -> None:
    r = client.post("/ask", data={"question": "What is the etymology?"})
    assert r.status_code == 200
    assert "Refused" in r.text
    assert "vault doesn&#39;t contain" in r.text or "vault doesn't contain" in r.text


def test_ask_rejects_empty_question(
    client: TestClient, fake_answered: None
) -> None:
    r = client.post("/ask", data={"question": "   "})
    assert r.status_code == 400
    assert "Question is empty" in r.text


@pytest.mark.asyncio
async def test_documents_list_renders_each_doc(
    settings: MemexSettings, client: TestClient
) -> None:
    a = await ingest_markdown_passthrough("# A\n\nFirst.\n", source_stem="alpha")
    b = await ingest_markdown_passthrough("# B\n\nSecond.\n", source_stem="beta")

    r = client.get("/documents")
    assert r.status_code == 200
    assert a.doc_id in r.text
    assert b.doc_id in r.text


@pytest.mark.asyncio
async def test_document_detail_renders_body(
    settings: MemexSettings, client: TestClient
) -> None:
    ref = await ingest_markdown_passthrough(
        "# Reflexivity\n\nSome paragraph about it.\n",
        source_stem="reflexivity",
    )

    r = client.get(f"/documents/{ref.doc_id}")
    assert r.status_code == 200
    # The body is wrapped in <pre>; check substring rather than exact HTML.
    assert "Some paragraph about it" in r.text
    assert "Reflexivity" in r.text


def test_document_detail_404s_on_unknown_doc(client: TestClient) -> None:
    r = client.get("/documents/nonexistent-doc")
    assert r.status_code == 404


# ----- Phase 4: PDF source, edit/body partials, review, graph -----


@pytest.mark.asyncio
async def test_document_source_serves_pdf(
    settings: MemexSettings, client: TestClient
) -> None:
    ref = await ingest_markdown_passthrough(
        "# Reflexivity\n\nBody.\n", source_stem="reflexivity_src"
    )
    asset_dir = settings.vault_path / "documents" / ref.doc_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "source.pdf").write_bytes(b"%PDF-1.7\nfake pdf body\n%%EOF\n")

    r = client.get(f"/documents/{ref.doc_id}/source")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/pdf")
    assert r.content.startswith(b"%PDF-")


def test_document_source_404s_without_source(
    settings: MemexSettings, client: TestClient
) -> None:
    """Markdown-passthrough docs have no source file copied — the route
    must 404 cleanly rather than serve the markdown."""
    import asyncio

    async def _make():
        return await ingest_markdown_passthrough(
            "# Solo md\n\n.\n", source_stem="solo_md"
        )

    ref = asyncio.run(_make())
    r = client.get(f"/documents/{ref.doc_id}/source")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_document_view_renders_pane_split_when_pdf_present(
    settings: MemexSettings, client: TestClient
) -> None:
    ref = await ingest_markdown_passthrough(
        "# With source\n\nMarkdown body.\n", source_stem="with_source"
    )
    asset_dir = settings.vault_path / "documents" / ref.doc_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "source.pdf").write_bytes(b"%PDF-1.7\n%%EOF\n")

    r = client.get(f"/documents/{ref.doc_id}")
    assert r.status_code == 200
    assert "pane-split" in r.text
    assert f"/documents/{ref.doc_id}/source" in r.text


def test_document_view_renders_solo_when_no_source(
    settings: MemexSettings, client: TestClient
) -> None:
    import asyncio

    async def _make():
        return await ingest_markdown_passthrough(
            "# No source\n\nBody.\n", source_stem="no_src"
        )

    ref = asyncio.run(_make())
    r = client.get(f"/documents/{ref.doc_id}")
    assert r.status_code == 200
    assert "pane-solo" in r.text
    assert "pane-split" not in r.text


@pytest.mark.asyncio
async def test_document_edit_returns_textarea_partial(
    settings: MemexSettings, client: TestClient
) -> None:
    ref = await ingest_markdown_passthrough(
        "# Edit me\n\nOriginal body.\n", source_stem="edit_me"
    )
    r = client.get(f"/documents/{ref.doc_id}/edit")
    assert r.status_code == 200
    assert "<html" not in r.text  # fragment, not full document
    assert "<textarea" in r.text
    assert "Original body" in r.text
    assert f"/documents/{ref.doc_id}/review" in r.text


@pytest.mark.asyncio
async def test_document_body_returns_view_partial(
    settings: MemexSettings, client: TestClient
) -> None:
    ref = await ingest_markdown_passthrough(
        "# View\n\nBody.\n", source_stem="view_me"
    )
    r = client.get(f"/documents/{ref.doc_id}/body")
    assert r.status_code == 200
    assert "<html" not in r.text
    assert "<pre" in r.text
    # No "saved" toast on a plain GET.
    assert "saved-flash" not in r.text


@pytest.mark.asyncio
async def test_document_review_writes_and_returns_saved_partial(
    settings: MemexSettings, client: TestClient
) -> None:
    from memex.vault.store import read_document

    ref = await ingest_markdown_passthrough(
        "# Original\n\nFirst draft.\n", source_stem="review_me"
    )

    new_body = "# Updated\n\nSecond draft, hand-corrected.\n"
    r = client.post(
        f"/documents/{ref.doc_id}/review", data={"body": new_body}
    )
    assert r.status_code == 200
    assert "saved-flash" in r.text
    assert "Second draft" in r.text

    # Markdown on disk reflects the new body.
    refreshed = await read_document(settings.vault_path, ref.doc_id)
    assert "Second draft" in refreshed.body
    assert refreshed.ref.content_sha256 != ref.content_sha256


@pytest.mark.asyncio
async def test_document_review_updates_manifest_to_avoid_watcher_race(
    settings: MemexSettings, client: TestClient
) -> None:
    """The watcher's `_confirm_user_edit` compares on-disk sha to the
    manifest's `content_sha256` — matching means Memex wrote it. The
    review route must update the manifest so a watcher event fired by
    its own write is correctly swallowed."""
    from memex.core.manifest import read_manifest
    from memex.vault.store import read_document

    ref = await ingest_markdown_passthrough(
        "# Watch race\n\nA.\n", source_stem="watch_race"
    )
    new_body = "# Watch race\n\nA.\n\nAppended by the UI.\n"

    r = client.post(
        f"/documents/{ref.doc_id}/review", data={"body": new_body}
    )
    assert r.status_code == 200

    manifest = await read_manifest(settings.vault_path, ref.doc_id)
    assert manifest is not None
    on_disk = await read_document(settings.vault_path, ref.doc_id)
    assert manifest.content_sha256 == on_disk.ref.content_sha256


@pytest.mark.asyncio
async def test_graph_renders_with_inline_data(
    settings: MemexSettings,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """We don't need a real RyuGraph install — patch `GraphStore.open`
    to return an in-memory fake whose `.neighbors()` returns a fixed
    list. The graph page should embed that data in its
    `<script id="graph-data">` tag for Cytoscape to pick up."""
    from memex.index.graph_store import GraphNeighbor

    ref = await ingest_markdown_passthrough(
        "# Center\n\nThe centerpiece.\n", source_stem="center"
    )

    class _FakeStore:
        @classmethod
        async def open(cls, vault_path):
            return cls()

        async def neighbors(self, doc_id, limit=50):
            return [
                GraphNeighbor(
                    doc_id="abc12345-neighbor-a",
                    title="Neighbor A",
                    relation="shares_entity",
                    via="reflexivity",
                ),
                GraphNeighbor(
                    doc_id="def67890-neighbor-b",
                    title="Neighbor B",
                    relation="shares_entity",
                    via="methodology",
                ),
            ]

        async def close(self):
            return None

    monkeypatch.setattr(
        "memex.webui.app.GraphStore.open",
        staticmethod(_FakeStore.open),
    )

    r = client.get(f"/graph/{ref.doc_id}")
    assert r.status_code == 200
    assert ref.doc_id in r.text
    assert "abc12345-neighbor-a" in r.text
    assert "Neighbor A" in r.text
    assert "def67890-neighbor-b" in r.text
    assert "reflexivity" in r.text
    assert 'id="graph-data"' in r.text
    assert "cytoscape" in r.text  # CDN script reference


@pytest.mark.asyncio
async def test_graph_shows_unavailable_when_graph_store_missing(
    settings: MemexSettings,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `GraphStore.open` raises ImportError (ryugraph not installed),
    the page still renders — with an `unavailable` chip and an empty
    neighbours list."""
    ref = await ingest_markdown_passthrough(
        "# Solo\n\nNo graph backing.\n", source_stem="solo_graph"
    )

    async def _raise(vault_path):
        raise ImportError("ryugraph not installed")

    monkeypatch.setattr(
        "memex.webui.app.GraphStore.open",
        staticmethod(_raise),
    )

    r = client.get(f"/graph/{ref.doc_id}")
    assert r.status_code == 200
    assert "graph store unavailable" in r.text
    # The center node is still rendered.
    assert ref.doc_id in r.text


def test_graph_404s_on_unknown_doc(client: TestClient) -> None:
    r = client.get("/graph/nonexistent-doc")
    assert r.status_code == 404
