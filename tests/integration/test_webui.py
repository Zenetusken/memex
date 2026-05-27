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
            wikilinks=["[[d1#Reflexivity]]"],
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


def test_healthz_reports_vault_path(settings: MemexSettings, client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "ok"
    assert payload["vault_path"] == str(settings.vault_path)


def test_ask_returns_answer_fragment(client: TestClient, fake_answered: None) -> None:
    r = client.post("/ask", data={"question": "What does Smith argue?"})
    assert r.status_code == 200
    # Fragment, not full document — no <html> tag.
    assert "<html" not in r.text
    assert "Smith treats reflexivity" in r.text
    assert "Reflexivity shapes the data" in r.text
    assert "01HZTESTWEBUI" in r.text  # correlation id printed


def test_ask_answer_fragment_renders_sources_wikilinks(
    client: TestClient, fake_answered: None
) -> None:
    """P4.1: the answered partial lists each wikilink as an `<a>` to the
    cited doc/section."""
    r = client.post("/ask", data={"question": "What does Smith argue?"})
    assert r.status_code == 200
    assert "Sources" in r.text
    assert 'href="/documents/d1#reflexivity"' in r.text
    assert 'class="wikilink"' in r.text


def test_ask_renders_refusal(client: TestClient, fake_refused: None) -> None:
    r = client.post("/ask", data={"question": "What is the etymology?"})
    assert r.status_code == 200
    assert "Refused" in r.text
    assert "vault doesn&#39;t contain" in r.text or "vault doesn't contain" in r.text
    # P4.1: no Sources section on a refusal (wikilinks default to []).
    assert "Sources" not in r.text
    # #256: no scope note when retrieval wasn't re-scoped (artifact_scope_doc_ids=[]).
    assert "Scoped to" not in r.text


@pytest.mark.asyncio
async def test_ask_refusal_surfaces_artifact_scope_titles(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """#256 observability: a refusal caused by a re-scope shows which document(s)
    the query was narrowed to — by HUMAN TITLE (the doc-id is the hover tooltip +
    the link target), so the narrowing is auditable + readable."""
    fw = await ingest_markdown_passthrough(
        "# Firewall\n\nCoupe-feu architectures.\n", source_stem="CR350 Firewall Diagrams"
    )
    lec = await ingest_markdown_passthrough(
        "# Cours 6\n\nPare-feu et mandataires.\n", source_stem="CR350 Cours 6 Coupe-feu"
    )

    async def _fake(question: str, **_kw: Any) -> FinalResponse:
        return FinalResponse(
            answered=False,
            refusal_reason="The firewall documents have no VLAN address range.",
            artifact_scope_doc_ids=[fw.doc_id, lec.doc_id],
            correlation_id="01HZTESTSCOPE00000000000000",
            tokens_used=20,
            nodes_traversed=6,
            regenerate_attempts=0,
        )

    monkeypatch.setattr("memex.webui.app.answer_query", _fake)

    r = client.post("/ask", data={"question": "Quelle plage VLAN dans le diagramme de coupe-feu ?"})
    assert r.status_code == 200
    assert "Refused" in r.text
    assert "Scoped to the documents you named" in r.text
    # Human titles render (not the raw doc-ids as the visible text).
    assert "CR350 Firewall Diagrams" in r.text
    assert "CR350 Cours 6 Coupe-feu" in r.text
    # The doc-id is retained as the link target + hover tooltip (stable identifier).
    assert f'href="/documents/{fw.doc_id}"' in r.text
    assert f'title="{fw.doc_id}"' in r.text


@pytest.mark.asyncio
async def test_index_renders_doc_picker(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """The doc-picker: the landing page offers each vault document as a tickable
    scope (title shown, doc-id the checkbox value)."""
    lec = await ingest_markdown_passthrough(
        "# STP\n\nSpanning Tree Protocol.\n", source_stem="CR350 Cours 5 STP"
    )
    r = client.get("/")
    assert r.status_code == 200
    assert "Scope to documents" in r.text
    assert 'name="scope_doc_ids"' in r.text
    assert f'value="{lec.doc_id}"' in r.text
    assert "CR350 Cours 5 STP" in r.text  # offered by human title


@pytest.mark.asyncio
async def test_ask_scopes_to_selected_docs_and_labels_note(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Ticking docs forwards `scope_doc_ids` to the agent, and the answer panel's
    scope note reads 'your selected document(s)' (not 'you named')."""
    lec = await ingest_markdown_passthrough(
        "# Cours 5\n\nSpanning Tree.\n", source_stem="CR350 Cours 5 STP"
    )
    captured: dict[str, Any] = {}

    async def _fake(question: str, **kw: Any) -> FinalResponse:
        captured["scope_doc_ids"] = kw.get("scope_doc_ids")
        return FinalResponse(
            answered=True,
            summary="STP elects a root bridge by lowest bridge ID.",
            claims=[
                CitedClaim(
                    claim="The lowest BID wins.",
                    source_chunk_id=f"{lec.doc_id}#c1",
                    confidence="high",
                )
            ],
            artifact_scope_doc_ids=[lec.doc_id],
            correlation_id="01HZTESTPICKER0000000000000",
            tokens_used=15,
            nodes_traversed=6,
            regenerate_attempts=0,
        )

    monkeypatch.setattr("memex.webui.app.answer_query", _fake)

    r = client.post(
        "/ask",
        data={"question": "How is the root bridge elected?", "scope_doc_ids": [lec.doc_id]},
    )
    assert r.status_code == 200
    assert captured["scope_doc_ids"] == [lec.doc_id]  # the route forwarded the selection
    assert "Scoped to your selected document" in r.text  # picker phrasing, not "you named"
    assert "CR350 Cours 5 STP" in r.text  # scoped doc shown by title


# ----- saved scope sets (persist + reapply a selection) -----


@pytest.mark.asyncio
async def test_scope_set_save_apply_delete_round_trip(
    settings: MemexSettings, client: TestClient
) -> None:
    """Save the ticked docs as a named set; it shows on the landing page; applying
    it re-ticks its docs server-side; deleting it removes it."""
    a = await ingest_markdown_passthrough("# A\n\nAlpha.\n", source_stem="alpha doc")
    b = await ingest_markdown_passthrough("# B\n\nBeta.\n", source_stem="beta doc")

    r = client.post(
        "/scope-sets",
        data={"set_name": "My Set", "scope_doc_ids": [a.doc_id, b.doc_id]},
    )
    assert r.status_code == 200
    assert "Saved" in r.text and "My Set" in r.text

    # Persisted — shows on the landing page's saved-set bar.
    r = client.get("/")
    assert "Saved sets" in r.text
    assert "My Set" in r.text

    # Applying ticks both docs (case-insensitive name match).
    r = client.post("/scope-sets/apply", data={"name": "my set"})
    assert "Applied" in r.text
    assert f'value="{a.doc_id}" checked>' in r.text
    assert f'value="{b.doc_id}" checked>' in r.text

    # Deleting removes it.
    r = client.post("/scope-sets/delete", data={"name": "My Set"})
    assert "Deleted" in r.text
    r = client.get("/")
    assert "My Set" not in r.text


@pytest.mark.asyncio
async def test_scope_set_save_empty_name_flashes_error(
    settings: MemexSettings, client: TestClient
) -> None:
    a = await ingest_markdown_passthrough("# A\n\nAlpha.\n", source_stem="alpha")
    r = client.post("/scope-sets", data={"set_name": "   ", "scope_doc_ids": [a.doc_id]})
    assert r.status_code == 200
    assert "scope-flash-error" in r.text  # error flash, not a 500
    # Nothing persisted.
    assert "Saved sets" not in client.get("/").text


@pytest.mark.asyncio
async def test_scope_set_save_no_docs_flashes_error(
    settings: MemexSettings, client: TestClient
) -> None:
    r = client.post("/scope-sets", data={"set_name": "Empty", "scope_doc_ids": []})
    assert r.status_code == 200
    assert "scope-flash-error" in r.text


@pytest.mark.asyncio
async def test_scope_set_apply_unknown_flashes_error(
    settings: MemexSettings, client: TestClient
) -> None:
    await ingest_markdown_passthrough("# A\n\nAlpha.\n", source_stem="alpha")
    r = client.post("/scope-sets/apply", data={"name": "nope"})
    assert r.status_code == 200
    assert "No saved set" in r.text
    assert "checked>" not in r.text  # nothing ticked


def test_ask_rejects_empty_question(client: TestClient, fake_answered: None) -> None:
    r = client.post("/ask", data={"question": "   "})
    assert r.status_code == 400
    assert "Question is empty" in r.text


@pytest.mark.asyncio
async def test_documents_list_renders_each_doc(settings: MemexSettings, client: TestClient) -> None:
    a = await ingest_markdown_passthrough("# A\n\nFirst.\n", source_stem="alpha")
    b = await ingest_markdown_passthrough("# B\n\nSecond.\n", source_stem="beta")

    r = client.get("/documents")
    assert r.status_code == 200
    assert a.doc_id in r.text
    assert b.doc_id in r.text


@pytest.mark.asyncio
async def test_document_detail_renders_body(settings: MemexSettings, client: TestClient) -> None:
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
async def test_document_source_serves_pdf(settings: MemexSettings, client: TestClient) -> None:
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


def test_document_source_404s_without_source(settings: MemexSettings, client: TestClient) -> None:
    """Markdown-passthrough docs have no source file copied — the route
    must 404 cleanly rather than serve the markdown."""
    import asyncio

    async def _make():
        return await ingest_markdown_passthrough("# Solo md\n\n.\n", source_stem="solo_md")

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
        return await ingest_markdown_passthrough("# No source\n\nBody.\n", source_stem="no_src")

    ref = asyncio.run(_make())
    r = client.get(f"/documents/{ref.doc_id}")
    assert r.status_code == 200
    assert "pane-solo" in r.text
    assert "pane-split" not in r.text


@pytest.mark.asyncio
async def test_document_edit_returns_textarea_partial(
    settings: MemexSettings, client: TestClient
) -> None:
    ref = await ingest_markdown_passthrough("# Edit me\n\nOriginal body.\n", source_stem="edit_me")
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
    ref = await ingest_markdown_passthrough("# View\n\nBody.\n", source_stem="view_me")
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

    ref = await ingest_markdown_passthrough("# Original\n\nFirst draft.\n", source_stem="review_me")

    new_body = "# Updated\n\nSecond draft, hand-corrected.\n"
    r = client.post(
        f"/documents/{ref.doc_id}/review",
        data={"body": new_body, "expected_sha": ref.content_sha256},
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

    ref = await ingest_markdown_passthrough("# Watch race\n\nA.\n", source_stem="watch_race")
    new_body = "# Watch race\n\nA.\n\nAppended by the UI.\n"

    r = client.post(
        f"/documents/{ref.doc_id}/review",
        data={"body": new_body, "expected_sha": ref.content_sha256},
    )
    assert r.status_code == 200

    manifest = await read_manifest(settings.vault_path, ref.doc_id)
    assert manifest is not None
    on_disk = await read_document(settings.vault_path, ref.doc_id)
    assert manifest.content_sha256 == on_disk.ref.content_sha256


@pytest.mark.asyncio
async def test_document_review_returns_409_conflict_on_stale_sha(
    settings: MemexSettings, client: TestClient
) -> None:
    """The user submits an edit based on an out-of-date sha. The server
    returns 409 with the conflict panel (diff + overwrite/discard buttons)
    and the on-disk file is unchanged.
    """
    from memex.vault.store import read_document

    ref = await ingest_markdown_passthrough(
        "# Conflict\n\nOriginal body.\n", source_stem="conflict"
    )
    # Out-of-band: the vault changes after the form loaded.
    out_of_band_body = "# Conflict\n\nSomeone else edited this.\n"
    r1 = client.post(
        f"/documents/{ref.doc_id}/review",
        data={"body": out_of_band_body, "expected_sha": ref.content_sha256},
    )
    assert r1.status_code == 200
    after_first = await read_document(settings.vault_path, ref.doc_id)

    # User submits based on the original (now-stale) sha.
    user_draft = "# Conflict\n\nMy local edit.\n"
    r2 = client.post(
        f"/documents/{ref.doc_id}/review",
        data={"body": user_draft, "expected_sha": ref.content_sha256},
    )
    assert r2.status_code == 409
    assert "Conflict" in r2.text
    assert "overwrite anyway" in r2.text.lower()
    assert "discard mine" in r2.text.lower()
    # The diff section should mention "your draft" vs "current".
    assert "your draft" in r2.text.lower()
    # On-disk content unchanged from the out-of-band write.
    after_conflict = await read_document(settings.vault_path, ref.doc_id)
    assert after_conflict.ref.content_sha256 == after_first.ref.content_sha256
    assert "Someone else edited this" in after_conflict.body
    assert "My local edit" not in after_conflict.body


@pytest.mark.asyncio
async def test_document_review_overwrite_anyway_succeeds(
    settings: MemexSettings, client: TestClient
) -> None:
    """After a 409, re-submitting with the CURRENT sha (which the conflict
    panel's "overwrite anyway" form does) succeeds and the user's draft
    lands.
    """
    from memex.vault.store import read_document

    ref = await ingest_markdown_passthrough(
        "# Overwrite\n\nStart.\n", source_stem="overwrite_anyway"
    )
    out_of_band_body = "# Overwrite\n\nIntervening write.\n"
    client.post(
        f"/documents/{ref.doc_id}/review",
        data={"body": out_of_band_body, "expected_sha": ref.content_sha256},
    )
    after_first = await read_document(settings.vault_path, ref.doc_id)
    current_sha = after_first.ref.content_sha256

    user_draft = "# Overwrite\n\nMy draft wins.\n"
    # "Overwrite anyway": resubmit with the *current* sha as expected.
    r = client.post(
        f"/documents/{ref.doc_id}/review",
        data={"body": user_draft, "expected_sha": current_sha},
    )
    assert r.status_code == 200
    assert "saved-flash" in r.text
    after = await read_document(settings.vault_path, ref.doc_id)
    assert "My draft wins" in after.body
    assert "Intervening write" not in after.body


@pytest.mark.asyncio
async def test_document_edit_form_includes_hidden_expected_sha(
    settings: MemexSettings, client: TestClient
) -> None:
    """The GET /documents/{id}/edit partial must include the hidden
    `expected_sha` input so the form submission can carry it back.
    """
    ref = await ingest_markdown_passthrough("# Hidden\n\nbody.\n", source_stem="hidden_sha_input")
    r = client.get(f"/documents/{ref.doc_id}/edit")
    assert r.status_code == 200
    assert 'name="expected_sha"' in r.text
    assert ref.content_sha256 in r.text


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

    ref = await ingest_markdown_passthrough("# Center\n\nThe centerpiece.\n", source_stem="center")

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


# ----- Inline title rename (metadata-only retitle) -----


@pytest.mark.asyncio
async def test_title_edit_renders_form_with_current_title(
    settings: MemexSettings, client: TestClient
) -> None:
    ref = await ingest_markdown_passthrough("# Doc\n\nBody.\n", source_stem="rename_me")
    r = client.get(f"/documents/{ref.doc_id}/title/edit")
    assert r.status_code == 200
    # The form carries the current title and posts back to .../title.
    assert 'name="title"' in r.text
    assert f"/documents/{ref.doc_id}/title" in r.text


@pytest.mark.asyncio
async def test_title_save_renames_and_persists(settings: MemexSettings, client: TestClient) -> None:
    from memex.vault.store import read_document

    ref = await ingest_markdown_passthrough("# Doc\n\nBody.\n", source_stem="rename_me")
    r = client.post(
        f"/documents/{ref.doc_id}/title",
        data={"title": "A Clean Human Title"},
    )
    assert r.status_code == 200
    # The returned view partial shows the new title + a rename trigger.
    assert "A Clean Human Title" in r.text
    assert "rename" in r.text
    # Frontmatter on disk reflects the new title.
    refreshed = await read_document(settings.vault_path, ref.doc_id)
    assert refreshed.frontmatter.title == "A Clean Human Title"


@pytest.mark.asyncio
async def test_title_save_blank_is_noop(settings: MemexSettings, client: TestClient) -> None:
    from memex.vault.store import read_document

    ref = await ingest_markdown_passthrough("# Doc\n\nBody.\n", source_stem="keep_title")
    before = (await read_document(settings.vault_path, ref.doc_id)).frontmatter.title
    r = client.post(f"/documents/{ref.doc_id}/title", data={"title": "   "})
    assert r.status_code == 200
    after = (await read_document(settings.vault_path, ref.doc_id)).frontmatter.title
    assert after == before


def test_title_save_404s_on_unknown_doc(client: TestClient) -> None:
    r = client.post("/documents/nonexistent-doc/title", data={"title": "X"})
    assert r.status_code == 404


# ----- co-residence resource mode UI (ADR-0007) -----


def test_header_shows_mode_chip(client: TestClient) -> None:
    # Default mode is "manual"; the chip renders on every page.
    body = client.get("/").text
    assert "mode-chip" in body
    assert "Manual" in body  # the active label


def test_resources_page_compares_all_modes(client: TestClient) -> None:
    r = client.get("/resources")
    assert r.status_code == 200
    assert "Active mode" in r.text
    for mode_id in ("fast", "full", "gpu_only"):
        assert mode_id in r.text  # every mode listed
    # the live per-mode Apply switch is wired (replaced the old `memex mode set` cmd)
    assert 'hx-post="/resources/mode"' in r.text
    assert "mode-apply" in r.text


def test_resources_page_highlights_active_curated_mode(
    tmp_vault: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With a curated mode active, its row is highlighted + the chip shows it.
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_vault))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("MEMEX_MODELS__CO_RESIDENCE_MODE", "full")
    set_settings(MemexSettings())  # type: ignore[call-arg]
    try:
        client = TestClient(create_app())
        r = client.get("/resources")
        assert r.status_code == 200
        assert "mode-row-active" in r.text  # the `full` row is the active one
        assert "~24,576 tokens (whole document)" in r.text
        assert "Full context" in client.get("/").text  # chip label
    finally:
        set_settings(None)


# ── live mode hot-switch (ADR-0007) — POST /resources/mode ──


class _FakeState:
    def __init__(self, *, alive: bool = True, reachable: bool = True) -> None:
        self.alive = alive
        self.reachable = reachable


class _FakeRegistry:
    """Records which models were unloaded (the app-side device swap)."""

    def __init__(self) -> None:
        self.unloaded: list[str] = []

    async def unload(self, name: str) -> None:
        self.unloaded.append(name)


def _fake_daemon(monkeypatch: pytest.MonkeyPatch, restarts: list[dict[str, Any]]) -> _FakeRegistry:
    reg = _FakeRegistry()

    async def _status(_s: Any) -> _FakeState:
        return _FakeState(alive=True)

    async def _restart(_s: Any, *, gpu_fraction: Any = None, max_model_len: Any = None) -> _FakeState:
        restarts.append({"gpu_fraction": gpu_fraction, "max_model_len": max_model_len})
        return _FakeState(reachable=True)

    monkeypatch.setattr("memex.webui.app.daemon_status", _status)
    monkeypatch.setattr("memex.webui.app.daemon_restart", _restart)
    monkeypatch.setattr("memex.webui.app.get_registry", lambda: reg)
    return reg


def test_resources_mode_hot_switch_restarts_and_swaps(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    restarts: list[dict[str, Any]] = []
    reg = _fake_daemon(monkeypatch, restarts)

    r = client.post("/resources/mode", data={"mode": "full"})

    assert r.status_code == 200
    assert "Switched to Full context" in r.text  # flash + new active label rendered
    assert settings.models.co_residence_mode == "full"  # settings mutated (registry shares it)
    assert reg.unloaded == ["embedder", "reranker"]  # retrieval models dropped for reload
    assert restarts == [{"gpu_fraction": 0.8, "max_model_len": 24576}]  # full posture applied


def test_resources_mode_manual_skips_daemon_restart(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    restarts: list[dict[str, Any]] = []
    reg = _fake_daemon(monkeypatch, restarts)

    r = client.post("/resources/mode", data={"mode": "manual"})

    assert r.status_code == 200
    assert restarts == []  # manual prescribes no orchestrator posture → no restart
    assert reg.unloaded == ["embedder", "reranker"]  # but the device swap still happens
    assert settings.models.co_residence_mode == "manual"


def test_resources_mode_unknown_flashes_error(client: TestClient) -> None:
    r = client.post("/resources/mode", data={"mode": "turbo"})
    assert r.status_code == 400
    assert "Unknown mode" in r.text
