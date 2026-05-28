"""FastAPI web UI — Phase 3 regression net.

Drives the routes via Starlette's `TestClient`. The answering agent is
faked; the rest of the stack is real (Jinja templates render, the vault
is a real tmp dir, the document list iterates real files).
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport

from memex.agents.answering import (
    CitedClaim,
    FinalResponse,
    reset_compiled_graph,
)
from memex.core.config import MemexSettings, set_settings
from memex.core.types import Chunk
from memex.ingest.pipeline import ingest_markdown_passthrough
from memex.parse.pdf_render import PDFPreviewError
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


async def _ask_to_completion(app: Any, question: str, **form: Any) -> str:
    """POST /ask, then long-poll /ask/{cid}/status (via httpx on the test's event
    loop, so the background agent task actually runs) until the answer/expired
    fragment replaces the progress fragment. Returns the final HTML — mirrors the
    browser's long-poll chain end-to-end. (A sync TestClient can't drive the
    background task between polls; the shared-loop AsyncClient can.)"""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        r = await ac.post("/ask", data={"question": question, **form})
        assert r.status_code == 200, r.text
        text = r.text
        m = re.search(r"/ask/([^/?\"]+)/status\?v=(\d+)", text)
        assert m is not None, f"POST /ask did not return a progress fragment: {text[:300]}"
        cid, v = m.group(1), int(m.group(2))
        for _ in range(100):
            await asyncio.sleep(0.01)
            r = await ac.get(f"/ask/{cid}/status?v={v}")
            assert r.status_code == 200
            text = r.text
            if 'class="progress"' not in text:
                return text  # answer / expired fragment — no poll trigger, chain stops
            mv = re.search(r"\?v=(\d+)", text)
            if mv is not None:
                v = int(mv.group(1))
        raise AssertionError(f"ask did not complete after polling: {text[:300]}")


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


@pytest.mark.asyncio
async def test_ask_returns_answer_fragment(client: TestClient, fake_answered: None) -> None:
    text = await _ask_to_completion(client.app, "What does Smith argue?")
    # Fragment, not full document — no <html> tag.
    assert "<html" not in text
    assert "Smith treats reflexivity" in text
    assert "Reflexivity shapes the data" in text
    assert "01HZTESTWEBUI" in text  # correlation id printed


@pytest.mark.asyncio
async def test_ask_answer_fragment_renders_sources_wikilinks(
    client: TestClient, fake_answered: None
) -> None:
    """P4.1: the answered partial lists each wikilink as an `<a>` to the
    cited doc/section."""
    text = await _ask_to_completion(client.app, "What does Smith argue?")
    assert "Sources" in text
    assert 'href="/documents/d1#reflexivity"' in text
    assert 'class="wikilink"' in text


@pytest.mark.asyncio
async def test_ask_renders_sources_and_claims_by_title(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#2/#3: a claim's source chip + the Sources list render the human document
    TITLE (› section), linked to the doc section, with the raw docid#hash only as
    the hover tooltip — not as the visible label."""

    async def _fake(question: str, **_kw: Any) -> FinalResponse:
        chunk = Chunk(
            chunk_id="abc12345#h1",
            document_id="abc12345",
            document_title="CS Notes",
            text="C++ was developed by Stroustrup.",
            heading_path=["C++ Tutorials", "History"],
        )
        return FinalResponse(
            answered=True,
            summary="C++ history.",
            claims=[
                CitedClaim(
                    claim="Developed by Stroustrup.",
                    source_chunk_id="abc12345#h1",
                    confidence="high",
                )
            ],
            used_chunks=[chunk],
            wikilinks=["[[abc12345#History]]"],
            correlation_id="01HZTITLE0000000000000000",
            tokens_used=10,
            nodes_traversed=4,
            regenerate_attempts=0,
        )

    monkeypatch.setattr("memex.webui.app.answer_query", _fake)
    text = await _ask_to_completion(client.app, "who made C++?")
    # claim source chip → "Title › Section", linked to the doc section
    assert 'class="claim-source-link"' in text
    assert "CS Notes › History" in text
    assert 'href="/documents/abc12345#history"' in text
    assert 'title="abc12345#h1"' in text  # raw id only as the tooltip
    assert ">abc12345#h1<" not in text  # NOT shown as a visible <code> chip


@pytest.mark.asyncio
async def test_ask_renders_refusal(client: TestClient, fake_refused: None) -> None:
    text = await _ask_to_completion(client.app, "What is the etymology?")
    assert "Refused" in text
    assert "vault doesn&#39;t contain" in text or "vault doesn't contain" in text
    # P4.1: no Sources section on a refusal (wikilinks default to []).
    assert "Sources" not in text
    # #256: no scope note when retrieval wasn't re-scoped (artifact_scope_doc_ids=[]).
    assert "Scoped to" not in text


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

    text = await _ask_to_completion(
        client.app, "Quelle plage VLAN dans le diagramme de coupe-feu ?"
    )
    assert "Refused" in text
    assert "Scoped to the documents you named" in text
    # Human titles render (not the raw doc-ids as the visible text).
    assert "CR350 Firewall Diagrams" in text
    assert "CR350 Cours 6 Coupe-feu" in text
    # The doc-id is retained as the link target + hover tooltip (stable identifier).
    assert f'href="/documents/{fw.doc_id}"' in text
    assert f'title="{fw.doc_id}"' in text


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

    text = await _ask_to_completion(
        client.app, "How is the root bridge elected?", scope_doc_ids=[lec.doc_id]
    )
    assert captured["scope_doc_ids"] == [lec.doc_id]  # the route forwarded the selection
    assert "Scoped to your selected document" in text  # picker phrasing, not "you named"
    assert "CR350 Cours 5 STP" in text  # scoped doc shown by title


# ----- live progress indicator (long-poll) -----


def test_ask_post_returns_progress_fragment(client: TestClient, fake_answered: None) -> None:
    """POST /ask returns IMMEDIATELY with the self-polling progress fragment (not
    the answer): a status-poll URL + the step list, no <html>."""
    r = client.post("/ask", data={"question": "anything"})
    assert r.status_code == 200
    assert "<html" not in r.text
    assert 'class="progress"' in r.text
    assert "/ask/" in r.text and "/status?v=" in r.text
    assert 'hx-trigger="load"' in r.text
    assert "Retrieving" in r.text  # the first step renders


def test_ask_status_progress_then_done_and_evicts(client: TestClient) -> None:
    """Status route: a running entry → the progress fragment (still polling); a
    finished entry → the answer (no poll trigger); a second poll → expired (the
    entry is evicted on delivery)."""
    registry = client.app.state.progress
    cid = "01HZSTATUSTEST0000000000000"
    registry.new(cid, scope_doc_ids=[], scope_source="named")
    registry.set_phase(cid, "Grounding")  # version → 1
    # Running: poll with v=0 (< current version) → immediate progress fragment.
    r = client.get(f"/ask/{cid}/status?v=0")
    assert r.status_code == 200
    assert 'class="progress"' in r.text
    assert "Grounding" in r.text
    assert "progress-step-active" in r.text
    # Finish → the answer fragment (no poll trigger → polling stops).
    fr = FinalResponse(
        answered=True,
        summary="The composed answer.",
        claims=[],
        correlation_id="cidX",
        tokens_used=1,
        nodes_traversed=1,
        regenerate_attempts=0,
    )
    registry.finish(cid, response=fr)
    r = client.get(f"/ask/{cid}/status?v=0")
    assert r.status_code == 200
    assert "The composed answer." in r.text
    assert 'class="progress"' not in r.text
    # Evicted on delivery → a second poll is "Expired".
    r = client.get(f"/ask/{cid}/status?v=0")
    assert r.status_code == 200
    assert "Expired" in r.text
    assert 'class="progress"' not in r.text


def test_ask_status_done_error_renders_banner_at_200(client: TestClient) -> None:
    """A finished-with-error entry renders the error banner as content at HTTP 200
    (not a 4xx/5xx — the long-poll outcome is decoupled from the HTTP status)."""
    registry = client.app.state.progress
    cid = "01HZERRTEST00000000000000000"
    registry.new(cid, scope_doc_ids=[], scope_source="named")
    registry.finish(cid, error="Couldn't answer: ModelCallError. boom")
    r = client.get(f"/ask/{cid}/status?v=0")
    assert r.status_code == 200
    assert "Couldn&#39;t answer" in r.text or "Couldn't answer" in r.text
    assert 'class="progress"' not in r.text


def test_ask_status_unknown_cid_is_expired(client: TestClient) -> None:
    r = client.get("/ask/01HZNOPE0000000000000000000/status?v=0")
    assert r.status_code == 200
    assert "Expired" in r.text
    assert 'class="progress"' not in r.text


@pytest.mark.asyncio
async def test_ask_live_progression_surfaces_phase(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end on a shared loop: the agent's `on_node` calls flow into the
    registry, so a poll observes a mid-run phase BEFORE the answer arrives. A gate
    makes it deterministic — the fake holds at "Grounding" until the test sees it."""
    gate = asyncio.Event()

    async def _fake(question: str, *, on_node: Any = None, **_kw: Any) -> FinalResponse:
        if on_node is not None:
            on_node("verify")  # maps to the "Grounding" phase
        await gate.wait()  # hold here until the test has observed "Grounding"
        return FinalResponse(
            answered=True,
            summary="The grounded answer.",
            claims=[],
            correlation_id="cidL",
            tokens_used=3,
            nodes_traversed=4,
            regenerate_attempts=0,
        )

    monkeypatch.setattr("memex.webui.app.answer_query", _fake)
    async with httpx.AsyncClient(
        transport=ASGITransport(app=client.app), base_url="http://t"
    ) as ac:
        r = await ac.post("/ask", data={"question": "go"})
        m = re.search(r"/ask/([^/?\"]+)/status\?v=(\d+)", r.text)
        assert m is not None
        cid, v = m.group(1), int(m.group(2))
        # Poll until a frame shows "Grounding" active (the held phase).
        saw_grounding = False
        for _ in range(100):
            await asyncio.sleep(0.005)
            r = await ac.get(f"/ask/{cid}/status?v={v}")
            if (
                'class="progress"' in r.text
                and "Grounding" in r.text
                and "progress-step-active" in r.text
            ):
                saw_grounding = True
                break
            mv = re.search(r"\?v=(\d+)", r.text)
            if mv is not None:
                v = int(mv.group(1))
        assert saw_grounding, "never observed the live 'Grounding' phase"
        # Release the agent → the next poll returns the answer; polling stops.
        gate.set()
        final = ""
        for _ in range(100):
            await asyncio.sleep(0.005)
            r = await ac.get(f"/ask/{cid}/status?v={v}")
            if 'class="progress"' not in r.text:
                final = r.text
                break
            mv = re.search(r"\?v=(\d+)", r.text)
            if mv is not None:
                v = int(mv.group(1))
        assert "The grounded answer." in final


# ----- summarize progress (long-poll, mirrors /ask) -----


def _summary_fr(cid: str = "cidS") -> FinalResponse:
    return FinalResponse(
        answered=True,
        summary="A grounded summary.",
        claims=[],
        sections=[],
        correlation_id=cid,
        tokens_used=5,
        nodes_traversed=2,
        regenerate_attempts=0,
    )


def test_summarize_post_returns_progress_fragment(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /summarize returns the progress fragment (the SUMMARY step list + the
    summarize status poll URL), not the summary."""

    async def _fake(doc_id: str, **_kw: Any) -> FinalResponse:
        return _summary_fr()

    monkeypatch.setattr("memex.webui.app.summarize_document", _fake)
    r = client.post("/documents/abcd1234/summarize", data={"detail": "standard"})
    assert r.status_code == 200
    assert "<html" not in r.text
    assert 'class="progress"' in r.text
    assert "/documents/abcd1234/summarize/status?cid=" in r.text
    assert "Summarizing" in r.text  # the SUMMARY_PHASES step list
    assert "Retrieving" not in r.text  # NOT the /ask phases


def test_summarize_status_progress_then_done_and_evicts(client: TestClient) -> None:
    registry = client.app.state.progress
    cid = "01HZSUMMARYTEST000000000000"
    registry.new(cid, scope_doc_ids=[], scope_source="named")
    registry.set_phase(cid, "Summarizing · section 2 of 5")  # version → 1
    r = client.get(f"/documents/abcd1234/summarize/status?cid={cid}&v=0")
    assert r.status_code == 200
    assert 'class="progress"' in r.text
    assert "Summarizing" in r.text
    assert "section 2 of 5" in r.text  # the eyebrow detail (the section counter)
    assert "progress-step-active" in r.text
    # Finish → the summary fragment (no poll trigger).
    registry.finish(cid, response=_summary_fr())
    r = client.get(f"/documents/abcd1234/summarize/status?cid={cid}&v=0")
    assert r.status_code == 200
    assert "A grounded summary." in r.text
    assert 'class="progress"' not in r.text
    # Evicted on delivery → expired.
    r = client.get(f"/documents/abcd1234/summarize/status?cid={cid}&v=0")
    assert "Expired" in r.text


def test_summarize_status_done_error_renders_banner_at_200(client: TestClient) -> None:
    registry = client.app.state.progress
    cid = "01HZSUMMARYERR0000000000000"
    registry.new(cid, scope_doc_ids=[], scope_source="named")
    registry.finish(cid, error="Couldn't summarise: ModelCallError. boom")
    r = client.get(f"/documents/abcd1234/summarize/status?cid={cid}&v=0")
    assert r.status_code == 200
    assert "Couldn&#39;t summarise" in r.text or "Couldn't summarise" in r.text
    assert 'class="progress"' not in r.text


@pytest.mark.asyncio
async def test_summarize_live_progression_surfaces_section(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: the summarizer's on_phase("Summarizing · section 1 of 2") flows
    into a poll frame (active + the counter), then the summary swaps in."""
    gate = asyncio.Event()

    async def _fake(doc_id: str, *, on_phase: Any = None, **_kw: Any) -> FinalResponse:
        if on_phase is not None:
            on_phase("Summarizing · section 1 of 2")
        await gate.wait()
        return _summary_fr()

    monkeypatch.setattr("memex.webui.app.summarize_document", _fake)
    async with httpx.AsyncClient(
        transport=ASGITransport(app=client.app), base_url="http://t"
    ) as ac:
        r = await ac.post("/documents/abcd1234/summarize", data={"detail": "standard"})
        text = r.text
        saw = False
        for _ in range(100):
            m = re.search(r'hx-get="([^"]+)"', text)
            assert m is not None
            await asyncio.sleep(0.005)
            text = (await ac.get(m.group(1))).text
            if "section 1 of 2" in text and "progress-step-active" in text:
                saw = True
                break
        assert saw, "never observed the live 'section 1 of 2' phase"
        gate.set()
        final = ""
        for _ in range(100):
            m = re.search(r'hx-get="([^"]+)"', text)
            if m is None:
                final = text
                break
            await asyncio.sleep(0.005)
            text = (await ac.get(m.group(1))).text
        assert "A grounded summary." in final


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
    # Served INLINE (not "attachment") so it can render in a browser; the
    # template's download link forces a download via the HTML attribute.
    assert "inline" in r.headers.get("content-disposition", "")


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
    settings: MemexSettings, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = await ingest_markdown_passthrough(
        "# With source\n\nMarkdown body.\n", source_stem="with_source"
    )
    asset_dir = settings.vault_path / "documents" / ref.doc_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "source.pdf").write_bytes(b"%PDF-1.7\n%%EOF\n")
    # The pane wiring is what's under test; fake the page count so pypdfium2
    # need not parse the stub bytes (the real render is covered by the page route).
    monkeypatch.setattr("memex.webui.app.pdf_page_count", lambda _p: 2)

    r = client.get(f"/documents/{ref.doc_id}")
    assert r.status_code == 200
    assert "pane-split" in r.text
    # Server-rendered page images (one <img> per page), NOT an <iframe> embed.
    assert f"/documents/{ref.doc_id}/source/page/0" in r.text
    assert f"/documents/{ref.doc_id}/source/page/1" in r.text
    assert "<iframe" not in r.text
    assert f'/documents/{ref.doc_id}/source"' in r.text  # download link → original


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
async def test_document_source_page_renders_png(
    settings: MemexSettings, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The preview pane's per-page route rasterises the page to a PNG (server
    side) — so the page renders inline regardless of the browser's PDF setting."""
    ref = await ingest_markdown_passthrough("# Pg\n\n.\n", source_stem="pg_src")
    asset_dir = settings.vault_path / "documents" / ref.doc_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "source.pdf").write_bytes(b"%PDF-1.7\n%%EOF\n")
    monkeypatch.setattr(
        "memex.webui.app.render_pdf_page_png", lambda _p, _n: b"\x89PNG\r\n\x1a\nDATA"
    )

    r = client.get(f"/documents/{ref.doc_id}/source/page/0")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content.startswith(b"\x89PNG")


@pytest.mark.asyncio
async def test_document_source_page_out_of_range_404s(
    settings: MemexSettings, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = await ingest_markdown_passthrough("# Pg2\n\n.\n", source_stem="pg2_src")
    asset_dir = settings.vault_path / "documents" / ref.doc_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "source.pdf").write_bytes(b"%PDF-1.7\n%%EOF\n")

    def _boom(_p: object, _n: object) -> bytes:
        raise PDFPreviewError("page index out of range", context={})

    monkeypatch.setattr("memex.webui.app.render_pdf_page_png", _boom)
    r = client.get(f"/documents/{ref.doc_id}/source/page/99")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_office_doc_preview_renders_converted_pdf(
    settings: MemexSettings, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An Office doc (`source.pptx`) has no `source.pdf`, but the parse stage left
    a `converted.pdf` — the preview pane renders THAT (labelled "(rendered)"),
    while the download link still points at the original `.pptx`."""
    ref = await ingest_markdown_passthrough("# Deck\n\nSlides.\n", source_stem="deck_src")
    asset_dir = settings.vault_path / "documents" / ref.doc_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "source.pptx").write_bytes(b"PK\x03\x04 fake pptx")
    (asset_dir / "converted.pdf").write_bytes(b"%PDF-1.7\n%%EOF\n")
    monkeypatch.setattr("memex.webui.app.pdf_page_count", lambda _p: 1)

    r = client.get(f"/documents/{ref.doc_id}")
    assert r.status_code == 200
    assert "pane-split" in r.text
    assert f"/documents/{ref.doc_id}/source/page/0" in r.text  # renders converted.pdf
    assert "(rendered)" in r.text  # the pane is labelled a render of the pptx
    assert "download pptx" in r.text


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

    async def _restart(
        _s: Any, *, gpu_fraction: Any = None, max_model_len: Any = None
    ) -> _FakeState:
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
