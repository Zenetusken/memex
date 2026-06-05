"""FastAPI web UI — Phase 3 regression net.

Drives the routes via Starlette's `TestClient`. The answering agent is
faked; the rest of the stack is real (Jinja templates render, the vault
is a real tmp dir, the document list iterates real files).
"""

from __future__ import annotations

import asyncio
import re
import tempfile
from collections.abc import Callable, Iterator
from datetime import UTC
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
from memex.daemon.supervisor import DaemonStatus
from memex.ingest.pipeline import ingest_markdown_passthrough
from memex.parse.pdf_render import PDFPreviewError
from memex.webui.app import create_app
from memex.webui.ingest_driver import IngestOutcome


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
async def test_ask_source_link_carries_page_when_chunk_attributed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the chunker attributed a chunk to a source PDF page (`Chunk.page`
    populated), the claim's source-link emits `data-page="N"` + the URL gains
    `?page=N` + the visible label gains "· p. N" — fueling the doc-page's
    preview-pane jump-to-page UX. Existing chunks (no page attribution) keep
    the previous behavior, pinned by the test above."""

    async def _fake(question: str, **_kw: Any) -> FinalResponse:
        chunk = Chunk(
            chunk_id="abc12345#p6",
            document_id="abc12345",
            document_title="Module 1",
            text="LED indicators table.",
            page=6,
            heading_path=["Switch LED Indicators (Cont.)"],
        )
        return FinalResponse(
            answered=True,
            summary="LED indicators.",
            claims=[
                CitedClaim(
                    claim="STAT amber means port-blocked.",
                    source_chunk_id="abc12345#p6",
                    confidence="high",
                )
            ],
            used_chunks=[chunk],
            wikilinks=[],
            correlation_id="01HZPAGE0000000000000000",
            tokens_used=10,
            nodes_traversed=4,
            regenerate_attempts=0,
        )

    monkeypatch.setattr("memex.webui.app.answer_query", _fake)
    text = await _ask_to_completion(client.app, "what does amber STAT mean?")
    # The link carries the page hint three ways: the URL param (cross-page
    # navigation), the data-page attribute (same-page JS scroll), and the
    # visible label (so the reader can see WHICH page they're being sent to).
    assert 'data-page="6"' in text
    assert 'href="/documents/abc12345?page=6#switch-led-indicators-cont' in text
    assert "· p. 6" in text


@pytest.mark.asyncio
async def test_ask_source_link_carries_time_anchor_for_audio_chunk(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An audio transcript chunk (ADR-0017, `Chunk.time_range` populated) gets a
    `· mm:ss` time chip on its source link — the audio analogue of the `· p. N`
    page chip. A non-audio chunk (no time_range) shows no chip (pinned above)."""

    async def _fake(question: str, **_kw: Any) -> FinalResponse:
        chunk = Chunk(
            chunk_id="lecture1#t1",
            document_id="lecture1",
            document_title="Lecture 1",
            text="VLANs segment the broadcast domain.",
            heading_path=["[01:02]"],
            time_range=(62.0, 66.0),
        )
        return FinalResponse(
            answered=True,
            summary="VLANs segment the network.",
            claims=[
                CitedClaim(
                    claim="A VLAN segments the broadcast domain.",
                    source_chunk_id="lecture1#t1",
                    confidence="high",
                )
            ],
            used_chunks=[chunk],
            wikilinks=[],
            correlation_id="01HZTIME0000000000000000",
            tokens_used=10,
            nodes_traversed=4,
            regenerate_attempts=0,
        )

    monkeypatch.setattr("memex.webui.app.answer_query", _fake)
    text = await _ask_to_completion(client.app, "what does a VLAN do?")
    # The transcript chunk's start time (62.0s) renders as a readable mm:ss chip;
    # no page chip (audio has no pages).
    assert "Lecture 1 · 1:02" in text
    assert "· p." not in text
    # The section heading IS the timestamp `[01:02]` for a transcript chunk, so the
    # redundant `› [01:02]` segment is suppressed in favour of the normalized chip
    # (F1: no `Lecture 1 › [01:02] · 1:02` duplication).
    assert "[01:02]" not in text


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


def test_report_summary_renders_paragraphs_in_one_block(client: TestClient) -> None:
    """A `report`-detail summary (ADR-0010) is a multi-paragraph body (blank-line
    separated). The done-render splits it into one <p> per paragraph inside ONE
    .ans-answer block — one blue rule spans them, NOT N separate answer blocks."""
    registry = client.app.state.progress
    cid = "01HZSUMMARYREPORT0000000000"
    registry.new(cid, scope_doc_ids=[], scope_source="named")
    body = "First paragraph of the report.\n\nSecond paragraph.\n\nThird paragraph."
    fr = FinalResponse(
        answered=True,
        summary=body,
        claims=[],
        sections=[],
        correlation_id=cid,
        tokens_used=5,
        nodes_traversed=2,
        regenerate_attempts=0,
    )
    registry.finish(cid, response=fr)
    r = client.get(f"/documents/abcd1234/summarize/status?cid={cid}&v=0")
    assert r.status_code == 200
    # Each paragraph is its OWN <p> (the \n\n was split, not collapsed into one run).
    assert "<p>First paragraph of the report.</p>" in r.text
    assert "<p>Second paragraph.</p>" in r.text
    assert "<p>Third paragraph.</p>" in r.text
    # Exactly ONE .ans-answer block wraps all three (a single blue rule).
    assert r.text.count('class="ans-answer"') == 1


def test_report_summary_surfaces_faithfulness_confidence(client: TestClient) -> None:
    """A report summary surfaces its INFORM-ONLY faithfulness confidence (ADR-0010):
    the overall score + the embedding/lexical breakdown, quietly under the footer."""
    from memex.agents.answering import ReportConfidence

    registry = client.app.state.progress
    cid = "01HZSUMMARYCONF00000000000"
    registry.new(cid, scope_doc_ids=[], scope_source="named")
    fr = FinalResponse(
        answered=True,
        summary="Para one.\n\nPara two.",
        claims=[],
        sections=[],
        report_confidence=ReportConfidence(
            overall=0.78, embedding=0.81, lexical=0.73, per_paragraph=[0.80, 0.76]
        ),
        correlation_id=cid,
        tokens_used=5,
        nodes_traversed=2,
        regenerate_attempts=0,
    )
    registry.finish(cid, response=fr)
    r = client.get(f"/documents/abcd1234/summarize/status?cid={cid}&v=0")
    assert r.status_code == 200
    assert "faithfulness" in r.text
    assert "0.78" in r.text  # overall
    assert "embedding" in r.text and "0.81" in r.text
    assert "lexical" in r.text and "0.73" in r.text


@pytest.mark.asyncio
async def test_document_detail_select_offers_report(
    settings: MemexSettings, client: TestClient
) -> None:
    """The Summarize control exposes the `report` detail (ADR-0010) alongside the
    existing levels."""
    ref = await ingest_markdown_passthrough(
        "# Summary Control\n\nBody.\n", source_stem="summary_control"
    )
    r = client.get(f"/documents/{ref.doc_id}")
    assert r.status_code == 200
    assert '<option value="report">report</option>' in r.text
    assert '<option value="detailed">detailed</option>' in r.text


def test_summary_labels_sources_by_section_not_repeated_doc_title(client: TestClient) -> None:
    """Item 5: a summary is of ONE doc, so the per-claim source chips label by
    SECTION (+ page) — NOT "DocTitle › Section" repeated on every claim — and
    there's no redundant "Sources" wikilink list re-listing the same doc once
    per section. (The answer view keeps the full title › section + Sources.)"""
    registry = client.app.state.progress
    cid = "01HZSUMMARYSRC0000000000000"
    registry.new(cid, scope_doc_ids=[], scope_source="named")
    chunk = Chunk(
        chunk_id="abcd1234#k",
        document_id="abcd1234",
        document_title="Access Control Exercise",
        text="Key components.",
        page=2,
        heading_path=["Key Components"],
    )
    registry.finish(
        cid,
        response=FinalResponse(
            answered=True,
            summary="A grounded summary.",
            claims=[
                CitedClaim(
                    claim="R1 and R2 are routers.", source_chunk_id="abcd1234#k", confidence="high"
                )
            ],
            used_chunks=[chunk],
            wikilinks=["[[abcd1234#Key Components]]"],
            correlation_id=cid,
            tokens_used=5,
            nodes_traversed=2,
            regenerate_attempts=0,
        ),
    )
    r = client.get(f"/documents/abcd1234/summarize/status?cid={cid}&v=0")
    assert r.status_code == 200
    # Section-first chip: "Key Components · p. 2", NOT "Access Control Exercise › …".
    assert "Key Components · p. 2" in r.text
    assert "Access Control Exercise › Key Components" not in r.text
    # No redundant Sources section (every source is this one doc).
    assert "Sources" not in r.text


def test_summary_source_chip_shows_time_anchor_for_audio_chunk(client: TestClient) -> None:
    """A summary of an audio transcript labels each key-point source by its `· mm:ss`
    time chip (F3 coverage, ADR-0017). The redundant `[00:30]` heading-as-section is
    suppressed in favour of the normalized chip — the summary analogue of the `/ask`
    suppression."""
    registry = client.app.state.progress
    cid = "01HZSUMMARYTIME000000000000"
    registry.new(cid, scope_doc_ids=[], scope_source="named")
    chunk = Chunk(
        chunk_id="lecture2#t0",
        document_id="lecture2",
        document_title="Lecture 2",
        text="Routers forward packets between networks.",
        heading_path=["[00:30]"],
        time_range=(30.0, 35.0),
    )
    registry.finish(
        cid,
        response=FinalResponse(
            answered=True,
            summary="A grounded transcript summary.",
            claims=[
                CitedClaim(
                    claim="A router forwards packets between networks.",
                    source_chunk_id="lecture2#t0",
                    confidence="high",
                )
            ],
            used_chunks=[chunk],
            wikilinks=[],
            correlation_id=cid,
            tokens_used=5,
            nodes_traversed=2,
            regenerate_attempts=0,
        ),
    )
    r = client.get(f"/documents/lecture2/summarize/status?cid={cid}&v=0")
    assert r.status_code == 200
    assert "· 0:30" in r.text  # the time chip
    assert "[00:30]" not in r.text  # the redundant bracketed section is suppressed
    assert "· p." not in r.text  # audio has no page chip


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


# ----- Scope-set suggestions ("docs related to your selection", ADR-0011) -----


class _SuggestFake:
    """A fake GraphStore whose related_documents returns one fixed suggestion regardless of
    seed — for the scope-picker suggestion tests."""

    @classmethod
    async def open(cls, vault_path: object) -> _SuggestFake:
        return cls()

    async def related_documents(self, doc_id: str, *, limit: int = 10, max_entities: int = 8):
        from memex.index.graph_store import RelatedDocument

        return [
            RelatedDocument(
                doc_id="rel-doc-1",
                title="Related Suggestion",
                score=5.0,
                shared_entities=["DNS spoofing"],
            )
        ]

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_scope_apply_auto_suggests_related(
    settings: MemexSettings, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Applying a saved set auto-surfaces "Suggested additions" — docs the graph relates to
    the set — each tick-able + with `/entity?name=` why-related tags."""
    a = await ingest_markdown_passthrough("# A\n\nAlpha.\n", source_stem="alpha s")
    b = await ingest_markdown_passthrough("# B\n\nBeta.\n", source_stem="beta s")
    monkeypatch.setattr("memex.webui.app.GraphStore.open", staticmethod(_SuggestFake.open))
    client.post("/scope-sets", data={"set_name": "S", "scope_doc_ids": [a.doc_id, b.doc_id]})
    r = client.post("/scope-sets/apply", data={"name": "S"})
    assert r.status_code == 200
    assert "Suggested additions" in r.text
    assert "Related Suggestion" in r.text  # the graph-related doc
    assert "/entity?name=DNS" in r.text  # the why-related entity is a traversal link


@pytest.mark.asyncio
async def test_scope_suggest_button_from_ticked_docs(
    settings: MemexSettings, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The "Suggest related" button (POST /scope-sets/suggest) surfaces suggestions for the
    posted ticks + a count flash."""
    a = await ingest_markdown_passthrough("# A\n\nAlpha.\n", source_stem="alpha t")
    monkeypatch.setattr("memex.webui.app.GraphStore.open", staticmethod(_SuggestFake.open))
    r = client.post("/scope-sets/suggest", data={"scope_doc_ids": [a.doc_id]})
    assert r.status_code == 200
    assert "Suggested additions" in r.text
    assert "Related Suggestion" in r.text
    assert "related document" in r.text  # the count flash


@pytest.mark.asyncio
async def test_scope_suggest_empty_selection_flashes(
    settings: MemexSettings, client: TestClient
) -> None:
    """Suggest with NO ticks → a hint flash, no suggestions section, no 500."""
    await ingest_markdown_passthrough("# A\n\nAlpha.\n", source_stem="alpha e")
    r = client.post("/scope-sets/suggest", data={})
    assert r.status_code == 200
    assert "Suggested additions" not in r.text
    assert "Tick one or more documents" in r.text


@pytest.mark.asyncio
async def test_scope_suggest_survives_graph_unavailable(
    settings: MemexSettings, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Graph absent (ImportError) → no suggestions, the picker still renders (fail-open)."""
    a = await ingest_markdown_passthrough("# A\n\nAlpha.\n", source_stem="alpha u")

    def _boom(vault_path: object) -> object:
        raise ImportError("ryugraph not installed")

    monkeypatch.setattr("memex.webui.app.GraphStore.open", staticmethod(_boom))
    r = client.post("/scope-sets/suggest", data={"scope_doc_ids": [a.doc_id]})
    assert r.status_code == 200
    assert "Suggested additions" not in r.text
    assert "No related documents" in r.text


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
    # The pane wiring is what's under test; fake the page count + size so
    # pypdfium2 need not parse the stub bytes (the real render is covered by the
    # page route). `pdf_page_size` returns a 16:9 deck size to also pin that the
    # aspect-ratio CSS var is emitted into the container.
    monkeypatch.setattr("memex.webui.app.pdf_page_count", lambda _p: 2)
    monkeypatch.setattr("memex.webui.app.pdf_page_size", lambda _p, _n=0: (1280.0, 720.0))

    r = client.get(f"/documents/{ref.doc_id}")
    assert r.status_code == 200
    assert "pane-split" in r.text
    # Server-rendered page images (one <img> per page), NOT an <iframe> embed.
    assert f"/documents/{ref.doc_id}/source/page/0" in r.text
    assert f"/documents/{ref.doc_id}/source/page/1" in r.text
    assert "<iframe" not in r.text
    assert f'/documents/{ref.doc_id}/source"' in r.text  # download link → original
    # Aspect-ratio var feeds the placeholder height so `loading="lazy"` actually
    # defers offscreen pages — without it every row reads as in-viewport and
    # all pages render on initial load (the stale-cache foot-gun we just hit).
    assert "--pdf-page-aspect: 1280.000 / 720.000" in r.text
    # Each preview <img> carries `id="page-{1-based}"` so a claim's
    # `data-page="N"` can `scrollIntoView` to it (the click-source→jump-to-PDF
    # UX). The inline `<script>` is gated on `has_preview` so it only ships
    # when a preview exists.
    assert 'id="page-1"' in r.text
    assert 'id="page-2"' in r.text
    assert "scrollPreviewTo" in r.text  # the inline script's hook


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
    monkeypatch.setattr("memex.webui.app.pdf_page_size", lambda _p, _n=0: (1280.0, 720.0))

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


def _fake_graph_store():
    """A fake GraphStore for the /graph view tests: a fixed two-doc neighbourhood bridged by
    a shared 'methodology' entity (+ 'reflexivity' on one doc). Provides BOTH lenses the route
    calls — `related_documents` (document lens) and `related_bridges` (concept lens)."""
    from memex.index.graph_store import BridgeDoc, DocumentBridge, RelatedDocument

    class _FakeStore:
        @classmethod
        async def open(cls, vault_path):
            return cls()

        async def related_documents(self, doc_id, *, limit=10, max_entities=8):
            return [
                RelatedDocument(
                    doc_id="abc12345-neighbor-a",
                    title="Neighbor A",
                    score=3.9,
                    shared_entities=["reflexivity", "methodology"],
                ),
                RelatedDocument(
                    doc_id="def67890-neighbor-b",
                    title="Neighbor B",
                    score=1.2,
                    shared_entities=["methodology"],
                ),
            ]

        async def related_bridges(
            self, doc_id, *, limit_bridges=24, max_docs_per_bridge=50, max_via=5
        ):
            return [
                DocumentBridge(
                    entity="methodology",
                    kind="concept",
                    doc_count=2,
                    strength=4.0,
                    docs=[
                        BridgeDoc(
                            doc_id="abc12345-neighbor-a",
                            title="Neighbor A",
                            score=3.9,
                            via_entities=["reflexivity"],
                        ),
                        BridgeDoc(
                            doc_id="def67890-neighbor-b",
                            title="Neighbor B",
                            score=1.2,
                            via_entities=[],
                        ),
                    ],
                ),
                DocumentBridge(
                    entity="reflexivity",
                    kind="concept",
                    doc_count=1,
                    strength=2.5,
                    docs=[
                        BridgeDoc(
                            doc_id="abc12345-neighbor-a",
                            title="Neighbor A",
                            score=3.9,
                            via_entities=["methodology"],
                        )
                    ],
                ),
            ]

        async def close(self):
            return None

    return _FakeStore


@pytest.mark.asyncio
async def test_graph_renders_bridges_view(
    settings: MemexSettings,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The redesigned /graph view is server-rendered (NO Cytoscape) — the concept lens groups
    related docs under their bridging ENTITY, each entity a `/entity?name=` traversal link and
    each doc a `/documents/` link. The single-doc bridge ('reflexivity') folds into the tail."""
    ref = await ingest_markdown_passthrough("# Center\n\nThe centerpiece.\n", source_stem="center")
    monkeypatch.setattr("memex.webui.app.GraphStore.open", staticmethod(_fake_graph_store().open))

    r = client.get(f"/graph/{ref.doc_id}")
    assert r.status_code == 200
    assert ref.doc_id in r.text
    # the bridging entity heading + its reached docs
    assert "methodology" in r.text
    assert "Neighbor A" in r.text and "abc12345-neighbor-a" in r.text
    assert "Neighbor B" in r.text and "def67890-neighbor-b" in r.text
    assert "bridges 2" in r.text  # the doc_count badge on the multi-doc bridge
    # entities are TRAVERSAL links into the entity-centric view (the "why")
    assert "/entity?name=methodology" in r.text
    # the single-doc bridge folds into the tail
    assert "single-doc bridge" in r.text
    # the Cytoscape hairball is GONE — no canvas, no inline graph-data, no vendored lib
    assert 'id="graph-data"' not in r.text
    assert "cytoscape" not in r.text.lower()
    # the lens toggle is present, concept active by default
    assert "?group=document" in r.text and "?group=concept" in r.text


@pytest.mark.asyncio
async def test_graph_document_lens_renders_ranked_list(
    settings: MemexSettings,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`?group=document` renders the flat strength-ranked neighbour list (the alternate lens),
    each row a doc link + its connecting entities."""
    ref = await ingest_markdown_passthrough("# Center\n\nThe centerpiece.\n", source_stem="center2")
    monkeypatch.setattr("memex.webui.app.GraphStore.open", staticmethod(_fake_graph_store().open))

    r = client.get(f"/graph/{ref.doc_id}?group=document")
    assert r.status_code == 200
    assert "Neighbor A" in r.text and "Neighbor B" in r.text
    assert "/documents/abc12345-neighbor-a" in r.text
    assert "/entity?name=reflexivity" in r.text  # connecting entity as a traversal link
    assert "cytoscape" not in r.text.lower()


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


@pytest.mark.asyncio
async def test_document_view_renders_related_documents(
    settings: MemexSettings,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The doc view shows the 'Related documents' section (explore connections),
    fed by GraphStore.related_documents — by human title + the connecting entities."""
    from memex.index.graph_store import RelatedDocument

    ref = await ingest_markdown_passthrough("# Center\n\nThe centerpiece.\n", source_stem="center")

    class _FakeStore:
        @classmethod
        async def open(cls, vault_path):
            return cls()

        async def related_documents(self, doc_id, *, limit=10, max_entities=8):
            return [
                RelatedDocument(
                    doc_id="abc12345-sibling-lecture",
                    title="Sibling Lecture",
                    score=3.91,
                    shared_entities=["DNS spoofing", "stateful firewall"],
                )
            ]

        async def citations(self, doc_id):
            from memex.index.graph_store import CitationLink, DocumentCitations

            return DocumentCitations(
                cites=[
                    CitationLink(
                        doc_id="ref99-cited-lecture", title="Cited Lecture", surface_text="Cours 4"
                    )
                ],
                cited_by=[],
            )

        async def close(self):
            return None

    monkeypatch.setattr("memex.webui.app.GraphStore.open", staticmethod(_FakeStore.open))

    r = client.get(f"/documents/{ref.doc_id}")
    assert r.status_code == 200
    assert "Related documents" in r.text
    assert "Sibling Lecture" in r.text
    assert "/documents/abc12345-sibling-lecture" in r.text
    assert "DNS spoofing" in r.text  # the connecting entity (the "why related")
    # The connecting entity is a LINK into the entity-centric view (the entry point).
    assert "/entity?name=DNS" in r.text
    # The 1-hop CITES "References" section reads the previously write-only CITES edges.
    assert "References" in r.text
    assert "Cited Lecture" in r.text
    assert "/documents/ref99-cited-lecture" in r.text
    assert "Cours 4" in r.text


@pytest.mark.asyncio
async def test_document_view_survives_graph_unavailable(
    settings: MemexSettings,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Related is OPTIONAL: an ImportError from GraphStore.open (ryugraph absent)
    must NOT 500 the doc view — it just omits the section."""
    ref = await ingest_markdown_passthrough("# Solo\n\nNo graph.\n", source_stem="solo")

    def _boom(vault_path):
        raise ImportError("ryugraph not installed")

    monkeypatch.setattr("memex.webui.app.GraphStore.open", staticmethod(_boom))

    r = client.get(f"/documents/{ref.doc_id}")
    assert r.status_code == 200
    assert "Related documents" not in r.text  # section omitted, page still renders


# ----- /ask "Related documents" discovery panel (ADR-0011) -----


def _answered_two_docs() -> FinalResponse:
    """An answer that CITES two docs (aaaa1111 + bbbb2222) but surfaces only aaaa1111 in its
    claim/wikilink — so bbbb2222 appears on the page ONLY if the related panel wrongly keeps
    it (it's a cited doc → must be excluded)."""
    return FinalResponse(
        answered=True,
        summary="Two-source answer.",
        claims=[CitedClaim(claim="A fact.", source_chunk_id="aaaa1111#h1", confidence="high")],
        used_chunks=[
            Chunk(chunk_id="aaaa1111#h1", document_id="aaaa1111", document_title="Doc A", text="a"),
            Chunk(chunk_id="bbbb2222#h1", document_id="bbbb2222", document_title="Doc B", text="b"),
        ],
        wikilinks=["[[aaaa1111#History]]"],
        correlation_id="01HZRELATED000000000000000",
        tokens_used=10,
        nodes_traversed=4,
        regenerate_attempts=0,
    )


class _RelatedFake:
    """A fake GraphStore whose related_documents returns per-seed lists — one of which
    includes a CITED doc (bbbb2222) to prove the exclusion."""

    @classmethod
    async def open(cls, vault_path: object) -> _RelatedFake:
        return cls()

    async def related_documents(self, doc_id: str, *, limit: int = 10, max_entities: int = 8):
        from memex.index.graph_store import RelatedDocument

        if doc_id == "aaaa1111":
            return [
                RelatedDocument(
                    doc_id="cccc3333-sibling",
                    title="Sibling C",
                    score=3.9,
                    shared_entities=["DNS spoofing"],
                ),
                RelatedDocument(
                    doc_id="bbbb2222", title="Doc B", score=2.0, shared_entities=["x"]
                ),  # CITED → excluded
            ]
        if doc_id == "bbbb2222":
            return [
                RelatedDocument(
                    doc_id="cccc3333-sibling",
                    title="Sibling C",
                    score=4.5,
                    shared_entities=["firewall"],
                ),
                RelatedDocument(
                    doc_id="dddd4444-other", title="Doc D", score=1.0, shared_entities=["y"]
                ),
            ]
        return []

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_ask_renders_related_panel_excluding_cited_docs(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The answered /ask shows a "Related documents" panel — entity-graph neighbours of the
    cited docs (merged across them), each a doc link + `/entity?name=` connecting-entity
    tags — and EXCLUDES the docs the answer itself cited."""

    async def _fake(question: str, **_kw: Any) -> FinalResponse:
        return _answered_two_docs()

    monkeypatch.setattr("memex.webui.app.answer_query", _fake)
    monkeypatch.setattr("memex.webui.app.GraphStore.open", staticmethod(_RelatedFake.open))

    text = await _ask_to_completion(client.app, "a two-source question?")
    assert "Related documents" in text
    assert "Sibling C" in text  # the merged neighbour
    assert "/documents/cccc3333-sibling" in text
    assert "/documents/dddd4444-other" in text
    # dedup keeps the HIGHER-score relation (bbbb2222's cccc3333 @4.5, entity "firewall"),
    # and its connecting entity is a `/entity?name=` traversal link.
    assert "/entity?name=firewall" in text
    # bbbb2222 is a CITED doc (in used_chunks, not surfaced via claim/wikilink) → the panel
    # must NOT list it as a related doc.
    assert "/documents/bbbb2222" not in text


@pytest.mark.asyncio
async def test_ask_related_panel_survives_graph_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Graph absent (ImportError) → no panel, the answer still renders (fail-open)."""

    async def _fake(question: str, **_kw: Any) -> FinalResponse:
        return _answered_two_docs()

    def _boom(vault_path: object) -> object:
        raise ImportError("ryugraph not installed")

    monkeypatch.setattr("memex.webui.app.answer_query", _fake)
    monkeypatch.setattr("memex.webui.app.GraphStore.open", staticmethod(_boom))

    text = await _ask_to_completion(client.app, "a question?")
    assert "Two-source answer." in text  # the answer rendered
    assert "Related documents" not in text  # panel omitted, no 500


@pytest.mark.asyncio
async def test_ask_refusal_has_no_related_panel(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal gets no related panel — `_related_for_answer` is answered-only (a refusal's
    chunks were retrieved, not 'used')."""

    async def _fake(question: str, **_kw: Any) -> FinalResponse:
        return FinalResponse(
            answered=False,
            refusal_reason="Not in the corpus.",
            used_chunks=[
                Chunk(
                    chunk_id="aaaa1111#h1", document_id="aaaa1111", document_title="Doc A", text="a"
                ),
            ],
            correlation_id="01HZREFUSE0000000000000000",
            tokens_used=5,
            nodes_traversed=3,
            regenerate_attempts=0,
        )

    # GraphStore.open should never be reached on the answered-only path; make it loud if it is.
    def _boom(vault_path: object) -> object:
        raise AssertionError("related must not be computed on a refusal")

    monkeypatch.setattr("memex.webui.app.answer_query", _fake)
    monkeypatch.setattr("memex.webui.app.GraphStore.open", staticmethod(_boom))

    text = await _ask_to_completion(client.app, "a counterfactual?")
    assert "Refused" in text
    assert "Related documents" not in text


# ----- Entity-centric discovery view (/entity, ADR-0011) -----


def _entity_overview_resolved() -> object:
    from memex.index.graph_store import (
        CoOccurringEntity,
        EntityMention,
        EntityProfile,
        EntitySuggestion,
    )
    from memex.retrieve import EntityOverview

    return EntityOverview(
        profile=EntityProfile(
            query_name="DNS",
            matched_names=["DNS"],
            kinds=["concept", "tool"],
            doc_count=2,
            mentions=[
                EntityMention(doc_id="aaaa1111-cours-3", title="CR350 — Cours 3"),
                EntityMention(doc_id="bbbb2222-cours-6", title="CR350 — Cours 6"),
            ],
            cooccurring=[
                CoOccurringEntity(name="TCP", kind="concept", shared_docs=2, score=12.3),
                CoOccurringEntity(name="DHCP", kind="concept", shared_docs=1, score=5.1),
            ],
            resolved=True,
            suggestions=[
                EntitySuggestion(
                    name="Domain Name System", kind="concept", doc_count=3, relation="expansion"
                )
            ],
        ),
        passages=[
            Chunk(
                chunk_id="aaaa1111-cours-3#abc",
                document_id="aaaa1111-cours-3",
                document_title="CR350 — Cours 3",
                text="DNS resolves names to addresses.",
                heading_path=["Services", "DNS"],
            )
        ],
        passages_scoped=True,
    )


@pytest.mark.asyncio
async def test_entity_view_renders_resolved_profile(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resolved entity renders identity (kinds + doc_count), the co-occurring
    neighbourhood (each a /entity traversal link), the mentioning docs, and scoped
    passages by human title › section."""

    async def _fake(name: str, **_kw: object) -> object:
        assert name == "DNS"
        return _entity_overview_resolved()

    monkeypatch.setattr("memex.webui.app.entity_overview", _fake)
    r = client.get("/entity", params={"name": "DNS"})
    assert r.status_code == 200
    assert "in graph" in r.text  # the resolved badge
    assert "mentioned in 2 documents" in r.text
    assert "Co-occurring concepts" in r.text
    assert "TCP" in r.text
    assert "/entity?name=TCP" in r.text  # the co-entity is a traversal link
    assert "CR350 — Cours 3" in r.text  # a mentioning doc
    assert "/documents/aaaa1111-cours-3" in r.text
    assert "DNS resolves names to addresses." in r.text  # a scoped passage
    assert "from the 2 mentioning documents" in r.text  # passages_scoped note
    # The acronym↔expansion bridge surfaces as an "Also see" traversal link.
    assert "Also see" in r.text
    assert "Domain Name System" in r.text
    assert "/entity?name=Domain" in r.text  # urlencoded link into the expansion's profile


@pytest.mark.asyncio
async def test_entity_view_renders_did_you_mean_when_unresolved(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unresolved name that HAS a bridge surfaces 'Did you mean?' + the link."""
    from memex.index.graph_store import EntityProfile, EntitySuggestion
    from memex.retrieve import EntityOverview

    async def _fake(name: str, **_kw: object) -> object:
        return EntityOverview(
            profile=EntityProfile(
                query_name=name,
                matched_names=[],
                kinds=[],
                doc_count=0,
                mentions=[],
                cooccurring=[],
                resolved=False,
                suggestions=[
                    EntitySuggestion(
                        name="Domain Name System", kind="concept", doc_count=3, relation="acronym"
                    )
                ],
            ),
            passages=[Chunk(chunk_id="z#1", document_id="z", document_title="Z", text="…")],
            passages_scoped=False,
        )

    monkeypatch.setattr("memex.webui.app.entity_overview", _fake)
    r = client.get("/entity", params={"name": "DNS"})
    assert r.status_code == 200
    assert "not a known entity" in r.text
    assert "Did you mean" in r.text
    assert "/entity?name=Domain" in r.text


@pytest.mark.asyncio
async def test_entity_unknown_with_no_bridge_stays_honest(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The STP UI invariant: an unresolved name with NO bridge shows the honest
    'not a known entity' note and NO 'Did you mean?' block (no fabricated suggestion)."""
    from memex.index.graph_store import EntityProfile
    from memex.retrieve import EntityOverview

    async def _fake(name: str, **_kw: object) -> object:
        return EntityOverview(
            profile=EntityProfile(
                query_name=name,
                matched_names=[],
                kinds=[],
                doc_count=0,
                mentions=[],
                cooccurring=[],
                resolved=False,
                suggestions=[],
            ),
            passages=[
                Chunk(
                    chunk_id="z#1",
                    document_id="z",
                    document_title="Z",
                    text="Spanning Tree Protocol prevents loops.",
                )
            ],
            passages_scoped=False,
        )

    monkeypatch.setattr("memex.webui.app.entity_overview", _fake)
    r = client.get("/entity", params={"name": "STP"})
    assert r.status_code == 200
    assert "not a known entity" in r.text
    assert "Did you mean" not in r.text  # no fabricated bridge
    assert "Spanning Tree Protocol prevents loops." in r.text  # the honest FTS fallback


@pytest.mark.asyncio
async def test_entity_view_unknown_falls_back_to_fts(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unknown name → the honest whole-corpus FTS fallback (resolved=False),
    NOT a pretend profile."""
    from memex.index.graph_store import EntityProfile
    from memex.retrieve import EntityOverview

    async def _fake(name: str, **_kw: object) -> object:
        return EntityOverview(
            profile=EntityProfile(
                query_name=name,
                matched_names=[],
                kinds=[],
                doc_count=0,
                mentions=[],
                cooccurring=[],
                resolved=False,
            ),
            passages=[
                Chunk(
                    chunk_id="z#1",
                    document_id="zzzz9999-mod-5",
                    document_title="Module 5",
                    text="Spanning Tree Protocol prevents loops.",
                    heading_path=["STP"],
                )
            ],
            passages_scoped=False,
        )

    monkeypatch.setattr("memex.webui.app.entity_overview", _fake)
    r = client.get("/entity", params={"name": "STP"})
    assert r.status_code == 200
    assert "not a known entity" in r.text
    assert "full-text search across the vault" in r.text  # the unscoped passages note
    assert "Spanning Tree Protocol prevents loops." in r.text
    assert "Co-occurring concepts" not in r.text  # no graph neighbourhood on a miss


def test_entity_lookup_form_renders_without_name(client: TestClient) -> None:
    """GET /entity with no name → just the lookup form (no profile, never errors)."""
    r = client.get("/entity")
    assert r.status_code == 200
    assert 'name="name"' in r.text  # the lookup input
    assert "in graph" not in r.text
    assert "Co-occurring concepts" not in r.text


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
    # Default mode is now "auto" (the dynamic VRAM manager); the chip renders on every page.
    body = client.get("/").text
    assert "mode-chip" in body
    assert "Auto" in body  # the active label


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
        # full surfaces its concrete leverage: a 24,576 window + deeper retrieval.
        assert "24,576 tokens" in r.text
        assert "top-18 chunks" in r.text
        assert "Full context" in client.get("/").text  # chip label
    finally:
        set_settings(None)


def test_resources_renders_vram_panel(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # The /resources page surfaces a live GPU-memory panel: total/used/free + the per-process
    # holder breakdown (orchestrator vs this web UI vs other) + the auto-mode placement rationale.
    import os

    from memex.core.types import GpuProcess

    monkeypatch.setattr("memex.core.vram.total_vram_gb", lambda: 12.0)
    monkeypatch.setattr("memex.core.vram.free_vram_gb", lambda: 4.0)
    monkeypatch.setattr(
        "memex.core.vram.gpu_processes",
        lambda: [
            GpuProcess(pid=99001, name="VLLM::EngineCore", used_mib=6090),
            GpuProcess(pid=os.getpid(), name="python3", used_mib=2026),
            GpuProcess(pid=99002, name="Xorg", used_mib=300),
        ],
    )
    r = client.get("/resources")
    assert r.status_code == 200
    assert "GPU memory" in r.text
    assert "<b>8.0</b> GB used" in r.text  # 12.0 total − 4.0 free
    assert "<b>4.0</b> GB free" in r.text
    assert "12.0 GB total" in r.text
    assert "Orchestrator (vLLM)" in r.text  # the vLLM holder labelled
    assert "This web UI" in r.text  # the in-process holder labelled (pid == os.getpid())
    assert "vram-seg-orchestrator" in r.text  # the bar-segment swatch
    # auto + reranker-on-GPU (4.0 GB free ≥ 2.0 GB floor) → the placement rationale is shown
    assert "Reranker on the GPU" in r.text
    assert "2.0 GB floor" in r.text


def test_resources_vram_panel_unavailable_fallback(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Off-GPU / no-CUDA: the probe returns None → a clear fallback message, NOT a blank figure
    # (the old `{% if free_vram_gb is not none %}` guard rendered "<b> GB</b>" on an Undefined var).
    monkeypatch.setattr("memex.core.vram.total_vram_gb", lambda: None)
    monkeypatch.setattr("memex.core.vram.free_vram_gb", lambda: None)
    r = client.get("/resources")
    assert r.status_code == 200
    assert "VRAM probe unavailable" in r.text
    assert "GB used" not in r.text  # no figures rendered when the probe is unavailable


def test_resources_vram_fragment_auto_refreshes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GET /resources/vram is the HTMX auto-refresh target — it returns JUST the panel partial,
    # which re-arms its own `every 5s` trigger so the cycle is self-sustaining. Read-only.
    from memex.core.types import GpuProcess

    monkeypatch.setattr("memex.core.vram.total_vram_gb", lambda: 12.0)
    monkeypatch.setattr("memex.core.vram.free_vram_gb", lambda: 5.0)
    monkeypatch.setattr(
        "memex.core.vram.gpu_processes",
        lambda: [GpuProcess(pid=99001, name="VLLM::EngineCore", used_mib=6090)],
    )
    r = client.get("/resources/vram")
    assert r.status_code == 200
    assert "GPU memory" in r.text
    assert "<b>5.0</b> GB free" in r.text  # live figure
    assert "Orchestrator (vLLM)" in r.text
    # the returned fragment carries its own poll trigger (the self-sustaining refresh loop)
    assert 'hx-get="/resources/vram"' in r.text
    assert 'hx-trigger="every 5s"' in r.text
    # …and the full page embeds the same auto-refreshing panel
    assert 'hx-trigger="every 5s"' in client.get("/resources").text


def test_resources_table_surfaces_manual_mode(client: TestClient) -> None:
    # `manual` (the escape hatch — pin to the explicit device knobs) is now a selectable table row,
    # even though `all_modes()` excludes it (no fixed profile). Applying it skips the daemon restart.
    r = client.get("/resources")
    assert r.status_code == 200
    assert '<code class="mode-table-id">manual</code>' in r.text  # the manual row is present
    assert '{"mode": "manual"}' in r.text  # …with an apply button wired to switch to it


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


def test_resources_mode_switch_rejected_during_ingest(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    # A mode switch restarts the orchestrator + unloads models — reject it during an ingest so it
    # can't race the ingest's exclusive-GPU pause (the 3-vLLM mode-switch-during-ingest race, B4).
    restarts: list[dict[str, Any]] = []
    _fake_daemon(monkeypatch, restarts)
    client.app.state.ingesting.active = True
    try:
        r = client.post("/resources/mode", data={"mode": "fast"})
    finally:
        client.app.state.ingesting.active = False
    assert r.status_code == 409
    assert "being ingested" in r.text
    assert restarts == []  # the switch was rejected BEFORE any daemon restart


async def test_ingest_rejected_during_mode_switch(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The REVERSE of the mode-switch-during-ingest guard: an upload started WHILE a resource-mode
    # switch holds mode_switch_lock (its ~40s daemon restart) is rejected — so the two
    # GPU-orchestrating ops are mutually exclusive BOTH ways (B4b, the reviewer's catch).
    app = create_app()
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        async with app.state.mode_switch_lock:  # a mode-switch is mid-flight
            r = await ac.post(
                "/ingest",
                files={"file": ("x.pdf", b"%PDF data", "application/octet-stream")},
            )
    assert "resource-mode switch is finishing" in r.text
    assert app.state.ingesting.active is False  # never entered the exclusive-GPU ingestion mode


# ----- Grounded multi-turn chat (Surface A) -----


@pytest.fixture
def fake_chat_answered(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake(conversation_id: str, user_text: str, **_kw: Any) -> Any:
        from memex.agents.chat import ChatTurnResult
        from memex.core.types import ConversationTurn

        resp = FinalResponse(
            answered=True,
            summary="Chat grounded answer.",
            claims=[CitedClaim(claim="a chat claim", source_chunk_id="d1#a", confidence="high")],
            used_chunks=[
                Chunk(chunk_id="d1#a", document_id="d1", document_title="Doc One", text="x")
            ],
            wikilinks=["[[d1#Section]]"],
            correlation_id="01HZCHATWEBUI00000000000000",
            tokens_used=7,
            nodes_traversed=5,
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

    monkeypatch.setattr("memex.webui.app.answer_turn", _fake)


async def _chat_turn_to_completion(app: Any, message: str) -> str:
    """Open a conversation, POST a turn, and long-poll the chat status until the
    grounded assistant bubble replaces the progress fragment (mirrors _ask_to_completion)."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        opened = await ac.get("/chat", follow_redirects=False)
        assert opened.status_code == 303
        conv = opened.headers["location"].rsplit("/", 1)[-1]
        r = await ac.post(f"/chat/{conv}/turn", data={"message": message})
        assert r.status_code == 200, r.text
        text = r.text
        m = re.search(r"/chat/[^/]+/status\?cid=([^&\"]+)&(?:amp;)?v=(\d+)", text)
        assert m is not None, f"POST turn did not return a progress fragment: {text[:300]}"
        cid, v = m.group(1), int(m.group(2))
        for _ in range(100):
            await asyncio.sleep(0.01)
            r = await ac.get(f"/chat/{conv}/status?cid={cid}&v={v}")
            assert r.status_code == 200
            text = r.text
            if 'class="progress"' not in text:
                return text
            mv = re.search(r"&(?:amp;)?v=(\d+)", text)
            if mv is not None:
                v = int(mv.group(1))
        raise AssertionError(f"chat turn did not complete: {text[:300]}")


def test_chat_nav_link_and_redirect(client: TestClient) -> None:
    assert '/chat"' in client.get("/").text  # nav link on every page
    r = client.get("/chat", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].startswith("/chat/")


def test_chat_view_renders_composer(client: TestClient) -> None:
    loc = client.get("/chat", follow_redirects=False).headers["location"]
    r = client.get(loc)
    assert r.status_code == 200
    assert "chat-composer" in r.text
    assert 'name="message"' in r.text
    assert f'hx-post="{loc}/turn"' in r.text


def test_chat_view_404_on_missing(client: TestClient) -> None:
    assert client.get("/chat/01HZNOPECONVERSATION00000000").status_code == 404


@pytest.mark.asyncio
async def test_chat_turn_renders_grounded_answer(
    client: TestClient, fake_chat_answered: None
) -> None:
    text = await _chat_turn_to_completion(client.app, "What does Smith argue?")
    assert "<html" not in text  # a fragment, not a full page
    assert "Chat grounded answer." in text
    assert "a chat claim" in text
    assert "01HZCHATWEBUI" in text  # the audit correlation id


async def test_chat_view_rehydrates_thread(settings: MemexSettings) -> None:
    from memex.core.conversation_store import ConversationStore

    resp = FinalResponse(
        answered=True,
        summary="Persisted answer.",
        claims=[CitedClaim(claim="persisted claim", source_chunk_id="d1#a", confidence="high")],
        used_chunks=[Chunk(chunk_id="d1#a", document_id="d1", document_title="Doc One", text="x")],
        wikilinks=["[[d1#S]]"],
        correlation_id="01HZPERSISTCHAT0000000000000",
        tokens_used=3,
        nodes_traversed=4,
        regenerate_attempts=0,
    )
    store = await ConversationStore.open(settings.vault_path)
    convo = await store.create_conversation()
    await store.append_turn(
        convo.conversation_id,
        user_text="prior question",
        standalone_query="prior question",
        is_followup=False,
        answered=True,
        answer_summary="Persisted answer.",
        cited_chunk_ids=["d1#a"],
        response_json=resp.model_dump_json(),
        correlation_id="cid",
    )
    await store.close()

    r = TestClient(create_app()).get(f"/chat/{convo.conversation_id}")
    assert r.status_code == 200
    assert "prior question" in r.text  # the user bubble
    assert "Persisted answer." in r.text  # the assistant bubble, rehydrated from response_json
    assert "persisted claim" in r.text
    assert "scope-picker" not in r.text  # picker hidden after turn 0 (turn_count > 0)


@pytest.fixture
def fake_chat_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake(conversation_id: str, user_text: str, **_kw: Any) -> Any:
        from memex.agents.chat import ChatTurnResult
        from memex.core.types import ConversationTurn

        resp = FinalResponse(
            answered=False,
            refusal_reason="The vault does not cover that topic.",
            correlation_id="01HZCHATREFUSE000000000000",
            tokens_used=3,
            nodes_traversed=3,
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

    monkeypatch.setattr("memex.webui.app.answer_turn", _fake)


@pytest.mark.asyncio
async def test_chat_turn_renders_refusal(client: TestClient, fake_chat_refused: None) -> None:
    text = await _chat_turn_to_completion(client.app, "something the vault can't answer")
    assert "No grounded answer" in text
    assert "The vault does not cover that topic." in text
    assert "chat-refused" in text
    assert "chat-claim" not in text  # a refusal has no grounded claims


async def test_chat_turn_zero_persists_scope_pin(
    settings: MemexSettings, fake_chat_answered: None
) -> None:
    from memex.core.conversation_store import ConversationStore

    app = create_app()
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        conv = (
            (await ac.get("/chat", follow_redirects=False)).headers["location"].rsplit("/", 1)[-1]
        )
        r = await ac.post(
            f"/chat/{conv}/turn", data={"message": "q", "scope_doc_ids": ["d1", "d2"]}
        )
        assert r.status_code == 200
    store = await ConversationStore.open(settings.vault_path)
    c = await store.load(conv)
    await store.close()
    assert c is not None
    assert c.scope_doc_ids == ["d1", "d2"]  # the turn-0 selection persisted as the pin


# ----- Ungrounded expert surface (Surface B, ADR-0013) -----


@pytest.fixture
def expert_settings(tmp_vault: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemexSettings]:
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_vault))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("MEMEX_AGENTS__EXPERT_MODE_ENABLED", "true")
    s = MemexSettings()  # type: ignore[call-arg]
    set_settings(s)
    yield s
    set_settings(None)


@pytest.fixture
def expert_client(expert_settings: MemexSettings) -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def fake_expert(monkeypatch: pytest.MonkeyPatch) -> None:
    from memex.agents.expert import EXPERT_PROVENANCE_NOTE, ExpertAnswer, ExpertEvidence

    async def _fake(question: str, **kw: Any) -> ExpertAnswer:
        on_phase = kw.get("on_phase")
        if callable(on_phase):
            on_phase("Retrieving evidence")
            on_phase("Reasoning")
        return ExpertAnswer(
            question=question,
            answer="RSTP converges faster than STP because of its handshake.",
            evidence=[
                ExpertEvidence(
                    chunk_id="d1#a",
                    document_id="d1",
                    title="RSTP Guide",
                    section="Convergence",
                    snippet="…",
                )
            ],
            provenance_note=EXPERT_PROVENANCE_NOTE,
            model="m",
            tokens=99,
            correlation_id=kw.get("correlation_id") or "cid",
        )

    monkeypatch.setattr("memex.webui.app.expert_answer", _fake)


async def _expert_to_completion(app: Any, question: str) -> str:
    """POST /expert, then long-poll /expert/status until the answer fragment swaps in."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        r = await ac.post("/expert", data={"question": question})
        assert r.status_code == 200, r.text
        text = r.text
        m = re.search(r"/expert/status\?cid=([^&\"]+)&(?:amp;)?v=(\d+)", text)
        assert m is not None, f"POST /expert did not return a progress fragment: {text[:300]}"
        cid, v = m.group(1), int(m.group(2))
        for _ in range(100):
            await asyncio.sleep(0.01)
            r = await ac.get(f"/expert/status?cid={cid}&v={v}")
            assert r.status_code == 200
            text = r.text
            if 'class="progress"' not in text:
                return text
            mv = re.search(r"&(?:amp;)?v=(\d+)", text)
            if mv is not None:
                v = int(mv.group(1))
        raise AssertionError(f"expert did not complete after polling: {text[:300]}")


def test_nav_hides_expert_when_disabled(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert 'href="/expert"' not in r.text  # default off → no dead nav link


def test_nav_shows_expert_when_enabled(expert_client: TestClient) -> None:
    r = expert_client.get("/")
    assert r.status_code == 200
    assert 'href="/expert"' in r.text


def test_expert_home_disabled_explains_how_to_enable(client: TestClient) -> None:
    r = client.get("/expert")
    assert r.status_code == 200
    assert "Disabled" in r.text
    assert "MEMEX_AGENTS__EXPERT_MODE_ENABLED" in r.text
    assert 'name="question"' not in r.text  # no form while disabled


def test_expert_home_renders_form_and_banner(expert_client: TestClient) -> None:
    r = expert_client.get("/expert")
    assert r.status_code == 200
    assert 'name="question"' in r.text
    assert 'hx-post="/expert"' in r.text
    assert "ungrounded" in r.text  # the contract-inversion banner


def test_expert_disabled_post_refuses(client: TestClient) -> None:
    r = client.post("/expert", data={"question": "anything"})
    assert r.status_code == 200
    assert "disabled" in r.text.lower()


def test_expert_post_empty_question_flashes(expert_client: TestClient) -> None:
    r = expert_client.post("/expert", data={"question": "   "})
    assert r.status_code == 200
    assert "analytical question" in r.text.lower()


@pytest.mark.asyncio
async def test_expert_post_then_answer(expert_client: TestClient, fake_expert: None) -> None:
    text = await _expert_to_completion(expert_client.app, "Compare STP and RSTP.")
    assert "RSTP converges faster than STP" in text
    assert "Evidence consulted" in text
    assert "RSTP Guide" in text
    assert "ungrounded" in text.lower()  # the standing provenance caveat
    assert "correlation_id" in text


# ── Reason-then-ground bridge surface (§11) — gated on the SAME flag as expert mode ──


def _install_fake_bridge(
    monkeypatch: pytest.MonkeyPatch, *, grounded: bool, responsive: bool = True
) -> None:
    from memex.agents.answering import CitedClaim
    from memex.agents.bridge import BRIDGE_PROVENANCE_NOTE, BridgeAnswer
    from memex.agents.expert import ExpertEvidence
    from memex.core.types import Chunk

    async def _fake(question: str, **kw: Any) -> BridgeAnswer:
        on_phase = kw.get("on_phase")
        if callable(on_phase):
            on_phase("Retrieving evidence")
            on_phase("Reasoning")
            on_phase("Grounding claims")
        claims = (
            [
                CitedClaim(
                    claim="OSPF is a link-state protocol.",
                    source_chunk_id="d1#a",
                    confidence="high",
                )
            ]
            if grounded
            else []
        )
        sources = (
            [
                Chunk(
                    chunk_id="d1#a",
                    document_id="d1",
                    document_title="OSPF Guide",
                    text="OSPF is link-state.",
                    heading_path=["Intro"],
                )
            ]
            if grounded
            else []
        )
        # Mirror the bridge's present-as-answer contract: the gate runs only when the caller asked
        # AND something grounded; `presented` follows from responsive (ADR-0016).
        present = bool(kw.get("present_as_answer"))
        gate_runs = present and bool(claims)
        return BridgeAnswer(
            question=question,
            analysis="OSPF converges quickly; BGP is policy-driven.",
            grounded_claims=claims,
            grounded_sources=sources,
            evidence=[
                ExpertEvidence(
                    chunk_id="d1#a",
                    document_id="d1",
                    title="OSPF Guide",
                    section="Intro",
                    snippet="…",
                )
            ],
            provenance_note=BRIDGE_PROVENANCE_NOTE,
            n_extracted=2,
            n_grounded=len(claims),
            model="m",
            tokens=99,
            correlation_id=kw.get("correlation_id") or "cid",
            present_as_answer=present,
            responsive=(responsive if gate_runs else None),
            relevance_reason=("" if responsive else "answers a related question")
            if gate_runs
            else "",
            answer_headline=(" ".join(c.claim for c in claims) if gate_runs else ""),
            # The name-only guard normally filters this; the fake's chunk text is substantive,
            # so presentable == grounded (the surface renders presented_claims when presented).
            presented_claims=(claims if gate_runs else []),
        )

    monkeypatch.setattr("memex.webui.app.reason_then_ground", _fake)


async def _bridge_to_completion(app: Any, question: str, **form: Any) -> str:
    """POST /bridge, then long-poll /bridge/status until the answer fragment swaps in."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        r = await ac.post("/bridge", data={"question": question, **form})
        assert r.status_code == 200, r.text
        text = r.text
        m = re.search(r"/bridge/status\?cid=([^&\"]+)&(?:amp;)?v=(\d+)", text)
        assert m is not None, f"POST /bridge did not return a progress fragment: {text[:300]}"
        cid, v = m.group(1), int(m.group(2))
        for _ in range(100):
            await asyncio.sleep(0.01)
            r = await ac.get(f"/bridge/status?cid={cid}&v={v}")
            assert r.status_code == 200
            text = r.text
            if 'class="progress"' not in text:
                return text
            mv = re.search(r"&(?:amp;)?v=(\d+)", text)
            if mv is not None:
                v = int(mv.group(1))
        raise AssertionError(f"bridge did not complete after polling: {text[:300]}")


def test_nav_hides_analysis_when_disabled(client: TestClient) -> None:
    r = client.get("/")
    assert 'href="/bridge"' not in r.text  # default off → no dead nav link


def test_nav_shows_analysis_when_enabled(expert_client: TestClient) -> None:
    r = expert_client.get("/")
    assert 'href="/bridge"' in r.text


def test_bridge_home_disabled_explains_how_to_enable(client: TestClient) -> None:
    r = client.get("/bridge")
    assert r.status_code == 200
    assert "Disabled" in r.text
    assert "MEMEX_AGENTS__EXPERT_MODE_ENABLED" in r.text
    assert 'name="question"' not in r.text  # no form while disabled


def test_bridge_home_renders_form_and_banner(expert_client: TestClient) -> None:
    r = expert_client.get("/bridge")
    assert r.status_code == 200
    assert 'name="question"' in r.text
    assert 'hx-post="/bridge"' in r.text
    assert "grounded" in r.text.lower()  # the dual-contract banner


def test_bridge_disabled_post_refuses(client: TestClient) -> None:
    r = client.post("/bridge", data={"question": "anything"})
    assert r.status_code == 200
    assert "disabled" in r.text.lower()


@pytest.mark.asyncio
async def test_bridge_post_then_answer(
    expert_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_bridge(monkeypatch, grounded=True)
    text = await _bridge_to_completion(expert_client.app, "Compare OSPF and BGP.")
    assert "OSPF converges quickly" in text  # the ungrounded analysis
    assert "OSPF is a link-state protocol." in text  # the grounded claim
    assert "OSPF Guide" in text  # the grounded source rendered by title
    assert "Grounded claims" in text
    assert "correlation_id" in text


@pytest.mark.asyncio
async def test_bridge_zero_grounded_shows_note_not_refusal(
    expert_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_bridge(monkeypatch, grounded=False)
    text = await _bridge_to_completion(expert_client.app, "An unverifiable question?")
    assert "OSPF converges quickly" in text  # the analysis is still returned
    assert "could be verified against your vault" in text  # the empty-grounded note
    assert 'class="ans-flash-error"' not in text  # NOT a refusal/error


@pytest.mark.asyncio
async def test_bridge_surfaces_evidence_documents_on_fallback(
    expert_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The user-journey fix: even with NOTHING grounded, the bridge surfaces the vault documents
    the analysis was reasoned over (the retrieved evidence) as navigable links — so the user can
    open them and see what the vault actually says. Labelled "reasoned over", NOT a grounding cite."""
    _install_fake_bridge(monkeypatch, grounded=False)  # zero-grounded fallback
    text = await _bridge_to_completion(expert_client.app, "An unverifiable question?")
    assert "Retrieved from your vault" in text  # the evidence section eyebrow
    assert 'href="/documents/d1"' in text  # the evidence doc is navigable
    assert "OSPF Guide" in text  # rendered by title
    assert "not grounding cites" in text  # the honest label (fallback wording)


# ── Consented A→B escalation from a Surface-A refusal (§11) ──


@pytest.mark.asyncio
async def test_ask_refusal_offers_escalation_when_expert_enabled(
    expert_client: TestClient, fake_refused: None
) -> None:
    text = await _ask_to_completion(expert_client.app, "an unanswerable question?")
    assert "Refused" in text
    assert 'hx-post="/bridge"' in text  # the consented escalation form targets the bridge
    assert "verify this" in text  # the "Reason & verify this →" affordance
    assert (
        "an unanswerable question?" in text
    )  # the original question carried into the hidden input


@pytest.mark.asyncio
async def test_ask_refusal_no_escalation_when_expert_disabled(
    client: TestClient, fake_refused: None
) -> None:
    text = await _ask_to_completion(client.app, "an unanswerable question?")
    assert "Refused" in text  # the refusal still renders
    assert 'hx-post="/bridge"' not in text  # gated off → no escalation affordance
    assert "Reason over this instead" not in text


@pytest.mark.asyncio
async def test_ask_refusal_escalation_form_carries_present_as_answer(
    expert_client: TestClient, fake_refused: None
) -> None:
    """The consented escalation form sets present_as_answer=true (ADR-0016) so the bridge presents
    the grounded subset AS an answer when responsive."""
    text = await _ask_to_completion(expert_client.app, "an unanswerable question?")
    assert 'name="present_as_answer"' in text
    assert 'value="true"' in text


# ── Present-as-answer escalation (ADR-0016): the bridge result IS the answer ──


@pytest.mark.asyncio
async def test_bridge_present_as_answer_renders_grounded_answer(
    expert_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """present_as_answer=true + grounded + responsive → the grounded claims are presented AS the
    answer (the distinct 'Reasoned, then grounded' surface), with the reasoning fenced in a
    <details> — NOT the leading 'Analysis ungrounded' labelled surface."""
    _install_fake_bridge(monkeypatch, grounded=True, responsive=True)
    text = await _bridge_to_completion(
        expert_client.app, "Compare OSPF and BGP.", present_as_answer="true"
    )
    assert "Reasoned, then grounded" in text  # the distinct presented eyebrow
    assert "OSPF is a link-state protocol." in text  # the grounded claim, as the answer
    assert "Show the model's reasoning" in text  # the reasoning is fenced in <details>
    assert "OSPF converges quickly" in text  # the ungrounded analysis lives inside the <details>
    assert ">Analysis " not in text  # NOT the labelled-analysis lead
    assert ">Grounded claims" not in text  # not the labelled-subset header either


@pytest.mark.asyncio
async def test_bridge_present_as_answer_non_responsive_falls_back(
    expert_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """present_as_answer=true but the grounded subset is non-responsive → fall back to the labelled
    analysis surface with a quiet 'related question' note; NOT presented as an answer."""
    _install_fake_bridge(monkeypatch, grounded=True, responsive=False)
    text = await _bridge_to_completion(
        expert_client.app, "Compare OSPF and BGP.", present_as_answer="true"
    )
    assert "Reasoned, then grounded" not in text  # NOT presented
    assert ">Analysis " in text  # the labelled-analysis surface
    assert "bridge-nonresponsive" in text  # the related-question note
    assert "related question" in text


@pytest.mark.asyncio
async def test_bridge_standalone_post_is_not_presented(
    expert_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The standalone composer (no present_as_answer marker) never presents-as-answer — the
    labelled-analysis surface, byte-identical to before."""
    _install_fake_bridge(monkeypatch, grounded=True, responsive=True)
    text = await _bridge_to_completion(expert_client.app, "Compare OSPF and BGP.")
    assert "Reasoned, then grounded" not in text
    assert "Grounded claims" in text  # the labelled-subset surface


@pytest.mark.asyncio
async def test_bridge_presented_renders_only_presented_claims_with_held_back_note(
    expert_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The name-only guard (ADR-0016): the presented body renders `presented_claims` only; a
    held-back (name-only-cited) grounded claim does NOT appear, and a held-back note explains it."""
    from memex.agents.answering import CitedClaim
    from memex.agents.bridge import BRIDGE_PROVENANCE_NOTE, BridgeAnswer

    shown = CitedClaim(claim="Shown grounded claim.", source_chunk_id="d1#a", confidence="high")
    held = CitedClaim(claim="Held back name-only claim.", source_chunk_id="d1#b", confidence="high")

    async def _fake(question: str, **kw: Any) -> BridgeAnswer:
        on_phase = kw.get("on_phase")
        if callable(on_phase):
            for p in ("Retrieving evidence", "Reasoning", "Grounding claims"):
                on_phase(p)
        return BridgeAnswer(
            question=question,
            analysis="Ungrounded reasoning.",
            grounded_claims=[shown, held],  # full gate output
            grounded_sources=[],
            provenance_note=BRIDGE_PROVENANCE_NOTE,
            n_extracted=2,
            n_grounded=2,
            present_as_answer=True,
            responsive=True,
            answer_headline="Shown grounded claim.",
            presented_claims=[shown],  # `held` filtered by the name-only guard
            correlation_id=kw.get("correlation_id") or "cid",
        )

    monkeypatch.setattr("memex.webui.app.reason_then_ground", _fake)
    text = await _bridge_to_completion(expert_client.app, "Q?", present_as_answer="true")
    assert "Reasoned, then grounded" in text
    assert "Shown grounded claim." in text
    assert "Held back name-only claim." not in text  # held back from the presented body
    assert "held back from the answer" in text  # the held-back note


@pytest.mark.asyncio
async def test_ask_answered_has_no_escalation(
    expert_client: TestClient, fake_answered: None
) -> None:
    text = await _ask_to_completion(expert_client.app, "What does Smith argue?")
    # Refusal-only by construction: an answered response never offers the escalation,
    # even with expert mode enabled.
    assert "Reason over this instead" not in text


@pytest.mark.asyncio
async def test_ask_scoped_refusal_escalation_carries_scope(
    expert_client: TestClient, fake_refused: None
) -> None:
    text = await _ask_to_completion(
        expert_client.app, "an unanswerable question?", scope_doc_ids=["d1", "d2"]
    )
    assert "verify this" in text  # the "Reason & verify this →" affordance
    # The escalation re-POSTs the ORIGINAL scope so the bridge respects the user's constraint
    # (not silently widened to the whole vault).
    assert 'name="scope_doc_ids" value="d1"' in text
    assert 'name="scope_doc_ids" value="d2"' in text


# ── Standalone /bridge document scope-picker (§11) ──


@pytest.mark.asyncio
async def test_bridge_composer_renders_scope_picker(expert_client: TestClient) -> None:
    """The /bridge "Analysis" composer offers the SAME document scope-picker as /ask."""
    lec = await ingest_markdown_passthrough(
        "# OSPF\n\nOpen Shortest Path First.\n", source_stem="ENSA Module 1 OSPF"
    )
    r = expert_client.get("/bridge")
    assert r.status_code == 200
    assert "Scope to documents" in r.text
    assert 'name="scope_doc_ids"' in r.text
    assert f'value="{lec.doc_id}"' in r.text
    assert "ENSA Module 1 OSPF" in r.text  # offered by human title


@pytest.mark.asyncio
async def test_bridge_scopes_to_selected_docs(
    expert_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ticking docs on /bridge forwards `scope_doc_ids` to `reason_then_ground`, and the
    result shows the 'Scoped to your selected document:' note (parity with /ask)."""
    from memex.agents.bridge import BRIDGE_PROVENANCE_NOTE, BridgeAnswer

    lec = await ingest_markdown_passthrough(
        "# OSPF\n\nOpen Shortest Path First.\n", source_stem="ENSA Module 1 OSPF"
    )
    captured: dict[str, Any] = {}

    async def _fake(question: str, **kw: Any) -> BridgeAnswer:
        captured["scope_doc_ids"] = kw.get("scope_doc_ids")
        on_phase = kw.get("on_phase")
        if callable(on_phase):
            on_phase("Retrieving evidence")
            on_phase("Reasoning")
            on_phase("Grounding claims")
        return BridgeAnswer(
            question=question,
            analysis="OSPF is a link-state protocol.",
            provenance_note=BRIDGE_PROVENANCE_NOTE,
            n_extracted=0,
            n_grounded=0,
            model="m",
            tokens=10,
            scope_doc_ids=[lec.doc_id],
            correlation_id=kw.get("correlation_id") or "cid",
        )

    monkeypatch.setattr("memex.webui.app.reason_then_ground", _fake)
    text = await _bridge_to_completion(
        expert_client.app, "How does OSPF converge?", scope_doc_ids=[lec.doc_id]
    )
    assert captured["scope_doc_ids"] == [lec.doc_id]  # the route forwarded the selection
    assert "Scoped to your selected document" in text  # the result-side scope note
    assert "ENSA Module 1 OSPF" in text  # scoped doc shown by title


@pytest.mark.asyncio
async def test_ask_renders_companion_chip_for_aligned_transcript_chunk(
    settings: MemexSettings, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-0018 B3: a cited TRANSCRIPT chunk that is aligned to a slide deck shows a `↔ slide N`
    companion chip on its source link (read fail-open from the alignment sidecar). No alignment → no
    chip; HARD-gate-neutral (presentation lookup over the cited chunks)."""
    from memex.core.companion_store import upsert_alignment
    from memex.core.types import AlignmentBlock, CompanionAlignment

    await upsert_alignment(
        settings.vault_path,
        CompanionAlignment(
            transcript_doc="lecture1",
            deck_doc="deck1",
            null_count=0,
            blocks=[
                AlignmentBlock(
                    transcript_chunk_id="lecture1#t1",
                    deck_chunk_id="deck1#g",
                    deck_page=12,
                    score=0.61,
                )
            ],
        ),
    )

    async def _fake(question: str, **_kw: Any) -> FinalResponse:
        chunk = Chunk(
            chunk_id="lecture1#t1",
            document_id="lecture1",
            document_title="Lecture 1",
            text="VLANs segment the broadcast domain.",
            heading_path=["[01:02]"],
            time_range=(62.0, 66.0),
        )
        return FinalResponse(
            answered=True,
            summary="VLANs segment the network.",
            claims=[
                CitedClaim(
                    claim="A VLAN segments the broadcast domain.",
                    source_chunk_id="lecture1#t1",
                    confidence="high",
                )
            ],
            used_chunks=[chunk],
            wikilinks=[],
            correlation_id="01HZCOMP0000000000000000",
            tokens_used=10,
            nodes_traversed=4,
            regenerate_attempts=0,
        )

    monkeypatch.setattr("memex.webui.app.answer_query", _fake)
    text = await _ask_to_completion(client.app, "what does a VLAN do?")
    assert "↔ slide 12" in text  # the cited transcript chunk's aligned slide


# ----- Frictionless document ingestion (upload → pipeline → browsable) -----


def _patch_daemon_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake daemon_status → reachable, so `_await_orchestrator_reachable` returns at once (no real
    daemon dependency, no 150s poll)."""

    async def _status(*_a: object, **_k: object) -> DaemonStatus:
        return DaemonStatus(
            pid=1, alive=True, reachable=True, base_url="http://t/v1", pid_file="", log_file=""
        )

    monkeypatch.setattr("memex.webui.app.daemon_status", _status)


def _patch_ingest(monkeypatch: pytest.MonkeyPatch, *, outcome: IngestOutcome) -> None:
    """Patch the ingest_driver subprocess calls with fakes (no real `memex ingest`/`enrich`)."""
    _patch_daemon_reachable(monkeypatch)

    async def _fake_run_ingest(
        _file_path: Path, *, on_phase: Callable[[str], None], **_kw: object
    ) -> IngestOutcome:
        for phase in ("Parsing", "Transcribing · page 1", "Indexing"):
            on_phase(phase)
        return outcome

    async def _fake_run_enrich(
        _doc_id: str, *, on_phase: Callable[[str], None], **_kw: object
    ) -> int:
        on_phase("Enriching")
        return 0

    monkeypatch.setattr("memex.webui.ingest_driver.run_ingest", _fake_run_ingest)
    monkeypatch.setattr("memex.webui.ingest_driver.run_enrich", _fake_run_enrich)


async def _ingest_to_completion(app: Any, *, content: bytes, filename: str) -> str:
    """POST a file to /ingest, long-poll /ingest/{cid}/status until the done-fragment swaps in
    (httpx on the shared loop, so the background subprocess-driver task actually runs)."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        r = await ac.post(
            "/ingest", files={"file": (filename, content, "application/octet-stream")}
        )
        assert r.status_code == 200, r.text
        m = re.search(r"/ingest/([^/?\"]+)/status\?v=(\d+)", r.text)
        assert m is not None, f"POST /ingest did not return a progress fragment: {r.text[:300]}"
        cid, v = m.group(1), int(m.group(2))
        text = r.text
        for _ in range(200):
            await asyncio.sleep(0.01)
            r = await ac.get(f"/ingest/{cid}/status?v={v}")
            assert r.status_code == 200
            text = r.text
            if 'class="progress"' not in text:
                return text
            mv = re.search(r"\?v=(\d+)", text)
            if mv is not None:
                v = int(mv.group(1))
        raise AssertionError(f"ingest did not complete after polling: {text[:300]}")


def test_ingest_page_renders_upload_form(client: TestClient) -> None:
    r = client.get("/ingest")
    assert r.status_code == 200
    assert 'type="file"' in r.text and 'name="file"' in r.text
    assert 'hx-post="/ingest"' in r.text
    assert 'hx-encoding="multipart/form-data"' in r.text
    # The accepted-types copy advertises standalone images (ADR-0020).
    assert "image" in r.text.lower()
    assert "PNG/JPEG/WebP" in r.text


def test_nav_has_add_document_link(client: TestClient) -> None:
    r = client.get("/")
    assert 'href="/ingest"' in r.text and "Add document" in r.text


def test_ingest_post_returns_progress_fragment(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_ingest(
        monkeypatch, outcome=IngestOutcome(accepted=True, exit_code=0, doc_id="abcd1234-doc")
    )
    r = client.post("/ingest", files={"file": ("note.pdf", b"%PDF-1.4 fake", "application/pdf")})
    assert r.status_code == 200
    assert 'class="progress"' in r.text
    assert "/ingest/" in r.text and "/status?v=" in r.text
    assert "Parsing" in r.text  # the first ingest step renders


def test_ingest_oversize_file_rejected(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("memex.webui.app._MAX_UPLOAD_BYTES", 8)
    r = client.post(
        "/ingest", files={"file": ("big.pdf", b"way more than eight bytes", "application/pdf")}
    )
    assert r.status_code == 200
    assert "exceeds the 2 GiB upload limit" in r.text
    assert 'class="progress"' not in r.text  # no background task started


async def test_ingest_full_flow_renders_done_fragment(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_ingest(
        monkeypatch, outcome=IngestOutcome(accepted=True, exit_code=0, doc_id="abcd1234-newdoc")
    )
    text = await _ingest_to_completion(create_app(), content=b"%PDF fake", filename="note.pdf")
    assert "Ingested" in text
    assert "/documents/abcd1234-newdoc" in text
    assert "Open document" in text


async def test_ingest_rejection_surfaces_reason(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_ingest(
        monkeypatch,
        outcome=IngestOutcome(
            accepted=False, exit_code=0, doc_id=None, rejection_reason="unsupported file type"
        ),
    )
    text = await _ingest_to_completion(create_app(), content=b"\x00\x01", filename="x.bin")
    assert "Ingest failed" in text
    assert "unsupported file type" in text


async def test_ingest_partial_doc_surfaces_browsable_link(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Accepted + parsed (the canonical .md WAS written) but the chain exited non-zero (half-doc):
    # the doc is genuinely browsable, so surface the link + a warning.
    (settings.vault_path / "documents").mkdir(parents=True, exist_ok=True)
    (settings.vault_path / "documents" / "abcd1234-half.md").write_text("# Half doc\n\nbody")
    _patch_ingest(
        monkeypatch,
        outcome=IngestOutcome(
            accepted=True,
            exit_code=1,
            doc_id="abcd1234-half",
            rejection_reason="the ingest process exited with code 1",
        ),
    )
    text = await _ingest_to_completion(create_app(), content=b"%PDF fake", filename="note.pdf")
    assert "Partially ingested" in text
    assert "/documents/abcd1234-half" in text


async def test_ingest_partial_without_md_renders_failure_no_dead_link(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A doc_id was assigned (ingest.accepted) but parse failed BEFORE writing the .md — don't render
    # a dead "Open document (browsable)" link that 404s (B10); render a plain failure instead.
    _patch_ingest(
        monkeypatch,
        outcome=IngestOutcome(
            accepted=True,
            exit_code=1,
            doc_id="abcd1234-nomd",
            rejection_reason="parse failed",
        ),
    )
    text = await _ingest_to_completion(create_app(), content=b"%PDF fake", filename="note.pdf")
    assert "Ingest failed" in text
    assert "/documents/abcd1234-nomd" not in text  # no dead browse link


async def test_ingest_reconciles_a_down_orchestrator(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # D3 (the headline fix): if the orchestrator is DOWN after the ingest subprocess (its own
    # serve-vllm.sh restart failed under GPU contention), _run_ingest must ACTIVELY restart it via
    # the daemon supervisor — not just passively wait and leave RAG 404ing. Stateful fake: down
    # until daemon_restart is called, reachable after (mirrors reality).
    restarts: list[dict[str, Any]] = []
    state = {"reachable": False}

    async def _status(*_a: object, **_k: object) -> DaemonStatus:
        return DaemonStatus(
            pid=1, alive=True, reachable=state["reachable"], base_url="http://t/v1",
            pid_file="", log_file="",
        )

    async def _restart(
        _s: object, *, gpu_fraction: float | None = None, max_model_len: int | None = None
    ) -> DaemonStatus:
        restarts.append({"gpu_fraction": gpu_fraction, "max_model_len": max_model_len})
        state["reachable"] = True  # the active reconcile brought it back
        return DaemonStatus(
            pid=2, alive=True, reachable=True, base_url="http://t/v1", pid_file="", log_file=""
        )

    monkeypatch.setattr("memex.webui.app.daemon_status", _status)
    monkeypatch.setattr("memex.webui.app.daemon_restart", _restart)

    async def _run_ingest(_fp: Path, *, on_phase: Callable[[str], None], **_kw: object) -> IngestOutcome:
        on_phase("Indexing")
        return IngestOutcome(accepted=True, exit_code=0, doc_id="abc12345-doc", rejection_reason=None)

    async def _run_enrich(_d: str, *, on_phase: Callable[[str], None], **_kw: object) -> int:
        on_phase("Enriching")
        return 0

    monkeypatch.setattr("memex.webui.ingest_driver.run_ingest", _run_ingest)
    monkeypatch.setattr("memex.webui.ingest_driver.run_enrich", _run_enrich)

    app = create_app()
    text = await _ingest_to_completion(app, content=b"%PDF", filename="d.pdf")

    assert "Ingested" in text
    assert len(restarts) >= 1  # the reconcile ACTIVELY restarted the down orchestrator (not a wait)
    assert app.state.ingesting.active is False  # lock released; RAG resumes against a live orchestrator


async def test_ingest_streams_upload_bytes_to_subprocess_file(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real streaming spine (the fakes otherwise ignore the temp file): the bytes the driver
    # receives on disk must equal the uploaded bytes (right cursor, no truncation).
    seen: dict[str, bytes] = {}

    async def _capture_ingest(
        file_path: Path, *, on_phase: Callable[[str], None], **_kw: object
    ) -> IngestOutcome:
        seen["content"] = file_path.read_bytes()
        on_phase("Indexing")
        return IngestOutcome(accepted=True, exit_code=0, doc_id="abcd1234-streamed")

    async def _ok_enrich(_doc_id: str, *, on_phase: Callable[[str], None], **_kw: object) -> int:
        return 0

    _patch_daemon_reachable(monkeypatch)
    monkeypatch.setattr("memex.webui.ingest_driver.run_ingest", _capture_ingest)
    monkeypatch.setattr("memex.webui.ingest_driver.run_enrich", _ok_enrich)
    payload = b"%PDF-1.7 the actual uploaded bytes \x00\x01\x02 " * 200
    text = await _ingest_to_completion(create_app(), content=payload, filename="real.pdf")
    assert "Ingested" in text
    assert seen["content"] == payload  # streamed to disk intact for the subprocess


async def test_ingest_driver_crash_renders_failed(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _boom(
        _file_path: Path, *, on_phase: Callable[[str], None], **_kw: object
    ) -> IngestOutcome:
        raise RuntimeError("driver exploded")

    _patch_daemon_reachable(monkeypatch)
    monkeypatch.setattr("memex.webui.ingest_driver.run_ingest", _boom)
    app = create_app()
    text = await _ingest_to_completion(app, content=b"x", filename="x.pdf")
    assert "Ingest failed" in text  # the crash is surfaced, not swallowed
    assert app.state.ingesting.active is False  # the lock is released even on a crash


async def test_ingest_enrich_failure_still_links_browsable_doc(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Enrich is best-effort (graph discovery, HARD-gate-neutral): a RAISING enrich must NOT
    # discard a parsed+indexed (askable) document — it's presented as success with a live link.
    async def _ok_ingest(
        _file_path: Path, *, on_phase: Callable[[str], None], **_kw: object
    ) -> IngestOutcome:
        on_phase("Indexing")
        return IngestOutcome(accepted=True, exit_code=0, doc_id="abcd1234-enrichfail")

    async def _boom_enrich(_doc_id: str, *, on_phase: Callable[[str], None], **_kw: object) -> int:
        raise RuntimeError("enrich exploded")

    _patch_daemon_reachable(monkeypatch)
    monkeypatch.setattr("memex.webui.ingest_driver.run_ingest", _ok_ingest)
    monkeypatch.setattr("memex.webui.ingest_driver.run_enrich", _boom_enrich)
    text = await _ingest_to_completion(create_app(), content=b"%PDF", filename="x.pdf")
    assert "Ingested" in text  # presented as success, NOT "Ingest failed"
    assert "/documents/abcd1234-enrichfail" in text  # the browsable link is preserved


# ----- Exclusive-GPU ingestion mode: the RAG-surface lock (Inc 3) -----


def test_ingesting_locks_all_rag_posts(client: TestClient) -> None:
    """While a document is ingesting, every RAG POST returns the action-pane 'still ingesting'
    fragment (the orchestrator is down + the GPU is exclusive — an honest pause beats a 404'd
    answer). The wording is distinct from the global page banner (B20)."""
    client.app.state.ingesting.active = True
    posts: list[tuple[str, dict[str, str]]] = [
        ("/ask", {"question": "x"}),
        ("/documents/abcd1234/summarize", {"detail": "standard"}),
        ("/chat/conv01/turn", {"message": "x"}),
        ("/expert", {"question": "x"}),
        ("/bridge", {"question": "x"}),
    ]
    for path, data in posts:
        r = client.post(path, data=data)
        assert r.status_code == 200, f"{path}: {r.status_code}"
        assert "Still ingesting" in r.text, f"{path} did not lock"


def test_ingesting_shows_banner_on_entry_page(client: TestClient) -> None:
    client.app.state.ingesting.active = True
    r = client.get("/")
    assert r.status_code == 200
    assert "answering is paused" in r.text  # the live jinja global drives the banner
    client.app.state.ingesting.active = False
    assert "answering is paused" not in client.get("/").text  # gone when not ingesting


def test_entry_page_banner_is_self_refreshing(client: TestClient) -> None:
    # The banner must carry the self-refresh poll so it CLEARS on its own when the ingest finishes
    # (the lock releases AFTER the done-fragment renders, so a one-shot OOB clear would be premature).
    client.app.state.ingesting.active = True
    r = client.get("/")
    assert 'hx-get="/ingest/banner"' in r.text and 'hx-trigger="every 3s"' in r.text


def test_ingest_banner_fragment_reflects_and_clears(client: TestClient) -> None:
    client.app.state.ingesting.active = True
    r = client.get("/ingest/banner")
    assert r.status_code == 200
    assert "answering is paused" in r.text and 'hx-get="/ingest/banner"' in r.text  # polls itself
    client.app.state.ingesting.active = False
    r = client.get("/ingest/banner")
    assert "answering is paused" not in r.text  # cleared
    assert "hx-trigger" not in r.text  # trigger-less empty div ⇒ the polling stops


def test_ingest_lock_fragment_clears_when_done(client: TestClient) -> None:
    # The RAG-paused notice a GPU POST got mid-ingest self-refreshes to a "ready" fragment once the
    # lock releases — so the stale "Still ingesting" notice clears without a manual reload.
    client.app.state.ingesting.active = True
    r = client.get("/ingest/lock")
    assert r.status_code == 200
    assert "Still ingesting" in r.text and 'hx-get="/ingest/lock"' in r.text  # keeps polling
    client.app.state.ingesting.active = False
    r = client.get("/ingest/lock")
    assert "Still ingesting" not in r.text
    assert "Ingestion finished" in r.text  # the ready fragment
    assert "hx-trigger" not in r.text  # no trigger ⇒ polling stops


def test_ingesting_keeps_browsing_open(client: TestClient) -> None:
    """Browsing routes touch no GPU and stay available during an ingest."""
    client.app.state.ingesting.active = True
    for path in ("/documents", "/resources"):
        assert client.get(path).status_code == 200


def test_ingesting_rejects_concurrent_upload(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_ingest(
        monkeypatch, outcome=IngestOutcome(accepted=True, exit_code=0, doc_id="abcd1234-x")
    )
    client.app.state.ingesting.active = True  # an ingest is already running
    r = client.post("/ingest", files={"file": ("a.pdf", b"%PDF", "application/pdf")})
    assert r.status_code == 200
    assert "Still ingesting" in r.text  # single-flight: rejected with the action-pane notice


def test_ingest_no_file_renders_friendly_fragment(client: TestClient) -> None:
    # A malformed POST (no `file` field) must NOT surface FastAPI's raw 422 JSON in the HTMX pane —
    # the optional file param + a None check render the friendly done-fragment instead (B18).
    r = client.post("/ingest", data={"notafile": "x"})
    assert r.status_code == 200
    assert "No file was provided" in r.text
    assert "Ingest failed" in r.text  # the failed branch of _ingest_done.html


async def test_ingest_zero_chunk_doc_not_claimed_searchable(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A doc that parsed + indexed but produced 0 chunks (e.g. an image-only PDF, VLM off) is
    # browsable but NOT searchable — the done-fragment must not claim "fully consumed / searchable"
    # (B12). chunk_count=0 routes to the honest partial fragment.
    _patch_ingest(
        monkeypatch,
        outcome=IngestOutcome(
            accepted=True, exit_code=0, doc_id="abcd1234-empty", chunk_count=0
        ),
    )
    text = await _ingest_to_completion(create_app(), content=b"%PDF", filename="blank.pdf")
    assert "no searchable text" in text
    assert "/documents/abcd1234-empty" in text  # still browsable
    assert "fully consumed" not in text  # NOT the success "searchable and browsable" claim


async def test_ingest_unknown_chunk_count_takes_normal_success(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # chunk_count=None (event not captured — old subprocess / unusual path) must NOT be gated as
    # 0-chunk; it takes the normal success path (status quo).
    _patch_ingest(
        monkeypatch,
        outcome=IngestOutcome(
            accepted=True, exit_code=0, doc_id="abcd1234-ok", chunk_count=None
        ),
    )
    text = await _ingest_to_completion(create_app(), content=b"%PDF", filename="doc.pdf")
    assert "fully consumed" in text  # the normal success claim
    assert "no searchable text" not in text


def test_ingest_page_resumes_in_flight_progress(client: TestClient) -> None:
    # B7/B8: returning to GET /ingest while an ingest is in flight resumes its live progress in the
    # pane, instead of a form-only page that drops the running ingest from view.
    app = client.app
    cid = "01HZRESUME0000000000000000"
    app.state.progress.new(cid, scope_doc_ids=[], scope_source="named")
    app.state.ingesting.active = True
    app.state.ingesting.cid = cid
    try:
        r = client.get("/ingest")
    finally:
        app.state.ingesting.active = False
        app.state.ingesting.cid = ""
    assert r.status_code == 200
    assert f"/ingest/{cid}/status" in r.text  # the pane auto-loads the in-flight progress
    assert "Resuming" in r.text


def test_ingest_page_fresh_when_idle(client: TestClient) -> None:
    # No in-flight ingest → a normal form-only page (no resume block).
    r = client.get("/ingest")
    assert r.status_code == 200
    assert "Resuming" not in r.text
    assert 'type="file"' in r.text  # the upload form is present


async def test_upload_size_limit_middleware_rejects_oversize() -> None:
    # B11: the ASGI middleware rejects an over-cap Content-Length on POST /ingest WITHOUT calling
    # the downstream app — so the body never streams to disk (the handler check is too late).
    from memex.webui.app import _UploadSizeLimitMiddleware

    downstream_called = False

    async def downstream(scope: Any, receive: Any, send: Any) -> None:
        nonlocal downstream_called
        downstream_called = True

    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    mw = _UploadSizeLimitMiddleware(downstream, max_bytes=2 * 1024**3)
    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/ingest",
        "headers": [(b"content-length", str(3 * 1024**3).encode())],
    }
    await mw(scope, receive, send)

    assert downstream_called is False  # short-circuited → no disk stream
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 200  # 200 so HTMX swaps it into the pane
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert b"exceeds the 2 GiB upload limit" in body


async def test_upload_size_limit_middleware_passes_normal_and_other_routes() -> None:
    from memex.webui.app import _UploadSizeLimitMiddleware

    downstream_called = False

    async def downstream(scope: Any, receive: Any, send: Any) -> None:
        nonlocal downstream_called
        downstream_called = True

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        pass

    mw = _UploadSizeLimitMiddleware(downstream, max_bytes=2 * 1024**3)
    # under-cap /ingest passes through; a non-/ingest route passes regardless of declared size.
    scopes: list[dict[str, Any]] = [
        {"type": "http", "method": "POST", "path": "/ingest", "headers": [(b"content-length", b"100")]},
        {"type": "http", "method": "POST", "path": "/ask", "headers": [(b"content-length", str(9 * 1024**3).encode())]},
    ]
    for scope in scopes:
        downstream_called = False
        await mw(scope, receive, send)
        assert downstream_called is True


async def test_scan_half_docs_detects_interrupted_ingest(tmp_path: Path) -> None:
    # B19: the startup scan surfaces interrupted ingests — a manifest with the `ingest` stage but no
    # `index` (parsed/partial but NOT searchable). Detect-only (the caller logs, never auto-deletes).
    from datetime import datetime

    from memex.core.manifest import IndexStage, IngestStage, Manifest, write_manifest
    from memex.webui.app import _scan_half_docs

    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ingest = IngestStage(
        correlation_id="c1",
        ingested_at=ts,
        source_path="/x.pdf",
        source_size_bytes=10,
        detected_mime="application/pdf",
    )
    index = IndexStage(
        correlation_id="c1", indexed_at=ts, embedding_model="m", embedding_dim=8, chunk_count=3
    )
    await write_manifest(tmp_path, Manifest(doc_id="halfdoc", content_sha256="a", ingest=ingest))
    await write_manifest(
        tmp_path, Manifest(doc_id="fulldoc", content_sha256="b", ingest=ingest, index=index)
    )
    await write_manifest(tmp_path, Manifest(doc_id="nostages", content_sha256="c"))

    assert await _scan_half_docs(tmp_path) == ["halfdoc"]  # only the ingested-but-not-indexed one


async def test_scan_half_docs_empty_when_no_manifests(tmp_path: Path) -> None:
    from memex.webui.app import _scan_half_docs

    assert await _scan_half_docs(tmp_path) == []


async def test_scan_half_docs_skips_corrupt_manifest(tmp_path: Path) -> None:
    # B19 robustness (reviewer finding): one corrupt/unreadable manifest must NOT hide the others —
    # `read_manifest` raises a pydantic ValidationError on garbage, so without the per-file guard the
    # whole scan would abort and every other half-doc go unreported.
    from datetime import datetime

    from memex.core.manifest import IngestStage, Manifest, write_manifest
    from memex.webui.app import _scan_half_docs

    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ingest = IngestStage(
        correlation_id="c1",
        ingested_at=ts,
        source_path="/x.pdf",
        source_size_bytes=10,
        detected_mime="application/pdf",
    )
    await write_manifest(tmp_path, Manifest(doc_id="halfdoc", content_sha256="a", ingest=ingest))
    # A corrupt manifest sitting right before "halfdoc" in sorted order — it must be skipped, not fatal.
    (tmp_path / ".memex" / "manifests" / "aaa-corrupt.json").write_text("{ not valid json")

    assert await _scan_half_docs(tmp_path) == ["halfdoc"]  # the corrupt file skipped, half-doc found


async def test_ingesting_lock_set_then_cleared_after_completion(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The lock must be released in _run_ingest's finally — a stuck lock = permanently dead RAG.
    _patch_ingest(
        monkeypatch, outcome=IngestOutcome(accepted=True, exit_code=0, doc_id="abcd1234-lk")
    )
    app = create_app()
    text = await _ingest_to_completion(app, content=b"%PDF", filename="x.pdf")
    assert "Ingested" in text
    assert app.state.ingesting.active is False  # released → the RAG surfaces resume


def test_ingesting_lock_released_on_stream_failure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A non-OSError mid-stream (e.g. a cancelled 2 GiB upload) must NOT leave the exclusive-GPU
    # lock stuck (the ship-blocker the validator found): the try/finally releases it before the
    # exception propagates — no task owns the release yet at that point.
    def _boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("stream exploded")

    monkeypatch.setattr("shutil.copyfileobj", _boom)
    with pytest.raises(RuntimeError):
        client.post("/ingest", files={"file": ("a.pdf", b"%PDF", "application/pdf")})
    assert client.app.state.ingesting.active is False  # released by the finally, not stuck


# ----- Exclusive-GPU ingestion mode: GPU release + orchestrator serve-env (Inc 4) -----


async def test_ingest_injects_orchestrator_serve_env(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The ADR-0015 fix: the subprocess's post-parse vLLM restart must bring up the CONFIGURED
    # orchestrator (not the serve-script default). Assert the driver receives the serve-env so a
    # silent post-ingest 404 storm can't regress unnoticed.
    captured: dict[str, dict[str, str]] = {}

    async def _capture(
        _fp: Path,
        *,
        on_phase: Callable[[str], None],
        extra_env: dict[str, str] | None = None,
        **_kw: object,
    ) -> IngestOutcome:
        captured["env"] = extra_env or {}
        return IngestOutcome(accepted=True, exit_code=0, doc_id="abcd1234-env")

    async def _ok_enrich(_d: str, *, on_phase: Callable[[str], None], **_kw: object) -> int:
        return 0

    _patch_daemon_reachable(monkeypatch)
    monkeypatch.setattr("memex.webui.ingest_driver.run_ingest", _capture)
    monkeypatch.setattr("memex.webui.ingest_driver.run_enrich", _ok_enrich)
    await _ingest_to_completion(create_app(), content=b"%PDF", filename="x.pdf")
    assert "MEMEX_VLLM_MODEL" in captured["env"]  # the orchestrator serve-env was injected


async def test_ingest_cleans_up_temp_upload_dir(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The per-upload temp dir (mkdtemp, named for the original file) must not leak.
    before = set(Path(tempfile.gettempdir()).glob("memex-upload-*"))
    _patch_ingest(
        monkeypatch, outcome=IngestOutcome(accepted=True, exit_code=0, doc_id="abcd1234-clean")
    )
    await _ingest_to_completion(create_app(), content=b"%PDF", filename="x.pdf")
    after = set(Path(tempfile.gettempdir()).glob("memex-upload-*"))
    assert after <= before  # no new memex-upload-* dir lingers after completion


async def test_ingest_drains_inflight_rag_before_unloading_gpu(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # B18: an ingest must let an ALREADY-in-flight RAG answer finish before it unloads the GPU
    # models + the subprocess pauses the orchestrator — else the answer is yanked out mid-run.
    # Prove _drain_inflight_rag awaits the in-flight task BEFORE get_registry().unload_all() runs.
    _patch_ingest(
        monkeypatch,
        outcome=IngestOutcome(accepted=True, exit_code=0, doc_id="abcd1234-doc", chunk_count=5),
    )

    slow_finished = asyncio.Event()
    inflight_done_at_unload: list[bool] = []

    class _Reg:
        async def unload_all(self) -> None:
            # Record whether the in-flight RAG answer had finished by the time the ingest unloads.
            inflight_done_at_unload.append(slow_finished.is_set())

    monkeypatch.setattr("memex.webui.app.get_registry", lambda: _Reg())

    app = create_app()

    async def _slow_answer() -> None:
        await asyncio.sleep(0.1)
        slow_finished.set()

    # Seed a slow in-flight RAG answer under its OWN cid (started before the ingest).
    rag_task = asyncio.create_task(_slow_answer())
    app.state.progress.new("rag-cid", scope_doc_ids=[], scope_source="named")
    app.state.progress.attach_task("rag-cid", rag_task)

    await _ingest_to_completion(app, content=b"%PDF", filename="x.pdf")
    await rag_task

    assert inflight_done_at_unload == [True]  # the drain awaited the in-flight answer first
