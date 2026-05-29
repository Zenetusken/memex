# pyright: reportUnusedFunction=false
# FastAPI route handlers are decorated with `@app.get`/`@app.post`
# which registers them in the ASGI app's route table. Pyright can't
# introspect the decorator's side effect and flags every route
# handler as "not accessed." All 10 routes in this module are
# reached via the FastAPI dispatcher.

"""Local FastAPI + HTMX web UI — see IMPLEMENTATION-PLAN §1.10.

Server-rendered HTML, no SPA build step. HTMX handles the partial
updates for the `/ask` form, the document edit flow, and the
neighbour-graph navigation so the page never reloads.

Routes:

- `GET  /`                              — landing page with the `ask` form
- `POST /ask`                           — runs the answering agent
- `POST /scope-sets`                    — save the ticked docs as a named set
- `POST /scope-sets/apply`              — tick a saved set's docs (re-render)
- `POST /scope-sets/delete`             — delete a saved set (re-render)
- `GET  /resources`                     — active co-residence mode + comparison
- `GET  /documents`                     — document list
- `GET  /documents/{id}`                — markdown render (with PDF
                                          side-by-side when a source is
                                          present)
- `GET  /documents/{id}/source`         — serve the source file with
                                          its detected media-type
- `GET  /documents/{id}/edit`           — HTMX partial: edit textarea
- `GET  /documents/{id}/body`           — HTMX partial: view-mode body
- `POST /documents/{id}/review`         — accept an edit, write through
                                          `vault.write_document`, return
                                          the updated body partial
- `GET  /graph/{id}`                    — Cytoscape one-hop neighbourhood
- `GET  /healthz`                       — daemon-status probe target
"""

from __future__ import annotations

import asyncio
import mimetypes
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
import ulid
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader

from memex.agents.answering import FinalResponse, answer_query
from memex.agents.document_summarizer import SummaryDetail, summarize_document
from memex.core.config import get_settings
from memex.core.errors import (
    MemexError,
    ScopeSetError,
    StaleDocumentError,
    VaultIntegrityError,
)
from memex.core.manifest import update_manifest
from memex.core.resources import CoResidenceMode, ResourceProfile, all_modes, resolve_profile
from memex.core.scope_sets import (
    delete_scope_set,
    get_scope_set,
    list_scope_sets,
    save_scope_set,
)
from memex.daemon import restart as daemon_restart
from memex.daemon import status as daemon_status
from memex.index.graph_store import GraphStore
from memex.index.pipeline import retitle_document
from memex.models.registry import ModelNotConfigured, get_registry

# webui → parse boundary edge (documented, like webui → daemon / models.registry):
# the source-preview pane rasterises PDF/Office pages to images server-side — the
# Phase-4 "side-by-side preview" job (IMPLEMENTATION-PLAN §1.10). `pdf_render` is the
# LIGHT pypdfium2-only renderer (no ML/Docling deps), so this import stays cheap.
from memex.parse.pdf_render import (
    PDFPreviewError,
    pdf_page_count,
    pdf_page_size,
    render_pdf_page_png,
)
from memex.vault.store import (
    VaultDocument,
    hash_bytes,
    list_documents,
    make_ref,
    read_document,
    read_document_title,
    write_document,
)
from memex.webui.progress import (
    PHASES,
    SUMMARY_PHASES,
    ProgressRegistry,
    phase_for,
    summary_phase_view,
)
from memex.webui.rendering import (
    clean_heading_text,
    extract_toc,
    render_body_html,
    render_wikilink,
    slugify_heading,
)

_HERE = Path(__file__).parent
_TEMPLATES_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"

# Mirrors `vault.store.assign_doc_id`'s output shape: 8-hex prefix
# optionally followed by `-` and a slug of [a-z0-9-]. Routes reject
# anything else — defence in depth against path-traversal via crafted
# {doc_id} segments (`..`, `%2F`, etc.) ever slipping past Starlette's
# default decoding.
_DOC_ID_RE = re.compile(r"^[0-9a-f]{8}(-[a-z0-9-]+)?$")

# Cap on UI form inputs. Question text is short by nature; body edits
# can be longer but should not exceed the ingest size cap (a body
# bigger than the largest accepted source is almost certainly a
# misbehaving client). Both are checked at request time.
_QUESTION_MAX_BYTES = 4_096
_BODY_MAX_BYTES = 16 * 1024 * 1024
_TITLE_MAX_LEN = 300


def _validate_doc_id(doc_id: str) -> str:
    """Guard every `{doc_id}` route param against traversal.

    Raises HTTPException(404) if `doc_id` doesn't match the canonical
    format `vault.assign_doc_id` produces. Returning 404 (not 400)
    avoids leaking whether the rejected path "might" exist somewhere
    on disk.
    """
    if not _DOC_ID_RE.match(doc_id):
        raise HTTPException(status_code=404, detail="document not found")
    return doc_id


# Map a `source.<ext>` filename to the source-kind label rendered in the
# pane header and to the HTTP media-type. Anything not in this map is
# served as `application/octet-stream` and rendered as "download only".
_SOURCE_KINDS: dict[str, tuple[str, str]] = {
    ".pdf": ("pdf", "application/pdf"),
    ".md": ("markdown", "text/markdown"),
    ".markdown": ("markdown", "text/markdown"),
    ".txt": ("text", "text/plain"),
    ".html": ("html", "text/html"),
    ".htm": ("html", "text/html"),
    ".docx": (
        "docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    ".pptx": (
        "pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ),
    ".xlsx": (
        "xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
}

logger = structlog.get_logger(__name__)


def _active_profile() -> ResourceProfile:
    """The active co-residence ResourceProfile (ADR-0007), resolved from the
    current settings (mode + the manual device knobs)."""
    s = get_settings()
    return resolve_profile(
        s.models.co_residence_mode,
        embedder_device=s.models.embedder_device,
        reranker_device=s.models.reranker_device,
    )


async def _apply_mode(mode: CoResidenceMode) -> tuple[ResourceProfile, str]:
    """Live co-residence mode switch (ADR-0007's runtime-transition). Two halves:

    1. **App-side (this process):** flip `settings.models.co_residence_mode` — the
       registry shares that exact `ModelSettings` object (`bootstrap` built it with
       `get_settings().models`), so the change is visible to it — then `unload` the
       embedder + reranker so the next retrieval reloads them on the new mode's
       device. `registry.unload` takes the per-model lock, so an in-flight `/ask`
       finishes on the old model before the swap (the quiesce).
    2. **Orchestrator-side (the daemon):** if the mode prescribes a posture
       (`orchestrator_gpu_fraction is not None`) and the daemon is alive, restart
       vLLM at the mode's util + context window (blocks ~40 s).

    Returns the new profile + a human note. Raises `MemexError` on a daemon failure
    (the route flashes it)."""
    s = get_settings()
    s.models.co_residence_mode = mode
    profile = resolve_profile(
        mode,
        embedder_device=s.models.embedder_device,
        reranker_device=s.models.reranker_device,
    )
    # Drop the retrieval models so the next use reloads them on the new device. The
    # registry may not be initialised (e.g. nothing loaded yet) — that's a no-op.
    try:
        registry = get_registry()
        await registry.unload("embedder")
        await registry.unload("reranker")
    except ModelNotConfigured:
        logger.info("resources.mode.registry_absent")

    if profile.orchestrator_gpu_fraction is None:
        return profile, "retrieval device applied; this mode leaves the orchestrator as launched."
    state = await daemon_status(s)
    if not state.alive:
        return profile, (
            "retrieval device applied; the orchestrator isn't daemon-managed here — "
            "run `memex daemon start` to apply its util/context window."
        )
    new_state = await daemon_restart(
        s,
        gpu_fraction=profile.orchestrator_gpu_fraction,
        max_model_len=profile.orchestrator_max_model_len,
    )
    return profile, (
        f"orchestrator restarted at util {profile.orchestrator_gpu_fraction} / "
        f"context {profile.orchestrator_max_model_len} "
        f"(reachable={new_state.reachable})."
    )


def _find_source(vault_path: Path, doc_id: str) -> Path | None:
    """Locate the original `source.<ext>` for a doc, if one was copied
    in by the ingest stage. Markdown-passthrough docs have no source
    file — they ARE the source."""
    asset_dir = vault_path / "documents" / doc_id
    if not asset_dir.is_dir():
        return None
    candidates = sorted(asset_dir.glob("source.*"))
    return candidates[0] if candidates else None


def _find_preview_pdf(vault_path: Path, doc_id: str) -> Path | None:
    """The PDF to rasterise in the source-preview pane, for ANY visual source
    type: a PDF doc's own `source.pdf`, or — for an Office/ODF doc — the
    `converted.pdf` the parse stage produced (the original `.pptx`/`.docx`
    can't render inline; the converted PDF is exactly what was parsed). A
    markdown-passthrough / text source has neither → no preview pane."""
    asset_dir = vault_path / "documents" / doc_id
    source_pdf = asset_dir / "source.pdf"
    if source_pdf.is_file():
        return source_pdf
    converted = asset_dir / "converted.pdf"
    if converted.is_file():
        return converted
    return None


def _source_view(response: FinalResponse) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """View-model for rendering answer/summary sources by HUMAN TITLE instead of
    the raw `docid#hash` / `[[doc#section]]` syntax. Returns:

    - `chunk_refs`: `chunk_id → {title, section, href, page}` for the per-claim
      source chips (a claim whose chunk isn't here — e.g. a synthetic table/SQL
      chunk — falls back to the raw id in the template). `page` is `""` when
      unknown (legacy chunks indexed before the chunker's page-attribution lever
      shipped, OR a doc parsed without per-page char counts in its manifest);
      otherwise the 1-based source page number, threaded into the href as
      `?page=N#section-slug` so the doc-page template can scroll the PDF
      preview pane to that page on landing.
    - `doc_titles`: `doc_id → title` for the `render_wikilink` Sources labels.

    Built from the cited chunks' OWN `document_title` + `heading_path` + `page` —
    no extra I/O (the same data the refusal panel already shows)."""
    chunk_refs: dict[str, dict[str, str]] = {}
    doc_titles: dict[str, str] = {}
    for c in response.used_chunks:
        # Clean inline-markdown (`**bold**`, `[x](url)`, `` `code` ``) out of the
        # heading text — a parsed heading like `**Zero Trust Architecture**` would
        # otherwise show its literal asterisks in the source chip. Same cleaner the
        # TOC uses; the slug (href) is derived independently via `slugify_heading`.
        section = clean_heading_text(c.heading_path[-1]) if c.heading_path else ""
        page_str = str(c.page) if c.page is not None else ""
        href = f"/documents/{c.document_id}"
        if page_str:
            href = f"{href}?page={page_str}"
        if section:
            href = f"{href}#{slugify_heading(section)}"
        title = c.document_title or c.document_id
        chunk_refs[c.chunk_id] = {
            "title": title,
            "section": section,
            "href": href,
            "page": page_str,
        }
        doc_titles[c.document_id] = title
    return chunk_refs, doc_titles


async def _answer_context(
    vault_path: Path, response: FinalResponse, scope_source: str
) -> dict[str, object]:
    """Build the `_answer.html` context from a FinalResponse — scope-doc titles
    (#256), the source-by-title view-model, and the scope-source label. Shared by
    the long-poll status route so the rendered answer is identical to the old
    synchronous path."""
    scope_docs = [
        {"doc_id": d, "title": await read_document_title(vault_path, d)}
        for d in response.artifact_scope_doc_ids
    ]
    chunk_refs, doc_titles = _source_view(response)
    return {
        "response": response,
        "error": None,
        "scope_docs": scope_docs,
        "scope_source": scope_source,
        "chunk_refs": chunk_refs,
        "doc_titles": doc_titles,
    }


def _kind_for(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()
    if ext in _SOURCE_KINDS:
        return _SOURCE_KINDS[ext]
    # Fall back to mimetypes for unmapped extensions; render label as
    # the bare extension stripped of the dot.
    media, _ = mimetypes.guess_type(str(path))
    return (ext.lstrip(".") or "unknown", media or "application/octet-stream")


def register_template_filter(env: Environment, name: str, func: Callable[..., Any]) -> None:
    """Register a Jinja2 filter on `env` under `name`.

    Localises the one unavoidable type-ignore: jinja2's
    `Environment.filters` has no class-level annotation (it's assigned
    `DEFAULT_FILTERS.copy()` in `__init__`), so pyright reports the
    member as unknown under --strict. jinja2 ships `py.typed`, so we
    don't shadow it with a competing stub (per src/memex/CLAUDE.md) —
    this wrapper keeps the ignore out of the route-heavy factory body.
    """
    env.filters[name] = func  # type: ignore[reportUnknownMemberType]  # reason: jinja2 leaves Environment.filters unannotated


def create_app() -> FastAPI:
    """Build the FastAPI app. Factory so tests can instantiate freely."""
    # Build the jinja env explicitly (typed local) so the wikilink
    # filter registration is pyright-clean — `Jinja2Templates.env`
    # resolves to Unknown under --strict (starlette assigns it via an
    # internal helper). Mirrors starlette's own defaults (FileSystemLoader
    # + autoescape) and registers the answer "Sources" `render_wikilink`
    # filter before handing the env to Jinja2Templates.
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )
    # jinja2's `Environment.filters` carries no class-level annotation
    # (it's set as `DEFAULT_FILTERS.copy()` in __init__), so pyright sees
    # the member as unknown under --strict. jinja2 ships `py.typed`, so we
    # don't shadow it with a competing stub (per src/memex/CLAUDE.md) —
    # `register_template_filter` localises the one unavoidable ignore.
    register_template_filter(env, "render_wikilink", render_wikilink)
    # The active co-residence mode (ADR-0007) is fixed for the process lifetime
    # (set at launch; switching needs a restart), so expose its label as a
    # template global for the header chip — every page can show it without each
    # route threading it through its context.
    env.globals["active_mode_label"] = _active_profile().label  # type: ignore[reportUnknownMemberType]  # reason: jinja2 leaves Environment.globals unannotated
    templates = Jinja2Templates(env=env)
    app = FastAPI(title="Memex", docs_url=None, redoc_url=None)

    if _STATIC_DIR.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(_STATIC_DIR)),
            name="static",
        )

    # ----- Landing + ask -----

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        """Landing page — the ask form + the optional document scope-picker
        (Notebook-LM-style: tick documents to scope the question to them, or
        apply a saved scope set)."""
        settings = get_settings()
        ctx = await _scope_picker_context(
            settings.vault_path, checked_ids=[], flash=None, picker_open=False
        )
        return templates.TemplateResponse(request, "index.html", ctx)

    # ----- Saved scope sets (persist + reapply a document selection) -----
    # Each route re-renders the `_scope_picker.html` partial (the HTMX swap
    # target #scope-picker) so the saved-set bar + ticked boxes update in place
    # without a page reload. `/ask` is unchanged — it still reads `scope_doc_ids`
    # from the ticked checkboxes; "apply" just pre-ticks them server-side.

    @app.post("/scope-sets", response_class=HTMLResponse)
    async def scope_set_save(
        request: Request,
        set_name: str = Form(""),
        scope_doc_ids: list[str] = Form([]),  # noqa: B008  # FastAPI Form default sentinel
    ) -> HTMLResponse:
        """Save the currently-ticked documents as a named scope set."""
        settings = get_settings()
        checked = scope_doc_ids
        try:
            record = await save_scope_set(settings.vault_path, set_name, scope_doc_ids)
            n = len(record.doc_ids)
            flash = {
                "kind": "ok",
                "text": f"Saved “{record.name}” ({n} document{'' if n == 1 else 's'}).",
            }
            checked = record.doc_ids
        except (ScopeSetError, VaultIntegrityError) as e:
            flash = {"kind": "error", "text": str(e)}
        ctx = await _scope_picker_context(
            settings.vault_path, checked_ids=checked, flash=flash, picker_open=True
        )
        return templates.TemplateResponse(request, "_scope_picker.html", ctx)

    @app.post("/scope-sets/apply", response_class=HTMLResponse)
    async def scope_set_apply(
        request: Request,
        name: str = Form(""),
    ) -> HTMLResponse:
        """Tick a saved set's documents (server-side) so a normal Ask submit
        scopes to them. Replaces the current selection."""
        settings = get_settings()
        checked: list[str] = []
        try:
            found = await get_scope_set(settings.vault_path, name)
        except VaultIntegrityError as e:
            flash = {"kind": "error", "text": str(e)}
        else:
            if found is None:
                flash = {"kind": "error", "text": f"No saved set named “{name}”."}
            else:
                n = len(found.doc_ids)
                flash = {
                    "kind": "ok",
                    "text": f"Applied “{found.name}” — {n} document{'' if n == 1 else 's'} ticked.",
                }
                checked = found.doc_ids
        ctx = await _scope_picker_context(
            settings.vault_path, checked_ids=checked, flash=flash, picker_open=True
        )
        return templates.TemplateResponse(request, "_scope_picker.html", ctx)

    @app.post("/scope-sets/delete", response_class=HTMLResponse)
    async def scope_set_remove(
        request: Request,
        name: str = Form(""),
    ) -> HTMLResponse:
        """Delete a saved scope set. Removes only the named collection."""
        settings = get_settings()
        try:
            removed = await delete_scope_set(settings.vault_path, name)
            flash = {
                "kind": "ok" if removed else "error",
                "text": (f"Deleted “{name}”." if removed else f"No saved set named “{name}”."),
            }
        except VaultIntegrityError as e:
            flash = {"kind": "error", "text": str(e)}
        ctx = await _scope_picker_context(
            settings.vault_path, checked_ids=[], flash=flash, picker_open=True
        )
        return templates.TemplateResponse(request, "_scope_picker.html", ctx)

    # Live-progress registry for in-flight asks (long-poll). One per app, like
    # `mode_switch_lock`; single-worker uvicorn → an in-process dict is safe.
    # Exposed on app.state so tests can pre-seed/inspect entries.
    progress = ProgressRegistry()
    app.state.progress = progress

    async def _run_ask(cid: str, question: str, scope_doc_ids: list[str]) -> None:
        """Background runner: drive the agent, stream node→phase updates into the
        registry, and store the result (or error) for the status poll. This is the
        top of a fire-and-forget task — it must never crash silently."""
        from memex.core.errors import MemexError

        try:
            response = await answer_query(
                question,
                scope_doc_ids=scope_doc_ids,
                correlation_id=cid,
                on_node=lambda n: progress.set_phase(cid, phase_for(n)),
            )
            progress.finish(cid, response=response)
        except MemexError as e:
            # Typed agent errors (schema-violating LLM output, OOM, tripped
            # breaker, …) → the same friendly banner the synchronous route showed.
            logger.warning("ask.failed", error_type=type(e).__name__, error=str(e)[:200])
            progress.finish(cid, error=f"Couldn't answer: {type(e).__name__}. {str(e)[:160]}")
        except Exception as e:
            # reason: top-of-task boundary — surface as a generic error + log,
            # rather than an unretrieved task exception. CancelledError (a
            # BaseException) is intentionally NOT caught, so cancellation still
            # propagates and tears the task down cleanly.
            logger.error("ask.crashed", error_type=type(e).__name__, error=str(e)[:200])
            progress.finish(cid, error="An unexpected error occurred while answering.")

    @app.post("/ask", response_class=HTMLResponse)
    async def ask(
        request: Request,
        question: str = Form(..., max_length=_QUESTION_MAX_BYTES),
        scope_doc_ids: list[str] = Form([]),  # noqa: B008  # FastAPI Form default sentinel
    ) -> HTMLResponse:
        """Start the answering agent in a background task and IMMEDIATELY return
        the `_progress.html` fragment, which long-polls `/ask/{cid}/status` for the
        live step (Retrieving → … → Composing) until the answer is ready."""
        question = question.strip()
        if not question:
            return templates.TemplateResponse(
                request,
                "_answer.html",
                {"response": None, "error": "Question is empty."},
                status_code=400,
            )
        cid = str(ulid.ULID())
        scope_source = "selected" if scope_doc_ids else "named"
        progress.new(cid, scope_doc_ids=scope_doc_ids, scope_source=scope_source)
        task = asyncio.create_task(_run_ask(cid, question, scope_doc_ids))
        progress.attach_task(cid, task)  # strong ref → the loop won't GC the task mid-run
        return templates.TemplateResponse(
            request,
            "_progress.html",
            {
                "poll_url": f"/ask/{cid}/status?v=0",
                "phases": PHASES,
                "active_index": 0,
                "elapsed": 0,
                "detail": "",
            },
        )

    @app.get("/ask/{cid}/status", response_class=HTMLResponse)
    async def ask_status(request: Request, cid: str, v: int = 0) -> HTMLResponse:
        """Long-poll the in-flight ask: block until the phase advances past `v`,
        the run finishes, or a ~1 s keepalive — so the indicator updates the INSTANT
        a node transition happens (SSE-like), while a held phase still ticks its
        elapsed timer. Always HTTP 200; the done / expired fragments carry no poll
        trigger, so polling stops on its own."""
        entry = await progress.wait_for_change(cid, v)
        if entry is None:  # swept / unknown cid (e.g. a server restart mid-poll)
            return templates.TemplateResponse(request, "_progress_expired.html", {})
        if not entry.done:
            return templates.TemplateResponse(
                request,
                "_progress.html",
                {
                    "poll_url": f"/ask/{cid}/status?v={entry.version}",
                    "phases": PHASES,
                    "active_index": entry.active_index(),
                    "elapsed": entry.phase_elapsed_s(),
                    "detail": "",
                },
            )
        # Done → render the final answer exactly like the old synchronous route.
        progress.evict(cid)
        if entry.error is not None or entry.response is None:
            return templates.TemplateResponse(
                request,
                "_answer.html",
                {"response": None, "error": entry.error or "Answering produced no result."},
            )
        settings = get_settings()
        ctx = await _answer_context(settings.vault_path, entry.response, entry.scope_source)
        return templates.TemplateResponse(request, "_answer.html", ctx)

    # ----- Resource mode (ADR-0007) -----

    # Serialize live mode switches — two concurrent daemon restarts would race.
    mode_switch_lock = asyncio.Lock()

    def _resources_ctx(
        *, flash: str | None = None, flash_error: bool = False, oob_chip: bool = False
    ) -> dict[str, object]:
        # `oob_chip` emits an out-of-band swap for the header mode chip — only on the
        # POST (a live switch), so the GET full-page render has no duplicate id.
        return {
            "active": _active_profile(),
            "modes": all_modes(),
            "flash": flash,
            "flash_error": flash_error,
            "oob_chip": oob_chip,
        }

    @app.get("/resources", response_class=HTMLResponse)
    async def resources(request: Request) -> HTMLResponse:
        """The active co-residence mode + the full mode comparison (ADR-0007), with
        a live per-mode Apply switch (POST /resources/mode)."""
        return templates.TemplateResponse(request, "resources.html", _resources_ctx())

    @app.post("/resources/mode", response_class=HTMLResponse)
    async def resources_set_mode(request: Request, mode: str = Form(...)) -> HTMLResponse:
        """Live co-residence mode switch (ADR-0007 runtime transition) — swaps the
        retrieval device (unload → lazy reload on the new device) and restarts the
        orchestrator at the mode's util/context window. Serialized; the daemon
        restart blocks ~40 s (the form shows an indicator). Returns the
        `_resources.html` partial (HTMX swap) reflecting the new active profile."""
        valid = ("fast", "full", "gpu_only", "manual")
        if mode not in valid:
            return templates.TemplateResponse(
                request,
                "_resources.html",
                _resources_ctx(flash=f"Unknown mode {mode!r}.", flash_error=True),
                status_code=400,
            )
        async with mode_switch_lock:
            try:
                profile, note = await _apply_mode(mode)
            except MemexError as exc:
                logger.warning("resources.mode.failed", mode=mode, error=str(exc)[:200])
                return templates.TemplateResponse(
                    request,
                    "_resources.html",
                    _resources_ctx(flash=f"Switch to {mode!r} failed: {exc}", flash_error=True),
                    status_code=503,
                )
            # Keep the header chip (a static jinja global, set once at startup) current.
            env.globals["active_mode_label"] = profile.label  # type: ignore[reportUnknownMemberType]  # reason: jinja2 leaves Environment.globals unannotated
            logger.info("resources.mode.switched", mode=mode)
            return templates.TemplateResponse(
                request,
                "_resources.html",
                _resources_ctx(flash=f"Switched to {profile.label}. {note}", oob_chip=True),
            )

    # ----- Documents -----

    @app.get("/documents", response_class=HTMLResponse)
    async def documents(request: Request) -> HTMLResponse:
        """List every document in the vault. Each row shows the
        frontmatter title (read cheaply via `read_document_title` —
        frontmatter block only, not the full body) with the doc_id +
        sha as the monospace secondary line."""
        settings = get_settings()
        docs: list[dict[str, str]] = []
        async for ref in list_documents(settings.vault_path):
            title = await read_document_title(settings.vault_path, ref.doc_id)
            docs.append(
                {
                    "doc_id": ref.doc_id,
                    "title": title,
                    "content_sha256": ref.content_sha256,
                }
            )
        # Sort by title for a stable, human-scannable listing.
        docs.sort(key=lambda d: d["title"].casefold())
        return templates.TemplateResponse(request, "documents.html", {"documents": docs})

    @app.get("/documents/{doc_id}", response_class=HTMLResponse)
    async def document(request: Request, doc_id: str) -> HTMLResponse:
        """Render the extracted markdown body for `doc_id`. When a
        `source.{pdf,png,...}` file is present, the template offers a
        side-by-side preview via `/documents/{doc_id}/source`."""
        doc_id = _validate_doc_id(doc_id)
        settings = get_settings()
        try:
            doc = await read_document(settings.vault_path, doc_id)
        except VaultIntegrityError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        source = _find_source(settings.vault_path, doc_id)
        if source is not None:
            source_kind, _ = _kind_for(source)
            has_source = True
        else:
            source_kind = None
            has_source = False
        # The preview pane rasterises a PDF to page images (the doc's own
        # source.pdf, or an Office doc's converted.pdf) — a server-side render
        # that works for every visual source type regardless of the browser's
        # PDF handling. A corrupt/unreadable PDF degrades to no-pane (never 500s).
        preview_pdf = _find_preview_pdf(settings.vault_path, doc_id)
        preview_pages = 0
        # Page-0 dimensions feed the placeholder `<img>` aspect-ratio so the
        # browser-native `loading="lazy"` actually defers offscreen pages — a
        # 0-height placeholder reads as "near viewport" and fires immediately,
        # which is what made all 49 deck pages render on initial load. Most PDFs
        # are uniform-sized so page 0 is a fair stand-in for the deck; a `None`
        # fallback (in the template) gives a sane 8.5/11 placeholder ratio.
        preview_aspect: str | None = None
        if preview_pdf is not None:
            try:
                # In a thread: both calls hold the pypdfium2 lock, so keep them
                # off the event loop (a concurrent page render may hold it ~200ms).
                preview_pages = await asyncio.to_thread(pdf_page_count, preview_pdf)
            except PDFPreviewError:
                logger.warning("document.preview_unreadable", doc_id=doc_id, pdf=str(preview_pdf))
                preview_pdf = None
                preview_pages = 0
            else:
                if preview_pages > 0:
                    try:
                        w, h = await asyncio.to_thread(pdf_page_size, preview_pdf, 0)
                        preview_aspect = f"{w:.3f} / {h:.3f}"
                    except PDFPreviewError:
                        # Non-fatal: the pages render fine; the CSS 8.5/11 fallback
                        # just gives a Letter-shaped placeholder for the unknown ratio.
                        logger.warning("document.preview_size_unreadable", doc_id=doc_id)
        # Related documents ("explore connections") — entity-specificity-ranked graph
        # discovery. Optional: a missing/unavailable graph fails OPEN to no section
        # (never 500s the doc view), mirroring the /graph route.
        related: list[dict[str, Any]] = []
        try:
            rstore = await GraphStore.open(settings.vault_path)
        except ImportError as e:
            logger.warning("webui.related_unavailable", doc_id=doc_id, reason=str(e))
        else:
            try:
                related = [r.model_dump() for r in await rstore.related_documents(doc_id, limit=8)]
            finally:
                await rstore.close()
        return templates.TemplateResponse(
            request,
            "document.html",
            {
                "document": doc,
                "rendered_body": render_body_html(doc.body),
                "toc": extract_toc(doc.body),
                "has_source": has_source,
                "source_kind": source_kind,
                "has_preview": preview_pdf is not None and preview_pages > 0,
                "preview_pages": preview_pages,
                "preview_aspect": preview_aspect,
                "related": related,
            },
        )

    async def _run_summarize(cid: str, doc_id: str, level: SummaryDetail) -> None:
        """Background runner: drive the summarizer, streaming phase updates into the
        registry (incl. the per-section counter), and store the result/error for the
        status poll. Top of a fire-and-forget task — never crash silently."""
        from memex.core.errors import MemexError

        try:
            response = await summarize_document(
                doc_id,
                detail=level,
                correlation_id=cid,
                on_phase=lambda p: progress.set_phase(cid, p),
            )
            progress.finish(cid, response=response)
        except MemexError as e:
            logger.warning("summarize.failed", error_type=type(e).__name__, error=str(e)[:200])
            progress.finish(cid, error=f"Couldn't summarise: {type(e).__name__}. {str(e)[:160]}")
        except Exception as e:
            logger.error("summarize.crashed", error_type=type(e).__name__, error=str(e)[:200])
            progress.finish(cid, error="An unexpected error occurred while summarising.")

    @app.post("/documents/{doc_id}/summarize", response_class=HTMLResponse)
    async def document_summarize(
        request: Request,
        doc_id: str,
        detail: str = Form("standard"),
    ) -> HTMLResponse:
        """Start the grounded summary (ADR-0008) in a background task and IMMEDIATELY
        return the `_progress.html` fragment, which long-polls
        `/documents/{id}/summarize/status` for the live phase ("Summarizing · section
        k of N" → "Reducing" → …) until the summary swaps in."""
        doc_id = _validate_doc_id(doc_id)
        level: SummaryDetail = (
            detail if detail in ("brief", "standard", "detailed", "report") else "standard"
        )
        cid = str(ulid.ULID())
        progress.new(cid, scope_doc_ids=[], scope_source="named")
        task = asyncio.create_task(_run_summarize(cid, doc_id, level))
        progress.attach_task(cid, task)
        return templates.TemplateResponse(
            request,
            "_progress.html",
            {
                "poll_url": f"/documents/{doc_id}/summarize/status?cid={cid}&v=0",
                "phases": SUMMARY_PHASES,
                "active_index": 0,
                "elapsed": 0,
                "detail": "",
            },
        )

    @app.get("/documents/{doc_id}/summarize/status", response_class=HTMLResponse)
    async def summarize_status(
        request: Request, doc_id: str, cid: str = "", v: int = 0
    ) -> HTMLResponse:
        """Long-poll the in-flight summary: block until the phase advances past `v`,
        the run finishes, or a ~1 s keepalive — then render `_progress.html` (current
        phase + the section counter as the eyebrow detail) or, when done,
        `_summary.html`. Always HTTP 200; done/expired carry no poll trigger."""
        entry = await progress.wait_for_change(cid, v)
        if entry is None:
            return templates.TemplateResponse(request, "_progress_expired.html", {})
        if not entry.done:
            active_index, detail = summary_phase_view(entry.phase)
            return templates.TemplateResponse(
                request,
                "_progress.html",
                {
                    "poll_url": f"/documents/{doc_id}/summarize/status?cid={cid}&v={entry.version}",
                    "phases": SUMMARY_PHASES,
                    "active_index": active_index,
                    "elapsed": entry.phase_elapsed_s(),
                    "detail": detail,
                },
            )
        progress.evict(cid)
        if entry.error is not None or entry.response is None:
            return templates.TemplateResponse(
                request,
                "_summary.html",
                {"response": None, "error": entry.error or "Summarising produced no result."},
            )
        chunk_refs, doc_titles = _source_view(entry.response)
        return templates.TemplateResponse(
            request,
            "_summary.html",
            {
                "response": entry.response,
                "error": None,
                "chunk_refs": chunk_refs,
                "doc_titles": doc_titles,
            },
        )

    @app.get("/documents/{doc_id}/source")
    async def document_source(doc_id: str) -> FileResponse:
        """Serve the original source file (PDF / image) with the
        correct Content-Type so the browser can render it inline
        in the side-by-side preview panel."""
        doc_id = _validate_doc_id(doc_id)
        settings = get_settings()
        source = _find_source(settings.vault_path, doc_id)
        if source is None:
            raise HTTPException(
                status_code=404,
                detail=f"no source file for doc_id={doc_id!r}",
            )
        _, media_type = _kind_for(source)
        return FileResponse(
            source,
            media_type=media_type,
            filename=source.name,
            # "inline", NOT FileResponse's default "attachment" — otherwise the
            # browser downloads the file instead of rendering it in the
            # side-by-side <iframe> preview, leaving the pane BLANK. The
            # template's "download" link forces a download via the HTML
            # `download` attribute, so the download path still works.
            content_disposition_type="inline",
        )

    @app.get("/documents/{doc_id}/source/page/{page}")
    async def document_source_page(doc_id: str, page: int) -> Response:
        """Rasterise one **0-based** page of the doc's preview PDF (its own
        `source.pdf`, or an Office doc's `converted.pdf`) to a PNG. The preview
        pane shows one `<img>` per page — a SERVER-side render, so the original
        page always displays inline regardless of the browser's "download PDFs"
        setting or iframe-PDF quirks (which left the old `<iframe>` blank). The
        render is CPU-bound, so it runs in a thread; the bytes are content-stable
        so the browser caches them."""
        doc_id = _validate_doc_id(doc_id)
        settings = get_settings()
        preview_pdf = _find_preview_pdf(settings.vault_path, doc_id)
        if preview_pdf is None:
            raise HTTPException(status_code=404, detail=f"no previewable PDF for doc_id={doc_id!r}")
        try:
            png = await asyncio.to_thread(render_pdf_page_png, preview_pdf, page)
        except PDFPreviewError as e:
            # out-of-range page or a render failure → 404 (the <img> just won't load)
            raise HTTPException(status_code=404, detail=str(e)) from e
        return Response(
            content=png,
            media_type="image/png",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.get("/documents/{doc_id}/edit", response_class=HTMLResponse)
    async def document_edit(request: Request, doc_id: str) -> HTMLResponse:
        """Render the document-body editor (HTMX-driven). Save POSTs to
        `/documents/{doc_id}/body` which round-trips through the vault
        CAS write path."""
        doc_id = _validate_doc_id(doc_id)
        settings = get_settings()
        try:
            doc = await read_document(settings.vault_path, doc_id)
        except VaultIntegrityError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return templates.TemplateResponse(
            request,
            "_document_edit.html",
            {"document": doc},
        )

    @app.get("/documents/{doc_id}/body", response_class=HTMLResponse)
    async def document_body(request: Request, doc_id: str) -> HTMLResponse:
        """View-mode partial — what `cancel` swaps back to."""
        doc_id = _validate_doc_id(doc_id)
        settings = get_settings()
        try:
            doc = await read_document(settings.vault_path, doc_id)
        except VaultIntegrityError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return templates.TemplateResponse(
            request,
            "_document_body.html",
            {
                "document": doc,
                "rendered_body": render_body_html(doc.body),
                "just_saved": None,
            },
        )

    @app.post("/documents/{doc_id}/review", response_class=HTMLResponse)
    async def document_review(
        request: Request,
        doc_id: str,
        body: str = Form(..., max_length=_BODY_MAX_BYTES),
        expected_sha: str = Form(..., min_length=64, max_length=64),
    ) -> HTMLResponse:
        """Apply an edit. Two layered concurrency stories:

        1. **Optimistic compare-and-swap (P1.4).** The form submits
           `expected_sha`, the sha the user's draft was based on.
           `write_document(expected_sha=...)` reads the current
           on-disk sha inside the per-doc lock and raises
           `StaleDocumentError` on mismatch. On stale, we render a
           409 conflict panel with a unified diff and "discard /
           overwrite" buttons (`_review_conflict.html`).

        2. **Manifest-before-write race (audit fix).** We pre-update
           the manifest's `content_sha256` with the anticipated
           post-write hash BEFORE the atomic file rename. A kill
           between manifest-update and file-write leaves the manifest
           claiming the NEW sha while the file is still OLD — on
           restart, `_confirm_user_edit` sees the mismatch and
           re-triggers enrich/index, which is the correct recovery.
        """
        doc_id = _validate_doc_id(doc_id)
        settings = get_settings()
        try:
            existing = await read_document(settings.vault_path, doc_id)
        except VaultIntegrityError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        new_doc = VaultDocument(
            ref=make_ref(
                settings.vault_path,
                doc_id,
                content_sha256=hash_bytes(body.encode("utf-8")),
                source_path=existing.ref.source_path,
            ),
            frontmatter=existing.frontmatter,
            body=body,
            mtime_ns=existing.mtime_ns,
        )
        anticipated = _anticipated_sha(new_doc)
        await update_manifest(
            settings.vault_path,
            doc_id,
            content_sha256=anticipated,
        )
        try:
            new_ref = await write_document(settings.vault_path, new_doc, expected_sha=expected_sha)
        except StaleDocumentError as e:
            # Roll back the optimistic manifest update — the file
            # didn't change, so the manifest must match the current
            # on-disk sha (not the anticipated one).
            current_sha = e.context["current_sha"]
            current_body = e.context["current_body"]
            if isinstance(current_sha, str):
                await update_manifest(
                    settings.vault_path,
                    doc_id,
                    content_sha256=current_sha,
                )
            logger.info(
                "webui.document_review.conflict",
                doc_id=doc_id,
                expected_sha=expected_sha[:16],
                current_sha=current_sha[:16] if isinstance(current_sha, str) else None,
                draft_size=len(body),
            )
            return _render_conflict(
                request,
                templates=templates,
                doc_id=doc_id,
                user_draft=body,
                current_body=current_body if isinstance(current_body, str) else "",
                expected_sha=expected_sha,
                current_sha=current_sha if isinstance(current_sha, str) else None,
            )

        if new_ref.content_sha256 != anticipated:
            # Frontmatter round-trip changed the serialized bytes
            # (e.g. yaml reorder, quoting). Reconcile so the manifest
            # tracks the canonical post-write sha.
            await update_manifest(
                settings.vault_path,
                doc_id,
                content_sha256=new_ref.content_sha256,
            )

        # Re-read so the template sees the canonical post-write state
        # (frontmatter is round-tripped through python-frontmatter).
        refreshed = await read_document(settings.vault_path, doc_id)

        logger.info(
            "webui.document_saved",
            doc_id=doc_id,
            new_sha=new_ref.content_sha256[:16],
            new_size=len(body),
        )
        return templates.TemplateResponse(
            request,
            "_document_body.html",
            {
                "document": refreshed,
                "rendered_body": render_body_html(refreshed.body),
                "just_saved": datetime.now().strftime("%H:%M:%S"),
            },
        )

    # ----- Title rename (inline, metadata-only) -----
    # These call `index.retitle_document` directly. The webui's normal
    # boundary is agents/vault/core, but a rename must fan the new title
    # out to the FTS/vector/graph copies *without* a re-embed — the
    # watcher's partial reindex can't, since the body (and thus every
    # chunk) is unchanged. `retitle_document` is the sanctioned write
    # path for that, the same way `vault.write_document` is for body edits.

    @app.get("/documents/{doc_id}/title", response_class=HTMLResponse)
    async def document_title(request: Request, doc_id: str) -> HTMLResponse:
        """View-mode title partial — what `cancel` swaps back to."""
        doc_id = _validate_doc_id(doc_id)
        settings = get_settings()
        try:
            doc = await read_document(settings.vault_path, doc_id)
        except VaultIntegrityError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return templates.TemplateResponse(request, "_document_title.html", {"document": doc})

    @app.get("/documents/{doc_id}/title/edit", response_class=HTMLResponse)
    async def document_title_edit(request: Request, doc_id: str) -> HTMLResponse:
        """Render the inline title-rename form."""
        doc_id = _validate_doc_id(doc_id)
        settings = get_settings()
        try:
            doc = await read_document(settings.vault_path, doc_id)
        except VaultIntegrityError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return templates.TemplateResponse(request, "_document_title_edit.html", {"document": doc})

    @app.post("/documents/{doc_id}/title", response_class=HTMLResponse)
    async def document_title_save(
        request: Request,
        doc_id: str,
        title: str = Form(..., max_length=_TITLE_MAX_LEN),
    ) -> HTMLResponse:
        """Apply a rename: fan the new title out to every store via
        `retitle_document` (no re-embed), then return the view partial."""
        doc_id = _validate_doc_id(doc_id)
        settings = get_settings()
        cleaned = title.strip()
        if not cleaned:
            # Empty title → no-op; re-render the current view partial.
            doc = await read_document(settings.vault_path, doc_id)
            return templates.TemplateResponse(request, "_document_title.html", {"document": doc})
        try:
            await retitle_document(doc_id, cleaned)
        except VaultIntegrityError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        refreshed = await read_document(settings.vault_path, doc_id)
        logger.info("webui.document_retitled", doc_id=doc_id, new_title=cleaned)
        return templates.TemplateResponse(request, "_document_title.html", {"document": refreshed})

    # ----- Graph -----

    @app.get("/graph/{doc_id}", response_class=HTMLResponse)
    async def graph(
        request: Request,
        doc_id: str,
        limit: int = 50,
    ) -> HTMLResponse:
        """Render the related-document neighbourhood for `doc_id` (Cytoscape.js
        client-side). Uses `related_documents` — neighbours ranked by shared-entity
        SPECIFICITY (ADR-0011), each edge labelled with the connecting entities (the
        "why") — NOT the raw unranked `neighbors()` (which surfaced generic connectors
        like an instructor's name). Returns `graph_available=False` + a fallback panel
        when ryugraph isn't installed."""
        # GraphStore is re-exported at module top (see the import at the
        # head of this file) as a test seam — `tests/integration/test_webui.py`
        # monkeypatches `memex.webui.app.GraphStore.open`. This re-export
        # is the only deliberate exception to the `webui/ → agents + vault
        # + core` import-direction rule documented in `src/memex/CLAUDE.md`,
        # and is justified by the testability win.
        doc_id = _validate_doc_id(doc_id)
        settings = get_settings()
        try:
            doc = await read_document(settings.vault_path, doc_id)
        except VaultIntegrityError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        related: list[dict[str, Any]] = []
        graph_available = True
        try:
            store = await GraphStore.open(settings.vault_path)
        except ImportError as e:
            logger.warning(
                "webui.graph_unavailable",
                doc_id=doc_id,
                reason=str(e),
            )
            graph_available = False
        else:
            try:
                related = [
                    r.model_dump() for r in await store.related_documents(doc_id, limit=limit)
                ]
            finally:
                await store.close()

        title = doc.frontmatter.title or doc_id
        nodes: list[dict[str, Any]] = [{"id": doc_id, "title": title, "kind": "center"}]
        edges: list[dict[str, Any]] = []
        # One node + one edge per related doc (already specificity-ranked + deduped by
        # related_documents). The edge label is the connecting entities — the "why" —
        # most-specific first, a few shown.
        for r in related:
            nodes.append(
                {
                    "id": r["doc_id"],
                    "title": r["title"] or r["doc_id"],
                    "kind": "neighbor",
                }
            )
            shared: list[str] = r.get("shared_entities") or []
            edges.append(
                {
                    "source": doc_id,
                    "target": r["doc_id"],
                    "label": ", ".join(shared[:3]) if shared else "shares entities",
                }
            )

        return templates.TemplateResponse(
            request,
            "graph.html",
            {
                "document": doc,
                "graph_data": {"nodes": nodes, "edges": edges},
                "neighbor_count": len(nodes) - 1,
                "graph_available": graph_available,
            },
        )

    # ----- Health -----

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        """Health check — returns the vault path so external monitors
        can confirm the right instance is responding. Always 200 OK
        as long as settings load; deeper integrity check lives in
        `memex doctor`."""
        settings = get_settings()
        return {
            "status": "ok",
            "vault_path": str(settings.vault_path),
        }

    return app


async def _scope_picker_context(
    vault_path: Path,
    *,
    checked_ids: list[str],
    flash: dict[str, str] | None,
    picker_open: bool,
) -> dict[str, Any]:
    """Build the context the scope-picker renders from: title-sorted vault docs,
    the saved scope sets, the ticked checkboxes, an optional flash, and whether
    the `<details>` opens. Shared by the index page and the three `/scope-sets`
    HTMX routes.

    A corrupt `scope_sets.json` would raise `VaultIntegrityError` from
    `list_scope_sets`; we swallow it to an empty list (+ a warning) so a damaged
    file never 500s the Ask page — `memex scope-set list` surfaces it loudly.
    """
    docs: list[dict[str, str]] = []
    async for ref in list_documents(vault_path):
        docs.append(
            {"doc_id": ref.doc_id, "title": await read_document_title(vault_path, ref.doc_id)}
        )
    docs.sort(key=lambda d: d["title"].lower())
    try:
        saved = await list_scope_sets(vault_path)
    except VaultIntegrityError as e:
        logger.warning("webui.scope_sets_unreadable", error=str(e)[:200])
        saved = []
    scope_sets: list[dict[str, object]] = [{"name": s.name, "count": len(s.doc_ids)} for s in saved]
    return {
        "documents": docs,
        "scope_sets": scope_sets,
        "checked_ids": checked_ids,
        "scope_flash": flash,
        "picker_open": picker_open,
    }


def _render_conflict(
    request: Request,
    *,
    templates: Jinja2Templates,
    doc_id: str,
    user_draft: str,
    current_body: str,
    expected_sha: str,
    current_sha: str | None,
) -> HTMLResponse:
    """Render the 409 conflict panel for `/review` stale-sha submissions.

    Computes a unified diff between the user's draft and the current
    vault body, splits it into per-line records with CSS classes
    (`diff-add`, `diff-del`, `diff-hunk`, `diff-context`) for the
    template to render with the right colour. Returns the rendered
    `_review_conflict.html` partial with status 409.
    """
    import difflib

    diff_records: list[dict[str, str]] = []
    for line in difflib.unified_diff(
        current_body.splitlines(keepends=False),
        user_draft.splitlines(keepends=False),
        fromfile="current",
        tofile="your draft",
        lineterm="",
        n=3,
    ):
        if line.startswith("@@"):
            css = "diff-hunk"
        elif line.startswith("+++") or line.startswith("---"):
            # File-header lines. Render as context so they don't drown
            # out the actual additions/removals.
            css = "diff-context"
        elif line.startswith("+"):
            css = "diff-add"
        elif line.startswith("-"):
            css = "diff-del"
        else:
            css = "diff-context"
        diff_records.append({"text": line, "css_class": css})

    return templates.TemplateResponse(
        request,
        "_review_conflict.html",
        {
            "doc_id": doc_id,
            "user_draft": user_draft,
            "current_body": current_body,
            "expected_sha": expected_sha,
            "current_sha": current_sha,
            "diff_lines": diff_records,
            "diff_line_count": len(diff_records),
        },
        status_code=409,
    )


def _anticipated_sha(doc: VaultDocument) -> str:
    """Compute the sha256 a `write_document` call would produce for
    `doc`, without writing. Used by the /review route to pre-update
    the manifest BEFORE the atomic file rename so a kill in the
    middle of the operation leaves the watcher able to reconcile.
    """
    import frontmatter

    fm_dict: dict[str, Any] = {
        **doc.frontmatter.model_dump(exclude={"custom"}, exclude_none=True),
        **doc.frontmatter.custom,
    }
    post = frontmatter.Post(doc.body, **fm_dict)
    serialized = frontmatter.dumps(post)
    return hash_bytes(serialized.encode("utf-8"))
