# pyright: reportUnusedFunction=false
# FastAPI route handlers are decorated with `@app.get`/`@app.post`
# which registers them in the ASGI app's route table. Pyright can't
# introspect the decorator's side effect and flags every route
# handler as "not accessed." Every route in this module is
# reached via the FastAPI dispatcher (the count is `git grep -c "@app\."`).

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
- `GET  /entity?name=`                  — entity-centric discovery (ADR-0011):
                                          graph profile + co-occurring + passages
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
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader

from memex.agents.answering import FinalResponse, answer_query
from memex.agents.bridge import BridgeAnswer, reason_then_ground
from memex.agents.chat import answer_turn
from memex.agents.document_summarizer import SummaryDetail, summarize_document
from memex.agents.expert import ExpertAnswer, expert_answer
from memex.core.config import get_settings
from memex.core.conversation_store import ConversationStore
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
from memex.core.types import Chunk, CompanionAlignment
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

# webui → retrieve boundary edge (documented, like webui → parse / daemon): the
# entity view is entity-centric DISCOVERY (ADR-0011) — graph identity + co-occurring
# neighbourhood + FTS passages. Read-only + HARD-gate-neutral; `entity_overview` is
# the orchestrator that composes GraphStore + FTSStore (fail-open when ryugraph absent).
from memex.retrieve import (
    entity_overview,
    related_documents_for_answer,
    related_documents_for_seeds,
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
    BRIDGE_PHASES,
    EXPERT_PHASES,
    PHASES,
    SUMMARY_PHASES,
    ProgressRegistry,
    bridge_phase_index,
    expert_phase_index,
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


def _format_time_anchor(time_range: tuple[float, float] | None) -> str:
    """The COMPACT chip form of an audio chunk's START time — `m:ss` (or `h:mm:ss`
    past an hour) — the transcript analogue of the `· p. N` page chip (ADR-0017),
    so it follows the chip convention (no zero-padded leading unit) rather than the
    body-heading form `[mm:ss]`/`[hh:mm:ss]` that `parse._format_timestamp` writes.
    `""` for a non-audio chunk (`time_range is None`). Kept local to the webui (no
    parse import — that boundary is closed)."""
    if time_range is None:
        return ""
    total = max(0, int(time_range[0]))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _source_view(response: FinalResponse) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """View-model for rendering answer/summary sources by HUMAN TITLE instead of
    the raw `docid#hash` / `[[doc#section]]` syntax. Returns:

    - `chunk_refs`: `chunk_id → {title, section, href, page, time}` for the per-claim
      source chips (a claim whose chunk isn't here — e.g. a synthetic table/SQL
      chunk — falls back to the raw id in the template). `page` is `""` when
      unknown (legacy chunks indexed before the chunker's page-attribution lever
      shipped, OR a doc parsed without per-page char counts in its manifest);
      otherwise the 1-based source page number, threaded into the href as
      `?page=N#section-slug` so the doc-page template can scroll the PDF
      preview pane to that page on landing. `time` is the audio time anchor
      (`mm:ss`, ADR-0017) for a transcript chunk, `""` otherwise — page/time are
      mutually exclusive (a doc is either paged or audio).
    - `doc_titles`: `doc_id → title` for the `render_wikilink` Sources labels.

    Built from the cited chunks' OWN `document_title` + `heading_path` + `page` +
    `time_range` — no extra I/O (the same data the refusal panel already shows)."""
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
            "time": _format_time_anchor(c.time_range),
        }
        doc_titles[c.document_id] = title
    return chunk_refs, doc_titles


_ENTITY_PASSAGE_PREVIEW_CHARS = 480


def _passage_refs(passages: list[Chunk]) -> list[dict[str, str]]:
    """View-model for the entity view's representative passages: each `Chunk` →
    `{title, section, href, text}` rendered by HUMAN title › section (mirrors
    `_source_view`), with the body truncated to a preview (the doc link reads on).
    The href carries `?page=N#slug` so the click lands on the source page+section."""
    out: list[dict[str, str]] = []
    for c in passages:
        section = clean_heading_text(c.heading_path[-1]) if c.heading_path else ""
        href = f"/documents/{c.document_id}"
        if c.page is not None:
            href = f"{href}?page={c.page}"
        if section:
            href = f"{href}#{slugify_heading(section)}"
        text = c.text.strip()
        if len(text) > _ENTITY_PASSAGE_PREVIEW_CHARS:
            text = text[:_ENTITY_PASSAGE_PREVIEW_CHARS].rstrip() + "…"
        out.append(
            {
                "title": c.document_title or c.document_id,
                "section": section,
                "href": href,
                "text": text,
            }
        )
    return out


async def _safe_doc_title(vault_path: Path, doc_id: str) -> str:
    """Read a doc's title, FAIL-OPEN to the doc_id. A single doc with corrupt frontmatter
    must not 500 a whole listing page (the Ask landing / Documents list / scope-picker)."""
    try:
        return await read_document_title(vault_path, doc_id)
    except VaultIntegrityError as e:
        logger.warning("webui.title_unreadable", doc_id=doc_id, reason=str(e))
        return doc_id


async def _related_for_docs(
    vault_path: Path,
    seed_ids: list[str],
    *,
    seed_limit: int = 5,
    per_seed: int = 8,
    out_limit: int = 6,
) -> list[dict[str, Any]]:
    """Aggregate `related_documents` over a SET of seed docs → the merged, deduped,
    seed-EXCLUDED, re-ranked, capped list (as dicts). The shared core behind the /ask
    "Related documents" panel and the scope-picker "Suggested additions" — both are just
    "graph neighbours of a set of docs". Read-only + HARD-gate-neutral; ImportError fail-open
    → `[]`. Expands the first `seed_limit` seeds (bounds graph calls) but EXCLUDES the full
    seed set from the output. Reuses the SHIPPED noise-filtered `related_documents` ranking
    (specificity + shared-docs floor), so callers inherit it."""
    ranked = await related_documents_for_seeds(
        vault_path, seed_ids, seed_limit=seed_limit, per_seed=per_seed, out_limit=out_limit
    )
    return [r.model_dump() for r in ranked]


async def _related_for_answer(vault_path: Path, response: FinalResponse) -> list[dict[str, Any]]:
    """The /ask "Related documents" panel (ADR-0011): graph neighbours of the docs THIS answer
    cited, as template dicts. Delegates to the shared `retrieve.related_documents_for_answer`
    (the same core MCP/CLI use) — HARD-gate-neutral, derived from the already-returned
    `FinalResponse`, never touches the agent/answer/refusal path."""
    ranked = await related_documents_for_answer(
        vault_path,
        [c.document_id for c in response.used_chunks],
        answered=response.answered,
    )
    return [r.model_dump() for r in ranked]


async def _answer_context(
    vault_path: Path, response: FinalResponse, scope_source: str
) -> dict[str, object]:
    """Build the `_answer.html` context from a FinalResponse — scope-doc titles
    (#256), the source-by-title view-model, the scope-source label, and the
    "Related documents" discovery panel (ADR-0011). Shared by the long-poll status
    route so the rendered answer is identical to the old synchronous path."""
    scope_docs = [
        {"doc_id": d, "title": await _safe_doc_title(vault_path, d)}
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
        "related": await _related_for_answer(vault_path, response),
        "companion": await _companion_labels(vault_path, response),
    }


async def _companion_labels(vault_path: Path, response: FinalResponse) -> dict[str, str]:
    """`chunk_id → a companion nav label` for the cited chunks (ADR-0018 B3): a cited TRANSCRIPT chunk
    gets `slide N` (the slide it explains); a cited DECK chunk gets `lecture mm:ss` (the spoken
    commentary on it). Read FAIL-OPEN from the alignment sidecar — no pair / no alignment → `{}`, so
    the answer renders exactly as before. HARD-gate-neutral (a presentation lookup over the cited
    chunks + a read-only sidecar; never touches the answer/grounding)."""
    from memex.core.companion_store import alignments_for_doc

    out: dict[str, str] = {}
    cache: dict[str, list[CompanionAlignment]] = {}
    for c in response.used_chunks:
        if c.document_id not in cache:
            cache[c.document_id] = await alignments_for_doc(vault_path, c.document_id)
        for a in cache[c.document_id]:
            if c.document_id == a.transcript_doc:
                block = next(
                    (b for b in a.blocks if b.transcript_chunk_id == c.chunk_id and b.deck_page),
                    None,
                )
                if block is not None:
                    out[c.chunk_id] = f"slide {block.deck_page}"
                    break
            elif c.document_id == a.deck_doc and c.page is not None:
                block = next(
                    (b for b in a.blocks if b.deck_page == c.page and b.time_range is not None),
                    None,
                )
                if block is not None:
                    out[c.chunk_id] = f"lecture {_format_time_anchor(block.time_range)}"
                    break
    return out


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
    # The ungrounded expert surface (Surface B, ADR-0013) is off by default; only
    # surface its nav link when it's enabled, so a disabled deployment shows no dead link.
    env.globals["expert_enabled"] = get_settings().agents.expert_mode_enabled  # type: ignore[reportUnknownMemberType]  # reason: jinja2 leaves Environment.globals unannotated
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

    @app.post("/scope-sets/suggest", response_class=HTMLResponse)
    async def scope_set_suggest(
        request: Request,
        scope_doc_ids: list[str] = Form([]),  # noqa: B008  # FastAPI Form default sentinel
    ) -> HTMLResponse:
        """Suggest documents the entity graph relates to the currently-ticked selection
        (ADR-0011 "docs related to your current selection"). Re-renders the picker — the
        ticks stay checked, and a "Suggested additions" section appears. The suggestions are
        computed inside `_scope_picker_context` from `checked_ids`; we just read the count
        for the flash. Read-only + HARD-gate-neutral (the agent never sees this)."""
        settings = get_settings()
        ctx = await _scope_picker_context(
            settings.vault_path, checked_ids=scope_doc_ids, flash=None, picker_open=True
        )
        n = len(ctx["suggested"])  # type: ignore[arg-type]  # reason: list[dict] from the context
        if not scope_doc_ids:
            ctx["scope_flash"] = {"kind": "error", "text": "Tick one or more documents, then Suggest related."}
        elif n:
            ctx["scope_flash"] = {
                "kind": "ok",
                "text": f"{n} related document{'' if n == 1 else 's'} — tick any to add to your scope.",
            }
        else:
            ctx["scope_flash"] = {"kind": "ok", "text": "No related documents found for your selection."}
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
        progress.new(
            cid, scope_doc_ids=scope_doc_ids, scope_source=scope_source, question=question
        )
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
        if entry.error is not None or not isinstance(entry.response, FinalResponse):
            return templates.TemplateResponse(
                request,
                "_answer.html",
                {"response": None, "error": entry.error or "Answering produced no result."},
            )
        settings = get_settings()
        ctx = await _answer_context(settings.vault_path, entry.response, entry.scope_source)
        # Carry the original question + scope so a REFUSAL panel can offer the consented A→B
        # escalation (§11) — a user-chosen "reason over this instead" into the bridge over the
        # SAME scope. Refusal-only + gated on expert_enabled in the template; answered path ignores.
        ctx["question"] = entry.question
        ctx["escalate_scope_ids"] = entry.scope_doc_ids
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
            title = await _safe_doc_title(settings.vault_path, ref.doc_id)
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
        # discovery. Optional: an uninstalled ryugraph (ImportError) fails OPEN to no
        # section, mirroring the /graph route. (A runtime graph error still surfaces —
        # only the optional-dependency case is caught, per the narrow-except rule.)
        related: list[dict[str, Any]] = []
        citations: dict[str, list[dict[str, Any]]] = {"cites": [], "cited_by": []}
        try:
            rstore = await GraphStore.open(settings.vault_path)
        except ImportError as e:
            logger.warning("webui.related_unavailable", doc_id=doc_id, reason=str(e))
        else:
            try:
                related = [r.model_dump() for r in await rstore.related_documents(doc_id, limit=8)]
                cites = await rstore.citations(doc_id)
                citations = {
                    "cites": [c.model_dump() for c in cites.cites],
                    "cited_by": [c.model_dump() for c in cites.cited_by],
                }
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
                "citations": citations,
            },
        )

    @app.get("/entity", response_class=HTMLResponse)
    async def entity(request: Request, name: str = "") -> HTMLResponse:
        """Entity-centric DISCOVERY (ADR-0011): "everything about entity X". An empty
        `name` renders just the lookup form; a name resolves to its graph profile
        (canonical identity + the authoritative mentioning-doc set + the co-occurring
        concept neighbourhood — each co-entity links back here to traverse) plus
        representative FTS passages. An unknown name → the honest whole-corpus FTS
        fallback (`resolved=False`). Read-only + fail-open (the orchestrator never
        raises on a missing graph); the co-occurring + mentions come from the graph,
        the passages from full-text search — the template says so."""
        name = name.strip()
        if not name:
            return templates.TemplateResponse(
                request, "entity.html", {"name": "", "overview": None, "passages": []}
            )
        overview = await entity_overview(name)
        return templates.TemplateResponse(
            request,
            "entity.html",
            {
                "name": name,
                "overview": overview,
                "passages": _passage_refs(overview.passages),
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
        if entry.error is not None or not isinstance(entry.response, FinalResponse):
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

    # ----- Grounded multi-turn chat (Surface A, docs/specs/grounded-agentic-chat.md) -----

    def _chat_assistant_ctx(response: FinalResponse) -> dict[str, object]:
        """Minimal context to re-render a chat assistant bubble from a stored response:
        the source view-model (cheap, no I/O); the related panel is skipped on
        rehydration (the LIVE turn adds it via `_answer_context`)."""
        chunk_refs, doc_titles = _source_view(response)
        return {
            "response": response,
            "error": None,
            "chunk_refs": chunk_refs,
            "doc_titles": doc_titles,
            "related": [],
        }

    async def _run_chat_turn(
        cid: str, conversation_id: str, message: str, scope_doc_ids: list[str] | None
    ) -> None:
        """Background runner: answer one grounded chat turn (persisted by `answer_turn`),
        streaming the answer-graph phases into the registry. Top of a fire-and-forget
        task — never crash silently."""
        from memex.core.errors import MemexError

        try:
            result = await answer_turn(
                conversation_id,
                message,
                scope_doc_ids=scope_doc_ids,
                correlation_id=cid,
                on_node=lambda n: progress.set_phase(cid, phase_for(n)),
            )
            progress.finish(cid, response=result.response)
        except MemexError as e:
            logger.warning("chat.failed", error_type=type(e).__name__, error=str(e)[:200])
            progress.finish(cid, error=f"Couldn't answer: {type(e).__name__}. {str(e)[:160]}")
        except Exception as e:
            logger.error("chat.crashed", error_type=type(e).__name__, error=str(e)[:200])
            progress.finish(cid, error="An unexpected error occurred while answering.")

    @app.get("/chat", response_class=HTMLResponse)
    async def chat_home() -> RedirectResponse:
        """Open a conversation: reuse the most recent EMPTY one (so repeated nav clicks
        don't litter the store) or create a fresh one, then redirect to it."""
        store = await ConversationStore.open(get_settings().vault_path)
        try:
            recent = await store.list_conversations(limit=20)
            empty = next((c for c in recent if c.turn_count == 0), None)
            convo = empty or await store.create_conversation()
        finally:
            await store.close()
        return RedirectResponse(f"/chat/{convo.conversation_id}", status_code=303)

    @app.get("/chat/{conversation_id}", response_class=HTMLResponse)
    async def chat_view(request: Request, conversation_id: str) -> HTMLResponse:
        """Render a conversation: each turn rehydrated (user message + grounded assistant
        bubble), the composer, and a rail of recent conversations to resume."""
        settings = get_settings()
        store = await ConversationStore.open(settings.vault_path)
        try:
            convo = await store.load(conversation_id)
            if convo is None:
                raise HTTPException(status_code=404, detail="conversation not found")
            recent = [c for c in await store.list_conversations(limit=20) if c.turn_count > 0]
        finally:
            await store.close()

        turns: list[dict[str, object]] = []
        for t in convo.turns:
            assistant: dict[str, object] | None = None
            if t.response_json:
                try:
                    resp = FinalResponse.model_validate_json(t.response_json)
                except ValueError:
                    assistant = None
                else:
                    # A degenerate refusal (answered=False with no reason — a model glitch
                    # or a forward-incompatible response_json) would render a blank bubble;
                    # drop it to a user-only turn rather than show an empty assistant.
                    if not resp.answered and not (resp.refusal_reason or "").strip():
                        logger.warning("chat.resume_empty_refusal", turn_id=t.turn_id)
                        assistant = None
                    else:
                        assistant = _chat_assistant_ctx(resp)
            turns.append({"user_text": t.user_text, "assistant": assistant})

        # The scope picker shows only on a fresh conversation (turn 0); later turns
        # inherit the persisted per-conversation scope pin.
        picker_ctx: dict[str, object] = {}
        if convo.turn_count == 0:
            picker_ctx = await _scope_picker_context(
                settings.vault_path,
                checked_ids=convo.scope_doc_ids,
                flash=None,
                picker_open=False,
            )

        return templates.TemplateResponse(
            request,
            "chat.html",
            {
                "conversation": convo,
                "turns": turns,
                "recent": recent,
                "show_picker": convo.turn_count == 0,
                **picker_ctx,
            },
        )

    @app.post("/chat/{conversation_id}/turn", response_class=HTMLResponse)
    async def chat_turn(
        request: Request,
        conversation_id: str,
        message: str = Form(..., max_length=_QUESTION_MAX_BYTES),
        scope_doc_ids: list[str] = Form([]),  # noqa: B008  # FastAPI Form default sentinel
    ) -> HTMLResponse:
        """Start a grounded chat turn in the background and return the user bubble + the
        progress fragment (appended to #conversation-log; the progress self-replaces with
        the assistant bubble on completion). A turn-0 scope selection is persisted as the
        conversation's scope pin so later turns inherit it."""
        message = message.strip()
        if not message:
            return HTMLResponse("", status_code=400)
        scope = list(dict.fromkeys(s for s in scope_doc_ids if s.strip()))
        if scope:
            store = await ConversationStore.open(get_settings().vault_path)
            try:
                await store.set_scope(conversation_id, scope)
            finally:
                await store.close()
        cid = str(ulid.ULID())
        progress.new(cid, scope_doc_ids=scope, scope_source="selected" if scope else "named")
        task = asyncio.create_task(_run_chat_turn(cid, conversation_id, message, scope or None))
        progress.attach_task(cid, task)
        return templates.TemplateResponse(
            request,
            "_chat_turn.html",
            {
                "user_text": message,
                "poll_url": f"/chat/{conversation_id}/status?cid={cid}&v=0",
                "phases": PHASES,
                "active_index": 0,
                "elapsed": 0,
                "detail": "",
            },
        )

    @app.get("/chat/{conversation_id}/status", response_class=HTMLResponse)
    async def chat_status(
        request: Request, conversation_id: str, cid: str = "", v: int = 0
    ) -> HTMLResponse:
        """Long-poll the in-flight chat turn (mirrors `summarize_status`): the progress
        fragment until the answer graph finishes, then the grounded assistant bubble
        (which replaces the progress in place). The turn is already persisted by
        `answer_turn`, so a later resume re-renders it identically."""
        entry = await progress.wait_for_change(cid, v)
        if entry is None:
            return templates.TemplateResponse(request, "_progress_expired.html", {})
        if not entry.done:
            return templates.TemplateResponse(
                request,
                "_progress.html",
                {
                    "poll_url": f"/chat/{conversation_id}/status?cid={cid}&v={entry.version}",
                    "phases": PHASES,
                    "active_index": entry.active_index(),
                    "elapsed": entry.phase_elapsed_s(),
                    "detail": "",
                },
            )
        progress.evict(cid)
        if entry.error is not None or not isinstance(entry.response, FinalResponse):
            return templates.TemplateResponse(
                request,
                "_chat_assistant.html",
                {"response": None, "error": entry.error or "Answering produced no result."},
            )
        ctx = await _answer_context(get_settings().vault_path, entry.response, "named")
        return templates.TemplateResponse(request, "_chat_assistant.html", ctx)

    @app.post("/chat/{conversation_id}/delete", response_class=HTMLResponse)
    async def chat_delete(conversation_id: str) -> RedirectResponse:
        """Delete a conversation and return to the chat home."""
        store = await ConversationStore.open(get_settings().vault_path)
        try:
            await store.delete_conversation(conversation_id)
        finally:
            await store.close()
        return RedirectResponse("/chat", status_code=303)

    # ----- Ungrounded reasoning EXPERT mode (Surface B, ADR-0013) -----

    async def _run_expert(cid: str, question: str, scope_doc_ids: list[str]) -> None:
        """Background runner: drive the ungrounded expert reasoning pass, streaming phase
        updates into the registry. Top of a fire-and-forget task — never crash silently."""
        from memex.core.errors import MemexError

        try:
            answer = await expert_answer(
                question,
                scope_doc_ids=scope_doc_ids or None,
                correlation_id=cid,
                on_phase=lambda p: progress.set_phase(cid, p),
            )
            progress.finish(cid, response=answer)
        except MemexError as e:
            logger.warning("expert.failed", error_type=type(e).__name__, error=str(e)[:200])
            progress.finish(cid, error=f"Couldn't reason: {type(e).__name__}. {str(e)[:160]}")
        except Exception as e:
            logger.error("expert.crashed", error_type=type(e).__name__, error=str(e)[:200])
            progress.finish(cid, error="An unexpected error occurred while reasoning.")

    @app.get("/expert", response_class=HTMLResponse)
    async def expert_home(request: Request) -> HTMLResponse:
        """The expert surface: a question form + the ungrounded-mode banner. When the
        feature is disabled, the template explains how to enable it instead of a form."""
        return templates.TemplateResponse(
            request,
            "expert.html",
            {"enabled": get_settings().agents.expert_mode_enabled, "question": ""},
        )

    @app.post("/expert", response_class=HTMLResponse)
    async def expert_run(request: Request, question: str = Form("")) -> HTMLResponse:
        """Start the ungrounded reasoning pass in a background task and IMMEDIATELY return
        the `_progress.html` fragment, which long-polls `/expert/status` until the answer
        swaps in. Refuses (a flash) when the feature is disabled or the question is empty."""
        if not get_settings().agents.expert_mode_enabled:
            return templates.TemplateResponse(
                request,
                "_expert.html",
                {
                    "answer": None,
                    "error": "Expert mode is disabled (set MEMEX_AGENTS__EXPERT_MODE_ENABLED=true).",
                },
            )
        q = question.strip()
        if not q:
            return templates.TemplateResponse(
                request,
                "_expert.html",
                {"answer": None, "error": "Ask an analytical question first."},
            )
        cid = str(ulid.ULID())
        progress.new(cid, scope_doc_ids=[], scope_source="named")
        task = asyncio.create_task(_run_expert(cid, q, []))
        progress.attach_task(cid, task)
        return templates.TemplateResponse(
            request,
            "_progress.html",
            {
                "poll_url": f"/expert/status?cid={cid}&v=0",
                "phases": EXPERT_PHASES,
                "active_index": 0,
                "elapsed": 0,
                "detail": "",
            },
        )

    @app.get("/expert/status", response_class=HTMLResponse)
    async def expert_status(request: Request, cid: str = "", v: int = 0) -> HTMLResponse:
        """Long-poll the in-flight reasoning pass: render `_progress.html` until the run
        finishes (or a keepalive), then `_expert.html` (the reasoned answer + provenance).
        Always HTTP 200; done/expired carry no poll trigger so the chain stops itself."""
        entry = await progress.wait_for_change(cid, v)
        if entry is None:
            return templates.TemplateResponse(request, "_progress_expired.html", {})
        if not entry.done:
            return templates.TemplateResponse(
                request,
                "_progress.html",
                {
                    "poll_url": f"/expert/status?cid={cid}&v={entry.version}",
                    "phases": EXPERT_PHASES,
                    "active_index": expert_phase_index(entry.phase),
                    "elapsed": entry.phase_elapsed_s(),
                    "detail": "",
                },
            )
        progress.evict(cid)
        answer = entry.response
        if entry.error is not None or not isinstance(answer, ExpertAnswer):
            return templates.TemplateResponse(
                request,
                "_expert.html",
                {"answer": None, "error": entry.error or "Reasoning produced no result."},
            )
        return templates.TemplateResponse(request, "_expert.html", {"answer": answer, "error": None})

    async def _run_bridge(
        cid: str, question: str, scope_doc_ids: list[str], present_as_answer: bool = False
    ) -> None:
        """Background runner: reason-then-ground, streaming phase updates into the registry.
        Top of a fire-and-forget task — never crash silently.

        `present_as_answer` (ADR-0016): the consented A→B escalation sets it so the grounded subset
        is presented AS an answer when responsive; the standalone composer leaves it False."""
        from memex.core.errors import MemexError

        try:
            answer = await reason_then_ground(
                question,
                scope_doc_ids=scope_doc_ids or None,
                present_as_answer=present_as_answer,
                correlation_id=cid,
                on_phase=lambda p: progress.set_phase(cid, p),
            )
            progress.finish(cid, response=answer)
        except MemexError as e:
            logger.warning("bridge.failed", error_type=type(e).__name__, error=str(e)[:200])
            progress.finish(cid, error=f"Couldn't reason: {type(e).__name__}. {str(e)[:160]}")
        except Exception as e:
            logger.error("bridge.crashed", error_type=type(e).__name__, error=str(e)[:200])
            progress.finish(cid, error="An unexpected error occurred while reasoning.")

    @app.get("/bridge", response_class=HTMLResponse)
    async def bridge_home(request: Request) -> HTMLResponse:
        """The reason-then-ground surface: a question form + the dual-contract banner + the
        document scope-picker (same as /ask — tick docs to scope the analysis to them). When the
        feature is disabled (the same flag as expert mode), the template explains how to enable it."""
        ctx = await _scope_picker_context(
            get_settings().vault_path, checked_ids=[], flash=None, picker_open=False
        )
        ctx.update({"enabled": get_settings().agents.expert_mode_enabled, "question": ""})
        return templates.TemplateResponse(request, "bridge.html", ctx)

    @app.post("/bridge", response_class=HTMLResponse)
    async def bridge_run(
        request: Request,
        question: str = Form(""),
        scope_doc_ids: list[str] = Form([]),  # noqa: B008  # FastAPI Form default sentinel
        present_as_answer: bool = Form(False),
    ) -> HTMLResponse:
        """Start the reason-then-ground pass in a background task and IMMEDIATELY return the
        `_progress.html` fragment, which long-polls `/bridge/status` until the answer swaps in.

        `scope_doc_ids` is carried by the A→B escalation form (§11) so a SCOPED /ask refusal
        escalates into a reason-then-ground pass over the SAME scope (the user's explicit
        constraint is preserved, not silently widened to the whole vault). The standalone
        /bridge composer sends none → whole-vault, unchanged.

        `present_as_answer` (ADR-0016) is the ONLY discriminator between the two POST callers:
        the consented escalation form sets it (`_answer.html`) so the grounded subset is presented
        AS an answer when responsive; the standalone composer omits it → the labelled-analysis
        surface, unchanged. It rides the result (`BridgeAnswer.present_as_answer`), so the status
        handler needs no extra plumbing."""
        if not get_settings().agents.expert_mode_enabled:
            return templates.TemplateResponse(
                request,
                "_bridge.html",
                {
                    "answer": None,
                    "sources": {},
                    "error": "Reason-then-ground is disabled (set MEMEX_AGENTS__EXPERT_MODE_ENABLED=true).",
                },
            )
        q = question.strip()
        if not q:
            return templates.TemplateResponse(
                request,
                "_bridge.html",
                {"answer": None, "sources": {}, "error": "Ask an analytical question first."},
            )
        scope = [d.strip() for d in scope_doc_ids if d.strip()]
        cid = str(ulid.ULID())
        progress.new(cid, scope_doc_ids=scope, scope_source="selected" if scope else "named")
        task = asyncio.create_task(_run_bridge(cid, q, scope, present_as_answer))
        progress.attach_task(cid, task)
        return templates.TemplateResponse(
            request,
            "_progress.html",
            {
                "poll_url": f"/bridge/status?cid={cid}&v=0",
                "phases": BRIDGE_PHASES,
                "active_index": 0,
                "elapsed": 0,
                "detail": "",
            },
        )

    @app.get("/bridge/status", response_class=HTMLResponse)
    async def bridge_status(request: Request, cid: str = "", v: int = 0) -> HTMLResponse:
        """Long-poll the in-flight reason-then-ground pass: `_progress.html` until the run
        finishes, then `_bridge.html` (the ungrounded analysis + the grounded-claims subset).
        Always HTTP 200; done/expired carry no poll trigger so the chain stops itself."""
        entry = await progress.wait_for_change(cid, v)
        if entry is None:
            return templates.TemplateResponse(request, "_progress_expired.html", {})
        if not entry.done:
            return templates.TemplateResponse(
                request,
                "_progress.html",
                {
                    "poll_url": f"/bridge/status?cid={cid}&v={entry.version}",
                    "phases": BRIDGE_PHASES,
                    "active_index": bridge_phase_index(entry.phase),
                    "elapsed": entry.phase_elapsed_s(),
                    "detail": "",
                },
            )
        progress.evict(cid)
        answer = entry.response
        if entry.error is not None or not isinstance(answer, BridgeAnswer):
            return templates.TemplateResponse(
                request,
                "_bridge.html",
                {
                    "answer": None,
                    "sources": {},
                    "error": entry.error or "Reasoning produced no result.",
                },
            )
        # Build the per-claim source view-model from the grounded chunks (no extra I/O — the
        # same data the answer panel shows), mirroring `_source_view`'s {chunk_id → {…}} shape.
        sources = {
            c.chunk_id: {
                "title": c.document_title or c.document_id,
                "section": (c.heading_path[-1] if c.heading_path else None),
                "href": f"/documents/{c.document_id}" + (f"?page={c.page}" if c.page else ""),
                "page": c.page,
            }
            for c in answer.grounded_sources
        }
        # The DOCUMENTS the analysis was reasoned over (the reranked evidence) — deduped to one
        # row per document, navigable. This closes the user-journey gap: when nothing grounded
        # (or to show the fuller retrieval scope), the user can still open the vault documents the
        # analysis drew on and see what they actually say. It is NOT a grounding cite — the
        # template labels it as "reasoned over", never "verified" (no I/O — same data the model read).
        evidence_docs: list[dict[str, object]] = []
        seen_doc_ids: set[str] = set()
        for e in answer.evidence:
            if e.document_id in seen_doc_ids:
                continue
            seen_doc_ids.add(e.document_id)
            evidence_docs.append(
                {
                    "doc_id": e.document_id,
                    "title": e.title or e.document_id,
                    "section": e.section,
                    "page": e.page,
                    "href": f"/documents/{e.document_id}" + (f"?page={e.page}" if e.page else ""),
                }
            )
        # Resolve the scope this analysis ran over → titles, for the "Scoped to …" note (parity
        # with /ask). Empty (whole-vault) → [] → the note is omitted. Mirrors `_answer_context`.
        scope_docs = [
            {"doc_id": d, "title": await _safe_doc_title(get_settings().vault_path, d)}
            for d in answer.scope_doc_ids
        ]
        return templates.TemplateResponse(
            request,
            "_bridge.html",
            {
                "answer": answer,
                "sources": sources,
                "evidence_docs": evidence_docs,
                "error": None,
                "scope_docs": scope_docs,
                "scope_source": entry.scope_source,
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
        group: str = "concept",
    ) -> HTMLResponse:
        """Render `doc_id`'s related-document neighbourhood as a server-rendered, ranked
        "Bridges" view (the redesign that retired the Cytoscape hairball — a 1-hop star has
        no topology to draw; the signal is a specificity RANKING + the entities that explain
        WHY). Two lenses over the SAME graph data, toggled by `?group=`:

          - `concept` (default) — `related_bridges`: related docs grouped UNDER the bridging
            ENTITY that connects them, ranked by Σ IDF×kind_weight (a rare concept shared by
            many docs wins). Answers "which concepts are this doc's connective tissue".
          - `document` — `related_documents`: the neighbours as a flat list ranked by
            shared-entity specificity, each with a strength bar + the connecting entities.

        Both rank by the SAME ADR-0011 specificity model (NOT the unranked `neighbors()`).
        Returns `graph_available=False` + a fallback panel when ryugraph isn't installed."""
        # GraphStore is re-exported at module top (see the import at the
        # head of this file) as a test seam — `tests/integration/test_webui.py`
        # monkeypatches `memex.webui.app.GraphStore.open`. This re-export
        # is the only deliberate exception to the `webui/ → agents + vault
        # + core` import-direction rule documented in `src/memex/CLAUDE.md`,
        # and is justified by the testability win.
        doc_id = _validate_doc_id(doc_id)
        group = group if group in ("concept", "document") else "concept"
        settings = get_settings()
        try:
            doc = await read_document(settings.vault_path, doc_id)
        except VaultIntegrityError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        bridges: list[dict[str, Any]] = []
        related: list[dict[str, Any]] = []
        graph_available = True
        try:
            store = await GraphStore.open(settings.vault_path)
        except ImportError as e:
            logger.warning("webui.graph_unavailable", doc_id=doc_id, reason=str(e))
            graph_available = False
        else:
            try:
                # Both lenses share one graph open — the document lens drives the header
                # count + the alternate view; the bridge lens is the default render.
                related = [
                    r.model_dump() for r in await store.related_documents(doc_id, limit=50)
                ]
                bridges = [b.model_dump() for b in await store.related_bridges(doc_id)]
            finally:
                await store.close()

        # Proportional strength bars (ordinal sugar — the count / rank are the honest signal,
        # so the % is never printed; WCAG 1.4.1: bar length is never the SOLE carrier).
        if related:
            max_score = max((r["score"] for r in related), default=0.0) or 1.0
            for r in related:
                r["bar_pct"] = round(100 * float(r["score"]) / max_score)
        if bridges:
            max_strength = max((b["strength"] for b in bridges), default=0.0) or 1.0
            for b in bridges:
                b["bar_pct"] = round(100 * float(b["strength"]) / max_strength)
        # Split the ranked bridges: multi-doc bridges are the headline (real connective
        # tissue); single-doc bridges fold into a quiet tail disclosure. If there are NO
        # multi-doc bridges (a sparse neighbourhood), promote the singles so the view isn't
        # empty behind a disclosure.
        primary = [b for b in bridges if b["doc_count"] >= 2]
        tail = [b for b in bridges if b["doc_count"] < 2]
        if not primary:
            primary, tail = tail, []

        return templates.TemplateResponse(
            request,
            "graph.html",
            {
                "document": doc,
                "group": group,
                "related": related,
                "bridges_primary": primary,
                "bridges_tail": tail,
                "bridge_count": len(bridges),
                "neighbor_count": len(related),
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
    the saved scope sets, the ticked checkboxes, an optional flash, whether the
    `<details>` opens, and `suggested` (docs the entity graph relates to the current
    selection — ADR-0011 "scope-set suggestions", below). Shared by the index page and
    the four `/scope-sets` HTMX routes.

    `suggested` is computed from `checked_ids` (empty selection ⇒ no graph query): the
    apply/save/suggest routes re-render with a non-empty selection, so suggestions appear
    there automatically; the index page + delete pass `[]` and compute nothing. Read-only +
    fail-open (a missing graph → []) — HARD-gate-neutral scoping UX, never the answer path.

    A corrupt `scope_sets.json` would raise `VaultIntegrityError` from
    `list_scope_sets`; we swallow it to an empty list (+ a warning) so a damaged
    file never 500s the Ask page — `memex scope-set list` surfaces it loudly.
    """
    docs: list[dict[str, str]] = []
    async for ref in list_documents(vault_path):
        docs.append(
            {"doc_id": ref.doc_id, "title": await _safe_doc_title(vault_path, ref.doc_id)}
        )
    docs.sort(key=lambda d: d["title"].lower())
    try:
        saved = await list_scope_sets(vault_path)
    except VaultIntegrityError as e:
        logger.warning("webui.scope_sets_unreadable", error=str(e)[:200])
        saved = []
    scope_sets: list[dict[str, object]] = [{"name": s.name, "count": len(s.doc_ids)} for s in saved]
    suggested = await _related_for_docs(vault_path, checked_ids) if checked_ids else []
    return {
        "documents": docs,
        "scope_sets": scope_sets,
        "checked_ids": checked_ids,
        "scope_flash": flash,
        "picker_open": picker_open,
        "suggested": suggested,
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
