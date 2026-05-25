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

import mimetypes
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader

from memex.agents.answering import answer_query
from memex.core.config import get_settings
from memex.core.errors import StaleDocumentError, VaultIntegrityError
from memex.core.manifest import update_manifest
from memex.index.graph_store import GraphStore
from memex.index.pipeline import retitle_document
from memex.vault.store import (
    VaultDocument,
    hash_bytes,
    list_documents,
    make_ref,
    read_document,
    read_document_title,
    write_document,
)
from memex.webui.rendering import extract_toc, render_body_html, render_wikilink

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


def _find_source(vault_path: Path, doc_id: str) -> Path | None:
    """Locate the original `source.<ext>` for a doc, if one was copied
    in by the ingest stage. Markdown-passthrough docs have no source
    file — they ARE the source."""
    asset_dir = vault_path / "documents" / doc_id
    if not asset_dir.is_dir():
        return None
    candidates = sorted(asset_dir.glob("source.*"))
    return candidates[0] if candidates else None


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
        """Landing page — renders the ask form."""
        return templates.TemplateResponse(request, "index.html", {})

    @app.post("/ask", response_class=HTMLResponse)
    async def ask(
        request: Request,
        question: str = Form(..., max_length=_QUESTION_MAX_BYTES),
    ) -> HTMLResponse:
        """Run the answering agent against `question` and render the
        HTMX `_answer.html` partial. Typed MemexError subclasses render
        as a refusal banner with status 503 rather than a bare 500."""
        from memex.core.errors import MemexError

        question = question.strip()
        if not question:
            return templates.TemplateResponse(
                request,
                "_answer.html",
                {"response": None, "error": "Question is empty."},
                status_code=400,
            )
        try:
            response = await answer_query(question)
        except MemexError as e:
            # The agent surfaces typed MemexError subclasses (ModelCallError
            # from a schema-violating LLM output, InsufficientVRAMError on
            # OOM, CircuitBreakerOpen on a tripped breaker, etc). Render
            # them as a refusal partial rather than a bare 500. The HTMX
            # caller swaps the same target either way, so a clean error
            # banner is friendlier than the browser's default error page.
            logger.warning(
                "ask.failed",
                error_type=type(e).__name__,
                error=str(e)[:200],
            )
            return templates.TemplateResponse(
                request,
                "_answer.html",
                {
                    "response": None,
                    "error": (f"Couldn't answer: {type(e).__name__}. {str(e)[:160]}"),
                },
                status_code=503,
            )
        return templates.TemplateResponse(
            request,
            "_answer.html",
            {"response": response, "error": None},
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
        return templates.TemplateResponse(
            request,
            "document.html",
            {
                "document": doc,
                "rendered_body": render_body_html(doc.body),
                "toc": extract_toc(doc.body),
                "has_source": has_source,
                "source_kind": source_kind,
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
        """Render the one-hop neighbourhood for `doc_id` (Cytoscape.js
        client-side). Returns `graph_available=False` + a fallback
        panel when ryugraph isn't installed."""
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

        neighbors: list[dict[str, Any]] = []
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
                raw = await store.neighbors(doc_id, limit=limit)
                neighbors = [n.model_dump() for n in raw]
            finally:
                await store.close()

        title = doc.frontmatter.title or doc_id
        nodes: list[dict[str, Any]] = [{"id": doc_id, "title": title, "kind": "center"}]
        edges: list[dict[str, Any]] = []
        seen_neighbor_ids: set[str] = {doc_id}
        for n in neighbors:
            other_id = n["doc_id"]
            if other_id in seen_neighbor_ids:
                # The graph store may return multiple edges per neighbor
                # (one per shared entity). Keep the first as a node;
                # accumulate the rest as additional edges.
                edges.append(
                    {
                        "source": doc_id,
                        "target": other_id,
                        "label": n.get("via") or n.get("relation", ""),
                    }
                )
                continue
            seen_neighbor_ids.add(other_id)
            nodes.append(
                {
                    "id": other_id,
                    "title": n.get("title") or other_id,
                    "kind": "neighbor",
                }
            )
            edges.append(
                {
                    "source": doc_id,
                    "target": other_id,
                    "label": n.get("via") or n.get("relation", ""),
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
