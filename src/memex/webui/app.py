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
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from memex.agents.answering import answer_query
from memex.core.config import get_settings
from memex.core.errors import VaultIntegrityError
from memex.core.manifest import update_manifest
from memex.index.graph_store import GraphStore
from memex.vault.store import (
    VaultDocument,
    hash_bytes,
    list_documents,
    make_ref,
    read_document,
    write_document,
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


def create_app() -> FastAPI:
    """Build the FastAPI app. Factory so tests can instantiate freely."""
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
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
        return templates.TemplateResponse(request, "index.html", {})

    @app.post("/ask", response_class=HTMLResponse)
    async def ask(
        request: Request,
        question: str = Form(..., max_length=_QUESTION_MAX_BYTES),
    ) -> HTMLResponse:
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
                    "error": (
                        f"Couldn't answer: {type(e).__name__}. "
                        f"{str(e)[:160]}"
                    ),
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
        settings = get_settings()
        refs: list[Any] = []
        async for ref in list_documents(settings.vault_path):
            refs.append(ref)
        return templates.TemplateResponse(
            request, "documents.html", {"documents": refs}
        )

    @app.get("/documents/{doc_id}", response_class=HTMLResponse)
    async def document(request: Request, doc_id: str) -> HTMLResponse:
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
                "has_source": has_source,
                "source_kind": source_kind,
            },
        )

    @app.get("/documents/{doc_id}/source")
    async def document_source(doc_id: str) -> FileResponse:
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
            {"document": doc, "just_saved": None},
        )

    @app.post("/documents/{doc_id}/review", response_class=HTMLResponse)
    async def document_review(
        request: Request,
        doc_id: str,
        body: str = Form(..., max_length=_BODY_MAX_BYTES),
    ) -> HTMLResponse:
        """Apply an edit. Updates the manifest's `content_sha256` with
        the post-write hash BEFORE writing the markdown, so a kill
        between the two operations leaves a manifest that anticipates
        the new file — when the doc is re-read, sha matches and the
        watcher's `_confirm_user_edit` doesn't treat the self-write as
        a fresh user edit.

        If the kill happens between manifest-update and file-write, the
        sha goes stale in the *opposite* direction: the on-disk content
        is the OLD file but the manifest claims the NEW sha. On next
        restart, `_confirm_user_edit` sees a mismatch and re-triggers —
        which re-enrich/re-index is the correct recovery: the edit is
        lost, but the index reflects what's actually on disk.
        """
        doc_id = _validate_doc_id(doc_id)
        settings = get_settings()
        try:
            existing = await read_document(settings.vault_path, doc_id)
        except VaultIntegrityError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        # Compute the post-write sha by serialising the new document
        # the same way `write_document` does (frontmatter is round-
        # tripped). We pre-update the manifest's content_sha256 with
        # this anticipated value, then perform the atomic file write.
        # See the docstring for the kill-window analysis.
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
        new_ref = await write_document(settings.vault_path, new_doc)
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
                "just_saved": datetime.now().strftime("%H:%M:%S"),
            },
        )

    # ----- Graph -----

    @app.get("/graph/{doc_id}", response_class=HTMLResponse)
    async def graph(
        request: Request,
        doc_id: str,
        limit: int = 50,
    ) -> HTMLResponse:
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
        nodes: list[dict[str, Any]] = [
            {"id": doc_id, "title": title, "kind": "center"}
        ]
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
        settings = get_settings()
        return {
            "status": "ok",
            "vault_path": str(settings.vault_path),
        }

    return app


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
