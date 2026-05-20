"""Markdown vault read/write — see IMPLEMENTATION-PLAN.md §1.1.

Sole owner of every byte under `vault/documents/`. All other modules
read and write the canonical store through this interface, never
directly. Implements ADR-0003's "Markdown wins" rule operationally:
atomic writes (tempfile + rename), frontmatter round-trip without loss,
doc-id-to-path resolution.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import tempfile
from collections.abc import AsyncIterator
from datetime import date as _date
from pathlib import Path
from typing import Any

import frontmatter
from pydantic import BaseModel, Field

from memex.core.errors import VaultIntegrityError

# Per-`doc_id` write serialisation. `_atomic_write` is atomic at the
# tempfile-rename level, but two coroutines calling `write_document` on
# the same `doc_id` concurrently would each prepare their own tempfile
# and the loser's data would be silently discarded by the last
# `os.replace`. The lock makes write+delete on a given doc id
# sequential within the process. Cross-process safety would need
# `fcntl.LOCK_EX`; today single-process is the only supported topology
# (CLI, daemon, watcher all run in the same Python process).
_DOC_LOCKS: dict[str, asyncio.Lock] = {}


def _lock_for(doc_id: str) -> asyncio.Lock:
    lock = _DOC_LOCKS.get(doc_id)
    if lock is None:
        lock = asyncio.Lock()
        _DOC_LOCKS[doc_id] = lock
    return lock

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_ID_PREFIX_LEN = 8
_SLUG_MAX_LEN = 48


class DocumentRef(BaseModel):
    """A vault document's identity and on-disk layout."""

    doc_id: str
    markdown_path: Path
    asset_dir: Path
    source_path: Path | None = None
    content_sha256: str


class Frontmatter(BaseModel):
    """The canonical frontmatter schema. Unknown fields land in `custom`.

    `date` is annotated `_date | None` rather than `date | None`
    because the field name shadows the imported `date` type during
    pydantic's annotation evaluation under `from __future__ import
    annotations`. Aliasing the import dodges the footgun.
    """

    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    date: _date | None = None
    source_url: str | None = None
    license: str | None = None
    tags: list[str] = Field(default_factory=list)
    custom: dict[str, Any] = Field(default_factory=dict)


class VaultDocument(BaseModel):
    """An in-memory view of a vault document."""

    ref: DocumentRef
    frontmatter: Frontmatter
    body: str
    mtime_ns: int


def assign_doc_id(content_sha256: str, source_stem: str) -> str:
    """Derive a stable, collision-resistant doc_id.

    First 8 hex chars of the content sha256 (collision-resistant key),
    plus a human-readable slug suffix (for grep / ls). The full sha256
    is recorded in the manifest; this is just the on-disk name.
    """
    prefix = content_sha256[:_ID_PREFIX_LEN]
    slug = _SLUG_RE.sub("-", source_stem.lower()).strip("-")[:_SLUG_MAX_LEN]
    return f"{prefix}-{slug}" if slug else prefix


def hash_bytes(data: bytes) -> str:
    """sha256 hex of bytes — exposed so callers can hash before write."""
    return hashlib.sha256(data).hexdigest()


def _docs_root(vault_path: Path) -> Path:
    return vault_path / "documents"


def _markdown_path(vault_path: Path, doc_id: str) -> Path:
    return _docs_root(vault_path) / f"{doc_id}.md"


def _asset_dir(vault_path: Path, doc_id: str) -> Path:
    return _docs_root(vault_path) / doc_id


def _atomic_write(path: Path, content: str) -> None:
    """Write `content` to `path` atomically (tempfile + fsync + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{path.name}.tmp.",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _ref_for(
    vault_path: Path,
    doc_id: str,
    content_sha256: str,
    source_path: Path | None,
) -> DocumentRef:
    return DocumentRef(
        doc_id=doc_id,
        markdown_path=_markdown_path(vault_path, doc_id),
        asset_dir=_asset_dir(vault_path, doc_id),
        source_path=source_path,
        content_sha256=content_sha256,
    )


def make_ref(
    vault_path: Path,
    doc_id: str,
    *,
    content_sha256: str = "0" * 64,
    source_path: Path | None = None,
) -> DocumentRef:
    """Public constructor for the layout the vault enforces.

    Use this from other modules that need to hand `write_document` a
    fresh `DocumentRef`. It exists so callers don't reach for the
    private `_ref_for`.
    """
    return _ref_for(vault_path, doc_id, content_sha256, source_path)


async def write_document(vault_path: Path, doc: VaultDocument) -> DocumentRef:
    """Atomically write `doc` to `vault/documents/{doc_id}.md`.

    Recomputes the content sha256 and updates the returned ref. Caller
    is responsible for re-reading mtime if they need it.

    Serialised per-`doc_id` via `_lock_for(doc_id)` so two concurrent
    callers on the same doc don't both `os.replace` and silently drop
    one's data. Different doc_ids are unaffected.
    """
    async with _lock_for(doc.ref.doc_id):
        fm_dict: dict[str, Any] = {
            **doc.frontmatter.model_dump(exclude={"custom"}, exclude_none=True),
            **doc.frontmatter.custom,
        }
        post = frontmatter.Post(doc.body, **fm_dict)
        serialized = frontmatter.dumps(post)
        _atomic_write(doc.ref.markdown_path, serialized)
        new_sha = hash_bytes(serialized.encode("utf-8"))
        return _ref_for(
            vault_path,
            doc.ref.doc_id,
            new_sha,
            doc.ref.source_path,
        )


async def read_document(vault_path: Path, doc_id: str) -> VaultDocument:
    """Load a vault document. Raises VaultIntegrityError if missing."""
    path = _markdown_path(vault_path, doc_id)
    if not path.exists():
        raise VaultIntegrityError(
            f"vault document not found: {doc_id}",
            context={"doc_id": doc_id, "path": str(path)},
        )
    text = path.read_text(encoding="utf-8")
    post = frontmatter.loads(text)

    canonical_keys = {"title", "authors", "date", "source_url", "license", "tags"}
    canonical = {k: post.metadata[k] for k in canonical_keys if k in post.metadata}
    custom = {k: v for k, v in post.metadata.items() if k not in canonical_keys}
    fm = Frontmatter(**canonical, custom=custom)

    return VaultDocument(
        ref=_ref_for(vault_path, doc_id, hash_bytes(text.encode("utf-8")), None),
        frontmatter=fm,
        body=post.content,
        mtime_ns=path.stat().st_mtime_ns,
    )


async def list_documents(vault_path: Path) -> AsyncIterator[DocumentRef]:
    """Yield refs for every `.md` file directly under `vault/documents/`."""
    root = _docs_root(vault_path)
    if not root.exists():
        return
    for md in sorted(root.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        yield _ref_for(
            vault_path,
            md.stem,
            hash_bytes(text.encode("utf-8")),
            None,
        )


async def create_document(
    vault_path: Path,
    *,
    body: str,
    source_stem: str,
    frontmatter_fields: Frontmatter | None = None,
) -> DocumentRef:
    """Convenience for tests and the `ingest --skip-parse` Phase-0 path.

    Hashes the body, derives a doc_id, writes the markdown atomically.
    """
    fm = frontmatter_fields or Frontmatter()
    content_sha = hash_bytes(body.encode("utf-8"))
    doc_id = assign_doc_id(content_sha, source_stem)
    ref = _ref_for(vault_path, doc_id, content_sha, None)
    doc = VaultDocument(ref=ref, frontmatter=fm, body=body, mtime_ns=0)
    return await write_document(vault_path, doc)


async def delete_document(vault_path: Path, doc_id: str) -> None:
    """Remove the markdown and its asset dir. Caller invalidates indexes.

    Serialised against concurrent writes on the same `doc_id` via the
    per-doc lock — a delete racing a write would otherwise leave a
    partially-removed asset directory.
    """
    async with _lock_for(doc_id):
        md = _markdown_path(vault_path, doc_id)
        md.unlink(missing_ok=True)
        asset = _asset_dir(vault_path, doc_id)
        if asset.exists():
            for child in asset.rglob("*"):
                if child.is_file():
                    child.unlink()
            for child in sorted(asset.rglob("*"), reverse=True):
                if child.is_dir():
                    child.rmdir()
            asset.rmdir()
    _DOC_LOCKS.pop(doc_id, None)
