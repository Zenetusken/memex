"""Saved scope sets — named collections of document ids that scope `ask`.

A scope set is the *persistence* layer over the document scope-picker
(2026-05-27): instead of re-ticking the same documents every time, the user
names a selection once ("CR350 networking") and reapplies it by name. The CLI
(`memex ask --scope-set NAME`), the MCP `ask(scope_set=...)` tool, and the webui
picker all resolve a set to its `doc_ids` and feed the EXISTING
`answer_query(scope_doc_ids=...)` path — the answering agent never learns about
scope sets, so this is a pure naming layer with the same HARD-gate guarantee
(it can only NARROW retrieval; a stale set whose docs were removed resolves to an
empty pool → the agent refuses cleanly).

Storage mirrors `core/manifest.py`: a single JSON file at
`vault/.memex/scope_sets.json`, written atomically (`mkstemp` → `fsync` →
`os.replace`). It is **user-authored, not regenerable** — unlike the embeddings /
FTS / graph stores it is NOT in the `reindex_vault(force=True)` teardown
allow-list, so a full rebuild preserves it. (`.memex/` already holds non-derived
operational state — `events.sqlite`, `daemon/`, `locks/` — so a user-data file
sits there consistently; it is kept out of `documents/` because it is metadata
about the vault, not a document in it.)

The set NAME is a JSON value, never a path component (the file path is fixed),
so an arbitrary name carries no path-traversal risk — unlike a `doc_id`.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from memex.core.errors import ScopeSetError, VaultIntegrityError
from memex.core.manifest import now_utc

# A display name is capped so a fat-fingered paste can't bloat the file or the
# pickers. Generous for a human label; rejected (not truncated) past the cap so
# the user notices.
_MAX_NAME_LEN = 100

# One shared file → one in-process write lock (not a per-key dict like
# `vault.store._DOC_LOCKS`). The atomic `os.replace` keeps the file un-corrupt
# even across processes; the only cross-process hazard is a lost update
# (last-writer-wins), acceptable for a rare, manual convenience write. A
# cross-process fcntl lock would mean importing `vault._file_lock`, which would
# invert the `core/` import direction (core has no Memex deps) — not worth it.
_WRITE_LOCK = asyncio.Lock()


class ScopeSet(BaseModel):
    """A named collection of document ids the user scopes questions to."""

    name: str
    doc_ids: list[str]
    created_at: datetime
    updated_at: datetime


class ScopeSetCollection(BaseModel):
    """Every saved scope set in a vault — the on-disk JSON shape."""

    sets: list[ScopeSet] = Field(default_factory=list[ScopeSet])


def normalize_set_name(name: str) -> str:
    """The case- and whitespace-insensitive LOOKUP key for a set name.

    Collapses internal whitespace runs to a single space, strips, and
    casefolds — so `"  CR350   Networking "` and `"cr350 networking"` are the
    same set, while `"CR350"` and `"CR 350"` stay distinct (a space is
    meaningful, case is not). The DISPLAY name (`_clean_display_name`) keeps the
    original case.
    """
    return " ".join(name.split()).casefold()


def _clean_display_name(name: str) -> str:
    """The stored display name: whitespace-collapsed + stripped, case kept."""
    return " ".join(name.split())


def _dedup_doc_ids(doc_ids: list[str]) -> list[str]:
    """Strip blanks and de-duplicate, preserving first-seen order (so the
    scope reads in the order the user picked)."""
    return [d for d in dict.fromkeys(s.strip() for s in doc_ids) if d]


def scope_sets_path(vault_path: Path) -> Path:
    """The on-disk location of the scope-set store for a vault."""
    return vault_path / ".memex" / "scope_sets.json"


async def read_scope_sets(vault_path: Path) -> ScopeSetCollection:
    """Load every scope set. Missing file → an empty collection.

    A present-but-malformed file raises `VaultIntegrityError` (loud at the
    management surface — `scope-set list` / `show`). The answer path uses
    `resolve_scope_set_doc_ids`, which swallows this so a corrupt file can never
    break `ask`.
    """
    path = scope_sets_path(vault_path)
    if not path.exists():
        return ScopeSetCollection()
    text = await asyncio.to_thread(path.read_text, encoding="utf-8")
    try:
        return ScopeSetCollection.model_validate_json(text)
    except ValueError as e:  # pydantic ValidationError + json decode both subclass ValueError
        raise VaultIntegrityError(
            "scope_sets.json is present but malformed",
            context={"path": str(path), "error": str(e)[:200]},
        ) from e


async def write_scope_sets(vault_path: Path, collection: ScopeSetCollection) -> Path:
    """Atomically persist the whole collection (mirrors `write_manifest`)."""
    path = scope_sets_path(vault_path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    content = collection.model_dump_json(indent=2)

    def _write() -> None:
        fd, tmp = tempfile.mkstemp(prefix=f"{path.name}.tmp.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    await asyncio.to_thread(_write)
    return path


async def list_scope_sets(vault_path: Path) -> list[ScopeSet]:
    """Every saved set, ordered by normalized name (stable run-to-run)."""
    collection = await read_scope_sets(vault_path)
    return sorted(collection.sets, key=lambda s: normalize_set_name(s.name))


async def get_scope_set(vault_path: Path, name: str) -> ScopeSet | None:
    """The set matching `name` (case/whitespace-insensitive), or None."""
    key = normalize_set_name(name)
    if not key:
        return None
    collection = await read_scope_sets(vault_path)
    for s in collection.sets:
        if normalize_set_name(s.name) == key:
            return s
    return None


async def save_scope_set(vault_path: Path, name: str, doc_ids: list[str]) -> ScopeSet:
    """Create or update a scope set (upsert by normalized name).

    Validates: a non-empty display name within `_MAX_NAME_LEN`; at least one
    non-blank doc id. `doc_ids` are stripped + de-duplicated. An update keeps the
    original `created_at` and bumps `updated_at`. Raises `ScopeSetError` on
    invalid input.
    """
    display = _clean_display_name(name)
    if not display:
        raise ScopeSetError(
            "a scope set needs a non-empty name",
            context={"name": name},
        )
    if len(display) > _MAX_NAME_LEN:
        raise ScopeSetError(
            f"scope set name is too long (max {_MAX_NAME_LEN} characters)",
            context={"name": display, "length": len(display), "max": _MAX_NAME_LEN},
        )
    cleaned_ids = _dedup_doc_ids(doc_ids)
    if not cleaned_ids:
        raise ScopeSetError(
            "a scope set needs at least one document",
            context={"name": display},
        )

    key = normalize_set_name(display)
    async with _WRITE_LOCK:
        collection = await read_scope_sets(vault_path)
        now = now_utc()
        kept: list[ScopeSet] = []
        existing: ScopeSet | None = None
        for s in collection.sets:
            if normalize_set_name(s.name) == key:
                existing = s  # drop the old record; replace below
            else:
                kept.append(s)
        record = ScopeSet(
            name=display,
            doc_ids=cleaned_ids,
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
        )
        kept.append(record)
        await write_scope_sets(vault_path, ScopeSetCollection(sets=kept))
        return record


async def delete_scope_set(vault_path: Path, name: str) -> bool:
    """Remove a scope set by name. Returns True if one was removed.

    Deletes only the named collection — never a document. A missing set is a
    no-op (returns False)."""
    key = normalize_set_name(name)
    if not key:
        return False
    async with _WRITE_LOCK:
        collection = await read_scope_sets(vault_path)
        kept = [s for s in collection.sets if normalize_set_name(s.name) != key]
        if len(kept) == len(collection.sets):
            return False
        await write_scope_sets(vault_path, ScopeSetCollection(sets=kept))
        return True


async def resolve_scope_set_doc_ids(vault_path: Path, name: str) -> list[str]:
    """The doc ids of the named set, for feeding `answer_query(scope_doc_ids=)`.

    Fail-OPEN: an unknown name, an empty name, or a corrupt store all return
    `[]` (the caller then takes the full-corpus path) — resolving a scope set
    must NEVER raise into the answer path. Surfaces that want to distinguish
    "no such set" (e.g. the CLI / MCP, to error helpfully) call `get_scope_set`
    first.
    """
    try:
        s = await get_scope_set(vault_path, name)
    except VaultIntegrityError:
        return []
    return list(s.doc_ids) if s is not None else []
