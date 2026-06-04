"""Companion-merge alignment sidecar (ADR-0018, spec docs/specs/companion-merge.md §5).

The atomic-write JSON store for per-pair transcript↔deck alignments. Mirrors `core/scope_sets.py`'s
storage pattern (single `vault/.memex/companion_alignments.json`, `mkstemp`→`fsync`→`os.replace`, one
in-process write lock) but is **DERIVED state** — regenerable from the indexed chunks + the embedder, so
unlike user-authored `scope_sets` it BELONGS in the `reindex_vault(force=True)` teardown allow-list.

Reads are FAIL-OPEN for the `/ask` `augment_companion` node + doc views (`read_alignments_open`): a
corrupt or missing file resolves to "no alignment" and NEVER breaks the answer path or a page render.
The MANAGEMENT surface (`memex link-slides list`) calls `read_alignments`, which raises
`VaultIntegrityError` loudly on a malformed file. This is `core/` (no Memex deps beyond `core`) so the
alignment COMPUTE (embedder + FTSStore) lives in `index/companion.py`, not here.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from memex.core.errors import VaultIntegrityError
from memex.core.types import CompanionAlignment

# One shared file → one in-process write lock (the `scope_sets` rationale: atomic `os.replace` keeps
# the file un-corrupt cross-process; the only hazard is a lost update, acceptable for a rare op).
_WRITE_LOCK = asyncio.Lock()


class CompanionAlignmentCollection(BaseModel):
    """Every transcript↔deck alignment in a vault — the on-disk JSON shape."""

    pairs: list[CompanionAlignment] = Field(default_factory=list[CompanionAlignment])


def companion_alignments_path(vault_path: Path) -> Path:
    """The on-disk location of the companion-alignment store for a vault."""
    return vault_path / ".memex" / "companion_alignments.json"


async def read_alignments(vault_path: Path) -> CompanionAlignmentCollection:
    """Load every alignment. Missing file → an empty collection; a present-but-malformed file raises
    `VaultIntegrityError` (loud at the management surface). The `/ask` + doc-view paths use
    `read_alignments_open`, which swallows that so a corrupt file can never break them."""
    path = companion_alignments_path(vault_path)
    if not path.exists():
        return CompanionAlignmentCollection()
    text = await asyncio.to_thread(path.read_text, encoding="utf-8")
    try:
        return CompanionAlignmentCollection.model_validate_json(text)
    except ValueError as e:  # pydantic ValidationError + json decode both subclass ValueError
        raise VaultIntegrityError(
            "companion_alignments.json is present but malformed",
            context={"path": str(path), "error": str(e)[:200]},
        ) from e


async def read_alignments_open(vault_path: Path) -> list[CompanionAlignment]:
    """FAIL-OPEN: a corrupt/missing store → `[]` (for the augmentation node + doc views)."""
    try:
        return (await read_alignments(vault_path)).pairs
    except VaultIntegrityError:
        return []


async def write_alignments(vault_path: Path, collection: CompanionAlignmentCollection) -> Path:
    """Atomically persist the whole collection (mirrors `write_scope_sets`)."""
    path = companion_alignments_path(vault_path)
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


async def get_alignment(
    vault_path: Path, transcript_doc: str, deck_doc: str
) -> CompanionAlignment | None:
    """The alignment for exactly this (transcript, deck) pair, or None (fail-open read)."""
    for a in await read_alignments_open(vault_path):
        if a.transcript_doc == transcript_doc and a.deck_doc == deck_doc:
            return a
    return None


async def alignments_for_doc(vault_path: Path, doc_id: str) -> list[CompanionAlignment]:
    """Every pair where `doc_id` is EITHER the transcript OR the deck — the bidirectional lookup the
    `augment_companion` node + the doc-view "Companion" surface use (fail-open read)."""
    return [
        a
        for a in await read_alignments_open(vault_path)
        if doc_id in (a.transcript_doc, a.deck_doc)
    ]


async def upsert_alignment(vault_path: Path, alignment: CompanionAlignment) -> None:
    """Add or REPLACE the (transcript, deck) pair, atomically (last-writer-wins per the lock)."""
    async with _WRITE_LOCK:
        existing = await read_alignments_open(vault_path)
        kept = [
            a
            for a in existing
            if not (a.transcript_doc == alignment.transcript_doc and a.deck_doc == alignment.deck_doc)
        ]
        kept.append(alignment)
        await write_alignments(vault_path, CompanionAlignmentCollection(pairs=kept))


async def delete_alignment(vault_path: Path, transcript_doc: str, deck_doc: str) -> bool:
    """Remove the (transcript, deck) pair. Returns True if one was removed (a missing pair → False)."""
    async with _WRITE_LOCK:
        existing = await read_alignments_open(vault_path)
        kept = [
            a
            for a in existing
            if not (a.transcript_doc == transcript_doc and a.deck_doc == deck_doc)
        ]
        if len(kept) == len(existing):
            return False
        await write_alignments(vault_path, CompanionAlignmentCollection(pairs=kept))
        return True
