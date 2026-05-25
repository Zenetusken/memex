"""Unit tests for the vault store — focused on the concurrency layers
P1.4 (optimistic `expected_sha` CAS) and P1.5 (cross-process flock)
added to `write_document` / `delete_document`.

The store has plenty of behaviour exercised end-to-end by integration
tests (ingest/parse/index). These tests target the concurrency paths
specifically so a regression shows up under `pytest tests/unit -q`.
The cross-process scenario lives in `tests/integration/` because it
requires spawning a real subprocess.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from memex.core.errors import StaleDocumentError
from memex.vault.store import (
    Frontmatter,
    VaultDocument,
    hash_bytes,
    make_ref,
    write_document,
)


def _build_doc(vault: Path, doc_id: str, body: str) -> VaultDocument:
    return VaultDocument(
        ref=make_ref(
            vault,
            doc_id,
            content_sha256=hash_bytes(body.encode("utf-8")),
        ),
        frontmatter=Frontmatter(title=doc_id),
        body=body,
        mtime_ns=0,
    )


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / "documents").mkdir()
    return tmp_path


@pytest.mark.asyncio
async def test_write_document_accepts_correct_expected_sha(vault: Path) -> None:
    """CAS with the actual current sha → write succeeds."""
    doc_id = "00000001-cas-ok"
    ref1 = await write_document(vault, _build_doc(vault, doc_id, "v1"))
    next_doc = _build_doc(vault, doc_id, "v2")
    ref2 = await write_document(vault, next_doc, expected_sha=ref1.content_sha256)
    assert ref2.content_sha256 != ref1.content_sha256


@pytest.mark.asyncio
async def test_write_document_raises_stale_document_error_on_mismatch(
    vault: Path,
) -> None:
    """CAS with a stale sha → StaleDocumentError, on-disk file untouched."""
    doc_id = "00000002-stale"
    ref1 = await write_document(vault, _build_doc(vault, doc_id, "v1"))
    on_disk_after_first = ref1.markdown_path.read_text(encoding="utf-8")

    stale = "0" * 64
    with pytest.raises(StaleDocumentError) as exc:
        await write_document(
            vault,
            _build_doc(vault, doc_id, "v2"),
            expected_sha=stale,
        )
    assert exc.value.context["doc_id"] == doc_id
    assert exc.value.context["expected_sha"] == stale
    assert exc.value.context["current_sha"] == ref1.content_sha256
    # The current body in context must match what's on disk right now.
    assert exc.value.context["current_body"] == on_disk_after_first
    # Crucially, the on-disk file was not touched by the failed write.
    assert ref1.markdown_path.read_text(encoding="utf-8") == on_disk_after_first


@pytest.mark.asyncio
async def test_write_document_expected_sha_none_skips_cas_check(
    vault: Path,
) -> None:
    """The default `expected_sha=None` keeps the pre-P1.4 last-write-wins
    behaviour — important for ingest, parse, and other callers that
    legitimately overwrite without checking.
    """
    doc_id = "00000003-no-cas"
    await write_document(vault, _build_doc(vault, doc_id, "v1"))
    # No expected_sha → no raise, second write lands.
    ref2 = await write_document(vault, _build_doc(vault, doc_id, "v2"))
    assert ref2.markdown_path.read_text(encoding="utf-8").endswith(
        "v2\n"
    ) or "v2" in ref2.markdown_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_write_document_stale_error_when_doc_deleted_concurrently(
    vault: Path,
) -> None:
    """If the doc was deleted out from under us, treat it as a conflict
    rather than blindly creating a new file.
    """
    doc_id = "00000004-deleted"
    ref1 = await write_document(vault, _build_doc(vault, doc_id, "v1"))
    ref1.markdown_path.unlink()
    with pytest.raises(StaleDocumentError) as exc:
        await write_document(
            vault,
            _build_doc(vault, doc_id, "v2"),
            expected_sha=ref1.content_sha256,
        )
    assert exc.value.context["current_sha"] is None
    assert exc.value.context["current_body"] == ""


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl not available")
@pytest.mark.asyncio
async def test_write_document_creates_and_releases_lock_file(
    vault: Path,
) -> None:
    """After a successful write, the lock file exists at the canonical
    path and is in the released state — a fresh non-blocking
    flock-exclusive acquisition should succeed instantly.
    """
    import fcntl

    from memex.vault._file_lock import _lock_path

    doc_id = "00000005-lockfile"
    await write_document(vault, _build_doc(vault, doc_id, "v1"))
    lock = _lock_path(vault, doc_id)
    assert lock.exists()
    assert lock.stat().st_mode & 0o777 == 0o600

    # The lock must be released. A non-blocking exclusive acquisition
    # from another fd should succeed — if it raises BlockingIOError,
    # the previous writer leaked the flock.
    fd = os.open(str(lock), os.O_WRONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
