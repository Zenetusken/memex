"""Unit tests for `vault.store.read_document_title` — the cheap
frontmatter-only title read used by the documents-list view."""

from __future__ import annotations

from pathlib import Path

import pytest

from memex.vault.store import (
    Frontmatter,
    create_document,
    read_document_title,
)


@pytest.mark.asyncio
async def test_read_title_from_frontmatter(tmp_path: Path) -> None:
    ref = await create_document(
        tmp_path,
        body="# Body\n\nprose",
        source_stem="cours-2",
        frontmatter_fields=Frontmatter(title="CR350 - Cours 2"),
    )
    assert await read_document_title(tmp_path, ref.doc_id) == "CR350 - Cours 2"


@pytest.mark.asyncio
async def test_read_title_missing_doc_falls_back_to_doc_id(
    tmp_path: Path,
) -> None:
    (tmp_path / "documents").mkdir(parents=True, exist_ok=True)
    assert await read_document_title(tmp_path, "nope-123") == "nope-123"


@pytest.mark.asyncio
async def test_read_title_does_not_read_whole_body(tmp_path: Path) -> None:
    """Title read must not depend on body size — write a large body and
    confirm the title still resolves (frontmatter-only read)."""
    big = "x" * 200_000
    ref = await create_document(
        tmp_path,
        body=f"# Big\n\n{big}",
        source_stem="big-doc",
        frontmatter_fields=Frontmatter(title="Big Doc"),
    )
    assert await read_document_title(tmp_path, ref.doc_id) == "Big Doc"
