"""Unit tests for `parse.pipeline.derive_title` — deriving a readable
frontmatter title from the manifest's original source filename instead
of the doc-id slug.

A meaningful title is what lets `enrich.citations` resolve cross-document
references (it scores candidates against other docs' titles), so this is
the enabler for the P4.1 wikilink chain on PDF-ingested docs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from memex.core.manifest import IngestStage, Manifest, write_manifest
from memex.parse.pipeline import derive_title


async def _write_manifest_with_source(
    vault: Path, doc_id: str, source_path: str
) -> None:
    await write_manifest(
        vault,
        Manifest(
            doc_id=doc_id,
            content_sha256="0" * 64,
            ingest=IngestStage(
                correlation_id="c0",
                ingested_at=datetime.now(UTC),
                source_path=source_path,
                source_size_bytes=123,
                detected_mime="application/pdf",
            ),
        ),
    )


@pytest.mark.asyncio
async def testderive_title_from_source_filename(tmp_path: Path) -> None:
    """The manifest's `ingest.source_path` stem becomes the title."""
    await _write_manifest_with_source(
        tmp_path,
        "5795b16a-cr350-cours-2",
        "/home/u/Documents/School/CR350/CR350 - Cours 2.pdf",
    )
    title = await derive_title(tmp_path, "5795b16a-cr350-cours-2")
    assert title == "CR350 - Cours 2"


@pytest.mark.asyncio
async def testderive_title_preserves_full_topic(tmp_path: Path) -> None:
    """Long descriptive filenames carry through intact (no truncation)."""
    name = "CR350 - Cours 6 - Coupe-feu, serveurs mandataires, détection d’intrusion, 802.1X"
    await _write_manifest_with_source(
        tmp_path, "abcd1234-cr350-cours-6", f"/x/{name}.pdf"
    )
    title = await derive_title(tmp_path, "abcd1234-cr350-cours-6")
    assert title == name


@pytest.mark.asyncio
async def testderive_title_falls_back_to_doc_id_when_no_manifest(
    tmp_path: Path,
) -> None:
    """No manifest on disk → fall back to the doc_id."""
    title = await derive_title(tmp_path, "deadbeef-orphan")
    assert title == "deadbeef-orphan"


@pytest.mark.asyncio
async def testderive_title_falls_back_when_no_ingest_stage(
    tmp_path: Path,
) -> None:
    """Manifest exists but has no ingest stage → fall back to doc_id."""
    await write_manifest(
        tmp_path,
        Manifest(doc_id="deadbeef-no-ingest", content_sha256="0" * 64),
    )
    title = await derive_title(tmp_path, "deadbeef-no-ingest")
    assert title == "deadbeef-no-ingest"


@pytest.mark.asyncio
async def testderive_title_skips_inline_passthrough_source(
    tmp_path: Path,
) -> None:
    """Inline markdown passthrough records `<inline:stem>` as the
    source_path — don't surface that placeholder as a title."""
    await _write_manifest_with_source(
        tmp_path, "abc1-inline", "<inline:some-stem>"
    )
    title = await derive_title(tmp_path, "abc1-inline")
    assert title == "abc1-inline"
