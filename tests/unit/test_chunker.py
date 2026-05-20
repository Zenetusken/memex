"""Chunker unit tests — runs without any models or external services."""

from __future__ import annotations

from pathlib import Path

import pytest

from memex.index.chunker import (
    OVERLAP_TOKENS,
    TARGET_TOKENS,
    chunk_document,
)
from memex.vault.store import DocumentRef, Frontmatter, VaultDocument


def _doc(body: str, *, doc_id: str = "abc12345-test", title: str = "Test") -> VaultDocument:
    ref = DocumentRef(
        doc_id=doc_id,
        markdown_path=Path(f"/tmp/{doc_id}.md"),  # noqa: S108 — test fixture
        asset_dir=Path(f"/tmp/{doc_id}"),  # noqa: S108
        source_path=None,
        content_sha256="0" * 64,
    )
    return VaultDocument(
        ref=ref,
        frontmatter=Frontmatter(title=title),
        body=body,
        mtime_ns=0,
    )


def test_empty_body_yields_no_chunks() -> None:
    assert chunk_document(_doc("")) == []


def test_short_body_yields_one_chunk() -> None:
    chunks = chunk_document(_doc("A short paragraph about reflexivity."))
    assert len(chunks) == 1
    assert chunks[0].document_title == "Test"
    assert chunks[0].text.startswith("A short paragraph")
    assert chunks[0].heading_path == []


def test_heading_path_tracks_h2_h3_nesting() -> None:
    body = (
        "Pre-heading paragraph.\n\n"
        "## Methodology\n\n"
        "Methods paragraph.\n\n"
        "### Cohort\n\n"
        "Cohort paragraph.\n\n"
        "## Results\n\n"
        "Results paragraph.\n"
    )
    chunks = chunk_document(_doc(body))
    by_text = {c.text.split("\n")[0]: c.heading_path for c in chunks}
    assert by_text.get("## Methodology") == ["Methodology"] or any(
        "Methodology" in c.heading_path for c in chunks
    )
    cohort_chunk = next(c for c in chunks if "Cohort paragraph" in c.text)
    assert cohort_chunk.heading_path == ["Methodology", "Cohort"]
    results_chunk = next(c for c in chunks if "Results paragraph" in c.text)
    assert results_chunk.heading_path == ["Results"]


def test_chunks_are_idempotent() -> None:
    body = "## A\n\nFirst paragraph.\n\n## B\n\nSecond paragraph."
    first = chunk_document(_doc(body))
    second = chunk_document(_doc(body))
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_target_window_is_respected_for_long_content() -> None:
    # ~1800-word body across multiple sections forces window splits.
    # Test input must contain real sentence boundaries — the chunker's
    # sentence splitter looks for `[.!?]\s+[A-Z0-9]`, so a "lorem lorem
    # lorem" run-on doesn't trigger it.
    sentence = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
        "eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    )
    body = "\n\n".join(
        f"## Section {i}\n\n" + (sentence * 50) for i in range(3)
    )
    chunks = chunk_document(_doc(body))
    assert len(chunks) >= 2, "expected at least one window split"
    for c in chunks:
        # Allow some slack for overlap carry-over.
        assert len(c.text.split()) <= TARGET_TOKENS + OVERLAP_TOKENS + 200


@pytest.mark.parametrize("body", ["", "   ", "\n\n\n"])
def test_pathological_inputs_dont_crash(body: str) -> None:
    chunks = chunk_document(_doc(body))
    assert isinstance(chunks, list)
