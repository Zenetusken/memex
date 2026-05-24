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
    body = "\n\n".join(f"## Section {i}\n\n" + (sentence * 50) for i in range(3))
    chunks = chunk_document(_doc(body))
    assert len(chunks) >= 2, "expected at least one window split"
    for c in chunks:
        # Allow some slack for overlap carry-over.
        assert len(c.text.split()) <= TARGET_TOKENS + OVERLAP_TOKENS + 200


@pytest.mark.parametrize("body", ["", "   ", "\n\n\n"])
def test_pathological_inputs_dont_crash(body: str) -> None:
    chunks = chunk_document(_doc(body))
    assert isinstance(chunks, list)


# ----- Oversized-table force-split cap (MAX_CHUNK_MULTIPLIER) -----

from memex.index.chunker import MAX_CHUNK_MULTIPLIER, _budget_word_count  # noqa: E402

_CAP = TARGET_TOKENS * MAX_CHUNK_MULTIPLIER


def _big_table(n_rows: int) -> str:
    header = "| Line item | FY25 | FY24 |\n|---|---|---|"
    rows = "\n".join(f"| Item number {i} | {i * 100} | {i * 97} |" for i in range(n_rows))
    return header + "\n" + rows


def test_oversized_table_force_split_into_bounded_chunks() -> None:
    body = f"## Financial Statements\n\n{_big_table(400)}\n"
    chunks = chunk_document(_doc(body))
    assert len(chunks) > 1
    # Every chunk's budget stays under the hard cap.
    assert all(_budget_word_count(c.text) <= _CAP for c in chunks)


def test_force_split_keeps_table_rows_intact() -> None:
    body = f"## Financial Statements\n\n{_big_table(400)}\n"
    chunks = chunk_document(_doc(body))
    for c in chunks:
        for line in c.text.splitlines():
            if line.lstrip().startswith("|"):
                assert line.rstrip().endswith("|"), f"row cut mid-pipe: {line!r}"
    # A known row appears whole, in exactly one chunk.
    needle = "| Item number 250 | 25000 | 24250 |"
    assert sum(needle in c.text for c in chunks) == 1


def test_force_split_offsets_roundtrip_exactly() -> None:
    body = f"## Financial Statements\n\n{_big_table(400)}\n"
    doc = _doc(body)
    chunks = chunk_document(doc)
    table_chunks = [c for c in chunks if c.text.lstrip().startswith("|")]
    assert table_chunks
    for c in table_chunks:
        # Single-element windows round-trip: body slice == chunk text verbatim.
        assert doc.body[c.char_start : c.char_end] == c.text
    # Offsets are monotonic non-decreasing across the table.
    starts = [c.char_start for c in table_chunks]
    assert starts == sorted(starts)


def test_force_split_preserves_heading_path() -> None:
    body = f"## Financial Statements\n\n{_big_table(400)}\n"
    chunks = chunk_document(_doc(body))
    table_chunks = [c for c in chunks if c.text.lstrip().startswith("|")]
    assert table_chunks
    assert all(c.heading_path == ["Financial Statements"] for c in table_chunks)


def test_cap_does_not_fire_on_normal_prose() -> None:
    # Many short sentences > the cap in total, but each sentence is tiny, so
    # the sentence-splitter handles it — the force-split path must NOT engage
    # (no mid-content line cuts) and chunking stays deterministic.
    body = "## Intro\n\n" + " ".join(f"Sentence number {i} here." for i in range(400))
    c1 = chunk_document(_doc(body))
    c2 = chunk_document(_doc(body))
    assert [c.chunk_id for c in c1] == [c.chunk_id for c in c2]  # deterministic
    assert all(_budget_word_count(c.text) <= _CAP for c in c1)


def test_oversized_chart_block_not_force_split() -> None:
    # A chart-extracted block is huge in raw words but ~0 budget-words, so the
    # cap must not fire — it stays a single chunk (gated by _budget_word_count).
    chart = "[chart-extracted]\n" + ("data point " * 2000) + "\n[/chart-extracted]"
    body = f"## Figure\n\n{chart}\n"
    chunks = chunk_document(_doc(body))
    assert sum("[chart-extracted]" in c.text for c in chunks) == 1
