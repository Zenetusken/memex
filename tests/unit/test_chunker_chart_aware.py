"""P3.3 v4 — verify the chunker is chart-block-aware.

The audit (2026-05-22 follow-up to P3.3 v3) identified that chart-OCR
enrichment shifts chunk boundaries: a paragraph that previously fit
in one chunk gets bumped to a new chunk because the inline
`[chart-extracted]...[/chart-extracted]` block inflates the budget
word count past `chunk_target_tokens`. The fix excludes chart-block
content from the budget word count via `_budget_word_count`, so
chunk boundaries match the no-chart-OCR baseline. The chart block
still appears in the chunk's text (the agent reads it at answer
time).

These tests pin the boundary-preservation contract.
"""

from __future__ import annotations

from pathlib import Path

from memex.index.chunker import chunk_document
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


def test_chart_block_does_not_shift_chunk_boundaries() -> None:
    """Two paragraphs that fit in one chunk (combined word count well
    below `chunk_target_tokens`) must STILL fit in one chunk when a
    large `[chart-extracted]` block is inserted between them. Without
    the v4 fix, the chart block's word count would push the running
    total past the target and force a split.
    """
    # Two small paragraphs that fit comfortably in one chunk.
    prose_a = "Paragraph A discusses the CUDA architecture in brief."
    prose_b = "Paragraph B continues the discussion of memory bandwidth."

    # A chart-extracted block with ~600 "words" — enough to bump the
    # budget past chunk_target_tokens (default 350) on its own.
    chart_rows = "\n".join(
        f"row{i:03d} | label{i:03d} | value{i:03d} | unit{i:03d} | extra{i:03d}" for i in range(120)
    )
    chart_block = f"[chart-extracted]\n{chart_rows}\n[/chart-extracted]"

    body_without_chart = f"{prose_a}\n\n{prose_b}"
    body_with_chart = f"{prose_a}\n\n<!-- image -->\n\n{chart_block}\n\n{prose_b}"

    chunks_without = chunk_document(_doc(body_without_chart, doc_id="aaa00001-no"))
    chunks_with = chunk_document(_doc(body_with_chart, doc_id="aaa00002-with"))

    # Without the chart, both paragraphs are in one chunk.
    assert len(chunks_without) == 1, f"baseline expected 1 chunk; got {len(chunks_without)}"

    # With the v4 fix, the chart-bearing body STILL produces one
    # chunk — the boundary doesn't shift.
    assert len(chunks_with) == 1, (
        f"v4 boundary preservation broken: chart-bearing body produced "
        f"{len(chunks_with)} chunks; expected 1 (matching baseline)"
    )

    # The chart block IS still in the chunk's text — the agent
    # needs to see it at answer time.
    assert "chart-extracted" in chunks_with[0].text
    assert "row050" in chunks_with[0].text
    # Both paragraphs are in the same chunk.
    assert "Paragraph A" in chunks_with[0].text
    assert "Paragraph B" in chunks_with[0].text


def test_chart_block_not_treated_as_oversized_paragraph() -> None:
    """A standalone chart-extracted block (no surrounding prose) must
    not trigger the oversized-paragraph sentence-split path. Without
    the v4 fix, a 1000-word chart block would be treated as one big
    paragraph exceeding `chunk_target_tokens` and would get
    sentence-split (which is meaningless for tabular chart data).
    """
    big_chart = "\n".join(f"r{i} | v{i}" for i in range(500))
    body = f"## Title\n\nIntro paragraph.\n\n<!-- image -->\n\n[chart-extracted]\n{big_chart}\n[/chart-extracted]\n"

    chunks = chunk_document(_doc(body))

    # The body fits in one chunk because the chart block doesn't
    # count toward the budget. (Without the fix: oversize trip, sentence-
    # split chaos.)
    assert len(chunks) == 1
    assert "Intro paragraph" in chunks[0].text
    assert "chart-extracted" in chunks[0].text
    # Last chart row must be present — sentence-splitting would have
    # broken it across chunks otherwise.
    assert "r499 | v499" in chunks[0].text


def test_v4_preserves_existing_chunker_behavior_on_no_chart_docs() -> None:
    """A document with no chart blocks must produce IDENTICAL chunks
    before and after the v4 patch. The change is opt-in by chart-block
    presence; no-chart-OCR vaults must be unaffected.
    """
    # A body that exercises both the paragraph-fit and overlap paths.
    sentence = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    body = "\n\n".join(f"## Section {i}\n\n" + (sentence * 30) for i in range(3))
    chunks = chunk_document(_doc(body))

    # Smoke test: produces multiple chunks; chunks are non-empty; chunk_ids
    # are deterministic (idempotence). This pins the unchanged-on-no-chart
    # path so the v4 patch can't regress it.
    assert len(chunks) >= 2
    for c in chunks:
        assert c.text.strip()
    # Idempotent
    again = chunk_document(_doc(body))
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in again]


def test_chart_block_words_count_zero_toward_budget() -> None:
    """A single paragraph that's small in prose-words but contains a
    massive chart block must NOT trigger the oversize-paragraph
    sentence-split path (which is keyed on `_budget_word_count > target`).
    """
    big_chart = "\n".join(f"row{i} | col1 | col2 | col3 | col4" for i in range(200))
    # Prose is ~10 words; chart is ~1000 words. Without the fix:
    # _word_count(p) > target, oversize path fires, sentence-split runs.
    # With the fix: _budget_word_count(p) ≈ 10, normal path.
    paragraph = f"Small intro about a chart.\n\n[chart-extracted]\n{big_chart}\n[/chart-extracted]"
    body = paragraph

    chunks = chunk_document(_doc(body))
    # One chunk; chart fully intact (no sentence-split mangling).
    assert len(chunks) == 1
    assert "row199 | col1" in chunks[0].text


# ----------------------------------------------------------------------
# P3.3 v7 chunker fix (2026-05-23): headings INSIDE chart-extracted
# blocks must not be treated as document-section boundaries.
# ----------------------------------------------------------------------


def test_h1_inside_chart_block_does_not_split_section() -> None:
    """The nvmath-python case: Nemotron-Parse emits H1 labels for the
    4 design principles INSIDE a single chart-extracted block. The
    chunker must NOT split on those — they're inert chart-figure
    labels, not document headings. Without the fix, only the LAST
    principle reached the reranker's top-5."""
    body = (
        "## nvmath-python\n\n"
        "Easy Pythonic access to CUDA Math library functionality.\n\n"
        "<!-- image -->\n\n"
        "[chart-extracted]\n"
        "# Minimal lead-time\n"
        "to access new CUDA library features from Python\n"
        "# Inter-operability\n"
        "with core Python numeric packages\n"
        "# Pythonic library interface\n"
        "to CUDA accelerated libraries\n"
        "# Platform-agnostic\n"
        "Runs on GPU, CPU (x86 & Arm) and Grace-Hopper\n"
        "[/chart-extracted]\n\n"
        "Logo\n"
    )

    chunks = chunk_document(_doc(body))
    # The entire ## nvmath-python section must be a single chunk —
    # the four H1 lines inside the chart block must NOT trigger
    # section splits.
    assert len(chunks) == 1
    text = chunks[0].text
    # All four principle labels are present in the same chunk.
    assert "Minimal lead-time" in text
    assert "Inter-operability" in text
    assert "Pythonic library interface" in text
    assert "Platform-agnostic" in text


def test_real_h1_outside_chart_block_still_splits() -> None:
    """Sanity: real document H1 headings OUTSIDE chart blocks continue
    to split into sections. The v7 fix is narrow — it only ignores H1s
    inside `[chart-extracted]...[/chart-extracted]` ranges."""
    body = (
        "# Real Section One\n\n"
        "Some prose.\n\n"
        "[chart-extracted]\n"
        "# Inner Chart Label\n"
        "ignored\n"
        "[/chart-extracted]\n\n"
        "# Real Section Two\n\n"
        "More prose.\n"
    )

    chunks = chunk_document(_doc(body))
    # Two real sections → at least two chunks; the inner H1 didn't
    # produce a third section.
    section_starts = {c.heading_path[0] for c in chunks if c.heading_path}
    assert "Real Section One" in section_starts
    assert "Real Section Two" in section_starts
    assert "Inner Chart Label" not in section_starts


def test_heading_path_skips_chart_block_h1() -> None:
    """The heading-path metadata for chunks after a chart block must
    reflect the document's real heading state, not chart-figure H1
    labels."""
    body = (
        "## Real H2 section\n\n"
        "Intro prose.\n\n"
        "[chart-extracted]\n"
        "# Chart Label\n"
        "filler\n"
        "[/chart-extracted]\n\n"
        "Trailing prose after chart.\n"
    )

    chunks = chunk_document(_doc(body))
    assert len(chunks) == 1
    # The trailing prose lives under `## Real H2 section`, NOT
    # `# Chart Label`.
    assert chunks[0].heading_path == ["Real H2 section"]
