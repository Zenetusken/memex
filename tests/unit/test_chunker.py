"""Chunker unit tests — runs without any models or external services."""

from __future__ import annotations

from pathlib import Path

import pytest

from memex.index.chunker import (
    OVERLAP_TOKENS,
    TARGET_TOKENS,
    _page_for_offset,
    chunk_document,
    page_intervals,
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


_GFM_HEADER = "| Line item | FY25 | FY24 |\n|---|---|---|"


def _rows_portion(text: str) -> str:
    """Strip a leading synthetic GFM header from a table chunk's text.

    Header-repeated chunks (every table sub-chunk after the first) carry the
    synthetic `header + "\\n"` that the chunker prepended — that prefix is NOT
    in the source, so the offset round-trip is to the ROWS portion only (the
    chunk text minus any leading repeated header). See the table-chunking spec
    §4 "Offset handling".
    """
    prefix = _GFM_HEADER + "\n"
    return text[len(prefix) :] if text.startswith(prefix) else text


def test_force_split_offsets_roundtrip_exactly() -> None:
    body = f"## Financial Statements\n\n{_big_table(400)}\n"
    doc = _doc(body)
    chunks = chunk_document(doc)
    table_chunks = [c for c in chunks if c.text.lstrip().startswith("|")]
    assert table_chunks
    for c in table_chunks:
        # Round-trip holds for the ROWS portion. The first table chunk's text
        # is wholly contiguous (its header is the table's real first line);
        # later chunks carry a synthetic repeated header that is the only text
        # not in [char_start, char_end). Accept either: body slice == full text
        # (group 0) OR == rows-only (header-repeated groups).
        slice_ = doc.body[c.char_start : c.char_end]
        assert slice_ == c.text or slice_ == _rows_portion(c.text)
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


# ----- Char-aware split + GFM table-header repetition (table-chunking spec) -----

from memex.index.chunker import MAX_CHUNK_CHARS, _force_split_oversized, _gfm_header  # noqa: E402

# The header _big_table emits — reused to detect/strip synthetic headers.
_TABLE_HEADER = "| Line item | FY25 | FY24 |"
_TABLE_DELIM = "|---|---|---|"


def _char_heavy_table(n_rows: int) -> str:
    """A char-heavy / word-light GFM table — the exact 10-K failure shape.

    Each row is long in characters (a long no-space identifier + two long
    no-space numbers → only ~4 whitespace "words") but light in budget-words,
    so the word cap (`MAX_CHUNK_MULTIPLIER`) never fires; only the char cap
    does. The whole table is one paragraph (no blank lines), so the sentence-
    splitter can't break it — it lands in `_force_split_oversized` whole.
    """
    header = f"{_TABLE_HEADER}\n{_TABLE_DELIM}"
    rows = "\n".join(
        # No internal spaces in the cells → each row is ~4 words but ~70 chars.
        f"| Segment_line_item_identifier_{i:04d} | {i:08d}00000 | {i:08d}99999 |"
        for i in range(n_rows)
    )
    return header + "\n" + rows


def test_char_heavy_word_light_table_splits_and_is_char_bounded() -> None:
    # ~big table that is well under the word cap but way over the char cap
    # (the exact 10-K failure: char-heavy, word-light → word cap never fires).
    table = _char_heavy_table(150)
    assert _budget_word_count(table) <= _CAP, _budget_word_count(table)  # word cap NOT fired
    assert len(table) > MAX_CHUNK_CHARS  # but the char cap must
    body = f"## Financials\n\n{table}\n"
    chunks = chunk_document(_doc(body))
    table_chunks = [c for c in chunks if c.text.lstrip().startswith("|")]
    assert len(table_chunks) > 1, "char-heavy table must split into multiple chunks"
    # Every chunk is char-bounded (allow one over-long final row of slack) and
    # word-bounded.
    longest_row = max(len(ln) for ln in table.split("\n"))
    for c in table_chunks:
        assert len(c.text) <= MAX_CHUNK_CHARS + longest_row + len(_TABLE_HEADER) + len(_TABLE_DELIM)
        assert _budget_word_count(c.text) <= _CAP


def test_table_header_repeated_on_every_post_first_chunk() -> None:
    table = _char_heavy_table(150)
    body = f"## Financials\n\n{table}\n"
    chunks = chunk_document(_doc(body))
    table_chunks = [c for c in chunks if c.text.lstrip().startswith("|")]
    assert len(table_chunks) > 1
    header_block = f"{_TABLE_HEADER}\n{_TABLE_DELIM}"
    # Every chunk after the first starts with the header + delimiter rows.
    for c in table_chunks[1:]:
        assert c.text.startswith(header_block), f"chunk missing repeated header: {c.text[:80]!r}"
    # The first chunk also starts with the header (it's the table's own first line).
    assert table_chunks[0].text.startswith(header_block)


def test_deep_row_is_colocated_with_the_header() -> None:
    # A row deep in the table must land in a chunk that ALSO carries the header,
    # so a value in that row is interpretable in isolation.
    table = _char_heavy_table(150)
    body = f"## Financials\n\n{table}\n"
    chunks = chunk_document(_doc(body))
    deep_row = "| Segment_line_item_identifier_0120 | 0000012000000 | 0000012099999 |"
    holders = [c for c in chunks if deep_row in c.text]
    assert len(holders) == 1, "deep row must appear in exactly one chunk"
    holder = holders[0]
    assert _TABLE_HEADER in holder.text
    assert _TABLE_DELIM in holder.text


def test_non_gfm_oversized_unit_gets_no_synthetic_header() -> None:
    # An oversized unit with no GFM header (a long pipe-less list/block) is
    # still force-split on lines, but NO synthetic header is prepended.
    lines = "\n".join(
        f"- Bullet item number {i} with some descriptive text here." for i in range(400)
    )
    body = f"## Notes\n\n{lines}\n"
    chunks = chunk_document(_doc(body))
    # It split (char + word bounded) ...
    assert len(chunks) > 1
    for c in chunks:
        assert _budget_word_count(c.text) <= _CAP
        # ... and no chunk grew a fabricated `|---|` delimiter row.
        assert _TABLE_DELIM not in c.text
    # _gfm_header returns None for a non-pipe unit.
    assert _gfm_header(lines) is None


def test_gfm_header_detection() -> None:
    table = "| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |\n| 4 | 5 | 6 |"
    assert _gfm_header(table) == "| A | B | C |\n|---|---|---|"
    # A pipe row with no following delimiter row is not a GFM header.
    assert _gfm_header("| A | B |\n| 1 | 2 |") is None
    # Prose is not a GFM table.
    assert _gfm_header("Just a paragraph.\nAnother line.") is None
    # A single line can't be a GFM table.
    assert _gfm_header("| A | B |") is None


def test_force_split_single_overlong_line_emitted_whole() -> None:
    # A pathological one-line "table" longer than max_chars is emitted whole
    # rather than cut mid-line (degenerate guard).
    one_line = "| " + " | ".join(str(i) for i in range(2000)) + " |"
    assert len(one_line) > MAX_CHUNK_CHARS
    groups = _force_split_oversized(one_line, target_tokens=10)
    assert groups == [one_line]


def test_prose_chunk_ids_identical_across_runs_and_versus_no_table_doc() -> None:
    # Prose path is byte-identical: a normal multi-paragraph prose doc chunks
    # deterministically AND identically to a doc with no table at all (the
    # char/header machinery is a no-op on prose).
    sentence = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
        "eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    )
    body = "\n\n".join(f"## Section {i}\n\n" + (sentence * 50) for i in range(3))
    run_a = chunk_document(_doc(body))
    run_b = chunk_document(_doc(body))
    assert [c.chunk_id for c in run_a] == [c.chunk_id for c in run_b]
    # The prose has no `|---|` rows at all → no chunk should carry a synthetic
    # delimiter row (proves the table machinery never touched it).
    assert all(_TABLE_DELIM not in c.text for c in run_a)


# --- chunk → source-page attribution (drives the webui's click-source UX) ---


def test_page_intervals_cumulates_with_delimiter() -> None:
    """Pages are joined with `\\n\\n` (2 chars) — each next page's interval starts
    at the prior end + 2. Pinned so a refactor of the join-delimiter convention
    can't silently shift the chunker's page mapping."""
    intervals = page_intervals([(1, 100), (2, 50), (3, 80)])
    assert intervals == [(1, 0, 100), (2, 102, 152), (3, 154, 234)]


def test_page_intervals_all_zero_returns_none() -> None:
    """Legacy manifest sentinel — every char_count=0 means the parser didn't
    record per-page sizes, so the chunker MUST treat it as 'no mapping' and
    skip attribution (else it would collapse every chunk to page 1)."""
    assert page_intervals([(1, 0), (2, 0)]) is None
    assert page_intervals([]) is None


def test_page_for_offset_picks_containing_interval() -> None:
    intervals = [(1, 0, 100), (2, 102, 152), (3, 154, 234)]
    assert _page_for_offset(intervals, 0) == 1
    assert _page_for_offset(intervals, 99) == 1
    assert _page_for_offset(intervals, 102) == 2
    assert _page_for_offset(intervals, 234) == 3  # past last end → falls to last page


def test_chunk_document_attributes_pages_when_counts_given() -> None:
    """End-to-end: with per-page char counts threaded in, each chunk gets
    `Chunk.page` set from its `char_start`. Two-paragraph body large enough
    to split into a section and a chunk per page."""
    body = "# H1\n\nFirst page content.\n\n# H2\n\nSecond page content."
    # Pretend the source had two pages of ~25 chars each (the actual values
    # don't matter for the attribution logic — only the cumulative offsets do).
    counts = [(1, 25), (2, len(body) - 27)]  # 27 = first page + the "\n\n" delimiter
    chunks = chunk_document(_doc(body), page_char_counts=counts)
    assert chunks  # non-empty
    pages = [c.page for c in chunks]
    assert all(p is not None for p in pages)
    # The first chunk starts at 0 → page 1; later chunks past the delimiter → page 2.
    assert pages[0] == 1


def test_chunk_document_skips_attribution_on_legacy_zero_counts() -> None:
    """A doc with a manifest predating `PageDecision.char_count` (all zeros)
    must NOT collapse every chunk to page 1 — `Chunk.page` stays `None`,
    the webui falls back to section-only anchors (no regression)."""
    body = "# H\n\nA paragraph that fits in one chunk."
    chunks = chunk_document(_doc(body), page_char_counts=[(1, 0), (2, 0)])
    assert chunks
    assert all(c.page is None for c in chunks)


def test_chunk_document_default_page_none_when_no_counts() -> None:
    """`page_char_counts=None` (the default) keeps the legacy behavior —
    `Chunk.page` is None, byte-identical to pre-attribution chunkers."""
    chunks = chunk_document(_doc("# H\n\nA tiny paragraph."))
    assert all(c.page is None for c in chunks)


def test_chunk_document_uses_exact_page_intervals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Companion arc-3: `exact_page_intervals` (already-measured `(page, start, end)`) attributes
    `Chunk.page` DIRECTLY, with no char_count→interval derivation."""
    body = "# H1\n\nFirst page content.\n\n# H2\n\nSecond page content."
    split = body.index("# H2")
    intervals = [(1, 0, split), (2, split, len(body))]
    chunks = chunk_document(_doc(body), exact_page_intervals=intervals)
    assert chunks
    assert chunks[0].page == 1
    # a chunk whose char_start is past the page-2 boundary is attributed to page 2
    assert {c.page for c in chunks} <= {1, 2}
    assert any(c.page == 2 for c in chunks)


def test_exact_page_intervals_takes_precedence_over_char_counts() -> None:
    """When BOTH are passed, the exact intervals win (the index path always supplies exact when the
    manifest is citation-grade) — proven by giving deliberately WRONG char_counts."""
    body = "# H1\n\nFirst page content.\n\n# H2\n\nSecond page content."
    split = body.index("# H2")
    exact = [(7, 0, split), (8, split, len(body))]  # pages renumbered 7/8 to detect which path ran
    chunks = chunk_document(_doc(body), page_char_counts=[(1, 5), (2, 5)], exact_page_intervals=exact)
    assert chunks[0].page == 7  # the exact intervals' page numbering, not the char_counts'
