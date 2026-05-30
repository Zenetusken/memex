"""Unit tests for Table-RAG Phase 1 — GFM table linearization + the
3-channel contract (see docs/specs/table-rag.md).

Covers:
  - `parse/table_linearize.linearize_gfm_tables` row formats (row-label /
    plain / named-first-column branches), section-heading prefix, footnote +
    whitespace cleanup, idempotence + determinism, header-sanity gate.
  - `core/text` helpers: `looks_like_value`, `table_rows_spans` (with
    orphan tolerance), `strip_superseded_gfm_tables`.
  - The 3-CHANNEL CONTRACT pins (a)-(e): FTS body, separate chunks, prose
    byte-identical, oversized char-split on row boundaries, no chart-exemption.
  - A BM25-pollution regression (a linearized row must not outrank prose on a
    shared common term), mirroring test_chart_block_retrieval_isolation.py.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from memex.core.table_linearize import (
    header_all_value_like,
    header_has_prose_cell,
    linearize_gfm_tables,
    nearest_heading_text,
    parse_gfm_table,
)
from memex.core.text import (
    looks_like_value,
    strip_chart_extracted_for_index,
    strip_superseded_gfm_tables,
    table_rows_spans,
)
from memex.core.types import Chunk
from memex.index.chunker import (
    MAX_CHUNK_CHARS,
    _force_split_oversized,
    chunk_document,
)
from memex.index.fts_store import FTSStore
from memex.vault.store import DocumentRef, Frontmatter, VaultDocument


def _doc(body: str, *, doc_id: str = "abc12345-test", title: str = "T") -> VaultDocument:
    ref = DocumentRef(
        doc_id=doc_id,
        markdown_path=Path(f"/tmp/{doc_id}.md"),  # noqa: S108 — test fixture
        asset_dir=Path(f"/tmp/{doc_id}"),  # noqa: S108
        source_path=None,
        content_sha256="0" * 64,
    )
    return VaultDocument(ref=ref, frontmatter=Frontmatter(title=title), body=body, mtime_ns=0)


def _rows_block(markdown: str) -> str:
    """Extract the text between `[table-rows]` and `[/table-rows]` (exclusive)."""
    start = markdown.index("[table-rows]") + len("[table-rows]\n")
    end = markdown.index("\n[/table-rows]")
    return markdown[start:end]


# ======================================================================
# looks_like_value (promoted from docling_tables._looks_like_value)
# ======================================================================


def test_looks_like_value_digits_dollar_percent() -> None:
    assert looks_like_value("123")
    assert looks_like_value("$22.5 billion")
    assert looks_like_value("12%")
    assert looks_like_value("FY2024")  # has digits


def test_looks_like_value_pure_text_is_false() -> None:
    assert not looks_like_value("Revenue")
    assert not looks_like_value("Compute & Networking")
    assert not looks_like_value("")


# ======================================================================
# linearize_gfm_tables — row formats
# ======================================================================


def test_row_label_column_branch() -> None:
    """Empty corner + textual col-0 → row-label layout:
    `{cell0}: {header[1]}={cell1}, …`."""
    md = "| | 2024 | 2025 |\n|---|---|---|\n| Compute | 10 | 20 |\n| Graphics | 5 | 8 |\n"
    out = _rows_block(linearize_gfm_tables(md))
    assert out == "Compute: 2024=10, 2025=20\nGraphics: 2024=5, 2025=8"


def test_plain_branch_named_first_column() -> None:
    """A named first column (header[0] non-empty) → plain layout:
    `{header[0]}={cell0}, …` — DIFFERENT output from the row-label branch."""
    md = "| Metric | 2024 | 2025 |\n|---|---|---|\n| Compute | 10 | 20 |\n"
    out = _rows_block(linearize_gfm_tables(md))
    assert out == "Metric=Compute, 2024=10, 2025=20"


def test_named_first_column_with_text_rows_is_still_plain() -> None:
    """Even when col-0 data cells are text, a NAMED header[0] forces the plain
    branch (the row-label branch requires an EMPTY corner)."""
    md = "| Segment | 2024 |\n|---|---|\n| Compute | 100 |\n"
    out = _rows_block(linearize_gfm_tables(md))
    assert out == "Segment=Compute, 2024=100"


def test_section_heading_prefix() -> None:
    """Nearest preceding heading is prefixed to each row as `[heading] `."""
    md = "## Revenue by segment\n\n| Metric | 2024 |\n|---|---|\n| Compute | 100 |\n"
    out = _rows_block(linearize_gfm_tables(md))
    assert out == "[Revenue by segment] Metric=Compute, 2024=100"


def test_no_heading_means_no_prefix() -> None:
    md = "| Metric | 2024 |\n|---|---|\n| Compute | 100 |\n"
    out = _rows_block(linearize_gfm_tables(md))
    assert not out.startswith("[")


def test_footnote_and_whitespace_cleanup_values_verbatim() -> None:
    """Trailing footnote markers stripped, whitespace collapsed — but the
    value text is kept verbatim (`$22.5 billion` survives)."""
    md = "| Item | Value |\n|---|---|\n| Revenue(1) | $22.5  billion |\n| Total[2] | $30* |\n"
    out = _rows_block(linearize_gfm_tables(md))
    lines = out.split("\n")
    assert lines[0] == "Item=Revenue, Value=$22.5 billion"  # (1) gone, ws collapsed, value kept
    assert lines[1] == "Item=Total, Value=$30"  # [2] gone, trailing * gone


def test_standalone_accounting_negative_value_survives() -> None:
    """A parenthesized number that IS the whole cell is an accounting-NEGATIVE value,
    not a footnote — keep it verbatim. A trailing `(1)` after a word still strips."""
    md = "| Item | Value |\n|---|---|\n| Net change | (45) |\n| Revenue(1) | $(56) |\n"
    lines = _rows_block(linearize_gfm_tables(md)).split("\n")
    assert lines[0] == "Item=Net change, Value=(45)"  # standalone (45) kept verbatim
    assert lines[1] == "Item=Revenue, Value=$(56)"  # word footnote (1) stripped; $(56) kept


# ======================================================================
# Header-sanity gate
# ======================================================================


def test_header_all_value_like_is_skipped() -> None:
    """A header row that is ALL value-like (digits/$/%) is a mis-structured
    header (e.g. the 10-K segment table) → no `[table-rows]` block emitted."""
    md = "| 2024 | 2025 | $ |\n|---|---|---|\n| 10 | 20 | 30 |\n"
    assert "[table-rows]" not in linearize_gfm_tables(md)


def test_year_header_NOT_skipped() -> None:
    """A year-header table (`| Metric | 2024 | 2025 |`) has a non-value-like
    `Metric` cell → NOT all-value-like → linearized."""
    md = "| Metric | 2024 | 2025 |\n|---|---|---|\n| Compute | 10 | 20 |\n"
    assert "[table-rows]" in linearize_gfm_tables(md)


def test_empty_corner_header_NOT_skipped() -> None:
    """An empty-corner header (`| | 2024 | 2025 |`) has an empty (not
    value-like) col-0 → NOT all-value-like → linearized (row-label layout)."""
    md = "| | 2024 | 2025 |\n|---|---|---|\n| Compute | 10 | 20 |\n"
    assert "[table-rows]" in linearize_gfm_tables(md)


# ======================================================================
# Idempotence + determinism
# ======================================================================


def test_idempotent() -> None:
    """A second pass over already-linearized markdown is a no-op (a table
    already followed by a `[table-rows]` block is skipped)."""
    md = "## S\n\n| Metric | 2024 |\n|---|---|\n| Compute | 100 |\n"
    once = linearize_gfm_tables(md)
    twice = linearize_gfm_tables(once)
    assert once == twice
    assert once.count("[table-rows]") == 1


def test_deterministic_byte_identical() -> None:
    md = "## S\n\n| Metric | 2024 | 2025 |\n|---|---|---|\n| A | 1 | 2 |\n| B | 3 | 4 |\n"
    assert linearize_gfm_tables(md) == linearize_gfm_tables(md)


def test_blank_line_separator_before_block() -> None:
    """The block is preceded by a blank line so it is a DISTINCT paragraph
    unit (load-bearing for the separate-chunks contract)."""
    md = "| Metric | 2024 |\n|---|---|\n| Compute | 100 |\n"
    out = linearize_gfm_tables(md)
    assert "100 |\n\n[table-rows]" in out


def test_no_table_passes_through_unchanged() -> None:
    md = "# Title\n\nJust prose, no tables here.\n\nMore prose.\n"
    assert linearize_gfm_tables(md) == md


# ======================================================================
# table_rows_spans (+ orphan tolerance)
# ======================================================================


def test_table_rows_spans_balanced() -> None:
    text = "before [table-rows]\nA=1\n[/table-rows] after"
    spans = table_rows_spans(text)
    assert len(spans) == 1
    s, e = spans[0]
    assert text[s:e].startswith("[table-rows]")
    assert text[s:e].endswith("[/table-rows]")


def test_table_rows_spans_orphan_opener() -> None:
    """A force-split first half carries only the opener → span to end-of-text."""
    text = "[table-rows]\nA=1\nB=2 (truncated mid-block)"
    spans = table_rows_spans(text)
    assert len(spans) == 1
    s, e = spans[0]
    assert text[s : s + len("[table-rows]")] == "[table-rows]"
    assert e == len(text)


def test_table_rows_spans_orphan_closer() -> None:
    """A force-split later half carries only the closer → span from start."""
    text = "C=3\nD=4\n[/table-rows] trailing"
    spans = table_rows_spans(text)
    assert len(spans) == 1
    s, e = spans[0]
    assert s == 0
    assert text[e - len("[/table-rows]") : e] == "[/table-rows]"


def test_table_rows_spans_none() -> None:
    assert table_rows_spans("no blocks") == []
    assert table_rows_spans("") == []


# ======================================================================
# strip_superseded_gfm_tables
# ======================================================================


def test_strip_superseded_removes_table_keeps_block() -> None:
    md = linearize_gfm_tables("| Metric | 2024 |\n|---|---|\n| Compute | 100 |\n")
    stripped = strip_superseded_gfm_tables(md)
    assert "|---|" not in stripped  # raw GFM table gone
    assert "[table-rows]" in stripped  # KV block kept
    assert "Metric=Compute, 2024=100" in stripped


def test_strip_leaves_unlinearized_table_intact() -> None:
    """A GFM table with NO following `[table-rows]` block keeps its raw form
    (its value tokens still reach BM25)."""
    md = "| Metric | 2024 |\n|---|---|\n| Compute | 100 |\n"
    assert strip_superseded_gfm_tables(md) == md


def test_strip_idempotent() -> None:
    md = linearize_gfm_tables("| Metric | 2024 |\n|---|---|\n| Compute | 100 |\n")
    once = strip_superseded_gfm_tables(md)
    assert strip_superseded_gfm_tables(once) == once


# ======================================================================
# 3-CHANNEL CONTRACT — (b) separate chunks
# ======================================================================


def _big_linearized_table(n_rows: int = 80) -> str:
    """A doc whose GFM table is oversized (char cap) — the motivating 10-K
    case — so the raw table force-splits and the `[table-rows]` block is its
    own distinct chunkable unit."""
    header = "| Line item | FY24 | FY25 |\n|---|---|---|"
    rows = "\n".join(
        f"| Revenue stream number {i} from continuing operations | {i}000000 | {i}999999 |"
        for i in range(n_rows)
    )
    return linearize_gfm_tables(f"## Financial Statements\n\n{header}\n{rows}\n")


def test_finalize_body_keeps_the_vault_md_clean() -> None:
    """audit-10 W1: the canonical `.md` is content-only — `_finalize_body` does NOT
    inject `[table-rows]` (the linearization moved to index time). The step-3 heading
    normalizer leaves an already-rooted tree (H1 title + H2 section) untouched."""
    from memex.parse.pipeline import _finalize_body

    body = "# Annual Report\n\n## Revenue\n\n| Metric | 2024 |\n|---|---|\n| Compute | 100 |\n"
    out = _finalize_body(body)
    assert "[table-rows]" not in out
    assert out == body  # pass-through (clean content, already-rooted headings)


def test_index_rederivation_is_retrieval_neutral() -> None:
    """audit-10 W1: re-deriving the linearization at INDEX time from the CLEAN vault
    body reproduces exactly the body the chunker used to see when `[table-rows]` lived
    in the `.md` — so chunk_ids (and embeddings) are unchanged. Idempotent, so a
    not-yet-re-parsed (still-polluted) `.md` re-derives to the same bytes too."""
    clean = "## Revenue\n\n| Metric | 2024 | 2025 |\n|---|---|---|\n| Compute | 10 | 20 |\n"
    assert "[table-rows]" not in clean
    indexed = linearize_gfm_tables(clean)  # exactly what index/pipeline.py now feeds the chunker
    assert "[table-rows]" in indexed  # retrieval still co-locates each cell with its label
    assert linearize_gfm_tables(indexed) == indexed  # idempotent → stable across migration


def test_contract_b_raw_table_and_block_in_separate_chunks() -> None:
    """(b) No chunk contains BOTH the raw GFM table AND the `[table-rows]`
    block — they never co-occur (no within-chunk dilution)."""
    chunks = chunk_document(_doc(_big_linearized_table()))
    mixed = [c for c in chunks if "|---|" in c.text and "[table-rows]" in c.text]
    assert not mixed, f"raw table + KV block concatenated in {len(mixed)} chunk(s)"
    # And both representations DO appear (somewhere, separately).
    assert any("|---|" in c.text for c in chunks)
    assert any("[table-rows]" in c.text for c in chunks)


def test_contract_b_small_table_colocates_in_one_chunk() -> None:
    """(b) DOCUMENTED deviation: a SMALL table (under the char cap) co-locates
    the raw GFM table + the `[table-rows]` block in ONE chunk. This is
    acceptable — a small table isn't truncation-diluted, the KV rows are
    additive co-located signal, and the FTS channel still strips the raw table
    (BM25 sees only KV). Pinned so the small-table behavior is explicit, not
    silently untested. (Per the spec's 3-channel contract, dense row.)"""
    body = (
        "## Reportable Segments\n\n"
        "| | Compute & Networking | Graphics | Total |\n|---|---|---|---|\n"
        "| Revenue | $193.5B | $22.5B | $215.9B |\n"
        "| Operating Income | $130.1B | $9.2B | $139.3B |\n"
    )
    chunks = chunk_document(_doc(linearize_gfm_tables(body)))
    colocated = [c for c in chunks if "|---|" in c.text and "[table-rows]" in c.text]
    assert colocated, "small table expected to co-locate raw + KV in one chunk"
    # The KV rows still carry the answerable column=value co-location.
    assert any("Graphics=$22.5B" in c.text for c in chunks)


# ======================================================================
# 3-CHANNEL CONTRACT — (c) prose chunking byte-identical
# ======================================================================


def test_contract_c_prose_chunking_byte_identical() -> None:
    """(c) A prose doc with NO table chunks identically before/after the
    linearizer is in the pipeline — the linearizer is a no-op on prose, so
    chunk_ids are unchanged (chunk_id stability)."""
    sentence = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
        "eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    )
    body = "\n\n".join(f"## Section {i}\n\n" + (sentence * 50) for i in range(3))
    # The linearizer must not touch prose.
    assert linearize_gfm_tables(body) == body
    before = chunk_document(_doc(body))
    after = chunk_document(_doc(linearize_gfm_tables(body)))
    assert [c.chunk_id for c in before] == [c.chunk_id for c in after]
    assert [c.text for c in before] == [c.text for c in after]


# ======================================================================
# 3-CHANNEL CONTRACT — (d) oversized [table-rows] char-splits on row
# boundaries + table_rows_spans tolerates orphaned halves
# ======================================================================


def _big_rows_block(n_rows: int = 120) -> str:
    rows = "\n".join(
        f"Metric=Item number {i}, FY24={i}00000000000000000, FY25={i}99999999999999999"
        for i in range(n_rows)
    )
    return "[table-rows]\n" + rows + "\n[/table-rows]"


def test_contract_d_oversized_block_char_splits_on_row_boundaries() -> None:
    """(d) An oversized `[table-rows]` block char-splits with each KV row
    WHOLE (never cut mid-row)."""
    block = _big_rows_block()
    assert len(block) > MAX_CHUNK_CHARS
    groups = _force_split_oversized(block, target_tokens=10)
    assert len(groups) > 1
    # Reassemble and confirm every Metric= line is intact (has its commas).
    for g in groups:
        for ln in g.split("\n"):
            if ln.startswith("Metric="):
                assert ln.count(", ") == 2, f"row cut mid-line: {ln!r}"


def test_contract_d_table_rows_spans_tolerates_split_halves() -> None:
    """(d) After a force-split, the first half is an orphan-opener and a later
    half an orphan-closer — `table_rows_spans` still recognises each half."""
    block = _big_rows_block()
    groups = _force_split_oversized(block, target_tokens=10)
    first, last = groups[0], groups[-1]
    assert "[table-rows]" in first and "[/table-rows]" not in first
    assert "[/table-rows]" in last and "[table-rows]" not in last
    # First half: opener with no closer → one span to end-of-text.
    fs = table_rows_spans(first)
    assert len(fs) == 1 and fs[0][1] == len(first)
    # Last half: closer with no opener → one span from start-of-text.
    ls = table_rows_spans(last)
    assert len(ls) == 1 and ls[0][0] == 0


# ======================================================================
# 3-CHANNEL CONTRACT — (e) [table-rows] does NOT get the chart-exemption
# ======================================================================


def test_contract_e_table_rows_block_is_not_chart_exempt() -> None:
    """(e) Unlike a `[chart-extracted]` block (kept whole as one chunk), an
    oversized `[table-rows]` block CHAR-SPLITS into multiple chunks."""
    body = f"## Financials\n\n{_big_rows_block(120)}\n"
    chunks = chunk_document(_doc(body))
    rows_chunks = [c for c in chunks if "Metric=Item number" in c.text]
    assert len(rows_chunks) > 1, (
        "[table-rows] must char-split (it is NOT given the chart-block exemption)"
    )
    # Contrast: a chart-extracted block of comparable size stays ONE chunk.
    chart = "[chart-extracted]\n" + ("data point " * 2000) + "\n[/chart-extracted]"
    chart_chunks = chunk_document(_doc(f"## Figure\n\n{chart}\n"))
    assert sum("[chart-extracted]" in c.text for c in chart_chunks) == 1


# ======================================================================
# 3-CHANNEL CONTRACT — (a) FTS body = KV rows, raw table stripped
# ======================================================================


@pytest.mark.asyncio
async def test_contract_a_fts_body_has_kv_rows_not_raw_table(tmp_path: Path) -> None:
    """(a) After upsert, a table chunk's FTS body contains the KV rows and the
    raw GFM table is STRIPPED (via strip_superseded_gfm_tables in upsert)."""
    vault_path = tmp_path / "vault"
    vault_path.mkdir(parents=True)

    body = linearize_gfm_tables(
        "## Segment revenue\n\n"
        "| Segment | FY24 |\n|---|---|\n"
        "| Datacenter | 47525 |\n| Gaming | 10447 |\n"
    )
    chunk = Chunk(
        chunk_id="tbl-1",
        document_id="doc-1",
        document_title="Doc 1",
        text=body,
        char_start=0,
        char_end=len(body),
    )
    store = await FTSStore.open(vault_path)
    await store.upsert([chunk])

    def _fts_body() -> str:
        row = store._db.execute(
            "SELECT text FROM chunks_fts WHERE chunk_id = ?", (chunk.chunk_id,)
        ).fetchone()
        assert row is not None
        return row[0]

    fts = await asyncio.to_thread(_fts_body)
    # KV rows present.
    assert "Segment=Datacenter, FY24=47525" in fts
    # Raw GFM delimiter row gone (table superseded).
    assert "|---|" not in fts
    # The stored chunk text keeps BOTH (source of truth untouched).
    assert "|---|" in chunk.text and "[table-rows]" in chunk.text
    await store.close()


# ======================================================================
# BM25-pollution regression (mirror test_chart_block_retrieval_isolation)
# ======================================================================


def _chunk(chunk_id: str, document_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title=f"Doc {document_id}",
        text=text,
        char_start=0,
        char_end=len(text),
    )


@pytest.mark.asyncio
async def test_linearized_row_does_not_outrank_prose_on_shared_term(tmp_path: Path) -> None:
    """A linearized KV row that incidentally mentions a common term must not
    outrank a genuinely-relevant prose chunk on that term — and the SUPERSEDED
    raw table must contribute nothing to BM25 (only the KV rows do).

    Mirrors test_chart_block_retrieval_isolation: the table chunk's raw GFM
    table is stripped, so the term only appears ONCE (in the KV row), not
    doubled (raw table + KV) — which is what would inflate BM25 rank.
    """
    vault_path = tmp_path / "vault"
    vault_path.mkdir(parents=True)

    # Prose chunk genuinely about "revenue": the term appears in real sentences.
    prose = _chunk(
        "prose-only",
        "doc-prose",
        (
            "## Revenue discussion\n\n"
            "Total revenue grew this year as revenue from datacenter products "
            "drove the bulk of the revenue increase. Revenue guidance remains "
            "strong heading into next year.\n"
        ),
    )
    # Table chunk: a linearized table whose KV rows mention "revenue" once each.
    table_body = linearize_gfm_tables(
        "## Segment table\n\n"
        "| Revenue segment | FY24 |\n|---|---|\n"
        "| Datacenter | 47525 |\n| Gaming | 10447 |\n"
    )
    table = _chunk("table-chunk", "doc-table", table_body)

    store = await FTSStore.open(vault_path)
    await store.upsert([prose, table])

    # 1) The raw GFM delimiter must not be in the table chunk's FTS body.
    def _fts_body(cid: str) -> str:
        row = store._db.execute("SELECT text FROM chunks_fts WHERE chunk_id = ?", (cid,)).fetchone()
        assert row is not None
        return row[0]

    table_fts = await asyncio.to_thread(_fts_body, "table-chunk")
    assert "|---|" not in table_fts, "raw table leaked into FTS body — strip broken"
    assert "Revenue segment=Datacenter" in table_fts  # KV row present

    # 2) The prose chunk must outrank the table chunk for "revenue".
    results = await store.search("revenue", k=5)
    ids = [r.chunk_id for r in results]
    assert "prose-only" in ids, f"prose chunk missing: {ids}"
    if "table-chunk" in ids:
        # If both match, the prose (term-dense, on-topic) must rank first.
        assert ids.index("prose-only") < ids.index("table-chunk"), (
            f"linearized table row outranked relevant prose: {ids}"
        )
    await store.close()


# ======================================================================
# Composition order in fts_store: both strips apply
# ======================================================================


def test_chart_and_table_strips_compose() -> None:
    """The fts strip pipeline drops the chart-extracted block AND the
    superseded raw table, keeping prose + KV rows."""
    table = linearize_gfm_tables("| Metric | 2024 |\n|---|---|\n| Compute | 100 |\n")
    full = (
        "Prose lead-in.\n\n"
        "<!-- image -->\n\n[chart-extracted]\nyear 2014 value 50\n[/chart-extracted]\n\n" + table
    )
    stripped = strip_superseded_gfm_tables(strip_chart_extracted_for_index(full))
    assert "[chart-extracted]" not in stripped
    assert "|---|" not in stripped
    assert "[table-rows]" in stripped
    assert "Prose lead-in." in stripped


def test_misbounded_prose_header_is_skipped() -> None:
    """The real 10-K failure: Docling MIS-BOUNDED the segment table, pulling the
    section heading + intro SENTENCE into the header row. The header is text (so
    the all-value-like gate misses it), but linearizing it maps the wrong label
    to each value (`Compute & Networking=$22.5B` — actually Graphics). The
    prose-cell gate must skip it so no WRONG KV rows (hallucination risk) ship."""
    body = (
        "## Reportable Segments\n\n"
        "| Fiscal 2026 Reportable Segments | Our two reportable segments are "
        "'Compute & Networking' and 'Graphics': | Compute & Networking | Graphics Total |\n"
        "|---|---|---|---|\n"
        "| Revenue | $193.5B | $22.5B | $215.9B |\n"
        "| Operating Income | $130.1B | $9.2B | $139.3B |\n"
    )
    out = linearize_gfm_tables(body)
    assert "[table-rows]" not in out, "mis-bounded prose header must be skipped"
    # the wrong mapping must NOT appear anywhere
    assert "Compute & Networking=$22.5B" not in out


# ======================================================================
# parse_gfm_table factor (Phase 2) — byte-identity pin + skip conditions
# ======================================================================

# Golden fixtures spanning the row-label / plain / heading / footnote / gate
# branches. The byte-identity pin re-runs the SHIPPED `linearize_gfm_tables`
# over each (the refactor to call `parse_gfm_table` must not change a byte).
_GOLDEN_FIXTURES = (
    "| | 2024 | 2025 |\n|---|---|---|\n| Compute | 10 | 20 |\n| Graphics | 5 | 8 |\n",
    "| Metric | 2024 | 2025 |\n|---|---|---|\n| Compute | 10 | 20 |\n",
    "## Revenue by segment\n\n| Metric | 2024 |\n|---|---|\n| Compute | 100 |\n",
    "| Item | Value |\n|---|---|\n| Revenue(1) | $22.5  billion |\n| Total[2] | $30* |\n",
    "| 2024 | 2025 | $ |\n|---|---|---|\n| 10 | 20 | 30 |\n",  # all-value-like → skipped
    "# Title\n\nJust prose, no tables here.\n\nMore prose.\n",  # no table
)


@pytest.mark.parametrize("md", _GOLDEN_FIXTURES)
def test_parse_factor_linearize_byte_identical(md: str) -> None:
    """The factor must not change `linearize_gfm_tables` output by a byte.

    A pin against itself can't catch a regression introduced in the same
    commit, so we also assert the EXACT expected bytes for the representative
    fixtures (the row-label + plain + heading branches)."""
    out = linearize_gfm_tables(md)
    # Idempotent + deterministic (re-run is byte-identical).
    assert linearize_gfm_tables(out) == linearize_gfm_tables(out)
    assert linearize_gfm_tables(md) == out


def test_parse_factor_exact_golden_bytes() -> None:
    """Exact expected bytes for the three load-bearing branches."""
    row_label = (
        "| | 2024 | 2025 |\n|---|---|---|\n| Compute | 10 | 20 |\n\n"
        "[table-rows]\nCompute: 2024=10, 2025=20\n[/table-rows]"
    )
    assert linearize_gfm_tables(
        "| | 2024 | 2025 |\n|---|---|---|\n| Compute | 10 | 20 |\n"
    ).startswith(row_label)
    plain = (
        "| Metric | 2024 |\n|---|---|\n| Compute | 100 |\n\n"
        "[table-rows]\nMetric=Compute, 2024=100\n[/table-rows]"
    )
    assert linearize_gfm_tables("| Metric | 2024 |\n|---|---|\n| Compute | 100 |\n").startswith(
        plain
    )


def test_parse_gfm_table_returns_header_and_rows() -> None:
    parsed = parse_gfm_table("| Metric | 2024 |\n|---|---|\n| Compute | 100 |\n")
    assert parsed is not None
    header, rows = parsed
    assert header == ["Metric", "2024"]
    assert rows == [["Compute", "100"]]


def test_parse_gfm_table_does_not_apply_header_gate() -> None:
    """parse_gfm_table returns the raw parse even for an all-value-like header
    — the header-sanity gate is the caller's responsibility (so both the
    linearizer and the table store apply it identically)."""
    parsed = parse_gfm_table("| 2024 | 2025 |\n|---|---|\n| 10 | 20 |\n")
    assert parsed is not None
    header, _rows = parsed
    assert header == ["2024", "2025"]
    assert header_all_value_like(header)  # the gate WOULD skip it


def test_parse_gfm_table_skip_conditions() -> None:
    # < 3 lines.
    assert parse_gfm_table("| Metric | 2024 |\n|---|---|\n") is None
    # Second line not a delimiter.
    assert parse_gfm_table("| a | b |\n| c | d |\n| e | f |\n") is None
    # No non-empty data rows.
    assert parse_gfm_table("| a | b |\n|---|---|\n| | |\n") is None


def test_header_gate_public_wrappers_match_behavior() -> None:
    assert header_all_value_like(["2024", "$", "%"])
    assert not header_all_value_like(["Metric", "2024"])
    assert header_has_prose_cell(["Our two reportable segments are these things:"])
    assert not header_has_prose_cell(["Metric", "2024"])


def test_nearest_heading_text() -> None:
    body = "## Revenue\n\n| Metric | 2024 |\n|---|---|\n| Compute | 100 |\n"
    table_start = body.index("| Metric")
    assert nearest_heading_text(body, table_start) == "Revenue"
    assert nearest_heading_text("| Metric | 2024 |\n", 0) == ""
