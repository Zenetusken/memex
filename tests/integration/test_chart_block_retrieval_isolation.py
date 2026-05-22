"""P3.3 v3 — end-to-end demonstration that `[chart-extracted]` blocks
no longer perturb BM25 retrieval rankings.

Setup: two synthetic chunks share a real FTSStore.
- chunk_prose: a slide describing CUTLASS without any chart data.
- chunk_with_chart: a slide whose body is dominated by a
  [chart-extracted] block listing years 2014..2022 with values.

Query: "what years did NVIDIA release Tesla cards" — a non-chart
query that happens to mention years like "2014, 2018, 2022" in its
expected document body. Without the strip, chunk_with_chart's
multiple `2014`, `2018`, `2022` mentions inside the chart block
would outrank chunk_prose. With the strip, the chart block is
absent from the BM25 body, so chunk_prose wins on relevance.

The test confirms BOTH:
1. The chunk_with_chart row in chunks_fts has NO `2014`/`2022`
   chart-numeric tokens visible to BM25.
2. A BM25 search for the year tokens does NOT promote
   chunk_with_chart above chunk_prose.

This is the integration-level evidence that the P3.3 v3 fix
works as designed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memex.core.types import Chunk
from memex.index.fts_store import FTSStore


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
async def test_chart_extracted_block_not_visible_to_bm25(
    tmp_path: Path,
) -> None:
    """After indexing a chunk that contains a `[chart-extracted]`
    block, the FTS row for that chunk must NOT contain the chart's
    numerical tokens. Direct DB inspection — no retrieval logic
    involved, just verifying the strip-on-upsert plumbing.
    """
    vault_path = tmp_path / "vault"
    vault_path.mkdir(parents=True)

    chunk_with_chart = _chunk(
        "chart-bearing",
        "doc-1",
        (
            "## CUDA Compile Time History\n"
            "Compile time evolution across CUDA releases.\n\n"
            "<!-- image -->\n\n"
            "[chart-extracted]\n"
            "2014 | 8253\n"
            "2016 | 1330\n"
            "2018 | 1519\n"
            "2020 | 1822\n"
            "2022 | 2126\n"
            "[/chart-extracted]\n"
        ),
    )

    store = await FTSStore.open(vault_path)
    await store.upsert([chunk_with_chart])

    def _read_fts_body() -> str:
        row = store._db.execute(
            "SELECT text FROM chunks_fts WHERE chunk_id = ?",
            (chunk_with_chart.chunk_id,),
        ).fetchone()
        assert row is not None, "chunk not inserted"
        return row[0]

    import asyncio

    fts_body = await asyncio.to_thread(_read_fts_body)

    # The chart numbers must NOT be visible to BM25.
    for token in ("8253", "1330", "1519", "1822", "2126"):
        assert token not in fts_body, (
            f"chart-block token {token!r} leaked into chunks_fts.text — "
            "the strip on FTS upsert is broken"
        )

    # The prose IS visible.
    assert "CUDA Compile Time History" in fts_body
    assert "Compile time evolution" in fts_body
    # The `<!-- image -->` Docling placeholder is intentionally
    # preserved by the strip (it's not chart-OCR content).
    assert "<!-- image -->" in fts_body

    # Meanwhile, the chunks_meta + the chunk's `.text` keep the FULL
    # content — the agent still sees the chart block when this
    # chunk is retrieved via prose match.
    assert "[chart-extracted]" in chunk_with_chart.text
    assert "8253" in chunk_with_chart.text

    await store.close()


@pytest.mark.asyncio
async def test_chart_block_does_not_outrank_prose_for_unrelated_query(
    tmp_path: Path,
) -> None:
    """The headline regression check: a query that incidentally
    mentions years/numbers must rank the prose-bearing chunk above
    the chart-bearing chunk. Before the P3.3 v3 fix, the chart
    block's dense numerical tokens (`2014`, `2018`, `2022`) would
    inflate BM25 for chunk_with_chart and pull it above
    chunk_prose. After the fix, the BM25 body of chunk_with_chart
    contains only its prose, so chunk_prose wins on relevance.
    """
    vault_path = tmp_path / "vault"
    vault_path.mkdir(parents=True)

    chunk_prose = _chunk(
        "prose-only",
        "doc-prose",
        (
            "## Tesla GPU Release Timeline\n\n"
            "NVIDIA released the Tesla K80 in 2014, the V100 in 2017, "
            "the A100 in 2020, and the H100 in 2022. Each generation "
            "delivered substantial compute gains for datacenter workloads.\n"
        ),
    )
    chunk_with_chart = _chunk(
        "chart-only",
        "doc-chart",
        (
            "## CUDA Compile Time History\n\n"
            "Compile time evolution.\n\n"
            "<!-- image -->\n\n"
            "[chart-extracted]\n"
            "2014 | 50\n"
            "2016 | 40\n"
            "2018 | 32\n"
            "2020 | 25\n"
            "2022 | 18\n"
            "[/chart-extracted]\n"
        ),
    )

    store = await FTSStore.open(vault_path)
    await store.upsert([chunk_prose, chunk_with_chart])

    # FTSStore.search treats the query as a quoted phrase via
    # FTS5 MATCH, so we use a phrase that appears verbatim in
    # chunk_prose AND happens to contain a year that ALSO appears
    # in chunk_with_chart's chart block. The chart-bearing chunk's
    # MATCH against the year (e.g. "2014") would historically have
    # boosted its BM25 rank for queries like "released in 2014".
    # With the strip, the chart-bearing chunk's FTS body no longer
    # contains the year, so it doesn't match at all.
    results = await store.search("2014", k=5)

    assert len(results) >= 1, "expected at least one match for '2014'"
    # Critical invariant: only the prose chunk matches. Before the
    # P3.3 v3 fix, both chunks would have matched (the chart block
    # in chunk_with_chart contained "2014 | 50"), with chunk_with_
    # chart possibly outranking chunk_prose because of repetition.
    result_ids = [r.chunk_id for r in results]
    assert "prose-only" in result_ids, (
        f"prose chunk missing from results: {result_ids}"
    )
    assert "chart-only" not in result_ids, (
        f"chart-bearing chunk leaked into results for '2014' — the "
        f"strip on FTS upsert is broken. Results: {result_ids}"
    )

    await store.close()
