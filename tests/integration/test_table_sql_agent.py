"""Integration test for the `query_tables` agent node (Table-RAG Phase 2).

Drives the full answering graph with all heavy I/O faked (vLLM, retrieve,
rerank, Langfuse). Validates:
  - the gate fires only on table-relevant aggregation-shaped queries;
  - the node returns the FULL `reranked + [synthetic]` list (not just the
    synthetic), with a non-colliding synthetic chunk_id;
  - the existing verify/compose machinery grounds + cites the synthetic chunk;
  - HARD GATE: a counterfactual aggregate over an absent column → SQL gen
    empty / recompute-disagree → no injection → refuse;
  - HARD GATE: a confident-wrong aggregate → recompute disagrees → no
    injection → refuse.

`memex.agents.table_sql.complete_structured` is patched with a `**_kw: object`
fake; `MemexSettings()` + `set_settings()` are constructed directly (tests
don't bootstrap).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from memex.agents.answering import (
    AnswerState,
    Chunk,
    CitedClaim,
    DraftAnswer,
    RelevanceAssessment,
    SufficiencyAssessment,
    VerificationResult,
    _build_synthetic_chunk,
    answer_query,
    query_tables,
    reset_compiled_graph,
)
from memex.core.config import MemexSettings, set_settings
from memex.core.types import GeneratedSQL, StoredTable, TableQueryResult
from memex.index.table_store import TableStore


@pytest.fixture(autouse=True)
def _fresh_graph() -> Iterator[None]:
    reset_compiled_graph()
    yield
    reset_compiled_graph()


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemexSettings]:
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(vault))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    s = MemexSettings()  # type: ignore[call-arg]
    set_settings(s)
    yield s
    set_settings(None)


def _segments_table() -> StoredTable:
    return StoredTable(
        doc_id="doc-1",
        table_id="abc1234567",
        section="Reportable Segments",
        header=["Segment", "Revenue"],
        rows=[
            ["Compute & Networking", "$116,193"],
            ["Graphics", "$17,109"],
        ],
        char_start=10,
        char_end=120,
    )


async def _seed_tables(vault: Path, tables: list[StoredTable]) -> None:
    store = await TableStore.open(vault)
    try:
        await store.upsert_document("doc-1", tables)
    finally:
        await store.close()


def _table_chunk() -> Chunk:
    return Chunk(
        chunk_id="doc-1#aaaaaaaaaa",
        document_id="doc-1",
        document_title="NVIDIA 10-K",
        text="## Reportable Segments\n\n[table-rows]\nSegment=Compute & Networking, Revenue=$116,193\n[/table-rows]",
        char_start=10,
        char_end=120,
        score=0.5,
    )


# ======================================================================
# Synthetic-chunk text bound (the whole text ≤ _SYNTHETIC_TEXT_MAX)
# ======================================================================


def test_synthetic_chunk_total_text_bounded_with_long_sql() -> None:
    """A pathologically long SQL line must NOT push the whole chunk text past
    900 chars — the evidence (contributing rows) leads and always survives;
    the [sql] tail is truncated/dropped to fit the budget."""
    long_sql = "SELECT SUM(revenue__num) FROM tbl WHERE " + " AND ".join(
        f"col{i} = 'value{i}'" for i in range(120)
    )
    assert len(long_sql) > 900  # genuinely oversized
    result = TableQueryResult(
        kind="aggregate",
        sql=long_sql,
        target_table_id="abc1234567",
        rows=None,
        aggregate_value=133302.0,
        contributing_rows=[
            ["Compute & Networking", "$116,193"],
            ["Graphics", "$17,109"],
        ],
        header=["Segment", "Revenue"],
        char_start=10,
        char_end=120,
        doc_id="doc-1",
        document_title="NVIDIA 10-K",
        heading_path=["Reportable Segments"],
        section="Reportable Segments",
    )
    chunk = _build_synthetic_chunk(result)
    # The WHOLE text is bounded.
    assert len(chunk.text) <= 900
    # Evidence survived (rows are present even though SQL is huge).
    assert "Compute & Networking" in chunk.text
    assert "Graphics" in chunk.text
    # The aggregate self-describes by its column header (2026-05-31) so the answer
    # node connects the scalar to the queried quantity.
    assert "SUM of Revenue = 133302" in chunk.text


# ======================================================================
# FIX 4 — superlative row never dropped (value preserved, truncated)
# ======================================================================


def test_superlative_oversized_row_keeps_value_and_framing() -> None:
    """A superlative `kind="rows"` result whose single row's KV rendering
    exceeds the 900-char budget must STILL surface the row's value (truncated to
    fit) AND the extremum framing — never a bare framing with no value (which
    would be an unsupported claim)."""
    wide_cells = ["Compute & Networking", "$116,193", "x" * 2000]
    result = TableQueryResult(
        kind="rows",
        sql="SELECT * FROM tbl ORDER BY revenue__num DESC LIMIT 1",
        target_table_id="abc1234567",
        rows=[wide_cells],
        aggregate_value=None,
        contributing_rows=[wide_cells],
        header=["Segment", "Revenue", "Notes"],
        char_start=10,
        char_end=120,
        doc_id="doc-1",
        document_title="NVIDIA 10-K",
        heading_path=["Reportable Segments"],
        section="Reportable Segments",
        superlative=("Revenue", "highest"),
    )
    chunk = _build_synthetic_chunk(result)
    # Whole text bounded.
    assert len(chunk.text) <= 900
    # The framing is present (only emitted because the row is present) and
    # NAMES the source table by its caption (the ar-15 fix — an anonymous
    # "this table" left the answer LLM unable to map the column to the query).
    assert 'Row with the highest Revenue in the "Reportable Segments" table:' in chunk.text
    # The row's value survived (the real cell values, even if the wide Notes
    # cell is truncated).
    assert "Compute & Networking" in chunk.text
    assert "$116,193" in chunk.text


def test_superlative_normal_row_unaffected() -> None:
    """A normal-width superlative row renders exactly as before (framing + row)."""
    result = TableQueryResult(
        kind="rows",
        sql="SELECT * FROM tbl ORDER BY total__num ASC LIMIT 1",
        target_table_id="dircomp01",
        rows=[["Ochoa", "321,309"]],
        aggregate_value=None,
        contributing_rows=[["Ochoa", "321,309"]],
        header=["Name", "Total ($)"],
        char_start=0,
        char_end=50,
        doc_id="doc-1",
        document_title="Proxy",
        heading_path=["Director Compensation"],
        section="Director Compensation",
        superlative=("Total ($)", "lowest"),
    )
    chunk = _build_synthetic_chunk(result)
    # The caption-named framing (ar-15): "Total ($)" alone is anonymous; the
    # "Director Compensation" table name is what lets the answer node map it to
    # the question's "total compensation".
    assert 'Row with the lowest Total ($) in the "Director Compensation" table:' in chunk.text
    assert "Ochoa" in chunk.text
    assert "321,309" in chunk.text


def test_plain_rows_oversized_first_row_kept_truncated() -> None:
    """A plain (non-superlative) rows result whose first row is oversized still
    keeps that row (truncated) — the value is never lost."""
    wide = ["A", "y" * 2000]
    result = TableQueryResult(
        kind="rows",
        sql="SELECT * FROM tbl",
        target_table_id="t000000000",
        rows=[wide],
        aggregate_value=None,
        contributing_rows=[wide],
        header=["Name", "Blob"],
        char_start=0,
        char_end=50,
        doc_id="doc-1",
        document_title="Doc",
        heading_path=[],
        section="",
        superlative=None,
    )
    chunk = _build_synthetic_chunk(result)
    assert len(chunk.text) <= 900
    # No section → the framing falls back to the anonymous "this table".
    assert "Matching rows in this table:" in chunk.text
    # The leading value survived.
    assert chunk.text.count("Name=A") == 1


# ======================================================================
# Gate behaviour (node-level, no full graph)
# ======================================================================


@pytest.mark.asyncio
async def test_gate_skips_non_table_shaped_query(settings: MemexSettings) -> None:
    """A query with no aggregation/superlative/filter keywords → gate-false →
    no model call, reranked untouched."""
    await _seed_tables(settings.vault_path, [_segments_table()])
    state = AnswerState(query="What is reflexivity?", reranked=[_table_chunk()])
    update = await query_tables(state)
    assert "reranked" not in update  # not augmented
    assert update.get("nodes_traversed") == state.nodes_traversed + 1


@pytest.mark.asyncio
async def test_gate_skips_when_no_stored_tables(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Table-shaped query but the chunk's doc has no stored tables AND no
    [table-rows] chunk → gate-false → no model call."""
    plain = Chunk(
        chunk_id="doc-x#bbbbbbbbbb",
        document_id="doc-x",
        document_title="Other",
        text="Just prose.",
        score=0.5,
    )

    async def _should_not_call(**_kw: object) -> tuple[object, int]:
        raise AssertionError("complete_structured must not be called when gated out")

    monkeypatch.setattr("memex.agents.table_sql.complete_structured", _should_not_call)
    state = AnswerState(query="What is the total revenue?", reranked=[plain])
    update = await query_tables(state)
    assert "reranked" not in update


@pytest.mark.asyncio
async def test_node_returns_full_augmented_list(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gated-in aggregate query injects ONE synthetic chunk and returns the
    FULL reranked list (real chunk + synthetic), not just the synthetic."""
    await _seed_tables(settings.vault_path, [_segments_table()])

    async def _fake(*, prompt: object, schema: type, **_kw: object) -> tuple[object, int]:
        return GeneratedSQL(
            sql="SELECT SUM(revenue__num) FROM abc1234567", target_table_id="abc1234567"
        ), 10

    monkeypatch.setattr("memex.agents.table_sql.complete_structured", _fake)
    monkeypatch.setattr("memex.agents.table_sql.render_prompt", lambda name, **_kw: f"[{name}]")

    real = _table_chunk()
    state = AnswerState(query="What is the total revenue across all segments?", reranked=[real])
    update = await query_tables(state)
    reranked = update.get("reranked")
    assert reranked is not None
    assert len(reranked) == 2  # real + synthetic, real NOT wiped
    assert reranked[0].chunk_id == real.chunk_id
    synthetic = reranked[1]
    assert synthetic.chunk_id == "doc-1#sql0001"
    # Non-colliding: 'sql' is non-hex, suffix len 7 (can't fuzzy-match a 10-hex hash).
    assert "sql" in synthetic.chunk_id
    # Evidence (contributing rows) leads; SQL appended.
    assert "Compute & Networking" in synthetic.text
    assert "133302" in synthetic.text or "Aggregate result" in synthetic.text
    assert "[sql]" in synthetic.text
    # The WHOLE chunk text — evidence + as much of the trailing [sql] line as
    # fits — is bounded by _SYNTHETIC_TEXT_MAX (900), so it survives the
    # smallest (1200) assess truncate.
    assert len(synthetic.text) <= 900


# ======================================================================
# FIX 1 — query_tables exception guard (query_doc_tables raises → no-op)
# ======================================================================


@pytest.mark.asyncio
async def test_query_tables_node_no_ops_when_query_doc_tables_raises(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If `query_doc_tables` raises a `ModelCallError` (the `complete_structured`
    inside it failed), the `query_tables` node must NOT propagate it — it logs
    and returns the no-op `nodes_traversed` bump, leaving `reranked` untouched so
    the agent proceeds on the normal set."""
    from memex.core.errors import ModelCallError

    await _seed_tables(settings.vault_path, [_segments_table()])

    async def _boom(*_a: object, **_k: object) -> object:
        raise ModelCallError("vLLM exploded", context={"node": "query_doc_tables"})

    monkeypatch.setattr("memex.agents.table_sql.query_doc_tables", _boom)

    real = _table_chunk()
    state = AnswerState(query="What is the total revenue across all segments?", reranked=[real])
    update = await query_tables(state)
    # No crash; no-op update.
    assert "reranked" not in update
    assert update.get("nodes_traversed") == state.nodes_traversed + 1


@pytest.mark.asyncio
async def test_query_tables_node_no_ops_when_store_read_raises(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `sqlite3.Error` from `tables_for_document` is also caught → no-op."""
    import sqlite3

    await _seed_tables(settings.vault_path, [_segments_table()])

    async def _bad_read(self: object, doc_id: str) -> object:
        raise sqlite3.OperationalError("malformed database")

    monkeypatch.setattr("memex.index.table_store.TableStore.tables_for_document", _bad_read)

    state = AnswerState(query="What is the total revenue?", reranked=[_table_chunk()])
    update = await query_tables(state)
    assert "reranked" not in update
    assert update.get("nodes_traversed") == state.nodes_traversed + 1


@pytest.mark.asyncio
async def test_full_graph_query_doc_tables_raises_still_refuses(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: `query_doc_tables` raising must NOT crash the answer. With no
    table injection and an insufficient assessment, the graph refuses normally
    (the table failure is fully swallowed)."""
    await _seed_tables(settings.vault_path, [_segments_table()])
    chunk = _table_chunk()

    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return [chunk]

    async def _rerank(query: str, candidates: list[Chunk], top_k: int = 10) -> list[Chunk]:
        return list(candidates[:top_k])

    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    monkeypatch.setattr("memex.agents.answering.render_prompt", lambda name, **_kw: f"[{name}]")
    monkeypatch.setattr(
        "memex.agents.answering.render_messages",
        lambda name, **_kw: [{"role": "user", "content": f"[{name}]"}],
    )

    from memex.core.errors import ModelCallError

    async def _boom(*_a: object, **_k: object) -> object:
        raise ModelCallError("vLLM exploded", context={})

    monkeypatch.setattr("memex.agents.table_sql.query_doc_tables", _boom)

    async def _llm(
        *, prompt: str | list[dict[str, str]], schema: type, **_kw: object
    ) -> tuple[Any, int]:
        name = schema.__name__
        if name == "SufficiencyAssessment":
            return SufficiencyAssessment(sufficient=False, reason="cannot answer"), 10
        raise AssertionError(f"unexpected schema {name} on a refusal path")

    monkeypatch.setattr("memex.agents.answering.complete_structured", _llm)

    response = await answer_query(
        "What was the total revenue across all segments?",
        graph_expansion_enabled=False,
    )
    # No crash; the table failure no-opped and the graph refused on its own.
    assert response.answered is False
    assert all(c.chunk_id != "doc-1#sql0001" for c in response.used_chunks)


@pytest.mark.asyncio
async def test_query_tables_node_does_not_swallow_cancellation(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard must NOT catch `asyncio.CancelledError` — it propagates."""
    import asyncio

    await _seed_tables(settings.vault_path, [_segments_table()])

    async def _cancel(*_a: object, **_k: object) -> object:
        raise asyncio.CancelledError()

    monkeypatch.setattr("memex.agents.table_sql.query_doc_tables", _cancel)

    state = AnswerState(query="What is the total revenue?", reranked=[_table_chunk()])
    with pytest.raises(asyncio.CancelledError):
        await query_tables(state)


# ======================================================================
# Full graph — happy path: aggregate answer grounded + cited
# ======================================================================


class _GraphFakeLLM:
    """Fake for both answering nodes (assess/answer/verify) AND the table_sql
    SQL generation. Routes by schema."""

    def __init__(self, sql: str, target: str) -> None:
        self._sql = sql
        self._target = target
        self.calls: list[type] = []

    async def __call__(
        self, *, prompt: str | list[dict[str, str]], schema: type, **_kw: object
    ) -> tuple[Any, int]:
        self.calls.append(schema)
        name = schema.__name__
        if name == "GeneratedSQL":
            return GeneratedSQL(sql=self._sql, target_table_id=self._target), 10
        if name == "SufficiencyAssessment":
            return SufficiencyAssessment(sufficient=True, reason="table answer present"), 10
        if name == "DraftAnswer":
            return (
                DraftAnswer(
                    summary="Total reportable-segment revenue is $133,302M.",
                    claims=[
                        CitedClaim(
                            claim="Total reportable-segment revenue is 133302.",
                            source_chunk_id="doc-1#sql0001",
                            confidence="high",
                        )
                    ],
                ),
                10,
            )
        if name == "VerificationResult":
            return VerificationResult(grounded=[0], ungrounded=[]), 10
        if name == "RelevanceAssessment":
            return RelevanceAssessment(responsive=True, reason="ok"), 6
        raise AssertionError(f"unexpected schema {name}")


@pytest.mark.asyncio
async def test_full_graph_aggregate_answer_grounded_and_cited(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _seed_tables(settings.vault_path, [_segments_table()])
    chunk = _table_chunk()

    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return [chunk]

    async def _rerank(query: str, candidates: list[Chunk], top_k: int = 10) -> list[Chunk]:
        return list(candidates[:top_k])

    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    monkeypatch.setattr("memex.agents.answering.render_prompt", lambda name, **_kw: f"[{name}]")
    monkeypatch.setattr(
        "memex.agents.answering.render_messages",
        lambda name, **_kw: [{"role": "user", "content": f"[{name}]"}],
    )
    monkeypatch.setattr("memex.agents.table_sql.render_prompt", lambda name, **_kw: f"[{name}]")

    fake = _GraphFakeLLM(sql="SELECT SUM(revenue__num) FROM abc1234567", target="abc1234567")
    monkeypatch.setattr("memex.agents.answering.complete_structured", fake)
    monkeypatch.setattr("memex.agents.table_sql.complete_structured", fake)

    response = await answer_query(
        "What was the total reportable-segment revenue?",
        graph_expansion_enabled=False,
    )
    assert response.answered is True
    assert response.summary is not None
    assert GeneratedSQL in fake.calls  # the SQL node fired
    # The synthetic chunk is cited.
    assert any(c.source_chunk_id == "doc-1#sql0001" for c in response.claims)
    assert any(c.chunk_id == "doc-1#sql0001" for c in response.used_chunks)


# ======================================================================
# HARD GATE 1 — counterfactual aggregate over absent column → refuse
# ======================================================================


@pytest.mark.asyncio
async def test_hard_gate_absent_column_no_injection_refuses(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A counterfactual aggregate over a column not in the table: the SQL
    references a nonexistent column → exec error → None → no injection. With
    no other grounding, the graph refuses (it never fabricates)."""
    await _seed_tables(settings.vault_path, [_segments_table()])
    chunk = _table_chunk()

    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return [chunk]

    async def _rerank(query: str, candidates: list[Chunk], top_k: int = 10) -> list[Chunk]:
        return list(candidates[:top_k])

    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    monkeypatch.setattr("memex.agents.answering.render_prompt", lambda name, **_kw: f"[{name}]")
    monkeypatch.setattr(
        "memex.agents.answering.render_messages",
        lambda name, **_kw: [{"role": "user", "content": f"[{name}]"}],
    )
    monkeypatch.setattr("memex.agents.table_sql.render_prompt", lambda name, **_kw: f"[{name}]")

    seen: dict[str, list[Chunk]] = {}

    async def _llm(
        *, prompt: str | list[dict[str, str]], schema: type, **_kw: object
    ) -> tuple[Any, int]:
        name = schema.__name__
        if name == "GeneratedSQL":
            # absent column → sqlite error inside query_doc_tables → None
            return GeneratedSQL(
                sql="SELECT SUM(profit__num) FROM abc1234567", target_table_id="abc1234567"
            ), 10
        if name == "SufficiencyAssessment":
            return SufficiencyAssessment(
                sufficient=False, reason="no profit data in the corpus"
            ), 10
        raise AssertionError(f"unexpected schema {name} — answer/verify must not run on a refusal")

    monkeypatch.setattr("memex.agents.answering.complete_structured", _llm)
    monkeypatch.setattr("memex.agents.table_sql.complete_structured", _llm)
    _ = seen

    response = await answer_query(
        "What was the total profit across all segments?",
        graph_expansion_enabled=False,
    )
    assert response.answered is False
    assert response.claims == []
    # No synthetic chunk was injected (the absent-column SQL errored).
    assert all(c.chunk_id != "doc-1#sql0001" for c in response.used_chunks)


# ======================================================================
# HARD GATE 2 — confident-wrong SQL → recompute disagrees → refuse
# ======================================================================


@pytest.mark.asyncio
async def test_hard_gate_confident_wrong_sql_recompute_disagrees_refuses(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The LLM emits a confident aggregate, but the independent recompute
    disagrees → no injection → refuse. We force disagreement by patching
    `_recompute_aggregate` (simulating a wrong column-mapping the recompute
    catches)."""
    await _seed_tables(settings.vault_path, [_segments_table()])
    chunk = _table_chunk()

    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return [chunk]

    async def _rerank(query: str, candidates: list[Chunk], top_k: int = 10) -> list[Chunk]:
        return list(candidates[:top_k])

    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    monkeypatch.setattr("memex.agents.answering.render_prompt", lambda name, **_kw: f"[{name}]")
    monkeypatch.setattr(
        "memex.agents.answering.render_messages",
        lambda name, **_kw: [{"role": "user", "content": f"[{name}]"}],
    )
    monkeypatch.setattr("memex.agents.table_sql.render_prompt", lambda name, **_kw: f"[{name}]")

    def _wrong_recompute(*_a: object, **_k: object) -> float:
        return 999999.0  # disagrees with the SQL SUM

    monkeypatch.setattr("memex.agents.table_sql._recompute_aggregate", _wrong_recompute)

    async def _llm(
        *, prompt: str | list[dict[str, str]], schema: type, **_kw: object
    ) -> tuple[Any, int]:
        name = schema.__name__
        if name == "GeneratedSQL":
            return GeneratedSQL(
                sql="SELECT SUM(revenue__num) FROM abc1234567", target_table_id="abc1234567"
            ), 10
        if name == "SufficiencyAssessment":
            return SufficiencyAssessment(sufficient=False, reason="cannot verify the total"), 10
        raise AssertionError(f"unexpected schema {name} on a refusal path")

    monkeypatch.setattr("memex.agents.answering.complete_structured", _llm)
    monkeypatch.setattr("memex.agents.table_sql.complete_structured", _llm)

    response = await answer_query(
        "What was the total revenue across all segments?",
        graph_expansion_enabled=False,
    )
    assert response.answered is False
    assert all(c.chunk_id != "doc-1#sql0001" for c in response.used_chunks)


# ======================================================================
# proximity table-scoping — GPU-acceptance regression 2026-05-24
# ======================================================================


def test_relevant_tables_scopes_to_near_chunks_and_caps() -> None:
    """`_relevant_tables` passes only tables near the retrieved chunks, capped —
    so a doc with ~74 tables doesn't dump every schema into the SQL-gen prompt
    and overflow the context (the bug GPU acceptance surfaced)."""
    from memex.agents.answering import _TABLE_CANDIDATE_CAP, _relevant_tables, _span_gap
    from memex.core.types import StoredTable

    # 20 tables spaced 1000 chars apart in one doc.
    tables = [
        StoredTable(
            doc_id="d",
            table_id=f"t{i:02d}",
            section="",
            header=["a"],
            rows=[["1"]],
            char_start=i * 1000,
            char_end=i * 1000 + 100,
        )
        for i in range(20)
    ]
    # One reranked chunk overlapping table #5's span.
    chunk = Chunk(
        chunk_id="d#abcdef0123",
        document_id="d",
        document_title="",
        text="...",
        char_start=5000,
        char_end=5100,
    )
    near = _relevant_tables(tables, [chunk], "d")
    assert near, "the table the chunk overlaps must be returned"
    assert near[0].table_id == "t05", "nearest (overlapping) table first"
    assert len(near) <= _TABLE_CANDIDATE_CAP, "candidate set is capped"
    # A faraway-only chunk (no table within the margin) → empty.
    far = Chunk(
        chunk_id="d#beefbeef00",
        document_id="d",
        document_title="",
        text="...",
        char_start=900_000,
        char_end=900_100,
    )
    assert _relevant_tables(tables, [far], "d") == []
    # A chunk with no span (0,0) contributes nothing.
    empty = Chunk(chunk_id="d#0000000000", document_id="d", document_title="", text="x")
    assert _relevant_tables(tables, [empty], "d") == []
    # _span_gap: overlap → 0; disjoint → positive gap.
    assert _span_gap(0, 100, 50, 150) == 0
    assert _span_gap(0, 100, 300, 400) == 200


def test_query_matched_tables_surfaces_by_section_header() -> None:
    """`_query_matched_tables` offers a table whose section/header lexically
    matches the query, regardless of where its chunk reranked — the recall path
    that fixed ar-14 (a 'director fees' query whose Director Compensation table
    chunk never reranked into the candidate pool)."""
    from memex.agents.answering import _TABLE_QUERY_MATCH_MIN, _query_matched_tables
    from memex.core.types import StoredTable

    def _t(tid: str, section: str, header: list[str]) -> StoredTable:
        return StoredTable(
            doc_id="d", table_id=tid, section=section, header=header, rows=[["1"]],
            char_start=0, char_end=10,
        )

    director_comp = _t(
        "dc", "Director Compensation", ["Name", "Fees Earned or Paid in Cash ($)", "Total ($)"]
    )
    exec_comp = _t(
        "ec", "Summary Compensation", ["Name", "Salary ($)", "Bonus ($)", "Stock Awards ($)"]
    )
    unrelated = _t("ux", "Lease Maturities", ["Year", "Operating Leases ($)"])
    tables = [exec_comp, director_comp, unrelated]

    q = "What was the total amount of fees earned or paid in cash to all of NVIDIA's directors?"
    matched = _query_matched_tables(tables, q)
    ids = [t.table_id for t in matched]
    assert ids[0] == "dc", "the header-matching Director Compensation table ranks first"
    assert "ux" not in ids, "an unrelated table (< min overlap) is not offered"

    # A query that matches nothing → empty (no spurious offering).
    assert _query_matched_tables(tables, "Who is the CEO?") == []
    # Below-threshold overlap (a single shared term) is not enough.
    assert _TABLE_QUERY_MATCH_MIN == 2


def test_select_tables_unions_proximity_and_query_match() -> None:
    """`_select_tables` = proximity ∪ query-match, query-matched first, deduped,
    capped — so the right table is offered whether retrieval reranked its chunk
    (proximity) or only its name matches (query-relevance)."""
    from memex.agents.answering import _TABLE_CANDIDATE_CAP, _select_tables
    from memex.core.types import StoredTable

    # A query-matching table FAR from any reranked chunk (proximity would miss it).
    director_comp = StoredTable(
        doc_id="d", table_id="dc", section="Director Compensation",
        header=["Name", "Fees Earned or Paid in Cash ($)"], rows=[["x", "1"]],
        char_start=150_000, char_end=150_500,
    )
    # A proximity table near the reranked chunk but NOT query-matching.
    near_table = StoredTable(
        doc_id="d", table_id="nt", section="Balance Sheet", header=["Assets", "Liabilities"],
        rows=[["1", "2"]], char_start=5000, char_end=5100,
    )
    chunk = Chunk(
        chunk_id="d#abcdef0123", document_id="d", document_title="", text="...",
        char_start=5000, char_end=5100,
    )
    q = "total fees earned or paid in cash to directors"
    sel = _select_tables([near_table, director_comp], [chunk], q, "d")
    ids = [t.table_id for t in sel]
    assert "dc" in ids, "the query-matched table is offered even though no chunk is near it"
    assert "nt" in ids, "the proximity table is still offered"
    assert ids[0] == "dc", "query-matched first"
    assert len(ids) == len(set(ids)), "no duplicates"
    assert len(sel) <= _TABLE_CANDIDATE_CAP

    # When proximity already covers the matched table, dedup keeps one copy.
    near_dc = director_comp.model_copy(update={"char_start": 5000, "char_end": 5100})
    sel2 = _select_tables([near_dc], [chunk], q, "d")
    assert [t.table_id for t in sel2] == ["dc"], "proximity ∪ query-match dedups to one"
