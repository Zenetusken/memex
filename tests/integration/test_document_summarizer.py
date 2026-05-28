"""Integration test for the structured grounded document summarizer (ADR-0008).

Fakes `FTSStore.chunks_for_document` (no SQLite) + `complete_structured`
(schema-dispatched) so the full map-reduce + per-section grounding + refusal
logic runs end-to-end without a model. Mirrors the fake-LLM pattern in
`test_answering_with_fakes.py`.

The grounding contract is the point: a key-point survives only if the verifier
confirms it, and a zero-grounded document refuses rather than ship an ungrounded
summary (the no-hallucination HARD gate, extended to summaries).
"""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, ClassVar

import pytest

from memex.agents import document_summarizer as ds
from memex.agents.answering import CitedClaim, DocAbstract, ReportStructure, SectionSummary
from memex.core.config import MemexSettings, set_settings
from memex.core.types import Chunk, StoredTable


@pytest.fixture
def _settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemexSettings:
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    (tmp_path / "documents").mkdir()
    s = MemexSettings(vault_path=tmp_path)
    set_settings(s)
    return s


def _chunk(cid: str, heading: str, text: str = "Some grounded body text.") -> Chunk:
    return Chunk(
        chunk_id=cid,
        document_id="docA",
        document_title="Doc A",
        text=text,
        heading_path=[heading],
    )


class _FakeFTS:
    """Stand-in for FTSStore: serves a fixed chunk list, no SQLite, no I/O."""

    chunks: ClassVar[list[Chunk]] = []

    @classmethod
    async def open(cls, vault_path: Any) -> _FakeFTS:
        return cls()

    async def chunks_for_document(self, doc_id: str) -> list[Chunk]:
        return list(_FakeFTS.chunks)

    async def close(self) -> None:
        return None


class _FakeTableStore:
    """Stand-in for TableStore: serves a fixed StoredTable list (the tabular route)."""

    tables: ClassVar[list[StoredTable]] = []

    @classmethod
    async def open(cls, vault_path: Any) -> _FakeTableStore:
        return cls()

    async def tables_for_document(self, doc_id: str) -> list[StoredTable]:
        return list(_FakeTableStore.tables)

    async def close(self) -> None:
        return None


def _stored_table(i: int) -> StoredTable:
    return StoredTable(
        doc_id="docA",
        table_id=f"t{i}",
        section=f"Segment {i}",
        header=["Segment", "Revenue"],
        rows=[["Gaming", "$16,042M"], ["Data Center", "$30,000M"]],
        char_start=0,
        char_end=20,
    )


async def _no_embed(*_a: object, **_k: object) -> None:
    """Force the report confidence to the lexical-only path in tests (no model load)."""
    return None


def _fake_complete(
    *, ground: bool, capture: list[str] | None = None, plan_starts: list[int] | None = None
):
    """A `complete_structured` stand-in dispatched on the schema name:

    - `SectionSummary`     → one key-point citing the FIRST chunk-id in the prompt
      (so grounding can match it), `section_title=""` to exercise the heading fallback.
    - `VerificationResult` → grounds index 0 when `ground`, else marks it ungrounded.
    - `DocAbstract`        → a fixed whole-doc abstract (proves REDUCE ran).
    - `ReportStructure`    → `plan_starts` if given, else `[0]` (trivial → the planner
      falls back to mechanical batching, the default report path).
    """

    async def _complete(
        prompt: str,
        schema: type,
        max_tokens: int = 0,
        prompt_tag: str = "",
        **_kw: object,
    ) -> tuple[Any, int]:
        if capture is not None:
            capture.append(prompt)
        name = schema.__name__
        if name == "ReportStructure":
            return ReportStructure(paragraph_starts=plan_starts if plan_starts is not None else [0]), 6
        if name == "SectionSummary":
            m = re.search(r"\[([^\]]*#[^\]]*)\]", prompt)
            cid = m.group(1) if m else "docA#x"
            return (
                SectionSummary(
                    section_title="",
                    digest="A concise digest of the section.",
                    key_points=[
                        CitedClaim(
                            claim="A grounded point.", source_chunk_id=cid, confidence="high"
                        )
                    ],
                ),
                10,
            )
        if name == "VerificationResult":
            return (
                schema(
                    grounded=[0] if ground else [],
                    ungrounded=[] if ground else [0],
                    ungrounded_reasons=[] if ground else ["not supported by the cited chunk"],
                ),
                5,
            )
        if name == "DocAbstract":
            return DocAbstract(sentences=["The whole-document abstract."]), 8
        raise AssertionError(f"unexpected schema {name!r}")

    return _complete


@pytest.mark.asyncio
async def test_long_doc_map_reduce_grounded(
    _settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two big sections → long route → MAP each + GROUND + REDUCE.
    chunks = [_chunk("docA#1", "Intro", "a" * 9_000), _chunk("docA#2", "Methods", "b" * 9_000)]
    monkeypatch.setattr(ds, "FTSStore", _FakeFTS)
    monkeypatch.setattr(_FakeFTS, "chunks", chunks)
    monkeypatch.setattr(ds, "complete_structured", _fake_complete(ground=True))

    resp = await ds.summarize_document("docA")

    assert resp.answered
    assert resp.summary == "The whole-document abstract."  # REDUCE produced it
    assert [s.section_title for s in resp.sections] == ["Intro", "Methods"]  # heading fallback
    assert all(len(s.key_points) == 1 for s in resp.sections)  # each grounded
    assert len(resp.claims) == 2  # doc-level points = grounded section points
    assert resp.artifact_scope_doc_ids == ["docA"]
    assert resp.wikilinks  # derived from the cited chunks


@pytest.mark.asyncio
async def test_on_phase_emits_section_sequence_and_threads_cid(
    _settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The opt-in progress hook (for the webui's live indicator): a caller-supplied
    correlation_id threads into the response, and on_phase fires the per-section
    sequence → Reducing → Composing. Observe-only; omitting it is unchanged (covered
    by the other tests, which call summarize_document without the new params)."""
    chunks = [_chunk("docA#1", "Intro", "a" * 9_000), _chunk("docA#2", "Methods", "b" * 9_000)]
    monkeypatch.setattr(ds, "FTSStore", _FakeFTS)
    monkeypatch.setattr(_FakeFTS, "chunks", chunks)
    monkeypatch.setattr(ds, "complete_structured", _fake_complete(ground=True))
    seen: list[str] = []

    resp = await ds.summarize_document("docA", correlation_id="fixed-cid", on_phase=seen.append)

    assert resp.correlation_id == "fixed-cid"
    assert seen == [
        "Summarizing · section 1 of 2",
        "Summarizing · section 2 of 2",
        "Reducing",
        "Composing",
    ]


@pytest.mark.asyncio
async def test_report_detail_builds_multi_paragraph_body(
    _settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`detail="report"` (ADR-0010) → a HIERARCHICAL reduce: the grounded section
    digests are batched `_REPORT_SECTIONS_PER_BATCH` at a time and EACH batch reduces
    to one paragraph, stitched with blank lines. So: one paragraph per batch, the
    reduce runs once per batch (every section reaches the output — no
    `_REDUCE_MAX_SECTIONS` truncation), and each batch is told (via the scope-note)
    that it covers a SPAN, not the whole document."""
    # 9 distinct sections → long route → 9 grounded SectionSummaries → ceil(9/4)=3 batches.
    chunks = [_chunk(f"docA#{i}", f"H{i}", chr(97 + i) * 9_000) for i in range(9)]
    monkeypatch.setattr(ds, "FTSStore", _FakeFTS)
    monkeypatch.setattr(_FakeFTS, "chunks", chunks)
    capture: list[str] = []
    monkeypatch.setattr(ds, "complete_structured", _fake_complete(ground=True, capture=capture))
    monkeypatch.setattr(ds, "_embedding_alignment", _no_embed)  # lexical-only, no model load
    seen: list[str] = []

    resp = await ds.summarize_document("docA", detail="report", on_phase=seen.append)

    assert resp.answered
    # The planner gets a trivial plan ([0]) → falls back to mechanical ceil(9/4)=3 batches.
    expected_batches = -(-9 // ds._REPORT_SECTIONS_PER_BATCH)  # 3
    # Multi-paragraph body: one (non-empty) paragraph per batch.
    paragraphs = resp.summary.split("\n\n")
    assert len(paragraphs) == expected_batches
    assert all(p.strip() for p in paragraphs)
    # The reduce ran once PER BATCH, each framed as part of a multi-part overview.
    batch_reduces = [p for p in capture if "multi-part overview" in p]
    assert len(batch_reduces) == expected_batches
    # ROLLING context: exactly the CONTINUATION batches (all but the first) carry the
    # preceding paragraph so they transition + add only new material (ADR-0010 coherence).
    rolling = [p for p in capture if "The overview SO FAR" in p]
    assert len(rolling) == expected_batches - 1
    # Proper ARC: exactly ONE opening (first batch) and ONE closing (last batch).
    assert len([p for p in capture if "This is the OPENING paragraph" in p]) == 1
    assert len([p for p in capture if "This is the CLOSING paragraph" in p]) == 1
    # Live progress: a "Reducing · paragraph k of N" per batch (parses cleanly in
    # the webui's summary_phase_view → base "Reducing" + the paragraph eyebrow).
    para_phases = [s for s in seen if s.startswith("Reducing · paragraph")]
    assert para_phases == [
        f"Reducing · paragraph {i + 1} of {expected_batches}" for i in range(expected_batches)
    ]
    # Every section is still present in the structured by-section breakdown.
    assert len(resp.sections) == 9
    # Close the loop: an inform-only faithfulness confidence, one score per paragraph.
    assert resp.report_confidence is not None
    assert len(resp.report_confidence.per_paragraph) == len(paragraphs)
    assert 0.0 <= resp.report_confidence.overall <= 1.0


@pytest.mark.asyncio
async def test_report_adaptive_structure_uses_plan(
    _settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-0010 adaptive structure: when the planner returns real break points, the report
    is grouped by the PLAN (combining sections), not the mechanical every-4 split. With 9
    sections and breaks at [0, 5], the body is 2 paragraphs (sections 0-4 and 5-8) — which
    mechanical batching (3 paragraphs of 4/4/1) would never produce."""
    chunks = [_chunk(f"docA#{i}", f"H{i}", chr(97 + i) * 9_000) for i in range(9)]
    monkeypatch.setattr(ds, "FTSStore", _FakeFTS)
    monkeypatch.setattr(_FakeFTS, "chunks", chunks)
    capture: list[str] = []
    monkeypatch.setattr(
        ds, "complete_structured", _fake_complete(ground=True, capture=capture, plan_starts=[0, 5])
    )
    monkeypatch.setattr(ds, "_embedding_alignment", _no_embed)

    resp = await ds.summarize_document("docA", detail="report")

    assert resp.answered
    # Two planned groups → two paragraphs (NOT the mechanical 3).
    assert len(resp.summary.split("\n\n")) == 2
    # The planner was actually consulted.
    assert any("PARAGRAPH STRUCTURE" in p for p in capture)
    # Confidence still computed over the (2) planned paragraphs.
    assert resp.report_confidence is not None
    assert len(resp.report_confidence.per_paragraph) == 2


@pytest.mark.asyncio
async def test_report_planner_coalesces_oversplit(
    _settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-0010 coalesce guardrail: an 8B that OVER-splits (a break at every section) must
    not shatter the report. With 9 sections and a break at all 9, the runs are coalesced up
    to `_REPORT_TARGET_SECTIONS_PER_PARAGRAPH` → ~3 fuller paragraphs, NOT 9 singletons."""
    chunks = [_chunk(f"docA#{i}", f"H{i}", chr(97 + i) * 9_000) for i in range(9)]
    monkeypatch.setattr(ds, "FTSStore", _FakeFTS)
    monkeypatch.setattr(_FakeFTS, "chunks", chunks)
    monkeypatch.setattr(
        ds,
        "complete_structured",
        _fake_complete(ground=True, plan_starts=[0, 1, 2, 3, 4, 5, 6, 7, 8]),
    )
    monkeypatch.setattr(ds, "_embedding_alignment", _no_embed)

    resp = await ds.summarize_document("docA", detail="report")

    assert resp.answered
    paragraphs = resp.summary.split("\n\n")
    # 9 single-section runs → coalesced to ceil(9/4) = 3 fuller paragraphs (never 9).
    assert len(paragraphs) == 3
    assert resp.report_confidence is not None
    assert len(resp.report_confidence.per_paragraph) == 3


def test_key_figures_numeric_guard() -> None:
    """The deterministic key-figures backstop (ADR-0008): a figure whose number is ABSENT
    from its cited chunk is dropped (the LLM verifier's near-number false-positive — it
    grounded a fabricated "$16,042M" against a cell holding "16384"); a verbatim figure and a
    no-number qualitative claim survive."""
    chunks = {"docA#tbl0": "Table — Model configs\nModel=GTEsmall; Params=30M; SeqLen=16384"}

    def _claim(text: str) -> CitedClaim:
        return CitedClaim(claim=text, source_chunk_id="docA#tbl0", confidence="high")

    # 2026 + 16042 are both absent from the chunk (which has 16384) → dropped.
    assert ds._figure_number_in_chunk(_claim("Fiscal 2026 Gaming revenue was $16,042M"), chunks) is False
    # 30 is present (30M) → a real verbatim figure survives.
    assert ds._figure_number_in_chunk(_claim("GTEsmall has 30M parameters"), chunks) is True
    # No number → left to the LLM grounding (qualitative key point).
    assert ds._figure_number_in_chunk(_claim("The table lists model configurations"), chunks) is True
    # 16384 IS in the chunk → a figure citing it survives (no false-drop of a real figure).
    assert ds._figure_number_in_chunk(_claim("The max sequence length is 16384"), chunks) is True


@pytest.mark.asyncio
async def test_report_summarizer_swap_in(
    _settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-0010 swap-in: with `models.summarizer` set, a `report` summary pauses the
    orchestrator, serves the swap-in model, and routes the map-reduce there — then the
    stack always closes (orchestrator restored). Mocks the GPU/process contexts; no vLLM."""
    _settings.models.summarizer = "fake/gemma-swap"
    set_settings(_settings)
    chunks = [_chunk(f"docA#{i}", f"H{i}", chr(97 + i) * 9_000) for i in range(5)]
    monkeypatch.setattr(ds, "FTSStore", _FakeFTS)
    monkeypatch.setattr(_FakeFTS, "chunks", chunks)
    monkeypatch.setattr(ds, "complete_structured", _fake_complete(ground=True))
    monkeypatch.setattr(ds, "_embedding_alignment", _no_embed)

    seen = {"paused": False, "served": None, "closed": False}

    @asynccontextmanager
    async def _fake_pause() -> AsyncGenerator[None]:
        seen["paused"] = True
        try:
            yield
        finally:
            seen["closed"] = True  # orchestrator restored on exit

    @asynccontextmanager
    async def _fake_serve(model: str) -> AsyncGenerator[str]:
        seen["served"] = model
        yield "http://fake-summarizer:8002/v1"

    monkeypatch.setattr("memex.parse.pipeline.pause_vllm_for_gpu", _fake_pause)
    monkeypatch.setattr("memex.agents.summarizer_serve.serve_summarizer_vllm", _fake_serve)

    resp = await ds.summarize_document("docA", detail="report")

    assert resp.answered
    assert seen["paused"] is True  # orchestrator paused for the swap
    assert seen["served"] == "fake/gemma-swap"  # the swap model was served
    assert seen["closed"] is True  # stack closed → orchestrator restored


@pytest.mark.asyncio
async def test_report_no_swap_when_summarizer_unset(
    _settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The swap is OPT-IN: with no `models.summarizer`, a report summary never pauses the
    orchestrator or serves anything (the default path is a pure no-op stack)."""
    chunks = [_chunk(f"docA#{i}", f"H{i}", chr(97 + i) * 9_000) for i in range(5)]
    monkeypatch.setattr(ds, "FTSStore", _FakeFTS)
    monkeypatch.setattr(_FakeFTS, "chunks", chunks)
    monkeypatch.setattr(ds, "complete_structured", _fake_complete(ground=True))
    monkeypatch.setattr(ds, "_embedding_alignment", _no_embed)

    served = {"called": False}

    @asynccontextmanager
    async def _fake_serve(model: str) -> AsyncGenerator[str]:
        served["called"] = True
        yield "http://nope:8002/v1"

    monkeypatch.setattr("memex.agents.summarizer_serve.serve_summarizer_vllm", _fake_serve)

    resp = await ds.summarize_document("docA", detail="report")

    assert resp.answered
    assert served["called"] is False  # no swap when summarizer unset


@pytest.mark.asyncio
async def test_non_report_reduce_keeps_whole_document_scope(
    _settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The single-level reduce (brief/standard/detailed) must keep the byte-identical
    whole-document prompt framing — the report `scope_note` is opt-in, never leaks
    into the default path."""
    chunks = [_chunk("docA#1", "Intro", "a" * 9_000), _chunk("docA#2", "Methods", "b" * 9_000)]
    monkeypatch.setattr(ds, "FTSStore", _FakeFTS)
    monkeypatch.setattr(_FakeFTS, "chunks", chunks)
    capture: list[str] = []
    monkeypatch.setattr(ds, "complete_structured", _fake_complete(ground=True, capture=capture))

    resp = await ds.summarize_document("docA", detail="standard")

    assert resp.answered
    assert "\n\n" not in resp.summary  # single paragraph, not a multi-paragraph body
    reduce_prompts = [p for p in capture if "per-section digests of" in p]
    assert reduce_prompts and all("a whole-document overview" in p for p in reduce_prompts)
    # None of the report-mode coherence framing leaks into the default path.
    assert all("multi-part overview" not in p for p in reduce_prompts)
    assert all("The overview SO FAR" not in p for p in reduce_prompts)


@pytest.mark.asyncio
async def test_short_doc_single_pass_uses_digest_as_abstract(
    _settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Small single-section doc → short route → one MAP pass, no REDUCE.
    chunks = [_chunk("docA#1", "Intro", "short body")]
    capture: list[str] = []
    monkeypatch.setattr(ds, "FTSStore", _FakeFTS)
    monkeypatch.setattr(_FakeFTS, "chunks", chunks)
    monkeypatch.setattr(ds, "complete_structured", _fake_complete(ground=True, capture=capture))

    resp = await ds.summarize_document("docA")

    assert resp.answered
    assert (
        resp.summary == "A concise digest of the section."
    )  # digest IS the abstract (short route)
    assert len(resp.sections) == 1
    assert resp.sections[0].section_title == "Doc A"  # single group keyed by doc title
    assert len(resp.claims) == 1
    # No DocAbstract (REDUCE) call on the short route.
    assert not any("whole-document" in p for p in capture)


@pytest.mark.asyncio
async def test_zero_grounded_refuses(
    _settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = [_chunk("docA#1", "Intro", "body")]
    monkeypatch.setattr(ds, "FTSStore", _FakeFTS)
    monkeypatch.setattr(_FakeFTS, "chunks", chunks)
    monkeypatch.setattr(ds, "complete_structured", _fake_complete(ground=False))

    resp = await ds.summarize_document("docA")

    assert not resp.answered
    assert resp.refusal_reason
    assert not resp.claims  # nothing grounded → nothing shipped
    assert resp.sections  # the section digests survive (key_points dropped)
    assert all(len(s.key_points) == 0 for s in resp.sections)


@pytest.mark.asyncio
async def test_no_indexed_chunks_refuses(
    _settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ds, "FTSStore", _FakeFTS)
    monkeypatch.setattr(_FakeFTS, "chunks", [])
    monkeypatch.setattr(ds, "complete_structured", _fake_complete(ground=True))

    resp = await ds.summarize_document("ghost")

    assert not resp.answered
    assert "ghost" in (resp.refusal_reason or "")
    assert resp.artifact_scope_doc_ids == ["ghost"]
    assert resp.tokens_used == 0


@pytest.mark.asyncio
async def test_detail_knob_threads_length_guidance_into_prompt(
    _settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = [_chunk("docA#1", "Intro", "short body")]
    capture: list[str] = []
    monkeypatch.setattr(ds, "FTSStore", _FakeFTS)
    monkeypatch.setattr(_FakeFTS, "chunks", chunks)
    monkeypatch.setattr(ds, "complete_structured", _fake_complete(ground=True, capture=capture))

    await ds.summarize_document("docA", detail="detailed")

    # detailed → the abstract guidance ("...5-8 sentences") threads into the MAP prompt
    # (short route maps with the abstract-length guidance).
    assert any("5-8 sentences" in p for p in capture)


@pytest.mark.asyncio
async def test_token_budget_stops_map_reduce_early(
    _settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = [_chunk("docA#1", "Intro", "a" * 9_000), _chunk("docA#2", "Methods", "b" * 9_000)]
    monkeypatch.setattr(ds, "FTSStore", _FakeFTS)
    monkeypatch.setattr(_FakeFTS, "chunks", chunks)
    monkeypatch.setattr(ds, "complete_structured", _fake_complete(ground=True))

    # token_budget=0 → after the first section the running total exceeds it → stop.
    resp = await ds.summarize_document("docA", token_budget=0)

    assert resp.answered  # the first section's point still grounds
    assert len(resp.sections) == 1  # the second section was skipped by the budget


def _fake_complete_drop_tabular():
    """Grounds prose key-points but NOT the tabular ones (the GROUND verify prompt
    for the tabular pass shows the table text — `Table —`). Models a key figure the
    LLM computed/invented that isn't a verbatim cell → must be dropped."""

    async def _complete(
        prompt: str, schema: type, max_tokens: int = 0, prompt_tag: str = "", **_kw: object
    ) -> tuple[Any, int]:
        name = schema.__name__
        if name == "SectionSummary":
            m = re.search(r"\[([^\]]*#[^\]]*)\]", prompt)
            cid = m.group(1) if m else "docA#x"
            return (
                SectionSummary(
                    section_title="",
                    digest="A digest.",
                    key_points=[
                        CitedClaim(claim="A figure.", source_chunk_id=cid, confidence="high")
                    ],
                ),
                10,
            )
        if name == "VerificationResult":
            tabular = "Table —" in prompt  # the table-chunk text only appears in tabular GROUND
            return (
                schema(
                    grounded=[] if tabular else [0],
                    ungrounded=[0] if tabular else [],
                    ungrounded_reasons=["not a verbatim cell"] if tabular else [],
                ),
                5,
            )
        if name == "DocAbstract":
            return DocAbstract(sentences=["The abstract."]), 8
        raise AssertionError(f"unexpected schema {name!r}")

    return _complete


@pytest.mark.asyncio
async def test_long_doc_sub_splits_huge_section_no_content_dropped(
    _settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "Intro" (small) + "Big" (14 chunks → 3 window-sized batches: b0.., b6.., b12..).
    chunks = [_chunk("docA#intro", "Intro", "intro text")] + [
        _chunk(f"docA#b{i}", "Big", "z" * 2_000) for i in range(14)
    ]
    monkeypatch.setattr(ds, "FTSStore", _FakeFTS)
    monkeypatch.setattr(_FakeFTS, "chunks", chunks)
    monkeypatch.setattr(ds, "complete_structured", _fake_complete(ground=True))

    resp = await ds.summarize_document("docA")

    assert resp.answered
    # the huge "Big" section was split into multiple parts (not truncated to one)
    big_parts = [s for s in resp.sections if s.section_title.startswith("Big (part")]
    assert len(big_parts) >= 2
    # a chunk from the LATE batch is cited → its content was summarized, not dropped
    cited = {kp.source_chunk_id for kp in resp.claims}
    assert "docA#b12" in cited


@pytest.mark.asyncio
async def test_tabular_route_surfaces_grounded_key_figures(
    _settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = [_chunk("docA#1", "Intro", "prose body")]
    monkeypatch.setattr(ds, "FTSStore", _FakeFTS)
    monkeypatch.setattr(_FakeFTS, "chunks", chunks)
    monkeypatch.setattr(ds, "TableStore", _FakeTableStore)
    monkeypatch.setattr(_FakeTableStore, "tables", [_stored_table(i) for i in range(4)])
    monkeypatch.setattr(ds, "complete_structured", _fake_complete(ground=True))

    resp = await ds.summarize_document("docA")

    assert resp.answered
    # The "Key figures" section leads, and a figure cites a synthetic table chunk.
    assert resp.sections[0].section_title == "Key figures"
    assert any("#tbl" in c.source_chunk_id for c in resp.claims)
    # The synthetic table-chunk is surfaced for Sources (so the wikilink resolves).
    assert any("#tbl" in c.chunk_id for c in resp.used_chunks)
    assert resp.wikilinks


@pytest.mark.asyncio
async def test_tabular_unsupported_figure_dropped(
    _settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = [_chunk("docA#1", "Intro", "prose body")]
    monkeypatch.setattr(ds, "FTSStore", _FakeFTS)
    monkeypatch.setattr(_FakeFTS, "chunks", chunks)
    monkeypatch.setattr(ds, "TableStore", _FakeTableStore)
    monkeypatch.setattr(_FakeTableStore, "tables", [_stored_table(i) for i in range(4)])
    monkeypatch.setattr(ds, "complete_structured", _fake_complete_drop_tabular())

    resp = await ds.summarize_document("docA")

    assert resp.answered  # the prose half still grounds
    # No ungrounded figure shipped; the empty key-figures section is not inserted.
    assert not any("#tbl" in c.source_chunk_id for c in resp.claims)
    assert all(s.section_title != "Key figures" for s in resp.sections)


@pytest.mark.asyncio
async def test_non_tabular_doc_skips_table_pass(
    _settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fewer than the threshold of tables → no tabular route, behaves like prose.
    chunks = [_chunk("docA#1", "Intro", "prose body")]
    monkeypatch.setattr(ds, "FTSStore", _FakeFTS)
    monkeypatch.setattr(_FakeFTS, "chunks", chunks)
    monkeypatch.setattr(ds, "TableStore", _FakeTableStore)
    monkeypatch.setattr(_FakeTableStore, "tables", [_stored_table(0)])  # 1 < _TABULAR_MIN_TABLES
    monkeypatch.setattr(ds, "complete_structured", _fake_complete(ground=True))

    resp = await ds.summarize_document("docA")

    assert resp.answered
    assert all(s.section_title != "Key figures" for s in resp.sections)
    assert not any("#tbl" in c.source_chunk_id for c in resp.claims)


@pytest.mark.asyncio
async def test_used_chunks_covers_every_section_key_point(
    _settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Root-cause regression for the by-section raw-`docid#hash` bug: `used_chunks`
    must resolve EVERY surfaced citation, not just the doc-level headline 12. The
    webui "By section" breakdown renders every `sections[*].key_points`; if a
    non-headline point's chunk isn't in `used_chunks`, `_source_view` has no
    `chunk_ref` for it → the template shows the raw id. With >12 grounded section
    points (so the doc-level cap excludes some), every section citation must still
    be in used_chunks — proving the resolution without any template fallback."""
    # 3 sections × 5 chunks (~1000 chars) = 15 chunks. 15k>14k → long route;
    # 5k/section < 12k → no sub-split; 3 < 12 sections → no packing.
    chunks = [
        _chunk(f"docA#{s}-{i}", f"Section {s}", "x" * 1000) for s in range(3) for i in range(5)
    ]
    monkeypatch.setattr(ds, "FTSStore", _FakeFTS)
    monkeypatch.setattr(_FakeFTS, "chunks", chunks)

    async def _fake(
        prompt: str, schema: type, max_tokens: int = 0, prompt_tag: str = "", **_kw: Any
    ) -> tuple[Any, int]:
        name = schema.__name__
        if name == "SectionSummary":
            ids = re.findall(r"\[([^\]]*#[^\]]*)\]", prompt)  # every chunk-id in THIS section
            kps = []
            for j, cid in enumerate(ids[:5]):
                # Corrupt the FIRST point's id per section — mangle the doc-id prefix
                # but keep the hash suffix intact ("docAXX#0-0"), exactly how the 8B
                # model fumbles a long id. Repair must snap it back via suffix-match
                # BEFORE grounding, else it can't resolve in used_chunks (raw-hash).
                emit = cid.replace("docA#", "docAXX#", 1) if j == 0 else cid
                kps.append(CitedClaim(claim=f"point {cid}", source_chunk_id=emit, confidence="high"))
            return SectionSummary(section_title="", digest="digest", key_points=kps), 10
        if name == "VerificationResult":
            return schema(grounded=[0, 1, 2, 3, 4], ungrounded=[], ungrounded_reasons=[]), 5
        if name == "DocAbstract":
            return DocAbstract(sentences=["The whole-document abstract."]), 8
        raise AssertionError(f"unexpected schema {name!r}")

    monkeypatch.setattr(ds, "complete_structured", _fake)

    resp = await ds.summarize_document("docA")

    assert resp.answered
    section_cited = {kp.source_chunk_id for s in resp.sections for kp in s.key_points}
    used = {c.chunk_id for c in resp.used_chunks}
    # The gap must actually exist (otherwise the test wouldn't catch the bug):
    # more distinct section citations than the doc-level headline cap.
    assert len(section_cited) > ds._MAX_DOC_KEY_POINTS
    # EVERY section citation resolves to a chunk in used_chunks — no raw-hash fallback.
    assert section_cited <= used, f"unresolved: {section_cited - used}"
    # The corrupted ids were REPAIRED (snapped to real ids), not left mangled —
    # so no "docAXX#" survives and there's nothing for a fallback to render.
    assert all("docAXX#" not in cid for cid in section_cited)
