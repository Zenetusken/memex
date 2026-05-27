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
from pathlib import Path
from typing import Any, ClassVar

import pytest

from memex.agents import document_summarizer as ds
from memex.agents.answering import CitedClaim, DocAbstract, SectionSummary
from memex.core.config import MemexSettings, set_settings
from memex.core.types import Chunk


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


def _fake_complete(*, ground: bool, capture: list[str] | None = None):
    """A `complete_structured` stand-in dispatched on the schema name:

    - `SectionSummary`     → one key-point citing the FIRST chunk-id in the prompt
      (so grounding can match it), `section_title=""` to exercise the heading fallback.
    - `VerificationResult` → grounds index 0 when `ground`, else marks it ungrounded.
    - `DocAbstract`        → a fixed whole-doc abstract (proves REDUCE ran).
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
            return DocAbstract(abstract="The whole-document abstract."), 8
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
