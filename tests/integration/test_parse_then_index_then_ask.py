"""End-to-end test of ingest → parse → index → answer with all heavy I/O faked.

Validates the full Phase-1 chain without a real vLLM, Langfuse,
Docling, LanceDB, or sentence-transformers. The vault is a real tmp
directory (so file copies, atomic writes, and manifests exercise the
real code), but every external library is monkey-patched.

This is the Phase-1 regression net IMPLEMENTATION-PLAN §3 calls for.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from memex.agents.answering import (
    AnswerState,
    Chunk,
    CitedClaim,
    DraftAnswer,
    SufficiencyAssessment,
    VerificationResult,
    answer_query,
    reset_compiled_graph,
)
from memex.core import config as config_mod
from memex.core.config import MemexSettings, set_settings
from memex.core.manifest import read_manifest
from memex.ingest.pipeline import IngestRequest, ingest_file
from memex.parse import pipeline as parse_pipeline
from memex.parse.docling_backend import DoclingConversion, DoclingPageOutput
from memex.parse.pipeline import parse_document, reset_docling_breaker

# ----- Fixtures: tmp vault, settings, fake heavy I/O -----


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    (tmp_path / "documents").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def settings(
    tmp_vault: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[MemexSettings]:
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_vault))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    s = MemexSettings()  # type: ignore[call-arg]
    set_settings(s)
    yield s
    set_settings(None)


@pytest.fixture(autouse=True)
def _fresh_graph() -> Iterator[None]:
    reset_compiled_graph()
    reset_docling_breaker()
    yield
    reset_compiled_graph()
    reset_docling_breaker()


@pytest.fixture
def fake_pdf(tmp_path: Path) -> Path:
    """A 'PDF' that satisfies the magic-byte check. Docling is faked, so
    the bytes after the header are irrelevant.
    """
    p = tmp_path / "input.pdf"
    p.write_bytes(b"%PDF-1.7\n%fake pdf content for tests\n%%EOF\n")
    return p


@pytest.fixture
def patch_docling(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_convert(
        source: Path, *, timeout_s: int, **_kw: object
    ) -> DoclingConversion:
        return DoclingConversion(
            markdown=(
                "# Smith 2024 on reflexivity\n\n"
                "Reflexivity in research design means the researcher's "
                "presence shapes the data they collect.\n\n"
                "Smith argues that reflexivity is constitutive rather "
                "than corrective.\n"
            ),
            pages=[
                DoclingPageOutput(page=1, markdown="(page 1)", confidence=0.92),
                DoclingPageOutput(page=2, markdown="(page 2)", confidence=0.88),
            ],
            docling_version="fake-1.0",
            figure_count=0,
            table_count=0,
            equation_count=0,
        )

    monkeypatch.setattr(
        "memex.parse.pipeline.docling_convert", _fake_convert
    )


@pytest.fixture
def patch_index_stores(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace LanceDB + FTS5 stores with in-memory fakes so we don't need
    a working LanceDB install for the parse → index → ask test.
    """
    written_chunks: list[Chunk] = []

    class _FakeStore:
        @classmethod
        async def open(cls, vault_path: Path) -> _FakeStore:
            return cls()

        async def upsert(self, chunks: list[Chunk], *args: object) -> None:
            written_chunks.extend(chunks)

        async def delete_document(self, doc_id: str) -> int:
            before = len(written_chunks)
            written_chunks[:] = [c for c in written_chunks if c.document_id != doc_id]
            return before - len(written_chunks)

        async def delete_chunks(self, chunk_ids: list[str]) -> int:
            """Selective delete — used by incremental re-indexing."""
            ids = set(chunk_ids)
            before = len(written_chunks)
            written_chunks[:] = [c for c in written_chunks if c.chunk_id not in ids]
            return before - len(written_chunks)

        async def chunk_ids_for_document(self, doc_id: str) -> set[str]:
            """Used by incremental re-indexing to compute the diff."""
            return {c.chunk_id for c in written_chunks if c.document_id == doc_id}

        async def search(self, *args: object, **kwargs: object) -> list[Chunk]:
            return list(written_chunks)

        async def close(self) -> None:
            return None

    monkeypatch.setattr("memex.index.pipeline.VectorStore", _FakeStore)
    monkeypatch.setattr("memex.index.pipeline.FTSStore", _FakeStore)
    monkeypatch.setattr(
        "memex.index.pipeline._embed_chunks",
        lambda chunks: _fake_embed(chunks),
    )
    return {"chunks": written_chunks}


async def _fake_embed(chunks: list[Chunk]) -> list[list[float]]:
    return [[0.0] * 768 for _ in chunks]


@pytest.fixture
def patch_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake retrieve + prompt + model layer so the agent runs without
    a real vLLM or model registry."""

    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        # Make up two chunks that look like what the chunker would have produced.
        return [
            Chunk(
                chunk_id="d1#a",
                document_id="d1",
                document_title="Smith 2024 on reflexivity",
                text="Reflexivity means the researcher shapes the data.",
            ),
            Chunk(
                chunk_id="d1#b",
                document_id="d1",
                document_title="Smith 2024 on reflexivity",
                text="Smith argues reflexivity is constitutive.",
            ),
        ]

    async def _rerank(query, candidates, top_k=10):  # type: ignore[no-untyped-def]
        return list(candidates[:top_k])

    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    monkeypatch.setattr(
        "memex.agents.answering.render_prompt",
        lambda name, **_kw: f"[fake {name} prompt]",
    )

    async def _structured(
        *, prompt: str, schema: type, **_kw: object
    ) -> tuple[Any, int]:
        if schema is SufficiencyAssessment:
            return SufficiencyAssessment(sufficient=True, reason="ok"), 5
        if schema is DraftAnswer:
            return DraftAnswer(
                summary="Smith treats reflexivity as constitutive.",
                claims=[
                    CitedClaim(
                        claim="Reflexivity is the researcher shaping the data.",
                        source_chunk_id="d1#a",
                        confidence="high",
                    ),
                ],
            ), 25
        if schema is VerificationResult:
            return VerificationResult(grounded=[0], ungrounded=[]), 10
        raise AssertionError(f"unexpected schema {schema}")

    monkeypatch.setattr("memex.agents.answering.complete_structured", _structured)


# ----- Tests -----


@pytest.mark.asyncio
async def test_ingest_then_parse_writes_markdown_and_manifest(
    settings: MemexSettings, fake_pdf: Path, patch_docling: None
) -> None:
    result = await ingest_file(IngestRequest(source_path=fake_pdf))
    assert result.accepted is True
    assert result.doc_id is not None
    doc_id = result.doc_id

    # ingest writes the source copy + initial manifest
    asset_dir = settings.vault_path / "documents" / doc_id
    assert (asset_dir / "source.pdf").exists()
    manifest = await read_manifest(settings.vault_path, doc_id)
    assert manifest is not None
    assert manifest.ingest is not None
    assert manifest.ingest.detected_mime == "application/pdf"

    # parse runs Docling (faked) and produces the canonical markdown
    parse_result = await parse_document(doc_id)
    assert parse_result.engine == "docling"
    canonical = settings.vault_path / "documents" / f"{doc_id}.md"
    assert canonical.exists()
    assert "reflexivity" in canonical.read_text(encoding="utf-8").lower()

    # manifest now carries the parse stage
    manifest = await read_manifest(settings.vault_path, doc_id)
    assert manifest is not None
    assert manifest.parse is not None
    assert manifest.parse.docling_version == "fake-1.0"
    assert len(manifest.parse.pages) == 2


@pytest.mark.asyncio
async def test_full_chain_ingest_parse_index_ask(
    settings: MemexSettings,
    fake_pdf: Path,
    patch_docling: None,
    patch_index_stores: dict[str, Any],
    patch_agent: None,
) -> None:
    result = await ingest_file(IngestRequest(source_path=fake_pdf))
    assert result.accepted and result.doc_id is not None
    await parse_document(result.doc_id)
    # index_document also needs the registry's embedder for real, but
    # patch_index_stores swapped `_embed_chunks` out, so no model is touched.
    from memex.index.pipeline import index_document

    index_result = await index_document(result.doc_id)
    assert index_result.chunk_count > 0

    # manifest now has all three stages
    manifest = await read_manifest(settings.vault_path, result.doc_id)
    assert manifest is not None
    assert manifest.ingest is not None
    assert manifest.parse is not None
    assert manifest.index is not None
    assert manifest.index.chunk_count == index_result.chunk_count

    # ask the agent — fully faked retrieve + model
    response = await answer_query("What does Smith say about reflexivity?")
    assert response.answered is True
    assert any("reflexivity" in c.claim.lower() for c in response.claims)


@pytest.mark.asyncio
async def test_validation_rejects_unknown_format(
    settings: MemexSettings, tmp_path: Path
) -> None:
    bogus = tmp_path / "weird.xyz"
    bogus.write_bytes(b"\x00\x01\x02 nothing recognisable here")
    result = await ingest_file(IngestRequest(source_path=bogus))
    assert result.accepted is False
    assert result.rejection_reason is not None
    assert "supported format" in result.rejection_reason


@pytest.mark.asyncio
async def test_validation_rejects_oversize(
    settings: MemexSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    big = tmp_path / "huge.pdf"
    big.write_bytes(b"%PDF-1.7\n" + b"x" * (1024 * 1024))
    # Shrink the cap so the test doesn't write hundreds of megs.
    monkeypatch.setenv("MEMEX_INGEST__MAX_BYTES", "1024")
    new_settings = MemexSettings()  # type: ignore[call-arg]
    set_settings(new_settings)
    try:
        result = await ingest_file(IngestRequest(source_path=big))
        assert result.accepted is False
        assert "max is" in (result.rejection_reason or "")
    finally:
        set_settings(settings)


@pytest.mark.asyncio
async def test_markdown_passthrough_skips_docling(
    settings: MemexSettings, tmp_path: Path
) -> None:
    src = tmp_path / "notes.md"
    src.write_text("# Notes\n\nSome content.\n", encoding="utf-8")
    ingest_result = await ingest_file(IngestRequest(source_path=src))
    assert ingest_result.accepted and ingest_result.is_markdown
    assert ingest_result.doc_id is not None

    parse_result = await parse_document(ingest_result.doc_id)
    assert parse_result.engine == "passthrough"

    canonical = settings.vault_path / "documents" / f"{ingest_result.doc_id}.md"
    assert canonical.exists()
    assert "Some content." in canonical.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_macros_rejected_unless_allowed(
    settings: MemexSettings, tmp_path: Path
) -> None:
    """A ZIP with a Word-shaped layout + vbaProject.bin is rejected."""
    import zipfile

    docx = tmp_path / "with_macros.docx"
    with zipfile.ZipFile(docx, "w") as zf:
        zf.writestr("[Content_Types].xml", "")
        zf.writestr("word/document.xml", "")
        zf.writestr("word/vbaProject.bin", b"\x00\x01\x02")

    result = await ingest_file(IngestRequest(source_path=docx))
    assert result.accepted is False
    assert "macros" in (result.rejection_reason or "")
