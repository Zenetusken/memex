"""End-to-end test of ingest → parse → index → answer with all heavy I/O faked.

Validates the full Phase-1 chain without a real vLLM, Langfuse,
Docling, LanceDB, or sentence-transformers. The vault is a real tmp
directory (so file copies, atomic writes, and manifests exercise the
real code), but every external library is monkey-patched.

This is the Phase-1 regression net IMPLEMENTATION-PLAN §3 calls for.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from memex.agents.answering import (
    Chunk,
    CitedClaim,
    DraftAnswer,
    RelevanceAssessment,
    SufficiencyAssessment,
    VerificationResult,
    answer_query,
    reset_compiled_graph,
)
from memex.core.config import MemexSettings, set_settings
from memex.core.manifest import read_manifest
from memex.ingest.pipeline import IngestRequest, ingest_file
from memex.parse.docling_backend import DoclingConversion, DoclingPageOutput
from memex.parse.pipeline import (
    parse_document,
    reset_docling_breaker,
    reset_pymupdf_breaker,
)
from memex.parse.pymupdf_backend import (
    PdfSignals,
    PyMuPDFConversion,
    PyMuPDFCrashed,
    PyMuPDFPageOutput,
    PyMuPDFUnavailable,
)

# ----- Fixtures: tmp vault, settings, fake heavy I/O -----


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    (tmp_path / "documents").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def settings(tmp_vault: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemexSettings]:
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
    reset_pymupdf_breaker()
    yield
    reset_compiled_graph()
    reset_docling_breaker()
    reset_pymupdf_breaker()


@pytest.fixture
def fake_pdf(tmp_path: Path) -> Path:
    """A 'PDF' that satisfies the magic-byte check. Docling is faked, so
    the bytes after the header are irrelevant.
    """
    p = tmp_path / "input.pdf"
    p.write_bytes(b"%PDF-1.7\n%fake pdf content for tests\n%%EOF\n")
    return p


@pytest.fixture
def patch_docling(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, list[dict[str, object]]]:
    """Fake Docling and also record every call so tests can assert it
    was (or wasn't) invoked and inspect the `force_ocr` kwarg.
    """
    calls: list[dict[str, object]] = []

    async def _fake_convert(
        source: Path,
        *,
        timeout_s: int,
        sandbox_network: bool = True,
        force_ocr: bool | None = None,
        **_kw: object,
    ) -> DoclingConversion:
        calls.append(
            {
                "source": source,
                "timeout_s": timeout_s,
                "sandbox_network": sandbox_network,
                "force_ocr": force_ocr,
            }
        )
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

    monkeypatch.setattr("memex.parse.pipeline.docling_convert", _fake_convert)
    return {"calls": calls}


def _make_pymupdf_conversion(
    *,
    creator: str | None = None,
    producer: str | None = None,
    chars_per_page_avg: float = 200.0,
    aspect_ratio: float = 1.78,
    is_tagged: bool = False,
    embedded_font_count: int = 0,
    image_area_fraction: float = 0.0,
    image_heavy_page_fraction: float = 0.0,
    image_count_total: int = 0,
    replacement_char_fraction: float = 0.0,
    empty_page_fraction: float = 0.0,
    chars_per_page_p90: float = 250.0,
    has_headings: bool = True,
    has_tables: bool = False,
    has_lists: bool = False,
    has_code_blocks: bool = False,
    page_count: int = 2,
) -> PyMuPDFConversion:
    """Build a PyMuPDFConversion with controllable signals for routing tests."""
    pages = [
        PyMuPDFPageOutput(
            page=i + 1,
            markdown=f"# page {i + 1}\n\n" + ("text " * int(chars_per_page_avg / 5)),
            char_count=int(chars_per_page_avg),
            image_count=3 if image_heavy_page_fraction > 0.5 else 0,
            aspect_ratio=aspect_ratio,
        )
        for i in range(page_count)
    ]
    return PyMuPDFConversion(
        markdown="\n\n".join(p.markdown for p in pages),
        pages=pages,
        pymupdf_version="fake-1.27",
        signals=PdfSignals(
            creator=creator,
            producer=producer,
            is_tagged=is_tagged,
            page_count=page_count,
            avg_aspect_ratio=aspect_ratio,
            embedded_font_count=embedded_font_count,
            image_count_total=image_count_total,
            image_heavy_page_fraction=image_heavy_page_fraction,
            image_area_fraction=image_area_fraction,
            total_chars=int(chars_per_page_avg * page_count),
            chars_per_page_avg=chars_per_page_avg,
            chars_per_page_median=chars_per_page_avg,
            chars_per_page_p10=chars_per_page_avg * 0.5,
            chars_per_page_p90=chars_per_page_p90,
            empty_page_fraction=empty_page_fraction,
            replacement_char_fraction=replacement_char_fraction,
            word_like_token_fraction=0.6,
            unique_char_variety=70,
            whitespace_fraction=0.15,
            has_headings=has_headings,
            has_tables=has_tables,
            has_lists=has_lists,
            has_code_blocks=has_code_blocks,
        ),
    )


@pytest.fixture
def patch_pymupdf_born_digital(monkeypatch: pytest.MonkeyPatch) -> None:
    """PyMuPDF fake that returns a born-digital PowerPoint document
    (portrait, text-dense — e.g., a whitepaper exported from
    PowerPoint, NOT a slide deck). Slide-shaped PowerPoint content
    is tested separately under the slide-deck routing fixtures.
    """

    async def _fake(source: Path, *, timeout_s: int, **_kw: object) -> PyMuPDFConversion:
        return _make_pymupdf_conversion(
            producer="Microsoft PowerPoint 2023",
            chars_per_page_avg=1200.0,  # text-dense, above slide-deck threshold
            aspect_ratio=0.77,  # portrait letter, fails slide-deck aspect gate
            has_headings=True,
        )

    monkeypatch.setattr("memex.parse.pipeline.pymupdf_convert", _fake)


@pytest.fixture
def patch_pymupdf_scan_producer(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake(source: Path, *, timeout_s: int, **_kw: object) -> PyMuPDFConversion:
        return _make_pymupdf_conversion(
            producer="ABBYY FineReader OCR",
            chars_per_page_avg=600.0,
        )

    monkeypatch.setattr("memex.parse.pipeline.pymupdf_convert", _fake)


@pytest.fixture
def patch_pymupdf_mixed_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mixed-content: native text + substantial images (charts/screenshots)."""

    async def _fake(source: Path, *, timeout_s: int, **_kw: object) -> PyMuPDFConversion:
        return _make_pymupdf_conversion(
            producer="Microsoft PowerPoint 2023",
            chars_per_page_avg=200.0,
            aspect_ratio=1.78,
            image_area_fraction=0.40,
            image_heavy_page_fraction=0.60,
            image_count_total=20,
        )

    monkeypatch.setattr("memex.parse.pipeline.pymupdf_convert", _fake)


@pytest.fixture
def patch_pymupdf_mojibake(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake(source: Path, *, timeout_s: int, **_kw: object) -> PyMuPDFConversion:
        return _make_pymupdf_conversion(
            producer="Generic PDF Writer",
            chars_per_page_avg=400.0,
            replacement_char_fraction=0.20,
        )

    monkeypatch.setattr("memex.parse.pipeline.pymupdf_convert", _fake)


@pytest.fixture
def patch_pymupdf_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake(source: Path, *, timeout_s: int, **_kw: object) -> PyMuPDFConversion:
        raise PyMuPDFUnavailable(
            "pymupdf not installed",
            context={"source": str(source)},
        )

    monkeypatch.setattr("memex.parse.pipeline.pymupdf_convert", _fake)


@pytest.fixture
def patch_pymupdf_crashes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake(source: Path, *, timeout_s: int, **_kw: object) -> PyMuPDFConversion:
        raise PyMuPDFCrashed(
            "subprocess exited 134",
            context={"source": str(source), "exit_code": 134},
        )

    monkeypatch.setattr("memex.parse.pipeline.pymupdf_convert", _fake)


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

    async def _structured(*, prompt: str, schema: type, **_kw: object) -> tuple[Any, int]:
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
        if schema is VerificationResult or schema.__name__ == VerificationResult.__name__:
            return VerificationResult(grounded=[0], ungrounded=[]), 10
        if schema.__name__ == "RelevanceAssessment":
            return RelevanceAssessment(responsive=True, reason="ok"), 6
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
async def test_validation_rejects_unknown_format(settings: MemexSettings, tmp_path: Path) -> None:
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
async def test_markdown_passthrough_skips_docling(settings: MemexSettings, tmp_path: Path) -> None:
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
async def test_pymupdf_routes_powerpoint_to_pymupdf(
    settings: MemexSettings,
    fake_pdf: Path,
    patch_docling: dict[str, list[dict[str, object]]],
    patch_pymupdf_born_digital: None,
) -> None:
    """PowerPoint producer + 200 chars/page → Tier 1.A → use PyMuPDF.
    Docling fake must not be called.
    """
    result = await ingest_file(IngestRequest(source_path=fake_pdf))
    assert result.accepted and result.doc_id is not None

    parse_result = await parse_document(result.doc_id)
    assert parse_result.engine == "pymupdf"
    assert all(p.engine == "pymupdf" for p in parse_result.pages)
    assert patch_docling["calls"] == []

    manifest = await read_manifest(settings.vault_path, result.doc_id)
    assert manifest is not None and manifest.parse is not None
    assert manifest.parse.pymupdf_version == "fake-1.27"


@pytest.mark.asyncio
async def test_pymupdf_records_citation_grade_page_starts_and_indexes_pages(
    settings: MemexSettings,
    fake_pdf: Path,
    patch_docling: dict[str, list[dict[str, object]]],
    patch_pymupdf_born_digital: None,
    patch_index_stores: dict[str, Any],
) -> None:
    """Companion arc-3 end-to-end wiring (PyMuPDF route — its fake reconstructs its own markdown,
    so the parse marker round-trip succeeds). Parse records each page's citation-grade `char_start`
    in `doc.body`; `index_document` maps those boundaries through the transforms to attribute
    `Chunk.page`. The on-disk body is unchanged (the helper always returns the canonical body)."""
    from memex.index.pipeline import index_document
    from memex.vault.store import read_document

    result = await ingest_file(IngestRequest(source_path=fake_pdf))
    assert result.accepted and result.doc_id is not None
    parse_result = await parse_document(result.doc_id)
    assert parse_result.engine == "pymupdf"

    # PARSE: every page carries a real (>= 0) citation-grade char_start, and the recorded offset
    # for page 2 lands on page 2's content in the written body (offsets are meaningful, not -1).
    manifest = await read_manifest(settings.vault_path, result.doc_id)
    assert manifest is not None and manifest.parse is not None
    pages = manifest.parse.pages
    assert len(pages) == 2
    assert all(p.char_start >= 0 for p in pages)  # citation-grade (not the -1 legacy fallback)
    doc = await read_document(settings.vault_path, result.doc_id)
    starts = {p.page: p.char_start for p in pages}
    assert starts[1] == 0
    assert doc.body[starts[2] :].startswith("# page 2")

    # INDEX: the exact page intervals drive Chunk.page — populated, monotonic, and a page-2-content
    # chunk is attributed to page 2.
    await index_document(result.doc_id)
    # The fake captures every upsert across BOTH the vector and FTS stores → dedupe by chunk_id
    # (document order preserved by dict insertion).
    chunks = list({c.chunk_id: c for c in patch_index_stores["chunks"]}.values())
    assert chunks and all(c.page is not None for c in chunks)
    page_seq = [c.page for c in chunks]
    assert page_seq == sorted(page_seq)  # non-decreasing in document order
    assert {c.page for c in chunks} == {1, 2}
    page2_chunks = [c for c in chunks if "page 2" in c.text]
    assert page2_chunks and all(c.page == 2 for c in page2_chunks)


@pytest.mark.asyncio
async def test_docling_page_aligned_markdown_reaches_citation_grade(
    settings: MemexSettings, fake_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Companion arc-3 on the PRIMARY engine. The Docling document-level markdown equals the
    per-page join ONLY after VLM escalation (re-stitched as the join) or on a deck with no
    serialization divergence; that shape reaches citation-grade. Fake exactly that shape and
    assert each content page records `char_start` and the offset lands on its content."""
    from memex.vault.store import read_document

    p1 = "# Slide 1\n\nIntro content for slide one."
    p2 = "# Slide 2\n\nSecond slide body text here."

    async def _fake(source: Path, *, timeout_s: int, **_kw: object) -> DoclingConversion:
        return DoclingConversion(
            markdown="\n\n".join([p1, p2]),  # the post-escalation / non-divergent shape
            pages=[
                DoclingPageOutput(page=1, markdown=p1, confidence=0.95),
                DoclingPageOutput(page=2, markdown=p2, confidence=0.95),
            ],
            docling_version="fake-1.0",
            figure_count=0,
            table_count=0,
            equation_count=0,
        )

    monkeypatch.setattr("memex.parse.pipeline.docling_convert", _fake)
    result = await ingest_file(IngestRequest(source_path=fake_pdf))
    assert result.accepted and result.doc_id is not None
    parse_result = await parse_document(result.doc_id)
    assert parse_result.engine == "docling"

    manifest = await read_manifest(settings.vault_path, result.doc_id)
    assert manifest is not None and manifest.parse is not None
    pages = manifest.parse.pages
    assert len(pages) == 2
    assert all(p.char_start >= 0 for p in pages if p.char_count > 0)  # citation-grade activated
    doc = await read_document(settings.vault_path, result.doc_id)
    starts = {p.page: p.char_start for p in pages}
    assert starts[1] == 0
    assert doc.body[starts[2] :].startswith("# Slide 2")


@pytest.mark.asyncio
async def test_docling_divergent_markdown_stays_navgrade(
    settings: MemexSettings, fake_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real NON-escalated Docling case: `conversion.markdown` is a whole-doc serialization that
    does NOT equal the per-page join. The parse round-trip guard must FAIL-SAFE to nav-grade
    (`char_start == -1` on every page) — never a wrong map, never a body change."""
    from memex.vault.store import read_document

    async def _fake(source: Path, *, timeout_s: int, **_kw: object) -> DoclingConversion:
        return DoclingConversion(
            markdown="# Whole\n\nA doc-level serialization that differs from the page join.",
            pages=[
                DoclingPageOutput(page=1, markdown="# A\n\nalpha", confidence=0.95),
                DoclingPageOutput(page=2, markdown="# B\n\nbeta", confidence=0.95),
            ],
            docling_version="fake-1.0",
            figure_count=0,
            table_count=0,
            equation_count=0,
        )

    monkeypatch.setattr("memex.parse.pipeline.docling_convert", _fake)
    result = await ingest_file(IngestRequest(source_path=fake_pdf))
    assert result.accepted and result.doc_id is not None
    await parse_document(result.doc_id)

    manifest = await read_manifest(settings.vault_path, result.doc_id)
    assert manifest is not None and manifest.parse is not None
    assert all(p.char_start == -1 for p in manifest.parse.pages)  # fail-safe nav-grade
    # The written body is still the canonical finalize of the doc-level markdown (unchanged behavior).
    doc = await read_document(settings.vault_path, result.doc_id)
    assert "doc-level serialization that differs" in doc.body


@pytest.mark.asyncio
async def test_pymupdf_scanner_producer_falls_through_with_force_ocr(
    settings: MemexSettings,
    fake_pdf: Path,
    patch_docling: dict[str, list[dict[str, object]]],
    patch_pymupdf_scan_producer: None,
) -> None:
    """ABBYY producer → Tier 1.B scan → fall through to Docling with
    OCR forced on. Verifies the mixed-content/scan routing carries
    `force_ocr=True` through to docling_convert.
    """
    result = await ingest_file(IngestRequest(source_path=fake_pdf))
    assert result.accepted and result.doc_id is not None

    parse_result = await parse_document(result.doc_id)
    assert parse_result.engine == "docling"
    assert len(patch_docling["calls"]) == 1
    call = patch_docling["calls"][0]
    assert call["force_ocr"] is True


@pytest.mark.asyncio
async def test_pymupdf_mixed_content_routes_to_docling_with_ocr(
    settings: MemexSettings,
    fake_pdf: Path,
    patch_docling: dict[str, list[dict[str, object]]],
    patch_pymupdf_mixed_content: None,
) -> None:
    """Born-digital PowerPoint with substantial image area (charts,
    screenshots, diagrams) → mixed-content tier → fall through to
    Docling with OCR forced on so image-embedded text is captured.

    This is the case the user explicitly flagged: documents with both
    native text AND images whose text needs OCR for full context.
    """
    result = await ingest_file(IngestRequest(source_path=fake_pdf))
    assert result.accepted and result.doc_id is not None

    parse_result = await parse_document(result.doc_id)
    assert parse_result.engine == "docling"
    assert len(patch_docling["calls"]) == 1
    call = patch_docling["calls"][0]
    assert call["force_ocr"] is True


@pytest.mark.asyncio
async def test_pymupdf_mojibake_falls_through_without_forced_ocr(
    settings: MemexSettings,
    fake_pdf: Path,
    patch_docling: dict[str, list[dict[str, object]]],
    patch_pymupdf_mojibake: None,
) -> None:
    """Broken encoding (20% U+FFFD) → Tier 3 mojibake → fall through.
    OCR won't help with a font-mapping bug, so force_ocr stays None.
    """
    result = await ingest_file(IngestRequest(source_path=fake_pdf))
    assert result.accepted and result.doc_id is not None

    parse_result = await parse_document(result.doc_id)
    assert parse_result.engine == "docling"
    assert len(patch_docling["calls"]) == 1
    assert patch_docling["calls"][0]["force_ocr"] is None


@pytest.mark.asyncio
async def test_pymupdf_unavailable_falls_through_silently(
    settings: MemexSettings,
    fake_pdf: Path,
    patch_docling: dict[str, list[dict[str, object]]],
    patch_pymupdf_unavailable: None,
) -> None:
    """PyMuPDF not installed → fall through to Docling, no crash record."""
    result = await ingest_file(IngestRequest(source_path=fake_pdf))
    assert result.accepted and result.doc_id is not None

    parse_result = await parse_document(result.doc_id)
    assert parse_result.engine == "docling"

    manifest = await read_manifest(settings.vault_path, result.doc_id)
    assert manifest is not None and manifest.parse is not None
    assert manifest.parse.crashed is False
    assert manifest.parse.pymupdf_version is None


@pytest.mark.asyncio
async def test_pymupdf_crash_falls_through_no_manifest_record(
    settings: MemexSettings,
    fake_pdf: Path,
    patch_docling: dict[str, list[dict[str, object]]],
    patch_pymupdf_crashes: None,
) -> None:
    """PyMuPDF subprocess crashes → fall through to Docling. Only
    Docling's own outcome touches the manifest; PyMuPDF crashes are
    logged but never recorded as parse-stage failures.
    """
    result = await ingest_file(IngestRequest(source_path=fake_pdf))
    assert result.accepted and result.doc_id is not None

    parse_result = await parse_document(result.doc_id)
    assert parse_result.engine == "docling"

    manifest = await read_manifest(settings.vault_path, result.doc_id)
    assert manifest is not None and manifest.parse is not None
    assert manifest.parse.crashed is False


@pytest.mark.asyncio
async def test_macros_rejected_unless_allowed(settings: MemexSettings, tmp_path: Path) -> None:
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
