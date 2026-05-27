"""VLMTranscriptionCache + its wiring into `vlm_backend.convert_pages`.

The cache makes the non-deterministic VLM reproducible by construction:
the model is invoked once per `(pdf-bytes, page, model, prompt)` key and
reused thereafter. These tests cover the store mechanics and the
convert_pages cache path (miss→transcribe→store, hit→reuse-skip-GPU,
refresh→re-transcribe, min-length guard) with a faked VLM (no torch).
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from memex.parse import vlm_backend
from memex.parse.docling_backend import DoclingPageOutput
from memex.parse.vlm_cache import VLMTranscriptionCache

# ── store mechanics ──────────────────────────────────────────────────────


async def test_put_get_roundtrip(tmp_path: Path) -> None:
    cache = await VLMTranscriptionCache.open(tmp_path)
    try:
        assert await cache.get("k1") is None  # miss
        await cache.put(
            "k1", pdf_sha256="abc", page_no=3, vlm_model="m", prompt_sha8="p", markdown="# Page"
        )
        assert await cache.get("k1") == "# Page"
    finally:
        await cache.close()


async def test_insert_or_ignore_first_writer_wins(tmp_path: Path) -> None:
    cache = await VLMTranscriptionCache.open(tmp_path)
    try:
        for md in ("first", "second"):
            await cache.put(
                "k1", pdf_sha256="abc", page_no=1, vlm_model="m", prompt_sha8="p", markdown=md
            )
        assert await cache.get("k1") == "first"  # first writer wins
    finally:
        await cache.close()


async def test_delete_by_pdf_scopes_to_one_document(tmp_path: Path) -> None:
    cache = await VLMTranscriptionCache.open(tmp_path)
    try:
        await cache.put(
            "a:1", pdf_sha256="A", page_no=1, vlm_model="m", prompt_sha8="p", markdown="x"
        )
        await cache.put(
            "a:2", pdf_sha256="A", page_no=2, vlm_model="m", prompt_sha8="p", markdown="y"
        )
        await cache.put(
            "b:1", pdf_sha256="B", page_no=1, vlm_model="m", prompt_sha8="p", markdown="z"
        )
        assert await cache.delete_by_pdf("A") == 2
        assert await cache.get("a:1") is None
        assert await cache.get("b:1") == "z"  # other document untouched
    finally:
        await cache.close()


async def test_persists_on_disk_across_reopen(tmp_path: Path) -> None:
    cache = await VLMTranscriptionCache.open(tmp_path)
    await cache.put("k", pdf_sha256="A", page_no=1, vlm_model="m", prompt_sha8="p", markdown="kept")
    await cache.close()
    assert (tmp_path / ".memex" / "vlm_cache.sqlite").is_file()
    reopened = await VLMTranscriptionCache.open(tmp_path)
    try:
        assert await reopened.get("k") == "kept"
    finally:
        await reopened.close()


# ── convert_pages wiring ─────────────────────────────────────────────────


def _fake_vlm_env(monkeypatch: Any, calls: list[int]) -> None:
    """Fake the VLM so convert_pages runs without torch: a per-page
    transcription that records which pages were actually transcribed, a
    no-op registry handle, and a settings stub for the model id."""

    async def fake_convert(
        handle: object,
        source_pdf: Path,
        page_number: int,
        max_new_tokens: int,
        samples: int = 1,
    ) -> DoclingPageOutput:
        calls.append(page_number)
        return DoclingPageOutput(
            page=page_number, markdown=f"VLM transcription of page {page_number}", confidence=1.0
        )

    class _FakeRegistry:
        @contextlib.asynccontextmanager
        async def use(self, _name: str) -> AsyncGenerator[object]:
            yield object()

    monkeypatch.setattr(vlm_backend, "_convert_with_handle", fake_convert)
    monkeypatch.setattr(vlm_backend, "get_registry", lambda: _FakeRegistry())
    monkeypatch.setattr(
        vlm_backend,
        "get_settings",
        lambda: SimpleNamespace(
            models=SimpleNamespace(vlm="test-vlm", vlm_serving="transformers"),
            parse=SimpleNamespace(vlm_transcription_samples=1),
        ),
    )


async def test_convert_pages_misses_then_reuses(tmp_path: Path, monkeypatch: Any) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.7 fake bytes")
    cache = await VLMTranscriptionCache.open(tmp_path)
    calls: list[int] = []
    _fake_vlm_env(monkeypatch, calls)
    try:
        # 1st call: both pages miss → VLM (fake) transcribes both.
        r1 = await vlm_backend.convert_pages(source_pdf=pdf, page_numbers=[1, 2], cache=cache)
        assert calls == [1, 2]
        assert isinstance(r1[1], DoclingPageOutput) and "page 1" in r1[1].markdown

        # 2nd call: both hit → VLM NOT invoked, byte-identical (reproducible).
        calls.clear()
        r2 = await vlm_backend.convert_pages(source_pdf=pdf, page_numbers=[1, 2], cache=cache)
        assert calls == []
        assert isinstance(r2[1], DoclingPageOutput) and r2[1].markdown == r1[1].markdown

        # refresh_vlm: bust this doc → re-transcribe.
        calls.clear()
        await vlm_backend.convert_pages(
            source_pdf=pdf, page_numbers=[1, 2], cache=cache, refresh_vlm=True
        )
        assert calls == [1, 2]
    finally:
        await cache.close()


async def test_convert_pages_min_length_guard_not_frozen(tmp_path: Path, monkeypatch: Any) -> None:
    """A near-empty draw (the VLM punting a hard diagram) is NOT cached, so
    the next parse retries rather than freezing the bad output."""
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF fake")
    cache = await VLMTranscriptionCache.open(tmp_path)
    calls: list[int] = []

    async def punt(
        handle: object,
        source_pdf: Path,
        page_number: int,
        max_new_tokens: int,
        samples: int = 1,
    ) -> DoclingPageOutput:
        calls.append(page_number)
        return DoclingPageOutput(page=page_number, markdown="x", confidence=1.0)  # sub-threshold

    class _FakeRegistry:
        @contextlib.asynccontextmanager
        async def use(self, _name: str) -> AsyncGenerator[object]:
            yield object()

    monkeypatch.setattr(vlm_backend, "_convert_with_handle", punt)
    monkeypatch.setattr(vlm_backend, "get_registry", lambda: _FakeRegistry())
    monkeypatch.setattr(
        vlm_backend,
        "get_settings",
        lambda: SimpleNamespace(
            models=SimpleNamespace(vlm="m", vlm_serving="transformers"),
            parse=SimpleNamespace(vlm_transcription_samples=1),
        ),
    )
    try:
        await vlm_backend.convert_pages(source_pdf=pdf, page_numbers=[1], cache=cache)
        calls.clear()
        await vlm_backend.convert_pages(source_pdf=pdf, page_numbers=[1], cache=cache)
        assert calls == [1]  # re-invoked — the punt was not cached
    finally:
        await cache.close()


async def test_convert_with_handle_keeps_longest_of_n(tmp_path: Path, monkeypatch: Any) -> None:
    """best-of-N: with samples>1, the LONGEST draw is kept — a content-
    completeness proxy for the non-deterministic VLM (a draw that drops
    content is shorter). The chosen draw is what convert_pages caches."""
    drafts = iter(["short draw", "the much longer and more complete draw", "medium draw here"])
    monkeypatch.setattr(vlm_backend, "_render_page_to_image", lambda pdf, page: object())
    monkeypatch.setattr(
        vlm_backend, "_vlm_transcribe_sync", lambda handle, image, prompt, mnt: next(drafts)
    )
    fake_handle: Any = object()
    out = await vlm_backend._convert_with_handle(
        fake_handle, tmp_path / "x.pdf", 1, 1024, samples=3
    )
    assert out.markdown == "the much longer and more complete draw"


# ── vLLM-served VLM backend (vlm_serving="vllm") ─────────────────────────


def _fake_vllm_env(monkeypatch: Any, calls: list[int], server_starts: list[str]) -> None:
    """Fake the vLLM-served VLM path: a no-launch server CM that records each
    start, and a per-page transcription that records pages — no subprocess,
    no torch, no openai."""

    @contextlib.asynccontextmanager
    async def fake_serve(model_id: str) -> AsyncGenerator[str]:
        server_starts.append(model_id)
        yield "http://fake-vlm:8001/v1"

    async def fake_convert(
        base_url: str,
        model_id: str,
        source_pdf: Path,
        page_number: int,
        max_new_tokens: int,
        samples: int = 1,
    ) -> DoclingPageOutput:
        calls.append(page_number)
        return DoclingPageOutput(
            page=page_number, markdown=f"vLLM transcription of page {page_number}", confidence=1.0
        )

    monkeypatch.setattr(vlm_backend, "_serve_vlm_vllm", fake_serve)
    monkeypatch.setattr(vlm_backend, "_convert_one_via_vllm", fake_convert)
    monkeypatch.setattr(
        vlm_backend,
        "get_settings",
        lambda: SimpleNamespace(
            models=SimpleNamespace(vlm="test-vlm", vlm_serving="vllm"),
            parse=SimpleNamespace(vlm_transcription_samples=1),
        ),
    )


async def test_convert_pages_vllm_backend_caches_and_skips_server_when_cached(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The vLLM path transcribes misses (server started ONCE for the batch),
    caches them, and on a full cache hit never starts the VLM vLLM at all
    (no wasteful ~30 s boot when nothing needs transcribing)."""
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.7 vllm fake")
    cache = await VLMTranscriptionCache.open(tmp_path)
    calls: list[int] = []
    server_starts: list[str] = []
    _fake_vllm_env(monkeypatch, calls, server_starts)
    try:
        # 1st: both miss → one server start, both transcribed + cached.
        r1 = await vlm_backend.convert_pages(source_pdf=pdf, page_numbers=[1, 2], cache=cache)
        assert calls == [1, 2]
        assert server_starts == ["test-vlm"]  # server lifecycle wraps the whole batch
        assert isinstance(r1[1], DoclingPageOutput) and "page 1" in r1[1].markdown

        # 2nd: both hit → VLM vLLM NEVER started (no misses), byte-identical.
        calls.clear()
        server_starts.clear()
        r2 = await vlm_backend.convert_pages(source_pdf=pdf, page_numbers=[1, 2], cache=cache)
        assert calls == []
        assert server_starts == []  # no misses → no vLLM boot
        assert isinstance(r2[2], DoclingPageOutput)
        assert isinstance(r1[2], DoclingPageOutput)
        assert r2[2].markdown == r1[2].markdown
    finally:
        await cache.close()
