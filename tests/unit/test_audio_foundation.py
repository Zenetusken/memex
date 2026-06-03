"""Unit tests for the audio-ASR foundation (Increment 1, ADR-0017).

Covers the additive, back-compatible config / manifest / types changes that the
audio ingestion route builds on: `ModelSettings.asr*`, `manifest.TranscriptSegment`
+ `ParseStage.segments`, and `Chunk.time_range`. All defaults must keep legacy
payloads loading unchanged (the route is purely additive).
"""

from __future__ import annotations

from memex.core.config import ASRServeSettings, MemexSettings
from memex.core.manifest import ParseStage, TranscriptSegment, now_utc
from memex.core.types import Chunk


def test_model_settings_asr_defaults() -> None:
    s = MemexSettings()  # type: ignore[call-arg]
    # Unconfigured by default → the route will raise ASRUnavailable until a model is set.
    assert s.models.asr is None
    # The in-process CTranslate2 backend is the recommended default (no per-doc cold start).
    assert s.models.asr_backend == "faster_whisper"
    # The serve recipe exists for the (non-default) vllm backend, on a port distinct from
    # the orchestrator (8000) / VLM (8001) / summarizer (8002).
    assert isinstance(s.models.asr_serve, ASRServeSettings)
    assert s.models.asr_serve.port == 8003


def test_transcript_segment_defaults() -> None:
    seg = TranscriptSegment(index=0, char_start=0, char_end=10, start_s=0.0, end_s=2.5)
    assert seg.language == ""
    assert seg.confidence == 1.0
    assert seg.rationale == ""


def test_parse_stage_segments_roundtrip() -> None:
    stage = ParseStage(
        correlation_id="cid",
        parsed_at=now_utc(),
        parser_version="audio-v1",
        segments=[
            TranscriptSegment(
                index=0, char_start=0, char_end=12, start_s=0.0, end_s=3.0, language="fr"
            ),
            TranscriptSegment(
                index=1,
                char_start=14,
                char_end=20,
                start_s=3.0,
                end_s=5.5,
                confidence=0.0,
                rationale="VLM-free ASR failed",
            ),
        ],
    )
    reloaded = ParseStage.model_validate_json(stage.model_dump_json())
    assert reloaded == stage
    assert [s.start_s for s in reloaded.segments] == [0.0, 3.0]


def test_parse_stage_legacy_manifest_has_empty_segments() -> None:
    # A manifest written before the audio route (no `segments` key) must load unchanged.
    legacy = (
        '{"correlation_id": "cid", "parsed_at": "2026-01-01T00:00:00Z", "parser_version": "v1"}'
    )
    stage = ParseStage.model_validate_json(legacy)
    assert stage.segments == []
    assert stage.pages == []


def test_chunk_time_range_default_and_roundtrip() -> None:
    plain = Chunk(chunk_id="a#1", document_id="a", document_title="t", text="hello")
    assert plain.time_range is None  # doc/PDF paths leave it None

    timed = Chunk(
        chunk_id="b#1",
        document_id="b",
        document_title="lecture",
        text="bonjour",
        time_range=(12.0, 18.5),
    )
    reloaded = Chunk.model_validate_json(timed.model_dump_json())
    assert reloaded.time_range == (12.0, 18.5)


def test_chunk_legacy_payload_has_no_time_range() -> None:
    legacy = '{"chunk_id": "a#1", "document_id": "a", "document_title": "t", "text": "x"}'
    chunk = Chunk.model_validate_json(legacy)
    assert chunk.time_range is None
