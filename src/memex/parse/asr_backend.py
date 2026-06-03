"""Audio transcription backend (ADR-0017; spec docs/specs/audio-asr-route.md §6).

Produces time-ordered `ASRSegment`s (text + GLOBAL timestamps) from an audio file. v1 ships the
in-process `faster_whisper` backend (the recommended default — CTranslate2, native VAD +
long-form + reproducible greedy decode); `vllm`/`transformers` are the deferred A/B
alternatives (they raise `ASRUnavailable`). The model load + transcribe sits behind a lazy
import + the `_run_faster_whisper` seam, so the route is unit-testable with no GPU/deps.

Per-file results are cached (`parse/asr_cache.py`) for re-parse reproducibility — faster-whisper
does its own internal VAD + long-form chunking, so the whole file is one cache unit
(`chunk_index=0`). The per-segment normalization (`core/text.normalize_transcript_text`) is
applied AFTER the cache (raw stays cached), gated by `ParseSettings.asr_normalize`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import structlog
from pydantic import BaseModel

from memex.core.config import MemexSettings, get_settings
from memex.core.errors import MemexError
from memex.core.text import normalize_transcript_text
from memex.parse.asr_cache import ASRTranscriptionCache, build_asr_cache_key, cfg_sha8

logger = structlog.get_logger(__name__)

# faster-whisper does its own VAD + long-form chunking, so the whole file is ONE cache unit.
_FULL_FILE_CHUNK_INDEX = 0


class ASRUnavailable(MemexError):
    """The configured ASR backend/model can't be used — not configured, not installed, or a
    backend deferred to a later increment."""


class ASRTranscriptionError(MemexError):
    """Transcription failed for an audio source."""


class ASRSegment(BaseModel):
    """One transcript segment as produced by the ASR backend — a TRANSIENT, text-carrying type,
    distinct from the manifest `TranscriptSegment` (which has no `text`; the text lives in the
    `.md`, addressed by char-spans — see spec §8). `start_s`/`end_s` are GLOBAL seconds vs the
    whole file; a failed segment carries `confidence=0.0` + the reason in `rationale`."""

    text: str
    start_s: float
    end_s: float
    language: str = ""
    confidence: float = 1.0
    rationale: str = ""


def _audio_sha256(source: Path) -> str:
    """Content hash of the audio file — the cache key's input-derived prefix."""
    h = hashlib.sha256()
    with open(source, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _decoding_cfg(settings: MemexSettings) -> dict[str, object]:
    """The decoding params that change the transcription → the cache `cfg` (so a change is a
    clean miss, never a stale replay)."""
    p = settings.parse
    return {
        "backend": settings.models.asr_backend,
        "model": settings.models.asr,
        "beam_size": p.asr_beam_size,
        "language": p.asr_language or "auto",
        "vad_filter": p.asr_vad_filter,
    }


def _run_faster_whisper(
    source: Path,
    *,
    model_id: str,
    device: str,
    beam_size: int,
    language: str | None,
    vad_filter: bool,
) -> list[ASRSegment]:
    """Load faster-whisper (lazy import) + transcribe the whole file. The monkeypatchable seam —
    tests fake this to avoid the GPU/dep. Greedy (beam_size 1) is reproducible for a fixed
    input + hardware + library version; the cache freezes the choice across versions."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:  # the [audio] extra isn't installed
        raise ASRUnavailable(
            "faster-whisper is not installed; run `uv sync --extra audio`",
            context={"backend": "faster_whisper"},
        ) from e

    compute_type = "float16" if device == "cuda" else "int8"
    try:
        model = WhisperModel(model_id, device=device, compute_type=compute_type)
        segments, info = model.transcribe(
            str(source), beam_size=beam_size, language=language, vad_filter=vad_filter
        )
        lang = info.language or (language or "")
        out: list[ASRSegment] = []
        for seg in segments:  # faster-whisper streams Segment objects lazily
            text = (seg.text or "").strip()
            if text:
                out.append(
                    ASRSegment(
                        text=text, start_s=float(seg.start), end_s=float(seg.end), language=lang
                    )
                )
        return out
    except (ASRUnavailable, asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:  # third-party ctranslate2/av errors aren't a typed family (cancellation re-raised above)
        raise ASRTranscriptionError(
            "faster-whisper transcription failed",
            context={"source": str(source), "model": model_id, "error": str(e)},
        ) from e


async def _transcribe_faster_whisper(
    source: Path, *, cache: ASRTranscriptionCache | None, refresh: bool, settings: MemexSettings
) -> list[ASRSegment]:
    """The faster-whisper path with the per-file cache: a hit replays the stored segments (no
    model load); a miss runs the model once and stores the result."""
    model_id = settings.models.asr
    if model_id is None:  # guaranteed by transcribe_audio; re-checked for the type-narrow
        raise ASRUnavailable("no ASR model configured", context={"source": str(source)})
    audio_sha = await asyncio.to_thread(_audio_sha256, source)
    cfg8 = cfg_sha8(_decoding_cfg(settings))
    key = build_asr_cache_key(
        audio_sha256=audio_sha, chunk_index=_FULL_FILE_CHUNK_INDEX, model=model_id, cfg=cfg8
    )
    log = logger.bind(component="asr.faster_whisper", model=model_id)
    if cache is not None:
        if refresh:
            await cache.delete_by_audio(audio_sha)
        else:
            hit = await cache.get(key)
            if hit is not None:
                log.info("asr.cache_hit")
                return [ASRSegment.model_validate(d) for d in json.loads(hit)]

    log.info("asr.transcribe.start", device=settings.parse.asr_device)
    raw = await asyncio.to_thread(
        _run_faster_whisper,
        source,
        model_id=model_id,
        device=settings.parse.asr_device,
        beam_size=settings.parse.asr_beam_size,
        language=settings.parse.asr_language,
        vad_filter=settings.parse.asr_vad_filter,
    )
    log.info("asr.transcribe.done", segments=len(raw))
    if cache is not None and raw:  # don't freeze an empty result — retry next parse
        payload = json.dumps([s.model_dump() for s in raw])
        await cache.put(
            key,
            audio_sha256=audio_sha,
            chunk_index=_FULL_FILE_CHUNK_INDEX,
            asr_model=model_id,
            cfg_sha8=cfg8,
            segments_json=payload,
        )
    return raw


async def transcribe_audio(
    *, source: Path, cache: ASRTranscriptionCache | None = None, refresh: bool = False
) -> list[ASRSegment]:
    """Transcribe an audio file to time-ordered `ASRSegment`s (GLOBAL timestamps), applying the
    deterministic per-segment normalization (gated by `ParseSettings.asr_normalize`). v1 routes
    `asr_backend="faster_whisper"`; the other backends raise `ASRUnavailable` (deferred A/B)."""
    settings = get_settings()
    if settings.models.asr is None:
        raise ASRUnavailable(
            "no ASR model configured — set models.asr (e.g. a faster-whisper build)",
            context={"source": str(source)},
        )
    backend = settings.models.asr_backend
    if backend == "faster_whisper":
        raw = await _transcribe_faster_whisper(
            source, cache=cache, refresh=refresh, settings=settings
        )
    else:
        raise ASRUnavailable(
            f"asr_backend={backend!r} is not implemented in v1; use faster_whisper",
            context={"backend": backend, "source": str(source)},
        )

    if not settings.parse.asr_normalize:
        return raw
    normed: list[ASRSegment] = []
    for s in raw:
        text = normalize_transcript_text(s.text)
        if text:  # a segment emptied by normalization (all-filler) is dropped
            normed.append(s.model_copy(update={"text": text}))
    return normed
