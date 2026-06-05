"""Lecture-video KEYFRAME OCR for the companion-merge accuracy lever (ADR-0018 §13).

The companion-merge v1 aligns a lecture transcript to its slide deck by cosine of
TRANSCRIPT-TEXT vs deck-page-text — MaViLS's *weakest* modality (F1 ≈ 0.53). When the
lecture is a SCREEN RECORDING that shows the projected slides (the CR350 ZOOM classes), the
video frames carry MaViLS's *strong* signal (frame-text F1 ≈ 0.76): a frame literally shows
which slide is on screen, so OCR-ing it and matching to the deck is near-exact.

This module is the parse-domain half: decode one video frame per transcript chunk (PyAV) at
the chunk's time-range midpoint, OCR it via the VLM (`vlm_backend.transcribe_images`), and
return `{chunk_id: frame_ocr_text}`. The EMBED + cosine-match to the deck and the
keyframe-primary-with-transcript-fallback live in `index/companion.py` (which takes these
texts as input — `index/` may not import `parse/`); the CLI (`link-slides --use-video`)
wires the two. The frame OCR is cached (content-addressed, `KeyframeOCRCache`) so re-runs of
the derived alignment sidecar stay byte-stable.

The slide PDFs remain the AUTHORITATIVE slide text — this OCR is the *alignment signal*
(which slide, when), not a slide-content replacement. HARD-gate-neutral by construction (it
feeds the derived alignment sidecar, never the grounding path).
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from memex.core.config import get_settings
from memex.core.errors import MemexError
from memex.parse.keyframe_ocr_cache import KeyframeOCRCache
from memex.parse.vlm_backend import transcribe_images

if TYPE_CHECKING:
    from PIL import Image

logger = structlog.get_logger(__name__)

# A lecture frame is a slide UNDER desktop chrome / an overlaid demo window / a webcam PiP
# (the CR350 recordings show, e.g., a DNS slide with a live Wireshark capture on top). The
# generic page-transcribe prompt would dump the dominant demo table; this one asks the VLM to
# isolate the MAIN SLIDE so the OCR text matches the deck page, not the demo. This is the
# v1 baseline; the K3 gold set (incl. demo-heavy frames) measures whether it lands the argmax.
_KEYFRAME_PROMPT = (
    "This is a frame from a lecture screen-recording that shows a presentation SLIDE "
    "(often with a live demo, application window, terminal, or webcam thumbnail overlaid "
    "on top of it). Transcribe ONLY the MAIN PRESENTATION SLIDE's content as clean "
    "Markdown — its title, headings, bullet points, and any diagram or figure labels. "
    "IGNORE the desktop taskbar, window chrome, webcam thumbnail, and any overlaid "
    "application / demo / terminal window that is not the slide itself. If no slide is "
    "visible (a full-screen demo or desktop), output nothing. Output ONLY the slide's "
    "Markdown — no preface, no commentary."
)

# Don't cache / return a near-empty OCR — the VLM occasionally punts a hard frame (or a frame
# with no slide), and leaving it uncached lets a later run retry. Mirrors vlm_backend's guard.
_MIN_OCR_CHARS = 20


class KeyframeOCRError(MemexError):
    """Keyframe OCR can't run — PyAV is missing, or the video source is absent. Not
    recoverable from here (a user-fix / gate prerequisite)."""


class KeyframeDecodeError(MemexError):
    """A single frame couldn't be decoded at the requested timestamp. Recoverable: the caller
    omits that chunk's keyframe (the alignment falls back to its transcript-text signal)."""


def _sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a (possibly large, ~GB) video file — content-addresses the cache
    without loading the whole file into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _cache_key(video_sha256: str, time_s: float, model: str, prompt_sha8: str) -> str:
    """`(video, timestamp_ms, model, prompt)` — see `keyframe_ocr_cache` for the rationale
    (timestamp, not frame-PNG bytes)."""
    return f"{video_sha256}:{round(time_s * 1000)}:m={model}:p={prompt_sha8}"


def _decode_frame(video_path: Path, time_s: float) -> Image.Image:
    """Decode the video frame at (or just after) `time_s` to a PIL image via PyAV.

    Seeks to the nearest keyframe at/before `time_s` (fast — one GOP), then decodes forward
    to the first frame whose presentation time reaches `time_s` (precise). The MONKEYPATCH
    SEAM: tests fake this so the orchestration runs with no video / no `av`. PyAV is a
    genuinely-untyped C-extension boundary → routed through `Any` (the transformers-`Any`
    pattern in `vlm_backend`)."""
    try:
        import av  # heavy optional dep, imported lazily (ships with the [audio] extra)
    except ImportError as e:
        raise KeyframeOCRError(
            "PyAV (`av`) is required to decode video keyframes; it ships with the [audio] "
            "extra (faster-whisper). Install with `uv sync --extra audio`.",
            context={"underlying": str(e)},
        ) from e

    container: Any = None
    try:
        container = av.open(str(video_path))
        stream: Any = container.streams.video[0]
        time_base = stream.time_base
        if time_base is not None:
            container.seek(int(time_s / float(time_base)), backward=True, any_frame=False, stream=stream)
        last: Any = None
        for frame in container.decode(stream):
            last = frame
            ftime = frame.time
            if ftime is not None and float(ftime) >= time_s:
                return frame.to_image()
        if last is not None:  # ran off the end (time_s past the last frame) → the last frame
            return last.to_image()
        raise KeyframeDecodeError(
            "no video frame decoded at timestamp",
            context={"video": str(video_path), "time_s": time_s},
        )
    except (KeyframeDecodeError, KeyframeOCRError):
        raise
    except Exception as e:  # PyAV (av.error.*) / OSError → a recoverable per-frame decode miss
        if type(e).__module__.split(".")[0] == "av" or isinstance(e, OSError):
            raise KeyframeDecodeError(
                "video frame decode failed",
                context={"video": str(video_path), "time_s": time_s, "error": str(e)},
            ) from e
        raise
    finally:
        if container is not None:
            container.close()


async def _ocr_images(images: list[Image.Image]) -> list[str]:
    """OCR seam (monkeypatched in tests) — delegates to the VLM with the keyframe prompt."""
    return await transcribe_images(images, prompt=_KEYFRAME_PROMPT)


def _avg_hash(image: Image.Image, size: int = 8) -> int:
    """Average-hash (aHash): a `size×size` grayscale perceptual fingerprint — bit i is set iff
    pixel i exceeds the frame's mean luminance. Near-identical frames (the SAME slide held on
    screen, under jittering demo/webcam chrome) hash within a small Hamming distance; a SLIDE
    CHANGE flips many bits. Pure PIL + stdlib → deterministic + cheap (~1 ms/frame)."""
    from PIL import Image as _PILImage  # lazy — the module stays importable without the [parse] PIL

    small = image.convert("L").resize((size, size), _PILImage.Resampling.LANCZOS)
    pixels = list(small.tobytes())  # "L" mode → one byte (int) per pixel — typed + not deprecated
    mean = sum(pixels) / len(pixels)
    bits = 0
    for i, p in enumerate(pixels):
        if p > mean:
            bits |= 1 << i
    return bits


def _hamming_distance(a: int, b: int) -> int:
    """Bit-difference between two average-hashes (0 = identical, 64 = inverted for an 8×8 hash)."""
    return (a ^ b).bit_count()


def _dedup_plan(images: list[Image.Image], *, hamming: int) -> list[int]:
    """Sliding-window perceptual dedup over decoded frames IN TIME ORDER.

    Returns `source[i]` = the index whose OCR result frame `i` should REUSE: `i` itself when the
    frame is kept (OCR'd fresh), or the index of the previous KEPT frame when `i` is within
    `hamming` of it (the slide hasn't changed). Comparing to the previous KEPT frame (not the
    immediate predecessor) avoids slow drift across a long slide-dwell from accumulating past the
    threshold."""
    source: list[int] = []
    hashes = [_avg_hash(im) for im in images]
    last_kept = -1
    for i in range(len(images)):
        if last_kept >= 0 and _hamming_distance(hashes[i], hashes[last_kept]) <= hamming:
            source.append(last_kept)  # reuse the kept frame's OCR — no VLM call for `i`
        else:
            source.append(i)
            last_kept = i
    return source


async def ocr_frames_for_chunks(
    video_path: Path,
    frames: list[tuple[str, float]],
    *,
    cache: KeyframeOCRCache | None = None,
) -> dict[str, str]:
    """Decode + OCR one video frame per `(chunk_id, time_s)` → `{chunk_id: ocr_text}`.

    Cache-aware: a chunk whose `(video, timestamp, model, prompt)` key is already cached is
    served without a decode or a VLM call. Misses are decoded (PyAV) and OCR'd in ONE VLM
    batch (the caller holds `pause_vllm_for_gpu()`); the chosen text is cached. A chunk whose
    frame fails to decode, fails to OCR, or yields near-empty text is OMITTED from the result
    (NO key) — the caller's alignment then falls back to that chunk's transcript-text signal.

    Raises `KeyframeOCRError` if the video source is missing (the CLI's `has_video_source`
    gate should preclude this) or PyAV is absent."""
    if not frames:
        return {}
    if not video_path.is_file():
        raise KeyframeOCRError(
            "video source not found for keyframe OCR",
            context={"video": str(video_path)},
        )

    settings = get_settings()
    model_id = settings.models.vlm
    prompt_sha8 = hashlib.sha256(_KEYFRAME_PROMPT.encode()).hexdigest()[:8]
    out: dict[str, str] = {}

    video_sha256 = ""
    keys: dict[str, str] = {}
    if cache is not None:
        video_sha256 = await asyncio.to_thread(_sha256_file, video_path)

    # First pass: serve cache hits, collect misses (no decode / no GPU yet).
    misses: list[tuple[str, float]] = []
    for chunk_id, time_s in frames:
        if cache is not None:
            key = _cache_key(video_sha256, time_s, model_id, prompt_sha8)
            keys[chunk_id] = key
            hit = await cache.get(key)
            if hit is not None:
                logger.info("keyframe.cache_hit", chunk_id=chunk_id, time_s=time_s)
                out[chunk_id] = hit
                continue
        misses.append((chunk_id, time_s))

    if not misses:
        return out

    # Second pass: decode the miss frames, then OCR them in one batch.
    images: list[Image.Image] = []
    decoded: list[tuple[str, float]] = []  # (chunk_id, time_s) parallel to `images`
    for chunk_id, time_s in misses:
        try:
            image = await asyncio.to_thread(_decode_frame, video_path, time_s)
        except KeyframeDecodeError as e:
            logger.warning("keyframe.decode_skipped", chunk_id=chunk_id, time_s=time_s, error=str(e))
            continue
        images.append(image)
        decoded.append((chunk_id, time_s))

    if not images:
        return out

    # Pass 1.5: perceptual dedup — OCR only the UNIQUE frames (a held slide's repeats reuse the
    # kept frame's text), then map each frame's text back via `source`. Each chunk is still cached
    # under its own key below, so a re-run replays without re-deciding the dedup.
    if settings.parse.keyframe_dedup_enabled:
        source = _dedup_plan(images, hamming=settings.parse.keyframe_dedup_hamming)
    else:
        source = list(range(len(images)))
    unique_indices = [i for i in range(len(images)) if source[i] == i]
    unique_texts = await _ocr_images([images[i] for i in unique_indices])
    text_by_index: dict[int, str] = dict(zip(unique_indices, unique_texts, strict=True))
    if len(unique_indices) < len(images):
        logger.info(
            "keyframe.dedup",
            frames=len(images),
            ocr_calls=len(unique_indices),
            deduped=len(images) - len(unique_indices),
        )

    for i, (chunk_id, time_s) in enumerate(decoded):
        cleaned = text_by_index[source[i]].strip()
        if len(cleaned) < _MIN_OCR_CHARS:
            continue  # punted / no-slide frame → omit → transcript-text fallback
        out[chunk_id] = cleaned
        if cache is not None:
            await cache.put(
                keys[chunk_id],
                video_sha256=video_sha256,
                timestamp_ms=round(time_s * 1000),
                vlm_model=model_id,
                prompt_sha8=prompt_sha8,
                markdown=cleaned,
            )
    return out
