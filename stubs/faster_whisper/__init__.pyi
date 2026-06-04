# Minimal stub for faster_whisper — covers only the surface `parse/asr_backend.py` uses:
# `WhisperModel(model_id, device=, compute_type=)`, `model.transcribe(audio, beam_size=,
# language=, vad_filter=) -> (segments, info)`, `Segment.text/start/end`, `info.language`.
# faster-whisper ships no `py.typed`.
from collections.abc import Iterable

class Segment:
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float

class TranscriptionInfo:
    language: str
    language_probability: float
    duration: float

class WhisperModel:
    def __init__(
        self,
        model_size_or_path: str,
        *,
        device: str = ...,
        compute_type: str = ...,
        **kwargs: object,
    ) -> None: ...
    def transcribe(
        self,
        audio: str,
        *,
        beam_size: int = ...,
        language: str | None = ...,
        vad_filter: bool = ...,
        **kwargs: object,
    ) -> tuple[Iterable[Segment], TranscriptionInfo]: ...
