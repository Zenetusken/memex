"""Parse stage — Docling primary, VLM fallback.

Routes each document by content type and per-page Docling confidence.
Low-confidence pages are escalated to the VLM in a single batched
call. Both backends produce `PageDecision` records recorded in the
manifest. The Docling worker runs out-of-process under a seccomp
network-egress sandbox (see `memex.parse.sandbox`).

See GUIDELINES.md Part II "The pipeline" and IMPLEMENTATION-PLAN §1.3.
"""

from memex.parse.asr_backend import (
    ASRSegment,
    ASRTranscriptionError,
    ASRUnavailable,
    transcribe_audio,
)
from memex.parse.asr_cache import ASRTranscriptionCache
from memex.parse.chart_ocr_cache import ChartOCRCache
from memex.parse.docling_backend import (
    DoclingConversion,
    DoclingCrashed,
    DoclingPageOutput,
    DoclingTimeout,
    DoclingUnavailable,
    SandboxLoadFailed,
)
from memex.parse.keyframe_ocr import (
    KeyframeOCRError,
    ocr_frames_for_chunks,
)
from memex.parse.keyframe_ocr_cache import KeyframeOCRCache
from memex.parse.office_convert import (
    OFFICE_SUFFIXES,
    OfficeConversionError,
    convert_to_pdf,
)
from memex.parse.pipeline import (
    AUDIO_SUFFIXES,
    MEDIA_SUFFIXES,
    VIDEO_SUFFIXES,
    ParseResult,
    derive_title,
    parse_document,
    pause_vllm_for_gpu,
    reset_docling_breaker,
    reset_pymupdf_breaker,
)
from memex.parse.pymupdf_backend import (
    PdfSignals,
    PyMuPDFConversion,
    PyMuPDFCrashed,
    PyMuPDFPageOutput,
    PyMuPDFTimeout,
    PyMuPDFUnavailable,
)
from memex.parse.sandbox import (
    SandboxStatus,
    enable_network_block,
)
from memex.parse.vlm_backend import VLMUnavailable, transcribe_images
from memex.parse.vlm_cache import VLMTranscriptionCache

__all__ = [
    "AUDIO_SUFFIXES",
    "MEDIA_SUFFIXES",
    "OFFICE_SUFFIXES",
    "VIDEO_SUFFIXES",
    "ASRSegment",
    "ASRTranscriptionCache",
    "ASRTranscriptionError",
    "ASRUnavailable",
    "ChartOCRCache",
    "DoclingConversion",
    "DoclingCrashed",
    "DoclingPageOutput",
    "DoclingTimeout",
    "DoclingUnavailable",
    "KeyframeOCRCache",
    "KeyframeOCRError",
    "OfficeConversionError",
    "ParseResult",
    "PdfSignals",
    "PyMuPDFConversion",
    "PyMuPDFCrashed",
    "PyMuPDFPageOutput",
    "PyMuPDFTimeout",
    "PyMuPDFUnavailable",
    "SandboxLoadFailed",
    "SandboxStatus",
    "VLMTranscriptionCache",
    "VLMUnavailable",
    "convert_to_pdf",
    "derive_title",
    "enable_network_block",
    "ocr_frames_for_chunks",
    "parse_document",
    "pause_vllm_for_gpu",
    "reset_docling_breaker",
    "reset_pymupdf_breaker",
    "transcribe_audio",
    "transcribe_images",
]
