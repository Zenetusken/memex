"""Parse stage — Docling primary, VLM fallback.

Routes each document by content type and per-page Docling confidence.
Low-confidence pages are escalated to the VLM in a single batched
call. Both backends produce `PageDecision` records recorded in the
manifest. The Docling worker runs out-of-process under a seccomp
network-egress sandbox (see `memex.parse.sandbox`).

See GUIDELINES.md Part II "The pipeline" and IMPLEMENTATION-PLAN §1.3.
"""

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
from memex.parse.office_convert import (
    OFFICE_SUFFIXES,
    OfficeConversionError,
    convert_to_pdf,
)
from memex.parse.pipeline import (
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
from memex.parse.vlm_backend import VLMUnavailable
from memex.parse.vlm_cache import VLMTranscriptionCache

__all__ = [
    "OFFICE_SUFFIXES",
    "ASRTranscriptionCache",
    "ChartOCRCache",
    "DoclingConversion",
    "DoclingCrashed",
    "DoclingPageOutput",
    "DoclingTimeout",
    "DoclingUnavailable",
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
    "parse_document",
    "pause_vllm_for_gpu",
    "reset_docling_breaker",
    "reset_pymupdf_breaker",
]
