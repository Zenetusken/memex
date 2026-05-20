"""Ingest stage — file validation, doc-id assignment, source copy.

Validates input files (magic numbers, size, format), assigns a stable
document ID (content hash + namespace), copies originals to
`vault/documents/{id}/source.{ext}`, writes the initial manifest entry.
The live filesystem watcher (`run_watcher` + `default_reaction`) runs
the re-enrich + re-index loop on confirmed user edits.

See GUIDELINES.md Part II "The pipeline" and IMPLEMENTATION-PLAN §1.2.
"""

from memex.ingest.pipeline import (
    IngestRequest,
    IngestResult,
    ingest_directory,
    ingest_file,
    ingest_markdown_passthrough,
)
from memex.ingest.validation import ValidationResult, validate_file
from memex.ingest.watcher import EditNotice, default_reaction, run_watcher

__all__ = [
    "EditNotice",
    "IngestRequest",
    "IngestResult",
    "ValidationResult",
    "default_reaction",
    "ingest_directory",
    "ingest_file",
    "ingest_markdown_passthrough",
    "run_watcher",
    "validate_file",
]
