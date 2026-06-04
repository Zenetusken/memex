"""Memex error hierarchy.

Errors are typed and structured: every exception inherits from `MemexError`
and carries a `context` dict that propagates to the user as part of the
error message ("an explanation with options"). Recoverable errors mark
the pipeline state for resumption.

See GUIDELINES.md Part VI "Error handling".
"""

from __future__ import annotations


class MemexError(Exception):
    """Base for all Memex-raised exceptions.

    Subclasses should be specific so callers can pattern-match. The
    `context` dict carries fields that help the user understand and act
    on the error (file paths, model names, confidence scores, etc.).
    """

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, object] | None = None,
        recoverable: bool = False,
    ) -> None:
        super().__init__(message)
        self.context: dict[str, object] = context or {}
        self.recoverable = recoverable


class InsufficientVRAMError(MemexError):
    """The requested model does not fit alongside currently resident models."""


class VRAMExhausted(MemexError):
    """A GPU-exclusive op (the parse-time VLM serve / chart-OCR) can't fit even after freeing the
    retrieval models — the GPU is held by another process. Carries the live free/total + the floor it
    needed + a best-effort holder list (which process holds the VRAM), so the user sees the actionable
    cause instead of vLLM's cryptic 'No available memory for the cache blocks'."""


class ParseConfidenceTooLow(MemexError):
    """Docling and the VLM both produced low-confidence output for a page."""


class RetrievalEmpty(MemexError):
    """No candidates met the relevance threshold for this query."""


class VaultIntegrityError(MemexError):
    """A vault invariant was violated (doc_id collision, manifest mismatch, ...)."""


class IndexOutOfSync(MemexError):
    """An index disagrees with the canonical Markdown; rebuild required."""


class ConfigurationError(MemexError):
    """A required configuration value is missing, invalid, or contradictory.

    Raised at startup by `configure_*` helpers; never at runtime.
    """


class StaleDocumentError(MemexError):
    """Optimistic-concurrency check failed on a vault write.

    Raised by `vault.store.write_document(..., expected_sha=...)` when
    the on-disk content has changed since the caller loaded it. The
    web UI's `/review` handler catches this and renders a conflict
    panel with a diff and "discard / overwrite" buttons.

    Context: `doc_id`, `expected_sha`, `current_sha`, `current_body`.
    The full current on-disk body is included so the UI can render
    the diff without a second vault read.
    """


class AnswerStateInvariantError(MemexError):
    """An agent node was reached with an `AnswerState` that violates its
    preconditions (e.g. `verify` called without a draft, `compose` called
    without a verification result). Indicates a graph wiring bug, not a
    user error.
    """


class ModelCallError(MemexError):
    """A model call (vLLM completion, embed, rerank) failed or returned
    malformed output that could not be coerced to the requested schema.
    """


class ScopeSetError(MemexError):
    """A saved scope set is invalid — empty/over-long name, no documents,
    or a duplicate. Raised by `core.scope_sets` on a malformed mutation; the
    CLI / webui / MCP surfaces render it as a user-facing validation message.
    """
