"""Full-document synthesis — the `full` co-residence mode's answering path.

Distinct from the top-k RAG agent (`answering.answer_query`): instead of
retrieving a handful of chunks, this feeds WHOLE document(s) into the
orchestrator's large context window (the `full` mode raises `--max-model-len`
to ~24k; see ADR-0007) and asks for a long-form summary/synthesis.

This is the **baseline** path (ADR-0007 Part 4 of the plan): free-form output,
no per-claim verify gate yet — its purpose is to MEASURE how far context +
output detail can go at the full setting before we design the structured,
grounded version. It deliberately reuses `FinalResponse` so every surface
renders it like a normal answer.

Module-boundary note: `agents/` may not import `vault/`, so the caller (CLI /
MCP / webui — which all may read the vault) loads the document text and passes
it in as `SourceDoc`s. This module stays pure of vault I/O.
"""

from __future__ import annotations

import structlog
import ulid
from pydantic import BaseModel, Field, create_model

from memex.agents.answering import FinalResponse
from memex.core.errors import ModelCallError
from memex.models.client import complete_structured
from memex.observability import bind_run_context, clear_run_context
from memex.prompts import render_prompt

logger = structlog.get_logger(__name__)

# Worst-case chars-per-token for dense text. The summary's grammar `max_length`
# (chars) must be small enough that the model ALWAYS closes the JSON string
# before exhausting `max_tokens` (tokens) — otherwise the output is cut
# mid-string and fails schema validation. At ~2.5 chars/token a `max_length` of
# `(max_tokens - scaffold) * 2.5` chars can't exceed the token budget.
_CHARS_PER_TOKEN_FLOOR = 2.5
_JSON_SCAFFOLD_TOKENS = 96

# ~3.5 chars/token is a conservative English estimate; we UNDER-fill the window
# (vs the orchestrator's hard max-model-len) so a slightly-off heuristic can't
# trip a request-too-long rejection. The caller sizes `max_input_chars` from the
# active mode's context window minus the output budget.
_TRUNCATION_MARKER = "\n\n[… document truncated to fit the context window …]\n"


class SourceDoc(BaseModel):
    """One document's content for synthesis (the caller loads it from the vault)."""

    doc_id: str
    title: str
    text: str


class Synthesis(BaseModel):
    """Free-form long-form summary. The per-call schema (built in
    `synthesize_documents`) sizes `summary`'s `max_length` to what `max_tokens`
    can actually close; this static base carries the field shape for the type
    checker. (The STRUCTURED schema — sections/key-points/entities — is the
    deferred Phase-B design; this baseline emits prose.)"""

    summary: str = Field(max_length=16000)


def _bounded_synthesis(max_output_tokens: int) -> type[Synthesis]:
    """A per-call `Synthesis` whose `summary` `max_length` is sized so the model
    closes the JSON within `max_output_tokens` (see `_CHARS_PER_TOKEN_FLOOR`).
    `__base__=Synthesis` keeps `.summary` typed for callers."""
    max_chars = max(2000, int((max_output_tokens - _JSON_SCAFFOLD_TOKENS) * _CHARS_PER_TOKEN_FLOOR))
    return create_model(
        "Synthesis",
        __base__=Synthesis,
        summary=(str, Field(max_length=max_chars)),
    )


def _assemble(sources: list[SourceDoc], max_input_chars: int) -> tuple[str, bool]:
    """Join the docs in order as titled sections, truncated to `max_input_chars`.
    Returns `(assembled_text, truncated)`."""
    blocks = [f"## {d.title}\n\n{d.text}".strip() for d in sources]
    assembled = "\n\n---\n\n".join(blocks)
    if len(assembled) <= max_input_chars:
        return assembled, False
    return assembled[:max_input_chars] + _TRUNCATION_MARKER, True


async def synthesize_documents(
    sources: list[SourceDoc],
    *,
    instruction: str | None = None,
    max_input_chars: int = 80_000,
    max_output_tokens: int = 4096,
) -> FinalResponse:
    """Summarise whole document(s) in one long-context pass.

    `sources` are loaded by the caller (vault read). `instruction` optionally
    focuses the summary ("focus on the security controls"). `max_input_chars`
    bounds the assembled context (caller sizes it from the active mode's
    `--max-model-len`); `max_output_tokens` is the per-call output cap (raised
    well above the RAG default so the summary can be long).

    Returns a `FinalResponse` (`answered=True`, `summary` = the synthesis,
    `artifact_scope_doc_ids` = the source docs). Free-form, ungated — the
    grounding/structure layer is deferred (ADR-0007).
    """
    correlation_id = str(ulid.ULID())
    clear_run_context()
    bind_run_context(correlation_id, query_preview=(instruction or "full-document summary")[:80])
    log = logger.bind(node="synthesize", doc_ids=[d.doc_id for d in sources])
    try:
        if not sources:
            log.info("synthesize.empty")
            return FinalResponse(
                answered=False,
                refusal_reason="No documents to summarise.",
                correlation_id=correlation_id,
                tokens_used=0,
                nodes_traversed=1,
                regenerate_attempts=0,
            )

        assembled, truncated = _assemble(sources, max_input_chars)
        log.info(
            "synthesize.start",
            docs=len(sources),
            input_chars=len(assembled),
            truncated=truncated,
            max_output_tokens=max_output_tokens,
        )
        prompt = render_prompt(
            "synthesize",
            instruction=instruction or "",
            documents=assembled,
        )
        try:
            synthesis, tokens = await complete_structured(
                prompt=prompt,
                schema=_bounded_synthesis(max_output_tokens),
                max_tokens=max_output_tokens,
                prompt_tag="synthesize@v1",
            )
        except ModelCallError as e:
            # The dominant failure mode for single-shot synthesis of a large or
            # table-heavy doc: the model regurgitates instead of condensing, runs
            # past max_tokens, and the JSON string is cut off → invalid (vLLM
            # guided-JSON does NOT enforce string maxLength). Degrade cleanly
            # rather than crash — this boundary is exactly what the structured /
            # map-reduce synthesis design (forthcoming) addresses.
            log.warning("synthesize.output_failed", error=str(e)[:160])
            return FinalResponse(
                answered=False,
                refusal_reason=(
                    "The summary exceeded the output budget without validating — the "
                    "document is likely too large or table-heavy for single-shot "
                    "synthesis. Raise --max-tokens, narrow the input, or wait for "
                    "structured/map-reduce summarization."
                ),
                artifact_scope_doc_ids=[d.doc_id for d in sources],
                correlation_id=correlation_id,
                tokens_used=0,
                nodes_traversed=1,
                regenerate_attempts=0,
            )
        log.info(
            "synthesize.done",
            tokens_used=tokens,
            summary_chars=len(synthesis.summary),
        )
        return FinalResponse(
            answered=True,
            summary=synthesis.summary,
            artifact_scope_doc_ids=[d.doc_id for d in sources],
            correlation_id=correlation_id,
            tokens_used=tokens,
            nodes_traversed=1,
            regenerate_attempts=0,
        )
    finally:
        clear_run_context()
