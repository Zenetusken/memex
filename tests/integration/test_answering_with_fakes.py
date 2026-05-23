"""End-to-end test of the answering agent with all I/O faked.

Validates the LangGraph state machine, structured-output handling,
routing, refusal, and regeneration without touching a real vLLM,
Langfuse, LanceDB, or the model registry. Runs in well under a second
on CPU.

This is the regression net IMPLEMENTATION-PLAN.md §2.6 calls for. Every
subsequent module change should keep it green.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from memex.agents.answering import (
    AnswerState,
    Chunk,
    CitedClaim,
    DraftAnswer,
    SufficiencyAssessment,
    VerificationResult,
    answer_query,
    reset_compiled_graph,
)


# ----- Fixtures: fake retrieve + fake model + reset graph cache -----


@pytest.fixture(autouse=True)
def _fresh_graph() -> Iterator[None]:
    """Tests that monkey-patch nodes need a fresh compile."""
    reset_compiled_graph()
    yield
    reset_compiled_graph()


@pytest.fixture
def fake_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="c1",
            document_id="d1",
            document_title="Smith 2024",
            text="Reflexivity in research design means the researcher's "
            "presence shapes the data they collect.",
            page=3,
            score=0.9,
        ),
        Chunk(
            chunk_id="c2",
            document_id="d1",
            document_title="Smith 2024",
            text="Smith argues that reflexivity is constitutive rather "
            "than corrective.",
            page=4,
            score=0.85,
        ),
    ]


@pytest.fixture
def patch_retrieve(
    monkeypatch: pytest.MonkeyPatch, fake_chunks: list[Chunk]
) -> None:
    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return list(fake_chunks)

    async def _rerank(
        query: str, candidates: list[Chunk], top_k: int = 10
    ) -> list[Chunk]:
        return list(candidates[:top_k])

    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)


@pytest.fixture
def patch_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the prompt loader with a no-op renderer; the fake LLM
    doesn't read the prompt anyway."""
    monkeypatch.setattr(
        "memex.agents.answering.render_prompt",
        lambda name, **_kw: f"[fake {name} prompt]",
    )


class FakeLLM:
    """Canned responses by (prompt-fragment, schema) pairs."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, type]] = []
        self._responses: dict[tuple[str, type], Any] = {}

    def respond(self, prompt_fragment: str, schema: type, value: Any) -> None:
        self._responses[(prompt_fragment, schema)] = value

    async def __call__(
        self,
        *,
        prompt: str | list[dict[str, str]],
        schema: type,
        **_kw: object,
    ) -> tuple[Any, int]:
        # `**_kw` absorbs forward-compatible kwargs the real
        # `complete_structured` accepts (e.g. `prompt_tag`, `model`,
        # `temperature`) without changing the fake's behaviour.
        # `prompt` may be a single string (legacy callers) OR a list
        # of OpenAI-style message dicts (callers that use
        # `render_messages` for the system/user split). Concatenate
        # the message contents for fragment matching so a single
        # `respond(...)` call works against either shape.
        prompt_text = (
            prompt
            if isinstance(prompt, str)
            else "\n".join(m["content"] for m in prompt)
        )
        self.calls.append((prompt_text, schema))
        for (frag, sch), value in self._responses.items():
            # Match by class identity OR by class name. The latter lets a
            # dynamically-constructed subclass (e.g. the bounded
            # `VerificationResult` built at runtime in the verify node
            # with `max_length=len(claims)` to constrain xgrammar) match
            # against a canned response keyed on the logical type. The
            # bounded variant retains `__name__ == "VerificationResult"`.
            if (sch is schema or sch.__name__ == schema.__name__) and frag in prompt_text:
                return value, 10
        raise AssertionError(
            f"no canned response for ({prompt_text!r}, {schema.__name__})"
        )


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    llm = FakeLLM()
    monkeypatch.setattr(
        "memex.agents.answering.complete_structured", llm
    )
    return llm


# ----- The actual tests -----


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_retrieve", "patch_prompt")
async def test_happy_path_returns_grounded_answer(fake_llm: FakeLLM) -> None:
    fake_llm.respond(
        "assess_sufficiency",
        SufficiencyAssessment,
        SufficiencyAssessment(sufficient=True, reason="Two on-point chunks"),
    )
    fake_llm.respond(
        "answer",
        DraftAnswer,
        DraftAnswer(
            summary="Smith treats reflexivity as constitutive.",
            claims=[
                CitedClaim(
                    claim="Reflexivity is the researcher shaping the data.",
                    source_chunk_id="c1",
                    confidence="high",
                ),
                CitedClaim(
                    claim="Smith argues reflexivity is constitutive.",
                    source_chunk_id="c2",
                    confidence="high",
                ),
            ],
        ),
    )
    fake_llm.respond(
        "verify_grounding",
        VerificationResult,
        VerificationResult(grounded=[0, 1], ungrounded=[]),
    )

    response = await answer_query("What does Smith say about reflexivity?")

    assert response.answered is True
    assert response.summary is not None
    assert len(response.claims) == 2
    assert {c.source_chunk_id for c in response.claims} == {"c1", "c2"}
    assert response.refusal_reason is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_retrieve", "patch_prompt")
async def test_insufficiency_refuses_without_drafting(
    fake_llm: FakeLLM,
) -> None:
    fake_llm.respond(
        "assess_sufficiency",
        SufficiencyAssessment,
        SufficiencyAssessment(
            sufficient=False,
            reason="Chunks discuss the topic but not the specific question.",
        ),
    )

    response = await answer_query("What is the etymology of reflexivity?")

    assert response.answered is False
    assert "specific question" in (response.refusal_reason or "")
    assert response.claims == []
    # Critical: no answer/verify call happened
    schemas_called = {schema for (_p, schema) in fake_llm.calls}
    assert DraftAnswer not in schemas_called
    assert VerificationResult not in schemas_called


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_retrieve", "patch_prompt")
async def test_empty_draft_short_circuits_verify(
    fake_llm: FakeLLM,
) -> None:
    """An answer-stage refusal — `DraftAnswer.claims == []` — must short-
    circuit the verify node instead of calling the model with an empty
    `Draft (indexed):` section.

    Why: `VerificationResult.grounded`/`ungrounded` are unbounded
    `list[int]`. With xgrammar's strict JSON-schema mode and a prompt
    that has no claims to bound the indices, the model can degenerate
    into a runaway integer emission (`grounded: [0, 1, 2, ...]+`) that
    hits `max_tokens` with `finish_reason: length` → schema validation
    fails → ModelCallError. The short-circuit fixes the pathology AND
    saves a useless round-trip — an empty draft is the refusal signal
    from the answer stage; downstream `compose` routes to refuse.
    """
    fake_llm.respond(
        "assess_sufficiency",
        SufficiencyAssessment,
        SufficiencyAssessment(sufficient=True, reason="ok"),
    )
    # Answer node legitimately returns zero claims — the literal-
    # presence rule in answer/v2 fires for queries whose answer
    # isn't in the chunks.
    fake_llm.respond(
        "answer",
        DraftAnswer,
        DraftAnswer(summary="No literal answer in chunks.", claims=[]),
    )

    response = await answer_query("What is the FP128 energy cost?")

    assert response.answered is False
    assert response.claims == []
    schemas_called = {schema for (_p, schema) in fake_llm.calls}
    assert DraftAnswer in schemas_called  # answer node DID fire
    assert VerificationResult not in schemas_called  # verify DIDN'T


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_retrieve", "patch_prompt")
async def test_ungrounded_triggers_regeneration_then_succeeds(
    fake_llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm.respond(
        "assess_sufficiency",
        SufficiencyAssessment,
        SufficiencyAssessment(sufficient=True, reason="ok"),
    )

    # First draft has one ungrounded claim; second draft is clean.
    drafts = iter(
        [
            DraftAnswer(
                summary="Smith argues reflexivity is constitutive.",
                claims=[
                    CitedClaim(
                        claim="Reflexivity shapes the data.",
                        source_chunk_id="c1",
                        confidence="high",
                    ),
                    CitedClaim(
                        claim="Reflexivity was invented by Bourdieu.",
                        source_chunk_id="c2",
                        confidence="medium",
                    ),
                ],
            ),
            DraftAnswer(
                summary="Smith argues reflexivity is constitutive.",
                claims=[
                    CitedClaim(
                        claim="Reflexivity shapes the data.",
                        source_chunk_id="c1",
                        confidence="high",
                    ),
                ],
            ),
        ]
    )
    verdicts = iter(
        [
            VerificationResult(
                grounded=[0],
                ungrounded=[1],
                ungrounded_reasons=["c2 doesn't mention Bourdieu"],
            ),
            VerificationResult(grounded=[0], ungrounded=[]),
        ]
    )

    async def fake_call(
        *, prompt: str, schema: type, **_kw: object
    ) -> tuple[Any, int]:
        fake_llm.calls.append((prompt, schema))
        if schema is SufficiencyAssessment:
            return fake_llm._responses[("assess_sufficiency", schema)], 8
        if schema is DraftAnswer:
            return next(drafts), 30
        if schema is VerificationResult or schema.__name__ == VerificationResult.__name__:
            return next(verdicts), 15
        raise AssertionError(f"unexpected schema {schema}")

    # Use monkeypatch so the patch is restored at test teardown — direct
    # module-attribute assignment used to leak into later tests in the
    # same session.
    monkeypatch.setattr(
        "memex.agents.answering.complete_structured", fake_call
    )

    response = await answer_query("What does Smith say about reflexivity?")

    assert response.answered is True
    assert response.regenerate_attempts == 1
    assert len(response.claims) == 1
    assert response.claims[0].source_chunk_id == "c1"


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_retrieve", "patch_prompt")
async def test_empty_retrieval_short_circuits_to_refuse(
    monkeypatch: pytest.MonkeyPatch, fake_llm: FakeLLM
) -> None:
    async def _empty(query: str, k: int = 50) -> list[Chunk]:
        return []

    monkeypatch.setattr("memex.agents.answering.hybrid_search", _empty)

    response = await answer_query("Anything")

    assert response.answered is False
    assert "No relevant content" in (response.refusal_reason or "")
    # No model calls at all on the empty path.
    assert fake_llm.calls == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_retrieve", "patch_prompt")
async def test_correlation_id_is_a_ulid_on_response(fake_llm: FakeLLM) -> None:
    fake_llm.respond(
        "assess_sufficiency",
        SufficiencyAssessment,
        SufficiencyAssessment(sufficient=False, reason="nope"),
    )
    response = await answer_query("Whatever")
    # Canonical ULIDs are 26 chars, base32-uppercase (Crockford alphabet).
    assert len(response.correlation_id) == 26
    assert response.correlation_id.isalnum()


@pytest.mark.asyncio
async def test_initial_state_uses_python_ulid_api() -> None:
    """Regression: AnswerState.correlation_id must construct without
    crashing on the python-ulid import path."""
    state = AnswerState(query="hello")
    assert len(state.correlation_id) == 26
