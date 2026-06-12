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
    RelevanceAssessment,
    SufficiencyAssessment,
    VerificationResult,
    answer,
    answer_query,
    assess,
    assess_relevance,
    compose,
    expand_graph,
    refuse,
    reset_compiled_graph,
    resolve_artifact_scope,
    verify,
)
from memex.core.errors import ModelCallError

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
            text="Smith argues that reflexivity is constitutive rather than corrective.",
            page=4,
            score=0.85,
        ),
    ]


@pytest.fixture
def patch_retrieve(monkeypatch: pytest.MonkeyPatch, fake_chunks: list[Chunk]) -> None:
    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return list(fake_chunks)

    async def _rerank(query: str, candidates: list[Chunk], top_k: int = 10) -> list[Chunk]:
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
        prompt_text = prompt if isinstance(prompt, str) else "\n".join(m["content"] for m in prompt)
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
        # The relevance gate (assess_relevance) runs on every grounded answer.
        # Unless a test explicitly cans a non-responsive verdict, default to
        # responsive so the grounded answer flows through to compose.
        if schema.__name__ == "RelevanceAssessment":
            return RelevanceAssessment(responsive=True, reason="(test default)"), 6
        raise AssertionError(f"no canned response for ({prompt_text!r}, {schema.__name__})")


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    llm = FakeLLM()
    monkeypatch.setattr("memex.agents.answering.complete_structured", llm)
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
async def test_non_responsive_grounded_answer_is_refused(fake_llm: FakeLLM) -> None:
    """The relevance gate: a fully GROUNDED answer that addresses a
    related-but-different question than asked is refused (the slide-decks-30
    conflation — library-usage content passed off as a kernel-language answer).
    """
    fake_llm.respond(
        "assess_sufficiency",
        SufficiencyAssessment,
        SufficiencyAssessment(sufficient=True, reason="on-topic chunk present"),
    )
    fake_llm.respond(
        "answer",
        DraftAnswer,
        DraftAnswer(
            summary="C++ and Python library abstractions are available.",
            claims=[
                CitedClaim(
                    claim="C++ and Python abstractions for tensor cores exist.",
                    source_chunk_id="c1",
                    confidence="high",
                ),
            ],
        ),
    )
    fake_llm.respond(
        "verify_grounding",
        VerificationResult,
        VerificationResult(grounded=[0], ungrounded=[]),
    )
    # The grounded claim is about library abstractions, not the asked-about
    # kernel-writing languages -> non-responsive -> the gate refuses.
    fake_llm.respond(
        "assess_relevance",
        RelevanceAssessment,
        RelevanceAssessment(
            responsive=False, reason="Answer is about libraries, not kernel languages."
        ),
    )

    response = await answer_query("Which language must I use to write CUDA kernels?")

    assert response.answered is False  # grounded but non-responsive -> refused
    assert response.refusal_reason is not None
    assert "related topic" in response.refusal_reason  # the relevance-refusal text
    assert not response.claims


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

    async def fake_call(*, prompt: str, schema: type, **_kw: object) -> tuple[Any, int]:
        fake_llm.calls.append((prompt, schema))
        if schema is SufficiencyAssessment:
            return fake_llm._responses[("assess_sufficiency", schema)], 8
        if schema is DraftAnswer:
            return next(drafts), 30
        if schema is VerificationResult or schema.__name__ == VerificationResult.__name__:
            return next(verdicts), 15
        if schema.__name__ == "RelevanceAssessment":
            return RelevanceAssessment(responsive=True, reason="(test default)"), 6
        raise AssertionError(f"unexpected schema {schema}")

    # Use monkeypatch so the patch is restored at test teardown — direct
    # module-attribute assignment used to leak into later tests in the
    # same session.
    monkeypatch.setattr("memex.agents.answering.complete_structured", fake_call)

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


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_retrieve", "patch_prompt")
async def test_verify_filters_phantom_indices(fake_llm: FakeLLM) -> None:
    """P3.3 v7 trace 2026-05-23: the verifier can emit indices > n-1
    when distracted by chunk content not in the draft. The verify node
    must filter those phantoms instead of treating them as real
    ungrounded claims (which would otherwise refuse a legitimate
    answer)."""
    fake_llm.respond(
        "assess_sufficiency",
        SufficiencyAssessment,
        SufficiencyAssessment(sufficient=True, reason="One on-point chunk"),
    )
    fake_llm.respond(
        "answer",
        DraftAnswer,
        DraftAnswer(
            summary="Bar charts work well combined with maps for context.",
            claims=[
                CitedClaim(
                    claim="Combine bar charts with maps.",
                    source_chunk_id="c1",
                    confidence="high",
                ),
            ],
        ),
    )
    # Verifier (incorrectly) emits index 1 referencing chunk content
    # ("Pareto chart") that is NOT in the draft. The draft has only
    # claim index 0. The verify node must filter index 1 out.
    fake_llm.respond(
        "verify_grounding",
        VerificationResult,
        VerificationResult(
            grounded=[0],
            ungrounded=[1],
            ungrounded_reasons=["Claim 1 refers to a Pareto chart not in cited chunk."],
        ),
    )

    response = await answer_query("Why do bar charts work well with maps?")

    # Phantom index 1 dropped; claim 0 is grounded; answer ships.
    assert response.answered is True
    assert response.summary is not None
    assert len(response.claims) == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_retrieve", "patch_prompt")
async def test_verify_missing_index_treated_as_ungrounded(
    fake_llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a claim index is missing from both `grounded` and `ungrounded`,
    treat it as ungrounded (conservative — don't default to grounded on
    verifier omission). Post-audit (2026-05-23): the phantom-index filter
    conservatively adds the omitted claim to `ungrounded`.

    Partial-grounded update (2026-05-26): with claim 0 grounded and claim 1
    omitted→ungrounded, the agent regenerates (trying to ground claim 1), then
    — since claim 1 stays ungrounded — ships the GROUNDED SUBSET (claim 0) and
    DROPS claim 1. The contract this pins is unchanged: the omitted claim is
    never treated as grounded (it's dropped, not shipped). The observable
    outcome moved from whole-answer refusal to a partial-grounded ship."""
    # M1 citation-retarget OFF: this test pins the PRE-retarget filter semantics
    # (a probe-aware fake would conflate the two contracts; the retarget has its own tests).
    monkeypatch.setenv("MEMEX_AGENTS__CITATION_RETARGET_ENABLED", "false")
    from memex.core.config import MemexSettings as _MS
    from memex.core.config import set_settings as _ss
    _ss(_MS())
    fake_llm.respond(
        "assess_sufficiency",
        SufficiencyAssessment,
        SufficiencyAssessment(sufficient=True, reason="Two on-point chunks"),
    )
    fake_llm.respond(
        "answer",
        DraftAnswer,
        DraftAnswer(
            summary="Two claims about reflexivity.",
            claims=[
                CitedClaim(claim="Claim A", source_chunk_id="c1", confidence="high"),
                CitedClaim(claim="Claim B", source_chunk_id="c2", confidence="high"),
            ],
        ),
    )
    # Verifier covers ONLY claim 0; claim 1 is omitted from both lists. The
    # phantom-index filter conservatively adds claim 1 to ungrounded → mixed
    # verdict → regenerate (same canned draft) → exhausts → ships claim 0.
    fake_llm.respond(
        "verify_grounding",
        VerificationResult,
        VerificationResult(grounded=[0], ungrounded=[]),
    )

    response = await answer_query("What does Smith say?")

    # Claim 0 (grounded) ships; claim 1 (omitted→ungrounded) is DROPPED, never
    # treated as grounded.
    assert response.answered is True
    assert [c.claim for c in response.claims] == ["Claim A"]
    assert "Claim B" not in [c.claim for c in response.claims], (
        "Omitted claim was silently treated as grounded + shipped"
    )
    # The summary is rebuilt from the surviving claim — it must NOT carry the
    # original draft summary that referenced both claims.
    assert response.summary == "Claim A"
    # It still regenerated (tried to ground claim 1) before shipping the partial.
    assert response.regenerate_attempts >= 1


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_retrieve", "patch_prompt")
async def test_verify_contested_index_is_not_shipped_as_grounded(fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch) -> None:
    """A claim index the verifier put in BOTH `grounded` and `ungrounded` must NOT ship
    as grounded — ungrounded is authoritative (the HARD gate: a verifier-flagged claim
    can't leak into the answer because `compose` keys only on `grounded`). The contested
    claim is dropped; the clean one ships."""
    # M1 citation-retarget OFF: this test pins the PRE-retarget filter semantics
    # (a probe-aware fake would conflate the two contracts; the retarget has its own tests).
    monkeypatch.setenv("MEMEX_AGENTS__CITATION_RETARGET_ENABLED", "false")
    from memex.core.config import MemexSettings as _MS
    from memex.core.config import set_settings as _ss
    _ss(_MS())
    fake_llm.respond(
        "assess_sufficiency",
        SufficiencyAssessment,
        SufficiencyAssessment(sufficient=True, reason="Two on-point chunks"),
    )
    fake_llm.respond(
        "answer",
        DraftAnswer,
        DraftAnswer(
            summary="Two claims.",
            claims=[
                CitedClaim(claim="Claim A", source_chunk_id="c1", confidence="high"),
                CitedClaim(claim="Claim B", source_chunk_id="c2", confidence="high"),
            ],
        ),
    )
    # Claim 1 is CONTESTED — the verifier listed it in BOTH grounded and ungrounded.
    fake_llm.respond(
        "verify_grounding",
        VerificationResult,
        VerificationResult(
            grounded=[0, 1],
            ungrounded=[1],
            ungrounded_reasons=["Claim B is not actually supported by its chunk."],
        ),
    )

    response = await answer_query("What does the source say?")

    assert response.answered is True
    # Claim B (contested → demoted to ungrounded) is dropped; only Claim A ships.
    assert [c.claim for c in response.claims] == ["Claim A"]
    assert "Claim B" not in [c.claim for c in response.claims], (
        "A verifier-contested claim was shipped as grounded"
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_retrieve", "patch_prompt")
async def test_partial_grounded_ships_grounded_subset(fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch) -> None:
    """#262 — the compound-question fix. A draft with one grounded claim (the
    answerable half) + one ungrounded claim (the half the corpus can't support)
    ships the GROUNDED subset: the ungrounded claim is dropped and the summary is
    rebuilt from the survivor, so no ungrounded assertion reaches the headline.
    `max_regenerate_attempts=0` exhausts the retry immediately → straight to the
    partial-ship branch."""
    # M1 citation-retarget OFF: this test pins the PRE-retarget filter semantics
    # (a probe-aware fake would conflate the two contracts; the retarget has its own tests).
    monkeypatch.setenv("MEMEX_AGENTS__CITATION_RETARGET_ENABLED", "false")
    from memex.core.config import MemexSettings as _MS
    from memex.core.config import set_settings as _ss
    _ss(_MS())
    fake_llm.respond(
        "assess_sufficiency",
        SufficiencyAssessment,
        SufficiencyAssessment(sufficient=True, reason="diagram present"),
    )
    fake_llm.respond(
        "answer",
        DraftAnswer,
        DraftAnswer(
            summary="The High Priority Queue is serviced first, so low-priority traffic is starved.",
            claims=[
                CitedClaim(
                    claim="The High Priority Queue is serviced first.",
                    source_chunk_id="c1",
                    confidence="high",
                ),
                CitedClaim(
                    claim="Low-priority traffic is starved.",
                    source_chunk_id="c2",
                    confidence="medium",
                ),
            ],
        ),
    )
    # Verifier grounds the queue-order claim, rejects the starvation claim
    # (not in the chunks).
    fake_llm.respond(
        "verify_grounding",
        VerificationResult,
        VerificationResult(
            grounded=[0],
            ungrounded=[1],
            ungrounded_reasons=["starvation risk not stated in the diagram"],
        ),
    )

    response = await answer_query(
        "Which queue is serviced first and what starvation risk does it create?",
        max_regenerate_attempts=0,
    )

    assert response.answered is True
    # Only the grounded claim ships; the ungrounded "starvation" claim is dropped.
    assert [c.claim for c in response.claims] == ["The High Priority Queue is serviced first."]
    # Summary rebuilt from the survivor — the ungrounded "starved" text is GONE.
    assert response.summary == "The High Priority Queue is serviced first."
    assert "starv" not in (response.summary or "").lower()


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_retrieve", "patch_prompt")
async def test_zero_grounded_refuses_even_with_partial(fake_llm: FakeLLM) -> None:
    """Partial-grounded never rescues a ZERO-grounded verdict — the
    counterfactual safety boundary (refusal_cf unaffected)."""
    fake_llm.respond(
        "assess_sufficiency",
        SufficiencyAssessment,
        SufficiencyAssessment(sufficient=True, reason="ok"),
    )
    fake_llm.respond(
        "answer",
        DraftAnswer,
        DraftAnswer(
            summary="Two claims.",
            claims=[
                CitedClaim(claim="Claim A", source_chunk_id="c1", confidence="high"),
                CitedClaim(claim="Claim B", source_chunk_id="c2", confidence="high"),
            ],
        ),
    )
    fake_llm.respond(
        "verify_grounding",
        VerificationResult,
        VerificationResult(grounded=[], ungrounded=[0, 1]),
    )

    response = await answer_query("an unanswerable thing", max_regenerate_attempts=0)
    assert response.answered is False
    assert response.claims == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_retrieve", "patch_prompt")
async def test_partial_grounded_kill_switch_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, fake_llm: FakeLLM
) -> None:
    """`MEMEX_AGENTS__PARTIAL_GROUNDED_ANSWERS=false` restores all-or-nothing:
    a mixed verdict refuses instead of shipping the grounded subset."""
    # M1 citation-retarget OFF: this test pins the PRE-retarget filter semantics
    # (a probe-aware fake would conflate the two contracts; the retarget has its own tests).
    monkeypatch.setenv("MEMEX_AGENTS__CITATION_RETARGET_ENABLED", "false")
    from memex.core.config import MemexSettings as _MS
    from memex.core.config import set_settings as _ss
    _ss(_MS())
    from memex.core.config import MemexSettings, set_settings

    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("MEMEX_AGENTS__PARTIAL_GROUNDED_ANSWERS", "false")
    set_settings(MemexSettings())  # type: ignore[call-arg]

    fake_llm.respond(
        "assess_sufficiency",
        SufficiencyAssessment,
        SufficiencyAssessment(sufficient=True, reason="ok"),
    )
    fake_llm.respond(
        "answer",
        DraftAnswer,
        DraftAnswer(
            summary="Two claims.",
            claims=[
                CitedClaim(claim="Claim A", source_chunk_id="c1", confidence="high"),
                CitedClaim(claim="Claim B", source_chunk_id="c2", confidence="high"),
            ],
        ),
    )
    fake_llm.respond(
        "verify_grounding",
        VerificationResult,
        VerificationResult(grounded=[0], ungrounded=[1]),
    )

    try:
        response = await answer_query("mixed verdict, switch off", max_regenerate_attempts=0)
    finally:
        set_settings(None)

    assert response.answered is False  # kill-switch → all-or-nothing refuse


# ----- P4.1: compose derives FinalResponse.wikilinks (deterministic) -----


def _compose_state(
    chunks: list[Chunk],
    claims: list[CitedClaim],
    *,
    grounded: list[int],
) -> AnswerState:
    """Build a minimal AnswerState ready for the `compose` node — a
    verified draft (`grounded` indices) over `chunks` already in
    `reranked`."""
    return AnswerState(
        query="q",
        reranked=list(chunks),
        draft=DraftAnswer(summary="A summary.", claims=list(claims)),
        verification=VerificationResult(grounded=list(grounded), ungrounded=[]),
    )


@pytest.mark.asyncio
async def test_compose_derives_wikilink_with_section() -> None:
    """A cited chunk with a populated heading_path → `[[doc#deepest]]`
    (deepest heading, raw text)."""
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        document_title="Annual Report",
        text="Director compensation totaled $1.2M.",
        heading_path=["Governance", "Director Compensation"],
    )
    claim = CitedClaim(claim="Director comp was $1.2M.", source_chunk_id="c1", confidence="high")
    update = await compose(_compose_state([chunk], [claim], grounded=[0]))
    final = update.get("final")
    assert final is not None
    assert final.answered is True
    assert final.wikilinks == ["[[d1#Director Compensation]]"]


@pytest.mark.asyncio
async def test_compose_derives_bare_wikilink_when_no_heading_path() -> None:
    """A cited chunk with empty heading_path → bare `[[doc]]`."""
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        document_title="Smith 2024",
        text="Reflexivity shapes the data.",
        heading_path=[],
    )
    claim = CitedClaim(claim="Reflexivity shapes data.", source_chunk_id="c1", confidence="high")
    update = await compose(_compose_state([chunk], [claim], grounded=[0]))
    final = update.get("final")
    assert final is not None
    assert final.wikilinks == ["[[d1]]"]


@pytest.mark.asyncio
async def test_compose_dedups_wikilinks_preserving_order() -> None:
    """Two cited chunks from the same doc+section → one wikilink; the
    first-seen order is preserved across distinct targets."""
    chunks = [
        Chunk(
            chunk_id="c1",
            document_id="d1",
            document_title="Report",
            text="first",
            heading_path=["Methods"],
        ),
        Chunk(
            chunk_id="c2",
            document_id="d2",
            document_title="Other",
            text="second",
            heading_path=["Results"],
        ),
        Chunk(
            chunk_id="c3",
            document_id="d1",
            document_title="Report",
            text="third",
            heading_path=["Methods"],
        ),
    ]
    claims = [
        CitedClaim(claim="A", source_chunk_id="c1", confidence="high"),
        CitedClaim(claim="B", source_chunk_id="c2", confidence="high"),
        CitedClaim(claim="C", source_chunk_id="c3", confidence="high"),
    ]
    update = await compose(_compose_state(chunks, claims, grounded=[0, 1, 2]))
    final = update.get("final")
    assert final is not None
    # d1#Methods cited twice → deduped to one; order: d1#Methods then d2#Results.
    assert final.wikilinks == ["[[d1#Methods]]", "[[d2#Results]]"]
    # Per-claim is ALIGNED 1:1 with claims and NOT deduped — the repeat is preserved so each
    # claim keeps its own source mapping (the whole point vs the flat deduped Sources list).
    assert final.claim_wikilinks == ["[[d1#Methods]]", "[[d2#Results]]", "[[d1#Methods]]"]
    assert len(final.claim_wikilinks) == len(final.claims)


@pytest.mark.asyncio
async def test_compose_per_claim_wikilinks_align_to_surviving_partial_grounded() -> None:
    """On a partial-grounded ship, `claim_wikilinks` aligns with the SURVIVING claims — the
    dropped (ungrounded) claim contributes no entry, so the list never drifts out of sync."""
    chunks = [
        Chunk(chunk_id="c1", document_id="d1", document_title="R", text="a", heading_path=["Intro"]),
        Chunk(chunk_id="c2", document_id="d2", document_title="S", text="b", heading_path=["Body"]),
    ]
    claims = [
        CitedClaim(claim="kept-0", source_chunk_id="c1", confidence="high"),
        CitedClaim(claim="dropped-1", source_chunk_id="c2", confidence="high"),
    ]
    update = await compose(_compose_state(chunks, claims, grounded=[0]))  # only claim 0 grounded
    final = update.get("final")
    assert final is not None
    assert [c.claim for c in final.claims] == ["kept-0"]
    assert final.claim_wikilinks == ["[[d1#Intro]]"]  # 1:1 with the surviving claim only


@pytest.mark.asyncio
async def test_compose_emits_no_claim_wikilinks_on_zero_grounded_refusal() -> None:
    """A zero-grounded result refuses (answered=False) and carries `claim_wikilinks=[]` —
    a refusal cited nothing (mirrors the flat `wikilinks=[]` refusal contract)."""
    chunk = Chunk(chunk_id="c1", document_id="d1", document_title="R", text="x", heading_path=["H"])
    claim = CitedClaim(claim="ungrounded", source_chunk_id="c1", confidence="low")
    update = await compose(_compose_state([chunk], [claim], grounded=[]))  # nothing grounded
    final = update.get("final")
    assert final is not None
    assert final.answered is False
    assert final.claim_wikilinks == [] and final.wikilinks == []


@pytest.mark.asyncio
async def test_refuse_node_emits_no_wikilinks() -> None:
    """The `refuse` node leaves `wikilinks=[]` — a refusal cited nothing,
    so emitting links would mislead."""
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        document_title="Report",
        text="off-topic",
        heading_path=["Methods"],
    )
    state = AnswerState(
        query="q",
        reranked=[chunk],
        sufficiency=SufficiencyAssessment(sufficient=False, reason="not enough"),
    )
    update = await refuse(state)
    final = update.get("final")
    assert final is not None
    assert final.answered is False
    assert final.wikilinks == []


@pytest.mark.asyncio
async def test_compose_no_surviving_claims_refusal_emits_no_wikilinks() -> None:
    """The in-compose no-surviving-claims refusal (verifier returned no
    grounded indices) keeps `wikilinks=[]` — it sits BELOW the derivation
    guard so a degenerate refusal carries no links."""
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        document_title="Report",
        text="content",
        heading_path=["Methods"],
    )
    claim = CitedClaim(claim="A", source_chunk_id="c1", confidence="high")
    # grounded=[] → no surviving claims → in-compose refusal branch.
    update = await compose(_compose_state([chunk], [claim], grounded=[]))
    final = update.get("final")
    assert final is not None
    assert final.answered is False
    assert final.wikilinks == []


# ===========================================================================
# #256 — deterministic artifact→doc re-scope
# ===========================================================================
#
# The `resolve_artifact_scope` node runs between `retrieve` and `expand_graph`.
# When a query NAMES a specific artifact it resolves the artifact's qualifier to
# the document(s) it lives in (via the FTS store) and REPLACES the candidate
# pool with chunks scoped there. These tests fake the FTS store (the resolver's
# per-token corpus search) alongside the existing retrieve/LLM fakes.


class _FakeFTSStore:
    """In-memory per-token BM25 stand-in for the resolver's corpus search.
    Replaces `FTSStore.open`. `search(token, k)` returns the corpus list for
    that exact token (the resolver searches one qualifier atom at a time)."""

    def __init__(self, corpus: dict[str, list[Chunk]]) -> None:
        self._corpus = corpus
        self.searched: list[str] = []
        self.closed = False

    @classmethod
    def opener(cls, corpus: dict[str, list[Chunk]]) -> tuple[Any, _FakeFTSStore]:
        instance = cls(corpus)

        async def _open(_vault_path: object) -> _FakeFTSStore:
            return instance

        return _open, instance

    async def search(self, query: str, *, k: int) -> list[Chunk]:
        self.searched.append(query)
        return list(self._corpus.get(query, []))[:k]

    async def close(self) -> None:
        self.closed = True


def _doc_chunk(cid: str, doc: str, title: str, *, text: str = "x", score: float = 1.0) -> Chunk:
    return Chunk(
        chunk_id=f"{doc}#{cid}",
        document_id=doc,
        document_title=title,
        text=text,
        page=1,
        score=score,
    )


_DIAG12 = "Quelle est la plage d'adresses VLAN configurée dans le diagramme de coupe-feu ?"
_IMG01 = (
    "Selon le diagramme de configuration VLAN du commutateur, "
    "quel port sert de port tronc et vers quels VLAN ?"
)
_FW_TITLE = "CR350 Diagrammes coupe-feu"
_LEC_TITLE = "Cours 6 coupe-feu (firewall)"


def _firewall_corpus() -> dict[str, list[Chunk]]:
    """coupe/feu live ONLY in the two firewall docs (diagram dominates by hits;
    the lecture is folded in by sibling-by-title)."""
    return {
        "coupe": [_doc_chunk(f"d{i}", "fw-diagram", _FW_TITLE, score=5.0 - i) for i in range(6)]
        + [
            _doc_chunk("l0", "cours-6", _LEC_TITLE, score=3.0),
            _doc_chunk("l1", "cours-6", _LEC_TITLE, score=2.0),
        ],
        "feu": [_doc_chunk(f"e{i}", "fw-diagram", _FW_TITLE, score=4.0 - i) for i in range(4)],
    }


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_artifact_scope_rescopes_and_refuses_wrong_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, fake_llm: FakeLLM
) -> None:
    """diag-12, the motivating bug. Full retrieval surfaces a semaine-4 VLAN
    chunk (the wrong source). The resolver scopes to the TWO firewall docs; the
    scoped re-retrieval has NO VLAN content; the reranker therefore never sees
    the VLAN chunk and the agent refuses — deterministically, regardless of LLM
    sampling. Also covers the multi-doc-scope case (both firewall ids passed)."""
    from memex.core.config import MemexSettings, set_settings
    from memex.index.fts_store import FTSStore

    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    set_settings(MemexSettings())  # type: ignore[call-arg]

    fake_open, _store = _FakeFTSStore.opener(_firewall_corpus())
    monkeypatch.setattr(FTSStore, "open", fake_open)

    vlan_chunk = _doc_chunk(
        "v1",
        "semaine-4",
        "Semaine 4 VLAN",
        text="Le port Fa0/21 est un trunk pour VLAN 10/20/30.",
        score=0.9,
    )
    fw_chunk = _doc_chunk(
        "fw1",
        "fw-diagram",
        _FW_TITLE,
        text="Architecture coupe-feu single-tier.",
        score=0.8,
    )

    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return [vlan_chunk]  # the WRONG-source chunk full retrieval would surface

    in_docs_calls: list[list[str]] = []

    async def _hybrid_in_docs(query: str, doc_ids: list[str], *, k: int) -> list[Chunk]:
        in_docs_calls.append(list(doc_ids))
        return [fw_chunk]

    rerank_saw: list[list[str]] = []

    async def _rerank(query: str, cands: list[Chunk], top_k: int = 10) -> list[Chunk]:
        rerank_saw.append([c.chunk_id for c in cands])
        return list(cands[:top_k])

    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.hybrid_search_in_docs", _hybrid_in_docs)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    # Scoped to firewall docs (no VLAN) → assess can't answer → refuse.
    fake_llm.respond(
        "assess_sufficiency",
        SufficiencyAssessment,
        SufficiencyAssessment(sufficient=False, reason="No VLAN range in the firewall diagram."),
    )

    try:
        response = await answer_query(_DIAG12)
    finally:
        set_settings(None)

    assert response.answered is False
    assert response.claims == []
    # Re-scope fired with BOTH firewall docs and excluded the wrong-source doc.
    assert in_docs_calls, "resolver should have re-scoped"
    assert set(in_docs_calls[0]) == {"fw-diagram", "cours-6"}
    assert "semaine-4" not in in_docs_calls[0]
    # Determinism: the VLAN chunk never reached the reranker; only the scoped
    # firewall chunk did — the LLM had no VLAN evidence to (mis)answer from.
    assert rerank_saw == [["fw-diagram#fw1"]]
    # Observability (#256): the refusal SURFACES the scope so a caller can see
    # WHY the pool was narrowed (it refused because the firewall docs lack VLAN).
    assert set(response.artifact_scope_doc_ids) == {"fw-diagram", "cours-6"}


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_artifact_scope_near_twin_answers_from_scoped_doc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, fake_llm: FakeLLM
) -> None:
    """img-01, the near-twin. The qualifier INCLUDES 'vlan' (it follows
    "diagramme de"), so it resolves to the VLAN deck — whose scoped chunk DOES
    answer the trunk-port question → the agent answers, citing semaine-4."""
    from memex.core.config import MemexSettings, set_settings
    from memex.index.fts_store import FTSStore

    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    set_settings(MemexSettings())  # type: ignore[call-arg]

    corpus = {
        "configuration": [
            _doc_chunk("c1", "semaine-4", "Semaine 4 VLAN", score=3.0),
            _doc_chunk("c2", "semaine-4", "Semaine 4 VLAN", score=2.0),
        ],
        "vlan": [
            _doc_chunk(f"v{i}", "semaine-4", "Semaine 4 VLAN", score=6.0 - i) for i in range(3)
        ],
        "commutateur": [_doc_chunk("m1", "semaine-4", "Semaine 4 VLAN", score=3.5)],
    }
    fake_open, _store = _FakeFTSStore.opener(corpus)
    monkeypatch.setattr(FTSStore, "open", fake_open)

    vlan_chunk = _doc_chunk(
        "v1",
        "semaine-4",
        "Semaine 4 VLAN",
        text="Le port Fa0/21 sert de port tronc vers les VLAN 10, 20 et 30.",
        score=0.9,
    )

    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return [vlan_chunk]

    in_docs_calls: list[list[str]] = []

    async def _hybrid_in_docs(query: str, doc_ids: list[str], *, k: int) -> list[Chunk]:
        in_docs_calls.append(list(doc_ids))
        return [vlan_chunk]

    async def _rerank(query: str, cands: list[Chunk], top_k: int = 10) -> list[Chunk]:
        return list(cands[:top_k])

    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.hybrid_search_in_docs", _hybrid_in_docs)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    fake_llm.respond(
        "assess_sufficiency",
        SufficiencyAssessment,
        SufficiencyAssessment(sufficient=True, reason="VLAN trunk-port chunk present"),
    )
    fake_llm.respond(
        "answer",
        DraftAnswer,
        DraftAnswer(
            summary="Fa0/21 is the trunk port for VLANs 10/20/30.",
            claims=[
                CitedClaim(
                    claim="Fa0/21 is the trunk port for VLANs 10/20/30.",
                    source_chunk_id="semaine-4#v1",
                    confidence="high",
                )
            ],
        ),
    )
    fake_llm.respond(
        "verify_grounding", VerificationResult, VerificationResult(grounded=[0], ungrounded=[])
    )

    try:
        response = await answer_query(_IMG01)
    finally:
        set_settings(None)

    assert response.answered is True
    assert in_docs_calls == [["semaine-4"]]
    assert [c.source_chunk_id for c in response.claims] == ["semaine-4#v1"]
    # The answered response surfaces the scope too (auditability).
    assert response.artifact_scope_doc_ids == ["semaine-4"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_retrieve", "patch_prompt")
async def test_artifact_scope_noop_when_no_artifact_named(
    monkeypatch: pytest.MonkeyPatch, fake_llm: FakeLLM
) -> None:
    """A query that names no artifact takes a path byte-identical to the
    pre-#256 pipeline: the resolver never opens the FTS store (detection short-
    circuits before any settings read) and the candidate pool is untouched."""
    from memex.index.fts_store import FTSStore

    def _explode_open(*_a: object, **_k: object) -> Any:
        raise AssertionError("FTSStore must not be opened for a no-artifact query")

    async def _explode_in_docs(query: str, doc_ids: list[str], *, k: int) -> list[Chunk]:
        raise AssertionError("no re-scope for a no-artifact query")

    monkeypatch.setattr(FTSStore, "open", _explode_open)
    monkeypatch.setattr("memex.agents.answering.hybrid_search_in_docs", _explode_in_docs)
    fake_llm.respond(
        "assess_sufficiency",
        SufficiencyAssessment,
        SufficiencyAssessment(sufficient=True, reason="ok"),
    )
    fake_llm.respond(
        "answer",
        DraftAnswer,
        DraftAnswer(
            summary="Reflexivity shapes data.",
            claims=[
                CitedClaim(
                    claim="Reflexivity shapes the data.", source_chunk_id="c1", confidence="high"
                )
            ],
        ),
    )
    fake_llm.respond(
        "verify_grounding", VerificationResult, VerificationResult(grounded=[0], ungrounded=[])
    )

    response = await answer_query("What does Smith say about reflexivity?")
    assert response.answered is True
    # No artifact named → no re-scope → the surfaced scope is empty (full corpus).
    assert response.artifact_scope_doc_ids == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_artifact_scope_fails_open_on_store_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, fake_llm: FakeLLM
) -> None:
    """A store error during resolution must NEVER turn an answerable artifact
    query into a refusal — the node fails open to the full-corpus pool."""
    from memex.core.config import MemexSettings, set_settings
    from memex.index.fts_store import FTSStore

    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    set_settings(MemexSettings())  # type: ignore[call-arg]

    async def _boom_open(_vault_path: object) -> Any:
        raise OSError("fts store unavailable")

    monkeypatch.setattr(FTSStore, "open", _boom_open)

    answerable = _doc_chunk("a1", "fw-diagram", _FW_TITLE, text="Coupe-feu single-tier.", score=0.9)

    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return [answerable]

    async def _explode_in_docs(query: str, doc_ids: list[str], *, k: int) -> list[Chunk]:
        raise AssertionError("must not re-scope after a store error")

    async def _rerank(query: str, cands: list[Chunk], top_k: int = 10) -> list[Chunk]:
        return list(cands[:top_k])

    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.hybrid_search_in_docs", _explode_in_docs)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    fake_llm.respond(
        "assess_sufficiency",
        SufficiencyAssessment,
        SufficiencyAssessment(sufficient=True, reason="ok"),
    )
    fake_llm.respond(
        "answer",
        DraftAnswer,
        DraftAnswer(
            summary="Single-tier firewall architecture.",
            claims=[
                CitedClaim(
                    claim="The firewall uses a single-tier architecture.",
                    source_chunk_id="fw-diagram#a1",
                    confidence="high",
                )
            ],
        ),
    )
    fake_llm.respond(
        "verify_grounding", VerificationResult, VerificationResult(grounded=[0], ungrounded=[])
    )

    try:
        response = await answer_query("Quelles architectures montre le diagramme de coupe-feu ?")
    finally:
        set_settings(None)

    assert response.answered is True  # fell back to full retrieval, did not refuse
    assert response.artifact_scope_doc_ids == []  # fail-open → no scope surfaced


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_artifact_scope_kill_switch_disables_rescope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, fake_llm: FakeLLM
) -> None:
    """`MEMEX_AGENTS__ARTIFACT_SCOPE_ENABLED=false` fully reverts: an artifact
    query is detected but the resolver is never invoked (no store open, no
    re-scope) — the full-corpus path runs unchanged."""
    from memex.core.config import MemexSettings, set_settings
    from memex.index.fts_store import FTSStore

    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("MEMEX_AGENTS__ARTIFACT_SCOPE_ENABLED", "false")
    set_settings(MemexSettings())  # type: ignore[call-arg]

    def _explode_open(*_a: object, **_k: object) -> Any:
        raise AssertionError("kill-switch: the resolver must not open the store")

    async def _explode_in_docs(query: str, doc_ids: list[str], *, k: int) -> list[Chunk]:
        raise AssertionError("kill-switch: no re-scope")

    monkeypatch.setattr(FTSStore, "open", _explode_open)
    monkeypatch.setattr("memex.agents.answering.hybrid_search_in_docs", _explode_in_docs)

    chunk = _doc_chunk("a1", "fw-diagram", _FW_TITLE, text="Coupe-feu single-tier.", score=0.9)

    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return [chunk]

    async def _rerank(query: str, cands: list[Chunk], top_k: int = 10) -> list[Chunk]:
        return list(cands[:top_k])

    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    fake_llm.respond(
        "assess_sufficiency",
        SufficiencyAssessment,
        SufficiencyAssessment(sufficient=True, reason="ok"),
    )
    fake_llm.respond(
        "answer",
        DraftAnswer,
        DraftAnswer(
            summary="Single-tier.",
            claims=[
                CitedClaim(
                    claim="Single-tier architecture.",
                    source_chunk_id="fw-diagram#a1",
                    confidence="high",
                )
            ],
        ),
    )
    fake_llm.respond(
        "verify_grounding", VerificationResult, VerificationResult(grounded=[0], ungrounded=[])
    )

    try:
        response = await answer_query(_DIAG12)
    finally:
        set_settings(None)

    assert response.answered is True
    assert response.artifact_scope_doc_ids == []  # kill-switch → no scope surfaced


def test_answer_state_artifact_scope_defaults_empty() -> None:
    assert AnswerState(query="x").artifact_scope_doc_ids == []


@pytest.mark.asyncio
async def test_resolve_node_populates_state_field_and_replaces_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Direct node call: a confident resolution returns the scoped doc ids AND
    replaces the candidate pool with the scoped re-retrieval."""
    from memex.core.config import MemexSettings, set_settings
    from memex.index.fts_store import FTSStore

    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    set_settings(MemexSettings())  # type: ignore[call-arg]

    corpus = {
        "gantt": [
            _doc_chunk("g1", "tableau-guide", "Tableau chart guide", score=7.0),
            _doc_chunk("g2", "tableau-guide", "Tableau chart guide", score=6.0),
        ]
    }
    fake_open, _store = _FakeFTSStore.opener(corpus)
    monkeypatch.setattr(FTSStore, "open", fake_open)

    scoped = _doc_chunk("g1", "tableau-guide", "Tableau chart guide", score=0.9)

    async def _hybrid_in_docs(query: str, doc_ids: list[str], *, k: int) -> list[Chunk]:
        return [scoped]

    monkeypatch.setattr("memex.agents.answering.hybrid_search_in_docs", _hybrid_in_docs)

    state = AnswerState(
        query="In the Gantt chart, who is assigned?",
        candidates=[_doc_chunk("x", "other", "Other Doc")],
    )
    try:
        update = await resolve_artifact_scope(state)
    finally:
        set_settings(None)

    assert update.get("artifact_scope_doc_ids") == ["tableau-guide"]
    assert [c.chunk_id for c in update.get("candidates", [])] == ["tableau-guide#g1"]


@pytest.mark.asyncio
async def test_expand_graph_skips_when_artifact_scope_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The determinism guard: with a scope active, `expand_graph` must NOT open
    the graph store (which could re-admit out-of-scope chunks and reintroduce
    the bug). It returns the counter-bump only."""

    def _explode_open(*_a: object, **_k: object) -> Any:
        raise AssertionError("expand_graph must not open the graph store under an active scope")

    monkeypatch.setattr("memex.index.graph_store.GraphStore.open", _explode_open)

    state = AnswerState(
        query="anything",
        candidates=[_doc_chunk("a", "fw-diagram", _FW_TITLE)],
        artifact_scope_doc_ids=["fw-diagram"],
    )
    update = await expand_graph(state)
    assert update == {"nodes_traversed": 1}


# ── Explicit document scope (the Notebook-LM doc-picker) ──────────────────────
# answer_query(scope_doc_ids=[...]) scopes retrieval to exactly those docs via the
# same resolve_artifact_scope node, TAKING PRECEDENCE over any inferred artifact
# reference (the node short-circuits before detection, so no FTS resolver runs).
# These harden the pipeline against the edge cases a manual picker introduces:
# bogus ids, empty pools, dedup/blanks, precedence. The scope sets
# artifact_scope_doc_ids, so expand_graph short-circuits (pinned just above) —
# hence each scoped test sees hybrid_search_in_docs called EXACTLY once.


def _scope_chunk(doc: str, cid: str = "s1", text: str = "scoped content") -> Chunk:
    return Chunk(
        chunk_id=f"{doc}#{cid}",
        document_id=doc,
        document_title=f"Doc {doc}",
        text=text,
        page=1,
        score=0.9,
    )


def _patch_scope_retrieve(
    monkeypatch: pytest.MonkeyPatch, *, full: list[Chunk], scoped: list[Chunk]
) -> tuple[list[list[str]], list[list[str]]]:
    """Fake full-corpus `hybrid_search` (→ `full`) + scoped `hybrid_search_in_docs`
    (→ `scoped`, recording the doc_ids of every call) + a passthrough rerank
    (recording what reached it). Returns (in_docs_calls, rerank_saw)."""
    in_docs_calls: list[list[str]] = []
    rerank_saw: list[list[str]] = []

    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return list(full)

    async def _hybrid_in_docs(query: str, doc_ids: list[str], *, k: int) -> list[Chunk]:
        in_docs_calls.append(list(doc_ids))
        return list(scoped)

    async def _rerank(query: str, cands: list[Chunk], top_k: int = 10) -> list[Chunk]:
        rerank_saw.append([c.chunk_id for c in cands])
        return list(cands[:top_k])

    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.hybrid_search_in_docs", _hybrid_in_docs)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    return in_docs_calls, rerank_saw


def _can_answer(fake_llm: FakeLLM, *, chunk_id: str) -> None:
    """Can the LLM to produce a grounded one-claim answer citing `chunk_id`."""
    fake_llm.respond(
        "assess_sufficiency",
        SufficiencyAssessment,
        SufficiencyAssessment(sufficient=True, reason="ok"),
    )
    fake_llm.respond(
        "answer",
        DraftAnswer,
        DraftAnswer(
            summary="Answer from the scoped doc.",
            claims=[
                CitedClaim(claim="The scoped fact.", source_chunk_id=chunk_id, confidence="high")
            ],
        ),
    )
    fake_llm.respond(
        "verify_grounding", VerificationResult, VerificationResult(grounded=[0], ungrounded=[])
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_explicit_scope_answers_from_selected_doc(
    monkeypatch: pytest.MonkeyPatch, fake_llm: FakeLLM
) -> None:
    """Scope to one doc → retrieval scoped there → answered from it; the DIFFERENT
    full-corpus chunk is bypassed and never reaches the reranker (graph-skip too)."""
    in_docs_calls, rerank_saw = _patch_scope_retrieve(
        monkeypatch,
        full=[_scope_chunk("other-doc", text="unrelated")],
        scoped=[_scope_chunk("lecture-5")],
    )
    _can_answer(fake_llm, chunk_id="lecture-5#s1")

    response = await answer_query(
        "How does STP elect the root bridge?", scope_doc_ids=["lecture-5"]
    )

    assert response.answered is True
    assert in_docs_calls == [
        ["lecture-5"]
    ]  # scoped to exactly the pick, ONCE (expand_graph skipped)
    assert rerank_saw == [["lecture-5#s1"]]  # the wrong-doc full-corpus chunk never reached rerank
    assert response.artifact_scope_doc_ids == ["lecture-5"]
    assert {c.source_chunk_id for c in response.claims} == {"lecture-5#s1"}


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_explicit_scope_takes_precedence_over_named_artifact(
    monkeypatch: pytest.MonkeyPatch, fake_llm: FakeLLM
) -> None:
    """A query that NAMES an artifact ('le diagramme de coupe-feu') is OVERRIDDEN
    by the user's explicit pick — the artifact inference (FTS resolver) is never
    even consulted (the node short-circuits on the explicit scope)."""
    from memex.index.fts_store import FTSStore

    async def _boom(_vault_path: object) -> Any:
        raise AssertionError("artifact inference must not run under an explicit scope")

    monkeypatch.setattr(FTSStore, "open", _boom)
    in_docs_calls, _ = _patch_scope_retrieve(
        monkeypatch, full=[_scope_chunk("fw-diagram")], scoped=[_scope_chunk("my-pick")]
    )
    _can_answer(fake_llm, chunk_id="my-pick#s1")

    response = await answer_query(_DIAG12, scope_doc_ids=["my-pick"])

    assert response.answered is True
    assert in_docs_calls == [["my-pick"]]  # the PICK, not the inferred firewall docs
    assert response.artifact_scope_doc_ids == ["my-pick"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_explicit_scope_dedups_and_strips_blanks(
    monkeypatch: pytest.MonkeyPatch, fake_llm: FakeLLM
) -> None:
    """Bogus shapes from a caller (dupes, empty/whitespace ids) are normalised."""
    in_docs_calls, _ = _patch_scope_retrieve(monkeypatch, full=[], scoped=[_scope_chunk("a")])
    _can_answer(fake_llm, chunk_id="a#s1")

    await answer_query("q", scope_doc_ids=["a", "a", "", "  ", "b"])

    assert in_docs_calls == [["a", "b"]]  # deduped, order preserved, blanks dropped


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_empty_scope_uses_full_corpus(
    monkeypatch: pytest.MonkeyPatch, fake_llm: FakeLLM
) -> None:
    """Empty selection = the full-corpus path: the scope node never re-scopes."""
    in_docs_calls, rerank_saw = _patch_scope_retrieve(
        monkeypatch, full=[_scope_chunk("d1")], scoped=[]
    )
    _can_answer(fake_llm, chunk_id="d1#s1")

    # graph_expansion off so the only possible hybrid_search_in_docs caller is the
    # scope node — proving it did NOT scope.
    response = await answer_query(
        "a plain open question", scope_doc_ids=[], graph_expansion_enabled=False
    )

    assert in_docs_calls == []  # never scoped
    assert response.artifact_scope_doc_ids == []
    assert response.answered is True
    assert rerank_saw == [["d1#s1"]]  # answered from the full-corpus pool


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_explicit_scope_refuses_when_selected_doc_cannot_answer(
    monkeypatch: pytest.MonkeyPatch, fake_llm: FakeLLM
) -> None:
    """The point of the picker: if the answer ISN'T in the selected docs, refuse —
    deterministically, like diag-12. The scope is surfaced on the refusal."""
    in_docs_calls, _ = _patch_scope_retrieve(
        monkeypatch,
        full=[_scope_chunk("has-answer", text="the answer lives here")],
        scoped=[_scope_chunk("wrong-pick", text="off-topic content")],
    )
    fake_llm.respond(
        "assess_sufficiency",
        SufficiencyAssessment,
        SufficiencyAssessment(sufficient=False, reason="Not in the selected document."),
    )

    response = await answer_query("the answer?", scope_doc_ids=["wrong-pick"])

    assert response.answered is False
    assert response.claims == []
    assert in_docs_calls == [["wrong-pick"]]
    assert response.artifact_scope_doc_ids == ["wrong-pick"]  # surfaced on the refusal


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_explicit_scope_empty_pool_refuses_cleanly(
    monkeypatch: pytest.MonkeyPatch, fake_llm: FakeLLM
) -> None:
    """Edge: a bogus/nonexistent id (or a doc with no matching chunk) → an empty
    scoped pool → a clean refusal (never a crash, never a hallucination)."""
    in_docs_calls, rerank_saw = _patch_scope_retrieve(
        monkeypatch, full=[_scope_chunk("real-doc")], scoped=[]
    )
    fake_llm.respond(
        "assess_sufficiency",
        SufficiencyAssessment,
        SufficiencyAssessment(sufficient=False, reason="No retrieved context."),
    )

    response = await answer_query("anything", scope_doc_ids=["does-not-exist"])

    assert response.answered is False
    assert response.claims == []
    assert in_docs_calls == [["does-not-exist"]]
    assert rerank_saw == [[]]  # the empty scoped pool reached rerank, not the full corpus
    assert response.artifact_scope_doc_ids == ["does-not-exist"]


# ----- Numeric-grounding backstop (2026-05-31) -----------------------------
# The verify node's 4th demotion filter closes the aggregate-numeric
# FALSE-POSITIVE: the LLM verifier rubber-stamps a SUMMED table total via the
# literal-table-row loophole (the live $159,748,343 10-K fabrication). These
# pin the wiring; the pure matchers are in tests/unit/test_numeric_grounding.py.

# An exec-comp TABLE chunk (GFM + [table-rows]); the fabricated $159,748,343 is
# a SUM of these PSU cells and equals no single cell at any unit scale.
_EXEC_TABLE_TEXT = (
    "| Name | SY PSU ($) | MY PSU ($) |\n|---|---|---|\n"
    "| Huang | 19,166,424 | 18,034,343 |\n"
    "| Kress | 6,099,993 | 7,350,483 |\n\n"
    "[table-rows]\n[Comp] Name=Huang, SY PSU=19,166,424, MY PSU=18,034,343\n"
)


def _vchunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id="d1", document_title="10-K", text=text, page=1, score=0.9
    )


async def _run_verify(
    fake_llm: FakeLLM,
    *,
    claim: str,
    chunk: Chunk,
    backstop: bool = True,
) -> VerificationResult:
    """Drive the verify node directly: the LLM marks the single claim grounded;
    the deterministic backstop then decides keep-vs-demote."""
    fake_llm.respond(
        "verify_grounding",
        VerificationResult,
        VerificationResult(grounded=[0], ungrounded=[], ungrounded_reasons=[]),
    )
    state = AnswerState(
        query="q",
        draft=DraftAnswer(
            summary="s",
            claims=[CitedClaim(claim=claim, source_chunk_id=chunk.chunk_id, confidence="high")],
        ),
        reranked=[chunk],
        numeric_grounding_backstop=backstop,
    )
    out = await verify(state)
    result = out["verification"]
    assert isinstance(result, VerificationResult)
    return result


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_verify_fails_closed_on_model_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A verify `ModelCallError` (e.g. a guided-decode truncation) must FAIL CLOSED —
    every claim → ungrounded → `route_after_verify` → refuse — and NOT propagate. The
    2026-06-06 slide-decks crash was exactly this error escaping the verify node (out
    of `/ask` + aborting the whole eval suite). HARD-gate-safe: never ships unverified."""

    async def _raise(**_kw: object) -> tuple[object, int]:
        raise ModelCallError("guided-decode truncation")

    monkeypatch.setattr("memex.agents.answering.complete_structured", _raise)
    state = AnswerState(
        query="q",
        draft=DraftAnswer(
            summary="s",
            claims=[
                CitedClaim(claim="a", source_chunk_id="c1", confidence="high"),
                CitedClaim(claim="b", source_chunk_id="c1", confidence="high"),
            ],
        ),
        reranked=[_vchunk("c1", "irrelevant")],
    )
    out = await verify(state)
    v = out["verification"]
    assert isinstance(v, VerificationResult)
    assert v.grounded == []  # zero grounded → route_after_verify → refuse
    assert v.ungrounded == [0, 1]  # all claims treated as ungrounded


@pytest.mark.asyncio
async def test_assess_fails_closed_on_model_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An `assess` `ModelCallError` (the guided-decode free-text `reason` overflow class that
    crashed verify) must FAIL CLOSED — `sufficient=False` → refuse — and NOT propagate out of
    `/ask`. The verify node got this guard; assess didn't until the 2026-06-07 audit."""

    async def _raise(**_kw: object) -> tuple[object, int]:
        raise ModelCallError("guided-decode truncation in assess")

    monkeypatch.setattr("memex.agents.answering.complete_structured", _raise)
    state = AnswerState(query="q", reranked=[_vchunk("c1", "some content")])
    out = await assess(state)
    s = out["sufficiency"]
    assert isinstance(s, SufficiencyAssessment)
    assert s.sufficient is False  # fail-closed → route to refuse


@pytest.mark.asyncio
async def test_assess_relevance_fails_open_on_model_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An `assess_relevance` `ModelCallError` must FAIL OPEN — `responsive=True` — because the
    claims are ALREADY grounded by verify (HARD-gate-safe); a gate failure must never manufacture
    a refusal of an already-grounded answer. Must not propagate out of `/ask`."""

    async def _raise(**_kw: object) -> tuple[object, int]:
        raise ModelCallError("guided-decode truncation in assess_relevance")

    monkeypatch.setattr("memex.agents.answering.complete_structured", _raise)
    state = AnswerState(
        query="q",
        draft=DraftAnswer(
            summary="s", claims=[CitedClaim(claim="a", source_chunk_id="c1", confidence="high")]
        ),
        verification=VerificationResult(grounded=[0], ungrounded=[]),
        reranked=[_vchunk("c1", "content")],
    )
    out = await assess_relevance(state)
    r = out["relevance"]
    assert isinstance(r, RelevanceAssessment)
    assert r.responsive is True  # fail-open → ship the already-grounded claims


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_numeric_backstop_demotes_fabricated_table_aggregate(fake_llm: FakeLLM) -> None:
    """The kill target: a fabricated SUM cited to a table chunk → demoted."""
    v = await _run_verify(
        fake_llm,
        claim="The total value of stock options granted to directors in fiscal 2026 is $159,748,343.",
        chunk=_vchunk("t1", _EXEC_TABLE_TEXT),
    )
    assert v.grounded == []
    assert 0 in v.ungrounded
    assert v.ungrounded_reasons  # a demotion reason is recorded


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_numeric_backstop_keeps_verbatim_cell(fake_llm: FakeLLM) -> None:
    """A verbatim cell read ($321,309 present in the table) stays grounded."""
    text = (
        "| Director | Fees | Stock | Total |\n|---|---|---|---|\n"
        "| Ochoa | 42,500 | 278,809 | 321,309 |\n"
        "[table-rows]\n[Comp] Director=Ochoa, Total=321,309\n"
    )
    v = await _run_verify(
        fake_llm,
        claim="The lowest director total compensation was $321,309 (Ellen Ochoa).",
        chunk=_vchunk("t2", text),
    )
    assert v.grounded == [0]
    assert v.ungrounded == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_numeric_backstop_keeps_millions_denomination(fake_llm: FakeLLM) -> None:
    """A '$16,042 million' claim grounds in a '16,042' $-millions cell."""
    text = "| Segment | Year |\n|---|---|\n| Gaming | 16,042 |\n[table-rows]\n[Seg] Gaming=16,042\n"
    v = await _run_verify(
        fake_llm,
        claim="Gaming revenue was $16,042 million in fiscal 2026.",
        chunk=_vchunk("t3", text),
    )
    assert v.grounded == [0]


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_numeric_backstop_exempts_sql_synthetic_chunk(fake_llm: FakeLLM) -> None:
    """A legit recompute-gated aggregate cites the synthetic #sql0001 chunk and
    is EXEMPT — even though its >=1e6 value renders as unreadable %g exponential
    so a verbatim re-check would wrongly drop it (the suffix is the durable guard)."""
    text = "Aggregate result = 1.59748e+08 over 10 rows:\n[Comp] Name=Huang, Total=19,166,424\n"
    v = await _run_verify(
        fake_llm,
        claim="The total fees paid to directors was $159,748,343.",
        chunk=_vchunk("d1#sql0001", text),
    )
    assert v.grounded == [0]
    assert v.ungrounded == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_numeric_backstop_exempts_prose_chunk(fake_llm: FakeLLM) -> None:
    """A rounded headline ('$216 billion') cited to a PROSE chunk (no table) is
    exempt — table-presence narrowing protects scale-reformatted prose answers."""
    text = "Revenue grew 65 percent to $216 billion in fiscal 2026, a record year."
    v = await _run_verify(
        fake_llm,
        claim="NVIDIA's total revenue in fiscal 2026 was $216 billion.",
        chunk=_vchunk("p1", text),
    )
    assert v.grounded == [0]


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_numeric_backstop_ignores_small_count_claim(fake_llm: FakeLLM) -> None:
    """A derived small count (< 1e4, no large figure) cited to a table is out of
    scope — the LLM verdict stands (no false-drop on enumerated counts)."""
    text = "| Tenet | Name |\n|---|---|\n| 1 | Data sources |\n[table-rows]\n[Z] Tenet=1\n"
    v = await _run_verify(
        fake_llm,
        claim="Zero trust defines 7 core tenets.",
        chunk=_vchunk("t4", text),
    )
    assert v.grounded == [0]


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_numeric_backstop_kill_switch_off_keeps_fabrication(fake_llm: FakeLLM) -> None:
    """With the backstop disabled (AnswerState flag False) the pre-fix behaviour
    is restored: the fabricated aggregate stays grounded."""
    v = await _run_verify(
        fake_llm,
        claim="The total value of stock options granted to directors in fiscal 2026 is $159,748,343.",
        chunk=_vchunk("t1", _EXEC_TABLE_TEXT),
        backstop=False,
    )
    assert v.grounded == [0]


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_numeric_backstop_fabrication_refuses_end_to_end(
    monkeypatch: pytest.MonkeyPatch, fake_llm: FakeLLM
) -> None:
    """End-to-end: a fabricated table aggregate is the ONLY claim → demoted →
    zero grounded → answer_query REFUSES (the ar-16 HARD-gate restoration)."""
    table_chunk = _vchunk("t1", _EXEC_TABLE_TEXT)

    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return [table_chunk]

    async def _rerank(query: str, candidates: list[Chunk], top_k: int = 10) -> list[Chunk]:
        return list(candidates[:top_k])

    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    monkeypatch.setattr(
        "memex.agents.answering.render_prompt", lambda name, **_kw: f"[fake {name} prompt]"
    )

    fake_llm.respond(
        "assess_sufficiency",
        SufficiencyAssessment,
        SufficiencyAssessment(sufficient=True, reason="A comp table is present"),
    )
    fake_llm.respond(
        "answer",
        DraftAnswer,
        DraftAnswer(
            summary="The total value of stock options granted to directors is $159,748,343.",
            claims=[
                CitedClaim(
                    claim="The total value of stock options granted to directors in fiscal 2026 is $159,748,343.",
                    source_chunk_id="t1",
                    confidence="high",
                ),
            ],
        ),
    )
    fake_llm.respond(
        "verify_grounding",
        VerificationResult,
        VerificationResult(grounded=[0], ungrounded=[], ungrounded_reasons=[]),
    )

    response = await answer_query("What was the total value of stock options granted to directors?")

    assert response.answered is False, "fabricated aggregate must be demoted → refused"
    assert response.claims == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_numeric_backstop_keeps_rounded_plus_exact_figure(fake_llm: FakeLLM) -> None:
    """A claim asserting a rounded headline AND an exact table figure (the gte-05
    shape: ~800M alongside the verbatim 788M) is KEPT — the gate demotes only
    when EVERY scoped figure is unsupported, so the verbatim 788M saves it."""
    text = "| Source | Pairs |\n|---|---|\n| Total | 788M |\n[table-rows]\n[T] Total=788M\n"
    v = await _run_verify(
        fake_llm,
        claim="We used ~800M text pairs for pre-training (Table 1 totals 788M).",
        chunk=_vchunk("t5", text),
    )
    assert v.grounded == [0]


# ======================================================================
# Name-only grounding backstop (2026-06-03): the 5th deterministic verify-node
# filter. The LLM grounds a BEHAVIORAL claim against a chunk that only NAMES the
# entity (a bare list/heading); the backstop demotes it. Fail-open + demotion-only
# ⇒ membership/value/unknown claims are KEPT (never over-refuses).
# ======================================================================

_NAME_LIST_CHUNK_TEXT = (
    "### Contrôle d'accès\n"
    "- Role-Based Access Control (RBAC)\n"
    "- Attribute-Based Access Control (ABAC)\n"
    "- Mandatory Access Control (MAC)\n"
)


async def _run_verify_name_only(
    fake_llm: FakeLLM, *, claim: str, chunk: Chunk, backstop: bool = True
) -> VerificationResult:
    """Drive the verify node: the LLM grounds the single claim; the NAME-ONLY backstop then
    decides keep-vs-demote. Numeric backstop off to isolate (a name-list isn't a table anyway)."""
    fake_llm.respond(
        "verify_grounding",
        VerificationResult,
        VerificationResult(grounded=[0], ungrounded=[], ungrounded_reasons=[]),
    )
    state = AnswerState(
        query="q",
        draft=DraftAnswer(
            summary="s",
            claims=[CitedClaim(claim=claim, source_chunk_id=chunk.chunk_id, confidence="high")],
        ),
        reranked=[chunk],
        numeric_grounding_backstop=False,
        name_only_grounding_backstop=backstop,
    )
    out = await verify(state)
    result = out["verification"]
    assert isinstance(result, VerificationResult)
    return result


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_name_only_backstop_demotes_behavioral_claim(fake_llm: FakeLLM) -> None:
    """The kill target: a BEHAVIORAL claim cited to a bare name-list → demoted (→ refuse)."""
    v = await _run_verify_name_only(
        fake_llm,
        claim="RBAC assigns permissions based on a user's job function.",
        chunk=_vchunk("nl1", _NAME_LIST_CHUNK_TEXT),
    )
    assert v.grounded == []
    assert 0 in v.ungrounded
    assert v.ungrounded_reasons


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_name_only_backstop_keeps_membership_claim(fake_llm: FakeLLM) -> None:
    """A MEMBERSHIP/existence claim IS grounded by a name-list → KEPT (the over-refusal trap)."""
    v = await _run_verify_name_only(
        fake_llm,
        claim="RBAC is one of the access control models listed.",
        chunk=_vchunk("nl2", _NAME_LIST_CHUNK_TEXT),
    )
    assert v.grounded == [0]
    assert v.ungrounded == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_name_only_backstop_keeps_table_value_claim(fake_llm: FakeLLM) -> None:
    """A behavioral-sounding claim cited to a TABLE chunk is never name-only → KEPT (Table-RAG)."""
    text = "| Model | Default priority |\n|---|---|\n| RBAC | 1 |\n[table-rows]\n[P] RBAC=1\n"
    v = await _run_verify_name_only(
        fake_llm,
        claim="RBAC uses a default priority of 1.",  # has a behavioral marker, but the chunk is a table
        chunk=_vchunk("tb1", text),
    )
    assert v.grounded == [0]


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_prompt")
async def test_name_only_backstop_kill_switch_off_keeps_behavioral(fake_llm: FakeLLM) -> None:
    """Kill-switch off → the behavioral name-list claim stays grounded (the prior behavior)."""
    v = await _run_verify_name_only(
        fake_llm,
        claim="RBAC assigns permissions based on a user's job function.",
        chunk=_vchunk("nl3", _NAME_LIST_CHUNK_TEXT),
        backstop=False,
    )
    assert v.grounded == [0]


# ======================================================================
# Answer-node context-overflow degradation (2026-05-31)
# A vLLM context-length 400 (rendered chunks + output reservation exceed the
# window) must NOT abort the run: the answer node drops the lowest-ranked chunk
# and retries (verify still grounds against the FULL reranked set), and refuses
# via an empty draft if even the top chunk overflows.
# ======================================================================

_OVERFLOW = ModelCallError(
    "vLLM call failed: Error code: 400 - This model's maximum context length is 6144 tokens."
)


def test_is_context_overflow_detector() -> None:
    from memex.agents.answering import _is_context_overflow

    assert _is_context_overflow(_OVERFLOW) is True
    assert _is_context_overflow(ModelCallError("vLLM call failed: Error code: 500 - boom")) is False


@pytest.mark.asyncio
async def test_answer_context_overflow_retries_with_fewer_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overflow on the full chunk set → drop the lowest-ranked chunk + retry;
    the draft is produced (not crashed) once it fits."""
    chunks = [_doc_chunk(f"{i:010x}", "d", "D", text="chunk " * 30) for i in range(4)]
    calls = {"n": 0}

    async def fake_call(*, prompt: object, schema: type, **_kw: object) -> tuple[object, int]:
        calls["n"] += 1
        if calls["n"] <= 2:  # 4 chunks then 3 chunks overflow; 2 chunks fits
            raise _OVERFLOW
        return DraftAnswer(summary="ok", claims=[]), 5

    monkeypatch.setattr("memex.agents.answering.complete_structured", fake_call)
    result = await answer(AnswerState(query="q", reranked=chunks))
    assert calls["n"] == 3  # 4 → 3 → 2 (fits)
    assert result["draft"].summary == "ok"


@pytest.mark.asyncio
async def test_answer_context_overflow_single_chunk_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If even the single top chunk overflows, refuse via an EMPTY draft (verify
    short-circuits → refuse) rather than propagate the crash."""
    chunks = [_doc_chunk(f"{i:010x}", "d", "D", text="chunk " * 30) for i in range(3)]

    async def always_overflow(*, prompt: object, schema: type, **_kw: object) -> tuple[object, int]:
        raise _OVERFLOW

    monkeypatch.setattr("memex.agents.answering.complete_structured", always_overflow)
    result = await answer(AnswerState(query="q", reranked=chunks))
    assert result["draft"].claims == []  # empty draft → route_after_verify refuses


@pytest.mark.asyncio
async def test_answer_non_overflow_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-overflow ModelCallError is a real failure — it must NOT be masked
    as a degradation."""
    chunks = [_doc_chunk("aaaaaaaaaa", "d", "D", text="x")]

    async def err500(*, prompt: object, schema: type, **_kw: object) -> tuple[object, int]:
        raise ModelCallError("vLLM call failed: Error code: 500 - boom")

    monkeypatch.setattr("memex.agents.answering.complete_structured", err500)
    with pytest.raises(ModelCallError):
        await answer(AnswerState(query="q", reranked=chunks))


@pytest.mark.asyncio
async def test_answer_context_overflow_preserves_synthetic_sql_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The synthetic Table-RAG `#sql0001` chunk is appended LAST to reranked but
    carries the aggregate/superlative answer — the overflow drop must remove the
    lowest-ranked REAL chunk and NEVER the synthetic (regressed ar-14/ar-15 under
    overflow at top_k=5/6144)."""
    reals = [_doc_chunk(f"{i:010x}", "d", "D", text="chunk " * 30) for i in range(4)]
    synthetic = _doc_chunk("sql0001", "d", "D", text="SUM of Fees = 956250 over 12 rows")
    chunks = [*reals, synthetic]  # synthetic appended last, as query_tables does
    calls = {"n": 0}
    synthetic_present: list[bool] = []

    async def fake_call(*, prompt: object, schema: type, **_kw: object) -> tuple[object, int]:
        calls["n"] += 1
        rendered = prompt if isinstance(prompt, str) else " ".join(str(m) for m in prompt)
        synthetic_present.append("956250" in rendered)
        if calls["n"] <= 3:  # overflow until the lowest 3 reals are dropped
            raise _OVERFLOW
        return DraftAnswer(summary="ok", claims=[]), 5

    monkeypatch.setattr("memex.agents.answering.complete_structured", fake_call)
    result = await answer(AnswerState(query="q", reranked=chunks))
    assert result["draft"].summary == "ok"
    # The synthetic chunk survived EVERY attempt (never dropped).
    assert all(synthetic_present), synthetic_present


# ---- Usage-intent rerank demotion at the `rerank` node (ADR-0021 / audits/14) ----


def _code_chunk(cid: str, heading: str, score: float, text: str = "code body") -> Chunk:
    return Chunk(
        chunk_id=cid,
        document_id="d",
        document_title="codex-rs",
        text=text,
        heading_path=[heading],
        rerank_score=score,
    )


async def _fake_score_rerank(query: str, candidates: list[Chunk], top_k: int = 10) -> list[Chunk]:
    """Stand-in cross-encoder: sort by the chunk's preset `rerank_score`, slice to top_k."""
    return sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)[:top_k]


def _usage_candidates() -> list[Chunk]:
    # The real mechanism (audits/14): X's definition (0.95) + a test (0.87) outrank the caller (0.68).
    return [
        _code_chunk("def", "fn is_known_safe_command", 0.95),
        Chunk(
            chunk_id="test",
            document_id="d",
            document_title="codex-rs",
            text="assert!(is_known_safe_command(c))",
            heading_path=["mod tests", "known_safe_examples"],
            rerank_score=0.87,
        ),
        _code_chunk(
            "caller", "fn assess_command_safety", 0.68, text="if is_known_safe_command(c) {}"
        ),
        _code_chunk("other", "enum AskForApproval", 0.50),
    ]


async def test_rerank_usage_intent_demotes_definition_and_tests_out_of_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A "which function calls X" query demotes X's definition + test chunks below the top_k
    cut, so the caller surfaces into the answer node's window."""
    from memex.agents.answering import rerank
    from memex.core.config import MemexSettings, set_settings

    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _fake_score_rerank)
    monkeypatch.setenv("MEMEX_RERANK_TOP_K", "2")
    monkeypatch.setenv("MEMEX_AGENTS__USAGE_INTENT_DEMOTION_ENABLED", "true")  # opt-in (default OFF)
    set_settings(MemexSettings())
    try:
        out = await rerank(
            AnswerState(
                query="Which function calls is_known_safe_command?",
                candidates=_usage_candidates(),
            )
        )
        ids = [c.chunk_id for c in out["reranked"]]
        # def (top raw score) + test demoted OUT of the top-2 window; the caller surfaces.
        assert ids == ["caller", "other"]
        assert "def" not in ids and "test" not in ids
    finally:
        monkeypatch.delenv("MEMEX_AGENTS__USAGE_INTENT_DEMOTION_ENABLED", raising=False)
        set_settings(MemexSettings())


async def test_rerank_usage_intent_kill_switch_keeps_plain_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag OFF → the rerank node is byte-identical to the plain cross-encoder order."""
    from memex.agents.answering import rerank
    from memex.core.config import MemexSettings, set_settings

    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _fake_score_rerank)
    monkeypatch.setenv("MEMEX_RERANK_TOP_K", "2")
    monkeypatch.setenv("MEMEX_AGENTS__USAGE_INTENT_DEMOTION_ENABLED", "false")
    set_settings(MemexSettings())
    try:
        out = await rerank(
            AnswerState(
                query="Which function calls is_known_safe_command?",
                candidates=_usage_candidates(),
            )
        )
        # Unfixed order: the definition + test occupy the top-2 window (the bug).
        assert [c.chunk_id for c in out["reranked"]] == ["def", "test"]
    finally:
        monkeypatch.delenv("MEMEX_AGENTS__USAGE_INTENT_DEMOTION_ENABLED", raising=False)
        set_settings(MemexSettings())


async def test_rerank_definition_query_is_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even with the lever ENABLED, a DEFINITION query ("what does X do") does not fire the
    detector → plain order; the definition rightly stays at the top of the window."""
    from memex.agents.answering import rerank
    from memex.core.config import MemexSettings, set_settings

    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _fake_score_rerank)
    monkeypatch.setenv("MEMEX_RERANK_TOP_K", "2")
    monkeypatch.setenv("MEMEX_AGENTS__USAGE_INTENT_DEMOTION_ENABLED", "true")  # opt-in (default OFF)
    set_settings(MemexSettings())
    try:
        out = await rerank(
            AnswerState(
                query="What does the is_known_safe_command function do?",
                candidates=_usage_candidates(),
            )
        )
        assert [c.chunk_id for c in out["reranked"]] == ["def", "test"]
    finally:
        monkeypatch.delenv("MEMEX_AGENTS__USAGE_INTENT_DEMOTION_ENABLED", raising=False)
        set_settings(MemexSettings())


# ---- M1: verify-time citation RETARGET (audit-15, promote-only) ----


def _retarget_fixture(_mp: object, *, support_in_sibling: bool):
    """Two-chunk window: the draft cites c1 (no support); c2 carries the support iff
    support_in_sibling. The fake verify grounds ONLY a 1-claim probe whose rendered
    prompt contains c2's marker text."""
    c1 = Chunk(chunk_id="d#c1", document_id="d", document_title="D", text="related prose, no figure")
    c2 = Chunk(chunk_id="d#c2", document_id="d", document_title="D", text="the overhead is 1.6x lower MARKER")
    calls = {"verify": 0, "probes": 0}

    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return [c1, c2]

    async def _rerank(query: str, candidates: list[Chunk], top_k: int = 10) -> list[Chunk]:
        return list(candidates[:top_k])

    async def fake_call(*, prompt: object, schema: type, **_kw: object) -> tuple[object, int]:
        name = schema.__name__
        text = prompt if isinstance(prompt, str) else " ".join(str(m) for m in prompt)
        if name == "SufficiencyAssessment":
            return SufficiencyAssessment(sufficient=True, reason="ok"), 3
        if name == "DraftAnswer":
            return (
                DraftAnswer(
                    summary="The overhead is 1.6x lower.",
                    claims=[CitedClaim(claim="The overhead is 1.6x lower.", source_chunk_id="d#c1", confidence="high")],
                ),
                5,
            )
        if name == "VerificationResult":
            calls["verify"] += 1
            if "MARKER" in text and "d#c1" not in text:  # the 1-claim retarget probe vs c2
                calls["probes"] += 1
                ok = support_in_sibling
                return schema(grounded=[0] if ok else [], ungrounded=[] if ok else [0]), 4
            return schema(grounded=[], ungrounded=[0]), 4  # main verify: cited pair fails
        if name == "RelevanceAssessment":
            return RelevanceAssessment(responsive=True, reason="on topic"), 2
        raise AssertionError(f"unexpected schema {name}")

    return c1, c2, calls, _hybrid, _rerank, fake_call


async def test_retarget_promotes_and_rewrites_citation(monkeypatch: pytest.MonkeyPatch) -> None:
    *_, calls, _hybrid, _rerank, fake_call = _retarget_fixture(monkeypatch, support_in_sibling=True)
    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    monkeypatch.setattr("memex.agents.answering.complete_structured", fake_call)
    monkeypatch.delenv("MEMEX_AGENTS__CITATION_RETARGET_ENABLED", raising=False)
    resp = await answer_query("How much lower is the overhead?", max_regenerate_attempts=0)
    assert resp.answered is True
    assert resp.claims and resp.claims[0].source_chunk_id == "d#c2"  # citation REWRITTEN
    assert calls["probes"] >= 1


async def test_retarget_no_sibling_support_still_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    *_, calls, _hybrid, _rerank, fake_call = _retarget_fixture(monkeypatch, support_in_sibling=False)
    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    monkeypatch.setattr("memex.agents.answering.complete_structured", fake_call)
    resp = await answer_query("How much lower is the overhead?", max_regenerate_attempts=0)
    assert resp.answered is False  # promote-only: no support, no promotion
    assert calls["probes"] >= 1


async def test_retarget_waits_for_regenerate_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    *_, calls, _hybrid, _rerank, fake_call = _retarget_fixture(monkeypatch, support_in_sibling=True)
    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    monkeypatch.setattr("memex.agents.answering.complete_structured", fake_call)
    resp = await answer_query("How much lower is the overhead?", max_regenerate_attempts=1)
    assert resp.answered is True
    # verify ran twice (attempt 0: no probe, regen; attempt 1 = final: probed)
    assert calls["probes"] == 1
    assert calls["verify"] >= 2 + calls["probes"]


async def test_retarget_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    from memex.core.config import MemexSettings, set_settings

    *_, calls, _hybrid, _rerank, fake_call = _retarget_fixture(monkeypatch, support_in_sibling=True)
    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    monkeypatch.setattr("memex.agents.answering.complete_structured", fake_call)
    monkeypatch.setenv("MEMEX_AGENTS__CITATION_RETARGET_ENABLED", "false")
    set_settings(MemexSettings())
    try:
        resp = await answer_query("How much lower is the overhead?", max_regenerate_attempts=0)
        assert resp.answered is False
        assert calls["probes"] == 0  # the lever never fired
    finally:
        monkeypatch.delenv("MEMEX_AGENTS__CITATION_RETARGET_ENABLED", raising=False)
        set_settings(MemexSettings())


# ---- M3: relevance world-knowledge override (audit-15) ----


async def _wk_fixture(monkeypatch: pytest.MonkeyPatch, *, reason: str) -> object:
    c1 = Chunk(chunk_id="d#c1", document_id="d", document_title="D",
               text="The notes show three stages: source code, compile, machine code.")

    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return [c1]

    async def _rerank(query: str, candidates: list[Chunk], top_k: int = 10) -> list[Chunk]:
        return list(candidates[:top_k])

    async def fake_call(*, prompt: object, schema: type, **_kw: object) -> tuple[object, int]:
        name = schema.__name__
        if name == "SufficiencyAssessment":
            return SufficiencyAssessment(sufficient=True, reason="ok"), 3
        if name == "DraftAnswer":
            return DraftAnswer(summary="Three stages: source, compile, machine code.",
                               claims=[CitedClaim(claim="The stages are source code, compile, machine code.",
                                                  source_chunk_id="d#c1", confidence="high")]), 5
        if name == "VerificationResult":
            return schema(grounded=[0], ungrounded=[]), 4
        if name == "RelevanceAssessment":
            return RelevanceAssessment(responsive=False, reason=reason), 2
        raise AssertionError(name)

    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    monkeypatch.setattr("memex.agents.answering.complete_structured", fake_call)
    return await answer_query("What are the stages shown in the notes?")


async def test_world_knowledge_nonresponsive_is_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = await _wk_fixture(
        monkeypatch,
        reason="The answer lists three stages instead of the standard four described in C++ documentation.",
    )
    assert resp.answered is True  # the override ships the grounded answer


async def test_topic_mismatch_nonresponsive_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = await _wk_fixture(
        monkeypatch,
        reason="The question asks about linker flags but the answer describes compilation stages.",
    )
    assert resp.answered is False  # a legit conflation still refuses


async def test_world_knowledge_override_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    from memex.core.config import MemexSettings, set_settings

    monkeypatch.setenv("MEMEX_AGENTS__RELEVANCE_WORLD_KNOWLEDGE_GUARD_ENABLED", "false")
    set_settings(MemexSettings())
    try:
        resp = await _wk_fixture(
            monkeypatch,
            reason="The answer lists three stages instead of the standard four described in C++ documentation.",
        )
        assert resp.answered is False  # without the guard, the gate's vote stands
    finally:
        monkeypatch.delenv("MEMEX_AGENTS__RELEVANCE_WORLD_KNOWLEDGE_GUARD_ENABLED", raising=False)
        set_settings(MemexSettings())


# ---- M2: denial-reframe retry (audit-15) ----


async def _denial_fixture(monkeypatch: pytest.MonkeyPatch, *, retry_has_claims: bool,
                          first_summary: str) -> tuple[object, dict[str, int]]:
    c1 = Chunk(chunk_id="d#c1", document_id="d", document_title="GTE paper",
               text="Training was conducted on up to 8 NVIDIA A100 GPUs with fp16.")
    calls = {"draft": 0}

    async def _hybrid(query: str, k: int = 50) -> list[Chunk]:
        return [c1]

    async def _rerank(query: str, candidates: list[Chunk], top_k: int = 10) -> list[Chunk]:
        return list(candidates[:top_k])

    async def fake_call(*, prompt: object, schema: type, **_kw: object) -> tuple[object, int]:
        name = schema.__name__
        if name == "SufficiencyAssessment":
            return SufficiencyAssessment(sufficient=True, reason="ok"), 3
        if name == "DraftAnswer":
            calls["draft"] += 1
            if calls["draft"] == 1:
                return DraftAnswer(summary=first_summary, claims=[]), 5
            if retry_has_claims:
                return DraftAnswer(summary="GTE was trained on up to 8 NVIDIA A100 GPUs.",
                                   claims=[CitedClaim(claim="GTE was trained on up to 8 NVIDIA A100 GPUs.",
                                                      source_chunk_id="d#c1", confidence="high")]), 5
            return DraftAnswer(summary="No literal answer in chunks.", claims=[]), 5
        if name == "VerificationResult":
            return schema(grounded=[0], ungrounded=[]), 4
        if name == "RelevanceAssessment":
            return RelevanceAssessment(responsive=True, reason="on topic"), 2
        raise AssertionError(name)

    monkeypatch.setattr("memex.agents.answering.hybrid_search", _hybrid)
    monkeypatch.setattr("memex.agents.answering.cross_encoder_rerank", _rerank)
    monkeypatch.setattr("memex.agents.answering.complete_structured", fake_call)
    resp = await answer_query("What GPUs was GTE trained on?")
    return resp, calls


DENIAL = ("The chunks do not state which specific GPUs were used, only that training "
          "was conducted on up to 8 NVIDIA A100 GPUs")


async def test_denial_reframe_retry_recovers_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    resp, calls = await _denial_fixture(monkeypatch, retry_has_claims=True, first_summary=DENIAL)
    assert calls["draft"] == 2  # exactly ONE retry
    assert resp.answered is True
    assert resp.claims and "A100" in resp.claims[0].claim


async def test_denial_retry_empty_second_draft_still_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    resp, calls = await _denial_fixture(monkeypatch, retry_has_claims=False, first_summary=DENIAL)
    assert calls["draft"] == 2
    assert resp.answered is False  # the retry's honest empty draft stands


async def test_true_refusal_summary_does_not_trigger_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    resp, calls = await _denial_fixture(monkeypatch, retry_has_claims=True,
                                        first_summary="No literal answer in chunks.")
    assert calls["draft"] == 1  # no retry — terminal refusal preserved
    assert resp.answered is False


async def test_denial_retry_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    from memex.core.config import MemexSettings, set_settings

    monkeypatch.setenv("MEMEX_AGENTS__DENIAL_REFRAME_RETRY_ENABLED", "false")
    set_settings(MemexSettings())
    try:
        resp, calls = await _denial_fixture(monkeypatch, retry_has_claims=True, first_summary=DENIAL)
        assert calls["draft"] == 1
        assert resp.answered is False
    finally:
        monkeypatch.delenv("MEMEX_AGENTS__DENIAL_REFRAME_RETRY_ENABLED", raising=False)
        set_settings(MemexSettings())
