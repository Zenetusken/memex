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
    answer_query,
    compose,
    expand_graph,
    refuse,
    reset_compiled_graph,
    resolve_artifact_scope,
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
async def test_partial_grounded_ships_grounded_subset(fake_llm: FakeLLM) -> None:
    """#262 — the compound-question fix. A draft with one grounded claim (the
    answerable half) + one ungrounded claim (the half the corpus can't support)
    ships the GROUNDED subset: the ungrounded claim is dropped and the summary is
    rebuilt from the survivor, so no ungrounded assertion reaches the headline.
    `max_regenerate_attempts=0` exhausts the retry immediately → straight to the
    partial-ship branch."""
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
