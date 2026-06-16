"""The provenance-scope backstop (audit-18 §9) — extraction, matching, and the
deterministic doc-identity verdict.

The provenance-class summary-scope breach: a query NAMES its source ("According to
the developer guidelines, …") and the shipped answer cites a chunk from a different
document (tg-13: `tui/src/log_layer.rs`). Audit-18 measured every semantic arm
non-separating here; the deterministic check is the lever that works. These tests pin
the measured probe behavior: tg-13 FIRES; tg-01/nist true-provenance PASS; every
ambiguous shape (artifact nouns, generic sources, topic frames, years) fails OPEN.
"""

from __future__ import annotations

import pytest

from memex.agents.answering import (
    AnswerState,
    CitedClaim,
    DraftAnswer,
    VerificationResult,
    _provenance_scope_violation,
)
from memex.core.text import extract_provenance_source, provenance_tokens_match
from memex.core.types import Chunk

# ---------------------------------------------------------------- extraction


class TestExtractProvenanceSource:
    def test_according_to_named_doc(self) -> None:
        got = extract_provenance_source(
            "According to the developer guidelines, what is the exact maximum line "
            "length in characters that the coding standards enforce?"
        )
        assert got == ("the developer guidelines", ["developer", "guidelines"])

    def test_doc_number_kept_short_token_dropped(self) -> None:
        """`SP 800-207` → 'sp' dropped (<3 chars, substring-hazard), '800-207' kept
        (the specific doc number that adjudicates)."""
        got = extract_provenance_source(
            "According to SP 800-207, who coined the term 'zero trust'?"
        )
        assert got == ("SP 800-207", ["800-207"])

    def test_artifact_noun_fails_open(self) -> None:
        """A figure/table/module reference is the #256 artifact-scope domain — the
        doc-identity check cannot adjudicate a sub-document artifact."""
        for q in (
            "According to the Compensation Committee process timeline figure, what did the CC do?",
            "According to the fiscal 2026 director compensation table, what was the total?",
            "Per the nvmath-python architectural figure, which libraries does it use?",
            "According to the NAT module, what kind of address does NAT translate?",
            "Selon le diagramme de configuration VLAN du commutateur, quel port sert de tronc?",
        ):
            assert extract_provenance_source(q) is None, q

    def test_generic_source_fails_open(self) -> None:
        """'the course' / 'the deck' name no specific document."""
        for q in (
            "According to the course, what is an operating system?",
            "According to the deck, what is the memory bandwidth of the AMD MI300X GPU?",
        ):
            assert extract_provenance_source(q) is None, q

    def test_bare_in_topic_frame_is_not_provenance(self) -> None:
        """'In X,' is a TOPIC frame in the measured query population, not provenance —
        excluded wholesale (the false-fire surface dwarfs the benefit)."""
        for q in (
            "In SP 800-207, the policy decision point is broken down into which components?",
            "In the Linux octal permission system, what numeric value is execute?",
            "In Memex's hybrid retrieval, how many fused candidates does the reranker take?",
        ):
            assert extract_provenance_source(q) is None, q

    def test_no_provenance_clause(self) -> None:
        assert (
            extract_provenance_source(
                "What was the gross margin of NVIDIA's Graphics segment in fiscal 2026?"
            )
            is None
        )

    def test_french_selon_named_source(self) -> None:
        got = extract_provenance_source(
            "Selon les directives de développement, quelle est la longueur maximale?"
        )
        assert got is not None
        assert "directives" in got[1]

    def test_year_only_source_fails_open(self) -> None:
        assert extract_provenance_source("According to the 2026, what changed?") is None

    def test_colloquial_generic_sources_fail_open(self) -> None:
        """The independent review's B2 catch: colloquial source words were probed
        FALSE-FIRE-CAPABLE against the live vault ('guide' substring-matches the
        guidelines doc; 'report' the 10-K; 'doc' project_doc.rs; 'user' 5 docs).
        Pinned: every one must yield NO adjudicable tokens."""
        for q in (
            "According to the docs, how do I mount a USB key?",
            "According to the guide, how do I mount a USB key?",
            "According to the user guide, how do I change permissions?",
            "According to the report, what was total revenue?",
            "According to the design doc, how does retrieval work?",
            "According to the spec, what is the chunk size?",
            "According to the readme, how do I install it?",
            "According to the study, what was the sample size?",
            "Per user, how much memory is allocated?",
            "According to the data, which quarter was strongest?",
        ):
            assert extract_provenance_source(q) is None, q

    def test_lone_alphabetic_token_fails_open(self) -> None:
        """The #256 single-token specificity gate: a lone usable token adjudicates
        only with a digit (doc-number shape). 'the NVIDIA filing' -> lone 'nvidia'
        (alphabetic) -> None; 'SP 800-207' -> lone '800-207' (digit) -> kept."""
        assert extract_provenance_source("According to NVIDIA, what was revenue?") is None
        got = extract_provenance_source("According to SP 800-207, who coined the term?")
        assert got is not None and got[1] == ["800-207"]

    def test_french_accents_tokenize_whole(self) -> None:
        """ASCII-only tokenization shattered 'résumé'->'sum', 'l'étude'->'tude'
        (which matched a real vault doc). Accented words must tokenize whole —
        and these generic FR sources then fail open."""
        for q in (
            "Selon le résumé du cours, quelle est la première phase ?",
            "D'après l'étude, quel est le résultat principal ?",
            "Selon le schéma du réseau, quel port est utilisé ?",
        ):
            assert extract_provenance_source(q) is None, q

    def test_french_specific_source_extracts_whole_tokens(self) -> None:
        got = extract_provenance_source(
            "Selon les directives de développement, quelle longueur maximale ?"
        )
        assert got is not None
        assert "développement" in got[1] and "directives" in got[1]


class TestExtractDocNameReference:
    """The EMBEDDED doc-class-noun recognizer (the cross-doc-scope fix): a doc-class noun
    + digit-bearing identifier is an adjudicable source ANYWHERE in the query."""

    def test_embedded_schedule_reference(self) -> None:
        # the held-out f8812-07 case: embedded "does the 2025 Schedule 8812 list" — the leading
        # "according to" clause does NOT match, the embedded recognizer does.
        got = extract_provenance_source(
            "What standard deduction amount does the 2025 Schedule 8812 list for head of household?"
        )
        assert got == ("Schedule 8812", ["8812"])  # class noun dropped; specific id kept

    def test_embedded_on_form_and_w_dash_number(self) -> None:
        assert extract_provenance_source("On the 2025 Schedule 8812, what is the credit?") == (
            "Schedule 8812", ["8812"])
        assert extract_provenance_source("What does Form W-4 Step 3 say?") == ("Form W-4", ["w-4"])
        assert extract_provenance_source("the standard deduction on Form 1040 for single filers") == (
            "Form 1040", ["1040"])

    def test_leading_clause_takes_precedence(self) -> None:
        # when BOTH a leading clause and an embedded ref are present, the leading wins (byte-identical
        # to the original behavior wherever the leading clause matched).
        got = extract_provenance_source("According to the developer guidelines, see Form 1040.")
        assert got == ("the developer guidelines", ["developer", "guidelines"])

    def test_comparison_cue_fails_open(self) -> None:
        # the comparison-CUE guard fires on a SINGLE-doc-ref comparison (the >=2-refs guard can't):
        # "differ"/"versus"/"relative to" name a doc to CONTRAST, not to scope to.
        for q in (
            "How does Form 1040 differ from the prior year?",
            "Form 1040 versus the worksheet — which applies?",
            "the deduction relative to Form 1040",
        ):
            assert extract_provenance_source(q) is None, q
        # a TWO-ref comparison fails open via the distinct-refs guard (separate mechanism, below)
        assert extract_provenance_source("Compare Form 1040 and Schedule 8812.") is None

    def test_two_distinct_doc_refs_fail_open(self) -> None:
        # spans two docs → ambiguous which is THE source → no-op (the >=2-distinct-refs guard, NOT
        # the comparison cue — "relate"/"differ" need not appear).
        assert extract_provenance_source("How does Schedule 8812 relate to Form 8995?") is None

    def test_year_id_fails_open(self) -> None:
        # a bare YEAR id is NON-scoping and substring-matches half the vault → dropped (the same
        # `_PROVENANCE_YEAR_RE` guard the leading clause applies; the embedded path must match it).
        assert extract_provenance_source("What were the main findings in publication 2024?") is None
        assert extract_provenance_source("Did Form 2025 change anything?") is None

    def test_topic_frame_and_bare_form_fail_open(self) -> None:
        # no doc-class-noun + digit-id → no embedded match (a topic frame is not provenance)
        assert extract_provenance_source("In the Linux octal permission system, what is 755?") is None
        # "form" with no digit-bearing id → no-op (the specificity rule)
        assert extract_provenance_source("How do I fill out the form correctly?") is None
        # common-English class-noun + a NON-id number → the <3-char/token filters drop it
        assert extract_provenance_source("I need to fill out form 3 times this week.") is None
        assert extract_provenance_source("Let's schedule 2 meetings about it.") is None

    def test_short_id_dropped(self) -> None:
        # a <3-char id ("17") is too unspecific to substring-adjudicate → no-op
        assert extract_provenance_source("What does Pub 17 say about deductions?") is None


# ---------------------------------------------------------------- matching


class TestProvenanceTokensMatch:
    def test_separator_normalized_doc_number(self) -> None:
        assert provenance_tokens_match(["800-207"], "0290d6ec-nist-sp-800-207 nist-sp-800-207")

    def test_any_token_clears_partial_title(self) -> None:
        """tg-01: the guidelines doc is titled just 'guidelines' — any-token."""
        assert provenance_tokens_match(["developer", "guidelines"], "f59860eb-guidelines guidelines")

    def test_no_match_on_foreign_doc(self) -> None:
        """tg-13: log_layer.rs carries neither 'developer' nor 'guidelines'."""
        assert not provenance_tokens_match(
            ["developer", "guidelines"],
            "cbb5e509-log-layer tui/src/log_layer.rs const _DEFAULT_MAX_LEN",
        )

    def test_heading_carried_source_matches(self) -> None:
        """A section titled 'Developer Guidelines' inside a larger doc IS true
        provenance — the cited blob includes heading_path."""
        assert provenance_tokens_match(
            ["developer", "guidelines"],
            "abc123-handbook Engineering Handbook Chapter 3 Developer Guidelines",
        )


# ---------------------------------------------------------------- the node verdict

_GUIDELINES_DOC = ("f59860eb-guidelines", "guidelines")
_LOG_LAYER_DOC = ("cbb5e509-log-layer", "tui/src/log_layer.rs")


class _FakeStore:
    def __init__(self, identities: list[tuple[str, str]]) -> None:
        self._identities = identities
        self.closed = False

    async def document_identities(self) -> list[tuple[str, str]]:
        return self._identities

    async def close(self) -> None:
        self.closed = True


def _patch_store(
    monkeypatch: pytest.MonkeyPatch,
    identities: list[tuple[str, str]],
    *,
    fake_settings: bool = True,
) -> _FakeStore:
    store = _FakeStore(identities)

    async def _open(_path: object) -> _FakeStore:
        return store

    monkeypatch.setattr("memex.index.fts_store.FTSStore.open", _open)
    if fake_settings:
        monkeypatch.setattr(
            "memex.core.config.get_settings",
            lambda: type("S", (), {"vault_path": "/nonexistent"})(),
        )
    return store


def _tg13_state(cited_doc: tuple[str, str], heading: list[str]) -> AnswerState:
    """The pinned tg-13 breach trace (docs/audits/data-17-scope-calibration/raw/
    tg13_full.json): the true grounded claim cited to the log_layer.rs chunk."""
    doc_id, title = cited_doc
    chunk = Chunk(
        chunk_id=f"{doc_id}#a270b0ee35",
        document_id=doc_id,
        document_title=title,
        text="## const _DEFAULT_MAX_LEN\n/// Maximum characters forwarded to the TUI.",
        heading_path=heading,
    )
    return AnswerState(
        query=(
            "According to the developer guidelines, what is the exact maximum line "
            "length in characters that the coding standards enforce?"
        ),
        reranked=[chunk],
        draft=DraftAnswer(
            summary=(
                "The developer guidelines specify a maximum line length of 120 "
                "characters for the TUI log layer."
            ),
            claims=[
                CitedClaim(
                    claim="The maximum line length enforced is 120 characters.",
                    source_chunk_id=f"{doc_id}#a270b0ee35",
                    confidence="high",
                )
            ],
        ),
        verification=VerificationResult(grounded=[0], ungrounded=[]),
    )


@pytest.mark.asyncio
async def test_tg13_breach_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    """THE kill target: 'according to the developer guidelines' cited to
    log_layer.rs, while a 'guidelines' doc exists in the vault → violation."""
    _patch_store(monkeypatch, [_GUIDELINES_DOC, _LOG_LAYER_DOC])
    reason = await _provenance_scope_violation(
        _tg13_state(_LOG_LAYER_DOC, ["const _DEFAULT_MAX_LEN"])
    )
    assert reason is not None
    assert "the developer guidelines" in reason
    assert "guidelines" in reason


@pytest.mark.asyncio
async def test_true_provenance_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """tg-01 shape: the same query class cited to the guidelines doc itself → None."""
    _patch_store(monkeypatch, [_GUIDELINES_DOC, _LOG_LAYER_DOC])
    reason = await _provenance_scope_violation(
        _tg13_state(_GUIDELINES_DOC, ["Coding standards"])
    )
    assert reason is None


@pytest.mark.asyncio
async def test_heading_carried_provenance_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The named source living in the CITED CHUNK's heading_path is true provenance
    even when the doc title doesn't carry it."""
    _patch_store(monkeypatch, [_GUIDELINES_DOC])
    reason = await _provenance_scope_violation(
        _tg13_state(("xyz-handbook", "Engineering Handbook"), ["Developer Guidelines"])
    )
    assert reason is None


@pytest.mark.asyncio
async def test_unadjudicable_source_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """X matching NO vault document → no verdict (the named source isn't checkable)."""
    _patch_store(monkeypatch, [_LOG_LAYER_DOC, ("abc-other", "Annual Review")])
    reason = await _provenance_scope_violation(
        _tg13_state(_LOG_LAYER_DOC, ["const _DEFAULT_MAX_LEN"])
    )
    assert reason is None


@pytest.mark.asyncio
async def test_store_error_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """An infrastructure failure must never manufacture a refusal."""

    async def _boom(_path: object) -> None:
        raise OSError("store unavailable")

    monkeypatch.setattr("memex.index.fts_store.FTSStore.open", _boom)
    monkeypatch.setattr(
        "memex.core.config.get_settings",
        lambda: type("S", (), {"vault_path": "/nonexistent"})(),
    )
    reason = await _provenance_scope_violation(
        _tg13_state(_LOG_LAYER_DOC, ["const _DEFAULT_MAX_LEN"])
    )
    assert reason is None


@pytest.mark.asyncio
async def test_dangling_cites_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """No cited chunk resolves in the window → no verdict."""
    _patch_store(monkeypatch, [_GUIDELINES_DOC])
    state = _tg13_state(_LOG_LAYER_DOC, [])
    state = state.model_copy(update={"reranked": []})  # window lost → dangling cite
    reason = await _provenance_scope_violation(state)
    assert reason is None


@pytest.mark.asyncio
async def test_no_provenance_query_is_settings_free(monkeypatch: pytest.MonkeyPatch) -> None:
    """A query with no provenance clause returns None before any store/settings
    access (the #256 cheap-detection-first pattern)."""

    async def _explode(_path: object) -> None:  # pragma: no cover — must not be hit
        raise AssertionError("store must not be opened")

    monkeypatch.setattr("memex.index.fts_store.FTSStore.open", _explode)
    state = _tg13_state(_LOG_LAYER_DOC, []).model_copy(
        update={"query": "What is the maximum line length?"}
    )
    assert await _provenance_scope_violation(state) is None


# ----------------------------------------- node wiring (assess_relevance integration)


@pytest.mark.asyncio
async def test_node_violation_refuses_without_llm_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a confirmed violation the node returns responsive=False DETERMINISTICALLY —
    the relevance LLM is never called, and (pinning the M3 interaction the review
    flagged) the world-knowledge override is structurally unreachable even when the
    deterministic reason happens to contain 'standard'-class words."""
    from memex.agents.answering import assess_relevance
    from memex.core.config import MemexSettings, set_settings

    set_settings(MemexSettings())
    _patch_store(
        monkeypatch,
        [("std-001", "Standard Operating Procedures guidelines doc")],
        fake_settings=False,  # the node reads the REAL settings (flag default-ON)
    )

    async def _explode(**_kw: object) -> tuple[object, int]:  # pragma: no cover
        raise AssertionError("relevance LLM must not be called on a violation")

    monkeypatch.setattr("memex.agents.answering.complete_structured", _explode)
    out = await assess_relevance(_tg13_state(_LOG_LAYER_DOC, ["const _DEFAULT_MAX_LEN"]))
    r = out["relevance"]
    assert r.responsive is False
    assert "according to" in r.reason


@pytest.mark.asyncio
async def test_node_kill_switch_skips_backstop(monkeypatch: pytest.MonkeyPatch) -> None:
    """provenance_scope_enabled=False -> the backstop never runs; the LLM verdict
    is used unchanged."""
    from memex.agents.answering import RelevanceAssessment, assess_relevance
    from memex.core.config import MemexSettings, set_settings

    settings = MemexSettings()
    settings.agents.provenance_scope_enabled = False
    set_settings(settings)

    async def _store_explode(_path: object) -> None:  # pragma: no cover
        raise AssertionError("store must not be opened with the kill-switch off")

    monkeypatch.setattr("memex.index.fts_store.FTSStore.open", _store_explode)

    async def _responsive(**_kw: object) -> tuple[RelevanceAssessment, int]:
        return RelevanceAssessment(responsive=True, reason="on topic"), 7

    monkeypatch.setattr("memex.agents.answering.complete_structured", _responsive)
    out = await assess_relevance(_tg13_state(_LOG_LAYER_DOC, ["const _DEFAULT_MAX_LEN"]))
    assert out["relevance"].responsive is True


# --------------------------------- the held-out f8812-07 cross-doc misattribution (audit-19+)

_F1040 = ("3d31c226-f1040", "f1040")
_F1040S8 = ("6936462d-f1040s8", "Schedule 8812 (Form 1040)")  # RETITLED (enabler A)
_F1040S8_SLUG = ("6936462d-f1040s8", "f1040s8")  # the pre-retitle slug


def _f8812_state(cited_doc: tuple[str, str]) -> AnswerState:
    """The held-out leak: the 1040's HoH standard deduction ($23,625) cited as Schedule 8812's
    (the synthetic chunk the bullet resolver injects, doc=1040), for an EMBEDDED doc-name query."""
    doc_id, title = cited_doc
    chunk = Chunk(
        chunk_id=f"{doc_id}#field0001",
        document_id=doc_id,
        document_title=title,
        text="Standard deduction for Head of household: $23,625",
        heading_path=[],
    )
    return AnswerState(
        query="What standard deduction amount does the 2025 Schedule 8812 list for head of household filers?",
        reranked=[chunk],
        draft=DraftAnswer(
            summary="The 2025 Schedule 8812 lists a standard deduction of $23,625 for head of household.",
            claims=[
                CitedClaim(
                    claim="Schedule 8812 lists a standard deduction of $23,625 for head of household filers.",
                    source_chunk_id=f"{doc_id}#field0001",
                    confidence="high",
                )
            ],
        ),
        verification=VerificationResult(grounded=[0], ungrounded=[]),
    )


@pytest.mark.asyncio
async def test_f8812_cross_doc_misattribution_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    """The robust layer: an EMBEDDED "Schedule 8812" reference whose grounded claim is cited to the
    1040 (a DIFFERENT vault doc) → violation. Needs BOTH enablers: the broadened extractor (catches
    the embedded ref) AND the retitle (so 'Schedule 8812' adjudicates to f1040s8)."""
    _patch_store(monkeypatch, [_F1040, _F1040S8])
    reason = await _provenance_scope_violation(_f8812_state(_F1040))
    assert reason is not None
    assert "Schedule 8812" in reason


@pytest.mark.asyncio
async def test_f8812_slug_title_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """WITHOUT the retitle (slug 'f1040s8'), 'Schedule 8812' matches no vault identity → no verdict.
    This pins WHY the corpus-side retitle is the enabling change, not optional polish."""
    _patch_store(monkeypatch, [_F1040, _F1040S8_SLUG])
    reason = await _provenance_scope_violation(_f8812_state(_F1040))
    assert reason is None


@pytest.mark.asyncio
async def test_f8812_legit_query_cited_to_schedule_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuine Schedule 8812 answer (cited to an f1040s8 chunk, whose identity now carries '8812')
    is true provenance → no verdict. This is why the retitle does NOT over-refuse f8812-01..06."""
    _patch_store(monkeypatch, [_F1040, _F1040S8])
    reason = await _provenance_scope_violation(_f8812_state(_F1040S8))
    assert reason is None


@pytest.mark.asyncio
async def test_numeric_token_does_not_collide_with_doc_id_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    """A numeric form-id token ("941") must NOT false-fire by substring-matching the random sha8
    prefix of an UNRELATED doc_id ("941" lives inside "2941523b-lib"). strip_content_hash drops the
    hash so only the stem/title carry identity. WITHOUT the strip this REFUSES a correct answer."""
    cited = ("abcd1234-employer-tax-guide", "Employer Tax Guide")  # the doc the answer came from
    collider = ("2941523b-lib", "cli/src/lib.rs")  # an unrelated doc whose HASH contains "941"
    _patch_store(monkeypatch, [cited, collider])
    doc_id, title = cited
    chunk = Chunk(
        chunk_id=f"{doc_id}#c", document_id=doc_id, document_title=title,
        text="Employer payroll taxes are reported quarterly.", heading_path=[],
    )
    state = AnswerState(
        query="What does Form 941 report for employer taxes?",
        reranked=[chunk],
        draft=DraftAnswer(
            summary="Form 941 reports employer payroll taxes quarterly.",
            claims=[CitedClaim(
                claim="Employer payroll taxes are reported quarterly.",
                source_chunk_id=f"{doc_id}#c", confidence="high",
            )],
        ),
        verification=VerificationResult(grounded=[0], ungrounded=[]),
    )
    assert await _provenance_scope_violation(state) is None  # no hash-collision false-fire
