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
