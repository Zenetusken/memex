"""The code-only FTS term-query helpers (Phase-3 Lever A, ADR-0021 / docs/audits/13).

`query_has_code_identifier` is the per-query gate (term-WHOLE fires ONLY when the query NAMES a
code identifier, so a prose natural-language question keeps the unchanged phrase-wrap).
`build_code_term_match` is the validated WHOLE builder — underscore-aware, NEVER case-split,
stopword-dropped, individually-quoted atoms. `code_term_query_enabled` fails open to the safe
phrase-wrap.
"""

from __future__ import annotations

import pytest

from memex.core.config import MemexSettings, set_settings
from memex.core.errors import ConfigurationError
from memex.core.types import Chunk
from memex.index import code_query
from memex.index.code_query import (
    _chunk_defines_symbol,
    _is_test_chunk,
    build_code_term_match,
    code_term_query_enabled,
    detect_usage_intent,
    last_ident,
    query_has_code_identifier,
    reorder_for_usage_intent,
    usage_intent_demotion_enabled,
)
from memex.index.fts_store import _fts_match_expr


def _chunk(heading_path: list[str], chunk_id: str = "d#1", text: str = "body") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="d",
        document_title="exec/src/lib.rs",
        text=text,
        heading_path=heading_path,
    )


# ---------------------------------------------------------------------------
# query_has_code_identifier — the per-query gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "Which function calls is_known_safe_command?",  # snake_case
        "What is the ModelClient struct used for?",  # PascalCase
        "How does Config::load_with_overrides work?",  # trait method (snake)
        "What does FunctionCallOutputPayload serialize?",  # camelCase
        "What type of identifier is the correlation_id?",  # snake in prose (accepted)
        "Quels paquets dans la configuration pfSense ?",  # mixed-case product name (accepted FP)
    ],
)
def test_detector_fires_on_code_identifier(query: str) -> None:
    assert query_has_code_identifier(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "What does Smith argue about reflexivity?",  # pure prose
        "What is the recommended daily dosage of vitamin D?",  # prose
        "How does the agent decide whether a command is safe?",  # prose, no identifier
        "What is the FP128 numeric format?",  # ALL-CAPS+digits acronym does NOT fire
        "How is a Cisco PIX firewall configured?",  # ALL-CAPS acronym does NOT fire
    ],
)
def test_detector_silent_on_prose_and_allcaps_acronyms(query: str) -> None:
    assert query_has_code_identifier(query) is False


# ---------------------------------------------------------------------------
# build_code_term_match — the WHOLE builder
# ---------------------------------------------------------------------------


def test_snake_identifier_is_quoted_WHOLE_not_split() -> None:
    """The literal snake_case identifier is one quoted phrase — NOT its `_`-split parts."""
    out = build_code_term_match("Which function calls is_known_safe_command?")
    assert out == '"is_known_safe_command"'
    assert '"is"' not in out and '"known"' not in out  # no underscore-splitting


def test_camel_identifier_is_lowercased_whole_not_case_split() -> None:
    """PascalCase → the one whole lowercased token (unicode61 keeps it whole); NEVER case-split."""
    out = build_code_term_match("What does FunctionCallOutputPayload serialize?")
    atoms = out.split(" OR ")
    assert '"functioncalloutputpayload"' in atoms  # the whole identifier, lowercased
    assert '"serialize"' in atoms  # a legit content token survives (not a stopword)
    # a case-split atom can't match the defining chunk → never emit one
    assert '"function"' not in out and '"call"' not in out and '"payload"' not in out


def test_stopwords_and_structural_words_dropped() -> None:
    """which/function/calls/the/struct… are dropped; only the identifier survives."""
    out = build_code_term_match("What is the ModelClient struct used for?")
    assert out == '"modelclient"'


def test_atoms_are_individually_quoted_and_or_joined() -> None:
    out = build_code_term_match("How does Config::load_with_overrides work?")
    # config + load_with_overrides survive; each individually quoted, OR-joined, sorted.
    assert out == '"config" OR "load_with_overrides"'


def test_empty_when_only_stopwords() -> None:
    """A query of pure stopwords/structural words yields no atoms → '' (caller falls back)."""
    assert build_code_term_match("What is the function?") == ""
    assert build_code_term_match("How does it work?") == ""


def test_each_atom_is_quoted_so_fts_keywords_stay_literal() -> None:
    """Every atom is double-quoted, so a token that happens to be an FTS5 keyword is literal."""
    out = build_code_term_match("the near_miss handler")  # 'near' is an FTS5 keyword bare
    for atom in out.split(" OR "):
        assert atom.startswith('"') and atom.endswith('"')


# ---------------------------------------------------------------------------
# code_term_query_enabled — the fail-open flag
# ---------------------------------------------------------------------------


def test_flag_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMEX_AGENTS__CODE_TERM_QUERY_ENABLED", raising=False)
    set_settings(MemexSettings())
    assert code_term_query_enabled() is True


def test_flag_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMEX_AGENTS__CODE_TERM_QUERY_ENABLED", "false")
    set_settings(MemexSettings())
    assert code_term_query_enabled() is False
    monkeypatch.delenv("MEMEX_AGENTS__CODE_TERM_QUERY_ENABLED", raising=False)
    set_settings(MemexSettings())


def test_flag_fails_open_to_false_on_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config-read failure degrades to the safe phrase-wrap (False), never raises."""

    def _boom() -> object:
        raise ConfigurationError("no settings")

    monkeypatch.setattr(code_query, "get_settings", _boom)
    assert code_term_query_enabled() is False


# ---------------------------------------------------------------------------
# _fts_match_expr — the fts_store branch (default byte-identical; term path; fallback)
# ---------------------------------------------------------------------------


def test_match_expr_default_is_the_unchanged_phrase_wrap() -> None:
    """term_query=False (every non-/ask caller) is byte-identical to the historical phrase-wrap."""
    q = "Which function calls is_known_safe_command?"
    assert _fts_match_expr(q, term_query=False) == '"Which function calls is_known_safe_command?"'


def test_match_expr_term_query_uses_code_terms() -> None:
    q = "Which function calls is_known_safe_command?"
    assert _fts_match_expr(q, term_query=True) == '"is_known_safe_command"'


def test_match_expr_term_query_falls_back_to_phrase_when_no_identifier() -> None:
    """A term query with no surviving identifier degrades to the phrase-wrap, never empty MATCH."""
    q = "What is the function?"
    assert _fts_match_expr(q, term_query=True) == '"What is the function?"'


def test_match_expr_empty_query_is_none() -> None:
    assert _fts_match_expr("   ", term_query=True) is None
    assert _fts_match_expr("", term_query=False) is None


# ===========================================================================
# Usage-intent rerank demotion (ADR-0021 / docs/audits/14)
# ===========================================================================

# detect_usage_intent — fires on "which <noun> <verb> X" / "where is X <verb>", returns X.
# These are the real find-the-code usage queries (the data the detector was designed against).


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Which function calls is_known_safe_command?", "is_known_safe_command"),
        ("Which function invokes spawn_command_under_seatbelt?", "spawn_command_under_seatbelt"),
        ("Where is the EMBEDDED_INSTRUCTIONS constant used?", "EMBEDDED_INSTRUCTIONS"),
        ("Which function uses is_safe_to_call_with_exec?", "is_safe_to_call_with_exec"),
        ("Where is get_platform_sandbox called?", "get_platform_sandbox"),  # past-tense verb
        ("Which function creates a RolloutRecorder?", "RolloutRecorder"),
        ("Which struct owns the McpConnectionManager?", "McpConnectionManager"),
        ("Which struct holds the ConversationHistory?", "ConversationHistory"),
        ("Which enum wraps a verified ApplyPatchAction?", "ApplyPatchAction"),
        ("Which method uses ansi_escape_line to format command output?", "ansi_escape_line"),
    ],
)
def test_usage_intent_fires_and_extracts_symbol(query: str, expected: str) -> None:
    assert detect_usage_intent(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        # Definition intent — X's OWN definition IS the answer → must stay silent.
        "What does the assess_command_safety function do?",
        "What fields does the ExecParams struct define?",
        "What are the variants of the SandboxType enum?",
        "How is process_exec_tool_call implemented?",
        "What is the ModelClient struct used for?",  # "used" present, but "What …" = definition
        "How does Config::load_with_overrides work?",
        "What is the DEFAULT_TIMEOUT_MS constant set to?",
        # Big-function (definition intent) queries.
        "What does the run_main function in the exec crate do?",
        "How does ProgramSpec::check validate a command invocation?",
        # A counterfactual phrased as definition — silent (and would be a no-op anyway).
        "Where is the FrobnicateWidget struct defined?",  # 'defined' is NOT a usage verb
        # Pure prose — no code identifier, no usage frame.
        "How does the agent decide whether a shell command is safe to run?",
        "is_known_safe_command",  # bare identifier, no interrogative frame
    ],
)
def test_usage_intent_silent_on_definition_and_prose(query: str) -> None:
    assert detect_usage_intent(query) is None


def test_usage_intent_fires_harmlessly_on_counterfactual_usage_phrasing() -> None:
    """A "which function calls <nonexistent>" counterfactual fires + extracts the absent symbol.
    Harmless: no chunk defines it, so `reorder_for_usage_intent` demotes nothing (the HARD gate
    still refuses on an empty grounded set)."""
    assert detect_usage_intent("Which function calls validate_jwt_token?") == "validate_jwt_token"


# last_ident — recover the bare symbol name from an injected `## <kind> <name>` heading.


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("fn is_known_safe_command", "is_known_safe_command"),
        ("struct Session", "Session"),
        ("enum AskForApproval", "AskForApproval"),
        ("Session::client", "client"),  # method: strip the Type:: prefix
        ("mod tests", "tests"),
        ("const EMBEDDED_INSTRUCTIONS", "EMBEDDED_INSTRUCTIONS"),
        ("tests::bash_lc_safe_examples", "bash_lc_safe_examples"),
    ],
)
def test_last_ident(heading: str, expected: str) -> None:
    assert last_ident(heading) == expected


# _chunk_defines_symbol — only the chunk whose OWN deepest symbol IS X.


def test_chunk_defines_symbol_true_for_definition() -> None:
    assert _chunk_defines_symbol(_chunk(["fn is_known_safe_command"]), "is_known_safe_command")


def test_chunk_defines_symbol_false_for_caller() -> None:
    # The caller's heading is the CALLER's symbol, not X → never demoted.
    assert not _chunk_defines_symbol(_chunk(["fn assess_command_safety"]), "is_known_safe_command")


def test_chunk_defines_symbol_false_when_x_is_only_an_ancestor() -> None:
    # A method living under a parent named like X is NOT X's definition.
    assert not _chunk_defines_symbol(_chunk(["impl Foo", "fn bar"]), "Foo")


# _is_test_chunk — Rust test markers; precise (no false positive on attest_/contest).


@pytest.mark.parametrize(
    ("heading_path", "is_test"),
    [
        (["mod tests", "known_safe_examples"], True),
        (["tests::bash_lc_safe_examples"], True),
        (["fn test_unified_diff"], True),
        (["impl Foo", "tests", "case_one"], True),
        (["fn assess_command_safety"], False),
        (["fn attest_signature"], False),  # 'attest' merely CONTAINS 'test' — not a test
        (["struct ContestEntry"], False),
    ],
)
def test_is_test_chunk(heading_path: list[str], is_test: bool) -> None:
    assert _is_test_chunk(_chunk(heading_path)) is is_test


# reorder_for_usage_intent — stable 2-tier demotion (def-of-X + tests to the bottom).


def test_reorder_demotes_definition_and_tests_to_bottom_preserving_order() -> None:
    other = _chunk(["enum AskForApproval"], "other")
    defn = _chunk(["fn is_known_safe_command"], "def")
    test = _chunk(["tests::ex"], "test")
    caller = _chunk(["fn assess_command_safety"], "caller")
    # input order: def, test, other, caller
    out = reorder_for_usage_intent([defn, test, other, caller], "is_known_safe_command")
    assert [c.chunk_id for c in out] == ["other", "caller", "def", "test"]


def test_reorder_is_noop_when_nothing_matches() -> None:
    # No chunk defines the (absent) symbol and none are tests → order is byte-identical.
    a = _chunk(["fn alpha"], "a")
    b = _chunk(["fn beta"], "b")
    out = reorder_for_usage_intent([a, b], "nonexistent_symbol")
    assert [c.chunk_id for c in out] == ["a", "b"]


# usage_intent_demotion_enabled — default on, kill-switch, fail-open.


def test_usage_intent_flag_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    # Default OFF (measured double-edged, audits/14) — opt-in only.
    monkeypatch.delenv("MEMEX_AGENTS__USAGE_INTENT_DEMOTION_ENABLED", raising=False)
    set_settings(MemexSettings())
    assert usage_intent_demotion_enabled() is False


def test_usage_intent_flag_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMEX_AGENTS__USAGE_INTENT_DEMOTION_ENABLED", "true")
    set_settings(MemexSettings())
    assert usage_intent_demotion_enabled() is True
    monkeypatch.delenv("MEMEX_AGENTS__USAGE_INTENT_DEMOTION_ENABLED", raising=False)
    set_settings(MemexSettings())


def test_usage_intent_flag_fails_open_to_false(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> object:
        raise ConfigurationError("no settings")

    monkeypatch.setattr(code_query, "get_settings", _boom)
    assert usage_intent_demotion_enabled() is False
