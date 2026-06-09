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
from memex.index import code_query
from memex.index.code_query import (
    build_code_term_match,
    code_term_query_enabled,
    query_has_code_identifier,
)
from memex.index.fts_store import _fts_match_expr

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
