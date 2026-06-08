"""Unit tests for the source-code display-language map (codebase-corpus arc, code-view).

`language_for_suffix` powers the webui code-view pane label ("source · rust"). It is a pure
lookup over `LANGUAGE_FOR_SUFFIX` with a case-insensitive key and a graceful bare-extension
fallback, so an unmapped code suffix never raises.
"""

from __future__ import annotations

import pytest

from memex.core.source_types import (
    CODE_SUFFIXES,
    LANGUAGE_FOR_SUFFIX,
    language_for_suffix,
)


@pytest.mark.parametrize(
    ("suffix", "language"),
    [
        (".rs", "rust"),
        (".py", "python"),
        (".ts", "typescript"),
        (".tsx", "typescript"),
        (".cpp", "c++"),
        (".cs", "c#"),
        (".sh", "shell"),
        (".kt", "kotlin"),
        (".mm", "objective-c++"),
    ],
)
def test_known_suffix_maps_to_language(suffix: str, language: str) -> None:
    assert language_for_suffix(suffix) == language


def test_lookup_is_case_insensitive() -> None:
    assert language_for_suffix(".PY") == "python"
    assert language_for_suffix(".Rs") == "rust"


def test_unmapped_suffix_falls_back_to_bare_extension() -> None:
    """An unknown suffix degrades to its extension (no dot), never raises."""
    assert language_for_suffix(".zig") == "zig"
    assert language_for_suffix(".UNKNOWN") == "unknown"


def test_every_code_suffix_has_a_nonempty_language() -> None:
    """The map stays in lockstep with CODE_SUFFIXES — every routable code suffix has a label."""
    for suffix in CODE_SUFFIXES:
        assert suffix in LANGUAGE_FOR_SUFFIX, f"{suffix} missing from LANGUAGE_FOR_SUFFIX"
        assert LANGUAGE_FOR_SUFFIX[suffix], f"{suffix} maps to an empty language name"
        # And the public helper agrees with the table (no fallback path for a known suffix).
        assert language_for_suffix(suffix) == LANGUAGE_FOR_SUFFIX[suffix]


def test_language_keys_are_normalized_leading_dot_lowercase() -> None:
    assert all(k.startswith(".") and k == k.lower() for k in LANGUAGE_FOR_SUFFIX)
