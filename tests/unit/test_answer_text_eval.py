"""`answer_mention_recall` — the deterministic answer-TEXT-correctness scorer (audit-14).

`citation_precision` grades CITED CHUNKS, never the answer text, which hid the usage-class
wrong-answer class ("which function calls X" answered by describing X's definition). The
scorer checks required SLOTS (str = the term; list = ANY-OF alternatives) via WHOLE-TOKEN
boundary match on normalized text — the slot vocabulary is code identifiers, so substring
matching would let `apply_patch` hit inside `maybe_parse_apply_patch_verified`.
"""

from __future__ import annotations

from memex.eval.scoring import answer_mention_recall

# ---------------------------------------------------------------------------
# Base semantics
# ---------------------------------------------------------------------------


def test_no_slots_is_ungraded_one() -> None:
    assert answer_mention_recall("any answer at all", []) == 1.0


def test_empty_answer_with_slots_is_zero() -> None:
    assert answer_mention_recall("", ["assess_command_safety"]) == 0.0


def test_fraction_of_slots() -> None:
    assert answer_mention_recall("one is here", ["one", "missing"]) == 0.5


def test_all_slots_found() -> None:
    text = "is_known_safe_command is called in assess_command_safety in safety.rs"
    assert answer_mention_recall(text, ["assess_command_safety"]) == 1.0


# ---------------------------------------------------------------------------
# Any-of groups (the multiple-valid-callers case)
# ---------------------------------------------------------------------------


def test_any_of_group_satisfied_by_either_alternative() -> None:
    slots: list[str | list[str]] = [["assess_patch_safety", "assess_command_safety"]]
    assert answer_mention_recall("called in assess_command_safety", slots) == 1.0
    assert answer_mention_recall("called in assess_patch_safety", slots) == 1.0


def test_any_of_group_unsatisfied_when_no_alternative_present() -> None:
    slots: list[str | list[str]] = [["assess_patch_safety", "assess_command_safety"]]
    assert answer_mention_recall("it checks the policy and returns", slots) == 0.0


def test_empty_any_of_group_is_unsatisfiable_not_vacuous() -> None:
    # A corpus-authoring error must be LOUD (recall < 1.0), never silently pass.
    assert answer_mention_recall("anything", [[]]) == 0.0


# ---------------------------------------------------------------------------
# The boundary traps — code identifiers must match as WHOLE tokens
# ---------------------------------------------------------------------------


def test_snake_case_slot_does_not_match_inside_longer_identifier() -> None:
    # THE trap from the corpus: `apply_patch` inside `maybe_parse_apply_patch_verified`.
    text = "The function maybe_parse_apply_patch_verified handles the hunk."
    assert answer_mention_recall(text, ["apply_patch"]) == 0.0


def test_snake_case_slot_matches_the_whole_identifier() -> None:
    assert answer_mention_recall("apply_patch calls apply_hunks", ["apply_patch"]) == 1.0


def test_prefix_identifier_does_not_match_its_extended_variant() -> None:
    # `maybe_parse_apply_patch` must not be satisfied by `maybe_parse_apply_patch_verified`.
    text = "see maybe_parse_apply_patch_verified for the verified path"
    assert answer_mention_recall(text, ["maybe_parse_apply_patch"]) == 0.0


def test_identifier_followed_by_punctuation_matches() -> None:
    # `(`, `.`, and backticks are non-word chars — the boundary accepts them.
    assert answer_mention_recall("calls apply_patch().", ["apply_patch"]) == 1.0


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_match_is_case_insensitive() -> None:
    assert (
        answer_mention_recall("Called by Assess_Command_Safety", ["assess_command_safety"]) == 1.0
    )


def test_backtick_wrapped_identifier_matches() -> None:
    assert answer_mention_recall("the `submission_loop` function", ["submission_loop"]) == 1.0


def test_double_colon_path_matches() -> None:
    # `:` is a non-word char; `re.escape` keeps the literal `::` — the whole path matches.
    assert answer_mention_recall("uses RolloutRecorder::new here", ["rolloutrecorder::new"]) == 1.0


def test_camelcase_compound_is_one_token_subtoken_slot_misses() -> None:
    # Documented behavior, not a bug: `MacosSeatbelt` normalizes to `macosseatbelt`,
    # one \w+ token — a sub-token slot (`seatbelt`) cannot boundary-match. Annotate
    # FULL identifiers (plus spaced variants as any-of alternatives).
    assert answer_mention_recall("the MacosSeatbelt variant", ["seatbelt"]) == 0.0
    assert answer_mention_recall("the MacosSeatbelt variant", ["macosseatbelt"]) == 1.0
    spaced: list[str | list[str]] = [["macosseatbelt", "macos seatbelt"]]
    assert answer_mention_recall("the macOS Seatbelt sandbox", spaced) == 1.0


def test_whitespace_collapse() -> None:
    assert answer_mention_recall("assess   \n  command", ["assess command"]) == 1.0
