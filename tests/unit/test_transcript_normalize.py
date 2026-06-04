"""Unit tests for deterministic ASR transcript normalization (ADR-0017).

`core/text.normalize_transcript_text` must clean non-lexical speech noise (filler
interjections + whitespace/punctuation artifacts) while being 100% faithful — it must
NEVER alter a content word — and deterministic/idempotent.
"""

from __future__ import annotations

from memex.core.text import normalize_transcript_text


def test_removes_inline_filler_en() -> None:
    assert normalize_transcript_text("I um think the router forwards it") == (
        "I think the router forwards it"
    )


def test_removes_filler_fr() -> None:
    assert normalize_transcript_text("le euh routeur transfère le paquet") == (
        "le routeur transfère le paquet"
    )


def test_removes_comma_set_off_filler() -> None:
    assert normalize_transcript_text("I, um, think so") == "I, think so"


def test_removes_multiple_consecutive_fillers() -> None:
    assert normalize_transcript_text("um uh I think") == "I think"


def test_sentence_initial_filler() -> None:
    # The dangling leading comma is cleaned; the content survives (casing left as-is).
    assert normalize_transcript_text("Um, so it works") == "so it works"


def test_does_not_touch_content_word_repetition() -> None:
    # Content-word stutters are the LLM follow-up's job, not this faithful pass.
    assert normalize_transcript_text("the the the cat") == "the the the cat"
    assert normalize_transcript_text("very very good") == "very very good"


def test_does_not_clip_filler_inside_a_word() -> None:
    assert normalize_transcript_text("an umbrella under the bridge") == (
        "an umbrella under the bridge"
    )


def test_does_not_mangle_hyphenated_backchannel() -> None:
    # "uh-huh" is a backchannel ("yes"), not a hesitation filler — must survive whole.
    assert normalize_transcript_text("uh-huh exactly") == "uh-huh exactly"


def test_collapses_whitespace() -> None:
    assert normalize_transcript_text("the   network    uses OSPF") == "the network uses OSPF"


def test_preserves_asr_nonspeech_markers() -> None:
    # Honest ASR annotations are signal, not noise — left untouched.
    assert normalize_transcript_text("[inaudible] the OSPF area") == "[inaudible] the OSPF area"


def test_all_filler_segment_becomes_empty() -> None:
    assert normalize_transcript_text("um uh euh") == ""


def test_clean_text_is_unchanged() -> None:
    clean = "The network uses OSPF to compute shortest paths."
    assert normalize_transcript_text(clean) == clean


def test_idempotent() -> None:
    raw = "I, um, think the, uh, router um forwards   it"
    once = normalize_transcript_text(raw)
    assert normalize_transcript_text(once) == once


def test_legit_leading_comma_preserved() -> None:
    # No filler was removed → a legitimate sentence-leading comma must NOT be stripped
    # (the prior over-strip; validator F1).
    assert normalize_transcript_text(", and then it works") == ", and then it works"


def test_filler_as_own_sentence_cleans_doubled_period() -> None:
    # A filler that was its own "sentence" leaves a doubled period that gets mopped up.
    assert normalize_transcript_text("I think. Uh. The router") == "I think. The router"


def test_trailing_filler_cleans_comma_before_terminal() -> None:
    # "Yes, um." → "Yes, ." → "Yes." (comma-before-terminal residue removed; AUDIO-NORM-002).
    assert normalize_transcript_text("Yes, um.") == "Yes."


def test_ahem_is_not_stripped() -> None:
    # `ahem` was dropped from the filler set — it is often an intentional attention-getter
    # (validator SC-3), so it must survive.
    assert normalize_transcript_text("ahem let me explain") == "ahem let me explain"


def test_capitalized_filler_homograph_is_removed() -> None:
    # DOCUMENTED accepted limit: a standalone capitalised filler-homograph (a surname "Heu")
    # is removed too. The verbatim raw stays cached as the faithfulness anchor.
    assert normalize_transcript_text("Heu is a name") == "is a name"


def test_prefix_overlapping_fillers_all_removed_deterministically() -> None:
    # Longest-match + boundary handling makes equal-length fillers order-invariant, so the
    # output is hash-seed-independent (the chunk-id-stability property). Prefix-overlapping
    # fillers are all removed regardless of alternation order.
    assert normalize_transcript_text("uh uhh uhhh uhm") == ""
    assert normalize_transcript_text("um umm ummm") == ""
