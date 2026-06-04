"""The deterministic faithful-transform guard + fidelity scorers (ADR-0017 §3/§15). Built for an
LLM transcript-structuring pass that was a VALIDATED NEGATIVE (large-v3-turbo already punctuates,
so the model returned every real block verbatim — the pass was reverted; see the
`transcript-structuring-negative-2026-06-04` memory). The guard + scorers are KEPT as a reusable,
adversarially-hardened primitive: a candidate may only re-format + collapse an ADJACENT stutter;
any addition / unique-content loss / reordering / number change must be REJECTED. Pure — no LLM."""

from __future__ import annotations

from memex.core.text import content_tokens, structure_block_is_faithful
from memex.eval.scoring import content_token_precision, content_token_recall

# ── content_tokens — the shared tokeniser ──


def test_content_tokens_lowercases_splits_and_drops_punctuation() -> None:
    assert content_tokens("The cat. Sat!") == ["the", "cat", "sat"]
    assert content_tokens("coupe-feu d'attaque") == ["coupe", "feu", "d", "attaque"]
    assert content_tokens("") == []
    # Numbers are content (kept); newlines are whitespace.
    assert content_tokens("port 53\n\nVLAN 10") == ["port", "53", "vlan", "10"]


# ── structure_block_is_faithful — the gate ──


def test_pure_reformatting_is_faithful() -> None:
    # Paragraph breaks + sentence-splitting + punctuation = identical content tokens.
    base = "the router forwards the packet then the switch learns the mac address"
    structured = "The router forwards the packet.\n\nThen the switch learns the MAC address."
    assert structure_block_is_faithful(structured, base)


def test_legit_stutter_collapse_is_faithful() -> None:
    # Light disfluency smoothing collapses an immediately-repeated content word.
    assert structure_block_is_faithful("the cat sat", "the the cat sat")
    assert structure_block_is_faithful("le réseau fonctionne", "le réseau réseau fonctionne")


def test_addition_is_rejected() -> None:
    # A hallucinated word is not a subsequence of the baseline.
    assert not structure_block_is_faithful("the big cat", "the cat")
    assert not structure_block_is_faithful("the cat sat down", "the cat sat")


def test_unique_word_deletion_is_rejected() -> None:
    # Dropping a content word ⇒ the adjacent-collapsed sequences differ.
    assert not structure_block_is_faithful("the cat", "the cat sat")
    assert not structure_block_is_faithful("port 53", "port 53 udp")


def test_reordering_is_rejected() -> None:
    # Same tokens, different order ⇒ equality (not subsequence) rejects it.
    assert not structure_block_is_faithful("cat the", "the cat")
    assert not structure_block_is_faithful("udp port 53", "port 53 udp")


def test_non_adjacent_duplicate_drop_is_rejected() -> None:
    # The bypass an independent adversarial review caught: dropping a NON-adjacent duplicate is real
    # content loss (only an IMMEDIATE stutter may collapse). A set/recall gate wrongly accepted these.
    assert not structure_block_is_faithful("buy low sell high", "buy low sell high buy")
    assert not structure_block_is_faithful(
        "open valve wait open", "open valve wait open valve open"
    )
    # …but an ADJACENT stutter still collapses fine.
    assert structure_block_is_faithful("open valve", "open open valve")


def test_number_value_or_format_change_is_rejected() -> None:
    # The headline bypass: `atomise` strips '.'/',', so 3.14 and 314 are the same content token —
    # the separate verbatim number check is what rejects a value (and any format) change.
    assert not structure_block_is_faithful("pi is 314", "pi is 3.14")  # value change (100x)
    assert not structure_block_is_faithful("the port is 54", "the port is 53")
    assert not structure_block_is_faithful("costs 1000", "costs 1,000")  # numbers must be verbatim
    assert not structure_block_is_faithful("subnet 192.168.0.2", "subnet 192.168.0.1")
    # A number left verbatim (the only faithful outcome) passes alongside reformatted prose.
    assert structure_block_is_faithful("The port is 53.\n\nUDP.", "the port is 53 udp")


def test_empty_edges() -> None:
    assert structure_block_is_faithful("", "")
    assert not structure_block_is_faithful("extra", "")  # adding to an empty baseline
    assert structure_block_is_faithful("", "lost everything") is False  # all content dropped


# ── content_token_precision / recall — the fidelity scorers ──


def test_scorers_perfect_on_faithful_reformatting() -> None:
    base = "the router forwards the packet"
    structured = "The router forwards the packet."
    assert content_token_precision(structured, base) == 1.0
    assert content_token_recall(structured, base) == 1.0


def test_scorers_perfect_on_legit_stutter_collapse() -> None:
    assert content_token_precision("the cat", "the the cat") == 1.0
    assert content_token_recall("the cat", "the the cat") == 1.0  # duplicate-collapse keeps the set


def test_precision_drops_on_addition() -> None:
    # {the,big,cat} vs {the,cat}: 2 of 3 structured tokens are legit.
    assert content_token_precision("the big cat", "the cat") == 2 / 3
    assert content_token_recall("the big cat", "the cat") == 1.0


def test_recall_drops_on_loss() -> None:
    assert content_token_recall("the cat", "the cat sat") == 2 / 3
    assert content_token_precision("the cat", "the cat sat") == 1.0


def test_scorer_empty_edges() -> None:
    assert content_token_precision("", "anything") == 1.0  # no false tokens
    assert content_token_recall("anything", "") == 1.0  # nothing to preserve
