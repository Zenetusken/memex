"""Unit tests for the deterministic acronym ↔ expansion bridge (ADR-0011):
the pure `index/initialism.py` helpers + the `_gate_suggestions` gate in
`graph_store.py`. No graph, no I/O — rows in, ranked suggestions out.
"""

from __future__ import annotations

from memex.index.graph_store import EntitySuggestion, _gate_suggestions
from memex.index.initialism import (
    derive_initialism,
    initialism_matches,
    looks_like_acronym,
)

# ---- derive_initialism ----


def test_derive_basic_expansions() -> None:
    assert derive_initialism("Domain Name System") == "DNS"
    assert derive_initialism("Address Resolution Protocol") == "ARP"
    assert derive_initialism("Dynamic Host Configuration Protocol") == "DHCP"


def test_derive_skips_en_and_fr_connectors() -> None:
    assert derive_initialism("Bureau of Investigation") == "BI"  # 'of' dropped
    assert derive_initialism("Système de Noms de Domaine") == "SND"  # FR 'de' dropped


def test_derive_single_word_is_none() -> None:
    assert derive_initialism("spanning") is None  # the exact-match case, not a bridge
    assert derive_initialism("DNS") is None  # single token, no initialism to derive
    assert derive_initialism("") is None
    assert derive_initialism("   ") is None


def test_derive_hyphen_and_slash_tokenised() -> None:
    assert derive_initialism("Address-Resolution-Protocol") == "ARP"  # hyphen splits
    assert derive_initialism("Input/Output Control") == "IOC"  # slash splits


def test_derive_too_long_is_none() -> None:
    # 8 significant words → over the _MAX_INITIALISM_LEN (7) cap → not an acronym expansion.
    assert derive_initialism("one two three four five six seven eight") is None


def test_derive_digit_token_takes_first_alnum() -> None:
    # Junk derivation is allowed here; the alpha-share guard lives in initialism_matches.
    assert derive_initialism("802.1X Authentication") == "8A"


# ---- looks_like_acronym ----


def test_looks_like_acronym_positive() -> None:
    assert looks_like_acronym("DNS")
    assert looks_like_acronym("dns")  # lowercase short → still a candidate
    assert looks_like_acronym("stp")
    assert looks_like_acronym("VLAN")  # uppercase


def test_looks_like_acronym_negative() -> None:
    assert not looks_like_acronym("spanning")  # 8-char lowercase real word → no scan
    assert not looks_like_acronym("Domain Name System")  # multi-token
    assert not looks_like_acronym("192.168")  # no letter
    assert not looks_like_acronym("A")  # too short
    assert not looks_like_acronym("routing")  # 7-char lowercase, no uppercase → excluded


# ---- initialism_matches ----


def test_initialism_matches() -> None:
    assert initialism_matches("DNS", "Domain Name System")
    assert initialism_matches("dns", "Domain Name System")  # case-insensitive query
    assert initialism_matches("ARP", "Address Resolution Protocol")
    assert not initialism_matches("DNS", "Domain Name")  # derives "DN"
    assert not initialism_matches("DNS", "spanning")  # single word → None → no match


def test_initialism_matches_alpha_share_guard() -> None:
    # A digit-only derivation ("88") is suppressed regardless of the query.
    assert derive_initialism("802.11 802.3") == "88"
    assert not initialism_matches("88", "802.11 802.3")  # 0% alpha → gated out


# ---- _gate_suggestions ----


def _c(entity_id: str, name: str, kind: str, doc_count: int, relation: str = "expansion"):
    return (entity_id, name, kind, doc_count, relation)


def test_gate_single_clean_bridge() -> None:
    out = _gate_suggestions(
        [_c("ent_dns_exp", "Domain Name System", "concept", 5)], n_docs=47, exclude_ids=set()
    )
    assert out == [
        EntitySuggestion(name="Domain Name System", kind="concept", doc_count=5, relation="expansion")
    ]


def test_gate_ambiguous_collision_drops_all() -> None:
    # Two DISTINCT names matched the same initialism → ambiguous → conservative no-op.
    out = _gate_suggestions(
        [
            _c("e1", "Domain Name System", "concept", 5),
            _c("e2", "Distributed Naming Service", "concept", 3),
        ],
        n_docs=47,
        exclude_ids=set(),
    )
    assert out == []


def test_gate_same_name_different_kind_is_not_a_collision() -> None:
    # One name across two kinds → ONE suggestion (highest-doc representative), not a collision.
    out = _gate_suggestions(
        [
            _c("e_concept", "Domain Name System", "concept", 5),
            _c("e_tool", "Domain Name System", "tool", 2),
        ],
        n_docs=47,
        exclude_ids=set(),
    )
    assert len(out) == 1
    assert out[0].name == "Domain Name System"
    assert out[0].doc_count == 5  # the max-doc representative
    assert out[0].kind == "concept"


def test_gate_excludes_generic_self_and_below_floor() -> None:
    # Generic (df > 0.6·N) dropped.
    assert _gate_suggestions([_c("e", "Generic Thing Name", "concept", 40)], n_docs=47, exclude_ids=set()) == []
    # Below the doc-count floor (a 1-doc cross-domain false-friend, e.g. "Short-term portion"
    # for query "STP") dropped — the key tweak from live validation.
    assert _gate_suggestions([_c("e", "Short Term Portion", "concept", 1)], n_docs=47, exclude_ids=set()) == []
    # Zero-doc orphan dropped.
    assert _gate_suggestions([_c("e", "Empty Name Here", "concept", 0)], n_docs=47, exclude_ids=set()) == []
    # Self (already the exact match) excluded.
    assert _gate_suggestions([_c("e_self", "Domain Name System", "concept", 5)], n_docs=47, exclude_ids={"e_self"}) == []


def test_gate_empty_corpus() -> None:
    assert _gate_suggestions([_c("e", "Domain Name System", "concept", 3)], n_docs=0, exclude_ids=set()) == []


# ---- edge cases: derive_initialism ----


def test_derive_lowercase_input_uppercases() -> None:
    assert derive_initialism("domain name system") == "DNS"


def test_derive_normalises_whitespace() -> None:
    assert derive_initialism("  Domain   Name  System  ") == "DNS"


def test_derive_all_connectors_is_none() -> None:
    assert derive_initialism("of the and") is None  # every token skipped → 0 significant


def test_derive_one_significant_word_plus_connectors_is_none() -> None:
    assert derive_initialism("Bureau of") is None  # only 'Bureau' is significant → 1 word


def test_derive_leading_connector_skipped() -> None:
    assert derive_initialism("the Domain Name System") == "DNS"


def test_derive_accented_leading_letter_kept() -> None:
    # The unicode-aware fix: an accented first letter is TAKEN, not skipped to the next
    # ASCII char (the old `[0-9a-z]` regex turned "Émetteur" into "m").
    assert derive_initialism("Émetteur Récepteur") == "ÉR"


def test_derive_strips_leading_punctuation_per_token() -> None:
    assert derive_initialism("Virtual (LAN) Network") == "VLN"  # first ALNUM of "(lan)" is "l"


def test_derive_length_boundaries() -> None:
    assert derive_initialism("Ay Bee") == "AB"  # exactly 2 significant words (min) → ok
    assert derive_initialism("xx yy zz pp qq rr ss") == "XYZPQRS"  # exactly 7 → ok
    assert derive_initialism("xx yy zz pp qq rr ss tt") is None  # 8 → over the cap


def test_derive_single_letter_connectors_skipped() -> None:
    # Single-letter tokens that ARE connectors ('a', FR 'd'/'l') drop out — so a string
    # of bare letters keeps only the non-connectors.
    assert derive_initialism("a b c d e f g") == "BCEFG"  # 'a' + 'd' are skip-words


# ---- edge cases: looks_like_acronym boundaries ----


def test_looks_like_acronym_length_boundaries() -> None:
    assert looks_like_acronym("NETWORK")  # 7 chars uppercase → ok
    assert not looks_like_acronym("ETHERNET")  # 8 chars → over the cap


def test_looks_like_acronym_punctuation_forms() -> None:
    assert looks_like_acronym("802.1X")  # digit+letter, len 6
    assert looks_like_acronym("PVST+")  # the '+' family is allowed
    assert not looks_like_acronym("")  # empty
    assert not looks_like_acronym("   ")  # whitespace-only


# ---- edge cases: initialism_matches alpha-share boundary ----


def test_initialism_matches_alpha_share_boundary() -> None:
    # Exactly 50% alpha passes (≥ threshold): "Apple 1st" → "A1" (1 of 2 alpha).
    assert derive_initialism("Apple 1st") == "A1"
    assert initialism_matches("A1", "Apple 1st")
    # Below 50% fails: "1st 2nd Apple" → "12A" (1 of 3 alpha = 0.33).
    assert derive_initialism("1st 2nd Apple") == "12A"
    assert not initialism_matches("12A", "1st 2nd Apple")


# ---- edge cases: _gate_suggestions interactions ----


def test_gate_relation_preserved() -> None:
    out = _gate_suggestions([_c("e", "Domain Name System", "concept", 3, "acronym")], n_docs=47, exclude_ids=set())
    assert out[0].relation == "acronym"


def test_gate_floor_short_circuits_before_collision() -> None:
    # Two DISTINCT names, BOTH below the floor → dropped by the floor (not the collision
    # path); the result is still [] but for the floor reason. Pinning the order of gates.
    out = _gate_suggestions(
        [_c("e1", "Alpha Beta Cee", "concept", 1), _c("e2", "Apple Banana Cherry", "concept", 1)],
        n_docs=47,
        exclude_ids=set(),
    )
    assert out == []


def test_gate_below_floor_sibling_does_not_trigger_false_collision() -> None:
    # One VALID name (≥floor) + a DIFFERENT below-floor name: the floored-out one must
    # NOT count toward the collision test, so the valid bridge survives as the lone result.
    out = _gate_suggestions(
        [
            _c("e_valid", "Domain Name System", "concept", 4),
            _c("e_noise", "Distributed Naming Service", "concept", 1),  # below floor → ignored
        ],
        n_docs=47,
        exclude_ids=set(),
    )
    assert [s.name for s in out] == ["Domain Name System"]


def test_gate_same_name_max_doc_count_across_floor() -> None:
    # Same name across two entities, one below floor + one above → deduped to the
    # above-floor representative (max doc_count).
    out = _gate_suggestions(
        [
            _c("e_low", "Domain Name System", "tool", 1),
            _c("e_high", "Domain Name System", "concept", 4),
        ],
        n_docs=47,
        exclude_ids=set(),
    )
    assert len(out) == 1
    assert out[0].doc_count == 4
    assert out[0].kind == "concept"


def test_gate_multiple_exclude_ids() -> None:
    out = _gate_suggestions(
        [_c("e1", "Domain Name System", "concept", 4), _c("e2", "Domain Name System", "tool", 3)],
        n_docs=47,
        exclude_ids={"e1", "e2"},  # both are the exact-resolved entity → nothing to suggest
    )
    assert out == []
