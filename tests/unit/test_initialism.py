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
    assert _gate_suggestions([_c("e", "Domain Name System", "concept", 1)], n_docs=0, exclude_ids=set()) == []
