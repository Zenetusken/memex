"""The deterministic NAME-ONLY grounding backstop (2026-06-03) — the pure
`core/text.claim_asserts_behavior` matcher behind the `verify` node's 5th demotion filter.

It closes the `verify_grounding/v2` entity-name-presence loophole: the gate grounds a BEHAVIORAL
claim against a chunk that merely NAMES the entity (a bare list/heading). The matcher is
FAIL-OPEN — membership/existence/definition AND any unrecognised phrasing return False (KEEP) — so
the backstop (demotion-only) can never manufacture an over-refusal. The wiring into `verify` is
pinned in `tests/integration/test_answering_with_fakes.py`.
"""

from __future__ import annotations

import pytest

from memex.core.text import claim_asserts_behavior

# claim text → expected (True = behavioral ⇒ demote-eligible on a name-list).
_BEHAVIORAL = [
    "RBAC assigns permissions based on a user's job function.",
    "ABAC evaluates attributes dynamically.",
    "ABAC allows access decisions based on a current confidence level.",
    "RBAC lacks the granularity to react to device state.",
    "ABAC is superior for a dynamic, device-health-aware environment.",
    "Local password authentication is not scalable.",
    "OSPF is faster than RIP at convergence.",
    # FR
    "RBAC repose sur des rôles prédéfinis.",
    "ABAC utilise une évaluation dynamique des attributs.",
    "Le protocole permet un contrôle granulaire.",
]

_NOT_BEHAVIORAL = [
    # membership / existence / definition — a name-list grounds these
    "RBAC is one of the access control models listed.",
    "RBAC is an access control model.",
    "The access control models include RBAC, ABAC, and MAC.",
    "MAC is listed among the access control types.",
    "ABAC est un type de contrôle d'accès.",
    "RBAC fait partie des modèles de contrôle d'accès.",
    # value / unknown — fail-open KEEP (a table chunk wouldn't be name-only anyway)
    "The default priority for RBAC is 1.",
    "OSPF is a link-state routing protocol.",
]


@pytest.mark.parametrize("claim", _BEHAVIORAL)
def test_behavioral_claims_are_flagged(claim: str) -> None:
    assert claim_asserts_behavior(claim) is True


@pytest.mark.parametrize("claim", _NOT_BEHAVIORAL)
def test_membership_value_and_unknown_claims_are_kept(claim: str) -> None:
    assert claim_asserts_behavior(claim) is False


def test_cross_language_membership_is_kept() -> None:
    """The critical trap: an EN membership claim about a FR-named entity must stay KEPT — the
    matcher keys on the predicate class, not lexical overlap with the (French) chunk."""
    assert claim_asserts_behavior("RBAC is one of the listed access-control models") is False


def test_membership_phrase_overrides_a_behavioral_marker() -> None:
    """Fail-open: when the claim's MAIN assertion is membership but it carries a behavioral word
    in a definitional qualifier, membership wins (checked first — the safe direction). The
    name-list DOES support 'RBAC is one of the access-control models'."""
    assert claim_asserts_behavior("RBAC is one of the models used for access control") is False


def test_empty_and_bare_claims_are_kept() -> None:
    assert claim_asserts_behavior("") is False
    assert claim_asserts_behavior("RBAC") is False
