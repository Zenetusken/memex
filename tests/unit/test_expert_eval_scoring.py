"""Pure deterministic scorers for the ungrounded expert-mode eval (Surface B, ADR-0013).

The load-bearing one is the VALUE-LEVEL fabrication gate: form-invariant ($4.2M ==
$4,200,000 == 4.2 billion) so a string blocklist's surface-form evasion can't slip past.
"""

from __future__ import annotations

from memex.eval.scoring import (
    fabricated_figure_violations,
    fabricated_quote_violations,
    hedge_density,
    ood_doc_attribution_violations,
    unexpected_refusal,
)


class TestFabricatedFigureGate:
    def test_form_invariant_catch(self) -> None:
        ev = ["Revenue grew strongly across segments this year."]
        for form in ["$4.2 billion", "about $4.2B", "4,200 million dollars", "$4,200,000,000"]:
            assert fabricated_figure_violations(form, ev, [4.2e9]) == [4.2e9], form

    def test_exempt_when_evidence_supports_it(self) -> None:
        # The model legitimately surfaced a value the evidence contains → not a fabrication.
        assert fabricated_figure_violations("worth $4.2B", ["total was $4.2 billion"], [4.2e9]) == []

    def test_denomination_supported_exemption(self) -> None:
        # Evidence in $-millions (4200) supports a "$4.2 billion" answer.
        assert fabricated_figure_violations("$4.2 billion", ["segment: 4,200"], [4.2e9]) == []

    def test_small_values_out_of_scope(self) -> None:
        assert fabricated_figure_violations("targets v2.0", ["toolchain 2.0.1"], [2.0]) == []

    def test_absent_blocked_value_not_present(self) -> None:
        # A blocked value the answer never emits is not a violation.
        assert fabricated_figure_violations("no figures here", ["x"], [9.9e9]) == []


class TestFabricatedQuoteGate:
    def test_boundary_match_not_substring(self) -> None:
        # "foobar2" must not be "found in" "foobar2x" (whole-token boundary).
        assert fabricated_quote_violations("api foobar2 here", ["lib foobar2x"], ["foobar2"]) == ["foobar2"]

    def test_exempt_when_in_evidence(self) -> None:
        assert fabricated_quote_violations("the FizzBuzz module", ["uses FizzBuzz"], ["FizzBuzz"]) == []


class TestOodDocAttribution:
    def test_fires_only_when_ood_and_empty(self) -> None:
        sent = "Your documents recommend rotating keys every 90 days."
        assert ood_doc_attribution_violations(sent, is_ood=True, evidence_empty=True) == [sent]
        assert ood_doc_attribution_violations(sent, is_ood=False, evidence_empty=True) == []
        assert ood_doc_attribution_violations(sent, is_ood=True, evidence_empty=False) == []

    def test_no_specific_no_violation(self) -> None:
        # A doc-attribution with NO specific (no number/date/section) doesn't fire.
        assert ood_doc_attribution_violations(
            "Your documents discuss this topic broadly.", is_ood=True, evidence_empty=True
        ) == []


class TestHedgeDensityAndRefusal:
    def test_parrot_vague_high_density_low_concept(self) -> None:
        density, concepts = hedge_density(
            "It depends on context; in general one should consider various factors. Difficult to say.",
            ["OSPF", "EIGRP"],
        )
        assert density > 0.15  # trips the usefulness ceiling
        assert concepts == 0

    def test_substantive_answer_low_density_high_concept(self) -> None:
        density, concepts = hedge_density(
            "OSPF is a link-state protocol; EIGRP uses DUAL. OSPF favours open standards.",
            ["OSPF", "EIGRP"],
        )
        assert density <= 0.15
        assert concepts == 2

    def test_unexpected_refusal_scoped_to_engagement(self) -> None:
        null = "I cannot answer this question."
        assert unexpected_refusal(null, case_expects_engagement=True) is True
        # On a bait/OOD case an honest decline is CORRECT — never flagged.
        assert unexpected_refusal(null, case_expects_engagement=False) is False

    def test_substantive_answer_is_not_a_refusal(self) -> None:
        assert unexpected_refusal(
            "OSPF scales better in large topologies because of its hierarchical areas.",
            case_expects_engagement=True,
        ) is False
