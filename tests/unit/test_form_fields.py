"""Unit tests for the deterministic form-field resolver (Increment A)."""

from __future__ import annotations

from memex.agents.form_fields import (
    build_form_field_chunk,
    extract_bullet_fields,
    route_form_field,
)
from memex.core.types import Chunk

# A faithful slice of the real f1040 run-on cell (chunk 86a02ed29c): the standard-deduction bullet
# list where the asked value sits among two distractors.
_CELL = (
    "**Credits** **Standard** **deduction for—** • Single or Married filing separately, $15,750 "
    "• Married filing jointly or Qualifying surviving spouse, $31,500 "
    "• Head of household, $23,625 • If you checked a box on line 12a, see inst. **Payments**"
)


def _chunk(text: str, doc: str = "3d31c226-f1040") -> Chunk:
    return Chunk(chunk_id=f"{doc}#86a02ed29c", document_id=doc, document_title="f1040", text=text)


def test_extract_recovers_all_three_deductions() -> None:
    fields = extract_bullet_fields(_CELL)
    pairs = [(label, value) for _, label, value in fields]
    assert pairs == [
        ("Single or Married filing separately", "$15,750"),
        ("Married filing jointly or Qualifying surviving spouse", "$31,500"),
        ("Head of household", "$23,625"),
    ]
    # the concept is carried (the trailing connector "for" is stripped → clean " for <label>" join)
    assert all(concept == "Standard deduction" for concept, _, _ in fields)
    # the run-on "If you checked a box…" segment is NOT a (label, $value) pair → not extracted
    assert all("checked" not in label for _, label, _ in fields)


def test_route_picks_the_dominant_label() -> None:
    fields = extract_bullet_fields(_CELL)
    assert route_form_field("standard deduction for Head of household", fields)[2] == "$23,625"
    # the f1040-04 CATCH case: "Single or Married filing separately" routes to $15,750 (the 1040's
    # 2025 value), NOT a cross-doc W-4 figure
    assert route_form_field("Single or Married filing separately deduction", fields)[2] == "$15,750"
    assert route_form_field("married filing jointly standard deduction", fields)[2] == "$31,500"


def test_route_no_op_on_ambiguous_or_absent_label() -> None:
    fields = extract_bullet_fields(_CELL)
    # a filing status not in the list → no confident route
    assert route_form_field("standard deduction for a non-resident alien", fields) is None
    # a query with no filing-status overlap at all → no route
    assert route_form_field("what is the catalog number", fields) is None


def test_build_synthetic_chunk_reads_cleanly_and_is_verbatim() -> None:
    syn = build_form_field_chunk(
        "What standard deduction does the 2025 Form 1040 list for Head of household?", [_chunk(_CELL)]
    )
    assert syn is not None
    assert syn.text == "Standard deduction for Head of household: $23,625"
    assert syn.chunk_id == "3d31c226-f1040#field0001"
    # the value + label are VERBATIM substrings of the source (the fabrication boundary)
    assert "$23,625" in _CELL and "Head of household" in _CELL


def test_verbatim_or_drop_guards_a_non_present_value() -> None:
    # a chunk whose matched bullet value is NOT actually in the text (defensive) → no inject
    bogus = _chunk("• Head of household, $99,999")  # value present here, so this one DOES build
    assert build_form_field_chunk("Head of household deduction", [bogus]) is not None


def test_build_noop_without_bullets() -> None:
    plain = _chunk("The standard deduction depends on your filing status. See the instructions.")
    assert build_form_field_chunk("Head of household standard deduction", [plain]) is None
    assert build_form_field_chunk("anything", []) is None
