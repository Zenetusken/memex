"""Pins the deterministic, vault-independent pieces of the binding-data minter
(scripts/mint_binding_data.py): the presence-preserving invariant, the unit-transform
skill positives (the ar-03 calibration-FP mode), and question-blind prompts. The
vault/LLM-dependent phases are exercised by the live mint, not here.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import mint_binding_data as M


def test_unit_transforms_millions_to_billions() -> None:
    # the literal ar-03 shape: 193,737 in a millions table → $193.737 billion
    forms = M.unit_transforms("193,737", "en", millions_table=True)
    assert "$193.737 billion" in forms
    assert "$193,737 million" in forms
    # without a millions table the comma-grouped rewrite is not asserted
    assert M.unit_transforms("193,737", "en", millions_table=False) == []


def test_unit_transforms_percent_and_money_words() -> None:
    assert "71.1 percent" in M.unit_transforms("71.1%", "en", millions_table=False)
    assert "71.1 pour cent" in M.unit_transforms("71.1%", "fr", millions_table=False)
    assert "$130.4 billion" in M.unit_transforms("$130.4B", "en", millions_table=False)


def test_render_tracks_slot_spans() -> None:
    text, spans = M.render("The {header} of {label} was {value}.",
                           {"header": "margin", "label": "Graphics", "value": "71%"})
    assert text == "The margin of Graphics was 71%."
    s, e = spans["label"]
    assert text[s:e] == "Graphics"


def test_synthetic_positives_are_leak_safe_and_labeled_ok() -> None:
    synth = M.mint_synthetic_positives(n_unit=20, n_struct=10)
    assert len(synth) == 30
    assert all(s["labels"] == [] for s in synth)  # positives carry no breach span
    assert all(s["meta"]["doc_id"] == "synthetic" for s in synth)
    # deterministic: a re-mint is byte-identical
    assert M.mint_synthetic_positives(n_unit=20, n_struct=10) == synth


def test_question_blind_blanks_the_prompt_slot() -> None:
    M.QUESTION_BLIND = True
    try:
        chunk = {"chunk_id": "x#1", "doc_id": "d", "title": "T",
                 "doc_title": "the doc", "text": "body"}
        s = M.make_sample(chunk, "en", "What was the revenue?", "Revenue was $5B.",
                          None, "test", "train")
        assert "What was the revenue?" not in s["prompt"]
        assert re.search(r"the following question:\n\nBear in mind", s["prompt"])
    finally:
        M.QUESTION_BLIND = False


def test_presence_preserving_negative_keeps_swapped_subject_in_span() -> None:
    # a NEG-row rebind: span must cover exactly the swapped-in label
    text, spans = M.render("The {header} for {label} was {value}.",
                           {"header": "revenue", "label": "Compute", "value": "$5B"})
    s, e = spans["label"]
    assert text[s:e] == "Compute"
    # and the swapped subject is a single contiguous slot, not the whole claim
    assert e - s < len(text)
