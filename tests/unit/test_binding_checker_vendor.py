"""Pins the vendored binding-checker dataset machinery (scripts/binding_checker_vendor.py)
— specifically the MEMEX sequence_ids answer-location fix: the upstream lettucedetect
heuristic (re-encode the context alone) loses the answer region entirely whenever the
context alone exceeds max_length, so span labels silently vanish for long passages.

Uses the lettucedect-en tokenizer from the local HF cache (present since the audit-18
checker round); skips cleanly if the cache is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from binding_checker_vendor import HallucinationDataset, HallucinationSample

_MODEL = "KRLabsOrg/lettucedect-base-modernbert-en-v1"


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
    except OSError:
        pytest.skip(f"{_MODEL} not in the local HF cache")


def _labels_for(tokenizer, prompt: str, answer: str, spans: list[dict]) -> list[int]:
    sample = HallucinationSample(
        prompt=prompt, answer=answer, labels=spans, split="train",
        task_type="qa", dataset="test", language="en",
    )
    ds = HallucinationDataset([sample], tokenizer, max_length=512)
    return ds[0]["labels"].tolist()


def test_span_maps_to_answer_tokens(tokenizer) -> None:
    answer = "Lyon is the capital of France."
    start = answer.index("Lyon")
    labels = _labels_for(
        tokenizer, "Paris is the capital of France. Lyon is a city.",
        answer, [{"start": start, "end": start + len("Lyon")}],
    )
    assert labels.count(1) >= 1  # the swapped subject is labeled
    assert labels.count(0) >= 3  # the rest of the answer is supported
    assert labels[0] == -100  # the prompt region is loss-masked


def test_unlabeled_answer_is_all_supported(tokenizer) -> None:
    labels = _labels_for(
        tokenizer, "Paris is the capital of France.",
        "Paris is the capital of France.", [],
    )
    assert labels.count(1) == 0
    assert labels.count(0) > 0


def test_long_context_truncation_keeps_answer_labels(tokenizer) -> None:
    """The MEMEX fix: with a context far beyond max_length, only_first truncation
    keeps the answer, and sequence_ids still locates it (the upstream heuristic
    returned an out-of-range answer_start and produced NO labeled tokens)."""
    answer = "The answer names NVLink here."
    start = answer.index("NVLink")
    labels = _labels_for(
        tokenizer, "word " * 3000, answer,
        [{"start": start, "end": start + len("NVLink")}],
    )
    assert labels.count(1) >= 1
    assert labels.count(0) >= 2
