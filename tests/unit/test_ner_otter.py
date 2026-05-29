"""Unit tests for the OTTER NER enrich backend (`enrich/ner_otter.py`).

Pure pieces (label presets, device no-op, backend toggle) + the ported extraction logic
exercised through faked model/tokenizer/collator handles — no real model is loaded.
"""

from __future__ import annotations

import contextlib
from typing import Any

from memex.core.config import AgentsSettings, MemexSettings, set_settings
from memex.core.types import Chunk
from memex.enrich.ner_otter import (
    _KINDS,
    _LABEL_PRESETS,
    _OtterHandle,
    _to_device,
    otter_backend_enabled,
)


def _chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id="doc1#abc", document_id="doc1", document_title="Doc 1", text=text, char_start=0
    )


class _FakeTok:
    def decode(self, ids: list[str], **_kw: object) -> str:
        return " ".join(ids)


class _FakeCollator:
    def __init__(self, input_ids: list[str]) -> None:
        self._input_ids = input_ids

    def __call__(self, _batch: object, **_kw: object) -> dict[str, Any]:
        return {"token_encoder_inputs": {"input_ids": [self._input_ids]}}


class _FakeModel:
    def __init__(self, spans: list[dict[str, Any]]) -> None:
        self._spans = spans

    def predict(self, _batch: object, *, threshold: float = 0.0, **_kw: object) -> object:
        return [self._spans]


class _FakeTorch:
    def no_grad(self) -> Any:
        return contextlib.nullcontext()


def _handle(
    input_ids: list[str], spans: list[dict[str, Any]], preset: str = "union"
) -> _OtterHandle:
    return _OtterHandle(
        model=_FakeModel(spans),
        tok_token=_FakeTok(),
        collator=_FakeCollator(input_ids),
        torch_mod=_FakeTorch(),
        device="cpu",
        label_to_kind=_LABEL_PRESETS[preset],
        threshold=0.05,
    )


def test_label_presets_union_is_superset_with_valid_kinds() -> None:
    union = _LABEL_PRESETS["union"]
    assert set(_LABEL_PRESETS["generic"]) <= set(union)
    assert set(_LABEL_PRESETS["domain"]) <= set(union)
    assert all(kind in _KINDS for kind in union.values())


def test_to_device_cpu_is_noop() -> None:
    obj = {"a": object()}
    assert _to_device(obj, "cpu") is obj


def test_predict_entities_maps_kinds_locates_spans_and_filters_garble() -> None:
    text = "configure VLAN trunk and STP on the switch"
    input_ids = ["configure", "VLAN", "trunk", "and", "STP", "XYZGARBLE"]
    spans = [
        {"start": 1, "end": 2, "label": "networking protocol", "confidence": 0.9},  # -> tool
        {"start": 4, "end": 4, "label": "protocol", "confidence": 0.8},  # -> tool
        {"start": 5, "end": 5, "label": "miscellaneous", "confidence": 0.1},  # garble, dropped
    ]
    ents = _handle(input_ids, spans).predict_entities(_chunk(text))
    by_name = {e.name: e for e in ents}
    # "XYZGARBLE" decodes to a string not present in the source → dropped by the filter.
    assert set(by_name) == {"VLAN trunk", "STP"}
    assert by_name["VLAN trunk"].kind == "tool"
    assert by_name["STP"].kind == "tool"
    assert by_name["VLAN trunk"].confidence == 0.9
    start = text.find("VLAN trunk")
    assert by_name["VLAN trunk"].char_span == (start, start + len("VLAN trunk"))
    assert by_name["STP"].chunk_ids == ["doc1#abc"]


def test_predict_entities_blank_chunk_returns_empty() -> None:
    assert _handle(["x"], []).predict_entities(_chunk("   ")) == []


def test_otter_backend_enabled_reflects_setting() -> None:
    set_settings(MemexSettings(agents=AgentsSettings(enrich_ner_backend="otter")))  # type: ignore[call-arg]
    assert otter_backend_enabled() is True
    set_settings(MemexSettings(agents=AgentsSettings(enrich_ner_backend="llm")))  # type: ignore[call-arg]
    assert otter_backend_enabled() is False
