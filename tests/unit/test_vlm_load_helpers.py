"""VLM load-path helpers in `models/registry`: the FA2->sdpa attention
fallback and the AutoAWQ `PytorchGELUTanh` import shim.

Both unblock the AWQ VLM load on the current stack — `flash_attn` is an
optional, separately-compiled native dependency, and AutoAWQ (deprecated)
imports a `transformers.activations` class that was renamed. Before these
fixes the VLM could never load (and was never reached, because the dead
confidence-only escalation never fired)."""

from __future__ import annotations

from typing import Any

from memex.models import registry as R


def test_attn_impl_falls_back_to_sdpa_when_flash_attn_absent(monkeypatch: Any) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    assert R._vlm_attn_implementation() == "sdpa"


def test_attn_impl_prefers_fa2_when_flash_attn_present(monkeypatch: Any) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: object())
    assert R._vlm_attn_implementation() == "flash_attention_2"


def test_awq_compat_aliases_renamed_gelu_symbol() -> None:
    import transformers.activations as _act

    act: Any = _act  # dynamic third-party module surface
    had = hasattr(act, "PytorchGELUTanh")
    saved = getattr(act, "PytorchGELUTanh", None)
    try:
        if had:
            delattr(act, "PytorchGELUTanh")  # simulate the renamed-away state
        R._ensure_awq_import_compat()
        # the old name AutoAWQ imports is restored as an alias to GELUTanh
        assert hasattr(act, "PytorchGELUTanh")
        assert act.PytorchGELUTanh is act.GELUTanh
    finally:
        if had:
            act.PytorchGELUTanh = saved
        elif hasattr(act, "PytorchGELUTanh"):
            delattr(act, "PytorchGELUTanh")
