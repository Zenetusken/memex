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
from memex.parse.vlm_backend import _strip_image_links


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


def test_strip_image_links_removes_spurious_image_markdown() -> None:
    # The VLM occasionally punts a hard diagram with an image link to a
    # file that doesn't exist instead of describing it — strip it.
    md = "## Zonage\n\n- DMZ Externe\n- Zone interne\n\n![Network diagram](DMZ_Zone.png)"
    out = _strip_image_links(md)
    assert "![" not in out
    assert ".png" not in out
    assert "DMZ Externe" in out  # real content kept
    assert out.endswith("Zone interne")  # trailing blank line collapsed


def test_strip_image_links_keeps_plain_links() -> None:
    # A normal (non-image) markdown link has no leading '!' — must survive.
    md = "See [the spec](https://example.com) for details."
    assert _strip_image_links(md) == md


def test_vlm_prompt_keeps_load_bearing_clauses() -> None:
    """Pin the VLM prompt's load-bearing, validated clauses so a future edit can't silently
    drop them — the 2026-05-25 forceful rewrite AND the 2026-05-30 W6 decorative-suppression
    attempt were both reverted (the latter for a −1 ANS on cr350-diagrams; see the NB comment in
    vlm_backend.py + audit-10 W6). This is the validated baseline prompt."""
    from memex.parse.vlm_backend import _PROMPT

    # The validated diagram clause + its inline-flow example + the image-placeholder guard.
    assert "Router -> Firewall -> Private Network" in _PROMPT
    assert "do NOT emit an image placeholder like ![...]" in _PROMPT
    assert "Code blocks (```fenced)" in _PROMPT
    assert "Output ONLY Markdown for the page contents" in _PROMPT
    # Calm register — the forceful absolutes that got reverted must NOT reappear.
    assert "MUST" not in _PROMPT and "NEVER" not in _PROMPT
