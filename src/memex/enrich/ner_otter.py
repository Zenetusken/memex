"""OTTER NER — an optional BERT span-NER backend for entity extraction at enrich.

A gated alternative to the LLM (Qwen3-8B) entity extractor: `whoisjones/otter-bi-mmbert`
— a GLiNER-style multilingual span NER on the mmBERT backbone. Selected via
`AgentsSettings.enrich_ner_backend == "otter"` (default `"llm"` ⇒ this module is never
imported/loaded). It produces the SAME per-chunk `list[Entity]` the LLM path's
`merge_entities` emits, so the output flows through the unchanged document-level `dedupe`
+ graph write — only the entity SOURCE changes; citations stay on the LLM.

Verified against the model's shipped code (`collate_fn.py` / `metrics.py`, 2026-05-29) and
A/B-validated by `scripts/entity_ner_ab_audit.py`: at threshold 0.05 + union labels it
lifts the `related_documents` discovery score +103% over the LLM entities on the 47-doc
vault, with far cleaner typing (place-noise 1539→359, method 380→4418). See
[[bert-ner-enrich-scope-2026-05-28]].

Loaded as a lazy process-global, NOT via `models/registry` — it's a CPU-side (or
pause-window GPU) enrich-time model with no answer-path co-residence, so the registry's
GPU OOM-breaker / co-residence machinery doesn't apply (same out-of-registry precedent as
the parse-time VLM / summarizer serves). The forward is lock-serialized: enrich is
CPU-bound here and per-chunk concurrency wouldn't help, and a single torch CPU model isn't
reentrancy-guaranteed.

`Any` is used for the model / tokenizer / collator handles by design: OTTER ships a custom
`trust_remote_code` architecture (`OtterBiEncoderModel` + `AllLabelsCollator`) with no
type stubs, so the whole inference surface is dynamic.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, cast

import structlog

from memex.core.config import get_settings
from memex.core.errors import ConfigurationError, MemexError
from memex.core.types import Chunk
from memex.enrich.entities import Entity, EntityKind

logger = structlog.get_logger(__name__)

# Mirror the live enrich passage cap (`extract_entities/v2.md` `truncate(6000)`); the
# collator further truncates to its token window. Parity with the LLM path's input.
_PASSAGE_CHARS = 6000


class OtterNERUnavailable(MemexError):
    """OTTER NER could not be loaded or produced an unusable result (missing deps/model)."""


# Zero-shot label presets → Memex's 7-kind taxonomy. The A/B winner is "union" (generic
# single words ∪ domain descriptive phrases): best discovery AND coverage on both the
# networking/security and general-English corpora (`scripts/entity_ner_ab_audit.py`).
_GENERIC: dict[str, str] = {
    "person": "person",
    "organization": "org",
    "company": "org",
    "location": "place",
    "place": "place",
    "concept": "concept",
    "method": "method",
    "technique": "method",
    "algorithm": "method",
    "tool": "tool",
    "software": "tool",
    "protocol": "tool",
    "standard": "tool",
    "miscellaneous": "other",
}
_DOMAIN: dict[str, str] = {
    "person": "person",
    "company or organization": "org",
    "location or place": "place",
    "networking protocol": "tool",
    "network device or hardware": "tool",
    "software or application": "tool",
    "network service": "tool",
    "technical standard or specification": "tool",
    "security attack or threat technique": "method",
    "security control or defense mechanism": "method",
    "cryptographic algorithm or method": "method",
    "vulnerability or weakness": "concept",
    "technical concept or term": "concept",
    "miscellaneous entity": "other",
}
_LABEL_PRESETS: dict[str, dict[str, str]] = {
    "generic": _GENERIC,
    "domain": _DOMAIN,
    "union": {**_DOMAIN, **_GENERIC},
}

_KINDS: frozenset[str] = frozenset({"person", "org", "place", "concept", "method", "tool", "other"})


def _to_device(obj: Any, device: str) -> Any:
    """Recursively move tensors in the (nested-dict) collated batch to `device` (no-op CPU)."""
    if device == "cpu":
        return obj
    if hasattr(obj, "to"):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in cast("dict[Any, Any]", obj).items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in cast("list[Any]", obj)]
    return obj


class _OtterHandle:
    """A loaded OTTER model + token tokenizer + collator. Forward is lock-serialized."""

    def __init__(
        self,
        *,
        model: Any,
        tok_token: Any,
        collator: Any,
        torch_mod: Any,
        device: str,
        label_to_kind: dict[str, str],
        threshold: float,
    ) -> None:
        self._model = model
        self._tok_token = tok_token
        self._collator = collator
        self._torch = torch_mod
        self._device = device
        self._label_to_kind = label_to_kind
        self._threshold = threshold
        self._lock = threading.Lock()

    def predict_entities(self, chunk: Chunk) -> list[Entity]:
        """Extract a chunk's entities as `list[Entity]` (same shape as `merge_entities`)."""
        text = chunk.text[:_PASSAGE_CHARS]
        if not text.strip():
            return []
        with self._lock:
            spans = self._predict_spans(text)
        out: list[Entity] = []
        for surface, label, score in spans:
            kind = self._label_to_kind.get(label.strip().lower(), "other")
            name = surface.strip()
            if not name or kind not in _KINDS:
                continue
            # Locate the span in the chunk for char offsets (mirrors `merge_entities`).
            idx = chunk.text.find(name)
            span = (
                (chunk.char_start + idx, chunk.char_start + idx + len(name)) if idx >= 0 else None
            )
            out.append(
                Entity(
                    name=name,
                    kind=cast("EntityKind", kind),  # validated against _KINDS above
                    confidence=score,
                    chunk_ids=[chunk.chunk_id],
                    char_span=span,
                )
            )
        return out

    def _predict_spans(self, text: str) -> list[tuple[str, str, float]]:
        batch: Any = self._collator([{"text": text, "char_spans": []}])
        batch = _to_device(batch, self._device)
        with self._torch.no_grad():
            preds: Any = self._model.predict(batch, threshold=self._threshold)
        if not (isinstance(preds, list) and preds and isinstance(preds[0], list)):
            raise OtterNERUnavailable(
                "OTTER predict() returned an unexpected shape (expected list[list[dict]])"
            )
        input_ids: Any = batch["token_encoder_inputs"]["input_ids"][0]
        text_norm = " ".join(text.lower().split())
        out: list[tuple[str, str, float]] = []
        for sp in cast("list[dict[str, Any]]", preds[0]):
            # start/end are SUBWORD-TOKEN indices → decode the surface from the input_ids.
            start, end = int(sp["start"]), int(sp["end"])
            surface = str(
                self._tok_token.decode(input_ids[start : end + 1], skip_special_tokens=True)
            ).strip()
            # Drop cross-token-boundary decode garble: a real span's decode is a
            # whitespace-normalised substring of source; artefacts are not.
            if surface and " ".join(surface.lower().split()) in text_norm:
                out.append((surface, str(sp["label"]), float(sp["confidence"])))
        return out


# lowercase (not a constant): reassigned under `_load_lock` by the lazy loader.
_handle: _OtterHandle | None = None
_load_lock = threading.Lock()


def _from_pretrained(auto_cls: Any, *args: Any, **kwargs: Any) -> Any:
    """Typed boundary for HF `from_pretrained` (partial py.typed) — returns Any by design
    (`auto_cls` is Any, so the access is too). Mirrors `models/registry._from_pretrained`,
    which we can't import (cross-module private)."""
    return auto_cls.from_pretrained(*args, **kwargs)


def _load() -> _OtterHandle:
    agents = get_settings().agents
    model_id = agents.enrich_ner_model
    device = agents.enrich_ner_device
    labels = agents.enrich_ner_labels
    try:
        import torch
        from transformers import (  # type: ignore[reportMissingTypeStubs]  # partial py.typed
            AutoConfig,
            AutoModelForTokenClassification,
            AutoTokenizer,
        )
        from transformers.dynamic_module_utils import (
            get_class_from_dynamic_module,  # type: ignore[reportUnknownVariableType]  # HF loader: partial types
        )
    except ImportError as e:
        raise OtterNERUnavailable(
            "transformers/torch unavailable for the OTTER NER backend",
            context={"error": str(e)},
        ) from e
    try:
        # `from_pretrained` ships partial types; route through the local typed boundary
        # (mirrors registry._from_pretrained — we can't import that private symbol) so the
        # dynamic surface is Any, not a pyright reportUnknownMemberType cascade. OTTER loads
        # F32 by default (model.safetensors is F32), so no torch_dtype arg is needed.
        config = _from_pretrained(AutoConfig, model_id, trust_remote_code=True)
        model = _from_pretrained(AutoModelForTokenClassification, model_id, trust_remote_code=True)
        model.to(device)
        model.eval()
        tok_token = _from_pretrained(AutoTokenizer, config.token_encoder)
        tok_type = _from_pretrained(AutoTokenizer, config.type_encoder)
        label_to_kind = _LABEL_PRESETS[labels]
        label2id = {lab: i for i, lab in enumerate(dict.fromkeys(label_to_kind))}
        collator_cls: Any = get_class_from_dynamic_module(
            "collate_fn.AllLabelsCollator", model_id, trust_remote_code=True
        )
        collator: Any = collator_cls(
            tok_token, tok_type, label2id, max_seq_length=agents.enrich_ner_max_seq_length
        )
    except Exception as e:
        raise OtterNERUnavailable(
            "failed to load the OTTER NER model",
            context={"model": model_id, "error": str(e)},
        ) from e
    logger.info(
        "enrich.otter_loaded",
        model=model_id,
        device=device,
        labels=labels,
        threshold=agents.enrich_ner_threshold,
    )
    return _OtterHandle(
        model=model,
        tok_token=tok_token,
        collator=collator,
        torch_mod=torch,
        device=device,
        label_to_kind=label_to_kind,
        threshold=agents.enrich_ner_threshold,
    )


def _get_handle() -> _OtterHandle:
    global _handle
    if _handle is not None:
        return _handle
    with _load_lock:
        if _handle is None:
            _handle = _load()
        return _handle


def otter_backend_enabled() -> bool:
    """True iff enrich should use OTTER for entities (fail-safe to LLM outside bootstrap)."""
    try:
        return get_settings().agents.enrich_ner_backend == "otter"
    except ConfigurationError:
        return False


async def extract_chunk_entities(chunk: Chunk) -> list[Entity]:
    """Per-chunk OTTER entities, shaped like the LLM path's `merge_entities` output."""
    handle = await asyncio.to_thread(_get_handle)
    return await asyncio.to_thread(handle.predict_entities, chunk)
