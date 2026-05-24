"""Unit tests for the reranker dispatcher (P2.1).

The public `rerank` function dispatches on
`settings.models.reranker_backend`:
  - `cross_encoder` (default) → CrossEncoder.predict(pairs, ...)
  - `qwen3` → Qwen3RerankerHandle, autoregressive softmax-over-yes/no

We don't load any real model here — the registry is monkeypatched to
yield fakes; the qwen3 path's forward pass is patched too. Live model
verification lives on the rig, not in pytest.
"""

from __future__ import annotations

import types
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from memex.core.config import MemexSettings, set_settings
from memex.core.types import Chunk
from memex.retrieve.rerank import _qwen3_format
from memex.retrieve.rerank import cross_encoder_rerank as rerank


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc1",
        document_title="doc1",
        text=text,
    )


@pytest.fixture
def settings_cross_encoder(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> Iterator[MemexSettings]:
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    monkeypatch.delenv("MEMEX_MODELS__RERANKER_BACKEND", raising=False)
    s = MemexSettings()  # type: ignore[call-arg]
    set_settings(s)
    yield s
    set_settings(None)


@pytest.fixture
def settings_qwen3(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemexSettings]:
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("MEMEX_MODELS__RERANKER_BACKEND", "qwen3")
    s = MemexSettings()  # type: ignore[call-arg]
    set_settings(s)
    yield s
    set_settings(None)


class _FakeCrossEncoder:
    """Stand-in for sentence_transformers.CrossEncoder.

    `predict(pairs, ...)` returns a fixed score per pair — the test
    asserts that `rerank` picks the highest-scoring pair as the top
    result and that the qwen3 path is NOT touched.
    """

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self._scores = scores_by_text
        self.predict_calls = 0

    def predict(self, pairs: Any, **kwargs: Any) -> list[float]:
        self.predict_calls += 1
        return [self._scores.get(text, 0.0) for _q, text in pairs]


@asynccontextmanager
async def _registry_with(model: Any) -> AsyncIterator[Any]:
    """Mock `registry.use("reranker")` to yield `model` directly."""
    yield model


@pytest.mark.asyncio
async def test_rerank_uses_cross_encoder_by_default(
    settings_cross_encoder: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default `reranker_backend=cross_encoder` runs CrossEncoder.predict
    and does not touch the qwen3 path.
    """
    fake = _FakeCrossEncoder({"alpha": 0.9, "beta": 0.5, "gamma": 0.1})

    class _FakeRegistry:
        def use(self, name: str) -> Any:
            return _registry_with(fake)

    monkeypatch.setattr("memex.retrieve.rerank.get_registry", lambda: _FakeRegistry())

    out = await rerank(
        "q",
        [_chunk("a", "alpha"), _chunk("b", "beta"), _chunk("c", "gamma")],
        top_k=2,
    )
    assert fake.predict_calls == 1
    assert [c.chunk_id for c in out] == ["a", "b"]
    assert out[0].rerank_score == pytest.approx(0.9)
    assert out[1].rerank_score == pytest.approx(0.5)


class _FakeQwen3Model:
    """Stand-in for the AutoModelForCausalLM call in `_score_qwen3`.

    Returns logits crafted so the softmax([no, yes]) per pair recovers
    a configurable score-by-document-text mapping.
    """

    def __init__(
        self,
        scores_by_text: dict[str, float],
        yes_id: int,
        no_id: int,
        vocab_size: int,
    ) -> None:
        self._scores = scores_by_text
        self.yes_id = yes_id
        self.no_id = no_id
        self.vocab_size = vocab_size
        self.forward_calls = 0
        self._last_batch_texts: list[str] = []

    def __call__(self, **inputs: Any) -> Any:
        import torch

        self.forward_calls += 1
        # Reconstruct which docs are in this batch by detokenising.
        # The fake tokenizer's pad_token_id == 0; non-pad positions
        # carry the integer offset of the document character within
        # the batch — not used for assertion, only structural.
        ids = inputs["input_ids"]
        batch_size, _ = ids.shape

        # Build logits: per row, set logits at yes_id / no_id so the
        # softmax matches the configured score for the doc the row
        # corresponds to. We retrieve the doc text from the side-
        # channel `self._last_batch_texts` (set by the fake tokenizer).
        logits = torch.full((batch_size, 1, self.vocab_size), -10.0, dtype=torch.float32)  # type: ignore[reportPrivateImportUsage]  # torch omits factory/dtype names from __all__
        for i in range(batch_size):
            doc = self._last_batch_texts[i]
            score = self._scores.get(doc, 0.5)
            # We want softmax([no, yes])[1] == score.
            # → yes - no = log(score / (1-score)).
            if 0.0 < score < 1.0:
                log_odds = float(__import__("math").log(score / (1.0 - score)))
            elif score >= 1.0:
                log_odds = 50.0
            else:
                log_odds = -50.0
            logits[i, 0, self.no_id] = 0.0
            logits[i, 0, self.yes_id] = log_odds

        return types.SimpleNamespace(logits=logits.to(torch.bfloat16))  # type: ignore[reportPrivateImportUsage]  # torch omits dtype names from __all__

    def parameters(self) -> Any:  # pragma: no cover — duck-typing
        import torch

        # Single fake parameter so `next(model.parameters()).device`
        # works. Stays on CPU; the inputs.to(device) call no-ops.
        yield torch.zeros(1)  # type: ignore[reportPrivateImportUsage]  # torch omits factory names from __all__


class _FakeQwen3Tokenizer:
    """Tokeniser fake that captures the document text per row so the
    paired fake model can look up its score.
    """

    pad_token_id = 0

    def __init__(self, fake_model: _FakeQwen3Model) -> None:
        self._fake_model = fake_model
        # The fake captures every batch so the model can map row → doc.

    def __call__(
        self,
        prompts: list[str],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 8192,
    ) -> Any:
        import torch

        # Pull `<Document>: …<|im_end|>` out of each prompt for the
        # paired model to score.
        docs: list[str] = []
        for p in prompts:
            start = p.index("<Document>: ") + len("<Document>: ")
            end = p.index("<|im_end|>", start)
            docs.append(p[start:end])
        self._fake_model._last_batch_texts = docs

        # Three dummy ids per row — the model only looks at logits[:, -1, :].
        ids = torch.tensor([[1, 2, 3]] * len(prompts), dtype=torch.long)  # type: ignore[reportPrivateImportUsage]  # torch omits factory/dtype names from __all__
        mask = torch.ones_like(ids)  # type: ignore[reportPrivateImportUsage]  # torch omits factory names from __all__
        return _MoveableTensorDict(input_ids=ids, attention_mask=mask)


class _MoveableTensorDict(dict):
    """Subclass that supports `.to(device)` like `BatchEncoding`."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def to(self, device: Any) -> _MoveableTensorDict:
        return self


@pytest.mark.asyncio
async def test_rerank_uses_qwen3_when_backend_flagged(
    settings_qwen3: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With backend=qwen3 the dispatcher routes to `_score_qwen3` and
    the softmax-over-yes/no extraction recovers the configured scores.
    """
    from memex.models.registry import Qwen3RerankerHandle

    scores = {"alpha-doc": 0.95, "beta-doc": 0.55, "gamma-doc": 0.05}
    vocab_size = 32
    yes_id, no_id = 10, 11
    fake_model = _FakeQwen3Model(scores, yes_id=yes_id, no_id=no_id, vocab_size=vocab_size)
    fake_tokenizer = _FakeQwen3Tokenizer(fake_model)
    handle = Qwen3RerankerHandle(
        tokenizer=fake_tokenizer,  # type: ignore[arg-type]
        model=fake_model,  # type: ignore[arg-type]
        yes_id=yes_id,
        no_id=no_id,
    )

    class _FakeRegistry:
        def use(self, name: str) -> Any:
            return _registry_with(handle)

    monkeypatch.setattr("memex.retrieve.rerank.get_registry", lambda: _FakeRegistry())

    out = await rerank(
        "q",
        [
            _chunk("a", "alpha-doc"),
            _chunk("b", "beta-doc"),
            _chunk("c", "gamma-doc"),
        ],
        top_k=2,
    )
    assert fake_model.forward_calls >= 1
    assert [c.chunk_id for c in out] == ["a", "b"]
    assert out[0].rerank_score == pytest.approx(0.95, abs=0.02)
    assert out[1].rerank_score == pytest.approx(0.55, abs=0.05)


def test_qwen3_format_includes_template_anchors() -> None:
    """The prompt template wires the right system + user + assistant
    markers so the last-token logits sit at the right place for the
    yes/no extraction.
    """
    out = _qwen3_format("the q", "the d")
    assert "<|im_start|>system" in out
    assert "<|im_start|>user" in out
    assert "<Query>: the q" in out
    assert "<Document>: the d" in out
    assert out.endswith("<think>\n\n</think>\n\n")
