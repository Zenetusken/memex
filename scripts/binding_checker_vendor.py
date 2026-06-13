"""Vendored LettuceDetect training stack (MIT, KRLabsOrg/LettuceDetect), pinned to
the lettucedetect==0.1.8 wheel cached at
/home/drei/.cache/uv/archive-v0/DiSmg5FZr2zmpe92C7t8d/ — the pip package overlay
previously broke the project torch combo (audit-18 ops lesson), so the ~250
load-bearing lines are transcribed here instead of installed.

Sources (verbatim except where marked MEMEX):
  datasets/hallucination_dataset.py  -> HallucinationSample, HallucinationDataset
  models/trainer.py                  -> Trainer
  models/evaluator.py                -> evaluate_model

MEMEX changes (each marked inline):
  1. `answer_start_token` is located via fast-tokenizer `sequence_ids()` instead of
     re-encoding the context alone — the upstream heuristic mislocates the answer
     whenever the context alone exceeds max_length (probed 2026-06-12: both the
     lettucedect-en tokenizer and mmBERT's fail identically on a 6000-word context;
     sequence_ids reconstructs the answer exactly on both).
  2. `HallucinationSample` keeps the upstream field set but accepts any `language`
     string and ignores unknown JSON keys (our mints carry a `meta` block).
  3. Trainer: gradient-accumulation + bf16 autocast options for the 12 GB rig;
     save-best logic unchanged (class-1 span F1 on the dev loader).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class HallucinationSample:
    prompt: str
    answer: str
    labels: list[dict]
    split: str
    task_type: str
    dataset: str
    language: str
    meta: dict = field(default_factory=dict)  # MEMEX: mint provenance

    def to_json(self) -> dict:
        return {
            "prompt": self.prompt,
            "answer": self.answer,
            "labels": self.labels,
            "split": self.split,
            "task_type": self.task_type,
            "dataset": self.dataset,
            "language": self.language,
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, d: dict) -> HallucinationSample:
        return cls(
            prompt=d["prompt"],
            answer=d["answer"],
            labels=d["labels"],
            split=d["split"],
            task_type=d.get("task_type", "qa"),
            dataset=d.get("dataset", "unknown"),
            language=d.get("language", "en"),
            meta=d.get("meta", {}),
        )


class HallucinationDataset(Dataset):
    """Token-classification dataset: prompt+answer pair-encoded, prompt loss-masked
    with -100, answer tokens labeled 1 where they overlap a hallucination span."""

    def __init__(self, samples: list[HallucinationSample], tokenizer: Any,
                 max_length: int = 4096):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def prepare_tokenized_input(
        tokenizer: Any, context: str, answer: str, max_length: int
    ) -> tuple[Any, list[int], torch.Tensor, int]:
        encoding = tokenizer(
            context,
            answer,
            truncation="only_first",
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        offsets = encoding.pop("offset_mapping")[0]

        # MEMEX change 1: locate the answer region via sequence_ids() — robust under
        # only_first truncation (upstream re-encoded the context alone, which
        # overshoots max_length and loses the answer entirely for long contexts).
        seq_ids = encoding.encodings[0].sequence_ids
        answer_token_idx = [i for i, s in enumerate(seq_ids) if s == 1]
        answer_start_token = answer_token_idx[0] if answer_token_idx else encoding[
            "input_ids"
        ].shape[1]

        labels = [-100] * encoding["input_ids"].shape[1]
        return encoding, labels, offsets, answer_start_token

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        encoding, labels, offsets, answer_start = self.prepare_tokenized_input(
            self.tokenizer, sample.prompt, sample.answer, self.max_length
        )

        answer_char_offset = offsets[answer_start][0] if answer_start < len(offsets) else None

        for i in range(answer_start, encoding["input_ids"].shape[1]):
            token_start, token_end = offsets[i]
            if token_start == token_end:  # special token inside/after the answer
                continue
            token_abs_start = (
                token_start - answer_char_offset if answer_char_offset is not None else token_start
            )
            token_abs_end = (
                token_end - answer_char_offset if answer_char_offset is not None else token_end
            )
            token_label = 0
            for ann in sample.labels:
                if token_abs_end > ann["start"] and token_abs_start < ann["end"]:
                    token_label = 1
                    break
            labels[i] = token_label

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def evaluate_model(model: Any, dataloader: Any, device: Any) -> dict[str, Any]:
    from sklearn.metrics import precision_recall_fscore_support

    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            predictions = torch.argmax(outputs.logits, dim=-1)
            mask = batch["labels"] != -100
            all_preds.extend(predictions.cpu()[mask].tolist())
            all_labels.extend(batch["labels"][mask].tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1], average=None, zero_division=0
    )
    return {
        "supported": {"precision": float(precision[0]), "recall": float(recall[0]),
                      "f1": float(f1[0])},
        "hallucinated": {"precision": float(precision[1]), "recall": float(recall[1]),
                         "f1": float(f1[1])},
    }


class Trainer:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        train_loader: Any,
        test_loader: Any,
        epochs: int = 6,
        learning_rate: float = 1e-5,
        save_path: str = "best_model",
        device: torch.device | None = None,
        grad_accum: int = 1,      # MEMEX change 3
        bf16: bool = False,       # MEMEX change 3
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.save_path = save_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grad_accum = grad_accum
        self.bf16 = bf16
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.model.to(self.device)

    def train(self) -> float:
        import time

        best_f1 = 0.0
        for epoch in range(self.epochs):
            t0 = time.time()
            self.model.train()
            total_loss, num_batches = 0.0, 0
            self.optimizer.zero_grad()
            for step, batch in enumerate(self.train_loader):
                ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if self.bf16 and self.device.type == "cuda"
                    else torch.enable_grad()
                )
                with ctx:
                    outputs = self.model(
                        batch["input_ids"].to(self.device),
                        attention_mask=batch["attention_mask"].to(self.device),
                        labels=batch["labels"].to(self.device),
                    )
                    loss = outputs.loss / self.grad_accum
                loss.backward()
                if (step + 1) % self.grad_accum == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                total_loss += float(loss.item()) * self.grad_accum
                num_batches += 1
                if num_batches % 100 == 0:
                    print(f"  epoch {epoch + 1} step {num_batches} "
                          f"avg_loss {total_loss / num_batches:.4f}", flush=True)
            metrics = evaluate_model(self.model, self.test_loader, self.device)
            f1 = metrics["hallucinated"]["f1"]
            print(
                f"epoch {epoch + 1}/{self.epochs} done in {time.time() - t0:.0f}s "
                f"avg_loss {total_loss / max(num_batches, 1):.4f} "
                f"dev hallucinated P/R/F1 "
                f"{metrics['hallucinated']['precision']:.3f}/"
                f"{metrics['hallucinated']['recall']:.3f}/{f1:.3f}",
                flush=True,
            )
            if f1 > best_f1:
                best_f1 = f1
                self.model.save_pretrained(self.save_path)
                self.tokenizer.save_pretrained(self.save_path)
                print(f"  new best F1 {best_f1:.4f} -> saved {self.save_path}", flush=True)
        return best_f1
