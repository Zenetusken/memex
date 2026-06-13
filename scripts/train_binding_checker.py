"""Fine-tune the binding-fabrication checker (audit-19 design §3).

VERDICT — DO NOT RE-WALK (audit-19 §9 K2): the content class (ar-12 binding
fabrication) is NOT separable by a scoped (evidence, ±question, answer) checker.
Three trained candidates + eight zero-shot arms exhausted the design space; the
kill is STRUCTURAL (question-echo entanglement, not data/scale — dev example-F1
0.83-0.88 every candidate). The companion GENERATION-time lever (answer@v6) was
ALSO tried + reverted (§10, prompt blast radius). This stack is retained ONLY as a
re-runnable artifact for a future MASKED-SUBJECT / complete-evidence-entailment
design — NOT a same-scoping tuning revisit. mxbai stays blocked on the content
class until that (order-of-magnitude-larger) machinery is warranted.

Training mix = vault mints (scripts/mint_binding_data.py) + RAGTruth-EN replay
(preprocessed VERBATIM per the vendored lettucedetect recipe) + ragtruth-fr-translated
replay — the replay keeps the general-hallucination skill and the clean-FP profile;
the mints add the binding class no public training set carries.

Checkpoint selection AND threshold freezing happen on the MINTED DEV SPLIT only
(by-document holdout). The frozen calibration set (data-17) is a one-shot gate that
this script never reads — design §2.6/§4.

Run with the vLLM daemon STOPPED (the 307M @ seq-4096 train step needs the GPU):
    uv run memex daemon stop
    uv run python scripts/train_binding_checker.py --out ~/.memex/binding-checker/cand1
    uv run memex daemon start
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from binding_checker_vendor import (
    HallucinationDataset,
    HallucinationSample,
    Trainer,
    evaluate_model,
)


def _h(*parts: object) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest(), 16)


def load_mints(path: str, span_mode: str) -> tuple[list[HallucinationSample], list[HallucinationSample]]:
    rows = json.load(open(path))
    train, dev = [], []
    for r in rows:
        if span_mode == "whole" and r["labels"]:
            r = {**r, "labels": [{"start": 0, "end": len(r["answer"])}]}
        s = HallucinationSample.from_json(r)
        (dev if r["split"] == "dev" else train).append(s)
    return train, dev


def load_ragtruth_en(data_dir: Path, n: int) -> list[HallucinationSample]:
    """The vendored preprocess_ragtruth.py recipe, verbatim: prompt =
    source['prompt'], answer = response['response'], labels = char spans."""
    sources = {
        s["source_id"]: s
        for s in (json.loads(line) for line in (data_dir / "source_info.jsonl").read_text().splitlines())
    }
    out = []
    for line in (data_dir / "response.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["split"] != "train":
            continue
        src = sources[r["source_id"]]
        labels = [
            {"start": lb["start"], "end": lb["end"], "label": lb["label_type"]}
            for lb in r["labels"]
        ]
        out.append(
            HallucinationSample(
                prompt=src["prompt"], answer=r["response"], labels=labels,
                split="train", task_type=src["task_type"], dataset="ragtruth",
                language="en",
            )
        )
    out.sort(key=lambda s: _h("en", s.prompt[:64], s.answer[:64]))
    return out[:n]


def load_ragtruth_fr(n: int) -> list[HallucinationSample]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        "KRLabsOrg/ragtruth-fr-translated", "data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    rows = pq.read_table(p).to_pylist()
    out = [
        HallucinationSample(
            prompt=r["prompt"], answer=r["answer"], labels=list(r["labels"] or []),
            split="train", task_type=r.get("task_type", "qa"), dataset="ragtruth-fr",
            language="fr",
        )
        for r in rows
        if r["prompt"] and r["answer"]
    ]
    out.sort(key=lambda s: _h("fr", s.prompt[:64], s.answer[:64]))
    return out[:n]


def make_collate(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate(batch):
        n = max(b["input_ids"].shape[0] for b in batch)

        def pad(t, v):
            return torch.cat([t, torch.full((n - t.shape[0],), v, dtype=t.dtype)])

        return {
            "input_ids": torch.stack([pad(b["input_ids"], pad_id) for b in batch]),
            "attention_mask": torch.stack([pad(b["attention_mask"], 0) for b in batch]),
            "labels": torch.stack([pad(b["labels"], -100) for b in batch]),
        }

    return collate


def example_scores(model, tokenizer, samples, max_length, device):
    """Per-example ld_max_conf (max class-1 prob among predicted-1 answer tokens) —
    the same thresholdable scalar the probe's lettuce_arm records."""
    model.eval()
    ds = HallucinationDataset(samples, tokenizer, max_length)
    scores = []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            logits = model(
                item["input_ids"].unsqueeze(0).to(device),
                attention_mask=item["attention_mask"].unsqueeze(0).to(device),
            ).logits[0]
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            mask = (item["labels"].to(device) != -100) & (preds == 1)
            scores.append(float(probs[mask, 1].max()) if bool(mask.any()) else 0.0)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mints", default="/tmp/binding_mints_full.json")  # noqa: S108 — mint artifact
    ap.add_argument("--ragtruth-dir", default=str(Path.home() / ".memex/binding-checker/ragtruth"))
    ap.add_argument("--replay-en", type=int, default=3000)
    ap.add_argument("--replay-fr", type=int, default=1500)
    ap.add_argument("--base", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--out", default=str(Path.home() / ".memex/binding-checker/cand1"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--span-mode", choices=["slot", "whole"], default="slot")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForTokenClassification.from_pretrained(args.base, num_labels=2)
    if device.type == "cuda":
        model.gradient_checkpointing_enable()  # seq-4096 activations on a 12 GB card

    mint_train, mint_dev = load_mints(args.mints, args.span_mode)
    en = load_ragtruth_en(Path(args.ragtruth_dir), args.replay_en)
    fr = load_ragtruth_fr(args.replay_fr)
    train = mint_train + en + fr
    train.sort(key=lambda s: _h("shuffle", s.prompt[:48], s.answer[:48]))
    print(
        f"[train] mints {len(mint_train)} (+dev {len(mint_dev)}) | replay en {len(en)} "
        f"fr {len(fr)} | total {len(train)} | device {device} | span-mode {args.span_mode}",
        flush=True,
    )

    collate = make_collate(tokenizer)
    train_loader = DataLoader(
        HallucinationDataset(train, tokenizer, args.max_length),
        batch_size=args.batch, shuffle=False, collate_fn=collate,
    )
    dev_loader = DataLoader(
        HallucinationDataset(mint_dev, tokenizer, args.max_length),
        batch_size=args.batch, shuffle=False, collate_fn=collate,
    )

    trainer = Trainer(
        model, tokenizer, train_loader, dev_loader,
        epochs=args.epochs, learning_rate=args.lr, save_path=args.out,
        device=device, grad_accum=args.grad_accum, bf16=device.type == "cuda",
    )
    best = trainer.train()
    print(f"[train] best dev span-F1: {best:.4f}", flush=True)

    # ---- threshold freeze on the minted dev split (NEVER on the calibration set)
    best_model = AutoModelForTokenClassification.from_pretrained(args.out).to(device)
    scores = example_scores(best_model, tokenizer, mint_dev, args.max_length, device)
    truth = [bool(s.labels) for s in mint_dev]
    grid = sorted({round(s, 3) for s in scores} | {0.5})
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        tp = sum(1 for s, y in zip(scores, truth, strict=True) if y and s >= t)
        fp = sum(1 for s, y in zip(scores, truth, strict=True) if not y and s >= t)
        fn = sum(1 for s, y in zip(scores, truth, strict=True) if y and s < t)
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    fp_rate = sum(1 for s, y in zip(scores, truth, strict=True) if not y and s >= best_t) / max(
        1, sum(1 for y in truth if not y)
    )
    # detect question-blind from the mints (the breach subject is in the question, so a
    # question-blind checker must be GATED in noq mode — audit-19 §8)
    question_blind = all(
        "the following question:\n\nBear in mind" in s.prompt for s in mint_dev[:20]
    )
    meta = {
        "threshold": best_t, "dev_example_f1": round(best_f1, 4),
        "dev_fp_rate": round(fp_rate, 4), "base": args.base,
        "span_mode": args.span_mode, "max_length": args.max_length,
        "train_size": len(train), "dev_size": len(mint_dev),
        "question_blind": question_blind,
    }
    json.dump(meta, open(Path(args.out) / "threshold.json", "w"), indent=1)
    print(f"[train] frozen threshold {best_t} (dev example-F1 {best_f1:.3f}, "
          f"dev FP rate {fp_rate:.3f}) -> {args.out}/threshold.json", flush=True)
    # token-level dev report for the record
    print(json.dumps(evaluate_model(best_model, dev_loader, device), indent=1), flush=True)


if __name__ == "__main__":
    main()
