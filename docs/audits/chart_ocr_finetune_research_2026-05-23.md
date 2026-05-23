# Chart-OCR fine-tuning research — 2026-05-23

Companion to `chart_ocr_landscape_2026-05-23.md`. Three research agents
investigated the official fine-tuning paths for the three top
candidates (Path A: UniChart / chart-to-table; Path B: NeMo Retriever
2-stage; Path C: Nemotron-Parse-v1.2). All findings come from
authoritative upstream sources only (HF model cards, GitHub repos,
arxiv papers, NVIDIA developer blog).

## Path A — `khhuang/chart-to-table` (UniChart Donut-style)

**Official workflow**: only training script is `finetune_chartqa.py` in
[vis-nlp/UniChart](https://github.com/vis-nlp/UniChart). The
`khhuang/CHOCOLATE` repo marks training/eval scripts as "Coming soon"
— not released. So fine-tuning must reuse UniChart's ChartQA harness
and substitute the chart-to-table prompt/target.

**Command (from UniChart README)**:
```bash
python finetune_chartqa.py \
  --data-path "ahmed-masry/chartqa_without_images" \
  --train-images '/path/to/train/png/' \
  --valid-images '/path/to/val/png' \
  --max-steps 40000 --batch-size 8 --valid-batch-size 1 \
  --num-workers 12 --lr 5e-5 --check-val-every-n-epoch 1 \
  --warmup-steps 100 --checkpoint-steps 7000 \
  --checkpoint-path "ahmed-masry/unichart-base-960"
```

**Pinned dependencies**: `transformers==4.28.1`, `pytorch-lightning==1.8.5`,
`datasets`, `sentencepiece`. README explicitly demands the exact
transformers version — newer versions break `VisionEncoderDecoderModel.generate()`
kwargs. **This is a potential conflict for Memex which is on 4.57+**;
fine-tuning would need a separate environment.

**Data format**:
```json
{
  "imgname": "deck_chart_001.png",
  "query": "<data_table_generation>",
  "label": "Year | Revenue | Users &&& 2020 | 1.2 | 50 &&& 2021 | 2.4 | 110"
}
```
- `&&&` = row delimiter, `|` = column delimiter
- Loss masking: prompt tokens up to `<s_answer>` set to `ignore_id=-100`
- Max sequence length: 512

**Hardware feasibility on 4070 12 GB**: paper used 4×V100 32GB at
batch 24 for 20 epochs. Single 4070 viable with `batch_size=1-2` +
`accumulate_grad_batches=8-16`, gradient checkpointing if OOM.
**LoRA/QLoRA not officially supported**.

**Hyperparameters** (from paper + script):
- Optimizer: Adam (not AdamW), default betas (0.9, 0.999), wd=0
- LR: 5e-5
- Scheduler: Linear warmup → cosine decay
- Warmup steps: 100 (README) / 50 (script default)
- Max steps: 40k (paper: 20 epochs ChartQA)
- Batch size: 24 (paper, multi-GPU) / 8 (README)
- Precision: fp16
- Max seq len: 512
- Loss: standard CE on token sequence (prompt masked with `-100`)

**Eval metric**: chart-to-table uses **RNSS** (Relative Number-Set
Similarity) per the UniChart paper — NOT ChartQA's relaxed accuracy.
The val loop in `finetune_chartqa.py` uses relaxed accuracy; you'd
need to swap it.

**Domain adaptation for slide-deck charts**:
- Base trained on PlotQA + Statista/OECD/OWID + ChartInfo + Beagle +
  ExcelChart (601k charts, web-scraped)
- Likely weak on Keynote/PowerPoint screenshots, stylized fonts,
  pastel palettes, decorative grid removal
- Recipe: synthetic (matplotlib/plotly themed) + ~500-2000 hand-
  labelled deck screenshots; continue-train from
  `khhuang/chart-to-table` (not `unichart-base-960`)

**Expected runtime on 4070**: ~6-10 hours for 4k steps at effective
batch 16 with ~2k slide-deck chart samples.

**Pitfalls**:
- `transformers==4.28.1` pin (incompatible with current Memex
  environment)
- No LoRA recipe ([UniChart Issue #7](https://github.com/vis-nlp/UniChart/issues/7) open since Oct 2023)
- No published training script for chart-to-table specifically —
  must re-purpose ChartQA harness
- RNSS eval not in the default val loop

## Path B — NeMo Retriever 2-stage (`graphic-elements-v1` + `ocr-v2`)

**NVIDIA does NOT publish official training/fine-tuning code or a
NeMo recipe for either model.** This is the headline finding.

**`graphic-elements-v1` (54M YOLOX)**:
- HF model card ships inference-only code
- NVIDIA explicitly defers to [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
  for training
- TAO Toolkit ships YOLOv3/v4 only, not YOLOX
- NeMo Microservices Customizer doesn't list these vision models

```bash
# Practical entry point (Megvii upstream, NOT NVIDIA-blessed):
python -m yolox.tools.train -n yolox-l -d 1 -b 8 --fp16 \
  -c <path-to-graphic-elements-checkpoint>
```

**`nemotron-ocr-v2` (84M RegNetX-8GF + transformer recognizer)**:
- HF model card only documents inference
- NVIDIA's emelryan stated on the v1 HF discussion: *"We do not have
  any immediate plans to release training code"* (still true for v2)
- **NO official fine-tuning path exists.**

**Hardware feasibility**:
- `graphic-elements-v1` YOLOX-L: bs=4-8 stable on 12 GB at 1024px
- `nemotron-ocr-v2`: NVIDIA's tested HW list = H100/A100/L40S/L4/A10G/
  H200/B200/RTX PRO 6000. 4070 not listed but matches supported
  microarchitecture (Lovelace). Training undocumented.

**Data format**:
- `graphic-elements-v1`: COCO JSON. Classes: chart_title, x_title,
  y_title, xlabel, ylabel, legend_title, legend_label, mark_label,
  value_label, other
- `nemotron-ocr-v2`: not documented by NVIDIA. Community pattern:
  `{image_path: text}` JSON dict. v2_multilingual charset = 14,244 chars

**Hyperparameters** (YOLOX upstream defaults):
- 300 epochs, 5-epoch warmup, SGD momentum 0.9
- LR 0.01 / 64-batch (linear-scaled)
- Cosine schedule
- Mosaic+mixup off in last 15 epochs

**License**: NVIDIA Open Model License explicitly permits derivative
models (fine-tuned weights), commercial redistribution. Requirement:
bundle license copy + NOTICE file. Post-processing scripts:
Apache 2.0.

**Concrete path forward**:
1. `graphic-elements-v1`: clone Megvii YOLOX, load NVIDIA weights as
   `-c`, COCO-format slide-deck corpus (~500-2000 labeled chart crops
   minimum), train at bs=4-8, LR=1e-4, 50-100 epochs. Wall-time:
   6-18 hours.
2. `nemotron-ocr-v2`: **no supported path**. Use out-of-the-box and
   route domain-shifted OCR elsewhere, or wait for NVIDIA's training-
   code release.

## Path C — `nvidia/NVIDIA-Nemotron-Parse-v1.2`

**Official workflow**: NeMo AutoModel (PyTorch DTensor-native), not
classic NeMo `.nemo`, not HF PEFT/Trainer. Tutorial published for
v1.1; same recipe applies to v1.2 (same architecture, only
prompt-prefix + input resolution changed).

```bash
torchrun --nproc_per_node=auto Automodel/examples/vlm_finetune/finetune.py \
    --config nemotron_parse_config.yaml
```

**Mode**: full fine-tune of the mBART decoder with the vision tower
frozen (`freeze_vision_tower: true`, `freeze_language_model: false`).
**No LoRA/PEFT path published by NVIDIA.**

**Data format**: HF Dataset rows of `{image: PIL.Image, ground_truth:
JSON-string}` where `ground_truth.gt_parse` is the target structure.
The AutoModel collator handles image preprocessing in code.

**Hardware feasibility on 4070 12 GB**:
- Inference/serving: **feasible** (<4 GB BF16, vLLM-compatible)
- Fine-tuning: **not officially supported on consumer GPUs**. Tutorial
  uses L40S 48GB. Full-decoder fine-tune at bf16 + image batch 1 needs
  ~16-22 GB just for activations at 1664×2048 + decoder optimizer
  states.
- No QLoRA recipe; reducing input resolution + grad-checkpointing +
  DeepSpeed-Zero-3 CPU offload is uncharted

**Hyperparameters** (verbatim from tutorial YAML):
- `global_batch_size`: 8, `local_batch_size`: 1
- `max_steps`: 50 (for 425-row invoice demo, ~1 epoch)
- `val_every_steps`: 25
- `lr`: 5e-5
- `freeze_vision_tower`: true, `freeze_language_model`: false
- dtype: bf16

**vLLM serving post-fine-tune** (fully supported):
```bash
vllm serve <your-finetuned-ckpt> --dtype bfloat16 --max-num-seqs 8 \
  --limit-mm-per-prompt '{"image": 1}' --trust-remote-code --port 8000 \
  --chat-template chat_template.jinja
```

**Domain adaptation**: NVIDIA's invoice demo showed 425 examples / 1
epoch driving NED from ~0.81 → ~0.10 and field-accuracy from 0 → 80%+.
No chart-specific generator released. Invoice tutorial is the
canonical analog.

**License**: NVIDIA Nemotron Open Model License — derivative weights
publicly redistributable, full commercial use, no field-of-use
restrictions, no distillation ban. Requires NOTICE attribution.

**Performance benchmarks (paper Table 4/5)**:
- vs Nougat: overall WER 0.131 vs Nougat 0.452 (~3.5× lower error)
- vs GOT-OCR2: near-parity on GOT Dense OCR (F1 0.9785); Nemotron
  stronger on layout-preserving + bbox tasks
- vs Donut: not compared head-to-head; Nemotron is ~5× the params
  with reading-order/bbox prompt tokens
- Best on PubTabNet TEDS 81.3, RD-TableBench 86.2, multilingual
  F1 0.96-0.98

**Realistic Memex play**: **inference-only as a chart-OCR backend**
(fine-tuning is out of budget on 4070).

```bash
vllm serve nvidia/NVIDIA-Nemotron-Parse-v1.2 --dtype bfloat16 \
  --max-num-seqs 4 --limit-mm-per-prompt '{"image": 1}' \
  --trust-remote-code --port 8001
# Prompt: "</s><s><predict_bbox><predict_classes><output_markdown>
#         <predict_no_text_in_pic>"
```

Expected throughput: 1-3 chart-images/sec on Ada at 1664×2048.

## Cross-path comparison

| Aspect | Path A (chart-to-table) | Path B (Nemotron 2-stage) | Path C (Nemotron-Parse-v1.2) |
|---|---|---|---|
| Fine-tune feasible on 4070 | ✅ Yes (6-10h, batch=1-2 + accumulation) | ✅ Partial (graphic-elements only via Megvii YOLOX) | ❌ No (needs L40S 48GB minimum) |
| Official NVIDIA recipe | N/A (community) | ❌ None published | ✅ NeMo AutoModel tutorial (for v1.1) |
| LoRA / PEFT support | ❌ Not published | ❌ Not published | ❌ Not published |
| Inference VRAM | ~0.5 GB BF16 | ~0.5 GB combined | ~4 GB BF16 |
| Data volume estimate | 500-2000 labeled samples | 500-2000 COCO annotations | 425+ samples (invoice demo) |
| Training script in tree | ✅ `finetune_chartqa.py` | ❌ Must use Megvii upstream | ✅ AutoModel tutorial |
| Eval metric | RNSS (need to add to val loop) | mAP (standard YOLOX) | NED, field accuracy |
| License (fine-tuned weights) | Apache 2.0 (UniChart base) | NVIDIA Open (redistributable) | NVIDIA Nemotron Open (redistributable) |
| transformers version | **Pin: 4.28.1** (incompat with Memex 4.57+) | Any | Any |

## Sources

**Path A:**
- [vis-nlp/UniChart GitHub](https://github.com/vis-nlp/UniChart)
- [UniChart paper (arXiv 2305.14761)](https://arxiv.org/html/2305.14761)
- [khhuang/chart-to-table HF](https://huggingface.co/khhuang/chart-to-table)
- [khuangaf/CHOCOLATE](https://github.com/khuangaf/CHOCOLATE)
- [UniChart Issue #7 (LoRA)](https://github.com/vis-nlp/UniChart/issues/7)

**Path B:**
- [nvidia/nemotron-graphic-elements-v1](https://huggingface.co/nvidia/nemotron-graphic-elements-v1)
- [nvidia/nemotron-ocr-v2](https://huggingface.co/nvidia/nemotron-ocr-v2)
- [NVIDIA/NeMo-Retriever GitHub](https://github.com/NVIDIA/NeMo-Retriever)
- [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)

**Path C:**
- [Nemotron-Parse-v1.2 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2)
- [Nemotron-Parse paper (arXiv 2511.20478)](https://arxiv.org/abs/2511.20478)
- [Nemotron-Parse fine-tune tutorial](https://github.com/NVIDIA-NeMo/Automodel/blob/main/tutorials/nemotron-parse/finetune.ipynb)
- [NVIDIA-NeMo/Automodel](https://github.com/NVIDIA-NeMo/Automodel)
- [Nemotron Parse 1.1 blog post](https://developer.nvidia.com/blog/turn-complex-documents-into-usable-data-with-vlm-nvidia-nemotron-parse-1-1/)
