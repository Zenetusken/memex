# Chart-OCR landscape research — 2026-05-23

Triggered by the P3.3-b OneChart catastrophic-failure verdict. Three
research agents ran in parallel to answer:

1. **Why did OneChart CUDA-fail?** (root cause)
2. **Does NVIDIA ship a "premier" chart-OCR model?** (official offerings)
3. **What's the broader 2025-2026 chart-OCR landscape?** (top candidates)

This audit captures the synthesis. Three independent agents converged on
the same conclusion: **specialist chart models like DePlot/OneChart
overfit on synthetic distributions**; the safer bet is a Donut-family
VisionEncoderDecoder (architecturally bounded; same family as Nougat
which is well-validated in transformers since 2022).

## 1. OneChart's CUDA failure — diagnosed and (technically) fixable

### Root cause

OPT decoder's `max_position_embeddings=4096` overflow. The model
degenerates on out-of-distribution slide-deck imagery (logos, photos,
GPU architecture diagrams, code-screenshot composites), the JSON
closing-brace never appears in the generated output, and generation
runs to the 4096-token cap. At step 4096, the next position computes
`position_ids = 4096 + 2 = 4098` (OPT's hardcoded `offset=2`); the
embedding table has `num_embeddings = 4098` rows; the `nn.Embedding`
lookup hits row 4098 — a one-past-the-end read.

PyTorch's CUDA `indexSelectSmallIndex` kernel
(`pytorch/aten/src/ATen/native/cuda/Indexing.cu:1515`) asserts
`srcIndex < srcSelectDimSize`. Once asserted, the CUDA context is
poisoned for the entire process; all subsequent kernel launches fail.

### Upstream status

Confirmed as a **known but unfixed upstream bug**: GitHub issue
[LingyvKong/OneChart#34](https://github.com/LingyvKong/OneChart/issues/34)
reports the identical assertion with the HF demo code. No maintainer
response, no PR addressing the unbounded-generation contract.
Repository effectively dormant since 2024.

### Mitigation cost vs value

A ~30-line monkey-patch of `.chat()` to inject a `MaxLengthCriteria(3800)`
plus bounded `max_new_tokens=512` would eliminate the assert. **But this
doesn't help us** — the deeper P3.3 issue isn't OCR accuracy; it's
*retrieval perturbation from chart-extracted blocks*. OneChart fixed
would still produce the −1 ANS regression DePlot causes.

**Decision (already taken in P3.3-b ship)**: OneChart stays in tree
behind the ADR-0006 carve-out for future re-attempts on chart-heavy
corpora; not the path forward on prose-heavy content.

## 2. NVIDIA's chart-OCR offerings — the honest answer

**NVIDIA does NOT ship a "premier chart-to-data foundation model."**
The NVIDIA worldview frames chart extraction as a *pipeline problem*,
decomposed into bounded-box detection + OCR of labels — explicitly
rejecting the DePlot/OneChart pixel-to-number-regression abstraction.

### NeMo Retriever Extraction stack

| Component | HF repo | Params | Role |
|---|---|---|---|
| `nemotron-ocr-v2` | `nvidia/nemotron-ocr-v2` | 84M | Text detection / recognition |
| `nemotron-graphic-elements-v1` | `nvidia/nemotron-graphic-elements-v1` | 54M YOLOX | Detects chart titles / axes / legends / labels (bounding boxes only) |
| `nemoretriever-table-structure-v1` | (NIM container) | YOLOX | Table cell / row / column structure |
| `nemoretriever-page-elements-v2` | (NIM container) | YOLOX | Page-level layout |

**Critical**: DePlot and CACHED were **deprecated from NeMo Retriever
25.3.0**. NVIDIA's official chart-extraction path no longer involves a
chart-to-data foundation model.

### Single-model NVIDIA candidates

| Model | Params | License | ChartQA | Fit | Notes |
|---|---|---|---|---|---|
| `nvidia/NVIDIA-Nemotron-Parse-v1.2` | 885M | NVIDIA Open | n/a (PubTabNet TEDS 81.4) | ~2 GB BF16 | **Strong**: vLLM-compatible, outputs markdown/latex/json/csv. Parser not chart-extractor. |
| `nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1` | 8B | NVIDIA Open + Llama 3.1 Community | 86.3 | 16-18 GB BF16 | **Too big** for 9.5 GB |
| `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` | 12.6B | NVIDIA Open | 89.7 | ~25 GB | Way too big |
| `nvidia/Eagle2.5-8B` | 8B | NSCLv1 (research-only) | 87.5 | 16-18 GB | **License disqualifies** |
| FP4-QAD variants | various | various | as above | Blackwell-only | Not for Ada |

## 3. Top candidates from the broader 2025-2026 landscape

| Candidate | Params | VRAM | License | Architecture | Risk |
|---|---|---|---|---|---|
| **`khhuang/chart-to-table`** | 0.2B | 0.8 GB | **Apache 2.0** | UniChart Donut-style VisionEncoderDecoder | **Lowest** — same family as Nougat, in stock transformers since 2022, bounded position embeddings |
| **`ahmed-masry/chartgemma`** | 3B | 6 GB | MIT | PaliGemma backbone | Low — stock `PaliGemmaForConditionalGeneration`, trained on "in-the-wild" charts |
| **`google/paligemma2-3b-mix-448`** | 3B | 6 GB | Gemma license | PaliGemma 2 | Low — stock transformers, broader visual prior |
| **`microsoft/Florence-2-large`** | 0.77B | 1.5 GB | MIT | DaViT + BART | Low — built-in `<OCR>` tasks; no native chart-to-table |
| ❌ `mPLUG/TinyChart-3B-768` | 3B | 6 GB | Unverified | Custom phi-2 + token-merging | **OneChart-shaped risk** — `trust_remote_code=True` |
| ❌ `OpenGVLab/InternVL3-2B` | 2B | 4 GB | Custom OSS | InternVL | Same custom-modeling risk |
| ❌ `stepfun-ai/GOT-OCR2_0` | varies | varies | Some forks research-only | Custom | License unclear |
| ❌ ChartLlama 13B / ChartAssistant 13B | 13B | 26+ GB | various | various | Exceeds VRAM budget |

## 4. Converged conclusion — Donut/VisionEncoderDecoder family is the safe bet

**All three agents independently arrived at the same architectural insight**:

- Donut-style `VisionEncoderDecoder` models (`khhuang/chart-to-table`,
  Nougat, NVIDIA's `Nemotron-Parse-v1.2` in spirit) use **bounded learned
  absolute position embeddings** — they CAN'T overflow the way OneChart
  did
- Same family has been **hardened in stock transformers since 2022**
- **No `trust_remote_code` risk** — modeling code lives in transformers,
  not in the HF repo
- Output is **delimiter-structured** (deterministic to parse, won't run
  away on weird inputs)

This is the architectural lesson from the OneChart failure: **avoid
custom-modeling chart specialists**. Stick to Donut family OR stock
PaliGemma family.

## 5. Recommendation — three paths to try in sequence

Decision matrix per the user's "go through all paths" directive:

| Path | Model | Verdict criteria |
|---|---|---|
| **A** | `khhuang/chart-to-table` (UniChart, 200M, Apache 2.0) | Highest architectural-safety prediction. Drop-in DePlot replacement. |
| **B** | NeMo Retriever 2-stage (`graphic-elements-v1` + `ocr-v2`) | NVIDIA-validated pipeline. Heavier integration but vendor-blessed. |
| **C** | `nvidia/Nemotron-Parse-v1.2` (885M, NVIDIA Open) | Best NVIDIA single-model. Document-parser; may underperform on chart-numeric estimation. |

For each path: implement → A/B against baseline (no chart-OCR) and
DePlot baseline → measure ANS, mcp_ans, refusal_cf, parse latency.

Subsequent commits ship each path in order; this audit gets updated
with the verdict matrix after all three have been measured.

## 6. What's NOT in scope here

- **Fine-tuning** — each candidate has its own fine-tuning path
  (UniChart's tutorial, Gemma's docs, NVIDIA's recipes). Fine-tuning
  research is separate documents (`docs/audits/chart_ocr_finetune_*.md`).
- **OneChart re-attempt** — explicitly out of scope per the P3.3-b ship
  verdict (in tree but not the recommended path).
- **Larger VLMs** (Nano-VL-8B, ChartGemma 7B, Llama-3.2-Vision-11B) —
  exceed the 9.5 GB chart-OCR slot on the reference rig.
- **TinyChart / InternVL** — explicitly disqualified due to custom-
  modeling risk identical to OneChart's.

## Sources (concatenated from all three agents)

**OneChart root cause:**
- [kppkkp/OneChart on Hugging Face](https://huggingface.co/kppkkp/OneChart)
- [OneChart config.json](https://huggingface.co/kppkkp/OneChart/blob/main/config.json)
- [OneChart modeling_OneChart.py](https://huggingface.co/kppkkp/OneChart/blob/main/modeling_OneChart.py)
- [OneChart GitHub repo](https://github.com/LingyvKong/OneChart)
- [OneChart Issue #34](https://github.com/LingyvKong/OneChart/issues/34) — identical CUDA assertion, unresolved
- [transformers OPT modeling source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py)
- [transformers issue #20894 — `max_length` vs `max_new_tokens` warning origin](https://github.com/huggingface/transformers/issues/20894)
- [OneChart paper (arXiv 2404.09987)](https://arxiv.org/abs/2404.09987)

**NVIDIA offerings:**
- [NVIDIA NeMo-Retriever GitHub](https://github.com/NVIDIA/NeMo-Retriever)
- [NeMo Retriever extraction overview](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/)
- [Nemotron OCR v2 model card](https://huggingface.co/nvidia/nemotron-ocr-v2)
- [Nemotron Parse v1.2 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2)
- [Nemotron Graphic Elements v1 model card](https://huggingface.co/nvidia/nemotron-graphic-elements-v1)
- [Llama-3.1-Nemotron-Nano-VL-8B-V1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1)
- [NVIDIA Open Model License](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf)
- [Nemotron-Parse paper (arXiv 2511.20478)](https://arxiv.org/abs/2511.20478)

**Broader landscape:**
- [ahmed-masry/chartgemma](https://huggingface.co/ahmed-masry/chartgemma)
- [ChartGemma paper (arXiv 2407.04172)](https://arxiv.org/html/2407.04172v1)
- [khhuang/chart-to-table](https://huggingface.co/khhuang/chart-to-table)
- [UniChart paper (arXiv 2305.14761)](https://arxiv.org/pdf/2305.14761)
- [google/paligemma2-3b-mix-448](https://huggingface.co/google/paligemma2-3b-mix-448)
- [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large)
- [ChartQAPro benchmark (arXiv 2504.05506)](https://arxiv.org/abs/2504.05506) — shows even Claude Sonnet 3.5 drops 90.5→55.8 on harder real-world charts
