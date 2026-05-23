# Chart-OCR backend shootout — final A/B/C verdict 2026-05-23

The three-way comparison shipped via the 2026-05-23 chart-OCR landscape
research arc. Tests all candidate chart-OCR backends against the
slide-decks corpus (30 queries) to find one that doesn't regress the
prose answering on the CUDA deck.

## Configurations tested

| Path | Backend | Params | License | Repo |
|---|---|---|---|---|
| A | Baseline | — | — | (no chart-OCR; `disable_chart_ocr=true`) |
| B | DePlot | 0.3B | Apache 2.0 | `google/deplot` (Pix2Struct family) |
| C | OneChart | 0.3B | Apache 2.0 | `kppkkp/OneChart` (custom Vary-derived) |
| D | UniChart | 0.2B | Apache 2.0 | `khhuang/chart-to-table` (Donut-style) |
| E | **Nemotron-Parse** | 0.88B | NVIDIA Open | `nvidia/NVIDIA-Nemotron-Parse-v1.2` |

All tested with: Qwen3-8B-AWQ orchestrator, bf16 dispatch, sampling
config (temperature=0.1, top_p=0.8, presence_penalty=1.0, seed=42),
`MEMEX_VLLM_GPU_FRACTION=0.68` for chart-OCR-enabled vaults.

## Results table

| Path | ANS | Δ vs base | refusal_cf | mcp_ans | Chart blocks | Reparse time | CUDA stable | Verdict |
|---|---|---|---|---|---|---|---|---|
| A — Baseline | 11/30 | — | 1.000 | 1.000 | 0 | ~250s | ✓ | reference |
| B — DePlot | 10/30 | **−1** | 1.000 | 0.350* | 50 | ~358s | ✓ | regression |
| C — OneChart | 11/30† | 0 | 1.000 | 1.000 | 0† | ~120s | ✗ CUDA assert | catastrophic |
| D — UniChart | 10/30 | **−1** | 1.000 | 0.233* | 60 | ~78s | ✓ | regression |
| E — **Nemotron-Parse** | **11/30** | **0** | **1.000** | **0.955** | **11** | **~100s** | ✓ | **WINS** |

\* mcp_ans is influenced by chunk_id-changes after reparse making the
   original `relevant_chunk_ids` labels stale. The mcp_ans column tracks
   downstream-impact magnitude: lower = more chunks were perturbed.

† OneChart "11 ANS / 0 blocks" reflects zero useful extractions — every
   figure triggered a CUDA device-side assertion. The "11 ANS" matches
   baseline because the markdown was effectively unchanged.

## Why Nemotron-Parse won — the deep insight

The problem all along was **not chart-OCR quality**, it was **chart-block
insertion frequency**. The P3.3 prose-regression mechanism (documented
in `p33_tracker.md`):

  1. Chart-OCR adds `[chart-extracted]...[/chart-extracted]` blocks to
     the markdown
  2. Chunking absorbs those blocks → new chunk_ids
  3. The downstream FTS5 + dense retrieval now sees DIFFERENT chunks
  4. Top-k retrieval rankings shift, pushing prose chunks below the
     truncate budget
  5. Net: −1 to −5 ANS on prose-heavy corpora

Path C generates only **11 chart blocks** vs Path B's 50 vs Path D's 60.
Fewer block insertions → less retrieval-rank perturbation → prose
retrieval preserved → no regression.

Path E's `<predict_no_text_in_pic>` task token instructs the model to
emit ONLY high-confidence structured content, skipping OCR-of-text-in-
picture noise. This is the architectural lever that finally aligns
chart-OCR with the project's prose-heavy use case.

## Per-backend technical notes

### Path A (Baseline)
Reference state with `MEMEX_PARSE__DISABLE_CHART_OCR=true`. ANS=11.

### Path B (DePlot)
- Pix2Struct VisionEncoderDecoder
- Outputs collapsed `&&&` table per figure
- Generates 50 chart blocks → −1 ANS regression
- CUDA-stable, well-validated
- Decision: superseded by Path E

### Path C (OneChart)
- Custom Vary-derived architecture
- Required `trust_remote_code=True` (ADR-0006 amendment)
- **Catastrophically failed** — every figure triggered CUDA
  `indexSelectSmallIndex` assertion (OPT decoder
  `max_position_embeddings=4096` overflow on OOD imagery)
- Defensive handling held HARD GATES but produced zero useful output
- Known upstream bug: github.com/LingyvKong/OneChart#34
- Decision: stays in tree behind ADR carve-out for future re-attempts
  on chart-heavy corpora or pinned older revisions

### Path D (UniChart / chart-to-table)
- Donut-style VisionEncoderDecoder (donut-swin encoder + mbart decoder)
- max_position_embeddings=1536 (architecturally bounded — no
  OneChart-style overflow)
- Stock transformers, Apache 2.0
- Generates 60 chart blocks → −1 ANS regression
- CUDA-stable; faster than DePlot
- Decision: stays in tree as a strictly safer DePlot alternative

### Path E (Nemotron-Parse-v1.2) — WINNER
- 885M VisionEncoderDecoder; C-RADIO ViT-H encoder + mBART decoder
- NVIDIA Nemotron Open License (derivatives redistributable)
- Required `trust_remote_code=True` (ADR-0006 broadened amendment)
- Loaded via `AutoModel.from_pretrained(..., trust_remote_code=True)`
- Generates 11 chart blocks (most conservative)
- Custom inference recipe with prompt prefix
  `</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>`
- Post-process strips `<bbox_*>`, `<class_*>`, `<x_*>`, `<y_*>`,
  `<md_*>`, `<patch_*>`, `<extra_*>` structure tags via regex
- CUDA-stable across all 245 figures in the CUDA deck
- 2-run determinism confirmed
- Decision: **new default chart-OCR backend** when chart-OCR is
  desired. Default `disable_chart_ocr` stays `True`; operators opt in
  via env var.

## Operator switching guide

```sh
# Default (no chart-OCR) — applies if MEMEX_PARSE__DISABLE_CHART_OCR
# unset or true
unset MEMEX_PARSE__DISABLE_CHART_OCR

# Recommended chart-OCR backend (Path E winner)
MEMEX_PARSE__DISABLE_CHART_OCR=false \
MEMEX_MODELS__CHART_OCR=nvidia/NVIDIA-Nemotron-Parse-v1.2 \
MEMEX_VLLM_GPU_FRACTION=0.68 \
  memex parse <doc-id>

# Required pip deps for Path E (not in core install)
uv pip install albumentations timm open_clip_torch

# Alternative: faster + smaller but with -1 ANS regression on
# prose-heavy corpora (Path D)
MEMEX_PARSE__DISABLE_CHART_OCR=false \
MEMEX_MODELS__CHART_OCR=khhuang/chart-to-table \
  memex parse <doc-id>

# Legacy: DePlot (P3.3 v6 default — -1 ANS regression)
MEMEX_PARSE__DISABLE_CHART_OCR=false \
MEMEX_MODELS__CHART_OCR=google/deplot \
  memex parse <doc-id>

# Disqualified for prose-heavy corpora: OneChart (CUDA-asserts)
# Keep in tree for chart-heavy-corpus future re-attempts only.
```

## Operational notes

- **VRAM contention**: 12 GB rig needs `MEMEX_VLLM_GPU_FRACTION=0.68`
  (vs default 0.72) when chart-OCR is enabled because the eval process
  loads embedder + reranker (~2 GB) alongside vLLM, and chart-stitched
  chunks may push KV cache over the budget.
- **Path E load time**: ~6s after first download (3.75 GB safetensors);
  ~10 min on first download depending on bandwidth.
- **Path E reparse**: ~100s for 63 chart-candidate figures on the
  CUDA deck.

## Path B Nemotron 2-stage pipeline — not separately benchmarked

The plan listed a Path B (`nvidia/nemotron-graphic-elements-v1` +
`nvidia/nemotron-ocr-v2` 2-stage pipeline as NVIDIA's "official"
DePlot replacement). Path E (Nemotron-Parse-v1.2) is NVIDIA's
**single-model** alternative on the same compute stack, with
documented superiority on PubTabNet (TEDS 81.4) + RD-TableBench.
Given Path E wins the prose-preservation criterion at the same
license + vendor tier, the additional integration cost of a
2-stage pipeline is not justified. Path B can be revisited if a
future chart-heavy corpus reveals Path E's coverage gaps.

## Fine-tuning paths (documented but not exercised)

See `docs/audits/chart_ocr_finetune_research_2026-05-23.md` for the
official fine-tuning recipes. Summary:

- **Path D (UniChart)**: feasible on 4070 with `transformers==4.28.1`
  pin (incompatible with Memex's 4.57+; would need a separate env).
  ~6-10h wall-time for ~2k slide-deck samples.
- **Path E (Nemotron-Parse)**: NVIDIA's NeMo AutoModel tutorial uses
  L40S 48GB minimum. Not feasible on 4070 for fine-tuning; inference
  works fine.
- **Path B**: NVIDIA publishes no official training code for
  either model.

For Memex's case, Path E's out-of-box accuracy on slide-deck charts
appears sufficient (the +0 ANS vs baseline tells us we're not LOSING
information by extracting; whether we're GAINING information requires
chart-heavy-corpus measurement, which is the P3.3-c blocker).

## Closing the P3.3 chart-OCR arc

This audit + the chart_ocr_landscape + chart_ocr_finetune audits
collectively close the P3.3 chart-OCR investigation. **Final verdict**:

1. ✅ **Path E (NVIDIA Nemotron-Parse-v1.2)** is the recommended
   chart-OCR backend when chart-OCR is desired
2. **Default stays `disable_chart_ocr=True`** — the +0 ANS result
   means chart-OCR is not LOSING info, but we don't have evidence
   it's GAINING info on the prose-heavy CUDA deck either
3. Path E is the right backend for a future P3.3-c chart-heavy
   corpus test — its conservative extraction profile means it won't
   over-extract on edge cases
4. OneChart + DePlot + UniChart all stay in tree as alternative
   backends; operators can switch via `MEMEX_MODELS__CHART_OCR`
5. ADR-0006 amendment (2026-05-23, P3.3-b + P3.3-c) covers both
   trust_remote_code carve-outs

## Sources

- [chart_ocr_landscape_2026-05-23.md](chart_ocr_landscape_2026-05-23.md) — initial research synthesis
- [chart_ocr_finetune_research_2026-05-23.md](chart_ocr_finetune_research_2026-05-23.md) — fine-tuning recipes for all paths
- [onechart_2026-05-23.md](onechart_2026-05-23.md) — P3.3-b OneChart failure verdict
- [Nemotron-Parse-v1.2 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2)
- [Nemotron-Parse paper (arXiv 2511.20478)](https://arxiv.org/abs/2511.20478)
