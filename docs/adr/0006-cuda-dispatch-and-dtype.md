# ADR-0006: CUDA Toolkit Pin, Dtype Dispatch, and Attention Implementation

- **Status**: Accepted
- **Date**: 2026-05-20
- **Deciders**: Memex core team
- **Tags**: stack, gpu, models, dispatch

## Context

ADR-0001 commits Memex to NVIDIA CUDA-only inference, but neither the code nor the docs encode the specific toolkit version, dtype policy, attention implementation, or PyTorch-wheel resolution strategy. A multi-agent audit on 2026-05-20 surfaced six places where this matters for correctness, not just performance:

- `pyproject.toml` ships the CPU-only torch wheel on default `uv sync` because no wheel index is declared. Every "GPU" path silently lands on CPU.
- `models/registry.py:_load_embedder` loads EmbeddingGemma 300M without a dtype, defaulting to FP16. The official model card states *"activations do not support FP16"* — required dtype is FP32 or BF16.
- `models/registry.py:_load_vlm` uses `AutoModelForCausalLM` for Qwen2.5/3-VL, which is the wrong class — VLMs need `AutoModelForImageTextToText` (transformers 4.57+) or the explicit model class.
- `cli/bootstrap.py` does not assert `torch.cuda.is_available()`, contradicting ADR-0001's "no CPU fallback" promise.
- `HardwareSettings.gpu_memory_fraction` is defined but never read by any callsite.
- The default `ModelSettings.vlm = "Qwen/Qwen2.5-VL-7B-Instruct"` is ~15–18 GB in BF16 — won't fit on the 12 GB reference card.

The stack also moved underneath ADR-0001's prose: vLLM is on V1 unconditionally (no `VLLM_USE_V1` flag), CUDA 12.8/12.9 wheels are current, Flash-Attention 3 is not available on Ada Lovelace (sm_89), and the `transformers` / `sentence-transformers` APIs both grew first-class dtype controls.

This ADR settles five questions in one place so the codebase, the pyproject, and the docs all agree.

## Decision Drivers

- Correctness on the reference RTX 4070 (12 GB, Ada Lovelace, sm_89) over generality
- Single source of truth for the GPU stack — neither the code nor the docs should encode an implicit dispatch rule
- Fail loudly at startup (per GUIDELINES.md "Configuration"); never lose performance silently
- The dependency landscape moves; pin to *versions that exist today*, document the assumed driver floor

## Decisions

### 1. CUDA toolkit and PyTorch wheel

- **CUDA toolkit floor**: 12.8 (paired with NVIDIA driver R570+).
- **PyTorch wheel**: `torch>=2.7` resolved against the PyTorch `cu128` index, declared explicitly in `pyproject.toml`. The default PyPI wheel ships CPU-only — relying on it is the difference between a working system and a system that silently runs every model on CPU.
- **uv configuration** (in `pyproject.toml`):
  ```toml
  [[tool.uv.index]]
  name = "pytorch-cu128"
  url = "https://download.pytorch.org/whl/cu128"
  explicit = true

  [tool.uv.sources]
  torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
  ```
- Memex requires NVIDIA on Linux. macOS and Windows users get whatever the default PyPI wheel resolves to (typically CPU on macOS, cu128 on Windows); the runtime CUDA assert below catches the rest.

### 2. Dtype dispatch

- **BF16 is the default across the stack on Ada.** sm_89 has native bf16 tensor cores and bf16's exponent range avoids the activation overflows FP16 hits on certain models. We pin bf16 explicitly rather than relying on `torch_dtype="auto"`:
  - **Embedder (EmbeddingGemma 300M)**: bf16. FP16 is *incorrect* per the official model card.
  - **Reranker (bge-reranker-v2-m3)**: bf16. FP16 also works but bf16 is the stack default.
  - **VLM (Qwen2.5/3-VL)**: bf16. The VLM weight quantisation knob (see §4) controls VRAM independently.
- **TF32**: enabled globally at bootstrap via `torch.set_float32_matmul_precision("high")` and `torch.backends.cudnn.benchmark = True`. Memex has no FP32-precision-critical paths.

### 3. Attention implementation

- **Flash-Attention 2** is the default for the VLM (and any other transformers model that supports it). Pass `attn_implementation="flash_attention_2"` explicitly on `from_pretrained` so the dispatch isn't implicit.
- **Flash-Attention 3 is forbidden on the reference rig.** FA3 requires shared-memory budgets that Ada Lovelace doesn't have (per [vLLM #12429](https://github.com/vllm-project/vllm/issues/12429)). Do not set `VLLM_FLASH_ATTN_VERSION=3` or pass `attn_implementation="flash_attention_3"`. Revisit only if a future reference rig (Hopper, Blackwell) becomes a first-class target.
- `[parse]` extra installs `flash-attn>=2.6 ; sys_platform == 'linux'`. SDPA is the documented fallback on macOS/Windows.

### 4. VLM class and quantization

- **Model class**: `AutoModelForImageTextToText` (transformers 4.57+). `AutoModelForCausalLM` does not correctly expose the vision tower; `Qwen2_5_VLForConditionalGeneration` and `Qwen3VLForConditionalGeneration` are explicit alternatives but `AutoModelForImageTextToText` is the 2026 idiom.
- **`trust_remote_code` is dropped for the VLM, embedder, reranker, and orchestrator** — Qwen2.5-VL and Qwen3-VL landed in transformers proper in 4.49; the other models load via in-tree classes. **Amendments 2026-05-23 (P3.3-b + P3.3-c) — chart-OCR carve-out**: the prohibition is relaxed for the **chart-OCR slot only**, and only for explicit-opt-in models with auditable modeling code. Two backends are explicitly allowed under this carve-out: (a) **OneChart** (`kppkkp/OneChart`, P3.3-b) — 0.3B params, Apache 2.0, custom Vary-derived architecture; (b) **NVIDIA Nemotron-Parse-v1.2** (`nvidia/NVIDIA-Nemotron-Parse-v1.2`, P3.3-c) — 885M params, NVIDIA Nemotron Open License, custom NemotronParseForConditionalGeneration class. Common mitigations: (1) loaded only on explicit opt-in via `MEMEX_MODELS__CHART_OCR=<model-id>`; (2) modeling code is human-auditable (a few hundred LoC each); (3) pinned to a specific HF revision via the loader's `revision=` kwarg so an upstream repo edit can't change behavior without a code change here; (4) **[corrected 2026-05-25]** the chart-OCR model loads in the **orchestrator process** via `ModelRegistry` (so it can reuse the GPU while vLLM is paused for parse — see `parse/chart_ocr_backend.py`), NOT the seccomp'd CPU-only Docling/PyMuPDF subprocess workers. So its `trust_remote_code` runs **in-process, UNSANDBOXED** — process isolation is therefore *not* a mitigation here (an earlier draft incorrectly claimed it "runs under seccomp"). The load-bearing mitigations are (1) explicit opt-in, (2) the human code audit, (3) the pinned revision, plus the air-gap (no network egress after the one-time weights+code fetch); (5) backend-specific output gates (OneChart's `reliable_check` token, Nemotron-Parse's stop-token bounded decoder) prevent runaway emission. The exception does **not** extend to any other model slot. Additional chart-OCR backends must each justify their own amendment with comparable mitigations.
- **Default quantization**: **AWQ-Int4** for the VLM. BF16 Qwen2.5-VL-7B is ~15 GB raw; BF16 Qwen3-VL-8B is ~18 GB; both OOM on a 12 GB card. AWQ-Int4 fits ~7–8 GB peak (weights + ViT + activations) and the quality regression on document tasks is negligible (Qwen2.5-VL-7B-Instruct-AWQ shows -0.3 DocVQA pts per the official card).
- **`ModelSettings` gains `vlm_quantization: Literal["awq_int4", "bf16"] = "awq_int4"`** mirroring the existing `orchestrator_quantization`. Choosing `bf16` requires a non-default-rig VRAM budget and is the user's affirmation.
- **`device_map`**: deterministic single-GPU placement (`{"": "cuda:0"}`), not `"auto"`. The accelerate auto-splitter silently CPU-offloads under VRAM pressure; ADR-0001 prefers fail-fast.

### 5. CUDA bootstrap discipline

Every CLI / daemon entry point calls `cli/bootstrap.py:bootstrap()`, which is extended to:

1. Set `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"` before any torch import where possible (i.e. earliest in the function). `expandable_segments:True` is the documented fix for VRAM fragmentation under VLM swap-in/swap-out (per [PyTorch #165419](https://github.com/pytorch/pytorch/issues/165419)).
2. Import torch, call `torch.set_float32_matmul_precision("high")` and `torch.backends.cudnn.benchmark = True`.
3. Assert `torch.cuda.is_available()`; raise `InsufficientVRAMError` with a clear remediation message if not.
4. Call `torch.cuda.set_per_process_memory_fraction(settings.hardware.gpu_memory_fraction)` so the dead field is wired.

`MemexSettings` gains a `@model_validator` that consults a static GB table (orchestrator 5.0 + embedder 0.6 + reranker 0.6 + VLM AWQ-Int4 5.0 + overhead 2.0 + chart-OCR 2.5 when opt-in) and asserts the fit at startup before any model load is attempted. The VLM and chart-OCR are gated on `disable_vlm` / `disable_chart_ocr` so the budget check only counts the models that will actually load; the live table lives in `cli/bootstrap.py::_VRAM_GB`.

## Consequences

### Positive

- One canonical pyproject configuration resolves the right torch wheel on every supported platform.
- Embedder no longer produces silently-corrupt embeddings on FP16-incompatible Gemma.
- VLM actually loads on the reference rig instead of OOM-ing on first use.
- The `gpu_memory_fraction` setting that ADR-0001 promised becomes real.
- A future contributor can read ADR-0006 and know exactly which dtype + class + flag combination to use without re-deriving from scratch.

### Negative / Trade-offs

- Pinning to `cu128` excludes users on older NVIDIA drivers (< R570). We document the floor in the README and accept this.
- `flash-attn` is a heavy build-from-source dep on first install. We pin `>=2.6` and document the prebuilt wheel install path.
- AWQ-Int4 VLM quality is empirically very close to BF16 but not identical. The eval corpus (when assembled) will quantify; until then we accept the same risk we accept for the orchestrator (also Q4-quantised).
- The dtype discipline closes off PyTorch's `torch_dtype="auto"` convenience and requires an explicit choice at every callsite. We accept this as the price of GUIDELINES.md's "no implicit defaults" rule.

### Neutral

- The TF32 / cudnn-benchmark flags are safe for Memex's stable batch shapes; quality unchanged.
- Dropping `trust_remote_code=True` does not affect inference behaviour but does foreclose the path where a community Qwen-VL build ships custom modeling code. Re-enable on a per-model basis if needed.

## Alternatives in Detail

### Use `torch_dtype="auto"` instead of pinning bf16

Convenient — reads the dtype from each model's `config.json`. Rejected because the implicit defaults are uneven across the stack: Gemma defaults to FP16 (the wrong choice for activations), Qwen-VL defaults to BF16 (right), bge-reranker has no preference. Pinning bf16 globally is one rule that's right for every Memex model on Ada.

### Use FP16 everywhere instead of BF16

Smaller memory in some kernels, but the EmbeddingGemma activation-overflow issue is disqualifying. Mixing FP16 for some models and BF16 for others is the worst of both worlds (more dispatch rules to remember, no headroom). Rejected.

### Default VLM to a smaller model (Qwen3-VL-4B)

A real option. Qwen3-VL-4B fits BF16 on 12 GB with room to spare and avoids the AWQ build dependency. We chose AWQ-Int4 for the VLM because the larger model's OCR quality is materially better for the parser fallback workload that motivates having a VLM at all. **Current default**: `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` (the 7B variant is what's in production today; Qwen3-VL-8B is the eval-gated successor under P2.3 — see ADR-0001's revisit candidates). Revisit if AWQ becomes a maintenance burden or if a Qwen3-VL-4B Int4 build matches the larger model's quality.

### Route VLM through a second vLLM process

Would unify the inference layer under ADR-0001. Rejected: vLLM's published Qwen-VL recipes assume ≥4 GPUs and don't include a single-12-GB recipe; a second vLLM process adds another ~600 MB CUDA-context tax on top of a card already running the orchestrator's vLLM. Stay in-process via `transformers` under the existing `ModelRegistry`.

### Skip the wheel-index declaration and trust users to install torch correctly

Rejected. The default PyPI wheel for `torch>=2.7` on Linux is `cu126`; users running `uv sync --extra models` on a fresh box get a near-CUDA but mismatched-version wheel. Making this work requires either documentation users won't read or a CI-style pre-flight check we don't ship. Pin once in `pyproject.toml`.

## Revisit When

- A reference rig with FA3-capable shared-memory budget (Hopper / Blackwell consumer card) becomes a first-class target
- PyTorch drops `cu128` wheels (no current signal) — pivot to whatever the supported pairing is
- Qwen3-VL-8B official FP8 build becomes `transformers`-loadable (currently vLLM/SGLang only)
- An open-weight VLM in the 4–8B range with materially better OCR + a transformers loader appears

## References

- ADR-0001: vLLM as sole inference engine
- ADR-0005: RyuGraph replaces Kuzu
- Memex CUDA audit memory note (2026-05-20)
- [PyTorch / uv integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [EmbeddingGemma 300M model card](https://huggingface.co/google/embeddinggemma-300m) — the "no FP16" warning
- [Qwen3-VL-8B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [vLLM #12429: FA3 unavailable on sm_89](https://github.com/vllm-project/vllm/issues/12429)
- [PyTorch #165419: expandable_segments details](https://github.com/pytorch/pytorch/issues/165419)
- [transformers `AutoModelForImageTextToText`](https://huggingface.co/docs/transformers/en/model_doc/auto)
