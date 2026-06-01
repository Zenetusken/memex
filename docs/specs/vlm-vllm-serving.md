# Spec: VLM served via a parse-time vLLM process

**Status:** shipped 2026-05-26 (P2.3). **Code:** `parse/vlm_backend.py` (`_serve_vlm_vllm`, `_reap_vlm_vllm`, `_convert_one_via_vllm`, `_vllm_transcribe`, the `convert_pages` backend branch), `core/config.py` (`VLMServeSettings`, `ModelSettings.vlm_serving`). **See also:** ADR-0006 §4 (the amendment that reverses "Route VLM through a second vLLM process"), `vlm-transcription-cache.md`, `office-pdf-conversion.md`.

> **Update (2026-06-01, ADR-0015):** the grounded *orchestrator* swapped to the unified `cyankiwi/Qwen3.5-4B-AWQ-4bit`. Unifying THIS doc-VLM role onto the same 4B was **attempted + reverted** — the dedicated `Qwen3-VL-8B` is stronger at hard diagram/scan transcription (the 4B-VL regressed `cr350-multidoc`/`slide-decks`/`handwritten`), and a VLM re-parse re-chunks docs, which can tip a borderline counterfactual. So this lifecycle still serves **`models.vlm` = `cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit`** (unchanged); the orchestrator and the doc-VLM are *different* models. See ADR-0015 §"VLM-role unification: attempted, reverted".

## Problem

The VLM upgrade Qwen2.5-VL-7B → **Qwen3-VL-8B** fixes the state-machine / flow-diagram *flattening* the 7B model exhibited (it read a directed diagram's boxes top-to-bottom as a flat list, losing the transitions). But Qwen3-VL's only int4 community build is **compressed-tensors `pack-quantized`** (`cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit`), and **transformers cannot run it in-process on a 12 GB card**:

- `compressed_tensors` 0.15 has no in-process int4 *kernel* for `pack-quantized`, so a `ct_decompress_hook` **decompresses int4 → dense (~16 GB) at first forward** → CUDA-OOM (it loads at ~7.5 GB, dies mid-decompress). `run_compressed=True` does not suppress the hook; `device_map="auto"` CPU-offload hits a `Tensor.item() cannot be called on meta tensors` bug inside the decompress hook.
- (Contrast: the legacy `Qwen2.5-VL-7B-Instruct-AWQ` is an **AutoAWQ** build, which transformers *does* run int4 in-process at ~5 GB. The blocker is the quant *format*, not the model.)

**vLLM runs the identical build via its Marlin int4 kernel at ~7.4 GB**, `trust_remote_code=False` (Qwen3-VL is native in vLLM ≥0.11 and transformers ≥4.57). So the VLM is served by a **short-lived vLLM process during parse**, not loaded in-process. This reverses ADR-0006's original rejection of "VLM via vLLM" (which assumed no single-12-GB recipe existed) — the recipe below is validated.

## Design

```
parse  ── pause_vllm_for_gpu() ──┐  (orchestrator vLLM down → GPU free)
                                 │
   _route_and_escalate → convert_pages(vllm branch):
       async with _serve_vlm_vllm(model_id) as base_url:   # start VLM vLLM on :8001
           for page in misses: _convert_one_via_vllm(...)  # OpenAI multimodal API
       # VLM vLLM torn down HERE (before convert_pages returns)
                                 │
   chart-OCR pass (in-process Nemotron, ~3 GB)             # never co-resident with the VLM vLLM
                                 │
   ── restart orchestrator vLLM ─┘
```

Key property: the VLM vLLM lives **only inside the parse window** and is torn down **before** the in-process chart-OCR pass. The two can't co-reside (7.4 + ~3 GB > 12 GB), so it's a *swap*, not concurrency — which is exactly why ADR-0006's original co-residence objection doesn't apply. The orchestrator vLLM is already paused by `pause_vllm_for_gpu`, so the VLM vLLM gets the freed GPU.

`convert_pages` branches on `settings.models.vlm_serving`:
- `"vllm"` (default) → the lifecycle above.
- `"transformers"` → the legacy in-process registry path (`registry.use("vlm")` + the FA2/sdpa + AutoAWQ `PytorchGELUTanh` shims), for the AutoAWQ Qwen2.5-VL build.

The cache (`vlm_cache.py`), best-of-N keep-longest, and the escalation logic are **backend-agnostic** — only the per-page transcription call differs. A full cache-hit parse never starts the VLM vLLM (the `if not misses: return` guard fires first).

## Recipe — `VLMServeSettings` (validated on the 12 GB RTX 4070)

| Flag | Value | Why |
|---|---|---|
| `gpu_memory_utilization` | **0.80** | Must absorb the desktop's **peak** graphics load (Xorg + compositor + the user's apps), not its idle. 0.89 worked on an idle desktop but **intermittently failed startup** (`Free memory < desired GPU memory utilization`) mid-bulk-reingest once a Zoom call + browser raised GPU use. It's a one-time startup gate, so the margin must cover the peak. |
| `max_model_len` | **3072** | A page transcription is the image (~1280 visual tokens, `max_pixels`-capped) + <800 output. 3072 leaves KV headroom even at the lower util. |
| `mm-processor-kwargs` | `max_pixels=1003520, min_pixels=200704` | Without the cap, the vLLM **vision encoder cache reserves for the model's MAX image** (~16384 tokens) → KV starves to ~0.25 GB → vLLM **refuses to start**. 1280·28·28 mirrors the rasteriser cap in `_render_page_to_image`. |
| `--enforce-eager` | on | Skips CUDA-graph capture — faster startup, frees memory. |
| `--kv-cache-dtype` | `auto` | (compressed-tensors weights; not `fp8_e5m2`.) |
| `--max-num-seqs` / `--max-num-batched-tokens` | 1 / `max_model_len` | One page at a time. |
| `port` | 8001 (≠ orchestrator's 8000) | The parse pause's reachability check targets the **orchestrator** base_url; a distinct port means a nested no-op pause never targets the VLM vLLM. |

Served model id = `settings.models.vlm`; the OpenAI client calls `model=<that id>`.

## Lifecycle hardening (load-bearing)

`_serve_vlm_vllm` (async context manager) and `_reap_vlm_vllm`:

1. **Capture the process-group gid AT SPAWN** (`os.getpgid(proc.pid)` while the launcher is alive). A *failed* startup exits the launcher, after which a teardown-time `os.getpgid(proc.pid)` raises `ProcessLookupError` — the original bug, which **orphaned a GPU-holding EngineCore** (vLLM's compute child) and **cascaded** every subsequent doc's startup into `Free memory < desired`. Teardown `killpg`s the *captured* gid.
2. **Reap by group-emptiness:** after SIGTERM (then SIGKILL on a 30 s timeout), poll `os.killpg(gid, 0)` until it raises `ProcessLookupError` — the **reliable signal the EngineCore released its VRAM**. The port going quiet does NOT guarantee VRAM release.
3. **Startup retry (once):** ride out a transient tight-card `Free memory < desired` after a 5 s GPU settle. (Doesn't help against *sustained* desktop pressure — that's what util 0.80 is for.)
4. **Capture the subprocess stdout/stderr** to a temp log; surface its tail in the `VLMUnavailable` error/log on failure. (The old `DEVNULL` hid vLLM's own error — this capture is what diagnosed the desktop-graphics root cause.)

## VRAM accounting

- **Answering** (the steady state): orchestrator vLLM + EmbeddingGemma + bge-reranker co-resident. The vLLM-served VLM is **not** counted here — `bootstrap._verify_vram_fit` adds the in-process VLM estimate ONLY when `vlm_serving == "transformers"`.
- **Parse** (VLM phase): orchestrator paused → VLM vLLM ~7.4 GB weights + KV alone on the GPU (+ the desktop's ~1–2 GB). Then VLM vLLM down → chart-OCR ~3 GB in-process.

## Migration mechanics

- A VLM model change is a natural **cache miss** (`vlm_cache.py` keys include the model id) → every doc re-transcribes on next parse. Bust per-doc with `memex parse --refresh-vlm <doc>`.
- A VLM swap changes parsed Markdown → chunk-id churn → forces a re-embed of VLM-affected docs on `reindex`. **Back up `vault/documents/*.md` first** (the vault isn't git-tracked).
- Re-baseline scope: all answer-eval corpora under `tests/eval-data/`; HARD gates `refusal_cf=1.0` + 0 hallucinations, **measured multi-run** on the VLM-dependent corpora (borderline-cf non-determinism). The Qwen3-VL re-baseline PASSED — all 11 corpora held, VLM corpora deterministic across N=3.

## Testing

`tests/unit/test_vlm_cache.py` — the vllm-path test fakes `_serve_vlm_vllm` (a no-launch CM) + `_convert_one_via_vllm` and asserts: misses transcribe + cache, the server starts **once** for the batch, and a full cache-hit starts the server **zero** times (no wasted ~30 s boot). The transformers-path tests pin `vlm_serving="transformers"`.

## Open follow-ups

- The per-doc VLM-vLLM startup (~30 s) is the bulk-reingest cost; a future optimization could hoist one VLM-vLLM across a whole bulk run — but that reintroduces the chart-OCR co-residence problem (deferred).
- An official Qwen3-VL AWQ (or a transformers-loadable build) would allow dropping the second-process dance — see ADR-0006 "Revisit When".
