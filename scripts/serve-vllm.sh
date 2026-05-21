#!/usr/bin/env bash
# scripts/serve-vllm.sh — Launch the vLLM orchestrator with Memex's flag set.
#
# Memex assumes a single consumer GPU (reference: RTX 4070 12 GB, Ada
# Lovelace, sm_89). The flags below are chosen for that profile and
# discussed in ADR-0001 (vLLM as sole inference engine), ADR-0006
# (CUDA dispatch + dtype), and the CUDA audit memo.
#
# Override the model + endpoint via environment:
#   MEMEX_VLLM_MODEL          (default: Qwen/Qwen3-8B-AWQ — Qwen3 dropped
#                              the "Instruct" suffix; the base model IS
#                              the instruct model.)
#   MEMEX_VLLM_HOST           (default: 127.0.0.1)
#   MEMEX_VLLM_PORT           (default: 8000)
#   MEMEX_VLLM_QUANTIZATION   (default: awq)
#   MEMEX_VLLM_MAX_MODEL_LEN  (default: 8192)
#   MEMEX_VLLM_GPU_FRACTION   (default: 0.80 — leaves headroom for Memex's
#                              embedder + reranker + VLM in the same process)

set -euo pipefail

MODEL="${MEMEX_VLLM_MODEL:-Qwen/Qwen3-8B-AWQ}"
HOST="${MEMEX_VLLM_HOST:-127.0.0.1}"
PORT="${MEMEX_VLLM_PORT:-8000}"
# Use single-dash `${VAR-default}` so an explicitly-empty override
# (e.g. MEMEX_VLLM_QUANTIZATION='') means "no --quantization flag"
# for unquantized models like Qwen3-0.6B.
QUANTIZATION="${MEMEX_VLLM_QUANTIZATION-awq}"
MAX_MODEL_LEN="${MEMEX_VLLM_MAX_MODEL_LEN:-8192}"
GPU_FRACTION="${MEMEX_VLLM_GPU_FRACTION:-0.80}"

# Explicit single-device — skips the multi-GPU discovery codepath at
# startup (~1–2 s faster cold start on a single-card rig).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Do NOT set VLLM_FLASH_ATTN_VERSION=3 — FA3 is unavailable on Ada
# Lovelace per vLLM issue #12429. Default auto-select lands on FA2.
unset VLLM_FLASH_ATTN_VERSION || true

# VLLM_USE_V1 was removed in Nov 2025; V1 is unconditional. Any old
# .env that exports it is harmless but dead.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Run through `uv` so the project's venv (with the pinned cu129 torch
# and vllm>=0.21,<0.22) is used regardless of the caller's shell state.
# Falls back to a bare `vllm serve` if uv isn't on PATH so a manually
# pre-activated venv still works.
if command -v uv >/dev/null 2>&1; then
    cd "$PROJECT_ROOT"
    QUANT_ARGS=()
    if [[ -n "$QUANTIZATION" ]]; then
        QUANT_ARGS=(--quantization "$QUANTIZATION")
    fi
    exec uv run --extra serve vllm serve "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        "${QUANT_ARGS[@]}" \
        --gpu-memory-utilization "$GPU_FRACTION" \
        --max-num-seqs 8 \
        --max-model-len "$MAX_MODEL_LEN" \
        --kv-cache-dtype fp8_e5m2 \
        --enable-prefix-caching \
        ${MEMEX_VLLM_EAGER:+--enforce-eager}
else
    QUANT_ARGS=()
    if [[ -n "$QUANTIZATION" ]]; then
        QUANT_ARGS=(--quantization "$QUANTIZATION")
    fi
    exec vllm serve "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        "${QUANT_ARGS[@]}" \
        --gpu-memory-utilization "$GPU_FRACTION" \
        --max-num-seqs 8 \
        --max-model-len "$MAX_MODEL_LEN" \
        --kv-cache-dtype fp8_e5m2 \
        --enable-prefix-caching \
        ${MEMEX_VLLM_EAGER:+--enforce-eager}
fi
