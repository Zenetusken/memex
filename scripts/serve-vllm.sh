#!/usr/bin/env bash
# scripts/serve-vllm.sh — Launch the vLLM orchestrator with Memex's flag set.
#
# Memex assumes a single consumer GPU (reference: RTX 4070 12 GB, Ada
# Lovelace, sm_89). The flags below are chosen for that profile and
# discussed in ADR-0001 (vLLM as sole inference engine), ADR-0006
# (CUDA dispatch + dtype), and the CUDA audit memo.
#
# Override the model + endpoint via environment:
#   MEMEX_VLLM_MODEL          (default: Qwen/Qwen3-8B-Instruct-AWQ)
#   MEMEX_VLLM_HOST           (default: 127.0.0.1)
#   MEMEX_VLLM_PORT           (default: 8000)
#   MEMEX_VLLM_QUANTIZATION   (default: awq)
#   MEMEX_VLLM_MAX_MODEL_LEN  (default: 8192)
#   MEMEX_VLLM_GPU_FRACTION   (default: 0.80 — leaves headroom for Memex's
#                              embedder + reranker + VLM in the same process)

set -euo pipefail

MODEL="${MEMEX_VLLM_MODEL:-Qwen/Qwen3-8B-Instruct-AWQ}"
HOST="${MEMEX_VLLM_HOST:-127.0.0.1}"
PORT="${MEMEX_VLLM_PORT:-8000}"
QUANTIZATION="${MEMEX_VLLM_QUANTIZATION:-awq}"
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

exec vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --quantization "$QUANTIZATION" \
    --gpu-memory-utilization "$GPU_FRACTION" \
    --max-num-seqs 8 \
    --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-dtype fp8_e5m2 \
    --enable-prefix-caching \
    --swap-space 4
