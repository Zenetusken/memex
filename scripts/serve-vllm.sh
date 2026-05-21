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
#                              the "Instruct" suffix; the base IS the
#                              instruct model.)
#   MEMEX_VLLM_HOST           (default: 127.0.0.1)
#   MEMEX_VLLM_PORT           (default: 8000)
#   MEMEX_VLLM_QUANTIZATION   (default: awq_marlin — vLLM logs that AWQ
#                              models can run with the Marlin kernel for
#                              ~1.5-2× faster decode. Same weights, same
#                              quality, just a better kernel. Set to
#                              empty for unquantized models, or to `awq`
#                              for the legacy kernel.)
#   MEMEX_VLLM_MAX_MODEL_LEN  (default: 6144 — sized to fit the agent's
#                              answer prompt at top_k=5 with chunks
#                              truncated to 1800 chars. The earlier
#                              4096 ceiling clipped at ~3070 input
#                              tokens with the v2 prompts, leaving
#                              insufficient room for the 1024-token
#                              output. The +2048 token bump costs
#                              ~1 GB of KV-cache reservation with
#                              fp8_e5m2 KV; the 12 GB reference rig
#                              fits comfortably at gpu_memory_
#                              utilization=0.72.)
#   MEMEX_VLLM_GPU_FRACTION   (default: 0.72 — measured from a vLLM
#                              0.21 cold start on RTX 4070 12 GB with
#                              Qwen3-8B-AWQ (5.7 GB weights) + the new
#                              CUDA-graph memory profiler (≈ 0.45 GB);
#                              0.72 leaves ~2.5 GB for KV cache and
#                              ~3.4 GB headroom for the in-process
#                              embedder + reranker. 0.65 left only
#                              0.09 GB for KV cache and vLLM refused
#                              to start; 0.80 OOMed the reranker.)

set -euo pipefail

MODEL="${MEMEX_VLLM_MODEL:-Qwen/Qwen3-8B-AWQ}"
HOST="${MEMEX_VLLM_HOST:-127.0.0.1}"
PORT="${MEMEX_VLLM_PORT:-8000}"
# Use single-dash `${VAR-default}` so an explicitly-empty override
# (e.g. MEMEX_VLLM_QUANTIZATION='') means "no --quantization flag"
# for unquantized models like Qwen3-0.6B.
QUANTIZATION="${MEMEX_VLLM_QUANTIZATION-awq_marlin}"
MAX_MODEL_LEN="${MEMEX_VLLM_MAX_MODEL_LEN:-6144}"
GPU_FRACTION="${MEMEX_VLLM_GPU_FRACTION:-0.72}"

# Explicit single-device — skips the multi-GPU discovery codepath at
# startup (~1–2 s faster cold start on a single-card rig).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Do NOT set VLLM_FLASH_ATTN_VERSION=3 — FA3 is unavailable on Ada
# Lovelace per vLLM issue #12429. Default auto-select lands on FA2.
unset VLLM_FLASH_ATTN_VERSION || true

# VLLM_USE_V1 was removed in Nov 2025; V1 is unconditional. Any old
# .env that exports it is harmless but dead.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── systemd-notify readiness gate (FU3.2.1) ─────────────────────────
# When this script runs under a Type=notify unit, $NOTIFY_SOCKET is
# set. Spawn a backgrounded poller that calls `systemd-notify --ready`
# once vLLM's OpenAI endpoint is genuinely reachable. systemd then
# only marks the unit `active` after that, so `After=memex-vllm.service`
# on downstream units (web, MCP, watcher) becomes a real readiness
# gate — not just "the process forked" ordering. No-op when
# $NOTIFY_SOCKET is unset, so the manual / `memex daemon start` /
# Pattern B / Pattern C flows are unaffected.
if [ -n "${NOTIFY_SOCKET:-}" ]; then
    if command -v systemd-notify >/dev/null 2>&1; then
        (
            # Brief grace so we don't race the `exec` below.
            sleep 1
            # Poll until vLLM answers /v1/models. systemd's
            # TimeoutStartSec=300 (set on the unit) caps the wait —
            # no second timeout here so the budgets don't compete.
            until curl -sf "http://${HOST}:${PORT}/v1/models" \
                       >/dev/null 2>&1; do
                sleep 1
            done
            systemd-notify --ready \
                --status="vLLM serving at ${HOST}:${PORT}"
        ) &
    else
        echo "serve-vllm.sh: NOTIFY_SOCKET set but systemd-notify" \
             "binary not found; readiness gating disabled." >&2
    fi
fi

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
