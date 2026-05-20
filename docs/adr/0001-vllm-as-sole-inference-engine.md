# ADR-0001: vLLM as the Sole Inference Engine

- **Status**: Accepted
- **Date**: 2026-05-19
- **Deciders**: Memex core team
- **Tags**: stack, inference, models

## Context

Memex runs a fully local agentic pipeline on consumer hardware — the reference target is an RTX 4070 with 12 GB of VRAM. The answering pipeline makes many sequential model calls per user query: retrieval, reranking, sufficiency assessment, draft generation, grounding verification, and (when needed) regeneration. The inference layer's throughput, structured-output reliability, and steady-state memory behavior dominate user-perceived latency and answer quality.

Multiple mature inference engines exist with materially different trade-offs. Supporting more than one is tempting — give the user a choice, support more hardware — but multiplies the surface area we must test, observe, and document, and undermines the vision principle "Small Models, Used Well," which depends on tight infrastructure discipline.

This ADR commits to a single engine for v1.

## Decision Drivers

- Sustained throughput under agentic load (10+ sequential model calls per query)
- Reliable structured outputs via grammar-constrained decoding
- VRAM efficiency with co-resident embedder, reranker, and orchestrator on a 12 GB card
- OpenAI-compatible API so client code remains portable
- Production maturity — we should not be the first user to hit production-grade bugs
- Active development — open-weight inference is a fast-moving field

## Considered Options

1. **vLLM** — production-grade serving, PagedAttention, mature guided decoding
2. **Ollama** — developer-friendly single binary, model swapping, weaker structured outputs
3. **llama.cpp** (direct, via `llama-cpp-python` or `llama-server`) — lowest overhead, less mature server story
4. **Text Generation Inference (TGI)** — HuggingFace's stack, similar profile to vLLM but smaller community
5. **MLX-LM** — Apple Silicon only; eliminated by reference hardware target

## Decision

**vLLM** is the sole inference engine for v1.

It is the only option that simultaneously delivers production throughput, mature guided decoding, an OpenAI-compatible API, and active community development on consumer NVIDIA hardware. The cost is a heavier dev environment requirement (CUDA toolkit, NVIDIA driver) and a long cold start that forces a daemon model for the CLI.

## Consequences

### Positive

- PagedAttention reduces KV cache fragmentation, letting an 8B Q4 orchestrator coexist with the embedder and reranker in 12 GB VRAM with usable headroom
- xgrammar / outlines backends give us reliable JSON-schema-constrained outputs, eliminating regex parsing of free-text model responses
- The OpenAI-compatible endpoint means any future client (Python, TypeScript, MCP) speaks the protocol without Memex-specific glue
- Continuous batching gives us a free path to multi-user mode later (lab teams, classrooms) without rearchitecting
- Sustained tokens/sec on agentic loops is materially higher than Ollama on the same hardware

### Negative / Trade-offs

- **Dev environment requires CUDA toolkit and a compatible driver from day one.** No "just install and run" experience on machines without NVIDIA GPUs. This is a real hiring/contributor consideration.
- **Cold start is ~30 seconds** to load 8B Q4 + embedder + reranker. Acceptable for a long-running daemon, prohibitive for a fresh subprocess per CLI invocation. This forces the daemon model documented in the dev guidelines.
- **The CLI must spawn or attach to a running vLLM daemon.** Adds operational complexity: process supervision, port allocation, graceful shutdown to release VRAM, `memex daemon stop` for surrendering the GPU.
- **One model per server process.** Swapping the orchestrator for a larger variant means restarting the server. Fine for a tuned default; not for ad-hoc experimentation. Multi-LoRA serving partially mitigates this but adds complexity we defer.
- **No first-class CPU fallback.** Users without a CUDA GPU cannot run Memex. We accept this as the honest cost of the local-agentic vision.

### Neutral

- vLLM releases move quickly. We pin to a known-good version in `pyproject.toml` and bump deliberately, not automatically. Every bump runs the full eval suite and benchmark.

## Alternatives in Detail

### Ollama

The most attractive alternative. Single-binary install, genuinely good developer experience, broad model coverage, easy swapping. Two disqualifying issues for v1:

- Structured-output enforcement is best-effort (`format=json`) rather than grammar-constrained. We cannot tolerate occasional malformed JSON in tool use; vLLM's xgrammar integration guarantees grammar compliance at the decoder.
- Sustained throughput on agentic loops is materially lower — no PagedAttention, less efficient KV cache management, no continuous batching.

We may ship an Ollama-compat path in the future as a second-class option for low-friction trials, with explicit caveats about structured-output reliability. It is not the production path.

### llama.cpp (direct)

The lowest-overhead option. Runs on more hardware (CPU, Metal, ROCm, CUDA), smallest footprint, simplest dependency tree. Disqualifying for v1:

- `llama-server` is less mature than vLLM's serving stack — fewer production deployments, less hardening
- Guided decoding via GBNF grammar is more limited and slower than vLLM's xgrammar
- We would spend time on inference-layer infrastructure that vLLM already solves

We revisit if llama.cpp's server reaches parity on guided decoding throughput, or if we add Apple Silicon as a first-class target.

### TGI (HuggingFace)

Genuinely close to vLLM on capability. Smaller community around small-model serving on consumer hardware — HuggingFace positions TGI more for hosted larger models. Not enough advantage to swim against ecosystem gravity.

### MLX-LM

Apple Silicon only. Eliminated by reference hardware target (RTX 4070). Worth revisiting only if/when Apple Silicon becomes a first-class deployment target, which would require its own ADR.

## Revisit When

- vLLM ever drops support for consumer GPUs or 12 GB VRAM configurations
- llama.cpp's server reaches parity on guided decoding throughput and stability
- We commit to Apple Silicon as a first-class deployment target
- The "must have CUDA to develop" tax becomes a hiring or contributor blocker we can measure
- A new entrant clearly dominates vLLM on the relevant metrics for 12+ months

### Candidates to evaluate when `tests/evals/` exists

The stack-currency audit on 2026-05-19 surfaced the following swap candidates. None is acted on until the eval harness can produce a quality delta, but they are tracked here so the candidate set isn't re-discovered each cycle.

- **Structured-output API**: vLLM's `extra_body={"guided_json": ...}` is deprecated in favour of OpenAI-standard `response_format={"type": "json_schema", "json_schema": {"name": "...", "schema": ...}}`. The new form is portable across vLLM / SGLang / llama-server. Adopt at first `complete_structured` implementation; no eval required (wire-format change only).
- **VLM upgrade**: Qwen3-VL-8B-Instruct supersedes Qwen2.5-VL 7B with +6–14 OCR points and native 256 K context. Q4_K_M is ~6.1 GB (vs 5.5 GB), still inside the swap budget. Action when Phase 2 wires the VLM.
- **Orchestrator alternatives**: Granite 4.1-8B (Apache 2.0, native OpenAI-style tool calling baked into the chat template, BFCL 68.3 — better suited to the LangGraph loop than Qwen3-8B-Instruct's prompt-template tool path) or Qwen3.5-9B (in-family successor). Run A/B against `tests/evals/` once the harness exists; commit only if the eval delta is positive.
- **Reranker alternative**: Qwen3-Reranker-0.6B beats bge-reranker-v2-m3 by ~8 MMTEB-R points at the same VRAM but is autoregressive (yes/no logit), so per-pair latency is higher. Gate on the < 15 s answer floor from GUIDELINES Part VI.
- **Parser fast-path**: PyMuPDF4LLM as a pre-filter for clean-text PDFs before Docling sees them. No licensing or architectural cost; cuts ingest time on the easy ~60% of inputs.

## References

- vLLM: https://github.com/vllm-project/vllm
- xgrammar: https://github.com/mlc-ai/xgrammar
- Memex vision: `../VISION.md`
- Memex developer guidelines: `../GUIDELINES.md`
