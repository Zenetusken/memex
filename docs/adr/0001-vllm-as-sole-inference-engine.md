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

- **Structured-output API**: vLLM's `extra_body={"guided_json": ...}` is deprecated in favour of OpenAI-standard `response_format={"type": "json_schema", "json_schema": {"name": "...", "schema": ...}}`. The new form is portable across vLLM / SGLang / llama-server. **✅ Adopted in `models/client.py::complete_structured`** (no eval required; wire-format change).
- **VLM upgrade — Qwen3-VL-8B-Instruct**: supersedes Qwen2.5-VL 7B with +6–14 OCR points and native 256 K context. AWQ-Int4 (per ADR-0006) is the target quantisation. **⏸️ P2.3 — eval-gated**; awaiting a scan-style corpus to exercise the VLM page-transcription path. Default stays Qwen2.5-VL-7B-AWQ.
- **Orchestrator alternative — Granite 4.1-8B-FP8** (Apache 2.0, native OpenAI-style tool calling). **❌ P2.2 attempted 2026-05-21; vLLM 0.21 hangs at FA2 init** for Granite 4.1's hybrid Mamba2+attention backend, in both cudagraph and `--enforce-eager` modes. Resolution: vLLM 0.22+ with hybrid-arch support, OR Granite GGUF variant via `--quantization gguf`, OR fall back to Granite 3.x.
- **Reranker alternative — Qwen3-Reranker-0.6B**: ~8 MMTEB-R points above bge-reranker-v2-m3 at similar VRAM but autoregressive. **❌ P2.1 resolved 2026-05-21**: 3 head-to-head runs, **cross_encoder bge-reranker-v2-m3 wins clearly** (median ANS=4 vs Qwen3-Reranker's 0); Qwen3-Reranker ranks generic-CUDA chunks above the literal-answer chunk for the slide-deck corpus. Default reranker stays `cross_encoder`; qwen3 backend kept as `MEMEX_MODELS__RERANKER_BACKEND=qwen3` opt-in.
- **Parser fast-path — PyMuPDF4LLM pre-filter**: **✅ P1.1 shipped 2026-05-21**: tiered classifier (`parse/pipeline.py::_classify`) with mixed-content OCR routing; canonical CUDA deck parses 30s vs 96s.
- **Smaller-orchestrator tier — Qwen3-4B-AWQ**: **✅ P4.2 shipped 2026-05-21** at `gpu_memory_utilization=0.50` as the 8 GB tier profile; HARD GATES (refusal_cf=1.0, hallucinations=0) preserved. Full env-var matrix in `docs/deploy/hardware-tiers.md`.

## References

- vLLM: https://github.com/vllm-project/vllm
- xgrammar: https://github.com/mlc-ai/xgrammar
- Memex vision: `../VISION.md`
- Memex developer guidelines: `../GUIDELINES.md`
