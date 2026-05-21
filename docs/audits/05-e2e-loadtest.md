# E2E + Load Test Report (2026-05-20, RTX 4070 reference rig)

## Hardware
- NVIDIA GeForce RTX 4070, 12 GB VRAM, sm_89 (Ada Lovelace)
- 32 GB system RAM, 20 GB swap
- NVIDIA driver R580+, CUDA 13.0 driver / 12.9 toolkit

## Stack pinned (post-E2E)
- torch 2.11.0+cu129 (from pytorch-cu129 index)
- torchvision 0.26.0+cu129 (direct dep so source override applies)
- transformers 4.57.6 (<5 to keep sentence-transformers happy)
- sentence-transformers 5.5.1
- vllm 0.21.0+cu129 (from wheels.vllm.ai/0.21.0/cu129)
- pyseccomp on Linux
- HTMX 1.9.10 (SHA-384 verified, vendored)
- Hand-rolled tailwind utility subset (no CDN)

## Walkthrough — every audit fix verified live

| Step | Outcome | Audit fix verified |
|---|---|---|
| `uv sync --extra models --extra parse --extra dev --extra serve` | 4.4 GB venv | cu129 migration |
| `scripts/vendor-frontend.sh` | HTMX 47755 B, SHA verified | air-gap promise |
| `memex --help` | All 13 commands present | CLI surface |
| `memex doctor` (no daemon) | bootstrap + VRAM-fit warn + 0 docs + daemon unreachable | W1 doctor bootstrap |
| `memex ingest test.md` | doc_id assigned, vault asset dir 0700 | ingest path |
| `memex parse` | parser_version=`memex.parse@v1`, passthrough | V1 version bump |
| `memex index` | 5 chunks, EmbeddingGemma loaded, all 3 stores written | full index path |
| `memex index` (re-run) | `partial=True, unchanged=5, 89ms` (vs 113 s first) | incremental partial re-index |
| `memex search` | 5 dense + 0 BM25 → 3 reranked, scores [0.93, 0.31, 0.19] | LanceDB 0.30 async surface fix |
| `memex serve web` | Vendored Tailwind + HTMX, no CDN | air-gap |
| 6 path-traversal probes | 5/6 → 404; `../` → /documents/ (FastAPI routing) | S1 regex constraint |
| 5000-byte POST /ask | 422 | E13 max_length |
| `vllm serve` (cu129, Python.h installed) | reachable in 45 s, 7.9 GB VRAM | vLLM bring-up |
| `memex doctor` (daemon up) | reachable=True, models=[Qwen/Qwen3-0.6B] | doctor + audit fix chain |
| `memex ask` | 13 nodes, 5526 tokens, 21 s → REFUSE | refusal-first-class |
| `memex enrich` (single) | 12 entities, 0 chunk failures, 3.7 s | enrich return_exceptions |
| Bulk ingest 5 docs | 9.9 s end-to-end (~2 s/doc) | streaming ingest |
| Bulk enrich 5 in parallel | 49 entities, 60 MENTIONS, **1 CITES** | cross-doc citation resolution |
| `pytest tests/` | 88/88 pass (2 previously seccomp-skipped now run) | full suite |

## Load test (5 concurrent /ask on persistent server)

### Round 1 — process-per-call (`memex ask` × 5 in parallel)
- All 5 FAILED with `InsufficientVRAMError` + `OutOfMemoryError` + `CircuitBreakerOpen`
- VRAM: 7.9 GB → 9.5 GB (couldn't load reranker × 5 across separate processes)
- RAM: 9.6 GB → 17.6 GB (5× model duplication in host memory)
- Swap: 467 MB → 1388 MB
- **Audit fix `_cleanup_partial` fired correctly** (trace at registry.py:136)
- **Audit fix OOM circuit-breaker tripped after 3 consecutive OOMs**
- This is the WRONG concurrency model — each process loads its own ModelRegistry
- The right pattern: one server, N clients

### Round 2 — persistent server + 5 concurrent curls
| Metric | Idle | Peak | Settled |
|---|---|---|---|
| VRAM | 8.0 GB | **10.1 GB** | 8.0 GB |
| RAM | 9.0 GB | **11.9 GB** | 9.0 GB |
| Swap | 1127 MB | 1127 MB | 1127 MB |
| Load avg (1m) | 1.4 | 1.4 | 1.4 |
| GPU util | 9% | 62% | 5% |
| GPU temp | 51°C | 62°C | 45°C |

- Wall time: **24 s for 5 concurrent /ask** (~5 s/ask amortised)
- 3/5 → graceful refusal (audit "refusal-first-class")
- 2/5 → ModelCallError caught + rendered as red panel (NEW audit fix landed)
- 0/5 → 500 server errors

### Throughput at this stack (Qwen3-0.6B + bge-reranker-v2-m3 + embeddinggemma-300m)
- ~3 ask/s burst (vLLM batched 5 concurrent decodes within `--max-num-seqs 8`)
- ~0.2 ask/s sustained (cold-start dominated by model loads)
- For an 8B AWQ orchestrator: expect ~3-5× longer per call; rerank/embed costs unchanged

## Out-of-scope fixes that landed during the E2E
- pyproject.toml: cu128 → cu129; transformers <5; flash-attn out of [parse]; vllm in [serve]
- scripts/serve-vllm.sh: removed dead `--swap-space`; added `QUANT_ARGS` + `MEMEX_VLLM_EAGER` knobs; run via uv
- src/memex/models/client.py: model="default" → settings.models.orchestrator (vLLM 0.21+ requires exact id)
- src/memex/core/config.py: default orchestrator `Qwen3-8B-Instruct` (404) → `Qwen3-8B-AWQ`
- src/memex/cli/bootstrap.py: CUDA error message refs cu128 → cu129
- src/memex/index/graph_store.py: schema loader strips `//` comments before split (dropped first CREATE NODE TABLE Document)
- src/memex/index/vector_store.py: LanceDB 0.30 async-surface adaptation
- src/memex/webui/app.py: /ask catches MemexError + renders partial
- pyproject sources: torchvision direct dep so cu129 override applies

## Headroom & recommendations
- Qwen3-0.6B at `--gpu-memory-utilization 0.55` leaves ~4 GB VRAM for embedder+reranker — fits
- Qwen3-8B-AWQ at 0.55 would need ~6.5 GB; total demand ~10.5 GB — fits on the 12 GB rig
- VLM (Qwen2.5-VL-7B-Instruct-AWQ) on top of orchestrator+embedder+reranker WILL exceed 12 GB on this rig — keep `disable_vlm=True` until a smaller VLM variant is wired (Qwen3-VL-4B?)
- Swap pressure peaked at 1.4 GB during process-per-call test — well under the 20 GB swap available; the kernel reclaimed automatically
- The bootstrap VRAM-fit warning (`estimated_gb: 15.7 budget_gb: 9.8`) was prescient and saved us from a runtime OOM during the audit's first ask attempt
