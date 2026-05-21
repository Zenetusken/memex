# 8B-AWQ Load Test (2026-05-20, RTX 4070 reference rig)

Follow-up to `05-e2e-loadtest.md`. After tuning the vLLM defaults
(quantization=`awq_marlin`, max-model-len=4096,
gpu-memory-utilization=0.72) and downloading `Qwen/Qwen3-8B-AWQ`,
this report re-runs the 5-concurrent `/ask` test against the
production-target orchestrator.

## Setup

| Component | Value |
|---|---|
| Hardware | NVIDIA GeForce RTX 4070, 12 GB, sm_89 (Ada) |
| Orchestrator | `Qwen/Qwen3-8B-AWQ` (5.71 GiB weights) |
| Quantization kernel | `awq_marlin` (vLLM-recommended over plain `awq`) |
| vLLM GPU fraction | 0.72 (~8.85 GB budget) |
| `--max-model-len` | 4096 |
| `--kv-cache-dtype` | `fp8_e5m2` |
| `--max-num-seqs` | 8 |
| Embedder | `google/embeddinggemma-300m` (lazy-loaded, ~0.6 GB BF16) |
| Reranker | `BAAI/bge-reranker-v2-m3` (lazy-loaded, ~2 GB BF16) |
| VLM | disabled (`parse.disable_vlm=True`) |
| Vault | 6 documents, 49 entities, 60 MENTIONS, 1 CITES edge |
| Concurrency | 5 parallel `curl POST /ask` against the persistent web server |

vLLM cold start to reachable: **40 s**.

## Per-query outcomes

| Query | Time | Outcome | Citation |
|---|---|---|---|
| Q1 — "How does Foucault relate power to discourse?" | **13.9 s** | ✅ ANSWERED | `038920a2-02-foucault-power#0fa49582d0` |
| Q3 — "What does Latour mean by symmetry?" | **13.9 s** | ✅ ANSWERED | `3b6089c5-04-latour-actor-network#86a8240abe` |
| Q4 — "How does Butler describe gender performativity?" | **14.0 s** | ✅ ANSWERED | `aa74a4db-05-butler-performativity#a64ff01629` |
| Q2 — "What is Haraway's critique of objectivity?" | **16.6 s** | ✅ ANSWERED | Haraway chunks |
| Q0 — "What does Bourdieu say about habitus?" | 24.5 s | ⚠️ REFUSED — 1 ungrounded claim after 2 regen attempts | refuse path held |

**Wall clock for all 5 in parallel: 24.5 s** (Q0's two regenerate
attempts dominate; the four answered queries returned in 13.9–16.6 s
each).

## Resource peaks under load

| Metric | Quiescent | Peak | Final |
|---|---|---|---|
| VRAM       | 9.6 GB    | **11.7 GB / 12 GB (95%)** | 9.7 GB |
| GPU util   | 0%        | **100% sustained 5–15 s** | 0% |
| GPU temp   | 42 °C     | 64 °C                     | 62 °C |
| RAM        | 9.6 GB    | 12.2 GB                   | ~10 GB |
| Swap       | 1.1 GB    | 1.1 GB (stable)           | 1.1 GB |
| Load avg   | 1.05      | 0.73 (server is async)    | 0.73 |

The peak VRAM of 11.7 GB validates the bootstrap warning's estimate
of 10.1 GB plus the in-flight KV-cache fill (~1.6 GB additional
under 5-way concurrency). 0.6 GB headroom is tight but stable —
no OOM, no swap pressure.

## Comparison vs the 0.6B baseline

|                 | Qwen3-0.6B | Qwen3-8B-AWQ |
|---|---|---|
| Answered        | 0/5        | **4/5**         |
| Refused/Error   | 5/5        | 1/5             |
| Wall clock      | 24.0 s     | 24.5 s          |
| Peak VRAM       | 10.1 GB    | 11.7 GB         |
| GPU util peak   | 62%        | **100%**         |
| Cold start      | ~3 s       | 40 s            |

The 8B-AWQ takes 10× longer to cold-start (model size + KV-cache
profile dominates), but produces useful answers where the 0.6B
either refused or hit schema-violation errors. The audit fix that
made schema-violations render a friendly red panel (instead of a
500) still applies — none fired here because 8B-AWQ doesn't produce
malformed JSON under schema constraints.

## What the audit fixes did under load

- **OOM circuit breaker** — never tripped (correct, no OOMs)
- **`_cleanup_partial`** — never fired (no partial loads)
- **Refuse path** — fired once (Q0), rendered the proper amber
  banner with all 10 retrieved chunks visible to the user
- **`/ask` MemexError catch** — never needed (no ModelCallError;
  8B produces schema-valid JSON consistently)
- **Per-doc `asyncio.Lock`** — irrelevant under read-only concurrent
  /ask (no writes), but the design held
- **Bootstrap VRAM-fit warning** — predicted 10.1 GB, real peak
  was 11.7 GB. The 1.6 GB delta is the KV-cache fill under 5-way
  concurrency — a known correction we accept as informational

## Headline

The **Qwen3-8B-AWQ + awq_marlin + 0.72 GPU fraction + 4096
max_model_len + fp8_e5m2 KV cache** combination is now the
verified-good production default on a 12 GB RTX 4070.

- 4 of 5 concurrent queries return grounded, cited answers in ≤17 s
- 100% GPU utilisation during decode — no idle bubbles, no waste
- 0.6 GB VRAM headroom under peak — within tolerance
- Refuse path is the only failure mode and it renders cleanly

If you want more headroom (e.g. to load the VLM on top), the
smallest-footprint reduction is the reranker: swap
`BAAI/bge-reranker-v2-m3` (2 GB) for `BAAI/bge-reranker-base`
(~0.25 GB) — costs some ranking quality, frees ~1.75 GB.
