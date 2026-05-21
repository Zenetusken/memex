# Hardware tiers

Memex ships two **eval-verified** hardware-tier profiles. Pick the one that matches your GPU; the env-var matrix below activates each profile end-to-end.

The tiers were validated empirically against the 30-query slide-decks evaluation corpus (`tests/eval-data/slide-decks/queries.json`). Both clear the project's **HARD GATES**:

- `refusal_rate_on_counterfactuals = 1.0` — every refusal-mode query refused correctly
- **Zero hallucinations** — no fabricated answers on near-miss or empty-retrieval queries

The trade-off between tiers is **how often the model attempts an answer** (answered_count) and **how reliably its citations point at the literal-answer chunk** (`mcp_answered_only`). Hard quality floors hold across both tiers; medical/legal/exam-study use cases are honoured by both.

---

## 12 GB tier (default — Qwen3-8B-AWQ)

**Recommended for**: RTX 4070, 4080, 3080-12GB, A4000, RTX 5070, any consumer GPU with 12-15 GB VRAM.

### Activation

```sh
# Defaults — no env var override needed.
./scripts/serve-vllm.sh
```

Or equivalently, with explicit env vars:

```sh
MEMEX_VLLM_MODEL=Qwen/Qwen3-8B-AWQ \
MEMEX_VLLM_QUANTIZATION=awq_marlin \
MEMEX_VLLM_MAX_MODEL_LEN=6144 \
MEMEX_VLLM_GPU_FRACTION=0.72 \
MEMEX_VLLM_KV_CACHE_DTYPE=fp8_e5m2 \
  ./scripts/serve-vllm.sh
```

The `cli/bootstrap.py` VRAM-fit check will pre-flight this configuration against detected GPU memory and fail fast with an actionable message if there isn't enough room.

### Eval-verified quality

3 baseline runs against the 30-query slide-decks corpus on RTX 4070 12 GB:

| Metric | median | range | HARD GATE |
|---|---|---|---|
| `answered_count` | 9 | 9-12 | — |
| `mean_citation_precision_answered_only` | 0.89 | 0.75-0.89 | — |
| `refusal_rate_on_counterfactuals` | **1.00** | stable | ✅ |
| Hallucinations | **0** | stable | ✅ |
| Wall-clock per eval | ~115 s | — | — |

### Memory footprint

Live VRAM usage observed on RTX 4070 12 GB (cold start, single eval pass):

- vLLM (Qwen3-8B-AWQ weights + KV cache at max-model-len=6144 + CUDA graphs): ~8.5-9.5 GB
- In-process embedder (EmbeddingGemma-300m, BF16): ~1 GB
- In-process reranker (bge-reranker-v2-m3, BF16): ~1.5 GB
- Transient activation overhead during inference: ~0.5 GB
- **Total live usage**: ~11.5 GB
- Headroom: ~0.5 GB (tight; back-to-back evals can OOM intermittently — see Operational notes below)

---

## 8 GB tier (smaller orchestrator — Qwen3-4B-AWQ)

**Recommended for**: RTX 3060 12GB (if you want more headroom), RTX 4060 Ti 8GB, RTX 3070 Ti 8GB, GTX 1080 12GB. Also viable on a 12 GB card if you want the 5 GB of headroom freed (the smaller orchestrator unblocks the optional `qwen3` reranker backend or future parallel-model experiments).

### Activation

```sh
MEMEX_VLLM_MODEL=Qwen/Qwen3-4B-AWQ \
MEMEX_VLLM_GPU_FRACTION=0.50 \
  ./scripts/serve-vllm.sh
```

The agent also needs to know the smaller orchestrator name when calling vLLM:

```sh
# At inference / eval time, after the daemon is up:
export MEMEX_MODELS__ORCHESTRATOR=Qwen/Qwen3-4B-AWQ
```

If you use the systemd templates from `docs/deploy/memex-vllm.env`, copy them and override these two variables. The 8 GB tier doesn't change anything in `memex-web.env` or `memex-mcp.env` beyond `MEMEX_MODELS__ORCHESTRATOR`.

### Eval-verified quality

3 runs against the same 30-query corpus on the same RTX 4070 12 GB, but with the smaller stack active:

| Metric | median | range | HARD GATE |
|---|---|---|---|
| `answered_count` | 4 | 4-5 | — |
| `mean_citation_precision_answered_only` | 0.75 | 0.75-0.80 | — |
| `refusal_rate_on_counterfactuals` | **1.00** | stable | ✅ |
| Hallucinations | **0** | stable | ✅ |
| Wall-clock per eval | ~80 s | — | — |

### Why fewer answers?

Qwen3-4B is **more conservative** than its 8B sibling under the same `prompts/answer/v2.md` (the literal-presence rule). It refuses more often where the 8B would attempt an answer. For high-stakes contexts (medical, legal, exam-study), that's a feature: the model leans toward "I don't know" over invention. For exploratory research, it can feel restrictive.

If you want to soften this without changing tiers, you can:

- Loosen `prompts/answer/v2.md` (drop the worked examples — but the literal-presence rule itself is what gives you the zero-hallucinations guarantee, so loosen at your own risk).
- Increase `MEMEX_RERANK_TOP_K` to surface more chunks (`5` → `8`) — risks `max-model-len` overflow, which surfaces as a `BadRequestError` from vLLM.
- Decrease the chunk truncate in `prompts/answer/v{1,2}.md` from `truncate(1800)` to `truncate(1500)` (frees ~1500 chars of budget at top_k=5) — pair with bumping top_k.

### Memory footprint

Live VRAM usage on RTX 4070 12 GB (the 4B model uses LESS than the 8B, even at gpu_fraction=0.50):

- vLLM (Qwen3-4B-AWQ weights + KV cache + CUDA graphs): ~6.9 GB
- In-process embedder + reranker + activations: ~2 GB
- **Total live usage**: ~9 GB
- Headroom: ~2.5 GB (comfortable; back-to-back evals do not OOM)

**Why gpu_fraction must drop:** vLLM reserves `gpu_fraction × total_gpu_memory` upfront regardless of the actual model size. At `gpu_fraction=0.72` on a 12 GB rig, vLLM holds ~8.5 GB whether the model is 8B (uses it all) or 4B (uses ~5 GB, wastes the rest). To realize the freed VRAM, lower `gpu_fraction=0.50`.

---

## Smaller-than-8GB — currently unsupported

Memex doesn't ship a sub-8GB tier. The minimum viable stack needs:

- ~3 GB for the orchestrator weights at AWQ-int4 (Qwen3-4B-AWQ)
- ~1.5 GB for the KV cache at max-model-len=6144 with fp8_e5m2
- ~1 GB for the embedder
- ~1.5 GB for the reranker
- ~1 GB transient overhead

That's ~8 GB. Anything tighter would require:

- Smaller orchestrator (Qwen3-1.7B exists but quality on the slide-decks corpus is unmeasured — could re-run Session 3a's pattern against it if there's demand)
- Smaller reranker (`bge-reranker-base` at ~0.3 GB instead of `bge-reranker-v2-m3` at ~1.5 GB) — also unmeasured against this corpus
- Lower `max-model-len` (drops KV cache cost linearly but blows the chunk-truncate budget; quality regresses)

Open an issue if you want to validate a sub-8GB profile. The bones are there; just needs a 3-run benchmark like P4.2 Sessions 3-4.

---

## Reranker backend (orthogonal to tier choice)

The reranker backend is a separate config dimension, settable on either tier:

| Backend | `MEMEX_MODELS__RERANKER_BACKEND` | Eval verdict |
|---|---|---|
| **cross_encoder** (default, recommended) | unset or `cross_encoder` | ✅ Median ANS=9 on 12 GB tier, ANS=4 on 8 GB tier |
| qwen3 | `qwen3` | ❌ Median ANS=0 on either tier — ranks thematically-general chunks above the literal-answer chunk |

The qwen3 backend is kept as an opt-in for users with different corpora (e.g., where topical relevance dominates over fact extraction). For slide-deck-style content with chart-text tables and structured-diagram answers, cross_encoder's fact-extraction-relevance ranking is markedly better-suited. See P4.2 Session 4 in `docs/ROADMAP.md` for the head-to-head numbers.

---

## Operational notes

### Back-to-back evals on the 12 GB tier

The 12 GB production stack lives at ~0.5 GB of headroom. We've observed intermittent OOM when running >1 eval sequentially within the same vLLM session — vLLM's prefix-cache grows slightly between runs (from ~8.45 GB to ~8.66 GB after one eval pass), and the embedder + reranker that the eval process re-loads each time can find themselves short on memory.

**Mitigations**:
- Restart vLLM between back-to-back benchmark runs: `systemctl --user restart memex-vllm` (if using the daemon stack) or `pkill -TERM -f "vllm serve" && nohup ./scripts/serve-vllm.sh > /tmp/vllm.log 2>&1 &`
- Move to the 8 GB tier for the duration of benchmarking — the smaller orchestrator's ~2.5 GB of extra headroom eliminates the OOM cycle entirely.

### vLLM prefix-cache and determinism

vLLM's prefix-cache (`--enable-prefix-caching` is on by default) introduces session-dependent variance even at `temperature=0`. We've observed `answered_count` shifting between 9 and 12 across runs at the same vLLM config. For benchmarking, characterize your stack across 3 cold-start runs (restart vLLM each time) and report median ± range rather than a single number.

### Switching tiers

If you swap orchestrator size mid-vault, **nothing in the vault needs re-indexing**. The vault is parser/orchestrator-agnostic — the LanceDB embeddings, the FTS5 index, and the markdown source are all reusable. Only the live inference daemon changes.

---

## Verified configurations

These are the only two tiers we've eval-verified against the slide-decks corpus. Anything outside these — different orchestrator, different reranker, different parameter sweep — is unmeasured and may shift quality in either direction. P4.2 Sessions 1-4 in `docs/ROADMAP.md` document the full verification protocol if you want to re-run it against your own corpus.

If you ship a measured profile against a different corpus or hardware setup, a PR is welcome.
