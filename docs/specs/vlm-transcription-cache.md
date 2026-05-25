# VLM transcription cache + SDPA-math determinism

**Status:** shipped 2026-05-25. **Touches:** `parse/vlm_cache.py` (NEW), `parse/vlm_backend.py`, `parse/pipeline.py`, `index/pipeline.py` (teardown), `cli/commands.py` (`--refresh-vlm`).

## Problem

The VLM (`Qwen2.5-VL-7B-Instruct-AWQ`) transcribes diagram/scan pages that Docling can't read. Its **greedy** output is non-deterministic: re-parsing the same page yields a materially different transcription (observed markdown 3283↔2824 bytes; content occasionally dropped). Root cause (3-subagent research): BF16's 7-bit mantissa lets near-tied top-token logits flip under FP accumulation-order variance (AWQ split-K atomics, the SDPA `mem_efficient` backend, CUDA scheduling), and a single early flip cascades. `do_sample=False` is set — it is NOT a sampling bug. Near-bit-reproducibility is unreachable on AWQ-BF16 without FP32 accumulation (OOMs the 12 GB rig).

Impact: churns the content-addressed `chunk_id = sha1(doc_id + chunk.text)` on every re-parse → re-embeds, eval-anchor drift (the `cr350-diagrams` ANS count drifted 11→9→10 across re-parses), occasional content drop. The **HARD gates stay stable** (dropped content → correct refusal, never fabrication) — this is a reproducibility/consistency issue, not a safety one.

The research team (see `vlm_path_revival_2026_05_25` memory) rejected vLLM-for-VLM (breaks the 12 GB pause-vLLM model; ADR-0001/0006), multi-sample voting (VRAM), constrained decoding (variance is in content words, not schema), and global `torch.use_deterministic_algorithms` (raises on the mem_efficient/AWQ ops; AWQ kernels stay non-deterministic regardless). Chosen: **cache (reproducible-by-construction) + SDPA-math (steadier first draw)**.

## Design

**Completeness via best-of-N** — because a non-deterministic draw can silently DROP content (we observed a package description present in one draw, gone in the next), `_convert_with_handle` takes `parse.vlm_transcription_samples` (default 1) independent greedy draws and keeps the **LONGEST** — a content-completeness proxy, since a draw that drops content is shorter. The chosen draw is what gets cached, so the completeness choice is made once and frozen reproducibly. N>1 costs N× per-page VLM time on the first parse only (cached after); draws are **sequential** so there is no extra VRAM. Default 1 (no cost change); raise to 2–3 to converge toward the most-complete transcription.

**(SDPA-math — tried and reverted 2026-05-25)** Forcing the deterministic SDPA *math* backend in the generate (to steady the draw) CUDA-OOMs on the 12 GB rig — the math backend materialises the full B×H×S×S attention matrix for Qwen2.5-VL's ~1k+ visual tokens. best-of-N (no extra VRAM) is the kept completeness mechanism; the **cache** is the reproducibility guarantee.

**Cache** — `VLMTranscriptionCache` (`parse/vlm_cache.py`) mirrors `index/fts_store.py`: sync `sqlite3` under `asyncio.to_thread`, writes gated by an `asyncio.Lock`. DB at `vault/.memex/vlm_cache.sqlite`; schema `vlm_page_cache(cache_key PK, pdf_sha256, page_no, vlm_model, prompt_sha8, markdown, created_at)` + a `pdf_sha256` index. Key = `f"{sha256(pdf_bytes)}:{page}:m={model_id}:p={prompt_sha8}"` — hashing the **source-PDF bytes** (not the rendered image) is content-true and cheap (one read/doc, shared across pages); model-id + prompt-sha in the key make a model/prompt change a natural miss (no explicit invalidation). `INSERT OR IGNORE` (first writer wins).

**Integration** — `convert_pages(cache=, refresh_vlm=)`: first pass serves cache hits as `DoclingPageOutput` and **skips the GPU** (the `registry.use("vlm")` load only happens if a page misses); second pass transcribes misses under one acquisition and `put`s each, guarded by `_MIN_CACHEABLE_CHARS=20` so a punted/empty draw is NOT frozen (the next parse retries). `cache=None` ⇒ pre-cache behaviour. `_route_and_escalate` takes `cache`/`refresh_vlm` (default `None`/`False` — keeps the unit tests' direct calls unchanged); `_parse_with_docling` opens the cache (only when `not disable_vlm`), passes it, closes it in a `finally`; `refresh_vlm` threads CLI → `parse_document` → `_parse_pdf` → `_parse_with_docling` → `_route_and_escalate` → `convert_pages`. `vlm_cache.sqlite` is regenerable derived state (ADR-0003), dropped by `reindex --force`; `memex parse --refresh-vlm <doc>` busts one document.

## Invariant

Re-parsing an unchanged `(pdf, model, prompt)` page is **byte-identical** to its first transcription → `chunk_id` stable, eval anchors stable, no content drop. The cache is parse-side only and never touches the assess/answer/verify path, so the HARD gates (`refusal_cf=1.0`, 0 hallucinations) are unaffected.

## Verification

Parse the diagram doc twice with the VLM enabled: run 1 escalates + transcribes 4 pages (cache populated); run 2 logs `vlm.cache_hit` ×4, runs **zero VLM forward passes**, and writes a **byte-identical** `.md` (so `memex index` reports all chunks unchanged). The eval baseline then reproduces run-over-run — the ANS-count drift is gone.
