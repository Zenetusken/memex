# pyright: reportConstantRedefinition=false
# `_SETTINGS` is an uppercase module-level singleton intentionally
# rebound by `set_settings()` (test setup / bootstrap) and cleared
# by `reset_settings()`.

"""Centralized settings — see GUIDELINES.md Part II "Configuration".

A single `MemexSettings` model is the source of truth. Loaded once at
startup from `~/.config/memex/config.toml` with environment overrides,
validated for things like "the configured model fits in available VRAM."

Fail loudly at startup, never silently at runtime.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from memex.core.resources import CoResidenceMode


class VLMServeSettings(BaseModel):
    """Recipe for the short-lived VLM vLLM process (parse-time only),
    used when `ModelSettings.vlm_serving == "vllm"`.

    Validated 2026-05-26 on the 12 GB RTX 4070 for
    `cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit` (POC, see ADR-0006
    §VLM-via-vLLM): the desktop display holds a VARIABLE ~1–2 GB (Xorg +
    compositor + whatever apps the user has open — a Zoom call or browser
    tab swings it by hundreds of MB), so `gpu_memory_utilization` is kept
    at **0.80** for headroom. 0.89 worked on an idle desktop but
    intermittently failed startup ("Free memory < desired GPU memory
    utilization") mid-bulk-reingest once the desktop's GPU use rose — a
    one-time startup gate, so the margin must absorb the desktop's peak,
    not its idle. The vision encoder cache reserves for the MAX image
    unless `max_pixels` is capped (else KV starves and vLLM refuses to
    start); `max_model_len` 3072 fits a page transcription (image
    ~1280 visual tokens + <800 output) with KV headroom even at the lower
    util; `enforce_eager` skips CUDA-graph capture (faster startup, frees
    memory). Runs on a port DISTINCT from the orchestrator so the parse
    pause's reachability check (against the orchestrator base_url) never
    targets it; the process is started + torn down inside
    `vlm_backend.convert_pages`, so it phase-separates from the in-process
    chart-OCR pass that follows (the two can't co-reside: ~7.4 + ~3 GB >
    12 GB)."""

    host: str = "127.0.0.1"
    port: int = Field(default=8001, ge=1, le=65535)
    gpu_memory_utilization: float = Field(default=0.80, ge=0.1, le=1.0)
    max_model_len: int = Field(default=3072, ge=512)
    # Cap the visual-token budget so the vLLM vision encoder cache doesn't
    # reserve for the model's max image (~16384 tokens). 1280*28*28 mirrors
    # the rasteriser cap in parse/vlm_backend._render_page_to_image.
    max_pixels: int = Field(default=1003520, ge=50176)
    min_pixels: int = Field(default=200704, ge=784)
    startup_timeout_s: int = Field(default=180, ge=10)


class SummarizerServeSettings(BaseModel):
    """Recipe for the short-lived SUMMARIZER vLLM process (summarize-time only), used when
    `ModelSettings.summarizer` is set (ADR-0010 swap-in). A stronger model (e.g.
    Gemma-3-12B-it-AWQ) is served briefly on the GPU freed by `pause_vllm_for_gpu` — the
    same proven parse-time VLM lifecycle, text-only — to break the cross-paragraph
    repetition an 8B can't, then torn down and the orchestrator restored.

    On the 12 GB rig the orchestrator (and VLM) are DOWN during the swap, so the card is
    free for the 12B; `gpu_memory_utilization` 0.85 (still leaves headroom for the
    desktop's variable ~1-2 GB), `max_model_len` 8192 (fits a report MAP/GROUND chunk batch
    + the rolling overview-so-far + bounded output), `enforce_eager` for fast startup. A
    port DISTINCT from the orchestrator (8000) and the VLM (8001); a 12B loads slower than
    the 8B VLM, so a longer `startup_timeout_s`."""

    host: str = "127.0.0.1"
    port: int = Field(default=8002, ge=1, le=65535)
    gpu_memory_utilization: float = Field(default=0.85, ge=0.1, le=1.0)
    max_model_len: int = Field(default=8192, ge=512)
    startup_timeout_s: int = Field(default=300, ge=10)


class ModelSettings(BaseModel):
    """Pydantic-settings record for every model the registry owns —
    orchestrator (out-of-process via vLLM), embedder, reranker, VLM,
    and chart-OCR. The defaults match the 12 GB RTX 4070 reference
    rig; tighter rigs override via env vars (see `docs/deploy/
    hardware-tiers.md`)."""

    # The unified Qwen3.5-4B (ADR-0015): hybrid-reasoning VL model serving the
    # grounded orchestrator role. Re-baseline 2026-06-01 held the HARD gate on
    # all 12 answer-eval corpora at N=3. Kill-switch = revert this id (+ quant)
    # to "Qwen/Qwen3-8B-AWQ"/"AWQ" and `memex daemon restart` (the swap touches
    # zero derived state — chunk_ids/embeddings/FTS/graph are orchestrator-agnostic).
    orchestrator: str = "cyankiwi/Qwen3.5-4B-AWQ-4bit"
    # vLLM's GGUF path is flagged experimental; AWQ/GPTQ are the production
    # path on Ada (see ADR-0001 Revisit + ADR-0006). `compressed_tensors` is the
    # W4A16 pack-quantized format of the 4B above (the serve OMITS --quantization
    # — vLLM auto-detects — and falls back to `auto` KV, since the 4B is an fp8
    # checkpoint that rejects fp8_e5m2 KV; both handled by
    # `daemon/supervisor.orchestrator_serve_env`). The Q*_K_M variants are kept
    # for users running gguf-via-vLLM experimentally.
    orchestrator_quantization: Literal[
        "AWQ", "GPTQ", "compressed_tensors", "Q4_K_M", "Q5_K_M", "Q8_0"
    ] = "compressed_tensors"
    # VLM default: Qwen3-VL-8B-AWQ, served via vLLM (`vlm_serving`). The
    # in-process transformers path CANNOT run this compressed-tensors
    # build on 12 GB — it decompresses int4→dense (~16 GB) and OOMs;
    # vLLM's Marlin int4 kernel runs it at ~7.4 GB (POC 2026-05-26
    # confirmed it fixes the diagram-flattening limit Qwen2.5-VL had).
    # The legacy AutoAWQ build `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` still
    # works via `vlm_serving="transformers"`. See ADR-0006 §VLM-via-vLLM.
    vlm: str = "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"
    vlm_quantization: Literal["awq_int4", "bf16"] = "awq_int4"
    # How the VLM runs during parse. "vllm": a short-lived vLLM process on
    # `vlm_serve.port` transcribes pages over the OpenAI multimodal API
    # (required for Qwen3-VL — no in-process int4 kernel for its
    # compressed-tensors format). "transformers": the legacy in-process
    # registry path (AutoAWQ Qwen2.5-VL). See VLMServeSettings.
    vlm_serving: Literal["transformers", "vllm"] = "vllm"
    vlm_serve: VLMServeSettings = Field(default_factory=VLMServeSettings)
    # OPTIONAL summarizer swap-in (ADR-0010): when set, `report`-detail summaries serve
    # this stronger model briefly at summarize-time (on the GPU freed by pausing the
    # orchestrator) to break the cross-paragraph repetition an 8B can't. None (default) =
    # no swap (the orchestrator does the summary). Researched pick:
    # `gaunernst/gemma-3-12b-it-int4-awq`. `MEMEX_MODELS__SUMMARIZER=...`.
    summarizer: str | None = None
    summarizer_serve: SummarizerServeSettings = Field(default_factory=SummarizerServeSettings)
    embedder: str = "google/embeddinggemma-300m"
    reranker: str = "BAAI/bge-reranker-v2-m3"
    # P3.3 chart-OCR model — default `google/deplot` per Session 1
    # verdict (Apache 2.0, fine-tuned for plot→linearised-table,
    # Chart-OCR backend. Default `nvidia/NVIDIA-Nemotron-Parse-v1.2`
    # since the 2026-05-23 P3.3-c shootout (see
    # `docs/audits/chart_ocr_shootout_2026-05-23.md`) — first
    # backend that doesn't regress on prose-heavy corpora
    # (ANS == baseline; mcp_ans 0.955; refusal_cf 1.0). 0.88B params,
    # ~3 GB live in BF16. Requires `trust_remote_code=True` per the
    # ADR-0006 broadened amendment (chart-OCR slot only, opt-in by
    # env var). Alternatives in tree: `google/deplot`,
    # `khhuang/chart-to-table`, `kppkkp/OneChart`.
    chart_ocr: str = "nvidia/NVIDIA-Nemotron-Parse-v1.2"
    # Reranker backend selector. `cross_encoder` (default) loads the
    # `sentence_transformers.CrossEncoder` model named in `reranker`
    # (today: bge-reranker-v2-m3). `qwen3` loads a Qwen3-Reranker
    # decoder via `transformers.AutoModelForCausalLM` and scores via
    # softmax over the yes/no token logits — see P2.1 in
    # docs/ROADMAP.md + retrieve/rerank.py. Set `reranker` to the
    # matching checkpoint id (e.g. `Qwen/Qwen3-Reranker-0.6B`) when
    # flipping the backend. Quality A/B between the two awaits the
    # eval corpus (P0); memory wins of ~1.4 GB on a 12 GB rig are
    # available today.
    reranker_backend: Literal["cross_encoder", "qwen3"] = "cross_encoder"
    # Co-residence MODE — the high-level VRAM-tradeoff knob (ADR-0007). A named
    # bundle that resolves (via core/resources.py::resolve_profile) to a
    # concrete posture: the retrieval device placement here PLUS the
    # orchestrator's gpu-fraction + max-model-len (applied by the daemon). When
    # set to anything other than "manual" it OVERRIDES the explicit
    # embedder_device / reranker_device fields below. Default "manual" keeps the
    # raw device knobs authoritative (backward-compatible). `fast` = low-latency
    # top-k RAG; `full` = whole-document context for long-form synthesis
    # (reranker→CPU); `gpu_only` = all-GPU at full util (>12 GB cards).
    # `MEMEX_MODELS__CO_RESIDENCE_MODE=full`.
    co_residence_mode: CoResidenceMode = "manual"
    # Device placement for the two retrieval models. Default "cuda" (bf16,
    # per ADR-0006). Set either to "cpu" (loads fp32 on CPU) to free GPU VRAM
    # for a fuller orchestrator KV cache when co-residing on a single 12 GB
    # card — e.g. run `memex serve web` with the orchestrator at its full
    # gpu_memory_utilization by pushing the reranker (the ~2 GB bf16
    # load-time OOM culprit) and/or the embedder onto the CPU. The retrieval
    # path follows each model's own device, so this is the only switch.
    # Trade-off: CPU rerank of ~50 candidates adds a few seconds of latency
    # (NOT per-token, unlike a vLLM weight offload) — acceptable for
    # single-user interactive asks. `MEMEX_MODELS__RERANKER_DEVICE=cpu`.
    embedder_device: Literal["cuda", "cpu"] = "cuda"
    reranker_device: Literal["cuda", "cpu"] = "cuda"


class HardwareSettings(BaseModel):
    """Torch-level CUDA budget + concurrency knobs. `gpu_memory_fraction`
    is the cap `torch.cuda.set_per_process_memory_fraction` enforces;
    vLLM has its own `gpu_memory_utilization` flag set in
    `scripts/serve-vllm.sh`."""

    gpu_memory_fraction: float = Field(default=0.85, ge=0.0, le=1.0)
    max_concurrent_documents: int = Field(default=2, ge=1)
    cpu_workers: int = Field(
        default_factory=lambda: max(1, (os.cpu_count() or 2) - 1),
        ge=1,
    )


class SamplingSettings(BaseModel):
    """Sampling defaults for `complete_structured` — see the Qwen3
    prompt-engineering research notes
    (`docs/audits/qwen3_prompt_engineering_2026-05-22.md`).

    These values came out of the published Qwen team guidance for
    non-thinking-mode Qwen3-8B-AWQ, scaled down for eval determinism:
    - `temperature=0.1` — escapes Qwen-team-cautioned pure greedy
      without introducing noticeable nondeterminism at this batch size.
    - `top_p=0.8` — Qwen team's non-thinking-mode default.
    - `presence_penalty=1.0` — suppresses repetition. Capped at 1.0
      because AWQ-quantized models trigger language mixing under
      `presence_penalty > 1.5` (relevant for multilingual content).
    - `seed=42` — reproducibility floor across runs.
    - `max_tokens=1024` — Path C tightening; comfortably above the
      ~940-token worst-case `DraftAnswer` schema output.

    Per-call kwargs to `complete_structured` still override these.
    """

    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    presence_penalty: float = Field(default=1.0, ge=-2.0, le=2.0)
    seed: int | None = 42
    max_tokens: int = Field(default=1024, ge=1)


class InferenceSettings(BaseModel):
    """vLLM endpoint — see ADR-0001."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"  # vLLM doesn't require a real key by default
    request_timeout_s: int = Field(default=120, ge=1)
    # Path to the vLLM launch script (used by `memex daemon start`).
    # Resolved relative to CWD if not absolute. Override via
    # `MEMEX_INFERENCE__SERVE_SCRIPT=/path/to/serve-vllm.sh`.
    serve_script: Path = Field(default=Path("scripts/serve-vllm.sh"))
    daemon_startup_timeout_s: int = Field(default=120, ge=10)
    # Sampling defaults for `complete_structured`. Override via
    # `MEMEX_INFERENCE__SAMPLING__TEMPERATURE=0.0` etc.
    sampling: SamplingSettings = SamplingSettings()


class IngestSettings(BaseModel):
    """Ingestion validation knobs — see GUIDELINES.md Part VI security."""

    max_bytes: int = Field(default=256 * 1024 * 1024, ge=1024)
    allow_macros: bool = False  # macro-bearing Office docs are rejected by default


class ParseSettings(BaseModel):
    """Parser routing knobs — see IMPLEMENTATION-PLAN §1.3."""

    vlm_confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    # Escalate a Docling page to the VLM when figures cover at least this
    # fraction of the page area. The confidence trigger above is the wrong
    # signal for diagram/figure pages: Docling reports high confidence for
    # a slide whose title it read cleanly while the diagram content (the
    # part only a VLM can read) is entirely lost. An image-area-dominant
    # page is exactly the case the VLM exists for, so route it there
    # regardless of text confidence. Calibrated on real CR350 network
    # diagrams (firewall-architecture 0.38, pfSense screenshot 0.26,
    # 802.1X sequence 0.26, network-zoning 0.24): 0.20 routes every
    # substantively diagrammatic slide to the VLM — the zoning diagram at
    # 0.24 is the smallest real diagram in that set, so the bar sits just
    # below it — while leaving prose pages that carry only a small
    # logo/icon (well under a fifth of the page) on the Docling path.
    vlm_image_area_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    # Second VLM-escalation arm: route a page to the VLM when it carries a
    # figure Docling's PictureClassifier labels as one of these DIAGRAM
    # types, even if the page is not image-area-dominant. The chart-OCR pass
    # handles data charts (bar/line/pie) but deliberately EXCLUDES diagrams
    # (no extractable rows+cols); a flow chart or engineering drawing on a
    # text-heavy slide then falls under vlm_image_area_threshold and is
    # transcribed by NEITHER pass. Observed on the CR350 networking lectures:
    # 28 engineering drawings + 25 flow charts stranded on sub-0.20 pages.
    # Empty tuple disables this arm. snake_case values taken verbatim from
    # docling_core's PictureClassificationLabel (NOT the prettified markdown
    # rendering — "Screenshot from computer" is the label `screenshot_from_computer`).
    # The diagram complement of parse/chart_ocr_backend._CHART_CLASS_NAMES:
    # block/flow diagrams, schematics, and tool/manual screenshots — the
    # figure types a VLM transcribes but the chart-OCR data-table extractor
    # cannot. (`electrical_diagram`/`cad_drawing` included for engineering
    # decks even though the CR350 set didn't exercise them.)
    vlm_diagram_classes: tuple[str, ...] = (
        "flow_chart",
        "engineering_drawing",
        "electrical_diagram",
        "cad_drawing",
        "screenshot_from_computer",
        "screenshot_from_manual",
        "screenshot",
    )
    # Minimum PictureClassifier confidence to trust a diagram label for the
    # escalation arm above (mirrors the chart-OCR pre-filter's 0.50 gate).
    # Below this the label is treated as unknown and the page is NOT
    # escalated on the classification arm — the image_fraction arm still
    # applies. Docling's v2.5 classifier is typically >0.85 when confident.
    vlm_diagram_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # The VLM's greedy decode is non-deterministic (BF16 mantissa + AWQ/SDPA
    # accumulation order — an early near-tied-logit flip cascades), so a
    # given transcription draw can silently DROP content (we observed a
    # package description present in one draw, gone in the next). Take this
    # many independent draws per page and keep the LONGEST (a completeness
    # proxy — a draw that drops content is shorter). 1 = a single draw
    # (fastest, current default). Raise to 2–3 to converge toward the
    # most-complete transcription at N× the per-page VLM cost — paid once,
    # then the chosen draw is cached (`vlm_cache.py`). No extra VRAM (draws
    # are sequential), so it does not OOM the way the SDPA-math approach did.
    vlm_transcription_samples: int = Field(default=1, ge=1, le=8)
    # Chart-OCR has the SAME greedy non-determinism as the VLM (BF16/AWQ
    # accumulation-order variance flips near-tied logits) and the same
    # failure mode: a given draw can DROP a chart row or trip the
    # ambiguous-header / UNREADABLE refusal and come back empty, so a
    # re-parse silently loses a chart-content answer (it churned the
    # chart-types / slide-decks re-baselines). Take this many independent
    # draws per figure and keep the LONGEST non-empty extraction (the same
    # completeness proxy the VLM uses — a draw that drops a row or refuses
    # is shorter/empty). 1 = a single draw (fastest, current default).
    # Raise to 2–3 to converge toward the most-complete extraction at N×
    # the per-figure cost — paid once, then the chosen draw is cached
    # (`chart_ocr_cache.py`). Sequential draws, so no extra VRAM.
    chart_ocr_extraction_samples: int = Field(default=1, ge=1, le=8)
    # Default-off: the VLM escalation path is fully wired, but it
    # demands ~5 GB of VRAM (Qwen2.5-VL-7B AWQ-Int4 + processor) on top
    # of the embedder + reranker resident set. Opt in once you've
    # confirmed your rig has the headroom; bootstrap's VRAM-fit check
    # will refuse to load it otherwise.
    disable_vlm: bool = True
    # P3.3 chart-OCR pass over Docling figures. Default-off: the pass
    # adds ~2.3 GB of VRAM (Pix2Struct/DePlot BF16) during parse plus
    # ~30s parse-call overhead (vLLM is paused, chart-OCR loaded,
    # figures processed, model unloaded, vLLM restarted). Opt in once
    # Session 5's eval verifies HARD GATES + Q4/Q16/Q21 flips.
    # Set via `MEMEX_PARSE__DISABLE_CHART_OCR=true` to disable.
    # Default flipped to False on 2026-05-23 after the P3.3-c shootout
    # confirmed Nemotron-Parse-v1.2 doesn't regress prose answering
    # (ANS == baseline on the CUDA deck).
    disable_chart_ocr: bool = False
    # Force Docling routing, bypassing the PyMuPDF pre-filter. The
    # classifier would normally win PyMuPDF on born-digital text-heavy
    # PDFs (Adobe InDesign / Acrobat output / etc.); this flag overrides
    # that. Useful when you want chart-OCR to fire on a doc the
    # classifier sees as "no need to OCR" (e.g. a Tableau visualization
    # guide or a 10-K with a few charts). Cost: Docling is ~10× slower
    # than PyMuPDF on text-heavy docs. Set via
    # `MEMEX_PARSE__FORCE_DOCLING=true` or per-call via the
    # `--force-docling` CLI flag on `memex parse` / `memex ingest`.
    force_docling: bool = False
    # 1200s headroom (20 min). The original 300s default fit 30-page
    # papers; 600s was still tight on 100-page slide decks because the
    # layout model is CPU-bound. Most slide decks finish in 3-5 min
    # with OCR disabled (the worker's default — set
    # MEMEX_PARSE_DOCLING_OCR=1 to re-enable for scanned PDFs).
    docling_timeout_s: int = Field(default=1200, ge=10)
    docling_crash_threshold: int = Field(default=5, ge=1)
    # Network-egress sandbox for the Docling worker. When True, the
    # worker installs a seccomp filter blocking all network syscalls
    # before importing docling — see memex/parse/sandbox.py and
    # GUIDELINES.md Part VI. Linux-only (gracefully no-op elsewhere).
    # Set False only for deployments that genuinely need network during
    # parse; the docs recommend pre-fetching models with
    # `huggingface-cli download` instead.
    docling_sandbox_network: bool = True

    # ----- PyMuPDF4LLM pre-filter -----
    # When True, PDFs are first inspected by the PyMuPDF4LLM worker,
    # which extracts the native text layer + a rich signal set
    # (producer metadata, char distribution, image area, mojibake
    # ratio, markdown structure). A tiered classifier then routes the
    # document: high-confidence born-digital → use PyMuPDF (10-20×
    # faster than Docling); mixed-content (text + substantial images)
    # → Docling with OCR forced on for image-embedded text; everything
    # else → Docling default. See pipeline._classify and
    # docs/audits/07-ocr-ab.md.
    pymupdf_enabled: bool = True
    # Confidence threshold for trusting PyMuPDF's output. The classifier
    # emits ~1.0 for born-digital docs with rich text, 0.0 for scans,
    # and ~0.20 for mixed-content (which intentionally falls through to
    # Docling-with-OCR). Default 0.5 rejects sparse and mixed content.
    pymupdf_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # Average fraction of page area covered by images that triggers
    # the "mixed-content" classification. 0.35 = 35% of the page area
    # on average is images → likely contains charts/screenshots/photos
    # with embedded text that needs OCR for full retrieval. Combined
    # with `pymupdf_mixed_content_min_image_heavy_pages` to avoid
    # false-positives on decorative-heavy docs whose image-text is
    # disconnected from retrieval-worthy context. The canonical
    # NVIDIA GTC slide deck (109 pages, 13 images/page avg) measures
    # 0.285 — *below* this threshold by design, because the audit at
    # docs/audits/07-ocr-ab.md showed OCR didn't change a single
    # query outcome for that deck. Users who want more aggressive
    # OCR (academic docs with figure-embedded text, technical
    # diagrams with critical labels) can lower to 0.20.
    pymupdf_mixed_content_image_area_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    # Minimum fraction of pages that must individually be image-heavy
    # (>3 images per page) for the mixed-content classification to
    # fire. 0.30 = 30% of pages must independently flag image-heavy
    # before we force-OCR. Both gates must pass (AND) — protects
    # against false-positives on text docs with a single front-cover
    # image, and on decorative-heavy decks whose image-text is noise.
    pymupdf_mixed_content_min_image_heavy_pages: float = Field(default=0.30, ge=0.0, le=1.0)
    # Slide-deck override (Tier 1.5 in the classifier). When the
    # document's average page aspect ratio is at or above this
    # threshold AND chars-per-page is below the companion threshold,
    # the classifier routes the document to Docling regardless of
    # producer metadata. PyMuPDF text extraction loses chart structure
    # on slide-deck-shaped content (interleaving chart imagery as
    # `[chart-text]` blocks the agent can't ground on); Docling
    # preserves layout as proper tables + figures. Verified on the
    # GTC 2024 CUDA deck: legitimate answer rate 4/7 → 6/7 (+50%).
    # Reference values: portrait letter ≈ 0.77; 4:3 slides ≈ 1.33;
    # 16:9 slides ≈ 1.78. 1.3 is the floor catching both 4:3 and 16:9.
    pymupdf_slide_deck_aspect_threshold: float = Field(default=1.3, ge=1.0, le=3.0)
    # Companion to pymupdf_slide_deck_aspect_threshold. When the
    # document's average chars-per-page is below this value AND the
    # aspect threshold is met, the document is classified as a slide
    # deck and routed to Docling. Reference values: typical slides
    # have 200–700 chars per page; documents typically 2000+. 800 is
    # the floor catching even text-heavy slides without trapping
    # cover-page-light documents (which fail the aspect gate anyway).
    pymupdf_slide_deck_max_chars_per_page: int = Field(default=800, ge=0)
    # P3.3 Session 2: chart-heavy slide-deck escape valve. When the
    # aspect-ratio gate is met AND image_area_fraction crosses this
    # threshold, the slide-deck classification fires regardless of
    # chars_per_page. This catches chart-heavy decks (e.g., GPU /
    # architecture slide decks where chart-text inflates the per-page
    # char count past 800 but the content is dominated by figures).
    # Without this gate, PyMuPDF emits `[chart-text]` blocks of axis
    # labels interleaved with body prose; chart-OCR over Docling
    # figures handles the same content as structured tables.
    pymupdf_slide_deck_chart_heavy_image_area_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    pymupdf_timeout_s: int = Field(default=120, ge=5)
    pymupdf_crash_threshold: int = Field(default=5, ge=1)
    # Network-egress sandbox for the PyMuPDF worker. Symmetric with
    # docling_sandbox_network — PyMuPDF should never need the network
    # but the principle of least authority applies all the same.
    pymupdf_sandbox_network: bool = True


class McpSettings(BaseModel):
    """MCP server knobs — see docs/deploy/mcp-http.md.

    When `auth_token` is set, `memex serve mcp --transport http`
    requires `Authorization: Bearer <token>` on every request,
    verified via constant-time comparison. When unset, the HTTP
    transport refuses to bind a non-loopback address at startup,
    and warns-but-runs on loopback as a developer affordance.

    Generate a token with `memex mcp generate-token` and put the
    output in `MEMEX_MCP__AUTH_TOKEN` (env) or
    `~/.config/memex/config.toml` (`[mcp] auth_token = "…"`). One
    token = full access to all MCP tools; rotate by setting a new
    value and restarting the server. Out of scope here: OAuth,
    mTLS, token expiry, multi-token / scoped access. Put a reverse
    proxy in front if you need those.

    stdio transport is unaffected — auth only applies to HTTP.
    """

    auth_token: SecretStr | None = None


class IndexSettings(BaseModel):
    """Indexing knobs — chunker target sizes.

    The chunker walks heading sections → paragraphs → sentences and
    accumulates paragraphs until the cumulative word-count exceeds
    `chunk_target_tokens`, then flushes a chunk and re-seeds with
    `chunk_overlap_tokens` words of tail. The "tokens" here are
    word-count (real transformer tokens are ~1.3× higher); a future
    swap to tiktoken-counted tokens is on the roadmap.

    Default 400 ≈ 520 transformer tokens; with the answer prompt's
    per-chunk 700-char truncate, 10 such chunks comfortably fit
    inside vLLM's 4096-token context. Drop to 300 for an 8 GB /
    4K-context tier; raise to 600 (the pre-P1.6 default) on rigs
    with `max-model-len >= 8192`.

    Note: changing these values changes chunk boundaries → chunk_ids
    (sha1 of chunk text) → forces a full re-embed on next index of
    any doc whose chunks shift.
    """

    chunk_target_tokens: int = Field(default=400, ge=50)
    chunk_overlap_tokens: int = Field(default=60, ge=0)


class ObservabilitySettings(BaseModel):
    """Logging and tracing — see ADR-0004.

    Defaults align with the local-first / no-telemetry vision in
    README + VISION.md: structured stdout logs are always on,
    Langfuse tracing is **opt-in**. Set both keys to enable it; if
    only one is set, startup fails loudly so we never half-configure.
    """

    log_json: bool = True
    langfuse_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    trace_sample_rate_parse: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _keys_required_when_enabled(self) -> ObservabilitySettings:
        """If Langfuse is enabled, both keys must be set. We catch this at
        config load rather than at first model call, so misconfiguration
        is a startup failure with a clear message.
        """
        if self.langfuse_enabled and (not self.langfuse_public_key or not self.langfuse_secret_key):
            raise ValueError(
                "langfuse_enabled=true but the keys are missing. Either\n"
                "  set MEMEX_OBSERVABILITY__LANGFUSE_PUBLIC_KEY and\n"
                "      MEMEX_OBSERVABILITY__LANGFUSE_SECRET_KEY in the env,\n"
                "  or write them to ~/.config/memex/config.toml under\n"
                "    [observability]\n"
                "    langfuse_enabled = true\n"
                '    langfuse_public_key = "pk-…"\n'
                '    langfuse_secret_key = "sk-…"\n'
                "  or just omit `langfuse_enabled` — it now defaults to\n"
                "  false (local-first; tracing is opt-in)."
            )
        return self


def _default_vault_path() -> Path:
    """Default vault location. `XDG_DATA_HOME` (e.g. `~/.local/share`) if
    set, otherwise `~/.memex/vault`. Either way the validator below
    creates it on first run with mode 0700, so a fresh user can just
    `memex serve web` and have everything land in a sane place."""
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "memex" / "vault"
    return Path.home() / ".memex" / "vault"


class AgentsSettings(BaseModel):
    """Answering-agent policy toggles.

    `artifact_scope_enabled` (#256): when a query NAMES a specific artifact
    ("the firewall diagram", "le diagramme de coupe-feu"), the agent resolves it
    to the document(s) it lives in and scopes retrieval there — the named
    artifact acts as an automatic doc-selection (deterministic regex + BM25, no
    LLM; see `agents/artifact_scope.py`). The normal pipeline then answers from
    the right source or refuses naturally. Conservative: queries that name no
    artifact, or whose artifact doesn't resolve confidently, take the unchanged
    full-corpus path. Kill-switch: `MEMEX_AGENTS__ARTIFACT_SCOPE_ENABLED=false`
    fully reverts to full-corpus retrieval for every query.

    `partial_grounded_answers`: when a verified draft has SOME grounded claims
    and some ungrounded ones (e.g. a compound question whose groundable half the
    corpus supports and whose other half it doesn't), and regeneration can't
    ground everything, ship the grounded SUBSET (the relevance gate still vets
    responsiveness; `compose` drops the ungrounded claims and rebuilds the
    summary from the survivors) instead of refusing the whole answer. Only a
    ZERO-grounded verdict refuses, so counterfactuals are unaffected. Kill-switch:
    `MEMEX_AGENTS__PARTIAL_GROUNDED_ANSWERS=false` restores the all-or-nothing
    behaviour (any ungrounded claim → regenerate-then-refuse).

    `report_pack_chars` / `report_coalesce_target` (ADR-0010): the `report`-detail
    granularity knobs, tunable WITHOUT code edits via the report-structure validator
    (`scripts/report_structure_audit.py`). `report_pack_chars` is the deck-packing group
    size for `report` detail ONLY (smaller → more, finer section_summaries → richer
    multi-paragraph reports; standard detail keeps the full `_MAX_SECTION_INPUT_CHARS`
    window) — clamped to `_MAX_SECTION_INPUT_CHARS` (the fast-window safe ceiling, 10,000
    RENDERED chars; the old 12k was a text-ONLY budget that ignored per-chunk wrapper
    overhead and OVERFLOWED dense decks → dropped sections, see the constant's comment).
    `report_coalesce_target` is the planner's coalesce fullness target (lower → more
    paragraphs). The defaults (pack 4,000 / coalesce 2) are the TUNED granularity winner
    from the `report-structure` sweep (2026-05-28): vs the 10k/4 corrected baseline they
    ~4× the paragraphs (2→8 on packed decks) AND raise faithfulness-confidence AND
    distinctness with 0 must-not-assert leaks — the hypothesized granularity↔repetition
    tension didn't exist (a narrower pack → tighter per-paragraph grounding). The residual
    cross-paragraph repetition is removed deterministically by the dedup gate
    (`_dedup_sentences`), NOT by backing off these knobs. See the deck-granularity tracker.
    `MEMEX_AGENTS__REPORT_PACK_CHARS=6000`.

    `graph_expansion_enabled`: the `expand_graph` node (between retrieve and rerank) pulls
    1-hop "shares-an-entity" neighbour-doc chunks into the candidate pool, per `/ask`.
    **Default OFF as of 2026-05-28** — a microscope audit proved it does real work for zero
    answer impact: an A/B (ON vs OFF, N=3) on the two cross-doc corpora (cr350-multidoc,
    ccna-multidoc) was BYTE-IDENTICAL, and a per-query trace showed the node adds 2-12
    neighbour chunks every query of which the reranker cites EXACTLY 0 — the neighbours are
    linked by GENERIC shared entities ('IP', 'HTTP', the course instructor's name), so the
    cross-encoder correctly ranks them #25-#53 of ~56 (or, on a low-signal query, ties them
    in the noise but they still go uncited). At this corpus scale (47 docs) hybrid k=50
    already has near-total recall, so there's no missed-doc for 1-hop expansion to recover
    (a large-corpus technique, premature here — like flat vector search). The graph STORE
    is kept (it's the substrate for the on-mission `related_documents` discovery feature —
    explicit, entity-SPECIFICITY-ranked, user-initiated). ANDed with the `answer_query`
    param; opt back in for a large entity-rich vault via
    `MEMEX_AGENTS__GRAPH_EXPANSION_ENABLED=true`. See [[db-audit-2026-05-28]].

    `cooccurring_min_shared_docs`: the entity-graph DISCOVERY-ranking neighbourhood FLOOR (the
    `memex entity` / `/entity` "Co-occurring concepts" list; ADR-0011). The
    `_RELATED_GENERIC_ENTITY_DF_FRACTION` (0.6) gate catches near-universal entities; this
    floor catches the OTHER bulk — a co-entity sharing only 1 doc with the seed is an
    incidental single-doc co-mention (~69% of the noise — port/PID numbers, sizes — sat at
    `shared_docs=1`), so requiring ≥2 keeps "what RECURS alongside X". Corpus-AGNOSTIC,
    structural, zero per-corpus tuning; tunable to 1. Read-only discovery (HARD-gate-neutral).
    (The curated by-name `entity_stopwords` band-aid was REMOVED 2026-05-29: a hand-curated,
    per-corpus name list — e.g. one user's `CR350` course code — doesn't generalise to a
    local-first app run on arbitrary corpora, and the OTTER NER backend types entities cleanly
    upstream, so it no longer earns its keep. See [[bert-ner-enrich-scope-2026-05-28]].)
    """

    artifact_scope_enabled: bool = True
    partial_grounded_answers: bool = True
    graph_expansion_enabled: bool = False
    # The deterministic numeric-grounding backstop (2026-05-31): a post-verify
    # gate that demotes a grounded claim whose principal LARGE figure is absent
    # from its cited TABLE chunk (a computed aggregate the LLM verifier
    # rubber-stamps via the "literal table-row reading" loophole). Fixes the
    # verify_grounding aggregate-numeric FALSE-POSITIVE that regressed the 10-K
    # (annual-report-16). Demotion-only ⇒ HARD-gate-safe; default on, fail-open.
    numeric_grounding_backstop_enabled: bool = True
    # Index-time column UNDER-SPLIT recovery (2026-05-31): split a Docling-MERGED
    # table column (a >=2-bold-group header over K>=2 clean number-runs, e.g. the
    # 10-K "Stock Awards ($) Total ($)" / "278,809 342,559") back into K columns
    # in extract_tables so Table-RAG can query the Total column (ar-15) and the
    # synthetic chunk renders ungarbled rows (ar-14). Validated 0-false-split on
    # the 47-doc vault; fail-open default True (a flip reverts on `reindex --force`).
    table_column_split_enabled: bool = True
    report_pack_chars: int = 4_000
    report_coalesce_target: int = Field(default=2, ge=1)
    cooccurring_min_shared_docs: int = Field(default=2, ge=1)

    # OTTER NER backend — the gated BERT-NER enrich swap ([[bert-ner-enrich-scope-2026-05-28]]).
    # `enrich_ner_backend="otter"` replaces the LLM (Qwen3-8B) entity extractor with the span
    # NER `whoisjones/otter-bi-mmbert` at enrich (citations stay on the LLM); default "llm" =
    # unchanged behaviour. The A/B-validated operating config is threshold 0.05 + "union"
    # labels (+103% `related_documents` discovery on the 47-doc vault, cleaner typing; see
    # `scripts/entity_ner_ab_audit.py`). CPU by default; "cuda" is viable during the CLI
    # enrich's pause-vLLM window. `enrich_ner_labels`: generic|domain|union (union = winner).
    enrich_ner_backend: Literal["llm", "otter"] = "llm"
    enrich_ner_model: str = "whoisjones/otter-bi-mmbert"
    enrich_ner_device: Literal["cuda", "cpu"] = "cpu"
    enrich_ner_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    enrich_ner_labels: Literal["generic", "domain", "union"] = "union"
    enrich_ner_max_seq_length: int = Field(default=512, ge=64)


class MemexSettings(BaseSettings):
    """Top-level settings. Construct once at startup; treat as immutable."""

    vault_path: Path = Field(default_factory=_default_vault_path)
    models: ModelSettings = Field(default_factory=ModelSettings)
    hardware: HardwareSettings = Field(default_factory=HardwareSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    parse: ParseSettings = Field(default_factory=ParseSettings)
    index: IndexSettings = Field(default_factory=IndexSettings)
    mcp: McpSettings = Field(default_factory=McpSettings)
    agents: AgentsSettings = Field(default_factory=AgentsSettings)

    model_config = SettingsConfigDict(
        toml_file=str(Path.home() / ".config" / "memex" / "config.toml"),
        env_prefix="MEMEX_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    @field_validator("vault_path", mode="after")
    @classmethod
    def _vault_path_must_be_usable(cls, v: Path) -> Path:
        """The vault path must exist (or be creatable) and be writable.
        Per GUIDELINES.md Part VI, the vault directory is `0700`.
        """
        v = v.expanduser().resolve()
        try:
            v.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError as e:
            raise ValueError(f"vault_path {v!s} is not creatable: {e}") from e
        if not os.access(v, os.W_OK):
            raise ValueError(f"vault_path {v!s} exists but is not writable.")
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Add the TOML source. Without this override pydantic-settings
        declares the `toml_file` in `model_config` but never reads it.

        Precedence (highest first): init kwargs > env vars > .env file >
        TOML file > file-based secrets.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


_SETTINGS: MemexSettings | None = None


def get_settings() -> MemexSettings:
    """Return the process settings. Configured at startup via `set_settings`."""
    if _SETTINGS is None:
        from memex.core.errors import ConfigurationError

        raise ConfigurationError(
            "MemexSettings is not initialised",
            context={
                "fix": (
                    "call set_settings() from the entry point (typically `cli.bootstrap.bootstrap`)"
                ),
            },
        )
    return _SETTINGS


def set_settings(settings: MemexSettings | None) -> None:
    """Install the process settings. Tests pass None to detach."""
    global _SETTINGS
    _SETTINGS = settings


def config_toml_path() -> Path:
    """The `config.toml` path `MemexSettings` loads from (the `toml_file` in
    `model_config`). Used by `memex mode set` to tell the user where to persist
    a chosen mode."""
    raw = MemexSettings.model_config.get("toml_file")
    if isinstance(raw, str):
        return Path(raw)
    return Path.home() / ".config" / "memex" / "config.toml"
