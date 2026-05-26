"""VLM fallback for pages Docling can't handle confidently.

Renders the source PDF page to an image (via pypdfium2) and passes it
through the VLM loaded by `ModelRegistry.use("vlm")` with a strict
"transcribe as clean markdown" instruction. The result replaces
Docling's output for that page; the manifest records the routing
decision either way.

Heavy deps (pypdfium2, transformers, torch) are imported inside the
functions so the rest of the package stays importable without the
[parse] + [models] extras.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import structlog

from memex.core.config import get_settings
from memex.core.errors import MemexError
from memex.models.registry import VLMHandle, get_registry
from memex.parse.docling_backend import DoclingPageOutput

if TYPE_CHECKING:
    from PIL import Image

    from memex.parse.vlm_cache import VLMTranscriptionCache

logger = structlog.get_logger(__name__)

_PROMPT = (
    "You are converting a single document page to clean Markdown. "
    "Preserve structure:\n"
    "- Headings (#, ##, ###)\n"
    "- Tables (GFM table syntax)\n"
    "- Bulleted and numbered lists\n"
    "- Equations as LaTeX ($inline$ or $$display$$)\n"
    "- Code blocks (```fenced)\n"
    "- Diagrams, flowcharts, network/architecture diagrams and figures: "
    "transcribe their content as text — every label and node name, plus "
    "the connections or flow between them (e.g. 'Router -> Firewall -> "
    "Private Network'). Describe what the figure shows; do NOT emit an "
    "image placeholder like ![...].\n\n"
    "Output ONLY Markdown for the page contents — no preface, "
    "no commentary, no closing remarks."
)
# NB: a more forceful "you MUST transcribe / NEVER emit an image link"
# rewrite was tried 2026-05-25 and REVERTED — it didn't rescue the one
# diagram the model punts on (page-26 zoning) and regressed the others
# (bulleted the firewall chains, code-fenced + mis-hyphenated the 802.1X
# sequence "EAPOL"→"EAPO-L"). This wording + the `_strip_image_links`
# safety net below is the validated combination. The model still
# occasionally punts a hard spatial diagram; the strip removes the
# resulting broken `![...]()` so it never reaches the vault.

# The VLM transcribes a rendered PAGE image, so any markdown image link it
# emits points at a file that does not exist — pure noise, and a broken
# link in the rendered doc. The prompt forbids them; this is the
# deterministic safety net (the model still occasionally punts a hard
# diagram with `![...](...)` rather than describing it).
_IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def _strip_image_links(markdown: str) -> str:
    """Remove spurious markdown image links from VLM output, then collapse
    the blank line a removed standalone-image line leaves behind."""
    cleaned = _IMAGE_LINK_RE.sub("", markdown)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# A single page's transcribed Markdown is typically 500–800 tokens; 1024
# leaves headroom for table/code-heavy pages and defends against runaway
# generation. Tuned per the CUDA audit.
_MAX_NEW_TOKENS = 1024

# Don't cache a near-empty transcription — the VLM occasionally punts a hard
# diagram with a few chars (or just a stripped image link). Leaving it
# uncached lets the NEXT parse retry rather than freezing a bad draw.
_MIN_CACHEABLE_CHARS = 20


class VLMUnavailable(MemexError):
    """The VLM stack is not installed (transformers / torch / pypdfium2)."""


class PDFRenderError(MemexError):
    """The page could not be rasterised to an image."""


def _render_page_to_image(pdf_path: Path, page_number: int) -> Image.Image:
    """Render a single 1-indexed page to a PIL Image at ~144 DPI.

    Qwen-VL processors use `max_pixels = 1280 * 28 * 28 ≈ 1.0 M` and
    downscale anything bigger. Rendering at scale=2.0 (~144 DPI ≈ 1.9 M
    pixels for letter pages) is half the rasterisation work of the
    previous scale=2.78 (~200 DPI ≈ 3.7 M pixels) while still leaving
    the processor a clean resize step. CUDA audit, item 9.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as e:
        raise VLMUnavailable(
            "pypdfium2 is required for VLM page rasterisation; "
            "install with `uv sync --extra parse`",
            context={"underlying": str(e)},
        ) from e

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        idx = page_number - 1
        if not 0 <= idx < len(doc):
            raise PDFRenderError(
                f"page {page_number} out of range",
                context={"page": page_number, "page_count": len(doc)},
            )
        page = doc[idx]
        bitmap = page.render(scale=2.0)
        # N7 (audit 2026-05-20): pypdfium2's `bitmap.to_pil()` returns
        # a PIL.Image.Image whose pixel buffer is a VIEW into the
        # bitmap's C-owned memory. When the `finally` block calls
        # `doc.close()`, that memory is freed; any subsequent access
        # to the returned image (e.g. `image.size`, `image.save`,
        # processor preprocessing) reads freed memory — silent
        # corruption or segfault depending on the libc. The `.copy()`
        # forces PIL to allocate its own buffer and memcpy the bytes
        # over BEFORE the doc closes, decoupling lifetimes.
        return bitmap.to_pil().copy()
    finally:
        doc.close()


def _vlm_transcribe_sync(
    handle: VLMHandle, image: Image.Image, prompt: str, max_new_tokens: int
) -> str:
    """Synchronous transcription; called via asyncio.to_thread."""
    import torch

    # A VLM processor is a model-specific class returned by
    # `AutoProcessor` (e.g. `Qwen2VLProcessor`); its `apply_chat_template`
    # / `__call__` / `batch_decode` kwargs aren't on the base
    # `ProcessorMixin` stub. transformers' stub also types
    # `PreTrainedModel.generate` as a broken `Tensor | Module` union
    # (not callable). Both are genuinely-dynamic transformers boundaries,
    # so we route them through explicit `Any`.
    processor: Any = handle.processor
    model: Any = handle.model

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    # NB: forcing the deterministic SDPA *math* backend here (to steady the
    # non-deterministic greedy draw) was tried 2026-05-25 and REVERTED — on
    # the 12 GB rig the math backend materialises the full B×H×S×S attention
    # matrix for Qwen2.5-VL's ~1k+ visual tokens and CUDA-OOMs mid-generate.
    # The VLM cache (`vlm_cache.py`) is the reproducibility guarantee, and a
    # best-of-N keep-longest draw addresses completeness — neither needs the
    # math backend.
    with torch.inference_mode():
        # With the default `return_dict_in_generate=False`, `generate`
        # returns a token-id tensor; cast for the slice + decode below.
        outputs = cast(
            "torch.Tensor",
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            ),
        )

    # Strip the prompt prefix from the decode.
    generated = outputs[:, inputs["input_ids"].shape[1] :]
    decoded: list[str] = processor.batch_decode(generated, skip_special_tokens=True)
    return _strip_image_links(decoded[0] if decoded else "")


# ── vLLM-served VLM backend (parse-time, for Qwen3-VL etc.) ──────────────
# Qwen3-VL ships only compressed-tensors int4 community builds, which
# transformers can't run in-process on 12 GB (it decompresses int4→dense and
# OOMs; see ADR-0006 §VLM-via-vLLM). vLLM runs them via the Marlin int4 kernel
# at ~7.4 GB. The VLM vLLM is a SHORT-LIVED process started on the GPU freed by
# the parse-time `pause_vllm_for_gpu()` (orchestrator down) and torn down before
# `convert_pages` returns — so it never co-resides with the in-process chart-OCR
# pass that follows. Recipe + the 12 GB gotchas live in `VLMServeSettings`.


def _png_b64(image: Image.Image) -> str:
    """Encode a PIL image to a base64 PNG payload (no `data:` prefix)."""
    import base64
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def _vlm_vllm_reachable(url: str, timeout_s: float = 2.0) -> bool:
    """True iff a GET to `url` returns 2xx — the VLM vLLM readiness probe."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(url)
            return 200 <= resp.status_code < 300
    except Exception:
        return False


@asynccontextmanager
async def _serve_vlm_vllm(model_id: str) -> AsyncGenerator[str]:
    """Start a short-lived vLLM serving `model_id`, yield its base_url, and tear
    it down (process group + GPU release) on exit.

    The recipe is `VLMServeSettings` (validated on the 12 GB rig). The process
    gets its own session group so teardown's `killpg` reaps the EngineCore
    children too; readiness + GPU-release are gated on `/v1/models`. Raises
    `VLMUnavailable` if the process exits or never becomes ready.
    """
    import os
    import signal

    serve = get_settings().models.vlm_serve
    base_url = f"http://{serve.host}:{serve.port}/v1"
    log = logger.bind(component="vlm.vllm", model=model_id, port=serve.port)
    cmd = [
        "uv",
        "run",
        "--extra",
        "serve",
        "vllm",
        "serve",
        model_id,
        "--host",
        serve.host,
        "--port",
        str(serve.port),
        "--gpu-memory-utilization",
        str(serve.gpu_memory_utilization),
        "--max-model-len",
        str(serve.max_model_len),
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        str(serve.max_model_len),
        "--enforce-eager",
        "--kv-cache-dtype",
        "auto",
        "--mm-processor-kwargs",
        json.dumps({"max_pixels": serve.max_pixels, "min_pixels": serve.min_pixels}),
        "--limit-mm-per-prompt",
        json.dumps({"image": 1, "video": 0}),
    ]
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    log.info("vlm.vllm.start")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        stdin=asyncio.subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )
    try:
        ready = False
        for _ in range(serve.startup_timeout_s):
            if await _vlm_vllm_reachable(f"{base_url}/models"):
                ready = True
                break
            if proc.returncode is not None:
                raise VLMUnavailable(
                    "VLM vLLM process exited during startup",
                    context={"model": model_id, "returncode": proc.returncode},
                )
            await asyncio.sleep(1.0)
        if not ready:
            raise VLMUnavailable(
                "VLM vLLM did not become ready in time",
                context={"model": model_id, "timeout_s": serve.startup_timeout_s},
            )
        log.info("vlm.vllm.ready")
        yield base_url
    finally:
        log.info("vlm.vllm.stop")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=30.0)
        except TimeoutError:
            with _suppress_proc_lookup():
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        # Settle: wait for the port to stop answering so the GPU is actually
        # freed before the caller (chart-OCR) loads its in-process model.
        for _ in range(15):
            if not await _vlm_vllm_reachable(f"{base_url}/models", timeout_s=1.0):
                break
            await asyncio.sleep(1.0)


def _suppress_proc_lookup() -> Any:
    """`contextlib.suppress(ProcessLookupError)` without the top-level import."""
    import contextlib

    return contextlib.suppress(ProcessLookupError)


async def _vllm_transcribe(
    base_url: str, model_id: str, image: Image.Image, prompt: str, max_new_tokens: int
) -> str:
    """Transcribe one page image via the VLM vLLM OpenAI multimodal API
    (greedy: temperature 0). Mirrors `_vlm_transcribe_sync`'s prompt + strip."""
    from openai import AsyncOpenAI

    b64 = await asyncio.to_thread(_png_b64, image)
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    resp = await client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=max_new_tokens,
        temperature=0.0,
    )
    content = resp.choices[0].message.content or ""
    return _strip_image_links(content)


async def _convert_one_via_vllm(
    base_url: str,
    model_id: str,
    source_pdf: Path,
    page_number: int,
    max_new_tokens: int,
    samples: int = 1,
) -> DoclingPageOutput:
    """Render + transcribe one page via the VLM vLLM; best-of-N keep-longest
    (the same completeness proxy as the in-process `_convert_with_handle`)."""
    log = logger.bind(page=page_number, source=str(source_pdf), backend="vllm")
    n = max(1, samples)
    log.info("vlm.start", samples=n)
    image = await asyncio.to_thread(_render_page_to_image, source_pdf, page_number)
    drafts: list[str] = []
    for _ in range(n):
        drafts.append(await _vllm_transcribe(base_url, model_id, image, _PROMPT, max_new_tokens))
    markdown = max(drafts, key=len)
    log.info("vlm.done", chars=len(markdown), samples=n, draft_chars=[len(d) for d in drafts])
    return DoclingPageOutput(page=page_number, markdown=markdown, confidence=1.0)


async def _convert_with_handle(
    handle: VLMHandle,
    source_pdf: Path,
    page_number: int,
    max_new_tokens: int,
    samples: int = 1,
) -> DoclingPageOutput:
    """Internal: render + transcribe one page given an already-acquired VLM.

    Takes `samples` independent greedy draws (the VLM is non-deterministic,
    so they differ) and keeps the LONGEST — a content-completeness proxy,
    since a draw that silently drops content is shorter. `samples=1` is a
    single draw. The chosen draw is what `convert_pages` caches, so the
    completeness choice is made once and frozen reproducibly.
    """
    log = logger.bind(page=page_number, source=str(source_pdf))
    n = max(1, samples)
    log.info("vlm.start", samples=n)

    image = await asyncio.to_thread(_render_page_to_image, source_pdf, page_number)
    drafts: list[str] = []
    for _ in range(n):
        drafts.append(
            await asyncio.to_thread(_vlm_transcribe_sync, handle, image, _PROMPT, max_new_tokens)
        )
    markdown = max(drafts, key=len)

    log.info("vlm.done", chars=len(markdown), samples=n, draft_chars=[len(d) for d in drafts])
    return DoclingPageOutput(page=page_number, markdown=markdown, confidence=1.0)


async def convert_pages(
    *,
    source_pdf: Path,
    page_numbers: list[int],
    max_new_tokens: int = _MAX_NEW_TOKENS,
    cache: VLMTranscriptionCache | None = None,
    refresh_vlm: bool = False,
) -> dict[int, DoclingPageOutput | Exception]:
    """Per-document batch transcription. CUDA audit, item 8.

    With a `cache`, each page's transcription is keyed by
    `sha256(source_pdf_bytes):page:model:prompt` and reused on re-parse.
    The VLM is non-deterministic, so this makes transcription reproducible
    by construction (see `vlm_cache.py`). Pages served from cache skip the
    GPU entirely — the `registry.use("vlm")` acquisition (and its load) only
    happens if at least one page misses. `refresh_vlm` busts this document's
    cached pages first (force a fresh draw). `cache=None` ⇒ pre-cache
    behaviour, unchanged.

    Per-page failures are returned as exception objects in the result dict;
    the caller routes each to an escalated `PageDecision(engine="vlm")` or a
    fallback `PageDecision(engine="docling", rationale="...")`.

    Acquiring the VLM once for all misses (not once per page): the
    `ModelRegistry` keeps the VLM resident after first load, so per-page
    calls don't reload — but once the registry grows OOM-driven eviction,
    per-page acquisition would thrash. Once per document is correct.
    """
    if not page_numbers:
        return {}

    settings = get_settings()
    samples = settings.parse.vlm_transcription_samples
    results: dict[int, DoclingPageOutput | Exception] = {}

    # Cache-key components — computed once per document. Hashing the source
    # PDF bytes (not the rendered image) is content-true and needs no render
    # to check the cache.
    keys: dict[int, str] = {}
    pdf_sha256 = ""
    prompt_sha8 = ""
    vlm_model = ""
    if cache is not None:
        pdf_sha256 = hashlib.sha256(source_pdf.read_bytes()).hexdigest()
        prompt_sha8 = hashlib.sha256(_PROMPT.encode()).hexdigest()[:8]
        vlm_model = settings.models.vlm
        if refresh_vlm:
            deleted = await cache.delete_by_pdf(pdf_sha256)
            logger.info("vlm.cache_refresh", source=str(source_pdf), deleted=deleted)
        keys = {p: f"{pdf_sha256}:{p}:m={vlm_model}:p={prompt_sha8}" for p in page_numbers}

    # First pass: serve cache hits, collect misses (no GPU touched yet).
    misses: list[int] = []
    for page_number in page_numbers:
        if cache is not None:
            hit = await cache.get(keys[page_number])
            if hit is not None:
                logger.info("vlm.cache_hit", page=page_number, source=str(source_pdf))
                results[page_number] = DoclingPageOutput(
                    page=page_number, markdown=hit, confidence=1.0
                )
                continue
        misses.append(page_number)

    if not misses:
        return results  # everything served from cache — no VLM load

    # Second pass: transcribe the misses. Either via a short-lived VLM vLLM
    # process (Qwen3-VL — no in-process int4 kernel for its compressed-tensors
    # build; ADR-0006 §VLM-via-vLLM) or in-process via the registry (the legacy
    # AutoAWQ Qwen2.5-VL path). The chosen draw is cached either way.
    if settings.models.vlm_serving == "vllm":
        model_id = settings.models.vlm
        async with _serve_vlm_vllm(model_id) as base_url:
            for page_number in misses:
                try:
                    output = await _convert_one_via_vllm(
                        base_url, model_id, source_pdf, page_number, max_new_tokens, samples
                    )
                except (VLMUnavailable, PDFRenderError) as e:
                    results[page_number] = e
                    continue
                results[page_number] = output
                if cache is not None and len(output.markdown.strip()) >= _MIN_CACHEABLE_CHARS:
                    await cache.put(
                        keys[page_number],
                        pdf_sha256=pdf_sha256,
                        page_no=page_number,
                        vlm_model=vlm_model,
                        prompt_sha8=prompt_sha8,
                        markdown=output.markdown,
                    )
        return results

    # In-process (transformers) path: acquire the VLM once for all misses.
    registry = get_registry()
    async with registry.use("vlm") as handle:
        for page_number in misses:
            try:
                output = await _convert_with_handle(
                    handle, source_pdf, page_number, max_new_tokens, samples=samples
                )
            except (VLMUnavailable, PDFRenderError) as e:
                results[page_number] = e
                continue
            results[page_number] = output
            if cache is not None and len(output.markdown.strip()) >= _MIN_CACHEABLE_CHARS:
                await cache.put(
                    keys[page_number],
                    pdf_sha256=pdf_sha256,
                    page_no=page_number,
                    vlm_model=vlm_model,
                    prompt_sha8=prompt_sha8,
                    markdown=output.markdown,
                )
    return results
