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


class ModelSettings(BaseModel):
    orchestrator: str = "Qwen/Qwen3-8B-AWQ"
    # vLLM's GGUF path is flagged experimental; AWQ/GPTQ are the production
    # path on Ada (see ADR-0001 Revisit + ADR-0006). Default matches the
    # `Qwen/Qwen3-8B-AWQ` model id above. The Q*_K_M variants are kept in
    # the literal for users running gguf-via-vLLM experimentally.
    orchestrator_quantization: Literal[
        "AWQ", "GPTQ", "Q4_K_M", "Q5_K_M", "Q8_0"
    ] = "AWQ"
    # Default VLM is the AWQ-Int4 build that fits 12 GB on the reference
    # rig. Qwen3-VL-8B is the eval-gated successor (see ADR-0001 Revisit);
    # swap the string + adjust vlm_quantization in tandem.
    vlm: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    vlm_quantization: Literal["awq_int4", "bf16"] = "awq_int4"
    embedder: str = "google/embeddinggemma-300m"
    reranker: str = "BAAI/bge-reranker-v2-m3"
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


class HardwareSettings(BaseModel):
    gpu_memory_fraction: float = Field(default=0.85, ge=0.0, le=1.0)
    max_concurrent_documents: int = Field(default=2, ge=1)
    cpu_workers: int = Field(
        default_factory=lambda: max(1, (os.cpu_count() or 2) - 1),
        ge=1,
    )


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


class IngestSettings(BaseModel):
    """Ingestion validation knobs — see GUIDELINES.md Part VI security."""

    max_bytes: int = Field(default=256 * 1024 * 1024, ge=1024)
    allow_macros: bool = False  # macro-bearing Office docs are rejected by default


class ParseSettings(BaseModel):
    """Parser routing knobs — see IMPLEMENTATION-PLAN §1.3."""

    vlm_confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    # Default-off: the VLM escalation path is fully wired, but it
    # demands ~5 GB of VRAM (Qwen2.5-VL-7B AWQ-Int4 + processor) on top
    # of the embedder + reranker resident set. Opt in once you've
    # confirmed your rig has the headroom; bootstrap's VRAM-fit check
    # will refuse to load it otherwise.
    disable_vlm: bool = True
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
    pymupdf_mixed_content_image_area_threshold: float = Field(
        default=0.35, ge=0.0, le=1.0
    )
    # Minimum fraction of pages that must individually be image-heavy
    # (>3 images per page) for the mixed-content classification to
    # fire. 0.30 = 30% of pages must independently flag image-heavy
    # before we force-OCR. Both gates must pass (AND) — protects
    # against false-positives on text docs with a single front-cover
    # image, and on decorative-heavy decks whose image-text is noise.
    pymupdf_mixed_content_min_image_heavy_pages: float = Field(
        default=0.30, ge=0.0, le=1.0
    )
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
    pymupdf_slide_deck_aspect_threshold: float = Field(
        default=1.3, ge=1.0, le=3.0
    )
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
    pymupdf_slide_deck_chart_heavy_image_area_threshold: float = Field(
        default=0.20, ge=0.0, le=1.0
    )
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
        if self.langfuse_enabled and (
            not self.langfuse_public_key or not self.langfuse_secret_key
        ):
            raise ValueError(
                "langfuse_enabled=true but the keys are missing. Either\n"
                "  set MEMEX_OBSERVABILITY__LANGFUSE_PUBLIC_KEY and\n"
                "      MEMEX_OBSERVABILITY__LANGFUSE_SECRET_KEY in the env,\n"
                "  or write them to ~/.config/memex/config.toml under\n"
                "    [observability]\n"
                "    langfuse_enabled = true\n"
                "    langfuse_public_key = \"pk-…\"\n"
                "    langfuse_secret_key = \"sk-…\"\n"
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


class MemexSettings(BaseSettings):
    """Top-level settings. Construct once at startup; treat as immutable."""

    vault_path: Path = Field(default_factory=_default_vault_path)
    models: ModelSettings = Field(default_factory=ModelSettings)
    hardware: HardwareSettings = Field(default_factory=HardwareSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    observability: ObservabilitySettings = Field(
        default_factory=ObservabilitySettings
    )
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    parse: ParseSettings = Field(default_factory=ParseSettings)
    index: IndexSettings = Field(default_factory=IndexSettings)
    mcp: McpSettings = Field(default_factory=McpSettings)

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
            raise ValueError(
                f"vault_path {v!s} is not creatable: {e}"
            ) from e
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
                    "call set_settings() from the entry point "
                    "(typically `cli.bootstrap.bootstrap`)"
                ),
            },
        )
    return _SETTINGS


def set_settings(settings: MemexSettings | None) -> None:
    """Install the process settings. Tests pass None to detach."""
    global _SETTINGS
    _SETTINGS = settings
