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

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class ModelSettings(BaseModel):
    orchestrator: str = "Qwen/Qwen3-8B-Instruct"
    # vLLM's GGUF path is flagged experimental; AWQ/GPTQ are the production
    # path on Ada (see ADR-0001 Revisit + ADR-0006). The literal is open
    # for that swap — change the orchestrator string in tandem.
    orchestrator_quantization: Literal[
        "Q4_K_M", "Q5_K_M", "Q8_0", "AWQ", "GPTQ"
    ] = "Q4_K_M"
    # Default VLM is the AWQ-Int4 build that fits 12 GB on the reference
    # rig. Qwen3-VL-8B is the eval-gated successor (see ADR-0001 Revisit);
    # swap the string + adjust vlm_quantization in tandem.
    vlm: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    vlm_quantization: Literal["awq_int4", "bf16"] = "awq_int4"
    embedder: str = "google/embeddinggemma-300m"
    reranker: str = "BAAI/bge-reranker-v2-m3"


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
    docling_timeout_s: int = Field(default=300, ge=10)
    docling_crash_threshold: int = Field(default=5, ge=1)
    # Network-egress sandbox for the Docling worker. When True, the
    # worker installs a seccomp filter blocking all network syscalls
    # before importing docling — see memex/parse/sandbox.py and
    # GUIDELINES.md Part VI. Linux-only (gracefully no-op elsewhere).
    # Set False only for deployments that genuinely need network during
    # parse; the docs recommend pre-fetching models with
    # `huggingface-cli download` instead.
    docling_sandbox_network: bool = True


class ObservabilitySettings(BaseModel):
    """Logging and tracing — see ADR-0004."""

    log_json: bool = True
    langfuse_enabled: bool = True
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
                "langfuse_enabled=true requires langfuse_public_key and "
                "langfuse_secret_key. Set them, or set langfuse_enabled=false."
            )
        return self


class MemexSettings(BaseSettings):
    """Top-level settings. Construct once at startup; treat as immutable."""

    vault_path: Path
    models: ModelSettings = Field(default_factory=ModelSettings)
    hardware: HardwareSettings = Field(default_factory=HardwareSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    observability: ObservabilitySettings = Field(
        default_factory=ObservabilitySettings
    )
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    parse: ParseSettings = Field(default_factory=ParseSettings)

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
