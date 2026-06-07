"""Model loading, the vLLM client wrapper, and VRAM management.

See GUIDELINES.md Part III "The model stack and VRAM budget" and
ADR-0001 (vLLM as the sole inference engine).
"""

from memex.models.client import (
    complete_reasoning,
    complete_structured,
    configure_client,
    get_client,
    split_think,
)
from memex.models.download import (
    ModelTarget,
    format_report,
    model_cache_status,
    resolve_model_targets,
    run_download,
)
from memex.models.registry import (
    ModelHandle,
    ModelName,
    ModelRegistry,
    get_registry,
    set_registry,
)

__all__ = [
    "ModelHandle",
    "ModelName",
    "ModelRegistry",
    "ModelTarget",
    "complete_reasoning",
    "complete_structured",
    "configure_client",
    "format_report",
    "get_client",
    "get_registry",
    "model_cache_status",
    "resolve_model_targets",
    "run_download",
    "set_registry",
    "split_think",
]
