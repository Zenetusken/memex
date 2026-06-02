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
    "complete_reasoning",
    "complete_structured",
    "configure_client",
    "get_client",
    "get_registry",
    "set_registry",
    "split_think",
]
