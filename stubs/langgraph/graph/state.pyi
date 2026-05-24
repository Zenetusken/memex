from collections.abc import Callable, Hashable
from typing import Any, Generic, TypeVar

_S = TypeVar("_S")

# Non-generic: callers only `.ainvoke()` it (→ the merged state dict);
# the state type param is unused at the call sites, and keeping it
# non-generic spares every annotation a `[State]` argument.
class CompiledStateGraph:
    async def ainvoke(
        self, input: Any, config: Any = ..., **kwargs: Any
    ) -> dict[str, Any]: ...

class StateGraph(Generic[_S]):
    def __init__(
        self, state_schema: type[_S], *args: Any, **kwargs: Any
    ) -> None: ...
    def add_node(
        self, node: str, action: Callable[..., Any], **kwargs: Any
    ) -> Any: ...
    def add_edge(self, start_key: str, end_key: str) -> Any: ...
    def add_conditional_edges(
        self,
        source: str,
        path: Callable[..., Hashable],
        path_map: dict[Any, str] | list[str] | None = ...,
        **kwargs: Any,
    ) -> Any: ...
    def compile(self, *args: Any, **kwargs: Any) -> CompiledStateGraph: ...
