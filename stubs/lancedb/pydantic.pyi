"""Minimal type stubs for `lancedb.pydantic` — the `LanceModel` base
and `Vector` field factory used by `memex.index.vector_store._ChunkRow`.

`LanceModel` subclasses `pydantic.BaseModel` at runtime (it adds Arrow
schema derivation), so the stub inherits from `BaseModel` to give
subclasses real pydantic field typing. `Vector(dim)` returns a type
suitable as a field annotation; the runtime object is a dynamic
`FixedSizeList` pydantic type, so it is typed as `type` here and the
call site keeps its `# type: ignore[valid-type]`.
"""

from pydantic import BaseModel

class LanceModel(BaseModel): ...

def Vector(dim: int, value_type: object = ...) -> type: ...
