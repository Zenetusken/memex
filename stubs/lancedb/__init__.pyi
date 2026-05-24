"""Minimal type stubs for the async surface of `lancedb` used by
`memex.index.vector_store`.

lancedb ships no `py.typed`, so pyright treats every symbol as Unknown
under strict mode. This stub covers ONLY what `vector_store.py` touches:

- `connect_async` → `AsyncConnection`
- the async connection's `list_tables` / `create_table` / `open_table`
- the async table's `add` / `delete` / `update` / `count_rows` / `search`
- the async query builder's `limit` / `where` / `to_pydantic`

`to_pydantic` is generic over the `LanceModel` subclass so callers get
back the concrete row type they asked for. The full runtime API is far
larger; intentionally omitted to keep the stub honest about the surface
this codebase depends on.
"""

from typing import TypeVar

from lancedb.pydantic import LanceModel

_M = TypeVar("_M", bound=LanceModel)

class ListTablesResponse:
    tables: list[str]
    page_token: str | None

class AsyncQueryBuilder:
    def limit(self, limit: int) -> AsyncQueryBuilder: ...
    def where(self, predicate: str) -> AsyncQueryBuilder: ...
    async def to_pydantic(self, model: type[_M]) -> list[_M]: ...

class AsyncTable:
    async def add(self, data: object) -> object: ...
    async def delete(self, where: str) -> object: ...
    async def update(
        self,
        updates: dict[str, object] | None = ...,
        *,
        where: str | None = ...,
    ) -> object: ...
    async def count_rows(self, filter: str | None = ...) -> int: ...
    async def search(self, query: object = ...) -> AsyncQueryBuilder: ...

class AsyncConnection:
    async def list_tables(self) -> ListTablesResponse: ...
    async def create_table(
        self,
        name: str,
        data: object | None = ...,
        schema: object | None = ...,
    ) -> AsyncTable: ...
    async def open_table(self, name: str) -> AsyncTable: ...

async def connect_async(uri: str) -> AsyncConnection: ...
