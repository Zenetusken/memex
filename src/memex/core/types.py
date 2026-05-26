"""Shared cross-module types.

When a module needs to expose a type that other modules import, it
goes here rather than into the producing module — keeps the import
graph acyclic and explicit. Module-internal types stay in their own
module.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Canonical set of DATA-chart PictureClassifier class names (docling_core's
# snake_case labels for bar/line/pie/scatter/etc.). These figures are
# chart-OCR's domain. The parse worker excludes them from the VLM image-area
# escalation (`docling_worker._CHART_CLASSES`, kept as a sandbox-local copy
# pinned equal to this by a unit test); chart_ocr_backend may adopt this too.
# NB diagram classes (flow_chart / engineering_drawing / screenshot) are NOT
# here — those are the VLM's domain (the classification-escalation arm).
CHART_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "bar_chart",
        "line_chart",
        "pie_chart",
        "scatter_chart",
        "box_plot",
        "stacked_bar_chart",
        "heatmap",
        "stratigraphic_chart",
    }
)


class Chunk(BaseModel):
    """A retrievable unit of document content with citation metadata.

    The same shape is produced by `memex.index` (when chunking a
    document for storage), consumed by `memex.retrieve` (BM25 + dense
    candidates + reranked results), and threaded through the answering
    agent's state. Lives here so neither side has to import from the
    other.
    """

    chunk_id: str
    document_id: str
    document_title: str
    text: str
    page: int | None = None
    char_start: int = 0
    char_end: int = 0
    score: float = 0.0
    rerank_score: float | None = None
    heading_path: list[str] = Field(default_factory=list)


class StoredTable(BaseModel):
    """A well-formed GFM table extracted from a document body and persisted
    in the per-vault table store (`index/table_store.py`, Table-RAG Phase 2).

    `header` + `rows` carry the cell text VERBATIM (the same cleaned text the
    Phase-1 linearizer emits — whitespace collapsed, trailing footnote markers
    stripped, otherwise unchanged). `char_start`/`char_end` are the table's
    span in the source `.md` so an answer grounded in this table can cite the
    source. Produced by `index.table_store.extract_tables`, consumed by the
    `agents.table_sql` text-to-SQL helper.
    """

    doc_id: str
    table_id: str
    section: str
    header: list[str]
    rows: list[list[str]]
    char_start: int
    char_end: int


class TableQueryResult(BaseModel):
    """The result of a single text-to-SQL query over one document's tables
    (`agents.table_sql.query_doc_tables`, Table-RAG Phase 2).

    `kind="rows"` carries verbatim document cells the SQL selected (no new
    number); `kind="aggregate"` carries a single recomputed-and-agreeing scalar
    in `aggregate_value`. `contributing_rows` are the source rows the result is
    built from (rendered first in the synthetic chunk so the evidence survives
    the answer/verify truncation budgets). The citation fields mirror the source
    table's so the synthetic chunk cites the real document span.

    `superlative`, when set on a `kind="rows"` result, is `(column_label,
    "highest"|"lowest")` — meaning the returned row was INDEPENDENTLY verified
    (Python recompute over the source column) to hold the extremum of that
    column. The synthetic chunk frames it as such so the agent can attribute the
    superlative confidently; it's only set when the verification agrees, so the
    framing is grounded, not an unchecked claim.
    """

    kind: Literal["rows", "aggregate"]
    sql: str
    target_table_id: str
    rows: list[list[str]] | None = None
    aggregate_value: float | None = None
    contributing_rows: list[list[str]]
    header: list[str]
    char_start: int
    char_end: int
    doc_id: str
    document_title: str
    heading_path: list[str]
    section: str
    superlative: tuple[str, str] | None = None


class GeneratedSQL(BaseModel):
    """The LLM-emitted text-to-SQL output (`agents.table_sql`, Phase 2).

    Both fields are bounded (`max_length`) per the backend convention: xgrammar
    enforces the bound at the grammar level so the model cannot emit a runaway
    string that trips `max_tokens` mid-emission and crashes JSON validation.
    A SELECT over a single small table fits comfortably under 600 chars; a
    `table_id` is a 10-hex-char sha1 prefix (16 is generous headroom).
    """

    sql: str = Field(max_length=600)
    target_table_id: str = Field(max_length=16)
