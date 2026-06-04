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
    # Time range (audio transcripts only) — (start_s, end_s) seconds in the source file, the
    # audio analogue of `page`; attributed by the chunker from `TranscriptSegment` spans
    # (ADR-0017). None on the doc/PDF paths. Navigation-grade, not citation-grade.
    time_range: tuple[float, float] | None = None
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


class RelatedDocument(BaseModel):
    """A document related to a seed doc via SHARED ENTITIES, ranked by the SPECIFICITY of
    those entities (IDF — a rare shared entity is a strong topical signal; a near-universal
    one is noise). The on-mission "explore connections" discovery surface, vs the retired
    passive `expand_graph` which linked on generic entities, unranked, and never helped.

    Lives in `core/types` (not `index/graph_store`) because it crosses module boundaries —
    produced by `index/graph_store::related_documents`, aggregated by `retrieve/related`, and
    surfaced on `FinalResponse` for MCP/CLI/webui parity. Re-exported from `index.graph_store`
    for back-compat."""

    doc_id: str
    title: str
    score: float  # Σ IDF(entity) over the shared, non-generic entities — higher = stronger
    shared_entities: list[str]  # the connecting entities, most-specific first


class GpuProcess(BaseModel):
    """One process holding GPU memory, as reported by `nvidia-smi` (the structured probe
    `core/vram.gpu_processes`). Lives in `core/types` because it crosses the boundary from
    `core/vram` (the torch/nvidia-smi probe) to the webui `/resources` VRAM panel + the CLI.
    Infrastructure, not domain state — a live snapshot, never persisted or grounded on."""

    pid: int
    name: str  # the process name nvidia-smi reports (e.g. "VLLM::EngineCore", "python3")
    used_mib: int  # GPU memory this process holds, in MiB


class AlignmentBlock(BaseModel):
    """One transcript chunk aligned to its best-matching slide-deck PAGE (ADR-0018 companion-merge).
    PRIMARY identity = the two content-addressed chunk_ids; `deck_page`/`time_range` are CACHED
    navigation hints for the webui label (re-derivable). `deck_chunk_id is None` ⇒ a NULL block (the
    chunk matched no slide above the score floor — an off-slide tangent). Derived sidecar state;
    navigation/discovery-grade, NEVER read on the grounding path. Lives in `core/types` because it
    crosses boundaries (`index/companion` writes; the `augment_companion` node + webui read)."""

    transcript_chunk_id: str
    time_range: tuple[float, float] | None = (
        None  # the transcript chunk's audio anchor (cached label)
    )
    deck_chunk_id: str | None = None  # None ⇒ NULL: no slide above the floor
    deck_page: int | None = None  # cached nav hint = the aligned deck chunk's Chunk.page
    score: float  # cosine similarity of the alignment (≈0..1 for L2-normalized embeddings)


class CompanionAlignment(BaseModel):
    """The per-pair companion-merge alignment (ADR-0018): every chunk of a lecture TRANSCRIPT doc
    aligned to a SLIDE-DECK doc's page. A DERIVED, regenerable sidecar
    (`vault/.memex/companion_alignments.json`); HARD-gate-neutral discovery/navigation state — the
    transcript + deck remain first-class grounded docs in their own right."""

    transcript_doc: str
    deck_doc: str
    embedding_recipe_version: str = ""
    blocks: list[AlignmentBlock] = Field(default_factory=list[AlignmentBlock])
    null_count: int = 0


class BridgeDoc(BaseModel):
    """One related document reached THROUGH a bridging entity — a row under a `DocumentBridge`.
    Carries the doc's OVERALL relatedness `score` (identical to its `RelatedDocument.score`, so
    the two graph lenses agree) and `via_entities`: the doc's OTHER connecting entities besides
    the bridge's own (the "·via X, Y" secondary tags), most-specific first."""

    doc_id: str
    title: str
    score: float
    via_entities: list[str]


class DocumentBridge(BaseModel):
    """A shared ENTITY that bridges a seed document to one or more related documents — the
    entity-grouped lens on the neighbourhood (the /graph "Bridges" view). Inverts
    `RelatedDocument`'s doc-grouping: instead of "which docs are related" it answers "which
    CONCEPTS connect this doc, and to what". Ranked by `strength` = mean per-edge IDF×kind_weight
    × ln(1 + doc_count) — per-edge SPECIFICITY dominates, with fan-out entering sub-linearly so a
    near-generic entity shared by many docs doesn't bury the specific ones. `doc_count` is the
    literal "bridges N" (how many related docs share it).

    Lives in `core/types` alongside `RelatedDocument` — produced by
    `index/graph_store::related_bridges`, surfaced by the webui. Not on `FinalResponse` (a
    pure discovery surface, never on the answer path)."""

    entity: str
    kind: str
    doc_count: int  # how many related docs share this entity with the seed ("bridges N")
    strength: float  # Σ IDF×kind_weight over its (doc, entity) links — the ranking key
    docs: list[BridgeDoc]  # the reached docs, strongest-first (by each doc's overall score)


class ConversationTurn(BaseModel):
    """One turn of a grounded multi-turn chat (Surface A — `docs/specs/grounded-agentic-chat.md`).

    Persisted in the conversation sqlite sidecar (`core/conversation_store.py`) and rehydrated by
    the CLI/webui surfaces. Lives in `core/types` because it crosses `core/` ↔ `agents/` ↔ `webui/`.

    `cited_chunk_ids` are chunk IDs ONLY (never chunk text) — the bounded prior-chunk carry re-fetches
    them live from the FTS store, and they can never become grounding evidence. `response_json` is an
    OPAQUE `FinalResponse.model_dump_json()` (core/ must not import `agents/FinalResponse`); the surface
    parses it to re-render a resumed thread. `created_at` is ISO-8601 UTC.
    """

    turn_id: str
    conversation_id: str
    turn_index: int
    user_text: str
    standalone_query: str
    is_followup: bool = False
    answered: bool = False
    answer_summary: str = ""
    cited_chunk_ids: list[str] = Field(default_factory=list)
    response_json: str | None = None
    correlation_id: str | None = None
    created_at: str = ""


class Conversation(BaseModel):
    """A grounded multi-turn chat thread (Surface A). Persisted in the conversation sqlite sidecar;
    **USER DATA** — deliberately excluded from the `reindex_vault(force=True)` teardown (the
    `scope_sets.json` precedent). `running_summary` is the compacted history of turns older than the
    verbatim window (the digest); `scope_doc_ids` is the per-conversation scope pin (set on turn 1).
    `turns` is the hydrated turn list (empty until loaded with turns)."""

    conversation_id: str
    title: str = ""
    created_at: str = ""
    updated_at: str = ""
    turn_count: int = 0
    running_summary: str = ""
    scope_doc_ids: list[str] = Field(default_factory=list)
    turns: list[ConversationTurn] = []
