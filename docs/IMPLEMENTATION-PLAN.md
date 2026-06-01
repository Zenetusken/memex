# Memex Implementation Plan

A long-form architecture plan that takes Memex from "scaffolded skeleton + working answering agent" to "working end-to-end pipeline." References ADRs 0001–0004 and the seven parts of `docs/GUIDELINES.md`.

---

## 1. Module-by-Module Implementation Plan

### 1.1 `memex.vault` — Markdown read/write

**Responsibility.** Sole owner of every byte that lives under `vault/documents/`. All other modules must read and write the canonical store through this interface, never directly. Implements ADR-0003's "Markdown wins" rule operationally: atomic writes, frontmatter round-trip without loss, doc-id-to-path resolution, and conflict detection when an on-disk file has been edited under the system's feet. This is the only module that reaches into the filesystem with write intent on the canonical tree.

**Public interface** (file: `src/memex/vault/store.py`):

```python
class DocumentRef(BaseModel):
    doc_id: str                      # content-hash + source-path namespace
    markdown_path: Path              # vault/documents/{doc_id}.md
    asset_dir: Path                  # vault/documents/{doc_id}/
    source_path: Path                # vault/documents/{doc_id}/source.{ext}
    content_sha256: str              # of the .md as last written

class Frontmatter(BaseModel):
    title: str | None = None
    authors: list[str] = []
    date: date | None = None
    source_url: str | None = None
    license: str | None = None
    tags: list[str] = []
    custom: dict[str, Any] = {}      # everything else, round-tripped verbatim

class VaultDocument(BaseModel):
    ref: DocumentRef
    frontmatter: Frontmatter
    body: str                        # markdown after the frontmatter delimiters
    mtime_ns: int                    # for watcher dedup

async def assign_doc_id(source_path: Path, content_sha256: str) -> str: ...
async def create_document(source_path: Path, content_sha256: str) -> DocumentRef: ...
async def read_document(doc_id: str) -> VaultDocument: ...
async def write_document(doc: VaultDocument) -> DocumentRef: ...   # atomic
async def list_documents(filters: DocFilter | None = None) -> AsyncIterator[DocumentRef]: ...
async def delete_document(doc_id: str) -> None: ...
def resolve_wikilink(target: str) -> DocumentRef | None: ...
```

**Dependencies.** `memex.core` (config, errors, events), `python-frontmatter`, stdlib (`pathlib`, `hashlib`, `os`, `tempfile`). No other Memex module.

**Doc-ID derivation.** `f"{first8(sha256(content))}-{slugify(source_stem)[:48]}"`. The slug suffix is for humans grepping the directory; the prefix is the collision-resistant key. The full sha256 is stored in `manifest.json`.

**Atomic writes.** Write to `vault/documents/{doc_id}.md.tmp.{pid}.{ulid}`, `fsync`, `os.replace`. The same pattern wraps every manifest write. Filesystem level only — no fancy WAL, since the source of truth is the markdown itself.

**Minimum viable.** `create_document`, `read_document`, `write_document`, `list_documents`. Skip `resolve_wikilink` and `DocFilter` filtering until enrich actually needs them.

**Risks.** Two writers racing on the same `doc_id` (e.g., re-enrich + manual edit). Resolution: take an OS advisory lock (`fcntl.flock`) on the `.md` path for the duration of the write, log a `vault.write_blocked` event if contention exceeds 1 s. The watcher (see §2.3) coordinates re-enrichment around manual edits using mtime + content hash; if a re-enrich would overwrite a manual change, it aborts and emits `vault.edit_conflict`.

---

### 1.2 `memex.ingest` — file watcher, validation, queuing

**Responsibility.** The on-ramp. Takes filesystem inputs (`memex ingest path/`, the watcher firing on a drop into a watched directory), validates them (magic numbers, size limit, MIME), assigns a `doc_id` via the vault, copies the original into `vault/documents/{doc_id}/source.{ext}` immediately, and emits `document.queued`. Does not parse; does not look inside.

**Public interface** (file: `src/memex/ingest/pipeline.py`):

```python
class IngestRequest(BaseModel):
    source_path: Path
    correlation_id: str = Field(default_factory=lambda: str(ulid.ULID()))
    detected_mime: str | None = None

class IngestResult(BaseModel):
    correlation_id: str
    doc_id: str
    accepted: bool
    rejection_reason: str | None = None

async def ingest_file(req: IngestRequest) -> IngestResult: ...
async def ingest_directory(path: Path, recursive: bool = True) -> AsyncIterator[IngestResult]: ...
async def start_watcher(watch_dir: Path) -> WatcherHandle: ...
```

**Dependencies.** `memex.vault`, `memex.core` (event bus), `python-magic` (proposed, add to `pyproject.toml`), stdlib `mimetypes` as fallback, `watchdog` (proposed, add to `pyproject.toml`).

**Validation rules.** Reject above `MEMEX_INGEST__MAX_BYTES` (default 256 MB), reject mismatched extension/magic, reject Office docs with macros unless `--allow-macros` (per Part VI security). Validation failure becomes an `IngestResult(accepted=False, …)` *and* an `ingest.rejected` event — never an exception, because callers process batches.

**Minimum viable.** `ingest_file` + `ingest_directory` (synchronous CLI driver). Defer the live watcher until §2.3 lands. The watcher is the right design for v1 because the vault is meant to live in a directory you drop PDFs into — but a 50-line `find | xargs` equivalent unblocks every downstream module on day one.

**Risks.** Partial copies if `source_path` is on a network mount. Mitigation: copy to a tempfile in `vault/.memex/inbox/` first, sha256, then `os.replace` into `vault/documents/{doc_id}/source.{ext}`. The hash is the doc_id input — if the file changes mid-copy, the input file's mtime changes and we restart.

---

### 1.3 `memex.parse` — Docling primary + VLM fallback

**Responsibility.** Turn `vault/documents/{doc_id}/source.{ext}` into `vault/documents/{doc_id}.md` plus the per-page provenance in `manifest.json`. Decides, per page, whether Docling's output is high-confidence enough to keep or whether to escalate to the VLM (as-built 2026-05-26: **Qwen3-VL-8B-AWQ**, served via a parse-time vLLM process — see Routing logic + `docs/specs/vlm-vllm-serving.md`). Extracts figures and tables to `vault/documents/{doc_id}/figures/` and `tables/`. Runs in a child process sandbox (Part VI: "Sandboxed parsing").

**Public interface** (files: `parse/router.py`, `parse/docling_backend.py`, `parse/vlm_backend.py`, `parse/manifest.py`):

```python
class PageDecision(BaseModel):
    page: int
    engine: Literal["docling", "vlm"]
    confidence: float
    rationale: str
    duration_ms: int

class ParseManifest(BaseModel):
    doc_id: str
    parser_version: str
    docling_version: str
    pages: list[PageDecision]
    figures: list[FigureRef]
    tables: list[TableRef]
    equations: list[EquationRef]
    parsed_at: datetime
    correlation_id: str

class ParseResult(BaseModel):
    doc_ref: DocumentRef
    manifest: ParseManifest
    markdown: str                    # already written by `parse_document` via the vault

async def parse_document(doc_id: str) -> ParseResult: ...
```

**Routing logic.** First pass: Docling on the whole document. Second pass: any page where Docling reports confidence < `MEMEX_PARSE__VLM_THRESHOLD` (default 0.65) gets re-run with the VLM under `ModelRegistry.use("vlm")`. Output for that page replaces Docling's. Both decisions go into the manifest. (As-built the escalation trigger is a per-page `image_fraction` + diagram-class signal, not Docling's confidence — see `parse/pipeline.py::_route_and_escalate`. As-built 2026-05-26 the VLM is **Qwen3-VL-8B-AWQ served by a short-lived parse-time vLLM process** (`vlm_serving="vllm"`, started/torn-down inside `vlm_backend.convert_pages`), NOT the in-process registry — its compressed-tensors int4 can't run in-process on 12 GB (decompresses to dense → OOM); `registry.use("vlm")` is now the legacy AutoAWQ Qwen2.5-VL path. The orchestrator vLLM is paused (`pause_vllm_for_gpu`) so the VLM vLLM gets the GPU, then torn down before chart-OCR. See `docs/specs/vlm-vllm-serving.md`.)

**Office sources.** `.pptx`/`.docx`/`.xlsx` + their ODF cousins can't be rendered by pypdfium2 (the VLM + chart-OCR figure renderers are PDF-only), so `parse_document` converts an Office source to a cached `documents/{doc_id}/converted.pdf` via headless LibreOffice (`parse/office_convert.py`) and runs the FULL PDF pipeline on it — Docling + VLM + chart-OCR all operate on the PDF, so figure bboxes align and diagrams transcribe. The converted PDF is cached + reused so its bytes (hence the content-addressed VLM/chart-OCR cache keys) stay byte-stable across re-parses.

**Dependencies.** `memex.vault`, `memex.core`, `memex.models` (for the VLM context manager), `memex.observability`, `docling` (optional extra `[parse]`), the VLM via the model registry. Spawns a child process via `multiprocessing.get_context("spawn")` with `start_method="spawn"` so parser crashes never take down the daemon.

**Minimum viable.** Docling-only path with `--no-vlm` config flag. Ship the routing skeleton so the VLM can be plugged in later. The `manifest.json` writer is required from day one — it is the audit trail.

**Risks.** Docling has heavy native deps (`torch`, `transformers`); ADR-0001 already commits to NVIDIA, but Docling segfaults on weird PDFs. Mitigation: the parser circuit breaker (Part VI) and a 5-minute per-document timeout. On segfault, mark the document `parse_failed` in the manifest and emit `parse.crashed`; do not retry automatically — escalate to user with an "open original / skip / retry with VLM" prompt.

---

### 1.4 `memex.enrich` — entities, citations, wikilinks

**Responsibility.** Augment a parsed markdown document with structure: extract named entities and concept mentions; resolve citation strings against other vault documents to insert `[[wikilinks]]`; populate frontmatter fields the parser couldn't infer (authors normalized to canonical form, language, topic tags). Every enrichment writes to the manifest with `{model, prompt_name, prompt_version, confidence}`.

**Public interface** (files: `enrich/entities.py`, `enrich/citations.py`, `enrich/pipeline.py`):

```python
class Entity(BaseModel):
    text: str
    kind: Literal["person", "org", "place", "concept", "method", "tool", "other"]
    canonical: str | None = None      # filled when resolution succeeds
    span: tuple[int, int]             # offsets into the markdown body
    confidence: float

class CitationLink(BaseModel):
    surface_text: str
    target_doc_id: str | None
    target_external: str | None       # DOI / arXiv ID if external
    confidence: float

class EnrichResult(BaseModel):
    doc_ref: DocumentRef
    entities: list[Entity]
    citations: list[CitationLink]
    wikilinks_inserted: int
    frontmatter_updates: dict[str, Any]

async def enrich_document(doc_id: str, *, threshold: float = 0.8) -> EnrichResult: ...
```

**Behavior.** Reads via `vault.read_document`, runs `extract_entities` (one structured model call per chunk), runs `resolve_citations` (graph-aware: candidates are existing vault docs filtered by author+year+title trigram match before the model decides). Wikilink insertion is in-place but always atomic — never write a half-enriched markdown. As of P4.1 (2026-05-23) `insert_wikilinks` emits `[[doc_id]]` OR `[[doc_id#section]]` per ADR-0003; the section anchor is chosen opportunistically when the citation's ±300-char context references a heading of the target document (heading discovery via `core/text.py::extract_heading_texts`, chart-block-aware). Read-side helpers in `core/wikilinks.py` (`parse_wikilink`, `extract_wikilinks`, `resolve_wikilink_section`) parse and resolve the section-anchored form for downstream consumers. **As of 2026-05-29 entity extraction is a pluggable backend** (`AgentsSettings.enrich_ner_backend`, default `"llm"`): `"otter"` routes entities through the OTTER span NER (`enrich/ner_otter.py`, `whoisjones/otter-bi-mmbert`), which returns the same per-chunk `list[Entity]` the LLM path emits, so the document-level dedupe + graph write are unchanged; **citations always stay on the LLM**. OTTER loads lazily out of `models/registry` (a CPU/pause-window enrich-time model), and `GraphStore.clear_mentions` makes a re-enrich REPLACE (not append) so a backend switch doesn't leave stale entities. See `docs/specs/ner-enrich.md` + ADR-0012.

**Dependencies.** `memex.vault`, `memex.models.client`, `memex.prompts`, `memex.core`, optionally `memex.index` (read-only — to query the citation graph for candidate resolution targets).

**Minimum viable.** Entity extraction with a single prompt template, frontmatter updates, *no* wikilink insertion. Adding citations safely requires the graph index to exist — defer to Phase 2.

**Risks.** Enrichment is the most expensive stage in tokens, and the easiest to silently regress. Mitigation: every enrichment run records prompt version in the manifest, and `memex eval --category enrichment` regresses the entity-extraction prompt against a frozen subcorpus.

---

### 1.5 `memex.index` — LanceDB + SQLite FTS5 + Kuzu writers

**Responsibility.** Convert markdown into queryable derived state. Chunk the body, embed via the model registry, write rows to LanceDB, SQLite FTS5, and Kuzu. Idempotent on `doc_id` — re-indexing a document deletes old rows first. Owns the schema for all three stores. This is the only module that opens those files for write.

**Public interface** (files: `index/chunker.py`, `index/vector_store.py`, `index/fts_store.py`, `index/graph_store.py`, `index/pipeline.py`):

```python
class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    page: int | None
    char_start: int
    char_end: int
    embedding: list[float] | None = None  # populated by embed step

async def chunk_document(doc: VaultDocument) -> list[Chunk]: ...
async def embed_chunks(chunks: list[Chunk]) -> list[Chunk]: ...
async def index_document(doc_id: str) -> IndexResult: ...
async def remove_document(doc_id: str) -> None: ...
async def reindex_vault(*, force: bool = False) -> ReindexReport: ...
```

**Chunking.** Header-aware: split on markdown headings first (`##`, `###`), then on paragraph boundaries, then by sentence to a target window (`IndexSettings.chunk_target_tokens`, default 400 word-budget ≈ 520 transformer tokens) with overlap. A unit the sentence-splitter cannot break — a GFM table is one paragraph with no sentence boundaries — that exceeds `target * MAX_CHUNK_MULTIPLIER` is **force-split on line/row boundaries** (`_force_split_oversized`) so no chunk becomes an unbounded 21K-char monolith (which would exceed the reranker window + the answer prompt's truncate budget). Budget decisions use `_budget_word_count`, which zeroes `[chart-extracted]` blocks. Each chunk inherits a stable `chunk_id = f"{doc_id}#{sha1(text)[:10]}"` so re-indexing the same content gives the same id. *(Note: making table CONTENT answerable — not just bounded — shipped via the two-phase Table-RAG arc, 2026-05-24: **Phase 1** linearizes each well-formed GFM table into a `[table-rows]` KV block (`parse/table_linearize.py`) so table-only values retrieve + ground; **Phase 2** adds a structured per-doc table store (`index/table_store.py::TableStore`, persisted at `.memex/tables.sqlite`) + a text-to-SQL agent tool (`agents/table_sql.py`) whose result is injected as a synthetic chunk for aggregation/superlative queries, with row results shipping verbatim cells and aggregates gated behind an independent recompute. See ROADMAP Tier 1 + `docs/specs/table-rag.md` + `docs/specs/table-sql.md`.)*

**Schemas.** Defined as `pydantic` models *and* SQL/Cypher DDL files in `index/schemas/` so they version together. The LanceDB schema lives in `index/vector_store.py` as an Arrow schema. The graph store's schema is `index/schemas/graph.cypher`, applied idempotently on connection.

**Dependencies.** `memex.vault`, `memex.models` (embedder via registry), `memex.core`. External: `lancedb`, the graph database, stdlib `sqlite3` (FTS5 built-in to most distros).

**Minimum viable.** Chunker + LanceDB writer + FTS5 writer. The graph store is required for citation traversal but not for v1 hybrid search; defer graph writes to Phase 2.

**Risks.** Embedding dimensionality is a one-way door — switching from 768 to a different dim later means rebuilding every vector. Pin in config, document in an ADR-to-be (see §5).

---

### 1.6 `memex.retrieve` — hybrid + rerank (stub → real)

**Responsibility.** The read side of `index/`. Receives a query string, runs BM25 against FTS5 + dense against LanceDB in parallel, fuses with reciprocal rank fusion, returns the top 50 (default). The rerank submodule loads the reranker on demand and scores the top 50 down to top 10. The agent calls these directly; the agent never imports from `index/`.

**Public interface** (already defined as stubs):

```python
async def hybrid_search(query: str, k: int = 50) -> list[Chunk]: ...
async def cross_encoder_rerank(query: str, candidates: list[Chunk], top_k: int = 10) -> list[Chunk]: ...
```

Add (file: `retrieve/fusion.py`):

```python
def reciprocal_rank_fusion(
    rankings: list[list[Chunk]],
    *,
    k: int = 60,                # RRF constant
) -> list[Chunk]: ...
```

**Implementation notes.** BM25 query via SQLite FTS5 `MATCH`; dense query embeds via `ModelRegistry.use("embedder")` and calls `lancedb.Table.search(...).limit(k)`. RRF with the standard constant 60. Rerank uses the same model handle, batched, GPU-resident.

**Dependencies.** `memex.index` (read-only access — opens the three stores read-only), `memex.models`, `memex.core`.

**Minimum viable.** A BM25-only path (`hybrid_search` returns FTS5 results, no embed) is enough to make the answering agent runnable on a tiny corpus before EmbeddingGemma is wired in. Replace with the real fused path before exiting Phase 0.

**Risk.** `Chunk` is currently defined inside `memex.agents.answering`. Move it to `memex.core.types` to break the implied cycle (the retrieve module imports from agents to satisfy the type checker today, which is wrong even though it's `TYPE_CHECKING`-guarded).

*(Note: artifact-scoped retrieval shipped 2026-05-26 (#256). When a query NAMES a specific artifact ("le diagramme de coupe-feu"), the agent's `resolve_artifact_scope` node — between `retrieve` and `expand_graph` — resolves it to its document(s) via a pure regex+BM25 resolver (`agents/artifact_scope.py`, no LLM) and REPLACES the candidate pool with `hybrid_search_in_docs(query, scope_docs, k=50)`, so a question about a named artifact is answered from the right document or refuses naturally. The named artifact is an automatic doc-selection — the deterministic substrate the planned Notebook-LM-style doc-picker reuses. Conservative + fail-open + HARD-gate-safe; spec `docs/specs/artifact-scope.md`.)*

---

### 1.7 `memex.prompts` — versioned prompt library

**Responsibility.** A typed loader for prompt templates stored as `.md` files with YAML frontmatter. Validates each call's keyword arguments against the prompt's declared pydantic input schema, renders via Jinja2, tags the rendered string with `prompt_name@version` for Langfuse correlation (via a comment header that the Langfuse OpenAI wrapper picks up as `name` metadata when set on the call site).

**Public interface** (file: `src/memex/prompts/loader.py`):

```python
class PromptSpec(BaseModel):
    name: str
    version: str                     # semver-like, manually bumped
    role: Literal["system", "user"]
    target_model: str                # e.g. "cyankiwi/Qwen3.5-4B-AWQ-4bit"
    input_schema: str                # dotted path to a pydantic class
    output_schema: str | None        # same, when applicable
    eval_suite: str | None           # name in tests/evals/

class RenderedPrompt(BaseModel):
    text: str
    name: str
    version: str
    tag: str                         # f"{name}@{version}"

def render_prompt(name: str, **kwargs: Any) -> str: ...    # returns text only; tag set on call site
def load_prompt_spec(name: str) -> PromptSpec: ...
def list_prompts() -> list[PromptSpec]: ...
```

**On-disk layout.**

```
src/memex/prompts/
  assess_sufficiency/
    v1.md                  # frontmatter + Jinja template body
  answer/
    v1.md
  verify_grounding/
    v1.md
  extract_entities/
    v1.md
```

The loader picks the highest version unless `MEMEX_PROMPTS__PIN__<name>=<version>` is set in the env (lets eval runs lock a specific version). The signature stays `render_prompt(name, **kwargs) -> str` to match the existing agent code; the version is resolved internally.

**Dependencies.** `jinja2`, `pyyaml` (transitively via `python-frontmatter` already), `pydantic`. No Memex deps beyond `core`.

**Minimum viable.** Three prompts only: `assess_sufficiency`, `answer`, `verify_grounding` — enough to make `answer_query` runnable end to end. Add `extract_entities` when enrich lands.

**Risk.** Schema-vs-prompt drift. Mitigation: a tiny `tests/unit/test_prompts.py` that imports every spec, resolves its `input_schema` and `output_schema`, and renders the template with a minimal example. Prompt changes that break the schema fail the unit test.

---

### 1.8 `memex.models` — vLLM client + ModelRegistry

**Responsibility.** Two parts. (a) `client.py`: thin wrapper around `from langfuse.openai import openai` (per ADR-0004) that calls vLLM with `guided_json` against a pydantic schema, parses, returns `(instance, tokens_used)`. (b) `registry.py`: the `ModelRegistry` that owns local-process handles for the embedder, reranker, and VLM (the orchestrator is the remote vLLM daemon and is *not* a registry resident — see "Open Architectural Questions" §5.5).

**Public interface** (file: `src/memex/models/registry.py`):

```python
ModelName = Literal["embedder", "reranker", "vlm"]

class ModelHandle(BaseModel):
    name: ModelName
    vram_mb: int
    loaded_at: datetime
    inference_count: int = 0

class ModelRegistry:
    async def use(self, name: ModelName) -> AbstractAsyncContextManager[Any]: ...
    async def status(self) -> list[ModelHandle]: ...
    async def evict(self, name: ModelName) -> None: ...
    async def assert_can_load(self, name: ModelName) -> None: ...    # raises InsufficientVRAMError

def get_registry() -> ModelRegistry: ...                              # singleton
```

`use("embedder")` and `use("reranker")` are no-ops after first load (resident). `use("vlm")` triggers load if not resident; if VRAM check fails, it evicts the reranker first (predictable LRU within "swappable" models) and tries again. If the load itself OOMs, the OOM-protection circuit breaker (Part VI) counts the failure; three consecutive fails halt new parse work and emit `system.degraded`.

`client.py` itself stays small:

```python
# src/memex/models/client.py
from langfuse.openai import openai

async def complete_structured(
    prompt: str,
    schema: type[T],                  # generic over the schema, T bound to BaseModel
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    prompt_tag: str | None = None,    # "assess_sufficiency@v1"
) -> tuple[T, int]: ...

async def embed_texts(texts: list[str]) -> list[list[float]]: ...
async def rerank_pairs(pairs: list[tuple[str, str]]) -> list[float]: ...
```

`embed_texts` and `rerank_pairs` go through `get_registry().use(...)` under the hood. `complete_structured` goes through the OpenAI client pointed at `inference.base_url` (vLLM).

**Dependencies.** `openai`, `langfuse`, optionally `transformers` / `sentence-transformers` for the local embedder + reranker (add to the `[parse]` extra or its own `[models]` extra — the agent layer should be importable without torch installed). `core` for config and errors.

**Minimum viable.** `complete_structured` against a running vLLM is the unblocker for everything else. `embed_texts` next, `rerank_pairs` after that. The full circuit breaker can be a counter-with-threshold; the eviction logic can start as "fail to load if full" and only add LRU eviction when the VLM is wired in.

**Risk.** Long cold start (ADR-0001 calls out ~30 s for the orchestrator). Tests cannot pay this cost. Solution: tests inject a fake client via dependency injection (see §2.6).

---

### 1.9 `memex.mcp` — MCP server

**Responsibility.** Expose the vault to external agents as MCP tools (Part V). The "public API of Memex" per the guidelines. The v1 core: `search`, `get_document`, `list_documents`, `get_graph_neighbors`, `ask` — since extended with `summarize` (ADR-0008), `list_scope_sets` (scope-sets), and `related_documents` (the entity-specificity-ranked discovery surface, ADR-0011). Runs over stdio for desktop agents; HTTP transport behind a flag.

**Public interface** (file: `src/memex/mcp/server.py`):

```python
class SearchRequest(BaseModel):
    query: str
    k: int = 10
    filters: DocFilter | None = None

class DocumentEnvelope(BaseModel):
    ref: DocumentRef
    frontmatter: Frontmatter
    body: str

def build_mcp_server() -> mcp.Server: ...
async def serve_stdio() -> None: ...
async def serve_http(host: str = "127.0.0.1", port: int = 7424) -> None: ...
```

`ask(question)` simply delegates to `memex.agents.answering.answer_query` and serializes the `FinalResponse`. `search` runs `hybrid_search` + `cross_encoder_rerank` and returns `Chunk` envelopes. Filters mirror `vault.list_documents`.

**Dependencies.** `mcp` (official Python SDK — proposed add, pin to `<2`), `memex.agents`, `memex.retrieve`, `memex.vault`, `memex.index` (read-only graph queries for `get_graph_neighbors`).

**Minimum viable.** Stdio only, three tools (`search`, `ask`, `get_document`). HTTP and graph traversal in Phase 3.

**Risk.** MCP's Python SDK is fast-moving; v2 (breaking transport rework) is on the near horizon. Pin minor and bump deliberately.

---

### 1.10 `memex.webui` — FastAPI + HTMX

**Responsibility.** Local-only browser interface for the visual workflows the CLI can't do well: side-by-side preview of source PDF and extracted markdown, citation-graph zoom, per-document correction. Server-rendered HTML with HTMX interactivity per Part V. Mounts on `127.0.0.1:7423` only.

**Public interface** (files: `webui/app.py`, `webui/routes/{ask,documents,graph,review}.py`):

- `GET /` — search-and-ask landing (now also lists the doc-picker checklist + the saved-scope-set bar)
- `POST /ask` — kicks off `answer_query` in a background task, returns the live progress fragment (the `streams a partial response if SSE is wired in` idea below is realized as HTMX long-polling)
- `GET /ask/{cid}/status?v=N` — long-polls the answering agent's phase advance; renders `_answer.html` on completion
- `GET /documents` — list with filters
- `GET /documents/{doc_id}` — side-by-side viewer (markdown + page-image preview when a source PDF / Office-converted PDF is present)
- `POST /documents/{doc_id}/summarize` — kicks off `summarize_document` in a background task, returns the live progress fragment
- `GET /documents/{doc_id}/summarize/status?cid=&v=N` — long-polls the summarizer's phase advance; renders `_summary.html` on completion
- `GET /documents/{doc_id}/source` — serves the original file (inline `Content-Disposition` for the pane header `download` link)
- `GET /documents/{doc_id}/source/page/{n}` — rasterises a **0-based** page to PNG (`parse.pdf_render`, pypdfium2-light, lock-serialized — pypdfium2 is NOT thread-safe)
- `GET /graph/{doc_id}` — the document-connections "Bridges" page (server-rendered/ranked, `?group=concept|document`; Cytoscape was dropped 2026-05-29 — a 1-hop star has no topology to draw)
- `POST /documents/{doc_id}/review` — apply a manual correction (delegates to `vault.write_document` after merging)
- `GET /resources` + `POST /resources/mode` — co-residence mode comparison + live hot-switch (ADR-0007 §"Runtime transitions")
- `POST /scope-sets`, `POST /scope-sets/apply`, `POST /scope-sets/delete` — saved-scope-set CRUD (re-rendered partial)
- `GET /healthz` — for `memex daemon status` polling

**Dependencies.** `fastapi`, `uvicorn`, `jinja2` (already pulled in via prompts), HTMX as a vendored `static/htmx.min.js`, Tailwind as a hand-curated utility subset at `static/tailwind.css`. No build step, no React, no CDN. (Cytoscape.js was vendored the same way for an earlier `/graph` viz; the 2026-05-29 "Bridges" redesign dropped it — the connections page is now pure server-rendered HTML/CSS, zero scripts.)

**Minimum viable (shipped).** Every endpoint above except `/graph` (Phase 3+).

**Live progress.** Both `POST /ask` and `POST /documents/{id}/summarize` return an immediate `_progress.html` fragment whose `hx-get` chains a long-poll on the corresponding status endpoint. The status route HOLDS the connection (`webui/progress.py::ProgressRegistry::wait_for_change` — `asyncio.Event` + monotonic `version`) until the phase advances or a ~1 s keepalive — SSE-accurate behaviour without any new vendored JS. The agent + summarizer both expose an opt-in, observe-only progress hook (`answer_query(correlation_id, on_node)` via a `_NodeProgressHandler` LangGraph callback; `summarize_document(correlation_id, on_phase)` is linear → calls the sink directly). CLI/MCP/eval pass neither → byte-identical behaviour; HARD gates untouched.

**Source-preview pane.** Server-rendered page images (`<img loading="lazy">` per page, one PNG per request), NOT an embedded `<iframe>` PDF — the latter rendered blank under the browser's "download PDFs" pref. Works in every browser + is the right affordance for scans/handwriting (the original page sits beside its transcription). The doc route reads page-0 dimensions from the source PDF (`parse.pdf_render.pdf_page_size`) and feeds them as `--pdf-page-aspect` on the `.pdf-pages` container so the placeholder `<img>` has real height before the PNG loads — without that, `loading="lazy"` is effectively eager (every row reads as in-viewport and all N requests fire on initial load).

**Risk.** "Just enough Tailwind" with no build step — addressed by vendoring a hand-curated utility subset. WCAG AA verified via the contrast-floor convention in `webui/CLAUDE.md` (secondary text at `zinc-400`; never `zinc-500/600` for text content).

---

### 1.11 `memex.cli` — commands beyond `version`

**Responsibility.** The canonical interface (Part V). Every other interface (web UI, MCP) ultimately calls the same backend functions the CLI calls. The CLI grows by one command per phase; commands are added under `src/memex/cli/commands/{name}.py` and registered on the `app` in `src/memex/cli/__init__.py`.

**Command surface** (matches guidelines):

```
memex version                                # exists
memex config init                            # write a default config.toml
memex config show

memex ingest <path>...                       # one or many files / directories
memex parse <doc_id>                         # explicit re-parse
memex enrich <doc_id>                        # explicit re-enrich
memex index <doc_id>                         # explicit re-index
memex reindex [--force]                      # rebuild .memex/ from documents/
memex retitle <doc_id> [TITLE] [--derive]    # metadata-only rename (no re-embed)
memex remove <doc_id> [--yes]                # drop a doc: Markdown + asset dir + manifest + all index state
memex ask <question>                         # JSON-when-piped, rich-when-TTY
memex search <query> [--k N]
memex graph --document <doc_id>
memex doctor                                 # vault integrity + daemon health
memex daemon start | stop | status           # manages vLLM + Memex server
memex serve mcp [--stdio | --http]
memex serve web [--port N]
memex eval [--category C] [--quick]
```

Every command imports `configure_logging`, `configure_tracing`, and a single `bootstrap(settings)` helper that sets up the event bus, observability, and model registry. Each command then calls the relevant module function and renders.

Output rule: when `sys.stdout.isatty()` use `rich` tables; otherwise emit JSON via `pydantic`'s `model_dump_json`. Already implied by the guidelines but worth a single helper, `cli/render.py`, so every command does it the same way.

**Dependencies.** `typer`, `rich`, every module the CLI fronts.

**Minimum viable.** `ingest`, `ask`, `reindex`, `serve mcp`. Everything else is icing.

**Risk.** The daemon model (per ADR-0001: vLLM cold start is ~30 s) means `memex ask` cannot afford to spin up vLLM each invocation. `memex daemon start` is required from Phase 1, and CLI commands that need the orchestrator must check `daemon status` and either start or instruct the user. Reranker + embedder live in the Memex process; the orchestrator is the remote vLLM daemon.

---

## 2. Cross-Cutting Architectural Concerns

### 2.1 Event bus: in-process asyncio with SQLite persistence

The bus is a single asyncio-based publish/subscribe with synchronous append-to-SQLite as the persistence sidecar. Two layers:

```python
# src/memex/core/bus.py
class EventBus:
    async def publish(self, event: MemexEvent) -> None: ...
    def subscribe(self, *, stage: EventStage | None = None, event_type: str | None = None) -> AsyncIterator[MemexEvent]: ...
    async def replay(self, *, since: datetime, until: datetime | None = None) -> AsyncIterator[MemexEvent]: ...

def get_bus() -> EventBus: ...
```

Internally, `publish` does two things in this order: (1) `await sqlite_writer.append(event)` to `vault/.memex/events.sqlite` (write-ahead is fine — events are derived, regenerable per ADR-0003), (2) `for q in subscribers: q.put_nowait(event)`. Subscribers receive via `asyncio.Queue` so a slow subscriber backpressures itself, not the publisher (`put_nowait` raises if the queue is full; the bus logs `bus.dropped` and keeps moving — pipelines are not allowed to wedge on observability).

Why asyncio and not threads: every stage is already `async`, and the in-process bus mirrors the I/O concurrency model the rest of the code uses. Why SQLite and not Postgres: GUIDELINES Part II is explicit ("single-machine application; no Redis, no Kafka"), and SQLite is already a transitive dependency via FTS5.

Retention: a rolling 30-day window per the guidelines, enforced by a background task that runs once per hour and `DELETE FROM events WHERE timestamp < ?`. This is also where `memex doctor` looks for stuck correlation IDs.

### 2.2 ModelRegistry + VRAM manager

`get_registry()` returns a process-singleton `ModelRegistry`. The registry holds an `asyncio.Lock` per model and a `dict[ModelName, ModelHandle | None]`. The context manager:

```python
@asynccontextmanager
async def use(self, name: ModelName) -> AsyncIterator[Any]:
    async with self._locks[name]:
        await self._ensure_loaded(name)            # may evict swappable peers
        try:
            yield self._handles[name].model
        finally:
            self._handles[name].inference_count += 1
```

`_ensure_loaded` runs `assert_can_load`, which queries `torch.cuda.mem_get_info()` (the only place `torch` is imported in the registry — and only when CUDA is available). If predicted residency exceeds `HardwareSettings.gpu_memory_fraction * total_vram`, eviction proceeds in this order: reranker, then VLM, then embedder (LRU on the swappable set; the embedder is "always resident" but evictable if explicitly requested).

The OOM circuit breaker is a wrapper around `_ensure_loaded`: catch `torch.cuda.OutOfMemoryError`, increment a per-model counter, log `model.oom`. Three consecutive OOMs trip the breaker — the registry emits `system.degraded` on the event bus, refuses further loads, and `memex doctor` reports the state with recovery steps ("close other GPU processes; `memex daemon restart`"). The breaker resets on successful load.

Hot tip for tests: the registry's `_ensure_loaded` accepts a swappable model factory. Tests inject a `FakeEmbedder` that returns deterministic vectors — see §2.6.

> **Update (2026-05-27): the VRAM tradeoff is now a named-mode policy layer.** Rather than the auto-eviction sketch above, the shipped design ([ADR-0007](adr/0007-co-residence-resource-modes.md)) encodes the embedder/reranker placement + the orchestrator's util/context-window as named *modes* (`fast`/`full`/`gpu_only`/`manual`) resolved by one pure seam, `core/resources.py::resolve_profile`. The registry reads it via `effective_devices(...)`; bootstrap, the daemon, the CLI (`memex mode show|set`), and the webui all consume the same `ResourceProfile`. The *dynamic* VRAM manager this section anticipated is delivered later by replacing that resolver's body (compute the profile from live free-VRAM) — callers untouched. `full` mode's ~24 K window is what the document summarizer ([ADR-0008](adr/0008-document-summarization.md)) was built against, though the summarizer is mode-independent by construction.

### 2.3 Vault file watcher

Watchdog observes `vault/documents/**/*.md`. The watcher class debounces fs events on a per-path 500 ms window (`asyncio.TimerHandle`) — typical editors emit several events per save. Atomic-write semantics dovetail: we only re-enrich on a `modified` event whose post-debounce stat shows mtime > last-known and content-sha256 ≠ last-known.

On a real edit, the watcher fires `vault.edited` with `{doc_id, mtime_ns, prev_sha, new_sha, correlation_id}`. The enrich and index modules subscribe and re-run on that single doc. Locking is via the same `fcntl.flock` `vault.write_document` uses, so the watcher's re-index can't race a manual edit and a manual edit can't race a re-index.

Conflict policy: if the watcher fires while a write to the same path is in flight by Memex itself, the watcher swallows the event (because the post-write `mtime + sha256` will match the one Memex just wrote). If a user edit overlaps a re-enrich, the re-enrich notices on save attempt (sha256 mismatch against the version it loaded), aborts, and emits `vault.edit_conflict` so the user is told.

### 2.4 Document ID + manifest lifecycle

Doc IDs are assigned exactly once, at ingest, by `vault.assign_doc_id(source_path, content_sha256)`. The ID is content-hash-plus-source-slug per ADR-0003's collision rule. Once assigned, the ID is immutable — re-parsing or re-enriching the same doc reuses the same ID. Re-ingesting a file with the same content but different source path produces a *different* doc ID (this is correct: the same paper from two different sources is two corpus citations, even if it's the same PDF).

Manifests live at `vault/.memex/manifests/{doc_id}.json` and grow as the document advances through the pipeline:

```json
{
  "doc_id": "abc12345-smith-2024",
  "content_sha256": "...",
  "source_path": "...",
  "ingested_at": "...",
  "parse": { "engine_decisions": [...], "parser_version": "...", "duration_ms": ... },
  "enrich": { "entities": [...], "wikilinks_inserted": 7, "prompt_versions": {"extract_entities": "v3"} },
  "index": { "chunk_count": 42, "embedding_model": "embeddinggemma-300m@FP16", "indexed_at": "..." },
  "correlation_ids": ["01HW...ingest", "01HW...enrich"]
}
```

The manifest is the single audit artifact per document. It is written atomically (tempfile + replace) after each successful stage. Recovery: on `memex doctor`, walk the manifests and compare `index.indexed_at` against the markdown's mtime; if mtime > indexed_at, the document is queued for re-index.

### 2.5 Correlation ID propagation

Two distinct correlation ID lifetimes, both ULIDs (so they sort naturally and embed a timestamp):

- **Document correlation ID**: generated at ingest, persists in the manifest, and is the ID under which all stages of that document's processing trace appears. A re-enrich generates a *new* correlation ID (because it is a new run, not a continuation), but `bind_run_context` also binds `doc_id` and the original `ingested_correlation_id` so traces are joinable. Logs and traces use the run's correlation ID; the manifest keeps a list of every correlation ID that has touched the document.
- **Query correlation ID**: generated at query receipt by `answer_query` (already implemented in the agent). One ID per ask. Independent of any document; the trace tree joins them naturally because the agent records which chunks (and therefore which doc_ids) it cited.

The single rule: `bind_run_context(correlation_id, **fields)` is called exactly once at the entry point of every public async function that represents a "run" (`ingest_file`, `parse_document`, `enrich_document`, `index_document`, `answer_query`, MCP tool handlers). Nested functions never call it. The matching `clear_run_context()` lives in a `finally`. This is already the pattern in `agents/answering.py`; every new entry point follows it.

### 2.6 Test ergonomics

Tests must not depend on a running vLLM, on a CUDA GPU, or on a multi-gigabyte model download. Three primitives make this work:

1. **Fake model client**. A `memex.models.client._client` module-level handle that defaults to the real OpenAI/vLLM client and is replaced in tests via a `pytest` fixture. `complete_structured` calls `_client.chat.completions.create(...)`. Tests register canned responses by `(prompt_substring, schema)` → instance.

2. **In-memory event bus**. The default `EventBus` writes to `vault/.memex/events.sqlite`. Tests construct one with `EventBus(persistence=None)` so nothing hits disk. The bus accepts an optional `db_path` argument; production callers use the configured path, tests omit it.

3. **Tmp-vault fixture**. `pytest`'s `tmp_path` becomes the `vault_path`. A `make_vault` helper builds a minimal `vault/documents/{doc_id}.md` plus its manifest so integration tests don't need ingest/parse to run before exercising retrieve.

Integration test for the agent (the most important one): `tests/integration/test_answering_with_fakes.py` builds three documents in a tmp vault, indexes them via the real `memex.index.index_document` (which uses a `FakeEmbedder` that returns hashed deterministic vectors), and runs `answer_query("…")` against them with the fake LLM client returning canned `DraftAnswer` and `VerificationResult` instances. The whole test runs in < 1 s on CPU. This is what unblocks every subsequent module — once this passes, you have a regression net.

Real-model integration tests live under `tests/integration/with_models/`, marked `@pytest.mark.slow`, skipped in the fast CI lane, and run on the reference rig in the nightly job.

---

## 3. Build Order — Critical Path

The constraint: deliver something a user can ask a question of as fast as possible, then expand both ends (ingest and interfaces) outward from a working answering loop. Five phases. Each phase ends with a demoable behavior.

### Phase 0 — "answer_query runs end to end on a hand-crafted vault" (1–2 weeks)

Goal: a developer can hand-write a markdown file in `vault/documents/`, run `memex index`, and ask a question via Python.

Ships together:

- `memex.core.bus` (event bus, persistence optional)
- `memex.vault` — minimum-viable (`create_document`, `read_document`, `write_document`, `list_documents`)
- `memex.prompts` — loader + three prompts (`assess_sufficiency@v1`, `answer@v1`, `verify_grounding@v1`)
- `memex.models.client.complete_structured` — real, pointed at a running vLLM
- `memex.models.registry` — minimal: only the embedder, no eviction yet
- `memex.index.chunker` + `index.vector_store` + `index.fts_store` (skip graph store)
- `memex.retrieve.hybrid_search` (real RRF over FTS5 + LanceDB) + `cross_encoder_rerank` (lazy load on first call)
- `memex.cli` commands: `ingest --skip-parse` (just `vault.create_document` on a `.md` input), `index`, `reindex`, `ask`
- `tests/integration/test_answering_with_fakes.py` (the regression net)

Exit criterion: `memex ask "…"` returns a `FinalResponse` against a 5-doc hand-built vault, with a Langfuse trace.

### Phase 1 — "real documents in, queryable out" (2–3 weeks)

Goal: drop a folder of PDFs in, get answers.

Ships together:

- `memex.ingest` — file copy, validation, doc-id assignment; no live watcher yet
- `memex.parse` — Docling-only path with the routing skeleton in place; VLM path stubbed behind `--no-vlm`
- Manifest writer (used by parse, enrich, index)
- `memex.cli` commands: `parse`, `daemon start|stop|status`
- The OOM circuit breaker and Docling parser circuit breaker (Part VI)
- `tests/integration/test_parse_then_index_then_ask.py` against three real PDFs in `tests/fixtures/`

Exit criterion: `memex ingest path/to/papers/ && memex ask "…"` works, with manifests recorded.

### Phase 2 — "the vault gets smarter" (2–3 weeks)

Goal: enrich, graph traversal, watcher, VLM fallback.

Ships together:

- `memex.enrich` — entity extraction + frontmatter updates first; citation resolution + wikilinks second
- `memex.index.graph_store` — graph schema and write path; `remove_document` clears all three stores
- `memex.parse` VLM path — real `ModelRegistry.use("vlm")`, escalation on low Docling confidence
- `memex.ingest.start_watcher` — watchdog + debounce + edit-conflict detection
- `memex.cli` commands: `enrich`, `graph`, `doctor`
- The eval corpus harness (`memex eval --quick`) running against ~10 documents

Exit criterion: editing a markdown file in `vault/documents/` triggers re-enrich + re-index within 2 s; `memex graph --document …` prints citation neighbors.

### Phase 3 — "external interfaces" (1–2 weeks)

Goal: agents and humans can talk to the vault.

Ships together:

- `memex.mcp` — stdio transport, three tools (`search`, `ask`, `get_document`); HTTP behind a flag
- `memex.webui` — `/`, `/ask`, `/documents`, `/documents/{id}` (markdown render); WCAG AA verified
- `memex.cli` commands: `serve mcp`, `serve web`, `search`

Exit criterion: Claude Code can connect to the running MCP server and answer a question about the vault; a browser can do the same at `localhost:7423`.

### Phase 4 — "production hardening" (continuous)

- Full eval corpus (125 docs across 7 categories per `docs/eval-corpus-plan.md`)
- Per-PR eval delta reports in CI
- The PDF side-by-side viewer in the web UI
- Cytoscape graph viz
- Annotation correction UI
- Benchmarks gated in CI (15% regression fails the build)
- Documentation site (`mkdocs-material`)

Phases 0–3 are the critical path. Phase 4 runs in parallel with Phase 2/3 because it accretes; everything inside Phase 4 is independently shippable.

---

## 4. Architecture Diagram (ASCII)

```
                            +-----------------------+
                            |  Markdown vault       |
                            |  vault/documents/     |  <-- ADR-0003: source of truth
                            |   {doc_id}.md         |
                            |   {doc_id}/source.pdf |
                            |   {doc_id}/figures/   |
                            +----+--+---------+-----+
                                 |  ^         ^
                          (watchdog)  |       | (atomic write)
                                 |  |         |
            INGEST PIPELINE (asyncio, in-process, event-driven)
            ============================================================
                                 |  |         |
   user drops file   --> ingest --|--+         |
   (CLI or watcher)      (validate,            |
                          copy source,         |
                          assign doc_id,       |
                          emit document.queued)|
                                 |             |
                          v event: document.queued
                                 |
                              parse --(Docling + optional VLM)----+
                              (writes {doc_id}.md atomically,     |
                               writes manifests/{doc_id}.json)    |
                                 |                                |
                          v event: document.parsed                |
                                 |                                |
                             enrich --(entities, citations,       |
                                       wikilinks)                 |
                                 |                                |
                          v event: document.enriched              |
                                 |                                |
                              index ---> vault/.memex/embeddings.lance
                                    ---> vault/.memex/search.sqlite
                                    ---> vault/.memex/graph.* (per chosen store)
                                    ---> vault/.memex/manifests/*
                                 |
                          v event: document.indexed

            QUERY PATH
            ============================================================

   CLI    --+
            |
   MCP   ---+---> answer_query()  (LangGraph state machine)
            |       |
   WebUI ---+       v
                  retrieve --> rerank --> assess --> answer --> verify
                  (FTS5 +     (cross-      (LLM)     (LLM)       (LLM)
                   LanceDB,    encoder)                          |
                   RRF)                                  +--------+--------+
                                                         |        |       |
                                                      compose  regenerate  refuse

            OBSERVABILITY  (joined by ULID correlation_id)
            ============================================================

   structlog (ContextVars)                  Langfuse (self-hosted)
        | JSON to disk                            | trace tree + spans
        v                                         v
   vault/.memex/logs/*.jsonl              http://localhost:3000
                                          (Postgres + ClickHouse + Redis + MinIO
                                           per Langfuse v4 self-host requirements)

            INFERENCE  (out-of-process per ADR-0001)
            ============================================================

   memex daemon ---spawns---> vLLM server (orchestrator)
                              localhost:8000/v1   (OpenAI-compatible)
                                 ^
                                 | langfuse.openai wrapper
                                 |
                              memex.models.client.complete_structured

            LOCAL MODEL REGISTRY  (in Memex process)
            ============================================================

   ModelRegistry --owns--> embedder (resident)
                  --owns--> reranker (resident)
                  --owns--> VLM      (on-demand, swapped in)
   OOM circuit breaker watches load failures, emits system.degraded.
```

The three interfaces (CLI, MCP, web UI) all call the same backend functions: `ingest_file`, `parse_document`, `enrich_document`, `index_document`, `answer_query`, `hybrid_search`. Nothing the UIs do is unique to a UI; they differ in rendering, not in capability.

---

## 5. Open Architectural Questions

### 5.1 Chunking strategy

**Question.** How do we segment a markdown body into chunks for embedding and retrieval?

**Options.** (a) Fixed-token windows with overlap. (b) Header-aware with token-window fallback. (c) Semantic chunking (let an LLM decide split points).

**Recommended default.** (b) Header-aware. Split on `##`/`###` first, then paragraph, then sentence-window to ~600 tokens with 100-token overlap. Each chunk carries its heading path as metadata (`"3.2 Methodology > 3.2.1 Cohort selection"`) which the answer node can render in citations.

**Trade-off.** (a) is trivially simple and reliable but ignores document structure, which is the whole point of preserving structure in the markdown. (c) is the most powerful but adds a model call per document and is the kind of thing that silently regresses. (b) gets 80% of the benefit at no model cost.

### 5.2 Embedding dimensionality

**Question.** Fixed at the embedder's native dim, or reduce?

**Options.** (a) EmbeddingGemma's native 768. (b) Matryoshka-reduced 256 for size and speed.

**Recommended default.** (a) 768. Switching is a one-way door — every chunk must be re-embedded. ADR this when chosen; a future "EmbedV2" requires a corpus rebuild.

**Trade-off.** 256 would let LanceDB index a 100k-chunk corpus in roughly a third of the disk and roughly twice the query speed. Not worth the risk while corpus sizes are small.

### 5.3 Prompt versioning storage

**Question.** Where does prompt history live?

**Options.** (a) Filesystem versioned by directory (`prompts/answer/v1.md`, `v2.md`). (b) Git history alone. (c) A registry table in `vault/.memex/`.

**Recommended default.** (a). Multiple versions coexist on disk; the loader picks the newest unless pinned. Git records the *why* of each version; the directory records the *what*. Eval runs pin a version via `MEMEX_PROMPTS__PIN__answer=v1` so a regression at HEAD doesn't taint a baseline.

**Trade-off.** Two files per prompt (current + previous) is a bit of clutter. Worth it because reverts are an `rm` and a pin, not a git-bisect.

### 5.4 Surfacing `assert all_grounded` failures in the UI

**Question.** When the verifier finds ungrounded claims and the agent refuses, what does the user see?

**Options.** (a) Show only `refusal_reason`. (b) Show the rejected draft alongside the refusal and the per-claim verdict. (c) Show the rejected draft with ungrounded claims struck through, no refusal.

**Recommended default.** (b). In CLI: refusal text + a collapsed table of the rejected draft. In web UI: refusal banner + the rejected draft below, with ungrounded claims highlighted and a "show retrieval" affordance. This is the *Observable at Every Layer* principle made user-visible. Never (c) — the rejected draft is not an answer and must not look like one.

**Trade-off.** (a) is the cleanest UX but the worst for trust: the user wonders what the agent saw. (b) is busier but trains the user to understand the system's limits.

### 5.5 MCP transport: stdio-only or HTTP too?

**Question.** Does the MCP server serve over HTTP from the start?

**Options.** (a) Stdio only for v1; HTTP later. (b) Both from v1.

**Recommended default.** (a) for v1 and Phase 3; HTTP follows in Phase 4 once authentication is designed. Stdio is the only transport every MCP client speaks today; HTTP without auth on a multi-user machine is a footgun.

**Trade-off.** Stdio doesn't work for an agent on another machine. That's a Phase 4+ problem.

### 5.6 Orchestrator: registry-managed or out-of-process daemon?

**Question.** Does the orchestrator live in the `ModelRegistry` like the embedder/reranker, or only as a remote vLLM endpoint?

**Options.** (a) Remote vLLM only — Memex process never holds the orchestrator. (b) Both — registry-managed local handle as a fallback if no daemon is running.

**Recommended default.** (a). ADR-0001 already commits to vLLM as the sole inference engine, and ADR-0001's "one model per server process" plus "cold start ~30 s" make in-process loading misery. The registry owns embedder + reranker + VLM; the orchestrator is the remote endpoint. This is implicit in ADR-0001 and worth making explicit in a follow-up ADR.

**Trade-off.** Forces the daemon model from day one. Acceptable; ADR-0001 already accepted it.

### 5.7 Trace storage and rolling cleanup

**Question.** How long do we keep Langfuse traces?

**Options.** (a) Forever, user manages. (b) 30 days by default, configurable.

**Recommended default.** (b), matching the event bus's retention. `vault/.memex/traces/` is derived state per ADR-0003 and should not grow unbounded.

---

## 6. What NOT to Build for v1

A short list of things in the vision that get cut from v1 explicitly, with reasons:

- **Cross-machine sync.** Vision mentions "documents that maintain their own state" as long-horizon. The vault is git-friendly already; users can sync with rsync or git themselves. Building first-class sync requires conflict resolution semantics that are out of scope.
- **Citation graph reasoning during answering.** ~~The graph index is written in Phase 2 and exposed via `get_graph_neighbors`, but the answering agent does *not* traverse it.~~ **RESOLVED with evidence (2026-05-28, [ADR-0011](adr/0011-entity-graph-from-expansion-to-discovery.md)):** graph traversal in the answer loop (`expand_graph`, 1-hop shares-an-entity) WAS built, then a microscope audit proved it adds nothing at this corpus scale (near-total k=50 recall; generic-entity neighbours the reranker discards) → **default-OFF**, env opt-in. The graph's on-mission role is now **explicit discovery** (`related_documents`, entity-specificity-ranked: CLI `memex related` / MCP `related_documents` / webui doc-view; spec `docs/specs/graph-discovery.md`), NOT passive retrieval augmentation. Re-introducing specificity-ranked expansion into the answer path is a documented LARGE-corpus-only hook (where retrieval starts missing docs).
- **Incremental re-indexing as a true optimization.** Phase 2's watcher re-indexes a single document on edit. It does not partial-re-index a document or do delta embeddings. The "thousand-paper corpus doesn't require a full rebuild" goal is a v1.x deliverable.
- **Speculative parsing on idle GPU.** Cute, not necessary.
- **Domain plugins.** No plugin system in v1. Plugins arrive when there's a real second user with a real second domain.
- **Multi-user mode / continuous batching.** vLLM gives us continuous batching for free, but the agent loop is sequential per query by design. Multi-user scheduling, queuing, and quotas are post-v1.
- **PDF annotation correction (visual).** The data path exists (the user can edit the markdown and the watcher picks it up). A graphical correction UI on the PDF itself is Phase 4+.
- **Cytoscape graph viz.** Mentioned in Part V but Phase 4. The graph data is exposed via MCP and CLI before any visualization ships.
- **Telemetry of any kind.** Per Part VII: no telemetry, ever. If we ever cross this line, it requires an ADR and a user-visible opt-in.
- **CPU fallback for inference.** ADR-0001 already declines to ship one. Reiterating: not in v1.
- **Lab notebook parser, code-aware parser, citation-style plugins.** These are real and valuable. They are v1.1+ work driven by users who actually have those documents.
- **Multi-LoRA serving.** Mentioned in ADR-0001 as a possible mitigation; deferred until there's a concrete need.
- **A second inference engine (Ollama, llama.cpp).** ADR-0001 may revisit later; v1 is vLLM only.
