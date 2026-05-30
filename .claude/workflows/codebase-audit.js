export const meta = {
  name: 'codebase-audit',
  description: 'Max-rigor bug + optimization audit of the whole Memex codebase: deep per-unit review → adversarial per-finding verification (re-read code + check CLAUDE.md/ADRs) → synthesized, deduped, convention-checked action list',
  phases: [
    { title: 'Review', detail: 'one deep reviewer per code unit (~26 units)' },
    { title: 'Verify', detail: 'adversarial verifier per finding — refute by default, check the documented record' },
    { title: 'Synthesize', detail: 'dedup + rank confirmed findings into a fix list' },
  ],
}

// ---- Review units: balanced, context-coherent slices of src/memex ----
const UNITS = [
  { name: 'agents/answering', files: 'src/memex/agents/answering.py', note: 'the LangGraph answer pipeline — HARD-gate critical (refusal_cf=1.0, zero-hallucination). repair_claim_chunk_ids MUST run before verify; chart blocks MUST flow to assess/answer/verify; partial-grounded ships the grounded subset.' },
  { name: 'agents/summarizer', files: 'src/memex/agents/document_summarizer.py src/memex/agents/summarizer_serve.py', note: 'ADR-0008/0009/0010 doc summarizer — bounded MAP→GROUND→REDUCE; output is maxItems-bounded lists; zero-grounded → refuse; mode-independent by construction.' },
  { name: 'agents/table_sql+artifact_scope', files: 'src/memex/agents/table_sql.py src/memex/agents/artifact_scope.py', note: 'Table-RAG text-to-SQL (row-vs-aggregate fabrication boundary; _recompute_aggregate gate; read-only single-SELECT) + #256 deterministic artifact→doc re-scope (positional-qualifier + single-token gates).' },
  { name: 'parse/pipeline', files: 'src/memex/parse/pipeline.py', note: 'parse orchestration — VLM escalation arms (image_fraction/diagram/confidence), scan route, pause_vllm_for_gpu, chart-OCR stitch (_figures_for_chart_ocr 1:1 placeholder alignment), per-page char_count.' },
  { name: 'parse/chart_ocr', files: 'src/memex/parse/chart_ocr_backend.py src/memex/parse/chart_ocr_cache.py', note: 'Nemotron chart-OCR + LaTeX→markdown normalize (_normalize_latex_tabulars); cache keys content+model+version; caches empty results.' },
  { name: 'parse/pymupdf', files: 'src/memex/parse/pymupdf_worker.py src/memex/parse/pymupdf_backend.py', note: 'PyMuPDF path — font-size→heading-level remap; <br>-in-table normalize; misdetected-heading demote; subprocess worker lifecycle.' },
  { name: 'parse/docling', files: 'src/memex/parse/docling_worker.py src/memex/parse/docling_backend.py src/memex/parse/docling_tables.py', note: 'Docling path — bbox-height heading recovery (rank among headers, pre-export); prose-header reclassification; image_fraction; sandbox/timeout/crash handling.' },
  { name: 'parse/vlm+render+office', files: 'src/memex/parse/vlm_backend.py src/memex/parse/vlm_cache.py src/memex/parse/pdf_render.py src/memex/parse/office_convert.py src/memex/parse/sandbox.py', note: 'short-lived vLLM VLM serve (pgid spawn-capture + group-empty reap; util 0.80; retry-once); VLM cache (content+model+prompt key, _MIN_CACHEABLE_CHARS); pypdfium2 NOT thread-safe (lock); office→PDF cached bytes; seccomp.' },
  { name: 'parse/table_linearize', files: 'src/memex/parse/table_linearize.py src/memex/parse/__init__.py', note: 'GFM table → [table-rows] KV linearization + header-sanity gate.' },
  { name: 'webui/app', files: 'src/memex/webui/app.py', note: 'FastAPI routes — lazy GraphStore open + fail-open; long-poll progress; scope-picker; _source_view; the /graph Bridges route just rewritten (related_bridges + related_documents lenses).' },
  { name: 'webui/rendering+progress', files: 'src/memex/webui/rendering.py src/memex/webui/progress.py src/memex/webui/__init__.py', note: 'markdown body render (escape-then-construct XSS; _walk_headings slug-dedup; chart-block-aware); ProgressRegistry (asyncio.Event + version, TTL/cap, evict-on-delivery).' },
  { name: 'cli', files: 'src/memex/cli/commands.py src/memex/cli/bootstrap.py src/memex/cli/__init__.py', note: 'Typer CLI; bootstrap _configure_cuda + _verify_vram_fit; pause_vllm_for_gpu chains.' },
  { name: 'index/graph_store', files: 'src/memex/index/graph_store.py', note: 'RyuGraph wrapper — Cypher RETURN DISTINCT must ORDER BY the PROJECTED ALIAS; _rank_related_documents/_rank_bridges/_rank_co_occurring (just edited); guarded ALTER migration; clear_mentions before re-enrich.' },
  { name: 'index/pipeline+chunker', files: 'src/memex/index/pipeline.py src/memex/index/chunker.py', note: 'index pipeline (embed recipe version force-detect; native-prompt doc-side wrap); chunker (page attribution binary search; chart-block-aware heading split; oversized-table force-split cap).' },
  { name: 'index/stores', files: 'src/memex/index/fts_store.py src/memex/index/vector_store.py src/memex/index/table_store.py', note: 'FTS5 (phrase-wrap is VALIDATED-BENIGN — do not "fix"; chart-block strip; index-driven chunks_for_document); LanceDB flat search; tables.sqlite extract (header-sanity gate).' },
  { name: 'index/small', files: 'src/memex/index/initialism.py src/memex/index/embed_prompts.py src/memex/index/__init__.py', note: 'initialism derivation (EN+FR connectors); EmbeddingGemma native task:/title: prompts (wraps only the transient embedding INPUT — must not leak into stored text/chunk_id).' },
  { name: 'models', files: 'src/memex/models/registry.py src/memex/models/client.py src/memex/models/__init__.py', note: 'model registry (per-model locks; device/dtype dispatch; AWQ import shims; bf16; complete_structured generic over schema; xgrammar guided-JSON).' },
  { name: 'core/config+resources', files: 'src/memex/core/config.py src/memex/core/resources.py', note: 'MemexSettings single source (loaded once, validated); pure resolve_profile (co-residence modes; NoDecode comma-env list patterns).' },
  { name: 'core/data', files: 'src/memex/core/text.py src/memex/core/wikilinks.py src/memex/core/types.py src/memex/core/manifest.py src/memex/core/scope_sets.py', note: 'text helpers (chart_extracted_spans, atomise, STOPWORDS); wikilinks read/write primitives; shared pydantic types; manifest mkstemp→fsync→os.replace atomic write; scope_sets atomic JSON (corrupt raises vs resolve fails-open).' },
  { name: 'core/infra', files: 'src/memex/core/bus.py src/memex/core/breakers.py src/memex/core/errors.py src/memex/core/events.py src/memex/core/sqlite_tuning.py', note: 'event bus; circuit breakers (lock); MemexError subclasses + context dict; sqlite WAL/PRAGMA tuning.' },
  { name: 'enrich', files: 'src/memex/enrich/citations.py src/memex/enrich/pipeline.py src/memex/enrich/ner_otter.py src/memex/enrich/entities.py src/memex/enrich/course_refs.py src/memex/enrich/__init__.py', note: 'enrich (OTTER NER backend lazy/lock-serialized/out-of-registry, threshold 0.05+union; citations always on the LLM; clear_mentions before write; bounded EntityList/CitationList max_length=24; insert_wikilinks section anchors).' },
  { name: 'retrieve', files: 'src/memex/retrieve/rerank.py src/memex/retrieve/entity.py src/memex/retrieve/hybrid.py src/memex/retrieve/related.py src/memex/retrieve/__init__.py', note: 'hybrid dense+FTS RRF; CrossEncoder rerank (reads model.parameters().device; CPU fp32 on 12GB); entity_overview orchestrator (lazy graph open, ImportError fail-open); related_documents_for_seeds (agent leaves it [], surfaces populate).' },
  { name: 'vault', files: 'src/memex/vault/store.py src/memex/vault/_file_lock.py src/memex/vault/__init__.py', note: 'vault read/write (per-doc asyncio locks; cross-process file lock; optimistic-CAS; VaultIntegrityError; the vault is the source of truth).' },
  { name: 'ingest', files: 'src/memex/ingest/watcher.py src/memex/ingest/pipeline.py src/memex/ingest/validation.py src/memex/ingest/__init__.py', note: 'watchdog file watcher (debounce; partial reindex); ingest pipeline; validation.' },
  { name: 'mcp', files: 'src/memex/mcp/server.py src/memex/mcp/auth.py src/memex/mcp/__init__.py', note: 'MCP server tools (ask/summarize/entity_overview/related_documents/scope-sets); auth.' },
  { name: 'eval', files: 'src/memex/eval/runner.py src/memex/eval/scoring.py src/memex/eval/__init__.py', note: 'eval runners (answer/parse/summary); scorers (refusal_cf, cite_prec, structural_f1, gold_chunk_recall, absent_assertion_violations).' },
  { name: 'daemon+obs+prompts', files: 'src/memex/daemon/supervisor.py src/memex/daemon/__init__.py src/memex/observability/tracing.py src/memex/observability/__init__.py src/memex/prompts/loader.py src/memex/prompts/__init__.py', note: 'vLLM daemon supervisor (PID-identity guard accepts exec\'d vllm serve + port; pgid; start timeout); structlog/Langfuse binding; prompt loader.' },
]

const REVIEW_SCHEMA = {
  type: 'object',
  required: ['module', 'findings'],
  properties: {
    module: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'file', 'line', 'category', 'severity', 'title', 'evidence', 'why_it_matters', 'proposed_fix'],
        properties: {
          id: { type: 'string' },
          file: { type: 'string' },
          line: { type: 'string' },
          category: { type: 'string', enum: ['correctness', 'async-concurrency', 'resource-leak', 'error-handling', 'type-safety', 'security', 'performance', 'api-contract', 'dead-code', 'inconsistency'] },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          title: { type: 'string' },
          evidence: { type: 'string' },
          why_it_matters: { type: 'string' },
          proposed_fix: { type: 'string' },
        },
      },
    },
  },
}

const VERDICTS_SCHEMA = {
  type: 'object',
  required: ['verdicts'],
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'verdict', 'confidence', 'severity', 'reasoning', 'contradicts_documented_decision', 'minimal_safe_fix', 'fix_risk'],
        properties: {
          id: { type: 'string' },
          verdict: { type: 'string', enum: ['confirmed-bug', 'confirmed-improvement', 'false-positive', 'intentional-per-docs', 'needs-human-judgment'] },
          confidence: { type: 'number' },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          reasoning: { type: 'string' },
          contradicts_documented_decision: { type: 'string' },
          minimal_safe_fix: { type: 'string' },
          fix_risk: { type: 'string', enum: ['trivial', 'low', 'moderate', 'high'] },
        },
      },
    },
  },
}

const SYNTH_SCHEMA = {
  type: 'object',
  required: ['confirmed', 'rejected_notable', 'stats'],
  properties: {
    confirmed: {
      type: 'array',
      items: {
        type: 'object',
        required: ['file', 'line', 'severity', 'category', 'title', 'fix', 'fix_risk', 'confidence'],
        properties: {
          file: { type: 'string' },
          line: { type: 'string' },
          severity: { type: 'string' },
          category: { type: 'string' },
          title: { type: 'string' },
          fix: { type: 'string' },
          fix_risk: { type: 'string' },
          confidence: { type: 'number' },
        },
      },
    },
    rejected_notable: {
      type: 'array',
      items: {
        type: 'object',
        required: ['title', 'why_rejected'],
        properties: { title: { type: 'string' }, why_rejected: { type: 'string' } },
      },
    },
    stats: {
      type: 'object',
      required: ['total_findings', 'confirmed', 'false_positive', 'intentional', 'needs_human'],
      properties: {
        total_findings: { type: 'number' }, confirmed: { type: 'number' },
        false_positive: { type: 'number' }, intentional: { type: 'number' }, needs_human: { type: 'number' },
      },
    },
  },
}

const reviewPrompt = (u) => `You are a meticulous staff engineer doing a MAXIMUM-RIGOR bug + optimization audit of a SLICE of the Memex codebase (local-first agentic RAG; Python 3.12; async-first; pyright --strict; the cwd is /home/drei/project/Doc_Flo).

FILES TO AUDIT (read every one fully with Read; grep for callers/usages as needed):
${u.files}

CONTEXT (governing invariants for this slice): ${u.note}

The codebase is ALREADY ruff-clean and pyright-strict-clean with 1052 passing tests. Therefore:
- Do NOT report formatting/lint/style, missing type annotations, or anything a type-checker already catches.
- DO hunt for SEMANTIC defects that survive a green type-check + green tests:
  • correctness: logic errors, off-by-one, inverted conditionals, wrong operator, mishandled empty/None/boundary inputs, incorrect math, wrong default
  • async-concurrency: blocking I/O or CPU-heavy work in an async function without asyncio.to_thread, missing await, lock acquired/released wrong, check-then-act races, shared mutable state across tasks
  • resource-leak: a file/db-connection/subprocess/lock/temp-file not closed on every path (esp. exception paths); missing try/finally or context manager
  • error-handling: over-broad except (Exception/BaseException swallowing CancelledError), swallowed exceptions, bare RuntimeError instead of MemexError, missing context dict, error path that loses data
  • security: path traversal, SQL/Cypher injection (string-built queries), unsafe deserialization, a runtime network fetch (air-gap violation), secrets in logs/URLs
  • performance: genuine N+1 DB/file access, redundant recomputation in a loop, re-reading a file already in memory, an O(n^2) where O(n) is trivial — ONLY if it's on a path that runs often enough to matter
  • api-contract: a dict crossing a module boundary where a pydantic model is required; importing a _private symbol across modules; a node return that isn't the documented TypedDict; an unbounded LLM-emit str/list field (must have max_length)
  • dead-code: a function/branch/param that is provably unreachable or unused (verify with grep before claiming)
  • inconsistency: a docstring/comment that contradicts the code in a way that will mislead a maintainer into a bug

CRITICAL — RESPECT THE DOCUMENTED RECORD. This codebase encodes MANY deliberate decisions in src/memex/CLAUDE.md, src/memex/webui/CLAUDE.md, and docs/adr/* that LOOK like bugs/suboptimalities but are INTENTIONAL and validated (examples: the FTS5 query is a literal phrase by design — validated benign, do NOT flag; entity_stopwords was deliberately removed; repair_claim_chunk_ids must run BEFORE verify; chart-extracted blocks must flow to assess/answer/verify; expand_graph is default-off on purpose; the reranker on CPU is deliberate on 12GB). BEFORE reporting anything, grep src/memex/CLAUDE.md (and the webui one for webui files) for the relevant symbol/decision and DROP any finding the docs mark intentional. When unsure, still report it but say so — the verifier will check.

Return ONLY via the StructuredOutput schema. Every finding MUST cite a real file:line and a concrete, checkable defect (quote the offending code in "evidence"). Prefer FEWER, HIGHER-CONFIDENCE findings over a long speculative list. If the slice is genuinely clean, return an empty findings array — that is a valid and valuable result. Use module="${u.name}" and ids like "${u.name.replace(/[^a-z]/gi, '')}-1".`

const verifyPrompt = (u, findings) => `You are an ADVERSARIAL verifier. A reviewer audited the "${u.name}" slice (${u.files}) and produced the findings below. Your default stance is to REFUTE each; only confirm one that is a real, reproducible defect.

FINDINGS (JSON):
${JSON.stringify(findings, null, 1)}

For EACH finding (cwd /home/drei/project/Doc_Flo):
1. Read the ACTUAL code at the cited file:line (and any caller/callee needed). Does the defect truly exist, or does surrounding code / a caller / a type bound already handle it?
2. Is it INTENTIONAL per the documented record? grep src/memex/CLAUDE.md, src/memex/webui/CLAUDE.md, and docs/adr/* for the relevant symbol/decision — many "obvious fixes" here are documented anti-fixes.
3. Would the proposed fix break a documented invariant, a HARD gate (refusal_cf / zero-hallucination), or an existing test?

Return ONE verdict per finding, keyed by the finding's "id" (return a verdict for EVERY finding id). Verdicts: "confirmed-bug" (real defect, will misbehave), "confirmed-improvement" (not a bug but a real, safe optimization/clarity win), "false-positive" (not a defect), "intentional-per-docs" (deliberate + documented), "needs-human-judgment" (a real tradeoff only the maintainer should decide). Set confidence honestly. If confirmed, give the MINIMAL safe fix (exact change) + its risk; else set minimal_safe_fix to "". In contradicts_documented_decision, name the CLAUDE.md/ADR rule the finding or its fix collides with, else "".`

// ---- Phase 1+2: throttled batches (avoid the rate-limit burst that killed the first run).
// Per unit: review → ONE adversarial verify pass over its findings. Resilient (a failed
// agent degrades to empty, never throws out of the workflow). ----
const reviewAndVerify = async (u) => {
  let review
  try {
    review = await agent(reviewPrompt(u), { label: `review:${u.name}`, phase: 'Review', schema: REVIEW_SCHEMA })
  } catch {
    return { unit: u.name, findings: [], verdicts: [] }
  }
  const findings = (review && review.findings) || []
  if (!findings.length) return { unit: u.name, findings: [], verdicts: [] }
  let verified
  try {
    verified = await agent(verifyPrompt(u, findings), { label: `verify:${u.name}`, phase: 'Verify', schema: VERDICTS_SCHEMA })
  } catch {
    return { unit: u.name, findings, verdicts: [] }
  }
  return { unit: u.name, findings, verdicts: (verified && verified.verdicts) || [] }
}

const chunk = (arr, n) => {
  const out = []
  for (let i = 0; i < arr.length; i += n) out.push(arr.slice(i, i + n))
  return out
}

phase('Review')
const BATCH = 4 // small bursts — the first run rate-limited at ~14 concurrent heavy agents
const batches = chunk(UNITS, BATCH)
const collected = []
for (let bi = 0; bi < batches.length; bi++) {
  const batch = batches[bi]
  log(`batch ${bi + 1}/${batches.length}: ${batch.map((u) => u.name).join(', ')}`)
  const results = await parallel(batch.map((u) => () => reviewAndVerify(u)))
  collected.push(...results.filter(Boolean))
}

// Join findings to their verdicts by id; only verifier-judged findings proceed.
const all = []
for (const c of collected) {
  const vmap = {}
  for (const v of c.verdicts || []) vmap[v.id] = v
  for (const f of c.findings || []) {
    const v = vmap[f.id]
    if (v) all.push({ finding: f, verdict: v })
  }
}
const survivors = all.filter((x) =>
  x && x.verdict &&
  (x.verdict.verdict === 'confirmed-bug' || x.verdict.verdict === 'confirmed-improvement' || x.verdict.verdict === 'needs-human-judgment') &&
  (x.verdict.confidence ?? 0) >= 0.5
)
log(`reviewed ${UNITS.length} units → ${all.length} verified findings → ${survivors.length} survived`)

// ---- Phase 3: synthesize the survivors into a deduped, ranked action list ----
phase('Synthesize')
const packed = survivors.map((x) => ({
  file: x.finding.file, line: x.finding.line, category: x.finding.category,
  title: x.finding.title, evidence: x.finding.evidence,
  verdict: x.verdict.verdict, severity: x.verdict.severity, confidence: x.verdict.confidence,
  fix: x.verdict.minimal_safe_fix, fix_risk: x.verdict.fix_risk,
  contradicts: x.verdict.contradicts_documented_decision,
}))
const rejected = all
  .filter((x) => x && x.verdict && (x.verdict.verdict === 'false-positive' || x.verdict.verdict === 'intentional-per-docs'))
  .map((x) => ({ title: x.finding.title, file: x.finding.file, verdict: x.verdict.verdict, why: x.verdict.reasoning }))

let synthesis = null
try {
  synthesis = await agent(
    `You are the audit lead. Below are the VERIFIER-SURVIVED findings from a whole-codebase audit of Memex, plus the count of rejected ones. Produce the final, deduped, prioritized ACTION LIST for the maintainer to apply.

SURVIVED FINDINGS (JSON):
${JSON.stringify(packed, null, 1)}

REJECTED (for the "considered but dismissed" note), ${rejected.length} total — a sample:
${JSON.stringify(rejected.slice(0, 25), null, 1)}

Do: (1) DEDUP findings that are the same defect reported by multiple units (same file+root cause). (2) DROP anything whose "contradicts" names a real documented invariant (note it in rejected_notable instead). (3) RANK confirmed items by severity then confidence. (4) For each, give the precise minimal fix + its risk. (5) Fill stats honestly (total_findings = ${all.length}). Return ONLY the schema. Be precise — this list will be applied as real code edits, so every "fix" must be concrete and safe.`,
    { label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA }
  )
} catch {
  synthesis = null // fall back to the raw survivors below
}

// Always return the raw survivors too, so a synthesis hiccup never loses the audit data.
return {
  synthesis,
  survivors: packed,
  rejected_sample: rejected.slice(0, 40),
  survivor_count: survivors.length,
  total_findings: all.length,
}
