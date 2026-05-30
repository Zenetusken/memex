export const meta = {
  name: 'raw-md-audit',
  description: 'Meta-audit the parsed raw-markdown OUTPUT quality across a representative doc set: per-doc structured catalog of noise/artifacts/structure defects, each root-caused to a parse stage with a SOURCE-level fix → synthesized weakness map for the rich-doc-view precursor',
  phases: [
    { title: 'Audit', detail: 'one deep auditor per representative document (read raw .md + manifest)' },
    { title: 'Synthesize', detail: 'unified, deduped, prioritized weakness map + architectural findings' },
  ],
}

const VAULT = '/home/drei/.memex/vault/documents'

// Representative set — spans every parse path so the audit covers the breadth of artifacts.
const DOCS = [
  { id: '0e725ba0-2026-annual-report-web', kind: 'financial 10-K (Docling; tables + charts + a "five-layer cake" layout graphic)', hint: 'the hardest case — 1MB, 7078 lines, 501 H2s. Look at heading leveling (H2 subtitle then H1 section = inverted?), layout-graphics-flattened-as-tables → garbage [table-rows], duplicated [table-rows], financial-table fidelity.' },
  { id: '770047e3-ensa-module-3-2021', kind: 'CCNA slide deck (Docling + per-page VLM escalation)', hint: 'a VLM-transcribed slide is wrapped in a leaked ```markdown code fence (per-page fence-wrapper not stripped); the VLM DESCRIBES decorative/meme imagery ("Left Panel", "**Image:** ...") instead of transcribing content; layout-table mis-parse; CLI config blocks.' },
  { id: '2f96ae1c-s62400-cuda-new-features-and-beyond-171114580640', kind: 'CUDA slide deck (Docling; charts)', hint: '2295 lines of slide fragments; slide titles emitted as ###### (H6); Docling PictureClassifier labels (Logo / Photograph / Line chart) leak as bare text after <!-- image -->; chart-OCR [chart-extracted] blocks.' },
  { id: 'd646b885-gte-2308-03281', kind: 'academic paper (born-digital; equations)', hint: 'check equation fidelity (inline/display LaTeX), reference-list formatting, multi-column reading order, heading hierarchy.' },
  { id: '2d420cbb-irs-w9', kind: 'IRS tax FORM', hint: 'forms have fields/checkboxes/instructions — does the structure survive, or is it a jumble? form-field labels vs values.' },
  { id: '6cf677b7-cs-notes-1', kind: 'handwritten scan (whole-doc VLM transcription)', hint: 'the cleanest baseline — but check for VLM editorialization (an added "*Note: the diagram shows...*"), redundant restatements, and whether a transcribed ASCII diagram is correctly fenced.' },
  { id: '0290d6ec-nist-sp-800-207', kind: 'NIST security standard (born-digital prose; likely PyMuPDF)', hint: 'the "good" born-digital contrast — verify heading hierarchy, figure/table handling, and whether [table-rows] / placeholders still pollute an otherwise-clean prose doc.' },
]

const AUDIT_SCHEMA = {
  type: 'object',
  required: ['doc_id', 'parse_engine', 'doc_type', 'structural_score', 'structural_notes', 'findings'],
  properties: {
    doc_id: { type: 'string' },
    parse_engine: { type: 'string' }, // from the manifest: pymupdf / docling / scan-vlm / office
    doc_type: { type: 'string' },
    structural_score: { type: 'number' }, // 0-10: how close the RAW .md is to "perfectly formatted structured headers/sections/blocks"
    structural_notes: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['category', 'severity', 'title', 'example', 'scope', 'root_cause_stage', 'source_fix'],
        properties: {
          category: {
            type: 'string',
            enum: [
              'derived-block-pollution', 'parser-placeholder-noise', 'classification-label-noise',
              'vlm-over-transcription', 'vlm-fence-wrap', 'heading-level-broken', 'heading-missing-or-flat',
              'layout-graphic-as-table', 'malformed-table', 'duplication', 'encoding-or-html-leak',
              'reading-order-broken', 'missing-content', 'code-fence-misuse', 'list-or-block-malformed', 'other',
            ],
          },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          title: { type: 'string' },
          example: { type: 'string' }, // a SHORT verbatim excerpt (<=20 words / 1-3 lines) + approx line number. For decorative/offensive image-descriptions, describe by TYPE — do NOT quote the offensive content.
          scope: { type: 'string' }, // "pervasive" | "~N occurrences" | "isolated"
          root_cause_stage: { type: 'string' }, // pymupdf_worker | docling_worker | vlm_backend | chart_ocr | table_linearize | _finalize_body | chunker | classifier | architectural
          source_fix: { type: 'string' }, // the SOURCE-level fix (parse worker / pipeline emission). Per the user's mandate: prefer fixing at the source over post-processing.
          post_processing_fallback: { type: 'string' }, // only if a source fix is genuinely impossible; else ""
        },
      },
    },
  },
}

const MAP_SCHEMA = {
  type: 'object',
  required: ['weaknesses', 'architectural_findings', 'scorecard', 'sequence', 'summary'],
  properties: {
    weaknesses: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'category', 'severity', 'title', 'description', 'affected_doc_count', 'example', 'root_cause_stage', 'recommended_source_fix', 'fixable_at_source', 'effort', 'priority'],
        properties: {
          id: { type: 'string' },
          category: { type: 'string' },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          title: { type: 'string' },
          description: { type: 'string' },
          affected_doc_count: { type: 'number' },
          example: { type: 'string' },
          root_cause_stage: { type: 'string' },
          recommended_source_fix: { type: 'string' },
          post_processing_fallback: { type: 'string' },
          fixable_at_source: { type: 'boolean' },
          effort: { type: 'string', enum: ['trivial', 'small', 'medium', 'large'] },
          priority: { type: 'number' }, // 1 = highest
        },
      },
    },
    architectural_findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['title', 'finding', 'recommendation'],
        properties: { title: { type: 'string' }, finding: { type: 'string' }, recommendation: { type: 'string' } },
      },
    },
    scorecard: {
      type: 'array',
      items: {
        type: 'object',
        required: ['doc_id', 'parse_engine', 'structural_score', 'headline_issue'],
        properties: { doc_id: { type: 'string' }, parse_engine: { type: 'string' }, structural_score: { type: 'number' }, headline_issue: { type: 'string' } },
      },
    },
    sequence: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
}

const auditPrompt = (d) => `You are a meticulous document-structure auditor. Memex parses PDFs/Office/scans into a single canonical markdown file per document (the "raw" view — meant to be the SOURCE OF TRUTH with perfectly structured headers, sections, and blocks). Your job: audit ONE document's raw markdown output and catalog EVERY structural defect, noise source, and weird artifact.

THE DOCUMENT (read it fully with Read; cwd is /home/drei/project/Doc_Flo):
  ${VAULT}/${d.id}.md
ALSO read its manifest to learn the parse engine + stages:
  /home/drei/.memex/vault/.memex/manifests/${d.id}.json   (if absent, try: find /home/drei/.memex/vault/.memex -name '${d.id}*' )
DOC KIND: ${d.kind}
KNOWN HINTS (verify + EXTEND — don't just confirm): ${d.hint}

WHAT'S "RAW-VIEW NOISE" vs "SOURCE CONTENT" — load-bearing distinction:
The pipeline injects DERIVED, retrieval-oriented content INTO this source-of-truth .md. These are NOT original document content and pollute a clean raw view:
  - \`[table-rows]...[/table-rows]\` blocks — a KV linearization of every GFM table, added by \`_finalize_body\`/\`linearize_gfm_tables\` for BM25. Flag as derived-block-pollution.
  - \`[chart-extracted]...[/chart-extracted]\` blocks — chart-OCR output stitched at \`<!-- image -->\` sites.
  - \`<!-- image -->\` placeholders — Docling image markers.
  - bare classifier labels (\`Logo\`, \`Photograph\`, \`Line chart\`, \`Picture\`) emitted as text lines.
Catalog these AND separately assess the underlying SOURCE structural quality (headings, sections, lists, tables, code blocks, reading order).

PIPELINE-STAGE VOCABULARY for root-causing (be precise): \`pymupdf_worker\` (born-digital text + font-size→heading-level remap), \`docling_worker\` (scanned/complex PDFs + bbox-height heading recovery + PictureClassifier), \`vlm_backend\` (per-page VLM transcription of diagram pages + whole-doc scan VLM), \`chart_ocr\` (Nemotron chart→markdown), \`table_linearize\` (the [table-rows] emitter), \`_finalize_body\`, \`chunker\`, \`classifier\`, or \`architectural\` (a design issue: derived state baked into the source .md).

USER MANDATE (apply to every source_fix): fix at the SOURCE (the parse worker / pipeline emission) with as LITTLE post-processing as possible — only propose a post_processing_fallback when a source fix is genuinely impossible.

Return ONLY via the StructuredOutput schema. Every finding needs a SHORT verbatim example (≤20 words / 1-3 lines) + an approx line number. IMPORTANT: if a doc contains decorative/offensive image-description content (e.g. a transcribed meme), flag it by TYPE ("VLM described a decorative meme image as prose") — do NOT quote the offensive text. Score \`structural_score\` 0-10 = how close the RAW .md already is to a perfectly-structured source-of-truth view. Be exhaustive but precise.`

phase('Audit')
const BATCH = 4
const results = []
for (let i = 0; i < DOCS.length; i += BATCH) {
  const batch = DOCS.slice(i, i + BATCH)
  log(`audit batch ${Math.floor(i / BATCH) + 1}: ${batch.map((d) => d.id.slice(0, 18)).join(', ')}`)
  const out = await parallel(
    batch.map((d) => () =>
      agent(auditPrompt(d), { label: `audit:${d.id.slice(0, 20)}`, phase: 'Audit', schema: AUDIT_SCHEMA }).catch(() => null)
    )
  )
  results.push(...out.filter(Boolean))
}

log(`audited ${results.length}/${DOCS.length} docs → ${results.reduce((n, r) => n + (r.findings?.length || 0), 0)} findings`)

phase('Synthesize')
let map = null
try {
  map = await agent(
    `You are the lead. Below are per-document structured audits of Memex's parsed raw-markdown OUTPUT (JSON). Synthesize the unified WEAKNESS MAP that will drive a "rich document view" precursor effort (clean raw/source view + rich + original side-by-side).

PER-DOC AUDITS (JSON):
${JSON.stringify(results, null, 1)}

Produce (schema): (1) \`weaknesses\` — DEDUPED across docs (one entry per distinct defect class, e.g. all [table-rows]-pollution findings → one weakness), each with affected_doc_count, a representative example, the root-cause stage, the RECOMMENDED SOURCE-LEVEL FIX (the user mandates fixing at the parse source over post-processing — set fixable_at_source + only give a post_processing_fallback when source-fixing is impossible), effort, and a priority (1=highest, by severity × breadth). (2) \`architectural_findings\` — the cross-cutting design issues, especially: derived retrieval blocks ([table-rows]/[chart-extracted]) + placeholders being baked into the SOURCE-OF-TRUTH .md when the rich-doc-view feature wants a CLEAN raw view separate from the retrieval substrate — recommend how to separate them. (3) \`scorecard\` — per doc. (4) \`sequence\` — the recommended fix order. (5) \`summary\`. Be concrete and precise; this map is the blueprint for the hardening work.`,
    { label: 'synthesize-map', phase: 'Synthesize', schema: MAP_SCHEMA }
  )
} catch {
  map = null
}

return { map, per_doc: results, audited: results.length }
