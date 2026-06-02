# ADR-0003: Markdown Vault as Source of Truth, Indexes Are Regenerable

- **Status**: Accepted
- **Date**: 2026-05-19
- **Deciders**: Memex core team
- **Tags**: architecture, storage, data-model

## Context

Memex processes documents into structured representations that downstream features (search, graph, agents) need to query efficiently. There is a tension between two reasonable approaches: keep everything in a database for query performance, or keep everything as files for portability. Most products pick the first and end up with proprietary storage their users can never extract from cleanly.

The vision document is explicit that Markdown files are the user-facing artifact and that the product must be useful even if Memex itself is abandoned. This ADR formalizes the architectural commitment that follows from that promise.

The question this ADR settles: **when the Markdown file disagrees with the index, who wins?**

## Decision Drivers

- Vision principle: "Markdown as the Source of Truth"
- Vision principle: "Composable, Not Captive" — other tools must work on the same data
- User data ownership: the user can leave Memex at any time with no loss
- Backup, sync, version control: all should work with plain filesystem tools
- Operational simplicity: regenerating indexes from scratch should always be safe

## Considered Options

1. **Markdown as truth, derived state regenerable** — files in `vault/documents/` win; `vault/.memex/` is disposable
2. **Database as truth, Markdown as export** — SQLite is canonical, Markdown is a read-only view
3. **Hybrid: Markdown for content, DB for metadata** — content lives in files, structured fields live in DB
4. **Custom binary format with import/export** — fastest, least portable; eliminated immediately

## Decision

**Option 1.** The Markdown files in `vault/documents/` are authoritative. The `vault/.memex/` sidecar (LanceDB, SQLite FTS5, Kuzu, manifests, caches) is derived state and is fully regenerable from the Markdown.

When Markdown and index disagree, the Markdown wins. The index is wrong and gets rebuilt.

> **Refinement (2026-06-02, #362):** "fully regenerable from the Markdown" is precise for everything except one derived-block type — the chart-OCR `[chart-extracted]` blocks are stripped off the now-**content-only** `.md` and cached on the manifest sidecar (`ParseStage.chart_extractions`), re-attached byte-identically at index time. They are NOT re-derivable from the `.md` alone (the source figure is consumed at parse + the OCR is non-deterministic) — they're re-derivable by re-parsing the retained **source**. The "Markdown wins" rule is unchanged: the `.md` is still authoritative for the content a user reads + edits. See "Content-only Markdown" below.

## Consequences

### Positive

- A user can open any document with any text editor or ripgrep over the vault, and they are looking at the canonical data
- The vault is git-friendly out of the box — diffs are readable, history is meaningful, branching works
- Backup is `cp -r vault elsewhere`. Restore is the reverse. No database dump tooling, no schema migrations on restore.
- Memex survives its own deprecation. If we stop shipping, the user's data is unaffected.
- A user can edit Markdown directly (fix an OCR error, add a wikilink, change frontmatter) and Memex's next sync incorporates the change. The user is a peer of the system, not a customer.
- Indexes can be deleted and rebuilt without data loss. This is also the disaster recovery story: if the LanceDB index corrupts, delete it and re-embed. No backup of the index is necessary.

### Negative / Trade-offs

- **Filesystem I/O is slower than database access.** We accept this at single-user scale. Mitigated by caching hot documents in memory and by content-addressed caches for expensive derived operations.
- **Syncing Markdown changes into indexes is non-trivial.** A file watcher detects edits and triggers re-enrichment/re-indexing of the affected document. Race conditions and partial writes need careful handling.
- **Bulk metadata operations are awkward.** Changing a tag across 500 documents means editing 500 files, not running a SQL update. We provide CLI tooling (`memex meta set-tag ...`) that does this safely with atomic writes.
- **Markdown is underspecified.** "Just Markdown" hides real choices: which flavor, which extensions, how YAML frontmatter is parsed. We commit to CommonMark + GitHub Flavored Markdown table extension + YAML frontmatter via `python-frontmatter`, with our specific extensions documented in a spec file.
- **Two documents with the same content hash collide in the namespace.** We resolve via content-hash-plus-source-path as the document ID, never just content hash.

### Neutral

- The on-disk layout is part of the public API. Changing it is a breaking change for any tool that has learned to read Memex vaults directly. We accept this constraint as the price of portability.

## Alternatives in Detail

### Database as truth, Markdown as export

Faster queries, simpler concurrency, clean transactional semantics. Eliminates the file-watcher complexity. We reject it because it makes Markdown export a second-class derived artifact, which means in practice it falls behind the database, fields go missing, and users discover at migration time that "export" was always lossy.

Every product that has chosen this path eventually traps its users. The vision says we won't, so the architecture has to make trapping impossible.

### Hybrid (content in files, metadata in DB)

Splits the difference and inherits the worst of both: the user can still inspect content with text tools, but the metadata they actually filter and search by lives somewhere they can't get to. Reconciliation between the two stores becomes a constant operational concern. We pay the complexity of Option 1 (file watching, regenerable indexes) without getting the simplicity benefit of one source of truth.

### Custom binary format

Performance optimization at the cost of every other goal. Not seriously considered.

## Operational Notes

### What lives where

- `vault/documents/{doc_id}.md` — the canonical processed document, with YAML frontmatter and the body in Markdown
- `vault/documents/{doc_id}/` — sibling directory for figures, tables-as-images, the original source file, and any per-document artifacts that don't belong inline
- `vault/.memex/embeddings.lance` — vector index (LanceDB)
- `vault/.memex/search.sqlite` — FTS5 + relational metadata
- `vault/.memex/graph.ryu` — entity and citation graph (RyuGraph — see ADR-0005, which superseded Kuzu after upstream archival)
- `vault/.memex/tables.sqlite` — per-document structured table store for text-to-SQL (Table-RAG Phase 2); rebuilt from each document's GFM tables on index, dropped on `reindex --force`
- `vault/.memex/manifests/{doc_id}.json` — per-document processing provenance, **including `parse.chart_extractions`** — the chart-OCR `[chart-extracted]` blocks lifted off the content-only `.md` and re-attached at index (the one piece of derived state that is NOT re-derivable from the `.md`; see "Content-only Markdown" below)
- `vault/.memex/traces/` — Langfuse-compatible trace exports (optional)
- `vault/.memex/cache/` — content-addressed derived artifacts

### Rebuild semantics

`memex reindex` deletes `vault/.memex/{embeddings,search,graph}.*` and rebuilds from `vault/documents/`. The command is safe to run at any time. It is also the supported migration path between Memex versions (including the Kuzu→RyuGraph swap per ADR-0005, which is a `memex reindex` and nothing more for any user who never had data in the first place).

### Markdown spec commitments

- CommonMark base
- GitHub Flavored Markdown table extension
- YAML frontmatter (delimited by `---`)
- Wikilinks: `[[document-id]]` and `[[document-id#section]]`
- Math: `$inline$` and `$$display$$` LaTeX
- Anything beyond this spec is undefined behavior and not preserved across re-processing

### Content-only Markdown + index-time derived blocks (audit-10 #362, 2026-06-02)

The canonical `vault/documents/{doc_id}.md` is **content-only**: where the parser found a figure/chart it leaves a `<!-- image -->` placeholder, not the chart's transcription. Two kinds of derived block are re-attached to the body **at index time** (so they reach retrieval / embedding / grounding) without polluting the file the user reads:

- **`[table-rows]`** — the GFM-table linearization (ADR-0014 / Table-RAG Phase 1). Fully **re-derivable** from the `.md`'s own GFM tables; recomputed each index, never stored.
- **`[chart-extracted]`** — chart-OCR markdown. **NOT re-derivable from the `.md`** (the source figure is consumed at parse and the OCR is non-deterministic + cached), so it is persisted on `parse.chart_extractions` in the manifest sidecar and re-attached by `core/text.reattach_chart_extractions` — byte-identical to the historical inline-stitched body, so content-addressed `chunk_id`s stay stable.

This **refines, does not break,** the "derived state regenerable from the vault" promise: the `.md` stays authoritative for the content a user reads + edits; the chart-OCR sidecar is regenerable from the retained **source** (a `--force` re-parse), not from the `.md` alone — the manifest caches it precisely *because* re-deriving it is non-deterministic. The index-time re-attach (and the matching enrich-side re-attach, #394) is the one seam that keeps the chunked/embedded body identical to what the old inline-stitched `.md` produced. (Migrating an existing vault to the content-only form is a GPU-free direct-split — extract the on-disk blocks into the sidecar + write the clean `.md` — never a VLM re-parse, which is non-deterministic.)

## Revisit When

- Filesystem I/O becomes the dominant bottleneck at the dominant user scale (unlikely)
- We add cross-machine sync (this becomes a sync layer over Markdown, not a replacement for it)
- A meaningful contingent of users wants to bypass Markdown entirely (this would not change the architecture, but might add a binary-export feature)

## References

- Memex vision document, §"Why Markdown, Not Notion"
- Memex developer guidelines, Part IV
- ADR-0001 (vLLM): inference is also stateless; this consistency simplifies the mental model
