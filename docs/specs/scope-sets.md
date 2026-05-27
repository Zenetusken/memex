# Spec — Saved scope sets

**Status:** Shipped 2026-05-27. The persistence layer over the document
scope-picker (`docs/specs/artifact-scope.md` → "Anti-scope").

## Problem

The doc-picker lets a user scope a question to selected documents
(`answer_query(scope_doc_ids=[...])`), but the selection is ephemeral — re-ticking
the same documents every session is friction. A saved scope set is a **named,
reusable** document selection ("CR350 networking" → those 16 decks), reapplied by
name from any surface.

## Design — a naming layer, not new agent machinery

A scope set is **purely a name → `doc_ids` mapping**. Each surface resolves the
name to `doc_ids` and feeds the EXISTING `answer_query(scope_doc_ids=...)` path
(which rides the `resolve_artifact_scope` node — see `artifact-scope.md`). The
answering agent **never imports `core/scope_sets`**: no new coupling, and the same
HARD-gate guarantee carries over unchanged — scope only ever NARROWS retrieval, so
a stale set (its docs since removed) resolves to an empty candidate pool and the
agent refuses cleanly. There is no way for a scope set to add a chunk, relax a
gate, or leak across documents.

## Storage (`core/scope_sets.py`)

A single JSON file at **`vault/.memex/scope_sets.json`**, written atomically
(`mkstemp` → `fsync` → `os.replace`, via `asyncio.to_thread`) — the exact pattern
of `core/manifest.py`. Models: `ScopeSet {name, doc_ids, created_at, updated_at}`
and `ScopeSetCollection {sets: list[ScopeSet]}`.

- **User-authored, NOT regenerable.** Unlike the embeddings / FTS / tables / graph
  stores it is deliberately absent from the `reindex_vault(force=True)` teardown
  allow-list, so a full rebuild preserves it. (`.memex/` already holds non-derived
  operational state — `events.sqlite`, `daemon/`, `locks/` — so a user-data file
  sits there consistently; it is kept out of `documents/` because it is metadata
  *about* the vault, not a document in it.)
- **One file → one in-process `asyncio.Lock`** guards the read-modify-write of
  `save`/`delete`. Cross-process writes are last-writer-wins but never corrupt (the
  atomic `os.replace`); a cross-process fcntl lock would force `core/` to import
  `vault/`, inverting the import direction, and isn't worth it for a rare manual
  write.
- **The set NAME is a JSON value, never a path component** (the file path is
  fixed), so an arbitrary name carries no path-traversal risk — unlike a `doc_id`.

### API

| Function | Behaviour |
|---|---|
| `normalize_set_name(name)` | The lookup key: whitespace-collapsed + casefolded. `"CR350"` ≠ `"CR 350"` (a space is meaningful) but `"CR350"` = `"cr350"` (case is not). |
| `save_scope_set(vault, name, doc_ids)` | Upsert by normalized name. Validates a non-empty name ≤ 100 chars + ≥ 1 non-blank doc id (`ScopeSetError` otherwise); strips + de-dups `doc_ids` (first-seen order); an update keeps `created_at`, bumps `updated_at`. |
| `get_scope_set` / `list_scope_sets` / `delete_scope_set` | Lookup (case/ws-insensitive) / list (sorted by normalized name, stable) / remove (only the named set — never a document). |
| `read_scope_sets` | Missing file → empty. **Corrupt file → `VaultIntegrityError`** (loud at the management surface). |
| `resolve_scope_set_doc_ids` | name → `doc_ids` for the answer path. **Fails OPEN** — an unknown/empty/corrupt store returns `[]` (full-corpus), never raises into `ask`. |

## Surfaces

- **CLI** — `memex scope-set create NAME --doc ID …` (validates ids against the
  vault — an unknown id is rejected so a typo can't scope to nothing),
  `scope-set list | show NAME | delete NAME [--yes]`; and `memex ask --scope-set
  NAME` (resolves + unions with any `--doc`; unknown name → exit 2 listing the
  available sets).
- **MCP** — `ask(scope_set=)` unions the set's docs with `scope_doc_ids` (unknown
  name → `ConfigurationError`, never a silent full-corpus search); a new
  `list_scope_sets` tool lets a client discover sets.
- **webui** — the picker is the partial `_scope_picker.html`; a saved-set bar of
  chips sits above the doc checklist (each chip = an `apply` button + a `✕`
  delete), a save-as control below it. `POST /scope-sets` (save the ticked boxes
  under a name), `/scope-sets/apply` (re-render with the set's docs **pre-ticked**
  — so the unchanged `/ask` scopes to them; applying a set is just a convenience
  that pre-checks the existing checkboxes, no `scope_set` param on `/ask`),
  `/scope-sets/delete`. All `type="button"` + `hx-post`, swapping `#scope-picker`.
  The shared context helper swallows a corrupt-store `VaultIntegrityError` to an
  empty list so the Ask page never 500s. See `src/memex/webui/CLAUDE.md`.

## HARD-gate invariant

Identical to the doc-picker's: a scope set can only narrow retrieval to a set of
documents. Worst case is a conservative false-refuse (a set whose docs can't answer
the question), never a hallucination or a leaked cross-document answer.

## Validation

- **Unit** (`tests/unit/test_scope_sets.py`): CRUD round-trip, upsert preserves
  `created_at`, dedup/strip, name + doc validation, case/ws-insensitive lookup,
  ordering determinism, corrupt-store raises (read) vs fails-open (resolve),
  on-disk JSON shape.
- **Integration**: `test_webui.py` (save→appears, apply→pre-ticks, delete→removed,
  empty-name/no-docs/unknown-apply flash errors); `test_mcp_server.py`
  (`scope_set` unions doc_ids, unknown raises, `list_scope_sets`).
- **Chrome-extension e2e** (2026-05-27): save "SRWE STP" (2 docs) → chip; reload →
  chip persists; apply → re-ticks SRWE_Module_4 + _5 exactly; the applied scope
  reaches the agent (live log `resolve_artifact_scope via=user-selected
  doc_ids=[…module-4, …module-5]`, `expand_graph` skipped); delete → bar removed.

## Anti-scope (deferred)

- NOT shared/multi-user sets (single-user, local-first — one file per vault).
- NOT auto-derived sets (e.g. "all SRWE decks" by tag) — a set is an explicit
  hand-picked selection; tag-derived scoping is a separate future idea.
- NOT a "cite everything in the scope" answer mode — the agent still grounds in
  the retrieved-and-reranked subset of the scoped pool, exactly as today.
