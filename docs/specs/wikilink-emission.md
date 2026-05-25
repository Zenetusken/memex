# Spec: Agent + MCP wikilink emission (P4.1 completion)

**Status:** spec draft (2026-05-25). Process: spec → independent validation → build → independent validation → tests + gates → doc/commit. **No GPU eval needed** — this is deterministic, post-grounding answer formatting that does NOT change `answered`/`refused`/claims/grounding, so the HARD GATES (`refusal_cf=1.0`, 0 hallucinations) are untouched by construction (the existing suite staying green is the gate).

## Goal
Complete P4.1: the answering agent should emit `[[doc_id#section]]` wikilinks for the chunks it cited, so a cited answer carries navigable links to the source doc/section, and MCP/webui/CLI surface them. **Deterministic + grounded:** derived from the cited chunks' `document_id` + `heading_path` — NOT LLM-generated (no hallucination risk; the links point only at chunks the answer actually grounded against).

## Design

### 1. Build-side primitive — `core/wikilinks.py::format_wikilink`
Add (symmetry with the read-side `parse_wikilink`/`resolve_wikilink_section`):
```python
def format_wikilink(doc_id: str, section: str | None = None) -> str:
    """`[[doc_id#section]]` (raw heading text) or `[[doc_id]]` when no section."""
    if section and section.strip():
        return f"[[{doc_id}#{section.strip()}]]"
    return f"[[{doc_id}]]"
```
- **Section is RAW heading text, NOT a slug** — `resolve_wikilink_section` does a case-insensitive exact match of the section against `chunk.heading_path` entries, and `webui/rendering.py::slugify_heading` slugifies on demand for the URL fragment. So emission must be raw heading text (e.g. `[[0e725ba0#Director Compensation]]`); consumers slugify. Confirm the emitted grammar round-trips through `parse_wikilink` + `resolve_wikilink_section` (pin in a test).
- **Sanitize the section against `[`/`]`** (validation finding): the read-side regex stops at the first `]`, so a section containing `[` or `]` (even a lone `]`) breaks the wikilink → `format_wikilink` falls back to bare `[[doc_id]]` when the section contains `[` or `]`. (A `#` in a heading is SAFE — `parse_wikilink` splits on the first `#` only and doc_ids contain no `#` — so no handling needed for `#`.)

### 2. `FinalResponse.wikilinks` + derivation in `compose` — `agents/answering.py`
- Add `wikilinks: list[str] = []` to `FinalResponse`.
- In the **`compose` node, in the ANSWERED branch — AFTER it builds `used_chunks` from the surviving claims (≈answering.py:1217), NOT before the "no surviving claims" early-return (≈:1201)** (validation finding: compose has an in-situ refusal branch that early-returns `answered=False` with `used_chunks=state.reranked`; the derivation must sit below it so a degenerate refusal keeps `wikilinks=[]`). Derive one wikilink per used chunk: `section = chunk.heading_path[-1] if chunk.heading_path else None` → `format_wikilink(chunk.document_id, section)`. **Dedup while preserving first-seen order** (multiple cited chunks from the same doc+section → one wikilink; a "Sources" list shouldn't repeat).
- **Answered-only:** derive wikilinks ONLY in `compose` (the answered path, from the GROUNDED used_chunks). The `refuse` node leaves `wikilinks=[]` (a refusal didn't cite anything — its `used_chunks=state.reranked` are what it COULDN'T ground against, so emitting links there would be misleading). Confirm refuse → `wikilinks=[]`.
- Pure-derivation, no model call → available without extra tokens/latency.

### 3. MCP — `mcp/server.py`
The `ask` tool returns `FinalResponse` (pydantic auto-serializes over MCP). Adding `wikilinks: list[str]` → it auto-appears in the tool payload. **No code change** — confirm the tool returns the full FinalResponse (not a hand-picked subset that would drop the new field). If it projects a subset, add `wikilinks`.

### 4. Webui — `webui/app.py` `/ask` + `templates/_answer.html`
Add a **Sources** section to `_answer.html` (answered branch only, when `response.wikilinks` non-empty) rendering each wikilink as an `<a>` to `/documents/{doc_id}#{slug}`. **Required build step (validation finding):** the existing rewrite `_replace_wikilinks_with_anchors` is PRIVATE + takes a full HTML-escaped line, and `webui/CLAUDE.md` forbids importing another module's `_private` symbols — so add a **public** `render_wikilink(wikilink: str) -> Markup` in `rendering.py` (HTML-escape the input, then reuse the same `parse_wikilink` + `slugify_heading` + `<a class="wikilink" href="/documents/{doc}#{slug}">` construction — factor `_replace_wikilinks_with_anchors`'s per-match `_sub` body into a shared helper called by both). Register it as a Jinja filter (`Jinja2Templates(...).env.filters["render_wikilink"] = render_wikilink` in `create_app`) so the template can call it. The webui already injects `<span id="{slug}">` anchors before headings (the P4.1 webui work), so the link target resolves. Chart-block-awareness isn't needed here (these are doc#section refs, not body rendering).

### 5. CLI — `cli/commands.py` `ask`
`wikilinks` auto-appears in the JSON output (no change). Optionally add a one-line "Sources: [[…]], [[…]]" row to the rich-table rendering (low priority; JSON is canonical).

## Tests
- `tests/unit/test_wikilinks.py`: `format_wikilink` (doc+section → `[[doc#section]]`; no/empty section → `[[doc]]`; whitespace section → `[[doc]]`); a round-trip pin (`format_wikilink` output parses via `parse_wikilink` back to the same doc_id+section).
- `tests/integration/test_answering_with_fakes.py`: `compose` derives `final.wikilinks` from used_chunks — empty heading_path → `[[doc]]`; single/multi-level → `[[doc#deepest]]`; **dedup** (two cited chunks same doc+section → one wikilink, order preserved); **refuse → `wikilinks=[]`**; an answered case → wikilinks match the cited used_chunks.
- `tests/integration/test_mcp_server.py`: the `ask` tool payload includes `wikilinks`.
- `tests/unit/test_webui_rendering.py` (or `test_webui.py`): the Sources section renders each wikilink as an `<a href="/documents/doc#slug">`; absent/empty when no wikilinks or on refusal.
- Gates: `uv run pytest tests/ -q`, `uv run pyright` (0/0), `ruff check` + `format`. The full suite staying green (esp. the HARD-gate integration tests) is the acceptance — no behavior change to answered/refused/claims.

## Risks / edge cases
- **Heading text with `]]` or `#`** — a heading containing `]]` or `#` could produce a malformed/ambiguous wikilink. `parse_wikilink`'s grammar splits on the first `#`; a heading with `#` would mis-split. Mitigate: `format_wikilink` could reject/sanitize a section containing `]]` (fall back to bare `[[doc_id]]`); assess against real headings (the vault's headings — unlikely to contain `]]`, but `#` in a heading is possible). Pin behavior in a test.
- **doc_id format** — doc_ids are slug#... no, doc_ids are like `0e725ba0-2026-annual-report-web` (no `#`/`]]`). Safe.
- **Refusal** — explicitly `wikilinks=[]` (see §2). No misleading links on a refusal.
- **HARD-gate neutrality** — the derivation runs AFTER grounding/compose; it reads `used_chunks` (already-decided) and adds a field. It cannot change `answered`, claims, or refusal. The existing HARD-gate tests must stay green (no behavior change).

## Anti-scope
- NOT LLM-emitted wikilinks (deterministic from cited chunks only — no hallucination surface).
- NOT changing answer content / claims / grounding / refusal logic.
- NOT wikilinks on refusals.
- NOT per-claim wikilinks in v1 (a flat deduped `FinalResponse.wikilinks` "Sources" list; per-claim is a possible later refinement).
- NOT a new MCP tool — the existing `ask` tool's payload just gains the field.
