# `src/memex/webui/` — Frontend conventions

FastAPI + Jinja2 + HTMX, server-rendered, no SPA build step. Tailwind is **vendored** (hand-curated utility subset under `static/tailwind.css` — see the file header for the policy). HTMX is vendored via `scripts/vendor-frontend.sh` with SHA-384 integrity verification. **No third-party fetches at runtime** — the air-gap test passes. Single-user, localhost-only, dark-mode default.

## Aesthetics — non-negotiable

**Never use generic AI-generated aesthetics.** Concretely, that means avoiding:

- **Overused font families** — Inter, Roboto, Arial, default system stacks. The UI should pick a typeface that fits the "developer tool for documents" character (e.g. a specific monospace pairing, a serif for body text in document views) and commit to it. Document the choice in this file if you swap one in.
- **Cliché color schemes** — particularly purple gradients on white/dark backgrounds, neon-on-black "futuristic" looks, gradient meshes used as backgrounds for no reason. Memex is a tool for reading and thinking; the palette should be quiet and confident.
- **Predictable layouts** — centered hero with subheading and CTA button; three-column "feature" grids; floating chat bubble in the bottom-right; toy "AI assistant" framings. Memex's UI is for working, not for selling.
- **Cookie-cutter component patterns** — generic card-with-icon-and-title-and-description; sidebar-with-collapsing-nav for a single-page tool; "modern" sticky headers with blur effects applied indiscriminately.

What we **do** want:

- **Cohesive theme.** One typographic system (one primary face, one optional accent), one color ramp (Memex defaults to zinc on near-black with a single accent for action — blue today, can change). Pick and stick. No introducing a green border for a "success" message that doesn't reference any other green elsewhere.
- **Unique character.** The UI should feel like a desk lamp, not an AI app. Specific affordances: monospace for IDs, ULIDs, hashes; condensed numerals for token counts; collapsible chunk evidence under refusals; correlation-ID footers on every answer so users learn to expect the audit trail.
- **Animations for effect and micro-interactions** — not for decoration. The HTMX `htmx-indicator` opacity transition (in `static/style.css`) is the right scale: 150 ms ease-in-out, communicates "request in flight." A page-load fade-in animation is wrong. A hover ripple on every button is wrong.
- **WCAG 2.1 AA** — contrast minimums (4.5:1 body text, 3:1 large), keyboard focus rings visible, color is never the sole carrier of information (confidence chips use color + label).

## Stack rules

- **No SPA framework.** No React, no Vue, no Svelte, no build step. HTMX is the only client-side library.
- **Tailwind is vendored.** `static/tailwind.css` is a hand-rolled subset of utility classes covering exactly the templates' usage. When a new utility class shows up in a template, add a rule for it; CDN tags are forbidden.
- **HTMX is vendored.** `scripts/vendor-frontend.sh` downloads the pinned 1.9.10 build with SHA-384 integrity verification into `static/htmx.min.js`. A placeholder lives at that path so static-file serving doesn't 404 on a fresh checkout — run the script once before `memex serve web`.
- **No third-party fetches at runtime.** No CDN scripts, no Google Fonts, no analytics, no telemetry. Local-first applies to the UI too — the page renders fully on an air-gapped laptop.
- **Templates use `{% extends "base.html" %}`.** Partials for HTMX returns are prefixed with `_` (e.g. `_answer.html`) and don't extend the base.
- **Routes return `HTMLResponse` from Jinja templates**, never raw strings, never JSON for UI endpoints. JSON belongs to MCP and to `/healthz`.
- **Forms use HTMX, not page reloads.** `hx-post` + `hx-target` + `hx-swap`. The submit button gets `hx-disabled-elt="button"`.

## Document-body rendering (`rendering.py`, P4.1)

The document view renders the canonical markdown body inside a `<pre>` for fidelity — **the raw markdown is the content; don't inject glyphs that read as markdown syntax.** `webui/rendering.py::render_body_html` does three server-side transforms, line-by-line over the *original* body so HTML-escaping never drifts the offsets the analysis depends on:

1. `[[doc]]` / `[[doc#section]]` wikilinks → `<a class="wikilink">` (section → `#slug` fragment).
2. Each real heading gets an **invisible** `<span id="slug" class="anchor-target">` prepended for fragment-scroll. (An earlier version appended a visible `#` permalink — removed 2026-05-23 because in a raw-`<pre>` view a trailing `#` reads as ATX closing-hash and pollutes fidelity.)
3. `extract_toc` builds the sidebar/drawer TOC.

`_walk_headings` is the single source of truth: chart-block-aware (skips inert `# H1` labels inside `[chart-extracted]`), inline-markdown-cleaned (`## [Tips:](url)` → "Tips:" via `clean_heading_text`), and **slug-deduplicated** (`tips`, `tips-1`, `tips-2` — duplicate `id=` is invalid HTML + breaks fragment nav). Both the anchor-span IDs and the TOC hrefs consume it, so they stay in lockstep. TOC gated to `3 ≤ headings ≤ 50` (below = not worth navigating; above = PDF-parse heading-noise, e.g. the 10-K's 501 H2s). When extending: keep slug logic in `_walk_headings`, never recompute it ad-hoc.

## Answer panel (`_answer.html` + `static/style.css`, 2026-05-25)

The `/ask` result (`_answer.html`) renders three **labelled zones** so the reader can tell the parts apart (an earlier version stacked the summary as an `<h3>` above unlabelled claim boxes that often restated it — users couldn't tell answer from citation):

1. **Answer** — `response.summary`, the prominent synthesized answer. A blue left-rule (`.ans-answer`, `border-left: 2px blue-600`) marks it as THE result, reusing the single action accent (focus rings / wikilinks / anchor pulse share it).
2. **Grounded claims · N** — each `claim` as a `.claim` card: the assertion + a meta row with the **confidence chip** (`.conf-{high,medium,low}`, colour **and** label per WCAG 1.4.1 — never colour alone), a "source" label, and the monospace `.chunk-id`.
3. **Sources** — `response.wikilinks` via the `render_wikilink` filter (omitted on refusal — `wikilinks` defaults to `[]`).
Plus the **audit footer** (`.ans-footer`): monospace `correlation_id` + `.tabular` token/node counts — the quiet audit trail the brand wants users to expect. Refusal/error reuse `.ans-flash{,-refused,-error}` with the same eyebrow labels; refusal keeps the collapsible `.ans-evidence` chunk disclosure.

All styling is **semantic classes in `style.css`** (`.ans-*`, `.claim-*`, `.conf-*`), NOT Tailwind utilities — so the answer component stays cohesive and adds nothing to the vendored `tailwind.css` subset. Eyebrow labels (`.ans-eyebrow`) mirror `.toc-sidebar-title` / `.pane-header`. Tests in `test_webui.py` assert on the rendered TEXT (summary, claim, "Sources", wikilink href, "Refused", correlation_id) — keep those strings present when restyling.

**Artifact-scope note (`.ans-scope`, #256, 2026-05-26).** When `response.artifact_scope_doc_ids` is non-empty (the query NAMED an artifact, so retrieval was re-scoped to its doc(s) — see `agents/artifact_scope.py`), a quiet `.ans-scope` line renders just above `.ans-footer` on BOTH the answered and refused paths: "Scoped to the document(s) you named:" + each scoped document by **human title** as a quiet `.ans-scope-doc` link-tag (links to the doc's detail view; the stable doc-id is the `href` target + the `title=` hover tooltip). The `/ask` route resolves each `artifact_scope_doc_ids` entry → title via `read_document_title` (cheap frontmatter read, falls back to the doc-id) and passes a `scope_docs` `[{doc_id, title}]` list to the template — titles are a presentation concern, so the `FinalResponse` / MCP / CLI keep the stable doc-ids. It's most valuable on a REFUSAL — it explains WHY the pool was narrowed (e.g. scoped to the firewall doc, which lacks the asked-about VLAN range). Empty `artifact_scope_doc_ids` → no note (the full-corpus path). Pinned by `test_webui.py::test_ask_refusal_surfaces_artifact_scope_titles` (+ the absence assertion in `test_ask_renders_refusal`).

## Adding a route

1. Define it in `webui/app.py:create_app` (the factory pattern is what `test_webui.py` depends on).
2. If it returns HTML, render a Jinja template via `templates.TemplateResponse(request, "name.html", ctx)`.
3. If it's an HTMX target, name the partial `_name.html` and **omit** `{% extends %}`.
4. Add a test in `tests/integration/test_webui.py` using `TestClient(create_app())`, or unit-test pure render helpers in `tests/unit/test_webui_rendering.py`.

## Two inline-edit flows (both HTMX view/edit toggles)

- **Body**: the `edit` button swaps `#md-pane` (`/documents/{id}/edit` → form; `/documents/{id}/review` POST writes through `vault.write_document` with optimistic-CAS conflict handling; `/documents/{id}/body` is the view partial).
- **Title** (2026-05-24): the `rename` button swaps `#doc-title` (`/documents/{id}/title/edit` → form; `/documents/{id}/title` POST → `index.retitle_document`; `/documents/{id}/title` GET is the view partial). Partials `_document_title.html` / `_document_title_edit.html`. The POST calls `retitle_document` directly — the one sanctioned `webui → index` write path besides the `graph_store` test seam — because the rename must fan the title out to the FTS/vector/graph copies *without* a re-embed, which the watcher's partial reindex can't do (the body, hence every chunk, is unchanged).

## When in doubt

The web UI's job is the *visual* parts of the workflow the CLI can't do well (per IMPLEMENTATION-PLAN §1.10): side-by-side preview of source PDF and extracted Markdown (Phase 4), graph visualisation, per-document annotation correction. Everything else should be a CLI invocation. If a route doesn't make a visual workflow easier, push back on adding it.
