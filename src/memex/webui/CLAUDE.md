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

## Document scope-picker (`index.html` + `.scope-picker`, 2026-05-27)

The Notebook-LM-style picker: the Ask page lets the user tick documents to scope the question to them. The `index` route lists the vault docs (`list_documents` + `read_document_title`, title-sorted) and passes `documents=[{doc_id, title}]`; `index.html` renders a **native `<details>` disclosure** ("Scope to documents — optional; leave all unchecked to search the whole vault") with a scrollable checklist — each row a `<input type="checkbox" name="scope_doc_ids" value="{doc_id}">` + the human **title** + a monospace **doc-id** (`.scope-picker-id`). **No JS** (native disclosure + checkboxes, inside the `<form>` so the ticks submit with the question). `/ask` takes `scope_doc_ids: list[str] = Form([])` and forwards it to `answer_query(scope_doc_ids=...)`; the agent scopes retrieval to exactly those docs (the SAME `resolve_artifact_scope` node, explicit-wins-over-inference — see `src/memex/CLAUDE.md`). The route passes `scope_source="selected"` so the `.ans-scope` note reads **"Scoped to your selected document(s):"** (vs "the document(s) you named:" for an inferred artifact); empty selection → the full-corpus path + no note. Styling is **semantic `.scope-picker*` in `style.css`** (zinc, the action-blue reserved for the tick), NOT new Tailwind utilities. Pinned by `test_webui.py::test_index_renders_doc_picker` + `test_ask_scopes_to_selected_docs_and_labels_note`.

**Saved scope sets (2026-05-27)** persist a selection so it's reusable. The picker is factored into the partial **`_scope_picker.html`** (the `index` page `{% include %}`s it; the three `/scope-sets*` routes return it as the HTMX swap target `#scope-picker`, `hx-swap="outerHTML"`). It renders, above the doc checklist, a **saved-set bar** of chips — each an `apply` button (`{{ s.name }}` + a `.scope-set-count` badge) and a `✕` `delete` button — and, below the checklist, a **save-as control** (`set_name` text input + a "Save as set" button). All three controls are `type="button"` with their own `hx-post` (so they DON'T submit the Ask form): `POST /scope-sets` (save the ticked `scope_doc_ids` under `set_name`), `POST /scope-sets/apply` (resolve a set → re-render with its docs **pre-ticked** server-side — the boxes feed the unchanged `/ask`, so applying a set is just a convenience that pre-checks the existing checkboxes; `/ask` gains NO `scope_set` param), `POST /scope-sets/delete` (remove + re-render). Apply/delete pass the set name via `hx-vals` JSON (`{{ s.name | tojson }}` — `tojson` unicode-escapes `'`/`<`/`>`, so a name with a quote stays valid in the single-quoted attribute). A `.scope-flash` line ("Saved … / Applied … / Deleted …" in blue; validation errors in `.scope-flash-error` red) reports each action. The shared `_scope_picker_context(vault_path, *, checked_ids, flash, picker_open)` helper builds the context (title-sorted docs + saved sets + ticked ids); it **swallows `VaultIntegrityError`** from `list_scope_sets` to an empty list so a corrupt `scope_sets.json` never 500s the Ask page (the CLI surfaces it loudly). Storage + the agent contract live in `core/scope_sets.py` / `src/memex/CLAUDE.md` / `docs/specs/scope-sets.md`. Semantic `.scope-set*` + `.scope-flash*` CSS (zinc, action-blue apply-hover, red only for the delete-hover + errors). Pinned by `test_webui.py::test_scope_set_save_apply_delete_round_trip` (+ the empty-name / no-docs / unknown-apply flash-error tests).

## Adding a route

1. Define it in `webui/app.py:create_app` (the factory pattern is what `test_webui.py` depends on).
2. If it returns HTML, render a Jinja template via `templates.TemplateResponse(request, "name.html", ctx)`.
3. If it's an HTMX target, name the partial `_name.html` and **omit** `{% extends %}`.
4. Add a test in `tests/integration/test_webui.py` using `TestClient(create_app())`, or unit-test pure render helpers in `tests/unit/test_webui_rendering.py`.

## Summarize action (`document.html` + `_summary.html`, ADR-0008, 2026-05-27)

The document view has a **Summarize** control in the header: a `detail` `<select>`
(brief/standard/detailed) + a button in one `<form>` that `hx-post`s to
`POST /documents/{id}/summarize` (`hx-target="#summary-pane"`, `hx-swap="innerHTML"`,
`hx-indicator="#summary-loading"`, `hx-disabled-elt="button[name='go']"`). The route
reads the `detail` Form field, calls `agents.document_summarizer.summarize_document`
(NOT the agent — summaries are their own path; see `src/memex/CLAUDE.md`), and renders
the partial `_summary.html`; a `MemexError` becomes a 503 `.ans-flash-error` banner.
No JS (native `<select>` + HTMX). The summary can take a moment (sequential
per-section map-reduce) — the `#summary-loading` `.htmx-indicator` covers it.

`_summary.html` **reuses the answer panel's zones** for cohesion: the abstract in
`.ans-answer` (the blue left-rule "Summary"), the grounded key-points as `.claim`
cards with `.conf-{high,medium,low}` chips (colour **and** label, WCAG 1.4.1) + the
monospace source `.chunk-id`, the Sources via the `render_wikilink` filter, and the
`.ans-footer` audit line (correlation_id + token/section counts). It adds a
collapsible **`.summary-sections`** `<details>` ("By section · N") whose
`.summary-section` items each show the section title, digest, and per-section cited
points — semantic `.summary-*` CSS in `style.css` (zinc + the one action-blue,
mirrors `.ans-*`/`.toc-*`), NOT new Tailwind. A zero-grounded summary renders the
`.ans-flash-refused` "No summary" partial (the HARD gate, surfaced). Chrome-extension
e2e'd (the button → a grounded render). Tests in `test_webui.py` assert the rendered
zones; keep "Summary"/"Key points"/"Sources"/refusal strings present when restyling.

## Live co-residence mode hot-switch (`/resources` + `_resources.html`, ADR-0007, 2026-05-27)

`/resources` shows the active co-residence mode + a comparison table; each non-active
row has an **Apply** button (`POST /resources/mode`, `hx-target="#resources-pane"`,
`hx-swap="outerHTML"`, `hx-indicator="#mode-loading"`, `hx-disabled-elt="this"`) that
switches the mode **live** — the realization of ADR-0007's runtime transition.
`_apply_mode(mode)` (in `app.py`) does both halves: **(1) app-side** — flips
`get_settings().models.co_residence_mode` (the registry shares that exact
`ModelSettings` object — see `src/memex/CLAUDE.md`) and `await registry.unload(
"embedder"/"reranker")` so the next retrieval reloads on the new device (the unload
takes the per-model lock, so an in-flight `/ask` finishes first — the quiesce);
**(2) orchestrator-side** — when the mode prescribes a posture, `daemon.restart(
gpu_fraction, max_model_len)` (blocks ~40 s; the `#mode-loading` indicator covers it).

This adds two **documented boundary exceptions**: `webui → daemon` and `webui →
models.registry` (the hot-switch inherently orchestrates the daemon + the registry;
imported at module top so `test_webui.py` monkeypatches `memex.webui.app.{daemon_status,
daemon_restart,get_registry}`). Switches are serialized by a per-app `asyncio.Lock`.
A daemon failure (`MemexError`) → a 503 `.mode-flash-error`; an unknown mode → 400.

The route returns the `_resources.html` partial (the swap target `#resources-pane`)
reflecting the new active profile + a `.mode-flash`. The **header chip** (`base.html`,
a static `active_mode_label` jinja global) is kept current two ways: `env.globals` is
updated (so the next full page load is right) AND the partial emits an
**`hx-swap-oob`** `#mode-chip-v` fragment (gated on an `oob_chip` flag — POST only, so
the GET full page has no duplicate id) so the chip updates live without a reload.
Validated live (Chrome e2e): Apply `fast` → the daemon restarted 24,576→6,144, the
panel + chip flipped to Fast. Pinned by `test_webui.py` (restart args + device unload +
manual-skips-restart + unknown-mode error).

## Source-preview pane (`document.html` + `pdf-page` images, 2026-05-27)

The side-by-side preview (the Phase-4 "source ↔ extracted markdown" view) renders
the source as **server-rasterised page images**, NOT an embedded PDF. The original
`<iframe src=".../source">` left the pane **blank**: a browser's "download PDFs
instead of opening" setting (a per-user pref no `Content-Disposition` can override)
plus general iframe-PDF flakiness defeat native in-browser PDF rendering. So the
document route computes a **preview PDF** via `_find_preview_pdf` — the doc's own
`source.pdf`, **or** an Office/ODF doc's `converted.pdf` (what the parse stage
actually rendered; the `.pptx`/`.docx` can't render inline) — and passes
`has_preview` + `preview_pages`; the template emits one `<img loading="lazy"
src="/documents/{id}/source/page/{n}">` per page (`.pdf-pages`/`.pdf-page` CSS). The
new route `GET /documents/{id}/source/page/{n}` rasterises a **0-based** page to PNG
via `memex.parse.pdf_render.render_pdf_page_png` (a documented `webui → parse` edge;
`pdf_render` is the LIGHT pypdfium2+PIL twin of `vlm_backend`'s renderer — no ML deps,
so the import stays cheap), `asyncio.to_thread`'d (CPU-bound) + browser-cached. This
works in **every** browser/setting (it's an `<img>`) and is the right affordance for
scans/handwriting (the original page sits beside its transcription). `pdf_render`
wraps pypdfium2 failures as `PDFPreviewError` (the route + the doc-view page-count both
catch ONE type → a corrupt PDF degrades to no-pane, never a 500). `/documents/{id}/
source` still serves the original file — now **`content_disposition_type="inline"`**
(was the default `attachment`, which forced a download) — for the pane header's
`download` link (which uses the HTML `download` attribute to force a save). Validated
live (Chrome e2e): a scanned handwritten note AND a `.pptx` deck both render their
pages beside the markdown. Pinned by `test_webui.py` (pane-split emits page-image
srcs / no `<iframe>` / inline disposition / Office `converted.pdf` "(rendered)" + the
page route serves PNG + out-of-range 404) + `test_pdf_render.py` (render + corrupt +
out-of-range).

## Two inline-edit flows (both HTMX view/edit toggles)

- **Body**: the `edit` button swaps `#md-pane` (`/documents/{id}/edit` → form; `/documents/{id}/review` POST writes through `vault.write_document` with optimistic-CAS conflict handling; `/documents/{id}/body` is the view partial).
- **Title** (2026-05-24): the `rename` button swaps `#doc-title` (`/documents/{id}/title/edit` → form; `/documents/{id}/title` POST → `index.retitle_document`; `/documents/{id}/title` GET is the view partial). Partials `_document_title.html` / `_document_title_edit.html`. The POST calls `retitle_document` directly — the one sanctioned `webui → index` write path besides the `graph_store` test seam — because the rename must fan the title out to the FTS/vector/graph copies *without* a re-embed, which the watcher's partial reindex can't do (the body, hence every chunk, is unchanged).

## When in doubt

The web UI's job is the *visual* parts of the workflow the CLI can't do well (per IMPLEMENTATION-PLAN §1.10): side-by-side preview of source PDF and extracted Markdown (Phase 4), graph visualisation, per-document annotation correction. Everything else should be a CLI invocation. If a route doesn't make a visual workflow easier, push back on adding it.
