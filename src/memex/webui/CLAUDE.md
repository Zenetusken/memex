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
- **WCAG 2.1 AA** — contrast minimums (4.5:1 body text, 3:1 large), keyboard focus rings visible, color is never the sole carrier of information (confidence chips use color + label). **Secondary / label / metadata text floors at `zinc-400` (rgb 161 161 170 ≈ 7.7:1 on the `zinc-950` page) — NOT `zinc-500` (≈4.07:1, fails AA for normal text) or `zinc-600` (≈2.56:1).** `zinc-500/600` were swept out of text in 2026-05-27's contrast pass and survive only as decorative borders / dividers (e.g. `.pane-divider`, the middot separators, `border-left-color`), which are exempt. Don't reintroduce `text-zinc-500/600` (or those rgb values as a `color:`) for anything a user reads — bump to `zinc-400` (the established secondary tier; primary stays `zinc-100`).

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
2. **Grounded claims · N** — each `claim` as a `.claim` card: the assertion + a meta row with the **confidence chip** (`.conf-{high,medium,low}`, colour **and** label per WCAG 1.4.1 — never colour alone), a "source" label, and the source rendered by **human title › section** (`.claim-source-link`, linked to the doc section) with the raw `docid#hash` only as the `title=` tooltip. A claim whose chunk isn't in `used_chunks` (e.g. a synthetic table/SQL chunk) falls back to the monospace `.chunk-id`.
3. **Sources** — `response.wikilinks` via the `render_wikilink` filter, rendered by **document title › section** (not raw `[[doc#section]]`); omitted on refusal (`wikilinks` defaults to `[]`).

**Sources-by-title (2026-05-27).** The route resolves the source view-model with `app._source_view(response)` → `(chunk_refs, doc_titles)` from the cited chunks' OWN `document_title` + `heading_path` (NO extra I/O — the same data the refusal panel shows), passed to both `_answer.html` and `_summary.html`. `chunk_refs` (`chunk_id → {title, section, href}`) drives the per-claim `.claim-source-link`; `doc_titles` (`doc_id → title`) is the second arg to `render_wikilink(wikilink, titles)`, which now labels by title (the body-rewrite `_anchor_for_wikilink` still keeps the literal `[[..]]` for raw-markdown fidelity in the `<pre>` — the two deliberately diverge). The stable doc-id/chunk-id survive as `href` + `title=` tooltip. Pinned by `test_webui_rendering.py` (title-map label + tooltip) + `test_webui.py::test_ask_renders_sources_and_claims_by_title`.
Plus the **audit footer** (`.ans-footer`): monospace `correlation_id` + `.tabular` token/node counts — the quiet audit trail the brand wants users to expect. Refusal/error reuse `.ans-flash{,-refused,-error}` with the same eyebrow labels; refusal keeps the collapsible `.ans-evidence` chunk disclosure.

All styling is **semantic classes in `style.css`** (`.ans-*`, `.claim-*`, `.conf-*`), NOT Tailwind utilities — so the answer component stays cohesive and adds nothing to the vendored `tailwind.css` subset. Eyebrow labels (`.ans-eyebrow`) mirror `.toc-sidebar-title` / `.pane-header`. Tests in `test_webui.py` assert on the rendered TEXT (summary, claim, "Sources", wikilink href, "Refused", correlation_id) — keep those strings present when restyling.

**Artifact-scope note (`.ans-scope`, #256, 2026-05-26).** When `response.artifact_scope_doc_ids` is non-empty (the query NAMED an artifact, so retrieval was re-scoped to its doc(s) — see `agents/artifact_scope.py`), a quiet `.ans-scope` line renders just above `.ans-footer` on BOTH the answered and refused paths: "Scoped to the document(s) you named:" + each scoped document by **human title** as a quiet `.ans-scope-doc` link-tag (links to the doc's detail view; the stable doc-id is the `href` target + the `title=` hover tooltip). The `/ask` route resolves each `artifact_scope_doc_ids` entry → title via `read_document_title` (cheap frontmatter read, falls back to the doc-id) and passes a `scope_docs` `[{doc_id, title}]` list to the template — titles are a presentation concern, so the `FinalResponse` / MCP / CLI keep the stable doc-ids. It's most valuable on a REFUSAL — it explains WHY the pool was narrowed (e.g. scoped to the firewall doc, which lacks the asked-about VLAN range). Empty `artifact_scope_doc_ids` → no note (the full-corpus path). Pinned by `test_webui.py::test_ask_refusal_surfaces_artifact_scope_titles` (+ the absence assertion in `test_ask_renders_refusal`).

**Consented A→B escalation (`.ans-escalate`, §11 / ADR-0016, 2026-06-02).** On a REFUSAL (and only when `expert_enabled`), `_answer.html` renders an **amber-tinted `.ans-escalate` form** (a `<form>` POSTing to `/bridge`) inviting the user to re-reason the SAME question over the SAME scope through the reason-then-ground bridge — **explicitly USER-CHOSEN, never automatic** (ADR-0013 R3). It carries the original `question` + the refusal's `escalate_scope_ids` as hidden inputs (so the bridge runs over the same docs, not the whole vault); both come from `ProgressEntry.{question,scope_doc_ids}` threaded through the long-poll (`webui/progress.py`) into the refusal render (`ask_status` sets `ctx["question"]`/`ctx["escalate_scope_ids"]`). It is INSIDE the refusal `{% else %}` branch, so it can't render on an answered response. Semantic `.ans-escalate*` CSS (amber border — the caution tier, NOT the grounded action-blue; zinc-400 note for AA), NOT Tailwind. Pinned by `test_webui.py` (`test_ask_refusal_offers_escalation_when_expert_enabled` / `…no_escalation_when_expert_disabled` / `…answered_has_no_escalation` / `…scoped_refusal_escalation_carries_scope`).

## Bridge "Analysis" surface (`/bridge` + `bridge.html`/`_bridge.html`, §11 / ADR-0016, 2026-06-02)

Gated on `expert_enabled` (the nav "Analysis" link in `base.html` is hidden when off, like the sibling **Expert** tab). `GET /bridge` is the composer (a base-extending page); `POST /bridge` starts `reason_then_ground` in a background task (keyed by a cid) and returns the shared `_progress.html` fragment, which long-polls `GET /bridge/status?cid=&v=` (`BRIDGE_PHASES` = Retrieving evidence → Reasoning → Grounding claims; `bridge_phase_index` in `webui/progress.py`) until `_bridge.html` swaps in. `_bridge.html` renders the labelled **UNGROUNDED analysis** followed by the **GROUNDED-CLAIMS subset** (the survivors that passed the same vault-grounding check as `/ask`), reusing the answer panel's `.ans-*`/`.claim-*`/`.conf-*` chrome + an amber "ungrounded" provenance banner. It is ALSO the target of the `.ans-escalate` escalation: the SAME route handles a scope-carrying POST → bridge over those docs; a bare composer POST → whole vault (`scope_doc_ids: list[str] = Form([])`). The progress-registry `response` union is widened to `FinalResponse | ExpertAnswer | BridgeAnswer`. NOT MCP. The agent's `/ask` graph is never touched ⇒ HARD-gate-neutral. Pinned by `test_webui.py` (the bridge POST→progress→`_bridge.html` flow, zero-grounded note, nav-shows/hides-when-enabled/disabled).

## Document scope-picker (`index.html` + `.scope-picker`, 2026-05-27)

The Notebook-LM-style picker: the Ask page lets the user tick documents to scope the question to them. The `index` route lists the vault docs (`list_documents` + `read_document_title`, title-sorted) and passes `documents=[{doc_id, title}]`; `index.html` renders a **native `<details>` disclosure** ("Scope to documents — optional; leave all unchecked to search the whole vault") with a scrollable checklist — each row a `<input type="checkbox" name="scope_doc_ids" value="{doc_id}">` + the human **title** + a monospace **doc-id** (`.scope-picker-id`). **No JS** (native disclosure + checkboxes, inside the `<form>` so the ticks submit with the question). `/ask` takes `scope_doc_ids: list[str] = Form([])` and forwards it to `answer_query(scope_doc_ids=...)`; the agent scopes retrieval to exactly those docs (the SAME `resolve_artifact_scope` node, explicit-wins-over-inference — see `src/memex/CLAUDE.md`). The route passes `scope_source="selected"` so the `.ans-scope` note reads **"Scoped to your selected document(s):"** (vs "the document(s) you named:" for an inferred artifact); empty selection → the full-corpus path + no note. Styling is **semantic `.scope-picker*` in `style.css`** (zinc, the action-blue reserved for the tick), NOT new Tailwind utilities. Pinned by `test_webui.py::test_index_renders_doc_picker` + `test_ask_scopes_to_selected_docs_and_labels_note`.

**Saved scope sets (2026-05-27)** persist a selection so it's reusable. The picker is factored into the partial **`_scope_picker.html`** (the `index` page `{% include %}`s it; the three `/scope-sets*` routes return it as the HTMX swap target `#scope-picker`, `hx-swap="outerHTML"`). It renders, above the doc checklist, a **saved-set bar** of chips — each an `apply` button (`{{ s.name }}` + a `.scope-set-count` badge) and a `✕` `delete` button — and, below the checklist, a **save-as control** (`set_name` text input + a "Save as set" button). All three controls are `type="button"` with their own `hx-post` (so they DON'T submit the Ask form): `POST /scope-sets` (save the ticked `scope_doc_ids` under `set_name`), `POST /scope-sets/apply` (resolve a set → re-render with its docs **pre-ticked** server-side — the boxes feed the unchanged `/ask`, so applying a set is just a convenience that pre-checks the existing checkboxes; `/ask` gains NO `scope_set` param), `POST /scope-sets/delete` (remove + re-render). Apply/delete pass the set name via `hx-vals` JSON (`{{ s.name | tojson }}` — `tojson` unicode-escapes `'`/`<`/`>`, so a name with a quote stays valid in the single-quoted attribute). A `.scope-flash` line ("Saved … / Applied … / Deleted …" in blue; validation errors in `.scope-flash-error` red) reports each action. The shared `_scope_picker_context(vault_path, *, checked_ids, flash, picker_open)` helper builds the context (title-sorted docs + saved sets + ticked ids); it **swallows `VaultIntegrityError`** from `list_scope_sets` to an empty list so a corrupt `scope_sets.json` never 500s the Ask page (the CLI surfaces it loudly). Storage + the agent contract live in `core/scope_sets.py` / `src/memex/CLAUDE.md` / `docs/specs/scope-sets.md`. Semantic `.scope-set*` + `.scope-flash*` CSS (zinc, action-blue apply-hover, red only for the delete-hover + errors). Pinned by `test_webui.py::test_scope_set_save_apply_delete_round_trip` (+ the empty-name / no-docs / unknown-apply flash-error tests).

**Scope-set suggestions (2026-05-29, ADR-0011 discovery, `04ef4e9`)** surface docs the entity graph relates to the current selection, each tick-able to expand the scope. `_scope_picker_context` now computes a `suggested` list from `checked_ids` (empty selection ⇒ NO graph query) via the shared **`_related_for_docs(vault_path, seed_ids)`** (the same merge/dedup/seed-exclude/re-rank/cap helper the `/ask` Related panel uses — `_related_for_answer` delegates to it). So the apply/save re-renders auto-show a **"Suggested additions"** section; a fourth control **`POST /scope-sets/suggest`** (the "Suggest related" button, mirrors the save button — `type="button"` + `hx-post`, no Ask-form submit) feeds the ticked `scope_doc_ids` for a manual selection + sets a count flash. Each suggestion is a `name="scope_doc_ids"` checkbox + `/entity?name=` why-related tags; ticking one + a re-render moves it to the checked main list and drops it from suggestions (the dup checkbox is harmless — the scope path dedups). Fail-open: a missing graph → no section + "No related documents found". Semantic `.scope-suggest*` CSS (reuses `.related-entity*`; blue-tinted border). Pinned by `test_webui.py` (auto-on-apply excludes the set's docs; the Suggest button + count flash; empty-selection hint; graph-unavailable fail-open). **NB the discovery surfaces — doc-view Related, `/entity`, the `/ask` panel, scope suggestions — all need ENRICHED docs** (entities + MENTIONS edges); a doc that was ingested+indexed but never enriched (e.g. the CCNA bulk-ingest gap, fixed 2026-05-29) is invisible to them, so verify enrich ran (`manifest` has an `enrich` stage) when discovery is unexpectedly empty.

## Adding a route

1. Define it in `webui/app.py:create_app` (the factory pattern is what `test_webui.py` depends on).
2. If it returns HTML, render a Jinja template via `templates.TemplateResponse(request, "name.html", ctx)`.
3. If it's an HTMX target, name the partial `_name.html` and **omit** `{% extends %}`.
4. Add a test in `tests/integration/test_webui.py` using `TestClient(create_app())`, or unit-test pure render helpers in `tests/unit/test_webui_rendering.py`.

## Summarize action (`document.html` + `_summary.html`, ADR-0008, 2026-05-27)

The document view has a **Summarize** control in the header: a `detail` `<select>`
(brief/standard/detailed/**report** — the last is the multi-paragraph hierarchical
reduce, ADR-0010) + a button in one `<form>` that `hx-post`s to
`POST /documents/{id}/summarize` (`hx-target="#summary-pane"`, `hx-swap="innerHTML"`,
`hx-indicator="#summary-loading"`, `hx-disabled-elt="button[name='go']"`). The route
reads the `detail` Form field, calls `agents.document_summarizer.summarize_document`
(NOT the agent — summaries are their own path; see `src/memex/CLAUDE.md`), and renders
the partial `_summary.html`; a `MemexError` becomes a 503 `.ans-flash-error` banner.
No JS (native `<select>` + HTMX). The summary can take a moment (sequential
per-section map-reduce) — the `#summary-loading` `.htmx-indicator` covers it.

`_summary.html` **reuses the answer panel's zones** for cohesion: the abstract in
`.ans-answer` (the blue left-rule "Summary") — for `report` detail the body is
multi-paragraph (blank-line separated, ADR-0010), so the template splits
`response.summary` on `\n\n` into one `<p>` per paragraph inside ONE `.ans-answer`
block (the single blue rule spans them; `.ans-answer > p + p` spaces them); a
single-paragraph summary is exactly one `<p>` (unchanged). A report also surfaces a
quiet INFORM-ONLY **`.summary-confidence`** line under the footer (gated on
`response.report_confidence`) — the hybrid embedding+lexical faithfulness of the
generated prose vs its source digests (overall + breakdown; zinc-400, never coloured
as pass/fail, NOT a gate). Then the grounded key-points as `.claim`
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

## Related documents — "explore connections" (`document.html` + `.related-*`, 2026-05-28)

The doc view renders a **"Related documents"** section (just below `#summary-pane`):
the entity-graph discovery surface that replaced the retired passive `expand_graph`. The
`document` route fetches `GraphStore.related_documents(doc_id, limit=8)` — neighbours
ranked by shared-entity SPECIFICITY (IDF, NOT the unranked `neighbors()`; see
`src/memex/CLAUDE.md` / `index/graph_store.py`) — and passes `related` as a list of
`{doc_id, title, score, shared_entities}` dicts. The template renders each as a
title-link (`.related-link`, the one action-blue) + the connecting entities as quiet
`.related-entity` tags (the "why related"). **Fail-open**: an `ImportError` from
`GraphStore.open` (ryugraph absent) → `related=[]` → the section is omitted, the doc view
still renders (mirrors the `/graph` route; NEVER 500s). The graph read uses the documented
`webui → index/graph_store` edge. Semantic `.related-*` CSS in `style.css` (zinc + the
action-blue for the link), NOT new Tailwind. No JS. Pinned by `test_webui.py`
(`test_document_view_renders_related_documents` + `…survives_graph_unavailable`).
**The connecting-entity tags are now `/entity?name=` links** (`.related-entity-link`) — the
organic entry point from any doc into the entity-centric view below.

**The same surface also appears on the `/ask` ANSWER (2026-05-29, `ffe23fe`):** after an
answer renders, `_answer.html` shows a "Related documents" panel (below Sources, before the
scope note/footer) of docs related to the ones the answer CITED. It's built in
**`_related_for_answer(vault_path, response, …)`** and wired into **`_answer_context`** — THE
single seam feeding `_answer.html` on the long-poll completion (so adding the `related` key
there covers the answered path; the only other render site is the POST-error 400, which has
no answer). Answered-only; seeds from the distinct cited `document_id`s (`response.used_chunks`),
expands each via `related_documents`, merges/dedups/**excludes the cited docs**/re-ranks/caps;
ImportError fail-open → no panel. Reuses the SAME `.related-*` markup + a small `.ans-related`
wrapper. **HARD-gate-neutral + webui-only by construction** — presentation-layer, from the
already-returned `FinalResponse` + a read-only graph read; never touches the agent/answer/
refusal, and the CLI/MCP `ask` payloads are unchanged. Pinned by `test_webui.py`
(`test_ask_renders_related_panel_excluding_cited_docs` / `…survives_graph_unavailable` /
`test_ask_refusal_has_no_related_panel`).

## Connections view — the "Bridges" page (`/graph` + `graph.html` + `graph.css`, 2026-05-29, `b48f8b2`)

`GET /graph/{doc_id}` is the document-neighbourhood view, **redesigned from the Cytoscape
node-link "hairball" to a server-rendered, ranked, no-JS page**. The diagnosis (design panel +
the codebase record): a 1-hop neighbourhood is a **STAR** — no clusters, no paths between
leaves — so a node-link diagram spends its whole visual budget (position, edges, pan/zoom)
encoding a topology that doesn't exist, and draws ~45 identical grey spokes that read as "all
equally related" when the data is a sharp specificity RANKING. **Cytoscape is dropped from this
page** (air-gap + maintenance win; `static/cytoscape.min.js` + `static/graph.js` stay vendored
on disk but UNREFERENCED — `graph.css` keeps its retired `.cy-canvas`/`.inspector-*` block ONLY
so `graph.js`'s classes stay covered by `scripts/check-tailwind-coverage.py`).

Two **lenses** over the same graph data, toggled by `?group=` (plain `<a>` links → shareable
URL, no JS):
- **concept** (default) — `GraphStore.related_bridges(doc_id)`: related docs grouped UNDER the
  bridging ENTITY that connects them. Each bridge is a native `<details>` (top 2 `open`, rest
  collapsed) showing the entity (a `/entity?name=` traversal link) + the kind chip
  (`.entity-kind`) + a right-aligned strength bar + "bridges N", then its docs (`.related-link`)
  with their "·via" entity tags. The route splits multi-doc bridges (the headline) from
  single-doc ones (a folded `.bridge-tail` disclosure); if there are no multi-doc bridges the
  singles are promoted. Ranked by `strength = mean(IDF×kind_weight) × ln(1 + doc_count)` — see
  `src/memex/CLAUDE.md` / `docs/specs/graph-discovery.md` for why the fan-out is log-damped.
- **document** — `related_documents` rendered as a flat strength-ranked rail (`.rail-*`): a
  left strength bar + the doc-title link + its connecting entities. The Ranked-Rail design as
  the alternate lens.

The route computes proportional bar percentages (ordinal sugar — the count/rank are the honest
signal, so the % is never printed; WCAG 1.4.1). **Fail-open**: an `ImportError` from
`GraphStore.open` → `graph_available=False` + the amber "graph store unavailable" panel; an
empty neighbourhood → a quiet "No related documents found" note; a `VaultIntegrityError` on the
doc → 404. Semantic `.bridge-*` / `.rail-*` / `.lens-*` CSS in `graph.css` (zinc + the one blue
accent — the strength-bar fill is a quieted blue, like `.ans-answer`'s structural rule;
secondary text floors at zinc-400). Reuses the shipped `.related-entity-link` / `.entity-kind`
markup; NOT new Tailwind. The `webui → index/graph_store` test-seam edge is unchanged
(`test_webui.py` monkeypatches `GraphStore.open`; the fake provides BOTH `related_documents` +
`related_bridges`). Live-verified in Chrome (both lenses legible, clean console — the page ships
zero scripts). Pinned by `test_webui.py` (`test_graph_renders_bridges_view` /
`test_graph_document_lens_renders_ranked_list` / `…shows_unavailable…` / `…404s…`).

## Entity-centric discovery view (`/entity` + `entity.html` + `.entity-*`, ADR-0011, 2026-05-28)

The visual surface for "everything about entity X" — the second graph-discovery surface
(spec `docs/specs/graph-discovery.md`). `GET /entity?name=` calls `entity_overview` (the
**`webui → retrieve` boundary edge**, documented like `webui → parse`; imported at module
top so `test_webui.py` can monkeypatch `memex.webui.app.entity_overview`) and renders
`entity.html` (a full page, `{% extends base %}` — NOT an HTMX partial). Three states:
**resolved** → the graph profile (the "in graph" badge + kind chips + true `doc_count`; the
co-occurring neighbourhood, **each tag a `/entity?name=` link so the user TRAVERSES the
graph**; the mentioning docs as links) + the scoped passages by human title › section;
**unknown name** → a quiet "not a known entity" badge (colour + label, never colour alone) +
an explanatory note (acronym-vs-expansion) + the whole-corpus FTS passages (no co-occurring
section); **empty name** → just the lookup form. A header **"Entities" nav link** (`base.html`)
+ the doc-view related-entity tags are the entry points. Fail-open is inherited from the
orchestrator (a missing graph never raises — the route doesn't even catch). `_passage_refs`
is the passage view-model (title › section + `?page=N#slug` href + a bounded ~480-char
preview), mirroring `_source_view`. **No JS** (a GET `<form>`). Semantic `.entity-*` CSS in
`style.css` (zinc + the single action-blue; secondary text floors at zinc-400 for AA; the
co-occurring tags + mention links hover to the action-blue), NOT new Tailwind. Live-validated
(Chrome e2e): `DNS` → the full resolved profile + a coherent co-occurring cluster; `STP` →
the graceful FTS fallback. Pinned by `test_webui.py` (`test_entity_view_renders_resolved_profile`
/ `…unknown_falls_back_to_fts` / `test_entity_lookup_form_renders_without_name` + the
related-tag-is-`/entity`-link assertion).

**Acronym ↔ expansion suggestions (2026-05-28).** `entity.html` renders `profile.suggestions`
(the deterministic initialism bridge — see `src/memex/CLAUDE.md`) as an **"Also see"** block
on the resolved path (above co-occurring) and a **"Did you mean?"** block on the unresolved
path (alongside the "not a known entity" note), each a `/entity?name={{ s.name|urlencode }}`
traversal link. Semantic `.entity-suggestion-*` CSS — a blue-TINTED border (`rgb(37 99 235 /
0.4)`) distinguishes a bridge ("same concept, other name") from the plain zinc co-occurring
tags. When `suggestions==[]` (e.g. the true `STP` miss) NOTHING renders — the honest FTS
fallback is unchanged. Pinned by `test_webui.py` (`…renders_did_you_mean_when_unresolved`,
`…unknown_with_no_bridge_stays_honest`, + the "Also see" assertion in the resolved test).

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

**Click-source → jump-to-page (2026-05-27).** A claim's `.claim-source-link`
carries `data-page="N"` (+ a visible "· p. N", + `?page=N` in the href) when the
chunker attributed its chunk to a source page (`Chunk.page`, populated from the
parser's per-page `char_count` in the manifest — see `src/memex/CLAUDE.md`). Each
preview `<img>` gets `id="page-{1-based}"`. An inline `<script>` in `document.html`
(gated on `has_preview`, vanilla DOM, no vendored deps) does two things: on load it
reads `?page=N` and `scrollIntoView`s `#page-N` (the cross-page case — a same-tab Ask
whose cited source click lands on the doc page); and it intercepts same-doc
`[data-page]` clicks to scroll the preview + the markdown section + `replaceState`
the URL (no reload — a same-doc-different-query href would otherwise full-navigate).
Cross-doc `[data-page]` links navigate normally (the `?page=N` survives → the target
doc scrolls on load). A markdown-only doc (no `#pdf-pages`) no-ops. Pinned by
`test_webui.py` (`data-page`/`?page=N`/"· p. N" on the source-link + `id="page-N"` +
the `scrollPreviewTo` hook present). Existing docs need `memex index <doc> --force`
(or a re-ingest) to backfill the page attribution — the content-addressed partial
index skips it on a same-content re-parse; new ingests get it automatically.

## Live progress (long-poll, `_progress.html` + `webui/progress.py`, 2026-05-27)

Both `/ask` AND **Summarize** share one long-poll progress widget. The summarize
flow mirrors `/ask`: `POST /documents/{id}/summarize` runs `summarize_document` in a
background `_run_summarize` task (keyed by a cid) and returns `_progress.html` into
`#summary-pane`, which long-polls `GET /documents/{id}/summarize/status?cid=&v=`.
The summarizer is LINEAR (not a graph), so it reports progress via an opt-in
`on_phase` sink (not a LangGraph callback): it emits `"Summarizing · section k of
N"` per section (the dominant phase — the COUNTER is the real signal), plus "Key
figures" (tabular), "Reducing", "Composing". `progress.set_phase` stores the full
label; `summary_phase_view(label)` splits it into `(active_index in SUMMARY_PHASES,
eyebrow detail)` so the step list (Reading → Summarizing → Reducing → Composing)
highlights the base phase while the eyebrow shows "section 3 of 9". The status route
renders `_summary.html` on completion (via `_source_view`, identical to before). The
`_progress.html` fragment is **generic** — the route passes `poll_url` (the full
next-poll URL incl. `v`) + `phases`/`active_index`/`elapsed` + an optional eyebrow
`detail`; the shared `ProgressRegistry`/`_progress_expired.html` are reused as-is.
The Summarize form dropped its `#summary-loading`/`hx-indicator`. Pinned by
`test_webui.py` (summarize POST→progress, status branches, a live progression) +
`test_progress.py` (`summary_phase_view`) + `test_document_summarizer.py`
(`on_phase` sequence + cid threading).

## Live ask-progress (long-poll, `_progress.html` + `webui/progress.py`, 2026-05-27)

`POST /ask` no longer blocks ~60–90 s. It starts the agent in a background
`asyncio.Task` (`_run_ask`, keyed by a `cid` ULID — held strongly in the registry
so the loop can't GC it) and IMMEDIATELY returns the `_progress.html` fragment into
`#answer`. That fragment **long-polls** `GET /ask/{cid}/status?v=N`
(`hx-trigger="load"` + `hx-swap="outerHTML"` on itself): the status route HOLDS the
connection (`ProgressRegistry.wait_for_change`) until the phase advances past `v`,
the run finishes, or a ~1 s keepalive — so the step list (Retrieving → Reranking →
Assessing → Drafting → Grounding → Checking relevance → Composing) updates the
INSTANT a node transition happens (SSE-like), while a held phase still ticks its
per-phase elapsed timer. This is the user's "as accurate as real-time SSE" ask done
with zero new vendored JS — it adapts to our agent's bursty timing (instant across
the quick LLM phases, a heartbeat during the slow CPU rerank). When the run is done
the route renders `_answer.html` (or the error / `_progress_expired.html` fragment),
none of which carry an `hx-trigger` → the poll chain stops by itself. **Every status
response is HTTP 200** — the outcome is content, decoupled from the HTTP status, so
the HTMX 1.9.10-vs-2.x `responseHandling` quirk in `base.html` is irrelevant here.

The agent stays oblivious: `answer_query(correlation_id=cid, on_node=…)` appends an
observe-only LangGraph callback (`_NodeProgressHandler`) that maps node starts →
node names; `webui/progress.py` owns the `ProgressRegistry` (in-process,
single-worker-safe, `cid → ProgressEntry`, event-driven via `asyncio.Event` + a
monotonic `version`, lazy TTL+cap cleanup, evict-on-delivery) and the node→phase
map. `_answer.html` is UNCHANGED — the status route reproduces the old synchronous
context via `_answer_context` (scope-doc titles + `_source_view` + scope_source).
The Ask form dropped its old `#loading`/`hx-indicator`/`hx-disabled-elt` (the
progress fragment is the in-flight feedback; a fresh ask while one runs just gets a
new cid, abandoning the prior poll, which the TTL sweep reaps). Semantic
`.progress-*` CSS only (step text floored at zinc-400 for AA; the active dot's pulse
gated behind `prefers-reduced-motion`), no new Tailwind. Pinned by `test_webui.py`
(the `_ask_to_completion` poll-flow helper + status branches + a live
`httpx.AsyncClient` progression with a gate) + `test_progress.py` (the callback
discriminator, the registry incl. `wait_for_change`, the `answer_query` threading).

## Two inline-edit flows (both HTMX view/edit toggles)

- **Body**: the `edit` button swaps `#md-pane` (`/documents/{id}/edit` → form; `/documents/{id}/review` POST writes through `vault.write_document` with optimistic-CAS conflict handling; `/documents/{id}/body` is the view partial).
- **Title** (2026-05-24): the `rename` button swaps `#doc-title` (`/documents/{id}/title/edit` → form; `/documents/{id}/title` POST → `index.retitle_document`; `/documents/{id}/title` GET is the view partial). Partials `_document_title.html` / `_document_title_edit.html`. The POST calls `retitle_document` directly — the one sanctioned `webui → index` write path besides the `graph_store` test seam — because the rename must fan the title out to the FTS/vector/graph copies *without* a re-embed, which the watcher's partial reindex can't do (the body, hence every chunk, is unchanged).

## When in doubt

The web UI's job is the *visual* parts of the workflow the CLI can't do well (per IMPLEMENTATION-PLAN §1.10): side-by-side preview of source PDF and extracted Markdown (Phase 4), graph visualisation, per-document annotation correction. Everything else should be a CLI invocation. If a route doesn't make a visual workflow easier, push back on adding it.
