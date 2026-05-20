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

## Adding a route

1. Define it in `webui/app.py:create_app` (the factory pattern is what `test_webui.py` depends on).
2. If it returns HTML, render a Jinja template via `templates.TemplateResponse(request, "name.html", ctx)`.
3. If it's an HTMX target, name the partial `_name.html` and **omit** `{% extends %}`.
4. Add a test in `tests/integration/test_webui.py` using `TestClient(create_app())`.

## When in doubt

The web UI's job is the *visual* parts of the workflow the CLI can't do well (per IMPLEMENTATION-PLAN §1.10): side-by-side preview of source PDF and extracted Markdown (Phase 4), graph visualisation, per-document annotation correction. Everything else should be a CLI invocation. If a route doesn't make a visual workflow easier, push back on adding it.
