# Spec: Docling table-header recovery (flag-respecting serialization + gated re-attach)

**Status:** in implementation (2026-05-24). **Owner module:** `src/memex/parse/docling_worker.py` (+ a new `src/memex/parse/docling_tables.py` for the table logic, keeping the worker thin). **Depends on** `docling_core` (installed).

## Problem (grounded by page-23 inspection of the 10-K)

Docling's `MarkdownTableSerializer` **ignores `TableCell.column_header` flags** — it blindly treats `grid[0]` as the GFM header. Two failure classes result:

- **Class A — well-captured, mis-rendered.** Docling correctly flags the header rows (`column_header=True`), possibly **multiple** rows. The serializer emits only `grid[0]` as the header and pushes the *other* header rows into the body. Example (p23 summary table): `column_header rows=2` (`"Fiscal 2026 Result"` ×4, then `Revenue|Gross Margin|Operating Income|Diluted EPS`) → the real labels (row 1) become a data row.
- **Class B — mis-structured/transposed.** Docling tags the wrong cells as headers and the real column labels are **detached** into preceding `section_header` + `text` items. Example (p23 segment table): grid row0 = `[('Revenue','.'), ('$193.5B','H'), ('$22.5B','H'), ('$215.9B','H')]` — values flagged as headers; the labels "Compute & Networking" / "Graphics" / "Total" sit in a `### Compute & Networking` heading + a stray `Graphics Total` text line above the table. Result: GFM header = the Revenue *data* row; values can't be mapped to a column → false-refuse.

## Goal

Tables render with correct GFM headers so column values are answerable. Class A: cleanly fixed by respecting the flags. Class B: best-effort re-attach, **carefully gated** so it only fires when it's safe (label count exactly matches the value-column count), never corrupting other tables.

## Design

### Part 1 — `docling_tables.py`: header-aware table serialization (Class A, clean)

`HeaderAwareTableSerializer(MarkdownTableSerializer)` — override `serialize(*, item, doc_serializer, doc, **kwargs) -> SerializationResult`:
- Count leading header rows exactly like `TableItem._export_to_dataframe_with_options` (document.py ~2280): walk `item.data.grid`, a row is a header row iff **any** cell has `column_header=True`; stop at the first non-header row → `num_headers`.
- If `num_headers <= 1`: defer to `super().serialize(...)` (current behaviour is already correct for a single header row; no change). This keeps the vast majority of tables byte-identical.
- If `num_headers >= 2`: build a **single** GFM header row by concatenating the header rows per column with a space (mirror dataframe's `.`-join but use `" "` for readability, dedicating one space; collapse repeats), then render `grid[num_headers:]` as the body. Reuse the parent's cell-escaping (newline/pipe escaping) — factor the parent's cell-render into the override or call its helpers. Return a `SerializationResult(text=...)` in the same shape the parent returns (preserve any leading/trailing markers the parent adds, e.g. captions — inspect the parent and match).
- If `num_headers == 0`: defer to `super().serialize(...)` (Class B is handled by the re-attach pass which ADDS a header row, after which `num_headers>=1`).

### Part 2 — worker export swap (use the custom serializer; non-table output byte-identical)

In `docling_worker._convert_to_payload`, replace `doc.export_to_markdown()` with a helper `export_markdown_header_aware(doc) -> str` (in `docling_tables.py`) that constructs the doc serializer exactly as `DoclingDocument.export_to_markdown()` does (replicate its `MarkdownParams` construction from document.py ~6072 — same defaults: escape_html, labels, layers, etc.) but with `table_serializer=HeaderAwareTableSerializer()`, and returns `.serialize().text`. **REGRESSION INVARIANT:** on a document with no multi-header / no-re-attached table, `export_markdown_header_aware(doc) == doc.export_to_markdown()` byte-for-byte (only Class-A multi-header tables + re-attached tables differ). Apply the same swap to the per-page export path (`page.export_to_markdown()` → header-aware equivalent if feasible; if the per-page object doesn't support it, leave it — it's the fallback path).

### Part 3 — `docling_tables.py`: gated re-attach of detached headers (Class B, best-effort)

`reattach_detached_table_headers(doc) -> int` — run BEFORE export, mutate `doc` in place, return count re-attached. For each `TableItem`:
1. **Only consider mis-structured tables**: `num_headers == 0` after the flag count, OR the flagged header row is "data-like" (every flagged-header cell in the first column-header row matches a value pattern — contains a digit or `$`/`%` — while non-header col-0 cells look like row labels). Skip tables that already have a clean text header.
2. **Find detached labels**: via `doc.iterate_items()` collect the contiguous run of `SectionHeaderItem`/`TextItem` immediately preceding this `TableItem` in reading order (stop at the first non-text item or a blank). Tokenize: each item's text contributes label tokens by splitting on runs of 2+ spaces first, then single spaces only if needed to reach the count (prefer fewer splits). Collect the ordered label list.
3. **GATE (must hold or skip):** let `V = num_cols - (1 if the table's column 0 is a row-label column else 0)` (row-label column = col-0 cells are non-numeric text and not header-flagged). Re-attach ONLY if `len(labels) == V` exactly. If the count doesn't match after tokenization attempts, **skip** (leave the table unchanged). This is the safety gate the user requires.
4. **Mutate the grid**: prepend a new header row — for the row-label column (if any) an empty `TableCell(text="", column_header=True, ...)`, then one `column_header=True` cell per label; shift every existing cell's `start/end_row_offset_idx += 1`; clear the bogus `column_header=True` flags on the previously-mis-flagged value row; set `num_rows += 1`. After this, `num_headers == 1` with the correct labels, and `HeaderAwareTableSerializer`/the base serializer renders a correct GFM header.
5. Order: `reattach_detached_table_headers(doc)` runs in `_convert_to_payload` AFTER `_recover_heading_levels` + `_demote_misdetected_headers`, BEFORE `export_markdown_header_aware`. Log `docling.table_header_reattached count=N` to stderr.

### Anti-scope / safety
- The re-attach gate (exact count match) is the guardrail: when in doubt, skip. No table is ever made *worse*. Pin a negative test (count mismatch → unchanged).
- No transposition, no cell-value parsing beyond the value-vs-label digit/`$`/`%` heuristic for detection.
- Do not change the prose/non-table markdown (byte-identity invariant).
- No `IndexSettings` field.

## Tests (`tests/unit/test_docling_tables.py`, NEW — `pytest.importorskip("docling_core")`)
Construct real `docling_core` `TableData`/`TableCell`/`TableItem`/`DoclingDocument` objects in-memory.
1. **Class A**: a 2-header-row table (`column_header=True` on rows 0-1) → `HeaderAwareTableSerializer` emits ONE merged GFM header row (`"Fiscal 2026 Result Revenue"` etc.), body = data rows; the buggy base serializer would have dropped row 1 — assert the difference.
2. **Single-header table** → `HeaderAwareTableSerializer` output == base serializer output (no change).
3. **Re-attach happy path**: a transposed table (col0 row-labels, value cells flagged header) preceded by a heading + a text line whose tokens count to `num_cols-1` → after `reattach_detached_table_headers`, grid has a correct header row (labels), bogus flags cleared, `num_rows+1`; serialized GFM has the labels as the header. Use the segment-table shape exactly (`Revenue`/`Operating Income` rows; `Compute & Networking` + `Graphics Total` labels).
4. **Re-attach gate skip**: same shape but label tokens DON'T match `num_cols-1` (e.g. a multi-word label that over-splits) → table unchanged, returns 0.
5. **No detached labels** (no preceding text) → unchanged.
6. **Byte-identity**: build a small DoclingDocument with text + a single-header table; assert `export_markdown_header_aware(doc) == doc.export_to_markdown()`.
7. Worker wiring (duck-typed or skip-if-no-docling): `_convert_to_payload` calls reattach then header-aware export in the right order.

## Local gates (execution agent must pass)
- `uv run pytest tests/unit/test_docling_tables.py -q` → green
- `uv run pytest tests/ -q` → green (436 + new)
- `uv run pyright src/memex/parse/docling_worker.py src/memex/parse/docling_tables.py` → 0/0
- `uv run ruff check` + `ruff format` the changed files

## Independent validation gate (separate review subagent)
Re-run all gates; verify the **byte-identity invariant** on a no-multi-header doc; confirm the re-attach gate cannot corrupt a table (mutation only on exact count match); confirm the serializer reuses the parent's escaping; confirm wiring/order in `_convert_to_payload`; mutation-test the new tests; check pyright 0/0 + anti-scope (no IndexSettings field, prose unchanged). Iterate until no issues.

## GPU validation gate (orchestrator)
1. Re-parse the 10-K: `memex parse --force-docling 0e725ba0-2026-annual-report-web` (vLLM stopped → full GPU). Inspect the segment table in the regenerated `.md`: it should now have a header row `| | Compute & Networking | Graphics | Total |` (re-attach fired) and/or multi-header tables render with merged labels.
2. `memex index` + `memex enrich` the 10-K (output-bounds + char-split already in place; manage vLLM/rerank-batch as documented).
3. Acceptance: ask "What was NVIDIA's Graphics segment revenue in fiscal 2026?" → expect ANS ≈ $22.5B (was REF); "Compute & Networking operating income?" → $130.1B. If they flip, add as ANS eval entries.
4. Re-resolve annual-report anchors; re-run the eval → HARD GATES (`refusal_cf=1.0`, 0 hallucinations) hold; chart-content 09/10 still ANS.
