# Spec: char-aware chunk splitting + GFM table-header repetition

**Status:** in implementation (2026-05-24). **Owner module:** `src/memex/index/chunker.py`.

## Problem

Table-*only* values in the vault (e.g. the 10-K's Graphics segment revenue, C&N
operating-income growth) **false-refuse**: a GFM table is one paragraph with no
sentence boundaries, so it becomes one oversized chunk. Two failures result:
1. The existing `MAX_CHUNK_MULTIPLIER` cap is **word-budget**-based; char-heavy
   but word-light tables (20951 chars / 493 budget-words) slip under it.
2. Even when split, a row-group past the first has **no column headers** (they
   were in the table's first line) → the answer LLM sees `| 2,345 | 1,890 |`
   with no idea what the columns mean → can't ground a value.

## Goal

Every table chunk is (a) **char-bounded** so it fits the reranker window + the
answer prompt's `truncate(1800)`, and (b) **self-contained** — each row-group
carries the table's header row(s) so a value is interpretable in isolation.
Outcome: table-only values become answerable; the previously-false-refusing
queries (Graphics segment revenue, C&N OI growth) flip REF→ANS, HARD GATES hold.

## Design

### 1. Char-aware oversize trigger (both branches)

Add module constants:
- `MAX_CHUNK_CHARS = 1800` — row-group char target (matches the answer prompt's
  `truncate(1800)` so each table chunk is fully answer-visible).
- Keep `MAX_CHUNK_MULTIPLIER = 3` (word cap = `target * 3`).

In `_split_section_into_chunks`, a unit is **oversized** when
`_budget_word_count(u) > max_tokens_per_chunk` **OR** `len(u) > MAX_CHUNK_CHARS`.
Apply in both the paragraph loop and the inner sentence loop. Prose is safe:
`_force_split_oversized` splits only on `\n`, and a prose sentence/paragraph with
no internal newline returns whole (degenerate guard) — so a prose unit that trips
the char threshold but has no row structure is a **no-op**. Only multi-line units
(tables) actually split.

### 2. `_force_split_oversized` — char + word bounded line packing

`_force_split_oversized(unit, *, target_tokens, max_chars=MAX_CHUNK_CHARS)`:
greedily pack consecutive lines into groups, flushing when adding the next line
would exceed **either** `target_tokens` (word-budget) **or** `max_chars`. Never
split a line. A single line longer than `max_chars` is emitted whole (degenerate
guard — pathological one-line table).

### 3. GFM header detection + repetition

- `_gfm_header(unit) -> str | None`: if the unit's first two non-blank lines are a
  pipe row (`|`…`|`) followed by a delimiter row (matches `^\s*\|?[\s:|-]+\|?\s*$`
  and contains `-`), return those two lines joined by `\n`; else `None` (not a GFM
  table → no repetition).
- When `_force_split_oversized` splits a unit that has a GFM header: **prepend the
  header block** to every group **after the first** (the first group already
  starts with the header). Header-repeated groups are valid standalone GFM tables.

### 4. Offset handling (the load-bearing part)

The existing re-locate loop assumes a window's text is a contiguous substring of
`section`. A header-repeated group's text (`header + "\n" + rows`) is **not**
(the header is synthetic for groups >0). So force-split must **emit chunk tuples
directly** with explicit offsets, bypassing the generic re-locate:

- For each group, the **rows** (the source portion, excluding any synthetic
  header) are a contiguous substring of `unit`, hence of `section`. Locate the
  rows: `idx = section.find(rows, cursor)`; `char_start = section_offset + idx`;
  `char_end = char_start + len(rows)`; advance `cursor = idx + len(rows)`.
- `chunk_text = header + "\n" + rows` for groups >0 (synthetic header prepended),
  else `rows` (group 0 already includes the header).
- Emit `(char_start, char_end, chunk_text)` directly.

Invariant: `body[char_start:char_end] == rows` (the source portion) — the synthetic
header is the only text not in `[cs,ce)`. `heading_path_at(body, char_start)` stays
correct (char_start is the real rows position, inside the right section). This
relaxes the round-trip to the rows portion for header-repeated chunks (documented).

Restructure `_split_section_into_chunks` to build `chunks: list[(cs,ce,text)]`
incrementally: keep the pending prose `windows` list; when a force-split happens,
**flush** the pending windows (re-locate them as today) **then** append the
force-split chunk tuples directly, then continue. Net: prose path byte-identical to
today; only oversized table units take the new path.

### 5. chunk_id / determinism

`chunk_id = sha1(doc_id + chunk_text)`. Header-repeated text → deterministic id
(same input ⇒ same id). chunk_ids for the affected tables change (re-index +
re-resolve anchors on migration; only docs with oversized tables re-chunk).

### 6. Chart-block safety

`_budget_word_count` already zeroes `[chart-extracted]` blocks. The char trigger,
however, is raw `len()`. Guard: do NOT char-trigger a unit that is (or is inside) a
chart-extracted block — a chart block is intentionally one chunk. Check
`"[chart-extracted]" not in unit` before applying the char trigger (the word
trigger already won't fire on chart blocks). Pin with a test.

## Tests (`tests/unit/test_chunker.py`, extend)

1. Char-heavy/word-light table (≈20K chars, <target words, rows newline-separated)
   → multiple chunks, **every** chunk `len(text) <= MAX_CHUNK_CHARS + one_row` and
   `_budget_word_count <= cap`.
2. Header repetition: every table sub-chunk after the first **starts with the
   header + delimiter rows**; a deep row (e.g. row 200) appears in a chunk that
   ALSO contains the column header.
3. Rows intact: no `|`-line cut mid-pipe.
4. Offsets: for each table chunk, `body[char_start:char_end]` equals the chunk's
   **rows portion** (chunk_text with any leading synthetic header stripped);
   offsets monotonic non-decreasing.
5. `heading_path` correct for all table sub-chunks (table under `## Financials`).
6. Prose regression: a normal multi-paragraph prose doc → chunk_ids **identical**
   to a pre-recorded run (cap/char-split is a no-op on prose); deterministic
   across two calls.
7. Non-GFM oversized unit (no `|` header) → split on lines/whitespace, **no**
   synthetic header prepended.
8. Oversized-in-raw-chars `[chart-extracted]` block → stays ONE chunk (chart-safe).
9. Keep all existing `test_chunker.py` + `test_chunker_chart_aware.py` green.

## Local gates (execution agent must pass before reporting done)

- `uv run pytest tests/unit/test_chunker.py tests/unit/test_chunker_chart_aware.py -q` → green
- `uv run pytest tests/ -q` → green (429 + new)
- `uv run pyright src/memex/index/chunker.py` → 0 errors / 0 warnings
- `uv run ruff check src/memex/index/chunker.py tests/unit/test_chunker.py` + `ruff format`

## Validation gates (orchestrator runs — GPU/vLLM; NOT the execution agent)

1. `memex index 0e725ba0-2026-annual-report-web` — table chunks split; chunk count rises.
2. Ask the previously-false-refusing table queries with `MEMEX_RERANK_BATCH_SIZE=1`:
   - "What was NVIDIA's Graphics segment revenue in fiscal 2026?" → expect ANS ≈ $22.5B
   - "By what percentage did NVIDIA's Graphics segment operating income grow…?" → expect ANS ≈ 80%
   Confirm they now ANSWER with the correct value (the win condition).
3. Re-resolve annual-report anchors; re-run the annual-report eval → HARD GATES
   (`refusal_cf=1.0`, 0 hallucinations) hold; chart-content (09/10) still ANS.
4. If 2 succeeds, convert those queries into passing ANS eval entries (or add new
   ones), re-resolve, commit.

## Anti-scope

- No new `IndexSettings` field (module constants; promote later if tuning needed).
- No table transposition / cell-level parsing — row-group + header repetition only.
- No change to the prose chunking path (must stay byte-identical — pinned by test 6).
