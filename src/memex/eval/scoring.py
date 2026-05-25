"""Eval metrics — CER, WER, refusal-rate, citation-precision skeletons.

This module ships the metric *signatures* and the in-house implementations
that don't depend on external libs (ragas, jiwer). The reference
implementations:

- CER/WER use a Levenshtein-distance helper from the stdlib (`difflib`
  is fine for the magnitudes Memex sees; jiwer can replace this when
  the dev extras are installed).
- Citation precision is a stub that scores against ground-truth chunk
  IDs; the LLM-as-judge variant lands when the eval corpus does.

See docs/eval-corpus-plan.md.
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel

from memex.core.text import chart_extracted_spans, is_inside_any_span


def _normalize(text: str) -> str:
    """NFC + lowercase + collapse whitespace. Per eval-corpus-plan."""
    return " ".join(unicodedata.normalize("NFC", text).strip().lower().split())


def _levenshtein(a: str, b: str) -> int:
    """Classic O(len(a)*len(b)) Levenshtein on character sequences."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(
                min(
                    cur[-1] + 1,  # insertion
                    prev[j] + 1,  # deletion
                    prev[j - 1] + cost,  # substitution
                )
            )
        prev = cur
    return prev[-1]


def character_error_rate(predicted: str, reference: str) -> float:
    """Standard CER: Levenshtein / max(len(reference), 1)."""
    pn = _normalize(predicted)
    rn = _normalize(reference)
    if not rn:
        return 0.0 if not pn else 1.0
    return _levenshtein(pn, rn) / max(len(rn), 1)


def _levenshtein_seq(a: list[str], b: list[str]) -> int:
    """Levenshtein on a sequence of tokens (one token = one unit of edit).

    The character-level `_levenshtein` above operates on str characters;
    this overload counts each *word* (or whatever token) as one unit so
    `word_error_rate` doesn't conflate "different-length words" with
    "multiple word edits".
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, start=1):
        cur = [i]
        for j, tb in enumerate(b, start=1):
            cost = 0 if ta == tb else 1
            cur.append(
                min(
                    cur[-1] + 1,  # insertion
                    prev[j] + 1,  # deletion
                    prev[j - 1] + cost,  # substitution
                )
            )
        prev = cur
    return prev[-1]


def word_error_rate(predicted: str, reference: str) -> float:
    """Word-level Levenshtein normalised by reference word count.

    One word edit = one unit of error, regardless of word length. A
    single-word substitution in a four-word reference scores 0.25.
    """
    pn = _normalize(predicted).split()
    rn = _normalize(reference).split()
    if not rn:
        return 0.0 if not pn else 1.0
    return _levenshtein_seq(pn, rn) / max(len(rn), 1)


_FRONTMATTER_RE = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.DOTALL)
_ATX_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")

# Inline-markdown patterns stripped from heading TEXT before structural
# F1 (links, bold, italic, code). A parser that emits `## **Overview**`
# represents the same heading as a clean `## Overview` — the `**` is
# styling, not structure. (CER/WER stay raw — markup noise is a
# legitimate character-fidelity signal there; only the structural
# heading comparison normalizes it.)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{1,3})(.+?)\1")
_MD_CODE_RE = re.compile(r"`([^`]+)`")


def _clean_heading_text(text: str) -> str:
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_CODE_RE.sub(r"\1", text)
    # Repeat emphasis stripping so `***bold-italic***` fully unwraps.
    for _ in range(3):
        new = _MD_EMPHASIS_RE.sub(r"\2", text)
        if new == text:
            break
        text = new
    return text.strip()


def strip_frontmatter(markdown: str) -> str:
    """Drop a leading YAML frontmatter block (`---\\n…\\n---\\n`).

    `vault.read_document` already splits frontmatter off the body, but a
    ground-truth `.md` file read from disk still carries it — strip it so
    CER/WER compare body-to-body (frontmatter is metadata, not prose)."""
    return _FRONTMATTER_RE.sub("", markdown, count=1)


def extract_markdown_headings(markdown: str) -> list[tuple[int, str]]:
    """Extract `(level, text)` ATX heading tuples for structural F1.

    Skips headings inside fenced code blocks (``` / ~~~) and inside
    `[chart-extracted]` blocks — the latter carry inert `# H1` chart-
    figure labels that aren't document structure (same defense the
    chunker applies via `core.text`). The trailing-`#` ATX closing
    sequence is stripped from the captured text.
    """
    spans = chart_extracted_spans(markdown)
    out: list[tuple[int, str]] = []
    in_fence = False
    offset = 0
    for line in markdown.splitlines(keepends=True):
        line_start = offset
        offset += len(line)
        stripped = line.rstrip("\n")
        if _FENCE_RE.match(stripped):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _ATX_HEADING_RE.match(stripped)
        if m is None or is_inside_any_span(line_start, spans):
            continue
        out.append((len(m.group(1)), _clean_heading_text(m.group(2))))
    return out


class ParseQualityScores(BaseModel):
    """The parse-fidelity metrics for one document: predicted markdown vs
    hand-curated ground truth. See `docs/eval-corpus-plan.md` §Scoring.

    Structural F1 has three facets — headings, tables, equations. Each is
    `1.0` when both sides are empty of that element (a doc with no tables
    isn't penalized on table F1)."""

    cer: float
    wer: float
    structural_f1_headings: float
    structural_f1_tables: float
    structural_f1_equations: float


def score_parse_quality(predicted: str, reference: str) -> ParseQualityScores:
    """Score predicted markdown against a ground-truth reference.

    Both sides have any leading YAML frontmatter stripped first so the
    text metrics compare body content. Headings/tables/equations feed the
    three structural-F1 facets. This is the single entry point the eval
    runner calls per document — it bundles the primitives so the runner
    stays thin.
    """
    pred = strip_frontmatter(predicted)
    ref = strip_frontmatter(reference)
    return ParseQualityScores(
        cer=character_error_rate(pred, ref),
        wer=word_error_rate(pred, ref),
        structural_f1_headings=structural_f1_headings(
            extract_markdown_headings(pred),
            extract_markdown_headings(ref),
        ),
        structural_f1_tables=structural_f1_tables(
            extract_markdown_tables(pred),
            extract_markdown_tables(ref),
        ),
        structural_f1_equations=structural_f1_equations(
            extract_markdown_equations(pred),
            extract_markdown_equations(ref),
        ),
    )


def structural_f1_headings(
    predicted: list[tuple[int, str]],
    reference: list[tuple[int, str]],
) -> float:
    """Precision/recall over `(level, text)` heading tuples."""
    pset = {(lvl, _normalize(txt)) for lvl, txt in predicted}
    rset = {(lvl, _normalize(txt)) for lvl, txt in reference}
    if not pset and not rset:
        return 1.0
    if not pset or not rset:
        return 0.0
    tp = len(pset & rset)
    precision = tp / len(pset)
    recall = tp / len(rset)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# --- Tables: structural F1 over GFM cell content ---------------------------

# A GFM delimiter row: optional outer pipes around `:--`/`---`/`--:` runs.
_TABLE_DELIM_RE = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(?:\|\s*:?-{1,}:?\s*)*\|?\s*$")


def _split_table_row(line: str) -> list[str]:
    """Split a GFM table row into stripped cells, dropping the outer pipes."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def extract_markdown_tables(markdown: str) -> list[list[list[str]]]:
    """Extract GFM pipe tables as `[table][row][cell]` — header row included,
    the `|---|` delimiter row dropped.

    Skips fenced code blocks and `[chart-extracted]` blocks (the same defense
    `extract_markdown_headings` applies). The `[table-rows]` linearization
    block Memex appends after a table is not a pipe table, so it is naturally
    ignored."""
    spans = chart_extracted_spans(markdown)
    lines = markdown.splitlines(keepends=True)
    starts: list[int] = []
    off = 0
    for ln in lines:
        starts.append(off)
        off += len(ln)
    plain = [ln.rstrip("\n") for ln in lines]

    tables: list[list[list[str]]] = []
    in_fence = False
    i = 0
    n = len(plain)
    while i < n:
        line = plain[i]
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            i += 1
            continue
        if in_fence or is_inside_any_span(starts[i], spans):
            i += 1
            continue
        # A table = a header row whose NEXT line is a `|---|` delimiter.
        if "|" in line and i + 1 < n and _TABLE_DELIM_RE.match(plain[i + 1]):
            rows: list[list[str]] = [_split_table_row(line)]
            j = i + 2
            while (
                j < n
                and "|" in plain[j]
                and not _FENCE_RE.match(plain[j])
                and not is_inside_any_span(starts[j], spans)
            ):
                rows.append(_split_table_row(plain[j]))
                j += 1
            tables.append(rows)
            i = j
            continue
        i += 1
    return tables


def structural_f1_tables(
    predicted: list[list[list[str]]],
    reference: list[list[list[str]]],
) -> float:
    """Precision/recall of cell content keyed by `(table, row, col)` position.

    Tables are aligned by document order; a cell is a true positive when both
    sides hold a cell at the same `(table_index, row_index, col_index)` and
    its normalized content matches. Extra tables/rows/columns on either side
    are unmatched, so count and shape mismatches are penalized naturally. Per
    docs/eval-corpus-plan.md ("cell content given table identity")."""

    def _cells(tabs: list[list[list[str]]]) -> dict[tuple[int, int, int], str]:
        out: dict[tuple[int, int, int], str] = {}
        for t, rows in enumerate(tabs):
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    out[(t, r, c)] = _normalize(cell)
        return out

    pmap = _cells(predicted)
    rmap = _cells(reference)
    if not pmap and not rmap:
        return 1.0
    if not pmap or not rmap:
        return 0.0
    tp = sum(1 for k, v in pmap.items() if rmap.get(k) == v)
    precision = tp / len(pmap)
    recall = tp / len(rmap)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# --- Equations: normalized-LaTeX structural F1 ------------------------------

_FENCED_BLOCK_RE = re.compile(r"(?ms)^[ \t]*(```|~~~).*?^[ \t]*\1[ \t]*$")
_DISPLAY_EQ_RES = (
    re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
    re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
)
_INLINE_EQ_RES = (
    re.compile(r"(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)"),
    re.compile(r"\\\((.+?)\\\)", re.DOTALL),
)
_FRAC_RE = re.compile(r"\\[dt]frac\b")
# Spacing macros + `\left`/`\right` — pure rendering, dropped before compare.
_EQ_SPACING_RE = re.compile(r"\\[,;:!> ]|\\quad\b|\\qquad\b|\\left\b|\\right\b")


def normalize_equation(equation: str) -> str:
    """Trivial-form LaTeX normalization for structural comparison: `\\dfrac`/
    `\\tfrac` → `\\frac`, drop spacing macros + `\\left`/`\\right`, collapse
    whitespace to single spaces. Per docs/eval-corpus-plan.md."""
    eq = _FRAC_RE.sub(r"\\frac", equation)
    eq = _EQ_SPACING_RE.sub(" ", eq)
    return " ".join(eq.split())


def extract_markdown_equations(markdown: str) -> list[str]:
    """Extract raw LaTeX math spans — `$$…$$`, `\\[…\\]`, `$…$`, `\\(…\\)` —
    skipping fenced code blocks so shell `$VAR` isn't mistaken for math.
    Returns the raw interiors; `structural_f1_equations` normalizes."""
    body = _FENCED_BLOCK_RE.sub("\n", markdown)
    found: list[str] = []
    for rx in _DISPLAY_EQ_RES:
        found.extend(m.group(1).strip() for m in rx.finditer(body))
    # Strip display blocks before scanning inline so `$$…$$` interiors aren't
    # re-matched by the single-`$` inline pattern.
    inline_src = body
    for rx in _DISPLAY_EQ_RES:
        inline_src = rx.sub(" ", inline_src)
    for rx in _INLINE_EQ_RES:
        found.extend(m.group(1).strip() for m in rx.finditer(inline_src))
    return [e for e in found if e]


def structural_f1_equations(predicted: list[str], reference: list[str]) -> float:
    """Precision/recall over normalized-LaTeX equation strings (set-based,
    mirroring `structural_f1_headings`)."""
    pset = {n for n in (normalize_equation(e) for e in predicted) if n}
    rset = {n for n in (normalize_equation(e) for e in reference) if n}
    if not pset and not rset:
        return 1.0
    if not pset or not rset:
        return 0.0
    tp = len(pset & rset)
    precision = tp / len(pset)
    recall = tp / len(rset)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class CitationPrecisionInput(BaseModel):
    """The agent cited chunks; the eval supplies the ground-truth allowed set."""

    cited_chunk_ids: list[str]
    relevant_chunk_ids: set[str]


def citation_precision(input_: CitationPrecisionInput) -> float:
    """Fraction of cited chunks that appear in the relevant set.

    Returns `1.0` when the agent emitted zero citations — a refused
    answer has no false-positive citations. This inflates the
    all-queries mean; `EvalReport` also reports the answered-only
    variant for an honest signal."""
    if not input_.cited_chunk_ids:
        return 1.0  # no citations to be wrong about
    correct = sum(1 for cid in input_.cited_chunk_ids if cid in input_.relevant_chunk_ids)
    return correct / len(input_.cited_chunk_ids)
