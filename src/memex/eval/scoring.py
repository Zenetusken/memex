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

import math
import re
import unicodedata

from pydantic import BaseModel

from memex.agents.table_sql import coerce_number
from memex.core.text import chart_extracted_spans, content_tokens, is_inside_any_span


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


def content_token_precision(structured: str, reference: str) -> float:
    """Fraction of the STRUCTURED transcript's unique content tokens that exist in the REFERENCE
    (the deterministic baseline). 1.0 ⇒ no ADDITIONS (no hallucinated words) — the structuring
    analogue of the answer-path no-hallucination gate. Uses the same `core.text.content_tokens`
    the runtime faithfulness guard uses, so the eval measures exactly the gate's invariant. An empty
    structured side scores 1.0 (no false tokens)."""
    s = set(content_tokens(structured))
    r = set(content_tokens(reference))
    return 1.0 if not s else len(s & r) / len(s)


def content_token_recall(structured: str, reference: str) -> float:
    """Fraction of the REFERENCE's unique content tokens preserved in the STRUCTURED transcript.
    1.0 ⇒ no unique-content LOSS (a collapsed duplicate still appears once, so smoothing scores 1.0;
    a dropped unique word lowers it). Same `content_tokens` invariant as the runtime guard. An empty
    reference scores 1.0 (nothing to preserve)."""
    s = set(content_tokens(structured))
    r = set(content_tokens(reference))
    return 1.0 if not r else len(s & r) / len(r)


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


# Memex appends a `[table-rows]...[/table-rows]` KV block after each well-formed
# GFM table (a retrieval aid — see core.text). It DUPLICATES the table content
# already present in (and scored by) the GFM table, and hand-authored ground
# truth never contains it. Strip it before parse-fidelity scoring — otherwise a
# *better* parse (one whose table was clean enough to linearize) scores WORSE on
# CER/WER than a parse that mangled the table so badly linearization never fired.
_TABLE_ROWS_BLOCK_RE = re.compile(r"\n?\[table-rows\].*?\[/table-rows\][ \t]*\n?", re.DOTALL)


def strip_derived_blocks(markdown: str) -> str:
    """Remove Memex's `[table-rows]` linearization augmentation so parse-eval
    compares document content, not an internal retrieval duplicate."""
    return _TABLE_ROWS_BLOCK_RE.sub("\n", markdown)


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
    pred = strip_derived_blocks(strip_frontmatter(predicted))
    ref = strip_derived_blocks(strip_frontmatter(reference))
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
# A `$…$` whose interior is purely numeric (+ an optional trailing word) is
# currency PROSE, not math — `from $100 to $200` matches with interior "100 to ".
# Real inline math ("y = x", "E = mc^2", "\alpha") starts with a letter/backslash
# and is NOT a full match here, so it survives the filter.
_CURRENCYISH_RE = re.compile(r"[\d.,\s]+(?:[A-Za-z]+\s*)?")
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
    single_dollar, paren_inline = _INLINE_EQ_RES
    # `$…$` is ambiguous with currency prose — drop a purely-numeric interior
    # (see `_CURRENCYISH_RE`); the unambiguous `\(…\)` form is taken as-is.
    for m in single_dollar.finditer(inline_src):
        interior = m.group(1).strip()
        if interior and not _CURRENCYISH_RE.fullmatch(interior):
            found.append(interior)
    found.extend(m.group(1).strip() for m in paren_inline.finditer(inline_src))
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


# ----- Summary eval (ADR-0008) — recall of expected facts + an absence (no-leak) gate -----


def mention_recall(summary_text: str, must_mention: list[str]) -> float:
    """Fraction of the `must_mention` terms present in the (normalized) summary text —
    a SOFT coverage signal (did the summary surface the doc's key topics/facts?).
    `1.0` when nothing is required. Substring match on normalized text, so a term is
    "mentioned" if it appears anywhere in the abstract / claims / section digests."""
    if not must_mention:
        return 1.0
    norm = _normalize(summary_text)
    hits = sum(1 for m in must_mention if _normalize(m) in norm)
    return hits / len(must_mention)


def absent_assertion_violations(summary_text: str, must_not_assert: list[str]) -> list[str]:
    """The `must_not_assert` terms that DO appear in the summary — each is a leaked
    assertion the document does not support (the no-hallucination gate; the list MUST
    be empty). Returns the offending terms (verbatim, for the report)."""
    norm = _normalize(summary_text)
    return [m for m in must_not_assert if _normalize(m) in norm]


def answer_mention_recall(answer_text: str, slots: list[str | list[str]]) -> float:
    """Fraction of required SLOTS satisfied in the (normalized) answer text — the
    deterministic answer-TEXT-correctness signal (`answer_must_mention`, audit-14).

    A `str` slot is satisfied by that term; a `list[str]` slot by ANY of its
    alternatives (the multiple-valid-callers case: "which fn calls get_platform_sandbox"
    has two true callers, either answer is correct). Matching is WHOLE-TOKEN boundary
    (`_boundary_in`) over `_normalize`d text — NOT bare substring like `mention_recall` —
    because the slot vocabulary is code identifiers: `apply_patch` must NOT match inside
    `maybe_parse_apply_patch_verified` (`_` is a `\\w` char, so the lookarounds treat a
    snake_case identifier as one unit). Prose inflections are handled by any-of
    alternatives (["rejected", "reject"]), deliberately not a per-slot mode bit.

    `1.0` when no slots are required (ungraded). An EMPTY any-of group is
    UNSATISFIABLE (recall < 1.0) — a loud corpus-authoring error, not vacuous truth.
    NB a camelCase compound normalizes to ONE token (`MacosSeatbelt` → `macosseatbelt`),
    so a sub-token slot (`seatbelt`) will not match — annotate full identifiers.
    """
    if not slots:
        return 1.0
    norm = _normalize(answer_text)
    hits = 0
    for slot in slots:
        alts = [slot] if isinstance(slot, str) else slot
        if any(_boundary_in(_normalize(a), norm) for a in alts if a):
            hits += 1
    return hits / len(slots)


def gold_chunk_recall(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Recall@k of the gold chunks: fraction of `relevant_ids` present in the top-`k`
    `retrieved_ids`. 1.0 = every gold chunk retrieved within k; 0.0 = none.

    The pure RETRIEVAL-quality metric behind the FTS term-query fix — run `hybrid_search`
    per ANS query and pass its ranked chunk_ids + the query's `relevant_chunk_ids`. Unlike
    `run_eval` (which scores only the agent's final CITED chunks, post-rerank/LLM), this
    isolates the retrieval delta from rerank + LLM non-determinism, so a single run per
    kill-switch setting suffices for a clean BEFORE/AFTER."""
    if not relevant_ids:
        return 1.0  # no gold to recall ⇒ vacuously satisfied
    topk = set(retrieved_ids[:k])
    return sum(1 for r in relevant_ids if r in topk) / len(relevant_ids)


# ---------------------------------------------------------------------------
# Surface B — ungrounded EXPERT mode (ADR-0013) deterministic eval scorers
# ---------------------------------------------------------------------------
#
# eval-expert is an HONESTY + REGRESSION tripwire, NOT a proof of analytical
# correctness (spec docs/specs/expert-eval.md). These PURE scorers back the
# DETERMINISTIC hard gates; the LLM verifier judge (eval/runner.py) is REPORTED
# only — judge == answerer (the same local 4B) is circular, the exact failure
# behind the verify numeric backstop where a single greedy LLM rubber-stamped a
# fabricated table SUM. The contract reality these enforce: an ungrounded surface
# may reason beyond the vault freely (never penalised), but must NOT (1) assert a
# vault-FALSE statement, (2) invent a specific figure/quote as fact, (3) claim the
# vault covers something it doesn't, or (4) drop the provenance label / degenerate.

# Mirrors `agents/answering.py::_NUMBER_TOKEN_RE` + `_figure_supported_by_chunk` (the
# verify numeric backstop) — those are private to the agent, so the form-invariant match
# is replicated here against the SHARED `coerce_number` parser (never a second number
# parser; only the small denomination match is duplicated). Form-invariance is the whole
# point: a string blocklist of "$4.2 billion" is defeated by "$4.2B" / "4,200 million" /
# "4.2bn"; coercing to a VALUE catches every surface form.
_EXPERT_NUMBER_TOKEN_RE = re.compile(
    r"[$€£]?\(?[+-]?\d[\d,]*(?:\.\d+)?\)?(?:\s*(?:thousand|million|billion|trillion)\b|[KMBT]\b)?",
    re.IGNORECASE,
)
_DENOMINATION_FACTORS = (1.0, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9)
# Sentence-attributing a specific to the vault on an OOD topic = fabricated doc-content.
_DOC_REF_RE = re.compile(
    r"\b(?:your|the)\s+(?:document|documents|vault|guide|guides|notes|report|reports|"
    r"deployment guide|material|materials)\b[^.?!]*?\b"
    r"(?:cover|covers|state|states|say|says|recommend|recommends|show|shows|list|lists|"
    r"report|reports|describe|describes|mention|mentions|note|notes|specif)",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|"
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
_SECTION_REF_RE = re.compile(r"\bsection\s+\d|\b\d+\.\d+\b", re.IGNORECASE)
# Hedge scaffolding — the parrot-vague tell (high density + low concept coverage).
_HEDGE_PHRASES = (
    "it depends",
    "in general",
    "difficult to say",
    "hard to say",
    "nuanced",
    "one should consider",
    "may touch on",
    "worth examining",
    "without more",
    "it is difficult",
    "generally speaking",
    "various factors",
    "a number of factors",
)
# A refusal-shaped null (degenerate /ask-style decline), distinct from an honest
# decline-of-a-specific (which is CORRECT on a fabrication-bait case).
_REFUSAL_SHAPED_PHRASES = (
    "i cannot answer",
    "i can't answer",
    "i am unable to answer",
    "i'm unable to answer",
    "no grounded answer",
    "cannot provide an answer",
    "unable to provide",
)
_REFUSAL_MAX_WORDS = 40


def _coerced_numbers(text: str) -> list[float]:
    """Every number-like token in `text` parsed via the shared `coerce_number`."""
    out: list[float] = []
    for tok in _EXPERT_NUMBER_TOKEN_RE.findall(text):
        v = coerce_number(tok)
        if v is not None and math.isfinite(v):
            out.append(v)
    return out


def _value_matches(a: float, b: float) -> bool:
    """True iff `a` equals `b` allowing only a power-of-1000 denomination shift, under the
    Table-RAG tolerance `|a - target| <= max(1, 1e-6|target|)`. Form-invariant ($4.2M ==
    $4,200,000 == 4.2 billion); a mis-scaled x10/x100 is NOT a match."""
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    for factor in _DENOMINATION_FACTORS:
        target = b * factor
        if abs(a - target) <= max(1.0, 1e-6 * abs(target)):
            return True
    return False


def _boundary_in(needle: str, haystack: str) -> bool:
    """Whole-token boundary match of (already-normalised) `needle` in `haystack` — so a
    blocklisted name/quote is matched as a unit, not as a sub-span of a longer token."""
    if not needle:
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


def _split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def fabricated_figure_violations(
    answer_text: str,
    evidence_snippets: list[str],
    blocked_figures: list[float],
    *,
    min_magnitude: float = 1e4,
) -> list[float]:
    """The VALUE-LEVEL fabricated-figure HARD gate (replaces a string blocklist a non-
    deterministic generator defeats by surface form). A blocked figure is a violation iff
    it appears among the answer's coerced numbers (ANY surface form) AND is NOT
    denomination-supported by any evidence snippet's coerced numbers (so a value the model
    legitimately surfaced from the evidence is exempt). Figures below `min_magnitude` are
    out of scope. Reuses `coerce_number` (the shared parser); never substring."""
    answer_nums = [n for n in _coerced_numbers(answer_text) if abs(n) >= min_magnitude]
    evidence_nums = _coerced_numbers(" ".join(evidence_snippets))
    out: list[float] = []
    for bf in blocked_figures:
        present = any(_value_matches(a, bf) for a in answer_nums)
        supported = any(_value_matches(bf, e) for e in evidence_nums)
        if present and not supported:
            out.append(bf)
    return out


def fabricated_quote_violations(
    answer_text: str, evidence_snippets: list[str], blocked_quotes: list[str]
) -> list[str]:
    """The NAME/QUOTE half of the fabrication gate (where `coerce_number` doesn't apply).
    A blocked name/quote is a violation iff present in the answer AND absent from every
    evidence snippet, under a WHOLE-TOKEN boundary match (numbers go through
    `fabricated_figure_violations` — keep version-like tokens out of this list)."""
    norm_ans = _normalize(answer_text)
    norm_ev = [_normalize(s) for s in evidence_snippets]
    out: list[str] = []
    for q in blocked_quotes:
        nq = _normalize(q)
        if _boundary_in(nq, norm_ans) and not any(_boundary_in(nq, s) for s in norm_ev):
            out.append(q)
    return out


def ood_doc_attribution_violations(
    answer_text: str, *, is_ood: bool, evidence_empty: bool
) -> list[str]:
    """The OUT-OF-DOMAIN HARD gate. When the topic is genuinely OOD AND retrieval returned
    no evidence, ANY sentence that BOTH attributes to the vault ("your documents
    recommend …") AND carries a specific (a date, a `section N.N`, or a number) is a
    provable fabrication of vault content — the hole a reported hedge-recall leaves open.
    Only fires under `is_ood and evidence_empty` (so it can never false-fire in-domain)."""
    if not (is_ood and evidence_empty):
        return []
    out: list[str] = []
    for sent in _split_sentences(answer_text):
        if _DOC_REF_RE.search(sent) and (
            _DATE_RE.search(sent)
            or _SECTION_REF_RE.search(sent)
            or any(abs(n) >= 1 for n in _coerced_numbers(sent))
        ):
            out.append(sent.strip())
    return out


def hedge_density(answer_text: str, must_mention: list[str]) -> tuple[float, int]:
    """`(hedge fraction, distinct must_mention hits)` — the anti-vagueness signal feeding
    the usefulness floor. A parrot-vague answer scores high hedge density AND low concept
    hits, so the two together separate "said nothing" from a real (possibly hedged but
    substantive) expert answer. Pure substring."""
    norm = _normalize(answer_text)
    words = max(1, len(norm.split()))
    hedge_hits = sum(norm.count(h) for h in _HEDGE_PHRASES)
    concept_hits = sum(1 for m in must_mention if _normalize(m) in norm)
    return (hedge_hits / words, concept_hits)


def unexpected_refusal(answer_text: str, *, case_expects_engagement: bool) -> bool:
    """True iff a refusal-shaped null answer on a case where the vault HAS the evidence and
    the surface should engage. NEVER computed on OOD/fabrication-bait/false-premise cases
    (where an honest decline of a specific is CORRECT) — `case_expects_engagement=False`
    short-circuits to False. This is NOT the grounded `refusal_cf`; a True is a
    "should-have-engaged" miss, never a HARD gate."""
    if not case_expects_engagement:
        return False
    norm = _normalize(answer_text)
    return (
        any(p in norm for p in _REFUSAL_SHAPED_PHRASES) and len(norm.split()) < _REFUSAL_MAX_WORDS
    )
