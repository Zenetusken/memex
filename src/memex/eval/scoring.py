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
        out.append((len(m.group(1)), m.group(2).strip()))
    return out


class ParseQualityScores(BaseModel):
    """The three parse-fidelity metrics for one document: predicted
    markdown vs hand-curated ground truth. See `docs/eval-corpus-plan.md`
    §Scoring."""

    cer: float
    wer: float
    structural_f1_headings: float


def score_parse_quality(predicted: str, reference: str) -> ParseQualityScores:
    """Score predicted markdown against a ground-truth reference.

    Both sides have any leading YAML frontmatter stripped first so the
    text metrics compare body content. Headings feed `structural_f1_
    headings`. This is the single entry point the eval runner calls per
    document — it bundles the three primitives so the runner stays thin.
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
