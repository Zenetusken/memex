"""Citation resolution + wikilink insertion — see IMPLEMENTATION-PLAN §1.4.

Two stages:

1. **Extraction** — per-chunk LLM call that emits `CitationCandidate`s
   (`surface_text` + `confidence`).
2. **Resolution** — heuristic match of each candidate against a built
   index of the vault's documents. Matching uses three signals in order
   of decreasing strength: title substring, author + year pattern,
   title-token overlap. A score ≥ 0.7 resolves; lower scores leave
   the candidate unresolved (recorded but not linked).

Resolved citations become two things:

- `MENTIONS`-style `CITES` edges in the graph store
- `[[doc_id]]` wikilinks substituted in-place in the markdown body —
  but only for **high-confidence** resolutions, to avoid replacing
  ambiguous "Smith" mentions with the wrong document.

Citation matching deliberately doesn't go through the LLM a second
time. The graph + frontmatter are authoritative; the heuristic keeps
the pipeline cheap and the decisions auditable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from memex.vault.store import Frontmatter

CitationConfidence = Literal["high", "medium", "low"]

_CONFIDENCE_TO_FLOAT: dict[CitationConfidence, float] = {
    "high": 0.92,
    "medium": 0.68,
    "low": 0.42,
}

# Surface-form patterns we generate from frontmatter for matching.
# Single-author "Smith 2024", multi-author "Smith and Jones 2024",
# "Smith et al. 2024", "Smith et al., 2024". All lowercased.
_AUTHOR_YEAR_PATTERNS = (
    "{a0} {year}",
    "{a0}, {year}",
    "{a0} and {a1} {year}",
    "{a0} & {a1} {year}",
    "{a0} et al. {year}",
    "{a0} et al., {year}",
    "({a0} {year})",
    "({a0}, {year})",
)


# ----- Model-emitted shapes -----


class CitationCandidate(BaseModel):
    surface_text: str = Field(min_length=1)
    confidence: CitationConfidence


class CitationList(BaseModel):
    """Top-level output schema for the `extract_citations` prompt."""

    citations: list[CitationCandidate] = Field(default_factory=list)


class CitationExtractionInput(BaseModel):
    """Documented input shape for the `extract_citations` prompt."""

    document_title: str
    passage: str


# ----- Post-processed types -----


class ResolvedCitation(BaseModel):
    """A citation that matched a vault document."""

    surface_text: str
    target_doc_id: str
    target_title: str
    confidence: float           # composite of model + resolver
    chunk_id: str
    via: str                    # what matched ("title", "author_year", "tokens")


@dataclass
class DocSignature:
    """Searchable signature derived from a vault document's frontmatter."""

    doc_id: str
    title: str
    title_lower: str
    title_tokens: set[str] = field(default_factory=set)
    author_year_forms: list[str] = field(default_factory=list)


@dataclass
class CitationIndex:
    """Map of `doc_id → DocSignature`, plus precomputed lookup helpers."""

    by_id: dict[str, DocSignature] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.by_id)


def _author_year_forms(frontmatter: Frontmatter) -> list[str]:
    """Generate lowercase author-year surface forms from frontmatter.

    A "year" is the 4-digit YYYY portion of the `date` field; if there
    is no date, no author-year forms are emitted (we won't risk
    bare-author matches).
    """
    if not frontmatter.authors or frontmatter.date is None:
        return []
    year = str(frontmatter.date.year)
    # Authors are commonly "Last, First" or "First Last". For matching we
    # want the last name only; take the first comma-separated chunk if a
    # comma is present, otherwise the last whitespace-separated word.
    surnames: list[str] = []
    for author in frontmatter.authors:
        a = author.strip().lower()
        if not a:
            continue
        if "," in a:
            surnames.append(a.split(",", 1)[0].strip())
        else:
            parts = a.split()
            surnames.append(parts[-1])

    if not surnames:
        return []

    forms: list[str] = []
    a0 = surnames[0]
    a1 = surnames[1] if len(surnames) > 1 else None
    for pat in _AUTHOR_YEAR_PATTERNS:
        if "{a1}" in pat and a1 is None:
            continue
        forms.append(pat.format(a0=a0, a1=a1 or "", year=year))
    return forms


def _title_tokens(title: str) -> set[str]:
    """Lowercase, alphanumeric-only tokens of length ≥ 3."""
    tokens = re.findall(r"[a-z0-9]+", title.lower())
    return {t for t in tokens if len(t) >= 3}


def make_signature(doc_id: str, frontmatter: Frontmatter) -> DocSignature:
    """Derive a `DocSignature` from a vault document's frontmatter."""
    title = (frontmatter.title or "").strip()
    return DocSignature(
        doc_id=doc_id,
        title=title,
        title_lower=title.lower(),
        title_tokens=_title_tokens(title),
        author_year_forms=_author_year_forms(frontmatter),
    )


# ----- Resolution -----


_RESOLVE_THRESHOLD = 0.70
_HIGH_CONFIDENCE_THRESHOLD = 0.85


def resolve_candidate(
    candidate: CitationCandidate,
    chunk_id: str,
    index: CitationIndex,
    *,
    skip_doc_id: str | None = None,
) -> ResolvedCitation | None:
    """Find the best-matching vault document for one candidate.

    `skip_doc_id` excludes the candidate's own document (we don't link
    a doc to itself). Returns `None` if the best score is below the
    resolve threshold (0.70).
    """
    norm = candidate.surface_text.lower().strip()
    if not norm:
        return None

    candidate_tokens = set(re.findall(r"[a-z0-9]+", norm))

    best: tuple[str, float, str] | None = None  # (doc_id, score, via)

    for doc_id, sig in index.by_id.items():
        if doc_id == skip_doc_id:
            continue
        score = 0.0
        via = ""

        # 1. Full-title substring (either direction): very strong signal.
        if sig.title_lower and len(sig.title_lower) >= 8:
            if sig.title_lower in norm:
                score, via = 0.95, "title"
            elif norm in sig.title_lower and len(norm) >= 12:
                score, via = 0.88, "title"

        # 2. Author-year forms: strong signal when frontmatter has both.
        if score < 0.90:
            for form in sig.author_year_forms:
                if form in norm:
                    if score < 0.90:
                        score, via = 0.90, "author_year"
                    break

        # 3. Title-token overlap: weakest, only used when the others miss.
        if score < 0.85 and sig.title_tokens and candidate_tokens:
            overlap = len(sig.title_tokens & candidate_tokens)
            ratio = overlap / max(len(sig.title_tokens), 1)
            if ratio >= 0.6 and overlap >= 2:
                token_score = min(0.78, 0.55 + 0.10 * overlap)
                if token_score > score:
                    score, via = token_score, "tokens"

        if best is None or score > best[1]:
            best = (doc_id, score, via)

    if best is None or best[1] < _RESOLVE_THRESHOLD:
        return None

    doc_id, resolver_score, via = best
    sig = index.by_id[doc_id]
    # Composite confidence: average the model's confidence and the
    # resolver's score. Both range 0..1; product would punish too hard.
    model_conf = _CONFIDENCE_TO_FLOAT[candidate.confidence]
    composite = (model_conf + resolver_score) / 2
    return ResolvedCitation(
        surface_text=candidate.surface_text,
        target_doc_id=doc_id,
        target_title=sig.title or doc_id,
        confidence=composite,
        chunk_id=chunk_id,
        via=via,
    )


# ----- Wikilink insertion -----


def insert_wikilinks(
    body: str,
    resolved: list[ResolvedCitation],
    *,
    high_confidence_threshold: float = _HIGH_CONFIDENCE_THRESHOLD,
) -> tuple[str, int]:
    """Replace high-confidence citation surface forms with `[[doc_id]]`.

    Only resolutions at or above `high_confidence_threshold` are
    rewritten. The first occurrence per surface form is replaced;
    subsequent occurrences are intentionally left alone so a single
    edit doesn't cascade through the whole document (the user can
    re-run enrich to deepen if desired).

    Returns `(new_body, count)` where count is the number of
    substitutions made.
    """
    # Sort by surface-text length DESC so longer matches consume their
    # text before shorter overlapping matches can fire.
    ordered = sorted(
        (r for r in resolved if r.confidence >= high_confidence_threshold),
        key=lambda r: len(r.surface_text),
        reverse=True,
    )
    new_body = body
    count = 0
    # Track which doc_ids have already been linked; one wikilink per
    # `(surface_text, target)` per document is enough.
    seen: set[tuple[str, str]] = set()
    for cit in ordered:
        key = (cit.surface_text, cit.target_doc_id)
        if key in seen:
            continue
        # Word-boundary substitution. We use a custom boundary that
        # accepts citation-adjacent punctuation (parens, brackets, etc).
        pattern = re.compile(
            r"(?<![A-Za-z0-9_])" + re.escape(cit.surface_text) + r"(?![A-Za-z0-9_])"
        )
        new_body, n = pattern.subn(
            f"[[{cit.target_doc_id}]]", new_body, count=1
        )
        if n > 0:
            count += n
            seen.add(key)
    return new_body, count
