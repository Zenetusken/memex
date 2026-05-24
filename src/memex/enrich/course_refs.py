"""Course cross-reference resolution — turn "Cours N" / "Semaine N"
mentions into wikilinks to sibling lecture documents.

The academic citation resolver (`enrich.citations`) is tuned for
author-year / paper-title references, so it emits nothing on lecture
slides, which cross-reference each other (and are cross-referenced by
their syllabus) by ordinal label instead: "Cours 2", "Semaine 4",
"voir le Cours 6". This resolver fills that gap.

Unlike `extract_citations`, this is a pure-text pass — no LLM. The
mapping from an ordinal mention to a target document is deterministic:

- Each document carries a *series key* (a course code like "CR350"
  parsed from its title) and an *ordinal* (the integer following a
  Cours/Semaine/Leçon/Module/Chapitre token in its title).
- A mention "Cours N" in a document resolves to the sibling document
  in the SAME series whose ordinal is N — regardless of whether that
  sibling's title labels itself "Cours N" or "Semaine N" (the CR350
  corpus mixes both labels for the same week).

Series gating is strict: a mention resolves only when the mentioning
document and the target share a non-empty series key. That stops
"Cours 4" in one course leaking a link into an unrelated course's
week 4. Documents without a parseable course code produce no links.

The resolver emits `ResolvedCitation`s so the rest of the enrich
pipeline (wikilink substitution in `insert_wikilinks`, `CITES` graph
edges, manifest counters) handles them exactly like academic
citations — including opportunistic `[[doc#section]]` anchoring.
"""

from __future__ import annotations

import re

from memex.enrich.citations import CitationIndex, ResolvedCitation

# A course code like "CR350", "INF8480", "MTH2302". Two-to-four letters
# followed by three-to-four digits. Used as the series key that gates
# cross-references to the same course.
_COURSE_CODE_RE = re.compile(r"\b([A-Za-z]{2,4}\d{3,4})\b")

# The ordinal label inside a document title, e.g. "Cours 6" or
# "Semaine 4". `(?!\d)` keeps a two-digit cap from greedily eating the
# first digit of a longer run (defends against parse artifacts like
# "Cours 121").
_TITLE_ORDINAL_RE = re.compile(
    r"\b(?:Cours|Semaine|Le[çc]on|Module|Chapitre)\s+(\d{1,2})(?!\d)",
    re.IGNORECASE,
)

# A cross-reference mention in body text. Group 1 is the full surface
# form (preserved verbatim for substitution); group 2 is the ordinal.
_COURSE_REF_RE = re.compile(
    r"\b((?:Cours|Semaine|Le[çc]on|Module|Chapitre)\s+(\d{1,2}))(?!\d)",
    re.IGNORECASE,
)

# Course references are exact structured labels — high confidence. Above
# `citations._HIGH_CONFIDENCE_THRESHOLD` (0.85) so `insert_wikilinks`
# rewrites them.
_COURSE_REF_CONFIDENCE = 0.95


def _series_key(title: str) -> str | None:
    """Parse the course code (series key) from a document title.

    Returns the uppercased code (e.g. "CR350") or None when the title
    carries no recognizable course code.
    """
    m = _COURSE_CODE_RE.search(title)
    return m.group(1).upper() if m else None


def _title_ordinal(title: str) -> int | None:
    """Parse the lecture ordinal from a title's Cours/Semaine label.

    "CR350 - Cours 6 - ..." → 6; "CR350 - Semaine 4 - ..." → 4;
    a syllabus titled "CR350 - Plan de cours officiel" → None (no
    ordinal: it's a cross-reference *source*, not a target).
    """
    m = _TITLE_ORDINAL_RE.search(title)
    return int(m.group(1)) if m else None


def extract_course_references(
    body: str,
    *,
    self_doc_id: str,
    self_title: str,
    index: CitationIndex,
    confidence: float = _COURSE_REF_CONFIDENCE,
) -> list[ResolvedCitation]:
    """Resolve "Cours N" / "Semaine N" mentions in `body` to sibling
    lecture documents in the same course series.

    `index` is the vault's citation index (already excludes the current
    document). `self_title` provides the mentioning document's series
    key; `self_doc_id` is used only for citation attribution.

    Returns one `ResolvedCitation` per distinct `(surface, target)`
    pair, in body order. Empty when the document has no course code,
    no same-series siblings carry ordinals, or no mention resolves.
    """
    self_series = _series_key(self_title)
    if self_series is None:
        return []

    # Map ordinal → (doc_id, title) for same-series siblings. First doc
    # per ordinal wins; duplicate ordinals within a series are not
    # expected, and a stable choice keeps the output deterministic.
    targets: dict[int, tuple[str, str]] = {}
    for doc_id, sig in index.by_id.items():
        if _series_key(sig.title) != self_series:
            continue
        ordinal = _title_ordinal(sig.title)
        if ordinal is None:
            continue
        targets.setdefault(ordinal, (doc_id, sig.title))

    if not targets:
        return []

    out: list[ResolvedCitation] = []
    seen: set[tuple[str, str]] = set()  # (surface_lower, target_doc_id)
    for m in _COURSE_REF_RE.finditer(body):
        surface = m.group(1)
        ordinal = int(m.group(2))
        target = targets.get(ordinal)
        if target is None:
            continue
        target_doc_id, target_title = target
        if target_doc_id == self_doc_id:  # never link a doc to itself
            continue
        key = (surface.lower(), target_doc_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            ResolvedCitation(
                surface_text=surface,
                target_doc_id=target_doc_id,
                target_title=target_title,
                confidence=confidence,
                chunk_id=f"{self_doc_id}#course-ref",
                via="course_ref",
            )
        )
    return out
