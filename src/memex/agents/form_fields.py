"""Deterministic form-field resolver — the disambiguation fix for form-detail Q&A (Increment A).

A born-digital form (IRS 1040 / W-4) parses into a dense run-on `|cell|` wall where a labeled
value sits among DISTRACTORS — e.g. one f1040 cell holds
`• Single or Married filing separately, $15,750 • Married filing jointly …, $31,500 •
Head of household, $23,625`. The 4B can't reliably pick the asked value from among the three, so
the answer node drafts "the requested information is not available" and the gates refuse — even
though the fact is IN-WINDOW (measured: 7/18 forms ANS queries false-refuse, retrieval is fine; the
gold chunk reaches the top-5 reranked window every time). The cross-doc catch f1040-04 is the same
disease: the "Single …" query pulled the W-4's 2026 $16,100 instead of the 1040's 2025 $15,750.

This node resolves a `• <label>, $<value>` bullet list against the query **deterministically (no
LLM)** — token-overlap routing to the single dominant-matching label (the `artifact_scope.py`
precedent) — and injects ONE clean synthetic chunk (`"Standard deduction for Head of household:
$23,625"`) into `state.reranked` so the gates see the label+value together, unambiguously.

**HARD-gate-safe BY CONSTRUCTION:** verbatim-value-or-drop + additive-only injection (the
`query_tables` synthetic-chunk seam). The synthetic chunk carries no table marker, so the numeric
backstop skips it; it is a single substantive line, so `is_name_only_chunk` is False; `verify`
still grounds it against the original cited chunks, and `route_after_verify` still refuses a
zero-grounded draft — so `refusal_cf` cannot regress. The `• label, $value` shape fires on **2
chunks vault-wide (both f1040; measured across 187 docs / 7,743 chunks)** → a deterministic no-op
everywhere else. Default-ON; kill-switch `MEMEX_AGENTS__FORM_FIELD_RESOLVER_ENABLED=false`.
"""

from __future__ import annotations

import re

from memex.core.config import get_settings
from memex.core.errors import ConfigurationError, MemexError
from memex.core.text import STOPWORDS
from memex.core.types import Chunk

# `• <label>, $<value>` — a labeled monetary bullet in a run-on form cell. The label is the text
# up to the comma (no embedded bullet/comma/newline); the value is a `$`-prefixed number.
_BULLET_RE = re.compile(r"•\s*(?P<label>[^•,\n]{2,80}?)\s*,\s*(?P<value>\$[\d,]+(?:\.\d+)?)")
_BOLD_RE = re.compile(r"\*\*\s*([^*]+?)\s*\*\*")
_WORD_RE = re.compile(r"[a-z0-9]+")
_MIN_LABEL_OVERLAP = 2  # a confident route needs >=2 shared content tokens
_MAX_FIELD_TEXT = 200


def _content_tokens(text: str) -> set[str]:
    """Lowercase content tokens (>1 char, non-stopword) — phrase-level, unlike per-word atomise."""
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 1 and w not in STOPWORDS}


def extract_bullet_fields(text: str) -> list[tuple[str, str, str]]:
    """Find every `• <label>, $<value>` bullet + its section concept.

    The concept is the nearest preceding **bold** phrase run (e.g. `**Standard** **deduction
    for—**` → "Standard deduction for"), which carries the WHAT (standard deduction) the bullets
    enumerate by filing status. Returns `(concept, label, value)` tuples in document order.
    """
    out: list[tuple[str, str, str]] = []
    for m in _BULLET_RE.finditer(text):
        label = m.group("label").strip()
        value = m.group("value")
        bolds = _BOLD_RE.findall(text[: m.start()])
        concept = " ".join(b.strip() for b in bolds[-2:]).replace("—", " ")
        concept = re.sub(r"\s+", " ", concept).strip()
        # drop a trailing connector ("Standard deduction for" → "Standard deduction") so the
        # builder's " for <label>" join reads cleanly instead of doubling the word.
        concept = re.sub(r"\s*\b(?:for|of|to)\b\s*$", "", concept, flags=re.IGNORECASE).strip()
        out.append((concept, label, value))
    return out


def route_form_field(query: str, fields: list[tuple[str, str, str]]) -> tuple[str, str, str] | None:
    """Route the query to the SINGLE dominant-matching bullet label, else None.

    Scores each bullet by content-token overlap between the query and the label; fires only on a
    clear unique winner with >= `_MIN_LABEL_OVERLAP` shared tokens (the conservative single-dominant
    gate from `artifact_scope.py` — a tie or weak match no-ops, so an ambiguous query is never
    mis-routed)."""
    q = _content_tokens(query)
    scored = sorted(
        ((len(q & _content_tokens(label)), concept, label, value) for concept, label, value in fields),
        key=lambda t: t[0],
        reverse=True,
    )
    if not scored or scored[0][0] < _MIN_LABEL_OVERLAP:
        return None
    if len(scored) > 1 and scored[0][0] <= scored[1][0]:
        return None  # not a clear winner → no-op (safe)
    _, concept, label, value = scored[0]
    return concept, label, value


def build_form_field_chunk(query: str, chunks: list[Chunk]) -> Chunk | None:
    """Resolve a form-field query to ONE synthetic `label: value` chunk, or None (no-op).

    Scans the reranked chunks for a `• label, $value` bullet list, routes the query to the dominant
    label, and — only when the label AND value are VERBATIM in the source chunk (the fabrication
    boundary) — builds a clean synthetic chunk. The first chunk that yields a confident route wins.
    """
    for c in chunks:
        fields = extract_bullet_fields(c.text)
        if not fields:
            continue
        routed = route_form_field(query, fields)
        if routed is None:
            continue
        concept, label, value = routed
        if label not in c.text or value not in c.text:  # verbatim-or-drop
            continue
        prefix = f"{concept} for " if concept else ""
        synthetic_text = f"{prefix}{label}: {value}"[:_MAX_FIELD_TEXT]
        return Chunk(
            chunk_id=f"{c.document_id}#field0001",
            document_id=c.document_id,
            document_title=c.document_title,
            text=synthetic_text,
            char_start=c.char_start,
            char_end=c.char_end,
            score=0.0,
            rerank_score=None,
            heading_path=c.heading_path,
        )
    return None


def form_field_resolver_enabled() -> bool:
    """Read the form-field-resolver flag, FAIL-OPEN to OFF on a config-read failure.

    **Default ON** (`AgentsSettings.form_field_resolver_enabled`); kill-switch
    `MEMEX_AGENTS__FORM_FIELD_RESOLVER_ENABLED=false`.
    """
    try:
        return bool(get_settings().agents.form_field_resolver_enabled)
    except (ConfigurationError, MemexError):
        return False
