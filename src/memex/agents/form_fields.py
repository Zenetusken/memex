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

**Increment D (2026-06-15) adds the `Multiply <desc> by $<value>` worksheet idiom** (W-4 Step 3:
`Multiply the number of qualifying children under age 17 by $2,200` / `… other dependents by $500`)
— the same ADDITIVE, verbatim-or-drop, single-dominant-route mechanism, reaching the per-dependent
credit values the bullet shape can't. Measured surgical: the multiply pattern fires on **1 chunk
vault-wide (the W-4 Step-3 cell; 2 matches)**, the bullet pattern on 2 f1040 cells. It is the
CHURN-FREE complement to the index-time form linearizer, which was BUILT then REVERTED as a measured
negative (re-rendering the chunking body churns retrieval; an additive synthetic chunk cannot —
see `realworld_corpora_2026_06_15`).

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
# `Multiply <desc> by $<value>` — the IRS worksheet idiom (W-4 Step 3, Deductions Worksheet, …):
# `Multiply the number of qualifying children under age 17 by $2,200`. `\s*by\s*` (not `\s+by\s+`)
# tolerates the parser's glued `dependentsby` (no space before `by`); `desc` stops at the first
# `. | • \n` so it never swallows a dotted leader or a column wall. Value `$`-prefixed (the
# fabrication boundary — only a real doc number ships).
_MULTIPLY_RE = re.compile(
    r"[Mm]ultiply\s+(?P<desc>[^.|•\n]{3,80}?)\s*by\s*(?P<value>\$[\d,]+(?:\.\d+)?)"
)
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


def extract_multiply_fields(text: str) -> list[tuple[str, str, str]]:
    """Find every `Multiply <desc> by $<value>` worksheet line + its VERBATIM span.

    Returns `(desc, value, span)` in document order — the routing description, the `$` value, and
    the verbatim matched substring (injected as-is so grounding stays verbatim-or-drop, even where
    the parser glued `dependentsby`)."""
    return [(m.group("desc").strip(), m.group("value"), m.group(0)) for m in _MULTIPLY_RE.finditer(text)]


def route_multiply_field(query: str, fields: list[tuple[str, str, str]]) -> tuple[str, str, str] | None:
    """Route the query to the SINGLE dominant-matching multiply line, else None.

    The same conservative single-dominant overlap gate as `route_form_field` — scores each line by
    query↔desc content-token overlap, fires only on a clear unique winner with
    >= `_MIN_LABEL_OVERLAP` shared tokens (a tie or weak match no-ops)."""
    q = _content_tokens(query)
    scored = sorted(
        ((len(q & _content_tokens(desc)), desc, value, span) for desc, value, span in fields),
        key=lambda t: t[0],
        reverse=True,
    )
    if not scored or scored[0][0] < _MIN_LABEL_OVERLAP:
        return None
    if len(scored) > 1 and scored[0][0] <= scored[1][0]:
        return None  # not a clear winner → no-op (safe)
    _, desc, value, span = scored[0]
    return desc, value, span


def _synthetic_field_chunk(c: Chunk, suffix: str, text: str) -> Chunk:
    """Build an additive synthetic chunk carrying a resolved form value (shared by both patterns)."""
    return Chunk(
        chunk_id=f"{c.document_id}#{suffix}",
        document_id=c.document_id,
        document_title=c.document_title,
        text=text[:_MAX_FIELD_TEXT],
        char_start=c.char_start,
        char_end=c.char_end,
        score=0.0,
        rerank_score=None,
        heading_path=c.heading_path,
    )


def build_form_field_chunk(query: str, chunks: list[Chunk]) -> Chunk | None:
    """Resolve a form-field query to ONE synthetic value chunk, or None (no-op).

    Scans the reranked chunks for either form idiom and routes the query to the single dominant
    match — only when the value is VERBATIM in the source chunk (the fabrication boundary). The
    first chunk that yields a confident route wins:

    1. **labeled monetary bullets** (Increment A) — `• <label>, $<value>` → `{concept} for {label}: {value}`.
    2. **multiply worksheet lines** (Increment D) — `Multiply <desc> by $<value>` → the verbatim span
       (the W-4 Step-3 `$2,200` / `$500` per-dependent credits the bullet shape can't reach).

    Both are ADDITIVE + verbatim-or-drop ⇒ HARD-gate-safe by construction (the `query_tables`
    synthetic-chunk seam); `verify` still grounds the draft against the original chunks and
    `route_after_verify` still refuses a zero-grounded answer, so `refusal_cf` cannot regress.

    Two deliberate, gate-safe design properties (audit-confirmed, no clean tightening): (1) the
    idioms are GENERAL, not IRS-anchored — a lone prose `Multiply <x> by $N` with >= 2 query-token
    overlap routes with only the floor (the dominance gate needs >= 2 matches), but the injected
    span is verbatim source the doc genuinely states for a query that genuinely references it, so a
    fire outside the form domain still surfaces a CORRECT value, never a fabrication; the measured
    blast radius (3 chunks vault-wide) is a corpus fact, not a regex anchor. (2) Precedence is
    FIRST-CONFIDENT-ROUTE-WINS in `state.reranked` order (bullets-then-multiply per chunk), so a
    query that confidently routes to two different form-value chunks ships the earlier one — both
    candidates are verbatim + verify-grounded, so this is an ordering choice, never a safety hole."""
    for c in chunks:
        # 1) labeled monetary bullets (Increment A)
        bullets = extract_bullet_fields(c.text)
        routed = route_form_field(query, bullets) if bullets else None
        if routed is not None:
            concept, label, value = routed
            if label in c.text and value in c.text:  # verbatim-or-drop
                prefix = f"{concept} for " if concept else ""
                return _synthetic_field_chunk(c, "field0001", f"{prefix}{label}: {value}")
        # 2) multiply worksheet lines (Increment D)
        mfields = extract_multiply_fields(c.text)
        mrouted = route_multiply_field(query, mfields) if mfields else None
        if mrouted is not None:
            _desc, value, span = mrouted
            if value in c.text:  # value verbatim (span is a verbatim substring by construction)
                return _synthetic_field_chunk(c, "mult0001", span)
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
