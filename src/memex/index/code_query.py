"""Code-aware FTS query construction (the Phase-3 "Lever A" code-only term path).

The live FTS wraps the whole query as one literal phrase (`fts_store.py::search`), which
matches ~nothing for a natural-language question — benign for PROSE (BM25 recall ⊆ dense,
`docs/audits/09`) but useless for CODE, where the gold chunk is often titled by a DIFFERENT
symbol and the queried identifier lives in the chunk BODY. The Phase-3 arm-separation
measurement (`docs/audits/13`) proved a term-WHOLE BM25 arm recovers those usage/reference
golds (dense 0.70 → bm25-term 1.00) with no benefit-loss on definitions.

This module builds that term arm — SCOPED to queries that NAME a code identifier, gated by a
config flag — so the prose phrase-wrap is never touched on a natural-language question. The
term builder uses **whole, individually-quoted identifiers** (underscore-aware, NEVER
case-split): the `unicode61` index keeps PascalCase whole and splits snake_case on `_`, and
`core.text.atomise` STRIPS `_` into a mash, so it is deliberately not used here. See
`docs/specs/code-chunking.md` §Phase 3 + ADR-0021.
"""

from __future__ import annotations

import re

from memex.core.config import get_settings
from memex.core.errors import ConfigurationError, MemexError
from memex.core.text import STOPWORDS

# Question / structural / Rust-keyword filler dropped before lexical matching — low-signal,
# and the structural words (`struct`/`fn`/`enum`/…) head EVERY symbol chunk, so keeping them
# would only add noise. The rare IDENTIFIERS are the signal. (Mirrors the validated builder
# in scripts/code_bm25_arm_probe.py.)
_QUESTION_STOP: frozenset[str] = frozenset(
    {
        "which",
        "what",
        "where",
        "when",
        "why",
        "how",
        "who",
        "whom",
        "whose",
        "whether",
        "does",
        "do",
        "did",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "from",
        "into",
        "by",
        "at",
        "as",
        "it",
        "its",
        "and",
        "or",
        "not",
        "no",
        "function",
        "functions",
        "fn",
        "method",
        "methods",
        "struct",
        "structs",
        "enum",
        "enums",
        "trait",
        "traits",
        "impl",
        "implementation",
        "mod",
        "module",
        "modules",
        "const",
        "constant",
        "constants",
        "static",
        "type",
        "types",
        "macro",
        "macros",
        "call",
        "calls",
        "called",
        "calling",
        "use",
        "uses",
        "used",
        "using",
        "define",
        "defined",
        "defines",
        "defining",
        "definition",
        "implement",
        "implemented",
        "implements",
        "work",
        "works",
        "compute",
        "computes",
        "set",
        "sets",
        "value",
        "values",
        "field",
        "fields",
        "variant",
        "variants",
        "own",
        "owns",
        "hold",
        "holds",
        "create",
        "creates",
        "created",
        "invoke",
        "invokes",
        "construct",
        "constructs",
        "constructed",
        "before",
        "after",
        "apply",
        "applying",
        "decide",
        "decides",
        "agent",
        "run",
        "runs",
        "without",
        "asking",
        "approval",
    }
)
_STOPSET: frozenset[str] = STOPWORDS | _QUESTION_STOP
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# A token is a code identifier if it has an internal `_` (snake_case) OR a lowercase letter
# immediately followed by an uppercase one (camelCase / PascalCase). Deliberately simple:
# ALL-CAPS acronyms (FP128, PIX) do NOT fire; mixed-case product names (pfSense, SaaS) DO
# (accepted — the prose HARD-gate re-run, not a cleverer regex, is the safety net).
_CODE_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9]_[A-Za-z0-9]|[a-z][A-Z]")


def query_has_code_identifier(query: str) -> bool:
    """Whether `query` NAMES a code identifier (snake_case or camelCase/PascalCase).

    The per-query gate: term-WHOLE applies ONLY when this is True, so a natural-language
    prose question (no code identifier) keeps the unchanged phrase-wrap.
    """
    return _CODE_IDENTIFIER_RE.search(query) is not None


def build_code_term_match(query: str) -> str:
    """An OR of individually-quoted WHOLE identifiers — the validated term-WHOLE builder.

    Extract identifier tokens, drop stopwords + len<2, and quote each WHOLE token (raw, with
    underscores → a contiguous FTS5 phrase that matches the literal identifier; PascalCase →
    the one whole `unicode61` token). NEVER case-split (a case-split atom can't match the
    Pascal-defining chunk and pollutes the ranking). Returns `""` when no atom survives (the
    caller then falls back to the phrase-wrap).
    """
    atoms: set[str] = set()
    for raw in _TOKEN_RE.findall(query):
        low = raw.lower()
        if len(low) < 2 or low in _STOPSET:
            continue
        atoms.add(low)
    # individually quote each atom (FTS5 operator keywords inside a quoted phrase stay literal)
    return " OR ".join(f'"{a}"' for a in sorted(atoms))


def code_term_query_enabled() -> bool:
    """Read the code-term-query flag, FAIL-OPEN to the safe phrase-wrap.

    Default ON (`AgentsSettings.code_term_query_enabled`); a config-read failure (e.g. outside
    bootstrap) returns False so retrieval degrades to the unchanged phrase-wrap, never an error.
    Kill-switch: `MEMEX_AGENTS__CODE_TERM_QUERY_ENABLED=false`.
    """
    try:
        return bool(get_settings().agents.code_term_query_enabled)
    except (ConfigurationError, MemexError):
        return False
