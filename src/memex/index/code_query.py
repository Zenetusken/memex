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

This module ALSO hosts the **usage-intent rerank demotion** lever (the answer-stage complement
to Lever A's retrieval lever). Lever A puts a usage/reference gold (the CALLER of a symbol) in
the rerank pool; but for a "which function calls X" query the cross-encoder still ranks X's own
DEFINITION (and X's test chunks) above the caller, and the answer LLM then describes the
definition whenever it is in the visible top-k window. `detect_usage_intent` +
`reorder_for_usage_intent` demote X's definition + test chunks BELOW the top-k cut for a
usage-intent query, so the answer node sees the caller instead. Pure reorder ⇒ HARD-gate-safe by
construction. **Measured DOUBLE-EDGED on the find-the-code usage set (+3 fixed / −2 regressed:
demoting the definition is a distractor when the LLM anchors on it, but necessary CONTEXT when a
similarly-named sibling needs disambiguating or the subject needs anchoring) → shipped
DEFAULT-OFF (opt-in, `usage_intent_demotion_enabled`).** See `docs/audits/14` + ADR-0021
(usage-intent-demotion amendment).
"""

from __future__ import annotations

import re

from memex.core.config import get_settings
from memex.core.errors import ConfigurationError, MemexError
from memex.core.text import STOPWORDS
from memex.core.types import Chunk

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


# ──────────────────────────────────────────────────────────────────────────────
# Usage-intent rerank demotion (the answer-stage complement to Lever A)
# ──────────────────────────────────────────────────────────────────────────────

# Usage verbs (with their past tenses): "which <noun> <verb> X" / "where is X <verb>". The query
# asks to IDENTIFY a DIFFERENT entity that acts on X (the caller/owner/wrapper) — X's own
# definition is, by construction, NOT the answer. Definitional verbs (`defines`/`implements`) are
# DELIBERATELY excluded — "where is X defined" is a definition query, not a usage query.
_USAGE_VERB_RE = re.compile(
    r"\b(?:call(?:s|ed)?|invoke[sd]?|use[sd]?|using|own(?:s|ed)?|hold(?:s|ed)?|held"
    r"|wrap(?:s|ped)?|create[sd]?|creating|instantiate[sd]?|construct(?:s|ed)?"
    r"|spawn(?:s|ed)?|reference[sd]?|consume[sd]?)\b",
    re.IGNORECASE,
)
# A whole identifier token (the candidate symbol X). The `_looks_like_code_identifier` filter then
# keeps only snake_case / camelCase-or-PascalCase tokens — the structural code-nouns the query
# uses (`function`/`struct`/`enum`/`method`) are plain lowercase words, so X is unambiguous.
_IDENT_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_INTERNAL_CAPS_RE = re.compile(r"[a-z][A-Z]")
# Rust symbol-kind keywords that prefix an injected `## <kind> <name>` chunk heading
# (`index/rust_symbols.py`). `last_ident` strips them to recover the bare symbol name.
_SYMBOL_KIND_KEYWORDS: frozenset[str] = frozenset(
    {
        "fn",
        "struct",
        "enum",
        "union",
        "trait",
        "type",
        "mod",
        "const",
        "static",
        "impl",
        "macro_rules",
        "macro",
    }
)


def _looks_like_code_identifier(token: str) -> bool:
    """Whether `token` is a code identifier: snake_case (`_`) or camelCase/PascalCase (a lowercase
    letter immediately followed by an uppercase). Mirrors `_CODE_IDENTIFIER_RE`'s intent — a bare
    ALL-CAPS acronym (PIX, TUI) does NOT qualify (it would be too ambiguous to treat as the symbol);
    a SCREAMING_CASE constant (EMBEDDED_INSTRUCTIONS) qualifies via its `_`.
    """
    return ("_" in token) or (_INTERNAL_CAPS_RE.search(token) is not None)


def _extract_query_symbol(query: str) -> str | None:
    """The first code-identifier token in `query` — the symbol X the usage query names. Every
    find-the-code usage query names exactly one (the structural nouns are plain words), so the
    first match is unambiguous.
    """
    for m in _IDENT_TOKEN_RE.finditer(query):
        tok = m.group(0)
        if _looks_like_code_identifier(tok):
            return tok
    return None


def detect_usage_intent(query: str) -> str | None:
    """The named symbol X iff `query` asks which entity CALLS/USES X (usage intent), else None.

    Fires on "which <function|struct|…> <usage-verb> … X" and "where is X <usage-verb>"; stays
    SILENT on DEFINITION intent ("what does X do", "how is X implemented", "what fields does X
    define") where X's own definition IS the answer. The discriminator is the interrogative frame
    (`which …` / `where is …`) PLUS a usage verb — NOT the mere presence of an identifier (both
    classes name one). Validated on all 47 find-the-code queries: 17/17 usage fire; 0/16
    definition; 0/6 big-function. (A counterfactual phrased "which function calls <nonexistent>"
    fires but is a harmless no-op — no chunk defines the absent symbol, so nothing is demoted.)
    """
    q = query.strip().lower()
    if q.startswith("what ") or q.startswith("how "):  # definition intent — X IS the answer
        return None
    if not _USAGE_VERB_RE.search(q):
        return None
    if not (re.search(r"\bwhich\b", q) or re.search(r"\bwhere\s+is\b", q)):
        return None
    return _extract_query_symbol(query)


def last_ident(heading: str) -> str:
    """The defined symbol NAME in an injected `## <kind> <name>` chunk heading.

    Strips a `Type::` path prefix (a method heading) then a leading kind keyword:
    `"fn is_known_safe_command"` → `"is_known_safe_command"`, `"struct Session"` → `"Session"`,
    `"Session::client"` → `"client"`, `"mod tests"` → `"tests"`. A bare heading with no kind
    keyword returns its last whitespace token.
    """
    seg = heading.split("::")[-1].strip()
    parts = seg.split()
    if len(parts) >= 2 and parts[0] in _SYMBOL_KIND_KEYWORDS:
        return parts[-1]
    return parts[-1] if parts else seg


def _chunk_defines_symbol(chunk: Chunk, symbol: str) -> bool:
    """Whether `chunk`'s OWN deepest heading IS the definition of `symbol` (not merely a chunk that
    lives under a module named like it) — the precise demotion target for a usage query.
    """
    hp = chunk.heading_path
    return bool(hp) and last_ident(hp[-1]) == symbol


def _is_test_chunk(chunk: Chunk) -> bool:
    """Whether `chunk` is a Rust test chunk — under a `#[cfg(test)] mod tests` (a `tests` segment in
    its heading path) OR a `test_…`-named symbol. Narrow on purpose: a symbol like `attest_…` whose
    NAME merely contains "test" is NOT a test (the `test_` prefix + the `tests` module are the real
    markers), so it is not demoted.
    """
    for seg in chunk.heading_path:
        s = seg.strip().lower()
        if s == "mod tests" or s == "tests" or s.startswith("tests::") or "::tests::" in s:
            return True
        if last_ident(seg).lower().startswith("test_"):
            return True
    return False


def reorder_for_usage_intent(reranked: list[Chunk], symbol: str) -> list[Chunk]:
    """Stable 2-tier demotion for a usage-intent query naming `symbol`: keep all
    non-(definition-of-symbol)-and-non-test chunks in their rerank order, then append the
    definition + test chunks (also in rerank order). The caller slices the result to the answer
    node's top-k, so `symbol`'s own definition + tests fall BELOW the cut and "which function calls
    X" surfaces the CALLER instead of X's definition. Order-preserving among kept chunks ⇒ minimal
    perturbation; pure reorder ⇒ HARD-gate-safe (the downstream `verify` gate is untouched).
    """
    keep: list[Chunk] = []
    demote: list[Chunk] = []
    for c in reranked:
        if _chunk_defines_symbol(c, symbol) or _is_test_chunk(c):
            demote.append(c)
        else:
            keep.append(c)
    return keep + demote


def usage_intent_demotion_enabled() -> bool:
    """Read the usage-intent demotion flag, FAIL-OPEN to OFF (no reorder) on a config-read failure.

    **Default OFF** (`AgentsSettings.usage_intent_demotion_enabled`) — measured double-edged
    (audits/14), kept as opt-in infra. `MEMEX_AGENTS__USAGE_INTENT_DEMOTION_ENABLED=true` enables it.
    """
    try:
        return bool(get_settings().agents.usage_intent_demotion_enabled)
    except (ConfigurationError, MemexError):
        return False
