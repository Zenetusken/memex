"""Deterministic acronym ↔ expansion bridging for entity discovery (ADR-0011).

Pure, stdlib-only, unit-tested without a graph — the ethos-twin of
`agents/artifact_scope.py` (no LLM, no fuzzy matching; the project resolves
identity exactly and only SUGGESTS bridges, never merges them). Consumed by
`index/graph_store.py::entity_profile` to surface "Also see" / "Did you mean?"
links between an acronym entity (`DNS`) and its multi-word expansion
(`Domain Name System`) when both exist as separate nodes in the graph.

The bridge is an INITIALISM match: the first letter of each significant word of
a multi-word name forms its acronym (`Domain Name System` → `DNS`). This is
immune to the substring-CONTAINS noise that makes a naive fallback useless
(CONTAINS `dns` matches 34 entities incl. `DNS Spoofing`; CONTAINS `stp` matches
only the junk substring `NISTPRIV`). It only connects a real acronym to a real
expansion — and where neither exists (the `STP`-stored-as-`spanning` case), it
correctly yields nothing.
"""

from __future__ import annotations

import re

# Closed-class connector words that do NOT contribute a letter to an initialism.
# Standard convention: "Domain Name System" → DNS, but the connectors in
# "Système de Noms de Domaine" (FR; the CR350 corpus is bilingual) are dropped so
# it derives SND. Only function words conventional acronyms omit — kept small and
# predictable (NOT a general stopword list).
_INITIALISM_SKIP: frozenset[str] = frozenset(
    {
        # EN
        "of", "the", "a", "an", "and", "or", "for", "to", "in", "on",
        # FR
        "de", "des", "du", "la", "le", "les", "d", "l",
    }
)

# Tokenise on whitespace, hyphen family, and slash so
# "Address-Resolution-Protocol" derives the same initialism as the spaced form.
_WORD_SPLIT = re.compile(r"[\s/–—-]+")

_MIN_INITIALISM_LEN = 2  # a 1-letter "acronym" is meaningless + collision-prone
_MAX_INITIALISM_LEN = 7  # matches the corpus's ^[A-Z0-9.+-]{2,7}$ acronym-shape family
_MIN_ALPHA_SHARE = 0.5  # ≥50% of a derived initialism's chars must be letters (kill "8.1"-junk)


def derive_initialism(name: str) -> str | None:
    """The bare-acronym form a MULTI-WORD name abbreviates to, or `None`.

    "Domain Name System" → "DNS"; "Address Resolution Protocol" → "ARP";
    "Dynamic Host Configuration Protocol" → "DHCP". Skips closed-class connectors
    (`of`/`the`/FR `de`/`la`). Returns `None` for a single significant word (no
    initialism is meaningful — that's the exact-match case, not a bridge) or when
    fewer than `_MIN_INITIALISM_LEN` letters survive. Deterministic + pure.

    Each significant token contributes its FIRST alphanumeric char, so a numeric
    token ("802.1X") contributes "8"; `initialism_matches`'s alpha-share guard then
    suppresses such junk derivations. Capped at `_MAX_INITIALISM_LEN` (a very long
    name is not an acronym expansion).
    """
    if not name or not name.strip():
        return None
    letters: list[str] = []
    significant_words = 0
    for raw in _WORD_SPLIT.split(name.strip()):
        tok = raw.strip().lower()
        if not tok or tok in _INITIALISM_SKIP:
            continue
        # First alphanumeric char — UNICODE-aware (`str.isalnum`, not an ASCII regex),
        # so an accented leading letter ("Émetteur" → "É") is taken, not skipped to the
        # next ASCII char ("m"). The bilingual CR350 corpus has accented entity names.
        first = next((c for c in tok if c.isalnum()), None)
        if first is None:
            continue
        significant_words += 1
        letters.append(first)
    if significant_words < 2 or len(letters) < _MIN_INITIALISM_LEN:
        return None
    if len(letters) > _MAX_INITIALISM_LEN:
        return None
    return "".join(letters).upper()


def looks_like_acronym(query: str) -> bool:
    """True for a query that could BE an acronym (so the acronym→expansion scan is
    worth running): a single token, `_MIN_INITIALISM_LEN`–`_MAX_INITIALISM_LEN`
    chars, only `[A-Za-z0-9.+-]`, at least one letter, AND (has an uppercase letter
    OR is short, len ≤ 6). The final clause keeps `DNS`/`dns`/`stp` in but excludes
    a long all-lowercase real word like `spanning` (8 chars) so a plain word doesn't
    trigger the full entity-name scan needlessly."""
    s = query.strip()
    if not (_MIN_INITIALISM_LEN <= len(s) <= _MAX_INITIALISM_LEN):
        return False
    if re.fullmatch(r"[A-Za-z0-9.+-]+", s) is None:
        return False
    if not any(c.isalpha() for c in s):
        return False
    return any(c.isupper() for c in s) or len(s) <= 6


def initialism_matches(query_acronym: str, candidate_name: str) -> bool:
    """True iff `candidate_name`'s derived initialism EQUALS `query_acronym`
    (case-insensitive) — the bridge condition for the acronym→expansion direction.
    Guards on a minimum alpha-share so a digit-heavy derivation can't match."""
    derived = derive_initialism(candidate_name)
    if derived is None:
        return False
    alpha = sum(c.isalpha() for c in derived)
    if alpha / len(derived) < _MIN_ALPHA_SHARE:
        return False
    return derived == query_acronym.strip().upper()
