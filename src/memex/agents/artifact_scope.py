"""Deterministic artifact→document resolution for the answering agent (#256).

When a query NAMES a specific artifact that lives in one document — e.g.
cr350-diag-12 "Quelle est la plage d'adresses VLAN configurée dans **le diagramme
de coupe-feu**?" — the answer must come from THAT document, not a different one
that merely contains the data term ("VLAN range"). Left unscoped, the agent
answered the firewall-diagram question from a *lecture's* VLAN content (grounded
but wrong-source). A prior LLM `assess_relevance` "source-scope" prompt clause
was validated INEFFECTIVE and reverted (Qwen3-8B gave 0 non-responsive
verdicts), so this is the **deterministic** replacement: regex detect + BM25
resolve, no model.

Two pure stages, each deterministic for a fixed FTS index:

    detect_artifact_reference(query)         -> ArtifactReference | None
    resolve_scope(ref, search, ...)          -> ResolvedArtifactScope

The answering node (`agents/answering.py::resolve_artifact_scope`) wires these
to a scoped re-retrieval: a confident resolution makes the node REPLACE the
candidate pool with chunks from the artifact's doc(s), so the existing
assess/verify gates answer from the right source or refuse naturally. For the
bug case this makes the outcome DETERMINISTIC: the wrong-source evidence is
removed before the LLM sees it, so the refusal no longer depends on greedy
sampling.

Conservative everywhere — no artifact, an ambiguous (multi-)artifact, a numbered
"Figure 3" ref, an empty qualifier ("the diagram"), a weak/spread BM25 result —
all yield NO scope, i.e. today's full-corpus path. The mechanism can therefore
only NARROW retrieval for clearly artifact-named queries; it never turns an
open / multi-file query into a refusal, and the existing gates still decide
answer-vs-refuse.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from memex.core.text import STOPWORDS as _STOP
from memex.core.text import atomise as _atomise
from memex.core.types import Chunk

# ── Artifact-type vocabulary (EN + FR). Deliberately omits common nouns that
# read as artifacts only rarely and carry high false-positive risk in this
# corpus: "table"/"tableau" (Table-RAG owns tabular Qs; "the periodic table"),
# "plan" ("le plan de cours" = the syllabus), "map"/"graph" (too generic). The
# qualifier-must-be-non-empty + confident-resolution gates make even an
# accidental match a no-op, but a tight vocabulary is the first line of defence.
_ARTTYPES_EN: tuple[str, ...] = (
    "diagrams",
    "diagram",
    "figures",
    "figure",
    "charts",
    "chart",
    "schematic",
    "schema",
    "topology",
    "timelines",
    "timeline",
    "screenshots",
    "screenshot",
    "illustrations",
    "illustration",
    "flowcharts",
    "flowchart",
)
_ARTTYPES_FR: tuple[str, ...] = (
    "diagrammes",
    "diagramme",
    "figures",
    "figure",
    "schémas",
    "schéma",
    "schemas",
    "schema",
    "graphiques",
    "graphique",
    "captures",
    "capture",
    "illustrations",
    "illustration",
    "topologie",
    "organigrammes",
    "organigramme",
)

# EN: "the/this/that <≤3 qualifier words> <arttype>" — the qualifier MODIFIES the
# artifact noun and precedes it; the data term ("VLAN range") sits elsewhere.
_EN_PATTERN = re.compile(
    r"\b(?:the|this|that)\s+"
    r"(?P<qual>(?:[\w'’-]+\s+){0,2}[\w'’-]+)\s+"
    r"(?P<arttype>" + "|".join(_ARTTYPES_EN) + r")\b",
    re.IGNORECASE,
)

# FR: "<arttype> <connector> <tail>" — the qualifier FOLLOWS the artifact noun via
# a connector ("diagramme de coupe-feu"). A REQUIRED connector means a bare "le
# diagramme <verbe>…" (no connector) does NOT capture the following verb/data as
# a qualifier — it falls through to no-op. The data term precedes the artifact
# noun ("…VLAN configurée dans le diagramme de coupe-feu"), so it is never in the
# tail.
_FR_PATTERN = re.compile(
    r"\b(?P<arttype>" + "|".join(_ARTTYPES_FR) + r")\b\s+"
    # Connector: a space-separated article ("de la"/"du"/"des"/"de" + required
    # trailing space) OR an ELIDED article ("de l'"/"d'" — the apostrophe binds
    # directly to the next word, so NO trailing space, e.g. "diagramme d'attaque").
    r"(?:(?:de\s+la|du|des|de)\s+|(?:de\s+l|d)['’]\s*)"
    r"(?P<tail>.+)$",
    re.IGNORECASE | re.DOTALL,
)

# Bare numbered ref ("Figure 3", "Tableau 2") — no content qualifier, and there
# is no vault-wide figure-number index, so it is a conservative no-op.
_NUMBERED_PATTERN = re.compile(
    r"\b(?:figures?|fig\.?|diagrammes?|diagrams?|tables?|tableaux?|tableau|charts?|"
    r"schémas?|schemas?|plates?)\s+\d{1,3}\b",
    re.IGNORECASE,
)

# FR words that END the qualifier run (interrogatives / prepositions / verbs that
# begin the *question* rather than the artifact name).
_BOUNDARY_FR: frozenset[str] = frozenset(
    {
        "quel",
        "quelle",
        "quels",
        "quelles",
        "comment",
        "combien",
        "pourquoi",
        "où",
        "ou",
        "selon",
        "dans",
        "par",
        "pour",
        "entre",
        "sur",
        "avec",
        "sans",
        "est",
        "sont",
        "ont",
        "et",
        "mais",
        "car",
        "donc",
        "que",
        "qui",
        "vers",
        "afin",
        "ainsi",
        "lorsque",
        "quand",
        "à",
    }
)
# Tokens dropped from a qualifier (articles/connectors + generic corpus words).
# `_STOP` (bilingual stopwords) + `_atomise` (diacritics-/hyphen-aware tokenizer) are now
# shared from `core/text.py` (imported above) so `index/fts_store.py`'s FTS query builder
# reuses the exact same set/tokenizer without importing UP from `agents/`.

# Max surface words captured as a qualifier (keeps it the artifact name, not a clause).
_MAX_QUALIFIER_WORDS = 4

# ── Resolution thresholds (deterministic gates; pinned by unit tests). ──
_QUALIFIER_SEARCH_K = 30  # per-token BM25 depth
_MIN_HITS = 2  # a doc needs >=2 qualifier-matching chunks to count
_STRONG_HITS = 3  # hits that qualify a doc for the small-set without a title hit
_DOMINANCE_RATIO = 2.0  # top.hits >= ratio * second.hits => single dominant doc
_SET_BAND = 0.5  # small-set members keep >= band * top.hits
_MAX_SCOPE_DOCS = 2  # a named artifact lives in 1, at most 2 (doc + its lecture), docs


@dataclass(frozen=True)
class ArtifactReference:
    """A named artifact detected in a query.

    `qualifier_tokens` are the diacritics-/hyphen-folded, stopword-dropped ATOMIC
    content tokens BOUND to the artifact noun (e.g. ``{"coupe", "feu"}`` /
    ``{"gantt"}``) — NEVER the data term. `arttype` is the matched artifact noun.
    `numbered` is True for a bare "Figure N" ref (empty qualifier → no-op).
    """

    arttype: str
    qualifier_tokens: frozenset[str]
    numbered: bool = False
    surface: str = ""


@dataclass(frozen=True)
class ResolvedArtifactScope:
    """The document(s) a named artifact resolves to. `doc_ids` empty ⇒ NO scope
    (full-corpus path unchanged). `via` records the decision branch for tracing."""

    doc_ids: tuple[str, ...]
    via: str  # dominant | small-set | none-no-qualifier | none-weak | none-spread


# ── Detection ──────────────────────────────────────────────────────────────


def _tokens_before(qual: str) -> frozenset[str]:
    """EN qualifier: the ≤3 surface words preceding the artifact noun."""
    toks: set[str] = set()
    for raw in qual.split():
        for atom in _atomise(raw):
            if atom not in _STOP and len(atom) >= 2:
                toks.add(atom)
    return frozenset(toks)


def _tokens_from_tail(tail: str) -> frozenset[str]:
    """FR qualifier: walk the tail after the connector, stop at the first FR
    boundary word or after `_MAX_QUALIFIER_WORDS` surface words, drop stopwords."""
    toks: set[str] = set()
    surface = 0
    for raw in re.split(r"[\s,.?!;:()«»\"]+", tail):
        if not raw:
            continue
        low = re.sub(r"[^0-9a-zàâäçéèêëîïôöùûüÿœæ'’-]", "", raw.lower())
        if not low:
            continue
        head = low.split("'")[0].split("’")[0]
        if head in _BOUNDARY_FR:
            break
        surface += 1
        if surface > _MAX_QUALIFIER_WORDS:
            break
        for atom in _atomise(low):
            if atom not in _STOP and len(atom) >= 2:
                toks.add(atom)
    return frozenset(toks)


def detect_artifact_reference(query: str) -> ArtifactReference | None:
    """Detect a single named artifact + its qualifier tokens, or None.

    Returns None when: no artifact noun is named; the query names ≥2 DISTINCT
    artifacts (ambiguous which to scope to → conservative no-op); or the only
    match is a bare numbered ref / has an empty qualifier ("the diagram"). Pure;
    no I/O; fully deterministic."""
    if not query or not query.strip():
        return None

    refs: list[ArtifactReference] = []
    for m in _EN_PATTERN.finditer(query):
        toks = _tokens_before(m.group("qual"))
        if toks:
            refs.append(
                ArtifactReference(m.group("arttype").lower(), toks, False, m.group(0).strip())
            )
    for m in _FR_PATTERN.finditer(query):
        toks = _tokens_from_tail(m.group("tail"))
        if toks:
            refs.append(
                ArtifactReference(m.group("arttype").lower(), toks, False, m.group(0)[:60].strip())
            )

    if refs:
        distinct = {r.qualifier_tokens for r in refs}
        if len(distinct) >= 2:
            return None  # ambiguous: multiple distinct named artifacts → no scope
        return refs[0]

    # No qualified artifact — a bare numbered ref is a recorded no-op (visible to tests).
    nm = _NUMBERED_PATTERN.search(query)
    if nm:
        return ArtifactReference(nm.group(0).split()[0].lower(), frozenset(), True, nm.group(0))
    return None


# ── Resolution ─────────────────────────────────────────────────────────────

_TITLE_SPLIT = re.compile(r"[^0-9a-zàâäçéèêëîïôöùûüÿœæ]+")


def _strip_accents(s: str) -> str:
    """NFD diacritic fold — so a French qualifier atom (`atomise` keeps accents: 'réseau') matches the
    ASCII-folded doc-id SLUG ('cr350-reseau') the title is built from. Without this, an accented
    single-token FR qualifier yields an empty `title_hit` → the single-token-specificity gate / the
    sibling-by-title expansion silently mis-fire (a #256 cross-doc misattribution or a false-refuse)."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def _title_overlap(qualifier_tokens: frozenset[str], title: str) -> bool:
    """True iff any qualifier atom appears as a token of the document title
    (tokenised the same atomic way — splits the doc-id slug + diacritics)."""
    title_atoms = {_strip_accents(t) for t in _TITLE_SPLIT.split(title.lower()) if t}
    return bool({_strip_accents(q) for q in qualifier_tokens} & title_atoms)


async def resolve_scope(
    ref: ArtifactReference | None,
    search: Callable[[str, int], Awaitable[list[Chunk]]],
    *,
    k: int = _QUALIFIER_SEARCH_K,
) -> ResolvedArtifactScope:
    """Resolve a named artifact to the confident document-set it lives in.

    `search(token, k)` is an injected per-token corpus BM25 search (the answering
    node passes `FTSStore.search`; tests fake it). One search PER qualifier atom
    (FTS phrase-quotes its input, so per-token gives token-OR semantics). Docs are
    ranked by hit-count (sign-agnostic; the flipped-BM25 score is only a
    tie-breaker) with a deterministic `doc_id`-ascending final tie-break, so the
    result is identical run-to-run for a fixed index. Returns an empty scope
    (→ no re-scope) whenever resolution isn't confident."""
    if ref is None or ref.numbered or not ref.qualifier_tokens:
        return ResolvedArtifactScope((), "none-no-qualifier")

    doc_hits: dict[str, int] = {}
    doc_best: dict[str, float] = {}
    doc_title: dict[str, str] = {}
    for token in sorted(ref.qualifier_tokens):
        for c in await search(token, k):
            doc_hits[c.document_id] = doc_hits.get(c.document_id, 0) + 1
            doc_best[c.document_id] = max(doc_best.get(c.document_id, float("-inf")), c.score)
            doc_title.setdefault(c.document_id, c.document_title)

    docs = [d for d, h in doc_hits.items() if h >= _MIN_HITS]
    if not docs:
        return ResolvedArtifactScope((), "none-weak")

    # Deterministic order: most hits, then best single-chunk score, then doc_id.
    ranked = sorted(docs, key=lambda d: (-doc_hits[d], -doc_best[d], d))
    title_hit = {d for d in docs if _title_overlap(ref.qualifier_tokens, doc_title[d])}

    top = ranked[0]
    if len(ranked) == 1 or doc_hits[top] >= _DOMINANCE_RATIO * doc_hits[ranked[1]]:
        base, via = [top], "dominant"
    else:
        band = [
            d
            for d in ranked
            if doc_hits[d] >= max(_MIN_HITS, _SET_BAND * doc_hits[top])
            and (d in title_hit or doc_hits[d] >= _STRONG_HITS)
        ]
        if 1 <= len(band) <= _MAX_SCOPE_DOCS:
            base, via = band, "small-set"
        else:
            return ResolvedArtifactScope((), "none-spread")  # topic, not a single artifact

    # Single-token specificity gate. A LONE qualifier token ("tsmc", "gantt") is
    # weak evidence of an artifact's HOME: a cross-document entity ("the TSMC
    # chart") can have its in-chart mention FTS-stripped, so the bare token then
    # dominates whatever *other* doc merely discusses the entity in prose —
    # mis-scoping an answerable query to the wrong doc (a false-refuse). So a
    # single-token qualifier only scopes when it's distinctive: the resolved doc
    # carries it in the TITLE, or it's the SOLE doc with any hit on the token.
    # Otherwise → no-scope (full-corpus path; the query answers there as before).
    # Multi-token qualifiers are exempt — a conjunction ("coupe"+"feu",
    # "configuration"+"vlan"+"commutateur") is a strong artifact-name signal, and
    # that's where the #256 disambiguation fixes (diag-12, img-01) actually live.
    if (
        len(ref.qualifier_tokens) == 1
        and len(doc_hits) > 1
        and not any(d in title_hit for d in base)
    ):
        return ResolvedArtifactScope((), "none-singletoken-ambiguous")

    # Sibling-by-title: when a resolved doc's TITLE carries the qualifier (the
    # firewall LECTURE), admit other title-matching docs (the firewall DIAGRAM
    # doc) so the scope spans the artifact's doc + its companion, capped.
    if any(d in title_hit for d in base):
        for d in ranked:
            if len(base) >= _MAX_SCOPE_DOCS:
                break
            if d not in base and d in title_hit and doc_hits[d] >= _MIN_HITS:
                base.append(d)

    return ResolvedArtifactScope(tuple(base[:_MAX_SCOPE_DOCS]), via)
