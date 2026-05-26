"""Unit tests for the deterministic artifact→doc resolver (#256).

Pure functions — no models, no store, ms-fast. Two stages:

    detect_artifact_reference(query)  — regex EN+FR named-artifact detection
    resolve_scope(ref, search)        — per-token BM25 qualifier → confident doc-set

The single most load-bearing assertion is the **diag-12-vs-img-01 discriminator**:
the firewall-diagram question's qualifier must EXCLUDE the data term 'vlan' (so it
scopes to the firewall docs, which have no VLAN content → refuse), while the
VLAN-diagram question's qualifier must INCLUDE 'vlan' (so it scopes to the VLAN
deck → answer). If those two ever collapse, the #256 fix is defeated.

Real corpus query strings are pinned verbatim (not paraphrased) so a query-set
edit that would change detection surfaces here.
"""

from __future__ import annotations

import pytest

from memex.agents.artifact_scope import (
    _MAX_SCOPE_DOCS,
    ArtifactReference,
    ResolvedArtifactScope,
    detect_artifact_reference,
    resolve_scope,
)
from memex.core.types import Chunk

# ===========================================================================
# Stage 1 — detection
# ===========================================================================

# (qid, real-corpus query, expected qualifier token set)
_DETECT_REAL: list[tuple[str, str, set[str]]] = [
    # cr350-diag-12 — THE BUG. 'vlan' precedes "diagramme de" → excluded.
    (
        "diag-12",
        "Quelle est la plage d'adresses VLAN configurée dans le diagramme de coupe-feu ?",
        {"coupe", "feu"},
    ),
    # cr350-img-01 — near-twin. 'vlan' FOLLOWS "diagramme de" → included.
    (
        "img-01",
        "Selon le diagramme de configuration VLAN du commutateur, "
        "quel port sert de port tronc et vers quels VLAN ?",
        {"configuration", "vlan", "commutateur"},
    ),
    (
        "img-02",
        "D'après le diagramme d'architecture réseau, par quels équipements "
        "le trafic passe-t-il entre Internet et l'ordinateur A ?",
        {"architecture", "réseau"},
    ),
    (
        "img-03",
        "Selon le diagramme d'attaque du cours, quelles techniques "
        "l'attaquant combine-t-il pour perturber le réassemblage des paquets ?",
        {"attaque"},
    ),
    (
        "img-ref-01",
        "Selon le diagramme de configuration VLAN du commutateur, "
        "quelle est l'adresse MAC du port Fa0/21 ?",
        {"configuration", "vlan", "commutateur"},
    ),
    (
        "chart-08",
        "In the Gantt chart project status example shown in the paper, "
        "how many projects are 'On Time' versus 'Late'?",
        {"gantt"},
    ),
    (
        "chart-09",
        "Which team members in the Gantt chart example are assigned to GIF projects?",
        {"gantt"},
    ),
]


@pytest.mark.parametrize("qid,query,expected", _DETECT_REAL, ids=[c[0] for c in _DETECT_REAL])
def test_detect_real_corpus_queries(qid: str, query: str, expected: set[str]) -> None:
    ref = detect_artifact_reference(query)
    assert ref is not None, f"{qid}: expected a detection"
    assert ref.qualifier_tokens == frozenset(expected), qid
    assert ref.numbered is False


def test_diag12_qualifier_excludes_the_data_term() -> None:
    """The most load-bearing assertion. If 'vlan' leaks into the firewall
    question's qualifier, re-scope would point at the VLAN deck and answer
    the wrong-source — the original bug."""
    ref = detect_artifact_reference(
        "Quelle est la plage d'adresses VLAN configurée dans le diagramme de coupe-feu ?"
    )
    assert ref is not None
    assert "vlan" not in ref.qualifier_tokens
    assert ref.qualifier_tokens == frozenset({"coupe", "feu"})


def test_img01_qualifier_includes_vlan() -> None:
    ref = detect_artifact_reference(
        "Selon le diagramme de configuration VLAN du commutateur, quel port sert de port tronc ?"
    )
    assert ref is not None
    assert "vlan" in ref.qualifier_tokens


# --- queries that MUST NOT detect (→ no-op → full-corpus path) ---

_DETECT_NONE: list[tuple[str, str]] = [
    ("empty", ""),
    ("whitespace", "   \t  "),
    ("no-artifact", "Quelle est la plage d'adresses VLAN ?"),
    ("generic-en", "What does the diagram show?"),
    ("generic-fr", "Que montre le diagramme ?"),  # bare arttype, no connector+tail
    ("indefinite-en", "What is a Pareto chart?"),  # 'a', not the/this/that
    ("concept-bar", "When should you use a bar chart according to this paper?"),
    ("which-chart", "Who wrote the Tableau paper on which chart or graph to use?"),
    (
        "sankey-sunburst",
        "What's the difference between a Sankey diagram and a Sunburst chart "
        "according to this paper?",
    ),
    ("non-en-fr", "Was zeigt das Diagramm der Brandmauer?"),  # German — no EN/FR trigger
    ("common-noun-table", "What is in the periodic table?"),  # 'table' off-vocab
    ("common-noun-plan", "What does the project plan say?"),  # 'plan' off-vocab
    ("multi-artifact", "Compare the firewall diagram and the network topology."),
]


@pytest.mark.parametrize("qid,query", _DETECT_NONE, ids=[c[0] for c in _DETECT_NONE])
def test_detect_returns_none(qid: str, query: str) -> None:
    assert detect_artifact_reference(query) is None, qid


def test_numbered_ref_detected_with_empty_qualifier() -> None:
    """A bare 'Figure 3' is detected as numbered with no qualifier — resolution
    then no-ops (there is no vault-wide figure-number index)."""
    ref = detect_artifact_reference("Explain Figure 3 in detail.")
    assert ref is not None
    assert ref.numbered is True
    assert ref.qualifier_tokens == frozenset()


def test_multi_artifact_same_qualifier_is_not_ambiguous() -> None:
    """Two artifact nouns sharing ONE qualifier (same artifact named twice) is
    not ambiguous — only DISTINCT qualifiers trigger the no-op guard."""
    ref = detect_artifact_reference("the firewall diagram, i.e. the firewall schema")
    assert ref is not None
    assert ref.qualifier_tokens == frozenset({"firewall"})


# --- folding + boundary edge cases ---


def test_hyphen_exploded_into_atoms() -> None:
    ref = detect_artifact_reference("le diagramme de coupe-feu")
    assert ref is not None
    assert ref.qualifier_tokens == frozenset({"coupe", "feu"})


def test_apostrophe_connector_and_atom() -> None:
    ref = detect_artifact_reference("le diagramme d'attaque")
    assert ref is not None
    assert ref.qualifier_tokens == frozenset({"attaque"})


def test_qualifier_word_cap() -> None:
    """At most `_MAX_QUALIFIER_WORDS` (4) surface words are captured as a
    qualifier — beyond that it's a clause, not an artifact name."""
    ref = detect_artifact_reference("le diagramme de alpha beta gamma delta epsilon zeta")
    assert ref is not None
    assert ref.qualifier_tokens == frozenset({"alpha", "beta", "gamma", "delta"})


def test_stopwords_and_generic_words_dropped() -> None:
    """Connectors ('de') and generic corpus words ('cours') never enter a
    qualifier; only the discriminating token survives."""
    ref = detect_artifact_reference("le diagramme du cours de réseau")
    assert ref is not None
    assert "cours" not in ref.qualifier_tokens
    assert "de" not in ref.qualifier_tokens
    assert "réseau" in ref.qualifier_tokens


def test_fr_boundary_word_ends_qualifier() -> None:
    """A FR interrogative/preposition ends the qualifier run — the question
    body never bleeds into the artifact name."""
    ref = detect_artifact_reference("le diagramme de coupe-feu, par quels équipements ?")
    assert ref is not None
    assert ref.qualifier_tokens == frozenset({"coupe", "feu"})


# ===========================================================================
# Stage 2 — resolution (pure given an injected per-token search)
# ===========================================================================


def _chunk(cid: str, doc: str, title: str, score: float) -> Chunk:
    return Chunk(
        chunk_id=f"{doc}#{cid}",
        document_id=doc,
        document_title=title,
        text="x",
        page=1,
        char_start=0,
        char_end=1,
        heading_path=[],
        score=score,
    )


_FW_TITLE = "CR350 Diagrammes coupe-feu"
_LEC_TITLE = "Cours 6 le coupe-feu (firewall)"


def _make_search(corpus: dict[str, list[Chunk]]):  # type: ignore[no-untyped-def]
    async def _search(token: str, k: int) -> list[Chunk]:
        return list(corpus.get(token, []))[:k]

    return _search


@pytest.mark.asyncio
async def test_resolve_dominant_doc_plus_title_sibling() -> None:
    """The real firewall model: the DIAGRAM doc dominates hit-count (so the
    dominant branch fires), and the firewall LECTURE — fewer hits but a title
    match — is folded in by sibling-by-title. Neither has VLAN → downstream
    refuses."""
    corpus = {
        "coupe": [_chunk(f"d{i}", "fw-diagram", _FW_TITLE, 5.0 - i * 0.1) for i in range(6)]
        + [_chunk("l0", "cours-6", _LEC_TITLE, 3.0), _chunk("l1", "cours-6", _LEC_TITLE, 2.0)],
        "feu": [_chunk(f"e{i}", "fw-diagram", _FW_TITLE, 4.0 - i * 0.1) for i in range(4)]
        + [_chunk("l2", "cours-6", _LEC_TITLE, 1.5)],
    }
    ref = detect_artifact_reference("le diagramme de coupe-feu")
    scope = await resolve_scope(ref, _make_search(corpus))
    assert scope.via == "dominant"
    assert set(scope.doc_ids) == {"fw-diagram", "cours-6"}
    assert "semaine-4" not in scope.doc_ids  # the wrong-source doc is excluded


@pytest.mark.asyncio
async def test_resolve_small_set_when_neither_dominates() -> None:
    """Two near-tied title-matching docs (neither dominant) → small-set."""
    corpus = {
        "coupe": [
            _chunk("a", "fw-diagram", _FW_TITLE, 5.0),
            _chunk("b", "fw-diagram", _FW_TITLE, 4.0),
            _chunk("c", "cours-6", _LEC_TITLE, 3.0),
            _chunk("d", "cours-6", _LEC_TITLE, 2.5),
        ],
        "feu": [
            _chunk("e", "fw-diagram", _FW_TITLE, 4.5),
            _chunk("f", "cours-6", _LEC_TITLE, 2.0),
        ],
    }
    ref = ArtifactReference("diagramme", frozenset({"coupe", "feu"}))
    scope = await resolve_scope(ref, _make_search(corpus))
    assert scope.via == "small-set"
    assert set(scope.doc_ids) == {"fw-diagram", "cours-6"}


@pytest.mark.asyncio
async def test_resolve_dominant_single_doc() -> None:
    """img-01: the VLAN qualifier resolves to the single VLAN deck."""
    corpus = {
        "vlan": [_chunk(f"v{i}", "semaine-4", "Semaine 4 VLAN", 6.0 - i) for i in range(3)],
        "configuration": [
            _chunk("c1", "semaine-4", "Semaine 4 VLAN", 3.0),
            _chunk("c2", "semaine-4", "Semaine 4 VLAN", 2.0),
        ],
        "commutateur": [_chunk("m1", "semaine-4", "Semaine 4 VLAN", 3.5)],
    }
    ref = ArtifactReference("diagramme", frozenset({"configuration", "vlan", "commutateur"}))
    scope = await resolve_scope(ref, _make_search(corpus))
    assert scope.via == "dominant"
    assert scope.doc_ids == ("semaine-4",)


@pytest.mark.asyncio
async def test_resolve_single_token_sole_doc_scopes() -> None:
    """chart-08: 'gantt' is the SOLE doc with the token (no competition) — a
    single-token qualifier IS distinctive here, so it scopes (no title hit
    needed)."""
    corpus = {
        "gantt": [
            _chunk("g1", "tableau-guide", "Tableau chart guide", 7.0),
            _chunk("g2", "tableau-guide", "Tableau chart guide", 6.0),
        ]
    }
    ref = ArtifactReference("chart", frozenset({"gantt"}))
    scope = await resolve_scope(ref, _make_search(corpus))
    assert scope.via == "dominant"
    assert scope.doc_ids == ("tableau-guide",)


@pytest.mark.asyncio
async def test_resolve_single_token_ambiguous_no_scope() -> None:
    """The slide-decks-04 'TSMC chart' failure mode: a lone qualifier token whose
    in-artifact mention is FTS-invisible (chart-stripped) dominantly matches an
    UNRELATED doc that merely discusses the entity in prose. With ≥2 docs and no
    title anchor, a single-token qualifier must NOT scope — else it would
    false-refuse an answerable query by pointing at the wrong doc."""
    corpus = {
        # 'tsmc' lands mostly in an unrelated report; the artifact's true home
        # has its TSMC mention locked inside a stripped chart block (invisible).
        "tsmc": [
            _chunk("r1", "annual-report", "2026 Annual Report", 5.0),
            _chunk("r2", "annual-report", "2026 Annual Report", 4.0),
            _chunk("c1", "cuda-deck", "CUDA New Features", 3.0),
        ]
    }
    ref = ArtifactReference("chart", frozenset({"tsmc"}))
    scope = await resolve_scope(ref, _make_search(corpus))
    assert scope.via == "none-singletoken-ambiguous"
    assert scope.doc_ids == ()


@pytest.mark.asyncio
async def test_resolve_single_token_title_hit_scopes_despite_competition() -> None:
    """img-03: 'attaque' competes across docs, but the artifact's home carries it
    in the TITLE — that title anchor makes the lone token distinctive enough to
    scope (the gate exempts title-matching single tokens)."""
    corpus = {
        "attaque": [
            _chunk(f"a{i}", "semaine-5", "Semaine 5 attaque réseau", 5.0 - i * 0.1)
            for i in range(8)
        ]
        + [
            _chunk("x1", "cours-2", "Cours 2 intro", 2.0),
            _chunk("x2", "cours-2", "Cours 2 intro", 1.5),
        ],
    }
    ref = ArtifactReference("diagramme", frozenset({"attaque"}))
    scope = await resolve_scope(ref, _make_search(corpus))
    assert scope.via == "dominant"
    assert scope.doc_ids == ("semaine-5",)


@pytest.mark.asyncio
async def test_resolve_spread_returns_no_scope() -> None:
    """The img-02 canary: a generic 'architecture réseau' qualifier spreads
    across ≥3 near-tied docs with no title anchor → NO scope (full-corpus path),
    so it can never false-refuse via re-scope."""
    corpus = {
        "architecture": [
            _chunk("ar1", "doc-x", "Doc X", 2.0),
            _chunk("ar2", "doc-y", "Doc Y", 2.0),
            _chunk("ar3", "doc-z", "Doc Z", 2.0),
        ],
        "réseau": [
            _chunk("r1", "doc-x", "Doc X", 3.0),
            _chunk("r2", "doc-x", "Doc X", 2.9),
            _chunk("r3", "doc-y", "Doc Y", 3.0),
            _chunk("r4", "doc-y", "Doc Y", 2.8),
            _chunk("r5", "doc-z", "Doc Z", 3.0),
            _chunk("r6", "doc-z", "Doc Z", 2.7),
        ],
    }
    ref = ArtifactReference("diagramme", frozenset({"architecture", "réseau"}))
    scope = await resolve_scope(ref, _make_search(corpus))
    assert scope.via == "none-spread"
    assert scope.doc_ids == ()


@pytest.mark.asyncio
async def test_resolve_weak_returns_no_scope() -> None:
    """A qualifier no document matches → none-weak → no scope."""
    ref = ArtifactReference("chart", frozenset({"zzznotacorpusterm"}))
    scope = await resolve_scope(ref, _make_search({}))
    assert scope.via == "none-weak"
    assert scope.doc_ids == ()


@pytest.mark.asyncio
async def test_resolve_single_hit_doc_below_min_hits() -> None:
    """A doc with only ONE qualifier-matching chunk is below `_MIN_HITS` → it
    does not count → none-weak."""
    corpus = {"gizmo": [_chunk("o1", "doc-q", "Doc Q", 3.0)]}
    ref = ArtifactReference("diagramme", frozenset({"gizmo"}))
    scope = await resolve_scope(ref, _make_search(corpus))
    assert scope.doc_ids == ()


@pytest.mark.asyncio
async def test_resolve_sibling_capped_at_two() -> None:
    """Sibling-by-title never exceeds `_MAX_SCOPE_DOCS`: a dominant doc plus
    THREE title-matching siblings yields exactly two docs."""
    corpus = {
        "coupe": [_chunk(f"d{i}", "fw-diagram", _FW_TITLE, 5.0 - i * 0.1) for i in range(10)]
        + [_chunk("a1", "cours-6", _LEC_TITLE, 3.0), _chunk("a2", "cours-6", _LEC_TITLE, 2.0)]
        + [
            _chunk("b1", "cours-7", "Cours 7 coupe-feu", 2.5),
            _chunk("b2", "cours-7", "Cours 7 coupe-feu", 2.0),
        ],
        "feu": [_chunk("e1", "fw-diagram", _FW_TITLE, 4.0)],
    }
    ref = ArtifactReference("diagramme", frozenset({"coupe", "feu"}))
    scope = await resolve_scope(ref, _make_search(corpus))
    assert len(scope.doc_ids) == _MAX_SCOPE_DOCS == 2
    assert "fw-diagram" in scope.doc_ids  # the dominant doc is always kept


@pytest.mark.asyncio
async def test_resolve_numbered_ref_no_scope() -> None:
    ref = detect_artifact_reference("Explain Figure 3 in detail.")

    async def _never(token: str, k: int) -> list[Chunk]:
        raise AssertionError("a numbered ref must short-circuit before searching")

    scope = await resolve_scope(ref, _never)
    assert scope == ResolvedArtifactScope((), "none-no-qualifier")


@pytest.mark.asyncio
async def test_resolve_none_ref_no_scope() -> None:
    async def _never(token: str, k: int) -> list[Chunk]:
        raise AssertionError("a None ref must short-circuit before searching")

    scope = await resolve_scope(None, _never)
    assert scope.doc_ids == ()


# ===========================================================================
# Determinism — the core property the user demanded
# ===========================================================================


@pytest.mark.asyncio
async def test_deterministic_tie_break_by_doc_id() -> None:
    """Two docs with IDENTICAL hit-count AND identical best score must order by
    `doc_id` ascending — never insertion order — so the scope is byte-stable."""
    corpus = {
        "tok": [
            _chunk("z1", "zzz-doc", "Shared coupe feu", 5.0),
            _chunk("z2", "zzz-doc", "Shared coupe feu", 4.0),
            _chunk("a1", "aaa-doc", "Shared coupe feu", 5.0),
            _chunk("a2", "aaa-doc", "Shared coupe feu", 4.0),
        ]
    }
    ref = ArtifactReference("diagramme", frozenset({"tok"}))
    # 'tok' is not in the titles, so neither doc gets a title hit; equal hits +
    # equal best → the small-set band would be empty (no title, hits<_STRONG_HITS)
    # → none-spread. Use a title-carrying qualifier to force a 2-doc small-set.
    corpus2 = {
        "coupe": [
            _chunk("z1", "zzz-doc", "Shared coupe feu", 5.0),
            _chunk("z2", "zzz-doc", "Shared coupe feu", 4.0),
            _chunk("a1", "aaa-doc", "Shared coupe feu", 5.0),
            _chunk("a2", "aaa-doc", "Shared coupe feu", 4.0),
        ],
        "feu": [
            _chunk("z3", "zzz-doc", "Shared coupe feu", 5.0),
            _chunk("a3", "aaa-doc", "Shared coupe feu", 5.0),
        ],
    }
    ref2 = ArtifactReference("diagramme", frozenset({"coupe", "feu"}))
    results = {await resolve_scope(ref2, _make_search(corpus2)) for _ in range(50)}
    assert len(results) == 1
    (only,) = results
    assert only.doc_ids == ("aaa-doc", "zzz-doc")  # id-ascending, every run
    _ = ref, corpus  # the first construction documents the no-title path


@pytest.mark.asyncio
async def test_determinism_detect_plus_resolve_n50() -> None:
    """Both stages run 50× on the same inputs → byte-identical outputs. Proves
    detection + resolution are 100% deterministic for a fixed index."""
    corpus = {
        "coupe": [_chunk(f"d{i}", "fw-diagram", _FW_TITLE, 5.0 - i * 0.1) for i in range(6)]
        + [_chunk("l0", "cours-6", _LEC_TITLE, 3.0), _chunk("l1", "cours-6", _LEC_TITLE, 2.0)],
        "feu": [_chunk(f"e{i}", "fw-diagram", _FW_TITLE, 4.0 - i * 0.1) for i in range(4)],
    }
    query = "Quelle est la plage d'adresses VLAN configurée dans le diagramme de coupe-feu ?"
    detections = {detect_artifact_reference(query) for _ in range(50)}
    assert len(detections) == 1
    scopes = {
        await resolve_scope(detect_artifact_reference(query), _make_search(corpus))
        for _ in range(50)
    }
    assert len(scopes) == 1


# ===========================================================================
# Config flag
# ===========================================================================


def test_config_flag_default_on() -> None:
    from memex.core.config import AgentsSettings

    assert AgentsSettings().artifact_scope_enabled is True
