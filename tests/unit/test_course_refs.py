"""Unit tests for `enrich.course_refs.extract_course_references` —
resolving "Cours N" / "Semaine N" mentions to sibling lecture docs.

Mirrors the live CR350 corpus: the syllabus labels every week
"Cours N", but the lecture docs are titled inconsistently ("Cours
1/2/3/6" and "Semaine 4/5"), so resolution is by ordinal number, not
by the Cours/Semaine word.
"""

from __future__ import annotations

from memex.enrich.citations import CitationIndex, make_signature
from memex.enrich.course_refs import extract_course_references
from memex.vault.store import Frontmatter


def _index(titles: dict[str, str]) -> CitationIndex:
    idx = CitationIndex()
    for doc_id, title in titles.items():
        idx.by_id[doc_id] = make_signature(doc_id, Frontmatter(title=title))
    return idx


# The CR350 sibling set (the syllabus's own doc is the source, excluded).
_CR350 = {
    "c1": "CR350 - Cours 1",
    "c2": "CR350 - Cours 2",
    "c3": "CR350 - Cours 3",
    "c6": "CR350 - Cours 6 - Coupe-feu, serveurs mandataires",
    "s4": "CR350 - Semaine 4 - Routeurs, commutateurs",
    "s5": "CR350 - Semaine 5 - Attaques réseau",
}


def test_syllabus_links_to_every_resolvable_lecture() -> None:
    """The syllabus program mentions Cours 1..13; only ordinals with a
    matching sibling resolve."""
    body = (
        "Programme du cours.\n"
        "Cours 1 : Introduction.\n"
        "Cours 2 : Modèles OSI et TCP/IP.\n"
        "Cours 3 : DNS et ARP.\n"
        "Cours 4 : Équipements réseau.\n"  # → Semaine 4 doc
        "Cours 5 : Attaques réseau.\n"  # → Semaine 5 doc
        "Cours 6 : Coupe-feu.\n"
        "Cours 7 : (pas encore dans le vault).\n"
    )
    refs = extract_course_references(
        body, self_doc_id="syllabus", self_title="CR350 - Plan de cours", index=_index(_CR350)
    )
    by_target = {r.target_doc_id: r.surface_text for r in refs}
    # Cours 1/2/3/6 + Cours 4→Semaine4 + Cours 5→Semaine5 = 6 links.
    assert by_target == {
        "c1": "Cours 1",
        "c2": "Cours 2",
        "c3": "Cours 3",
        "s4": "Cours 4",
        "s5": "Cours 5",
        "c6": "Cours 6",
    }
    # Cours 7 has no sibling → not resolved.
    assert all(r.target_doc_id != "c7" for r in refs)


def test_ordinal_matches_across_cours_semaine_label() -> None:
    """A 'Cours 4' mention resolves to the doc titled 'Semaine 4'."""
    refs = extract_course_references(
        "Comme abordé au Cours 4, les commutateurs...",
        self_doc_id="x",
        self_title="CR350 - Cours 3",
        index=_index(_CR350),
    )
    assert len(refs) == 1
    assert refs[0].target_doc_id == "s4"
    assert refs[0].via == "course_ref"


def test_series_gating_blocks_cross_course_links() -> None:
    """A 'Cours 4' mention does not link to a different course's week 4."""
    other = {"o4": "INF8480 - Cours 4"}
    refs = extract_course_references(
        "voir Cours 4",
        self_doc_id="x",
        self_title="CR350 - Cours 2",
        index=_index(other),
    )
    assert refs == []


def test_no_series_key_yields_nothing() -> None:
    """A document whose title has no course code produces no links."""
    refs = extract_course_references(
        "voir Cours 4",
        self_doc_id="x",
        self_title="Some generic lecture notes",
        index=_index(_CR350),
    )
    assert refs == []


def test_self_reference_skipped() -> None:
    """A doc mentioning its own ordinal doesn't link to itself."""
    # Source is the Cours 2 doc; index here happens to include c2.
    refs = extract_course_references(
        "Dans ce Cours 2 nous voyons...",
        self_doc_id="c2",
        self_title="CR350 - Cours 2",
        index=_index(_CR350),
    )
    assert all(r.target_doc_id != "c2" for r in refs)


def test_parse_artifact_three_digit_run_not_matched() -> None:
    """A glitchy 'Cours 121' must not match 'Cours 12' or 'Cours 1'."""
    refs = extract_course_references(
        "Cours 121 est un artefact.",
        self_doc_id="x",
        self_title="CR350 - Plan de cours",
        index=_index(_CR350),
    )
    assert refs == []


def test_duplicate_mention_emitted_once() -> None:
    body = "Cours 2 introduit le modèle. Plus tard, Cours 2 revient."
    refs = extract_course_references(
        body, self_doc_id="x", self_title="CR350 - Plan de cours", index=_index(_CR350)
    )
    assert len([r for r in refs if r.target_doc_id == "c2"]) == 1


def test_semaine_label_in_mention_also_resolves() -> None:
    """Mentions can use 'Semaine N' too, matched by ordinal."""
    refs = extract_course_references(
        "Voir la Semaine 5 pour la Cyber Kill Chain.",
        self_doc_id="x",
        self_title="CR350 - Cours 6",
        index=_index(_CR350),
    )
    assert len(refs) == 1
    assert refs[0].target_doc_id == "s5"
    assert refs[0].surface_text == "Semaine 5"


def test_confidence_above_high_threshold() -> None:
    """Course refs must clear the wikilink high-confidence bar (0.85)."""
    refs = extract_course_references(
        "voir Cours 1", self_doc_id="x", self_title="CR350 - Cours 2", index=_index(_CR350)
    )
    assert refs[0].confidence >= 0.85
