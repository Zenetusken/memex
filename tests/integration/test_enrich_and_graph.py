"""Phase 2 regression net — enrich + graph + watcher with heavy deps faked.

Validates: per-chunk entity extraction wired to the LLM client, the
post-extraction dedupe, the graph upsert/link path, and the watcher's
debounce + sha-comparison logic.

RyuGraph is not installed in CI; we substitute an in-memory fake
graph store so the enrich pipeline exercises its full call sequence.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from memex.core.config import MemexSettings, set_settings
from memex.core.manifest import read_manifest
from memex.enrich import entities as entities_mod
from memex.enrich.entities import EntityList, ExtractedEntity
from memex.enrich.pipeline import enrich_document
from memex.index.graph_store import entity_id
from memex.ingest.pipeline import ingest_markdown_passthrough
from memex.ingest.watcher import EditNotice, _confirm_user_edit


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    (tmp_path / "documents").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def settings(tmp_vault: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemexSettings]:
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_vault))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    s = MemexSettings()  # type: ignore[call-arg]
    set_settings(s)
    yield s
    set_settings(None)


class _FakeGraphStore:
    """In-memory stand-in for the real RyuGraph store."""

    def __init__(self) -> None:
        self.documents: dict[str, str] = {}
        self.entities: dict[str, tuple[str, str]] = {}
        self.mentions: list[tuple[str, str, float]] = []
        self.cites: list[tuple[str, str, str, float]] = []
        self.deleted: list[str] = []
        self.closed = False

    @classmethod
    async def open(cls, vault_path: Path) -> _FakeGraphStore:
        return cls()

    async def upsert_document(self, doc_id: str, title: str) -> None:
        self.documents[doc_id] = title

    async def upsert_entity(self, name: str, kind: str) -> str:
        eid = entity_id(name, kind)
        self.entities[eid] = (name, kind)
        return eid

    async def link_mentions(self, doc_id: str, entity_id_: str, confidence: float) -> None:
        self.mentions.append((doc_id, entity_id_, confidence))

    async def link_cites(
        self,
        from_doc_id: str,
        to_doc_id: str,
        surface_text: str,
        confidence: float,
    ) -> None:
        self.cites.append((from_doc_id, to_doc_id, surface_text, confidence))

    async def delete_document(self, doc_id: str) -> None:
        self.deleted.append(doc_id)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_graph(monkeypatch: pytest.MonkeyPatch) -> _FakeGraphStore:
    instance = _FakeGraphStore()

    async def _open(vault_path: Path) -> _FakeGraphStore:
        return instance

    monkeypatch.setattr("memex.enrich.pipeline.GraphStore.open", staticmethod(_open))
    return instance


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace `complete_structured` with fixed entity + citation lists.

    The enrich pipeline now makes two LLM calls per chunk (entities +
    citations), so the fake routes by schema. Tests that only care
    about entities still see the original entity output.
    """
    from memex.enrich.citations import CitationList

    call_log: list[type] = []

    async def _fake(*, prompt: str, schema: type, **_kw: Any) -> tuple[Any, int]:
        call_log.append(schema)
        if schema is EntityList:
            return (
                EntityList(
                    entities=[
                        ExtractedEntity(
                            name="Reflexivity",
                            kind="concept",
                            confidence="high",
                            span_text="Reflexivity",
                        ),
                        ExtractedEntity(
                            name="Smith",
                            kind="person",
                            confidence="high",
                            span_text="Smith",
                        ),
                    ]
                ),
                12,
            )
        if schema is CitationList:
            # By default no citations — individual tests override via a
            # nested monkeypatch when they want citation behaviour.
            return CitationList(citations=[]), 6
        raise AssertionError(f"unexpected schema {schema}")

    monkeypatch.setattr("memex.enrich.pipeline.complete_structured", _fake)
    monkeypatch.setattr(
        "memex.enrich.pipeline.render_prompt",
        lambda name, **_kw: f"[fake {name} prompt]",
    )
    return {"calls": call_log}


@pytest.mark.asyncio
async def test_enrich_extracts_entities_and_writes_graph(
    settings: MemexSettings,
    fake_graph: _FakeGraphStore,
    fake_llm: dict[str, Any],
) -> None:
    body = (
        "# Reflexivity in research design\n\n"
        "Reflexivity in research design means the researcher's "
        "presence shapes the data they collect.\n\n"
        "Smith argues that reflexivity is constitutive.\n"
    )
    ref = await ingest_markdown_passthrough(body, source_stem="reflexivity")

    result = await enrich_document(ref.doc_id)

    assert len(result.entities) >= 2
    names = {e.name for e in result.entities}
    assert "Reflexivity" in names
    assert "Smith" in names

    # Graph: one document node, two entity nodes, mentions linking them.
    assert ref.doc_id in fake_graph.documents
    assert len(fake_graph.entities) == 2
    assert len(fake_graph.mentions) >= 2
    assert all(doc == ref.doc_id for (doc, _, _) in fake_graph.mentions)

    # Manifest now has the enrich section.
    manifest = await read_manifest(settings.vault_path, ref.doc_id)
    assert manifest is not None
    assert manifest.enrich is not None
    assert manifest.enrich.entity_count == len(result.entities)
    # The enrich pipeline now invokes both `extract_entities` and
    # `extract_citations` per chunk; the manifest records both prompt
    # versions. Entity prompt presence + version is what this test cares
    # about; citations have their own dedicated tests below.
    assert manifest.enrich.prompt_versions.get("extract_entities") == "v2"
    assert "extract_citations" in manifest.enrich.prompt_versions


def test_entity_dedupe_merges_by_lowered_name_and_kind() -> None:
    raw = [
        entities_mod.Entity(
            name="Reflexivity",
            kind="concept",
            confidence=0.95,
            chunk_ids=["c1"],
        ),
        entities_mod.Entity(
            name="reflexivity",
            kind="concept",
            confidence=0.70,
            chunk_ids=["c2"],
        ),
        entities_mod.Entity(
            name="Reflexivity",
            kind="other",
            confidence=0.45,
            chunk_ids=["c3"],
        ),
    ]
    merged = entities_mod.dedupe(raw)
    by_kind = {e.kind: e for e in merged}
    assert set(by_kind) == {"concept", "other"}
    concept = by_kind["concept"]
    assert concept.confidence == 0.95
    assert sorted(concept.chunk_ids) == ["c1", "c2"]


def test_entity_id_is_stable_across_case() -> None:
    a = entity_id("Reflexivity", "concept")
    b = entity_id("reflexivity ", "concept")
    assert a == b


@pytest.mark.asyncio
async def test_watcher_confirm_returns_none_when_hash_matches(
    settings: MemexSettings,
) -> None:
    """Memex's own write produces a hash that matches the manifest;
    the watcher correctly swallows the event."""
    ref = await ingest_markdown_passthrough("# Hello\n\nbody\n", source_stem="hello")
    md = settings.vault_path / "documents" / f"{ref.doc_id}.md"
    notice = await _confirm_user_edit(settings.vault_path, md)
    assert notice is None


@pytest.mark.asyncio
async def test_watcher_confirm_detects_real_edit(
    settings: MemexSettings,
) -> None:
    ref = await ingest_markdown_passthrough("# Hello\n\nbody\n", source_stem="hello")
    md = settings.vault_path / "documents" / f"{ref.doc_id}.md"
    # Simulate a user-side edit by appending to the file directly.
    with md.open("a", encoding="utf-8") as f:
        f.write("\n\nUser-added paragraph.\n")
    notice = await _confirm_user_edit(settings.vault_path, md)
    assert notice is not None
    assert notice.doc_id == ref.doc_id
    assert isinstance(notice, EditNotice)
    assert notice.new_sha256 != notice.prev_sha256


@pytest.mark.asyncio
async def test_eval_runner_scores_a_synthetic_query_set(
    settings: MemexSettings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """End-to-end eval harness smoke: synthetic 2-query set, faked agent."""
    from memex.agents import answering as ans
    from memex.agents.answering import FinalResponse
    from memex.eval.runner import run_eval

    async def _fake_answer(query: str, **_kw: Any) -> FinalResponse:
        if "refuse" in query.lower():
            return FinalResponse(
                answered=False,
                refusal_reason="should refuse",
                correlation_id="cid_refuse",
                tokens_used=0,
                nodes_traversed=1,
                regenerate_attempts=0,
            )
        return FinalResponse(
            answered=True,
            summary="ok",
            claims=[],
            correlation_id="cid_ok",
            tokens_used=10,
            nodes_traversed=4,
            regenerate_attempts=0,
        )

    monkeypatch.setattr(ans, "answer_query", _fake_answer)
    # The runner imports `answer_query` at module load, so patch the
    # alias the runner actually uses too.
    monkeypatch.setattr("memex.eval.runner.answer_query", _fake_answer)

    query_set = tmp_path / "queries.json"
    query_set.write_text(
        '{"queries": ['
        '{"qid": "q1", "question": "what is X?"},'
        '{"qid": "q2", "question": "please refuse this", "should_refuse": true}'
        "]}",
        encoding="utf-8",
    )

    report = await run_eval(query_set)
    assert report.query_count == 2
    assert report.answered_count == 1
    assert report.refused_count == 1
    assert report.refusal_rate_on_counterfactuals == 1.0


def test_scoring_cer_and_wer() -> None:
    from memex.eval.scoring import character_error_rate, word_error_rate

    assert character_error_rate("hello", "hello") == 0.0
    assert character_error_rate("hello", "world") > 0.5
    assert word_error_rate("the quick brown fox", "the quick brown fox") == 0.0
    assert word_error_rate("the quick brown fox", "the slow brown fox") == 0.25


# ----- v1.x: citation resolution + wikilink insertion -----


def test_citation_resolver_matches_author_year() -> None:
    """A citation surface like 'Smith 2024' resolves to a vault doc with
    that author + year, scoring >= the author-year threshold."""
    from datetime import date

    from memex.enrich.citations import (
        CitationCandidate,
        CitationIndex,
        make_signature,
        resolve_candidate,
    )
    from memex.vault.store import Frontmatter

    idx = CitationIndex()
    idx.by_id["d1-smith-reflexivity"] = make_signature(
        "d1-smith-reflexivity",
        Frontmatter(
            title="On Reflexivity in Research",
            authors=["Smith, Jane"],
            date=date(2024, 3, 1),
        ),
    )
    cand = CitationCandidate(surface_text="Smith 2024", confidence="high")
    result = resolve_candidate(cand, chunk_id="c0", index=idx)
    assert result is not None
    assert result.target_doc_id == "d1-smith-reflexivity"
    assert result.via == "author_year"
    assert result.confidence >= 0.85


def test_citation_resolver_matches_title_substring() -> None:
    from memex.enrich.citations import (
        CitationCandidate,
        CitationIndex,
        make_signature,
        resolve_candidate,
    )
    from memex.vault.store import Frontmatter

    idx = CitationIndex()
    idx.by_id["doc-method"] = make_signature(
        "doc-method",
        Frontmatter(title="A New Method for Cohort Selection"),
    )
    cand = CitationCandidate(
        surface_text='see "A New Method for Cohort Selection"',
        confidence="high",
    )
    result = resolve_candidate(cand, chunk_id="c0", index=idx)
    assert result is not None
    assert result.target_doc_id == "doc-method"
    assert result.via == "title"


def test_citation_resolver_returns_none_below_threshold() -> None:
    from datetime import date

    from memex.enrich.citations import (
        CitationCandidate,
        CitationIndex,
        make_signature,
        resolve_candidate,
    )
    from memex.vault.store import Frontmatter

    idx = CitationIndex()
    idx.by_id["d1"] = make_signature(
        "d1",
        Frontmatter(
            title="Far afield topic",
            authors=["Brown"],
            date=date(2020, 1, 1),
        ),
    )
    # Surface that shouldn't match anything in the index.
    cand = CitationCandidate(surface_text="some unrelated reference", confidence="low")
    assert resolve_candidate(cand, chunk_id="c0", index=idx) is None


def test_insert_wikilinks_respects_confidence_threshold() -> None:
    from memex.enrich.citations import (
        ResolvedCitation,
        insert_wikilinks,
    )

    body = "As Smith 2024 argues, reflexivity matters. See also Jones 2019."
    resolved = [
        ResolvedCitation(
            surface_text="Smith 2024",
            target_doc_id="abc-smith-reflex",
            target_title="Smith 2024",
            confidence=0.95,  # high — should rewrite
            chunk_id="c0",
            via="author_year",
        ),
        ResolvedCitation(
            surface_text="Jones 2019",
            target_doc_id="def-jones",
            target_title="Jones 2019",
            confidence=0.70,  # below high-confidence threshold
            chunk_id="c0",
            via="tokens",
        ),
    ]
    new_body, count = insert_wikilinks(body, resolved)
    assert count == 1
    assert "[[abc-smith-reflex]]" in new_body
    # Low-confidence match left untouched.
    assert "Jones 2019" in new_body
    assert "[[def-jones]]" not in new_body


def test_insert_wikilinks_only_replaces_first_occurrence() -> None:
    """One wikilink per (surface, target) per document — re-running
    enrich shouldn't cascade through every mention."""
    from memex.enrich.citations import (
        ResolvedCitation,
        insert_wikilinks,
    )

    body = "Smith 2024 argues X. Later, Smith 2024 says Y."
    resolved = [
        ResolvedCitation(
            surface_text="Smith 2024",
            target_doc_id="abc",
            target_title="Smith 2024",
            confidence=0.95,
            chunk_id="c0",
            via="author_year",
        )
    ]
    new_body, count = insert_wikilinks(body, resolved)
    assert count == 1
    # The first occurrence becomes a link; the second is left as text.
    assert new_body.count("[[abc]]") == 1
    assert new_body.count("Smith 2024") == 1


@pytest.mark.asyncio
async def test_enrich_resolves_citations_against_vault_docs(
    settings: MemexSettings,
    fake_graph: _FakeGraphStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the vault contains a paper by Smith 2024 and a current doc
    cites 'Smith 2024', the citation should resolve into a CITES edge
    and an inline [[doc_id]] wikilink."""
    from datetime import date

    from memex.enrich.citations import CitationCandidate, CitationList
    from memex.enrich.entities import EntityList
    from memex.enrich.pipeline import enrich_document
    from memex.ingest.pipeline import ingest_markdown_passthrough
    from memex.vault.store import Frontmatter, read_document, write_document

    # Seed the vault with the target doc (Smith 2024).
    target = await ingest_markdown_passthrough(
        "# Reflexivity in research\n\nSmith's argument.\n",
        source_stem="smith_reflexivity",
    )
    # Set its frontmatter so the resolver's author-year keys match.
    target_doc = await read_document(settings.vault_path, target.doc_id)
    target_doc.frontmatter = Frontmatter(
        title="Reflexivity in Research",
        authors=["Smith, Jane"],
        date=date(2024, 3, 1),
    )
    await write_document(settings.vault_path, target_doc)

    # The current doc references it.
    current = await ingest_markdown_passthrough(
        "# Citing paper\n\nAs Smith 2024 argues, this is constitutive.\n",
        source_stem="citing_paper",
    )

    # Override the fake LLM so the citation extractor returns "Smith 2024".
    async def _fake(*, prompt: str, schema: type, **_kw: Any) -> tuple[Any, int]:
        if schema is EntityList:
            return EntityList(entities=[]), 6
        if schema is CitationList:
            return (
                CitationList(
                    citations=[CitationCandidate(surface_text="Smith 2024", confidence="high")]
                ),
                6,
            )
        raise AssertionError(f"unexpected schema {schema}")

    monkeypatch.setattr("memex.enrich.pipeline.complete_structured", _fake)
    monkeypatch.setattr(
        "memex.enrich.pipeline.render_prompt",
        lambda name, **_kw: f"[fake {name}]",
    )

    result = await enrich_document(current.doc_id)

    # One citation resolved.
    assert len(result.citations) == 1
    assert result.citations[0].target_doc_id == target.doc_id
    assert result.citations[0].via == "author_year"

    # Graph has the CITES edge.
    assert any(
        from_id == current.doc_id and to_id == target.doc_id
        for (from_id, to_id, _surface, _conf) in fake_graph.cites
    )

    # Wikilink inserted into the markdown.
    assert result.wikilinks_inserted == 1
    updated = await read_document(settings.vault_path, current.doc_id)
    assert f"[[{target.doc_id}]]" in updated.body
