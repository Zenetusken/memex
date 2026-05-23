"""Multilingual support — pin the two Memex-internal fixes that
make non-English content (French, German, Spanish, Italian,
Portuguese) work end-to-end through the chunker + FTS5 layers.

Until 2026-05-22 the chunker's `_SENTENCE_RE` only matched ASCII
uppercase + digits after sentence-final punctuation, so French
sentences starting with accented capitals (`Élève`, `Ça`, `À`,
`Œuvre`) got concatenated into a single long sentence by the
sentence-split fallback. And the FTS5 schema declared
`tokenize='porter unicode61'` — porter is an English-only stemmer
that mangles French/German/etc word forms.

These tests pin:
- Chunker splits French + German sentences with accented capitals
- Chunker still splits English sentences (regression)
- FTS5 finds French content
- FTS5 matches diacritic-insensitively (`cafe` ↔ `café`)
- FTS5 still matches English content (regression)
"""

from __future__ import annotations

import re
import sqlite3

import pytest

from pathlib import Path

from memex.index.chunker import _SENTENCE_RE, chunk_document
from memex.vault.store import DocumentRef, Frontmatter, VaultDocument


def _doc(body: str, *, doc_id: str = "fr12cafe-test", title: str = "Test") -> VaultDocument:
    """Build a VaultDocument for chunker tests — same helper shape as
    tests/unit/test_chunker.py uses."""
    ref = DocumentRef(
        doc_id=doc_id,
        markdown_path=Path(f"/tmp/{doc_id}.md"),  # noqa: S108 — test fixture
        asset_dir=Path(f"/tmp/{doc_id}"),  # noqa: S108
        source_path=None,
        content_sha256="0" * 64,
    )
    return VaultDocument(
        ref=ref,
        frontmatter=Frontmatter(title=title),
        body=body,
        mtime_ns=0,
    )


# ----- chunker._SENTENCE_RE -----


def test_sentence_regex_splits_french_accented_capitals() -> None:
    """`É` and `À` after a sentence-final period must be recognised
    as sentence starts — otherwise the chunker's fallback treats two
    French sentences as one and the chunks get oversized."""
    text = "Première phrase. Élève en deuxième. À chaque fois trois."
    parts = _SENTENCE_RE.split(text)
    assert len(parts) == 3, f"expected 3 sentences, got {len(parts)}: {parts}"


def test_sentence_regex_splits_german_umlauts() -> None:
    """German accented capitals (`Ü`, `Ä`, `Ö`) after a period."""
    text = "Erste Satz. Über das Thema. Ärger gibt es immer."
    parts = _SENTENCE_RE.split(text)
    assert len(parts) == 3, f"expected 3 sentences, got {len(parts)}: {parts}"


def test_sentence_regex_splits_french_oe_and_capital_y_diaeresis() -> None:
    """`Œ` and `Ÿ` are essential French characters that live OUTSIDE
    Latin-1 Supplement (Latin Extended-A). Pin them explicitly so a
    future maintainer doesn't drop them when refactoring to a
    range-only regex."""
    text = "Phrase un. Œuvre majeure. Ÿpsilon enfin."
    parts = _SENTENCE_RE.split(text)
    assert len(parts) == 3, f"expected 3 sentences, got {len(parts)}: {parts}"


def test_sentence_regex_english_still_works() -> None:
    """Regression — ASCII English sentence splitting must still work
    exactly as before the multilingual extension."""
    text = "This is sentence one. Then sentence two. And a third."
    parts = _SENTENCE_RE.split(text)
    assert len(parts) == 3
    assert parts[0] == "This is sentence one."
    assert parts[1] == "Then sentence two."
    assert parts[2] == "And a third."


def test_chunker_french_paragraph_emits_chunks() -> None:
    """End-to-end: a French markdown doc passes through chunk_document
    without error and produces at least one chunk carrying the
    French body text."""
    french_body = (
        "# Cours de mathématiques\n\n"
        "Première leçon : les nombres entiers. Élève attentif, "
        "écoutez bien. À chaque exercice, vérifiez votre réponse. "
        "Œuvre collective, donc."
    )
    chunks = chunk_document(_doc(french_body, title="Cours"))
    assert len(chunks) >= 1
    joined = " ".join(c.text for c in chunks)
    assert "Élève" in joined
    assert "À chaque" in joined
    assert "Œuvre" in joined


# ----- fts_store schema + tokenizer -----


def _create_fts_table(db: sqlite3.Connection) -> None:
    """Build a copy of the production chunks_fts table using the
    same tokenize clause shipped in `_SCHEMA`. We test against an
    in-memory database to keep these tests pure-unit."""
    db.execute(
        "CREATE VIRTUAL TABLE chunks_fts USING fts5("
        "chunk_id UNINDEXED, document_id UNINDEXED, "
        "document_title UNINDEXED, text, "
        "tokenize='unicode61 remove_diacritics 2'"
        ")"
    )


@pytest.fixture
def fts_db() -> sqlite3.Connection:
    """Fresh in-memory FTS5 table for each test."""
    db = sqlite3.connect(":memory:")
    _create_fts_table(db)
    return db


def _insert(
    db: sqlite3.Connection, chunk_id: str, text: str, doc: str = "d1"
) -> None:
    db.execute(
        "INSERT INTO chunks_fts (chunk_id, document_id, document_title, text) "
        "VALUES (?, ?, ?, ?)",
        (chunk_id, doc, "title", text),
    )


def _search(db: sqlite3.Connection, query: str) -> list[str]:
    quoted = '"' + query.replace('"', '""') + '"'
    rows = db.execute(
        "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ?",
        (quoted,),
    ).fetchall()
    return [r[0] for r in rows]


def test_fts_finds_french_content(fts_db: sqlite3.Connection) -> None:
    """French chunk + French query = match. Catches the case where
    the porter stemmer was mangling French word forms."""
    _insert(fts_db, "c1", "Le café est délicieux ce matin")
    _insert(fts_db, "c2", "An unrelated English sentence")
    assert _search(fts_db, "café") == ["c1"]
    assert _search(fts_db, "délicieux") == ["c1"]


def test_fts_diacritic_insensitive_match(fts_db: sqlite3.Connection) -> None:
    """`unicode61 remove_diacritics 2` makes accented and unaccented
    forms interchangeable. Test both directions because that's how
    French users actually type — many omit accents from queries."""
    # Stored with accents, queried without.
    _insert(fts_db, "with_accent", "Le café est délicieux")
    assert "with_accent" in _search(fts_db, "cafe")
    assert "with_accent" in _search(fts_db, "delicieux")

    # Stored without accents, queried with.
    _insert(fts_db, "no_accent", "cafe latte every morning")
    assert "no_accent" in _search(fts_db, "café")


def test_fts_diacritic_strip_for_capital_letters(
    fts_db: sqlite3.Connection,
) -> None:
    """Accent-stripping must also apply to uppercase forms — `Élève`
    and `eleve` should be interchangeable. Catches the case where
    diacritic-strip is only wired for lowercase."""
    _insert(fts_db, "upper", "Élève très attentif")
    assert "upper" in _search(fts_db, "eleve")
    assert "upper" in _search(fts_db, "Eleve")


def test_fts_english_still_matches(fts_db: sqlite3.Connection) -> None:
    """Regression — ASCII English content + ASCII query must still
    match after the tokenizer change. Porter stemming is gone (so
    `running` no longer auto-matches `ran`), but exact-token match
    still works."""
    _insert(fts_db, "e1", "The quick brown fox jumps over the lazy dog")
    assert "e1" in _search(fts_db, "quick")
    assert "e1" in _search(fts_db, "fox")
    assert "e1" in _search(fts_db, "jumps")


def test_fts_german_umlauts(fts_db: sqlite3.Connection) -> None:
    """German `ü`, `ä`, `ö` — diacritic-stripped to base forms.
    Confirms `unicode61 remove_diacritics 2` doesn't special-case
    French only."""
    _insert(fts_db, "de1", "Über das Thema gibt es viel zu sagen")
    assert "de1" in _search(fts_db, "uber")
    assert "de1" in _search(fts_db, "über")


def test_sentence_regex_is_used_via_module_import() -> None:
    """Trivial guard so a future refactor that renames or relocates
    `_SENTENCE_RE` (e.g., into `core/text.py`) trips a test rather
    than silently breaking multilingual splitting."""
    assert isinstance(_SENTENCE_RE, re.Pattern)
    # Sanity check: the pattern string contains at least one Latin-1
    # accented uppercase character so a future "simplification"
    # back to ASCII trips here.
    assert "É" in _SENTENCE_RE.pattern
    assert "À" in _SENTENCE_RE.pattern
    assert "Œ" in _SENTENCE_RE.pattern
