"""Integration: Phase-2 symbol-aware code chunking (codebase-corpus arc).

A Rust code doc (Phase-1 verbatim ingest → `source.rs`) gets symbol-boundary chunks via the
TRANSIENT `build_chunking_body` (chart-reattach → GFM-linearize → Rust symbol-heading injection)
feeding the UNCHANGED `chunk_document`. The canonical `.md` is never modified. Prose docs and
non-Rust code fall through the prose path BYTE-IDENTICALLY (no chunk_id churn on the live vault).
Plus the `chunking_recipe_version` recorded by `index_document` + its force-rechunk-on-mismatch.

`build_chunking_body` + `chunk_document` are exercised directly (pure, GPU-free); the recipe
behaviour reuses the faked-store harness from `test_partial_reindex`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from memex.core.config import MemexSettings, set_settings
from memex.core.source_types import code_language_for_doc
from memex.core.table_linearize import linearize_gfm_tables
from memex.core.text import reattach_chart_extractions
from memex.index.chunker import chunk_document
from memex.index.pipeline import build_chunking_body
from memex.ingest.pipeline import ingest_markdown_passthrough
from memex.vault.store import read_document

_RUST = """\
use std::fmt;

/// A widget.
#[derive(Debug, Clone)]
pub struct Widget {
    pub id: u32,
}

impl Widget {
    pub fn new(id: u32) -> Self {
        Widget { id }
    }

    fn label(&self) -> String {
        format!("widget-{}", self.id)
    }
}

pub fn build() -> Widget {
    Widget::new(1)
}
"""


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


async def _seed_code(settings: MemexSettings, body: str, *, stem: str, suffix: str) -> str:
    """Seed a code doc the way Phase-1 lands one: verbatim `.md` body + a suffix-preserved
    `source.<ext>` in the asset dir (what `code_language_for_doc` keys off)."""
    ref = await ingest_markdown_passthrough(body, source_stem=stem)
    asset = settings.vault_path / "documents" / ref.doc_id
    asset.mkdir(parents=True, exist_ok=True)
    (asset / f"source{suffix}").write_text(body, encoding="utf-8")
    return ref.doc_id


@pytest.mark.asyncio
async def test_rust_doc_chunks_on_symbol_boundaries(settings: MemexSettings) -> None:
    doc_id = await _seed_code(settings, _RUST, stem="widget", suffix=".rs")
    doc = await read_document(settings.vault_path, doc_id)
    lang = code_language_for_doc(doc.ref.asset_dir)
    assert lang == "rust"

    body, recipe = build_chunking_body(doc, [], code_language=lang)
    assert recipe == "code-rust-v1"
    chunks = chunk_document(doc.model_copy(update={"body": body}))

    # The deepest heading_path entry = the symbol = the embed title + citation label.
    deepest = {c.heading_path[-1] for c in chunks if c.heading_path}
    assert "struct Widget" in deepest
    assert "impl Widget" in deepest
    assert "Widget::new" in deepest  # method, fully qualified
    assert "Widget::label" in deepest
    assert "fn build" in deepest

    # The method chunk is nested UNDER its impl (heading_path = [impl Widget, Widget::new]).
    new_chunk = next(c for c in chunks if c.heading_path and c.heading_path[-1] == "Widget::new")
    assert new_chunk.heading_path == ["impl Widget", "Widget::new"]
    # Its text carries the actual method body (content preserved, just whitespace-normalized).
    assert "Widget { id }" in new_chunk.text


@pytest.mark.asyncio
async def test_prose_doc_is_byte_identical_to_pre_phase2_transform(settings: MemexSettings) -> None:
    """A prose (markdown-passthrough) doc has no `source.*` → gate None → the chunker-input body
    is EXACTLY the old chart-reattach→linearize transform → chunk_ids unchanged. This is the
    load-bearing regression net: Phase 2 must not perturb the existing vault."""
    prose = "# Title\n\nFirst paragraph.\n\n## Section\n\nSecond paragraph with detail.\n"
    ref = await ingest_markdown_passthrough(prose, source_stem="prose_doc")
    doc = await read_document(settings.vault_path, ref.doc_id)

    lang = code_language_for_doc(doc.ref.asset_dir)
    assert lang is None  # no source.* for a passthrough doc

    body, recipe = build_chunking_body(doc, [], code_language=lang)
    assert recipe == "v0"
    # Identical to the pre-Phase-2 inline transform (index/pipeline old lines 271-272).
    assert body == linearize_gfm_tables(reattach_chart_extractions(doc.body, []))
    # No injected ATX headings appeared (the doc's own `#`/`##` survive, nothing synthetic added).
    assert "## fn " not in body and "### " not in body


@pytest.mark.asyncio
async def test_oversized_symbol_force_splits_keeping_heading_path(settings: MemexSettings) -> None:
    """A Rust fn larger than the chunker's MAX_CHUNK_CHARS force-splits into multiple chunks —
    every piece must carry the SAME `heading_path = [impl Big, Big::huge]` (computed per-chunk at
    its char_start). models.rs has only small methods, so this pins the oversized path the full
    codex-rs ingest (Phase 4) will certainly hit. Correct by construction (the existing chunker
    force-split + per-offset `_heading_path_at`), pinned here."""
    huge = "\n".join(f"    let v{i} = {i} + {i} + {i};" for i in range(220))  # >> MAX_CHUNK_CHARS
    rust = f"impl Big {{\n    fn huge(&self) {{\n{huge}\n    }}\n}}\n"
    doc_id = await _seed_code(settings, rust, stem="big", suffix=".rs")
    doc = await read_document(settings.vault_path, doc_id)
    lang = code_language_for_doc(doc.ref.asset_dir)
    body, _ = build_chunking_body(doc, [], code_language=lang)
    chunks = chunk_document(doc.model_copy(update={"body": body}))

    huge_chunks = [c for c in chunks if c.heading_path and c.heading_path[-1] == "Big::huge"]
    assert len(huge_chunks) >= 2  # the big method force-split into multiple chunks
    # Every force-split piece keeps the full nested heading_path (the embed title stays the symbol).
    assert all(c.heading_path == ["impl Big", "Big::huge"] for c in huge_chunks)


@pytest.mark.asyncio
async def test_non_rust_code_falls_through_to_prose_path_with_v0_tag(settings: MemexSettings) -> None:
    """A `.py` code doc IS detected as code (language 'python') but injection is Rust-only, so it
    falls through the prose path and is tagged "v0" — NOT "code-rust-v1" (the advisor's
    transform-keyed recipe contract: a non-Rust doc must not claim a code-chunking recipe, else a
    future Python splitter couldn't force-rechunk it)."""
    py = "# a module comment\ndef f():\n    return 1\n\n\ndef g():\n    return 2\n"
    doc_id = await _seed_code(settings, py, stem="script", suffix=".py")
    doc = await read_document(settings.vault_path, doc_id)

    lang = code_language_for_doc(doc.ref.asset_dir)
    assert lang == "python"  # detected as code …

    body, recipe = build_chunking_body(doc, [], code_language=lang)
    assert recipe == "v0"  # … but NOT chunked by the Rust grammar
    assert body == linearize_gfm_tables(reattach_chart_extractions(doc.body, []))  # untouched
