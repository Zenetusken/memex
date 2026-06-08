"""Integration: source-code files ingest VERBATIM (codebase-corpus arc, Phase 1).

A `.rs` ingests as kind `"code"` and parses via the verbatim passthrough — the canonical `.md` is
the RAW source, NOT the Docling-fallback pipe-table that mangled code before this change (a `.rs`
used to land as `| mod cli | |` rows). `engine == "passthrough"`; `strip_frontmatter=False` keeps a
leading `//!`/`---`-style line intact. Ingest+parse only (no GPU — index/retitle are live-verified).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from memex.core.config import MemexSettings, set_settings
from memex.ingest.pipeline import IngestRequest, ingest_file
from memex.parse.pipeline import parse_document
from memex.vault.store import read_document


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemexSettings]:
    (tmp_path / "documents").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    s = MemexSettings()  # type: ignore[call-arg]
    set_settings(s)
    yield s
    set_settings(None)


_RUST = """//! crate-level docs
use std::io::Write;

#[derive(Debug)]
struct Foo {
    bar: i32,
}

fn main() {
    let x = 1;
    println!("value = {}", x);
}
"""


@pytest.mark.asyncio
async def test_rust_file_ingests_verbatim_not_mangled(
    settings: MemexSettings, tmp_path: Path
) -> None:
    src = tmp_path / "lib.rs"
    src.write_text(_RUST, encoding="utf-8")

    res = await ingest_file(IngestRequest(source_path=src))
    assert res.accepted is True
    assert res.doc_id is not None
    assert res.detected_kind == "code"  # the new suffix-detected kind

    parse = await parse_document(res.doc_id)
    assert parse.engine == "passthrough"  # routed to the verbatim passthrough, NOT Docling

    doc = await read_document(settings.vault_path, res.doc_id)
    body = doc.body
    # VERBATIM: the canonical .md body IS the raw Rust — structure intact.
    assert "fn main() {" in body
    assert "struct Foo {" in body
    assert "#[derive(Debug)]" in body
    # strip_frontmatter=False → the leading `//!` doc-comment + `use` survive (not stripped).
    assert "//! crate-level docs" in body
    # NOT the Docling-fallback mangling, which prefixed each code line as a `| ... |` table row.
    assert not body.startswith("| ")
    assert "\n| " not in body


@pytest.mark.asyncio
async def test_python_hash_comment_does_not_block_code_routing(
    settings: MemexSettings, tmp_path: Path
) -> None:
    """A Python `# comment` line (which looks like a markdown H1) still ingests as code, verbatim —
    the heading-as-chunking concern is Phase 2; here only verbatim storage + the route matter."""
    src = tmp_path / "app.py"
    src.write_text("# module comment\ndef f():\n    return 42\n", encoding="utf-8")
    res = await ingest_file(IngestRequest(source_path=src))
    assert res.accepted and res.detected_kind == "code"
    parse = await parse_document(res.doc_id)  # type: ignore[arg-type]
    assert parse.engine == "passthrough"
    doc = await read_document(settings.vault_path, res.doc_id)  # type: ignore[arg-type]
    assert "# module comment" in doc.body and "def f():" in doc.body


@pytest.mark.asyncio
async def test_code_leading_yaml_block_is_not_frontmatter_stripped(
    settings: MemexSettings, tmp_path: Path
) -> None:
    """`strip_frontmatter=False` is the reason a code file is NOT the markdown passthrough: a source
    file whose first lines look like a YAML `---` block keeps them VERBATIM (the markdown passthrough
    would strip them). This pins the exact case the flag exists for."""
    src = tmp_path / "weird.rs"
    src.write_text("---\nkey: val\n---\nfn x() {}\n", encoding="utf-8")
    res = await ingest_file(IngestRequest(source_path=src))
    assert res.accepted and res.detected_kind == "code"
    await parse_document(res.doc_id)  # type: ignore[arg-type]
    doc = await read_document(settings.vault_path, res.doc_id)  # type: ignore[arg-type]
    assert doc.body.startswith("---")  # the leading block is KEPT, not stripped
    assert "key: val" in doc.body and "fn x() {}" in doc.body
