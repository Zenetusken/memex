"""Unit tests for source-code file ingest detection (codebase-corpus arc, Phase 1).

A source file (`.rs`/`.py`/…) is detected as kind `"code"` by suffix + the UTF-8 text guard, so it
stays ABSENT from `_EXTENSION_FOR_KIND` (the original suffix is preserved → the parse routing keys on
`.rs`) and routes to the VERBATIM passthrough. Non-code text stays `"text"`; markdown stays
`"markdown"`; a binary file carrying a code suffix is NOT `"code"` (the `_looks_like_text` guard).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memex.core.source_types import CODE_SUFFIXES
from memex.ingest.validation import ValidationResult, validate_file


def _write(tmp_path: Path, name: str, data: bytes | str) -> Path:
    p = tmp_path / name
    if isinstance(data, bytes):
        p.write_bytes(data)
    else:
        p.write_text(data, encoding="utf-8")
    return p


def _validate(p: Path) -> ValidationResult:
    return validate_file(p, max_bytes=100_000, allow_macros=False)


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("lib.rs", 'fn main() {\n    println!("hi");\n}\n'),  # Rust
        ("app.py", "# a comment\ndef f():\n    return 1\n"),  # Python (note the `# ` comment)
        ("index.ts", "export const x: number = 1;\n"),  # TypeScript
        ("main.go", "package main\nfunc main() {}\n"),  # Go
        ("util.c", "#include <stdio.h>\nint main(){return 0;}\n"),  # C
        ("Foo.java", "class Foo {}\n"),  # Java
        ("run.sh", "#!/bin/sh\necho hi\n"),  # shell
    ],
)
def test_code_suffix_detects_as_code(tmp_path: Path, name: str, content: str) -> None:
    p = _write(tmp_path, name, content)
    v = _validate(p)
    assert v.accepted is True
    assert v.kind == "code"
    assert v.mime == "text/plain"


def test_non_code_text_stays_text(tmp_path: Path) -> None:
    """An off-suffix UTF-8 file is still `"text"` — the code branch is suffix-gated, not catch-all."""
    p = _write(tmp_path, "notes.txt", "just plain prose, not code\n")
    v = _validate(p)
    assert v.accepted is True and v.kind == "text"


def test_markdown_unchanged(tmp_path: Path) -> None:
    p = _write(tmp_path, "readme.md", "# Heading\n\nbody text\n")
    v = _validate(p)
    assert v.accepted is True and v.kind == "markdown"


def test_binary_with_code_suffix_is_not_code(tmp_path: Path) -> None:
    """A `.rs` that's actually binary (a NUL byte) fails `_looks_like_text` → NOT `"code"`."""
    p = _write(tmp_path, "weird.rs", b"\x00\x01\x02binary\x00data")
    v = _validate(p)
    assert v.kind != "code"


def test_code_suffixes_include_the_codex_rs_language() -> None:
    assert ".rs" in CODE_SUFFIXES
    assert ".py" in CODE_SUFFIXES and ".ts" in CODE_SUFFIXES
    # All entries are normalized leading-dot lowercase (the suffix-match contract).
    assert all(s.startswith(".") and s == s.lower() for s in CODE_SUFFIXES)
