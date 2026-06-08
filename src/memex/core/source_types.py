"""Source-code file types — the shared `CODE_SUFFIXES` set.

Lives in `core/` because BOTH `ingest/validation.py` (to detect a `code` kind by
suffix) and `parse/pipeline.py` (to route a code file to the verbatim passthrough)
need it, and `ingest/ ↔ parse/` is a forbidden import edge — `core/` is the common
ancestor both may import.

A source-code file is a TEXT document ingested VERBATIM (the canonical `.md` is the
raw source — no Docling/PDF parse, which would mangle aligned code into a markdown
table). This set is the routing signal; it is deliberately SOURCE code, not config
(`.toml`/`.yaml`/`.json` are a separate concern).
"""

from __future__ import annotations

from typing import Final

CODE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        ".rs",  # Rust
        ".py",  # Python
        ".ts",
        ".tsx",
        ".js",
        ".jsx",  # TypeScript / JavaScript
        ".go",  # Go
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hpp",
        ".hh",  # C / C++
        ".java",  # Java
        ".kt",
        ".kts",  # Kotlin
        ".scala",  # Scala
        ".swift",  # Swift
        ".rb",  # Ruby
        ".php",  # PHP
        ".cs",  # C#
        ".sh",
        ".bash",
        ".zsh",  # shell
        ".lua",  # Lua
        ".pl",
        ".pm",  # Perl
        ".r",  # R
        ".m",
        ".mm",  # Objective-C / Objective-C++
        ".sql",  # SQL
    }
)
