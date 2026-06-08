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

from pathlib import Path
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

# Human-readable language name for the webui code-view pane label ("source · rust").
# Covers every CODE_SUFFIXES entry; `language_for_suffix` falls back to the bare
# extension for anything unmapped, so a missing entry degrades gracefully (never
# raises). Keep this in lockstep with CODE_SUFFIXES when adding a suffix.
LANGUAGE_FOR_SUFFIX: Final[dict[str, str]] = {
    ".rs": "rust",
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".c": "c",
    ".cc": "c++",
    ".cpp": "c++",
    ".cxx": "c++",
    ".h": "c",
    ".hpp": "c++",
    ".hh": "c++",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".swift": "swift",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "c#",
    ".sh": "shell",
    ".bash": "bash",
    ".zsh": "zsh",
    ".lua": "lua",
    ".pl": "perl",
    ".pm": "perl",
    ".r": "r",
    ".m": "objective-c",
    ".mm": "objective-c++",
    ".sql": "sql",
}


def language_for_suffix(suffix: str) -> str:
    """Map a file suffix (e.g. ``.rs``) to a display language name (``rust``).

    Case-insensitive. Falls back to the bare extension (``suffix.lstrip(".")``)
    for an unmapped suffix, so a code suffix without an explicit entry still
    reads sensibly (e.g. a hypothetical ``.zig`` → ``zig``). Pure — no I/O.
    """
    return LANGUAGE_FOR_SUFFIX.get(suffix.lower(), suffix.lower().lstrip("."))


def code_language_for_doc(asset_dir: Path) -> str | None:
    """Detect a code document by its ingested `source.<ext>` file and return its language name.

    The shared code-detection gate for the index + enrich pipelines (Phase 2 symbol-aware
    chunking). Globs ``asset_dir/source.*`` (the suffix-preserved copy Phase 1's `_copy_source`
    writes for a `code` kind — markdown-passthrough docs have NO `source.*`, PDFs carry `.pdf`)
    and returns `language_for_suffix(suffix)` when the suffix ∈ `CODE_SUFFIXES`, else `None`.
    Lives in `core/` so BOTH `index/` and `enrich/` can import it (webui's `_find_source` /
    `_is_code_source` are private + in `webui/`, which `index/` cannot import). Touches the
    filesystem (a single glob) but no models — cheap to call per index/enrich run.
    """
    if not asset_dir.is_dir():
        return None
    for candidate in sorted(asset_dir.glob("source.*")):
        if candidate.suffix.lower() in CODE_SUFFIXES:
            return language_for_suffix(candidate.suffix)
    return None
