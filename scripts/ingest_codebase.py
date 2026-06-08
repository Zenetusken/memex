"""Ingest a CODEBASE into the Memex vault as documents (the codebase-corpus arc, Phase 1).

Walks a repo dir, filters to SOURCE-CODE files (`core.source_types.CODE_SUFFIXES`), skips
build/VCS dirs, and per file runs the standard pipeline: ingest -> parse (the verbatim CODE
passthrough — the canonical `.md` is the raw source, NOT a Docling-mangled pipe-table) ->
index -> `retitle_document` to the **repo-relative path** (so `lib.rs`/`mod.rs`/`main.rs`
across crates don't collide to one indistinguishable "lib" title).

    uv run python scripts/ingest_codebase.py ~/project/codex/codex-rs            # all code files
    uv run python scripts/ingest_codebase.py <repo> --limit 3                    # first N (verification)
    uv run python scripts/ingest_codebase.py <repo> --dry-run                    # list, no changes
    uv run python scripts/ingest_codebase.py <repo> --suffix .rs                 # restrict suffixes

Composes the existing APIs (ingest_file / parse_document / index_document / retitle_document)
+ holds `pause_vllm_for_gpu()` across the batch (the index embed OOMs co-resident with vLLM on
12 GB — the CLI-ingest precedent). Symbol-aware CHUNKING is a later phase; this is ingest+index.

NB doc_id = `assign_doc_id` = `sha256(bytes)[:8] + "-" + slug(basename stem)`, so two files collide
to ONE doc ONLY when byte-identical AND sharing a basename (e.g. two trivial identical `mod.rs`) —
distinct stems with identical bytes do NOT collide. On a real collision the 2nd ingest merges onto
the 1st doc_id and `retitle_document` OVERWRITES the title → one of the two real paths silently
drops from the corpus. Tolerable for `--limit` verification; for a full ingest, pre-check that
`#distinct (sha256, stem) == #files` first (the Phase-4 entry condition) and decide deliberately.

Exit codes: 0 ok (>=1 ingested) · 1 all failed · 2 usage error.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from memex.cli.bootstrap import bootstrap
from memex.core.source_types import CODE_SUFFIXES
from memex.index.pipeline import index_document, retitle_document
from memex.ingest.pipeline import IngestRequest, ingest_file
from memex.parse.pipeline import parse_document, pause_vllm_for_gpu

# Build artefacts / VCS / vendored trees never belong in a source-code corpus.
_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "target",  # Rust / Maven build
        "node_modules",
        ".venv",
        "venv",
        "dist",
        "build",
        "vendor",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".next",
        ".tox",
    }
)


def _iter_code_files(root: Path, suffixes: frozenset[str]) -> list[Path]:
    """Source-code files under `root`, sorted, skipping build/VCS dirs + hidden files."""
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(root).parts
        if any(part in _SKIP_DIRS or part.startswith(".") for part in rel_parts[:-1]):
            continue
        if p.suffix.lower() in suffixes:
            out.append(p)
    return out


async def _ingest_one(root: Path, f: Path) -> tuple[str, str | None, str]:
    """ingest -> parse -> index -> retitle(repo-relative). Returns (rel, doc_id|None, status)."""
    rel = str(f.relative_to(root))
    res = await ingest_file(IngestRequest(source_path=f))
    if not res.accepted or res.doc_id is None:
        return rel, None, f"REJECTED: {res.rejection_reason}"
    doc_id = res.doc_id
    await parse_document(doc_id)
    idx = await index_document(doc_id)
    await retitle_document(doc_id, rel)
    return rel, doc_id, f"ok · kind={res.detected_kind} · chunks={getattr(idx, 'chunk_count', '?')}"


def _normalize_suffixes(raw: list[str] | None) -> frozenset[str]:
    if not raw:
        return CODE_SUFFIXES
    return frozenset((s if s.startswith(".") else f".{s}").lower() for s in raw)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("repo", type=Path, help="repo root to walk")
    ap.add_argument("--limit", type=int, default=0, help="ingest only the first N files (0 = all)")
    ap.add_argument(
        "--suffix",
        action="append",
        help="restrict to these suffixes (repeatable, e.g. --suffix .rs); default = all CODE_SUFFIXES",
    )
    ap.add_argument("--dry-run", action="store_true", help="list files that WOULD ingest; no changes")
    args = ap.parse_args()

    root = args.repo.expanduser().resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    files = _iter_code_files(root, _normalize_suffixes(args.suffix))
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} code file(s) under {root}{' [DRY RUN]' if args.dry_run else ''}")
    if args.dry_run:
        for f in files:
            print(f"  {f.relative_to(root)}")
        return 0
    if not files:
        print("no code files matched", file=sys.stderr)
        return 2

    bootstrap()

    async def _run() -> tuple[int, int]:
        ok = fail = 0
        # Hold the GPU across the whole batch (the index embed OOMs co-resident with vLLM on
        # 12 GB; the parse passthrough itself needs no GPU). Restarts vLLM once at the end.
        async with pause_vllm_for_gpu():
            for f in files:
                rel, doc_id, status = await _ingest_one(root, f)
                print(f"  {'✓' if doc_id else '✗'} {rel:52} {doc_id or '':22} {status}")
                if doc_id:
                    ok += 1
                else:
                    fail += 1
        print(f"\n  {ok} ingested · {fail} failed/skipped")
        return ok, fail

    ok, fail = asyncio.run(_run())
    return 1 if (fail and not ok) else 0


if __name__ == "__main__":
    sys.exit(main())
