"""Download and cache the models the configured Memex system needs — the one online
bootstrap step for a local-first / air-gapped install.

Thin CLI shim over `memex.models.download` (the shared logic the `memex download-models`
command + the webui `/resources` model-cache panel also use). Resolves model ids from
`MemexSettings`, fetches each repo into the HF cache via `huggingface_hub.snapshot_download`
(faster-whisper's own downloader for the ASR CTranslate2 model — which itself lands in the
SAME HF cache), verifies presence, and reports on-disk usage. After this runs once ONLINE,
the runtime loads every model OFFLINE — the air-gap test (VISION.md Principle 1) passes.

`snapshot_download` fetches a whole repo to the cache WITHOUT loading/instantiating it
(no torch, no GPU, no weights in RAM) and verifies file etags/hashes — so this is a
fast, CUDA-free bootstrap, not a model load. Prefer `memex download-models` (the discoverable
command); this script path is kept for back-compat + scripting.

Usage:
    uv run python scripts/download-models.py             # fetch the configured set
    uv run python scripts/download-models.py --check      # verify the cache only (no network)
    uv run python scripts/download-models.py --all        # + the gated capability models (full kit)
    uv run python scripts/download-models.py --only embedder reranker
    uv run python scripts/download-models.py --json        # machine-readable report

Exit codes:
    0  every targeted model is present (--check) or fetched OK.
    1  one or more models are MISSING (--check) or FAILED to download.
    2  setup error (e.g. ASR is configured but the `audio` extra isn't installed).
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify the cache only — no network, no download")
    ap.add_argument(
        "--all", action="store_true", help="include the gated capability models (the full offline kit)"
    )
    ap.add_argument(
        "--only", nargs="+", metavar="NAME", help="restrict to these model names (e.g. embedder reranker)"
    )
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = ap.parse_args()

    from memex.core.config import MemexSettings, set_settings
    from memex.models.download import format_report, resolve_model_targets, run_download

    settings = MemexSettings()  # type: ignore[call-arg]  # no bootstrap() — downloads need no CUDA
    set_settings(settings)

    rows, code = run_download(settings, check=args.check, include_all=args.all, only=args.only)
    if args.only and not rows:
        known = [t.name for t in resolve_model_targets(settings, include_all=True)]
        print(f"--only matched no models; known: {known}", file=sys.stderr)
        return 2

    if args.json:
        total = sum(r.get("size", 0) for r in rows)
        print(json.dumps({"check": args.check, "rows": rows, "total_bytes": total}, indent=1, default=str))
    else:
        print(format_report(rows, check=args.check))
    return code


if __name__ == "__main__":
    sys.exit(main())
