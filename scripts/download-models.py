"""Download and cache the models the configured Memex system needs — the one online
bootstrap step for a local-first / air-gapped install.

Resolves model ids from `MemexSettings`, fetches each repo into the HF cache via
`huggingface_hub.snapshot_download` (faster-whisper's own downloader for the ASR
CTranslate2 model — which itself lands in the SAME HF cache), verifies presence, and
reports on-disk usage. After this runs once ONLINE, the runtime loads every model
OFFLINE — the air-gap test (VISION.md Principle 1) passes.

`snapshot_download` fetches a whole repo to the cache WITHOUT loading/instantiating it
(no torch, no GPU, no weights in RAM) and verifies file etags/hashes — so this is a
fast, CUDA-free bootstrap, not a model load.

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
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# `kind` drives the fetch path:
#   "hf"    → huggingface_hub.snapshot_download(repo_id)
#   "asr"   → faster_whisper.utils.download_model(repo_id)  (maps a size name → CT2 repo; same HF cache)
#   "otter" → snapshot_download(repo_id) PLUS the transitive `config.token_encoder` repo (mmBERT base)


@dataclass(frozen=True)
class ModelTarget:
    """One model to fetch/verify: a display `name`, its HF `repo_id`, the fetch `kind`,
    and a human `reason` it's in the set (which config flag pulled it in)."""

    name: str
    repo_id: str
    kind: str  # "hf" | "asr" | "otter"
    reason: str


def resolve_model_targets(settings: Any, *, include_all: bool) -> list[ModelTarget]:
    """Resolve the model set the CONFIGURED system needs, from `MemexSettings`.

    CORE (always — the `/ask` spine + the orchestrator vLLM serves from the HF cache, so it
    must be cached for offline serving): orchestrator, embedder, reranker.
    CAPABILITY (gated by config, or forced by `--all` — the full offline kit): VLM, chart-OCR,
    summarizer (if set), OTTER NER (if the backend is otter), ASR (if an id is set — `--all`
    can't invent one when it's None). `reasoner` is a reserved hook that never auto-serves → skip.
    Deduped by repo_id (preserve first-seen order)."""
    m = settings.models
    targets: list[ModelTarget] = [
        ModelTarget("orchestrator", m.orchestrator, "hf", "core: the grounded /ask orchestrator (vLLM)"),
        ModelTarget("embedder", m.embedder, "hf", "core: dense retrieval embeddings"),
        ModelTarget("reranker", m.reranker, "hf", f"core: reranker ({settings.models.reranker_backend})"),
    ]

    if include_all or not settings.parse.disable_vlm:
        why = "parse: VLM page transcription" + (" (--all)" if settings.parse.disable_vlm else "")
        targets.append(ModelTarget("vlm", m.vlm, "hf", why))
    if include_all or not settings.parse.disable_chart_ocr:
        why = "parse: chart-OCR" + (" (--all)" if settings.parse.disable_chart_ocr else "")
        targets.append(ModelTarget("chart-ocr", m.chart_ocr, "hf", why))
    if m.summarizer:
        targets.append(ModelTarget("summarizer", m.summarizer, "hf", "models.summarizer is set"))
    if include_all or settings.agents.enrich_ner_backend == "otter":
        why = "enrich: OTTER NER" + (" (--all)" if settings.agents.enrich_ner_backend != "otter" else "")
        targets.append(ModelTarget("otter", settings.agents.enrich_ner_model, "otter", why))
    if m.asr:
        targets.append(ModelTarget("asr", m.asr, "asr", "models.asr is set (faster-whisper)"))

    # Dedup by repo_id, first-seen order (e.g. a config where two roles share a repo).
    seen: set[str] = set()
    deduped: list[ModelTarget] = []
    for t in targets:
        if t.repo_id and t.repo_id not in seen:
            seen.add(t.repo_id)
            deduped.append(t)
    return deduped


def _dir_size_bytes(path: str | os.PathLike[str]) -> int:
    """Total bytes under `path`, following the HF cache's blob symlinks (`stat()` follows)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size  # stat() follows the snapshot→blob symlink
            except OSError:
                pass
    return total


def _human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def _token_encoder_repo(snapshot_dir: str) -> str | None:
    """OTTER loads a SECOND repo via `config.token_encoder` (ner_otter.py:228 — the mmBERT base).
    Read it out of the downloaded `config.json` so it can be cached too. Best-effort + defensive."""
    cfg = Path(snapshot_dir) / "config.json"
    if not cfg.is_file():
        return None
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    te = data.get("token_encoder")
    # Only treat it as a fetchable repo when it's an `org/name` id (not a local path / nested dict).
    if isinstance(te, str) and "/" in te and not te.startswith((".", "/")):
        return te
    return None


def _snapshot(repo_id: str, *, check: bool) -> str:
    """Fetch (or, with check=True, locate-without-network) a repo; return its local snapshot dir."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, local_files_only=check)


def _asr_download(repo_id: str, *, check: bool) -> str:
    """faster-whisper resolves a size name → its CT2 repo and snapshot_downloads it (same HF cache)."""
    from faster_whisper.utils import download_model

    return download_model(repo_id, local_files_only=check)


def process_target(t: ModelTarget, *, check: bool) -> dict[str, Any]:
    """Fetch or verify one target. Returns a result row: ok/present, size, detail, extra.
    Never raises for an expected miss/HF error (the row records the failure); a missing
    optional dependency surfaces via the `setup_error` flag so main can exit 2."""
    row: dict[str, Any] = {"name": t.name, "repo_id": t.repo_id, "reason": t.reason}
    try:
        if t.kind == "asr":
            path = _asr_download(t.repo_id, check=check)
        else:  # "hf" or "otter"
            path = _snapshot(t.repo_id, check=check)
    except ImportError as e:
        # ASR configured but `faster_whisper` (the `audio` extra) isn't installed → setup error.
        row.update(ok=False, present=False, size=0, detail=f"dependency missing: {e}", setup_error=True)
        return row
    except Exception as e:  # an expected miss (check) or HF/network error: record, don't abort the batch
        verb = "not cached" if check else "download failed"
        row.update(ok=False, present=False, size=0, detail=f"{verb}: {type(e).__name__}: {str(e)[:160]}")
        return row

    size = _dir_size_bytes(path)
    row.update(ok=True, present=True, size=size, detail=str(path))

    if t.kind == "otter":  # also handle the transitive token_encoder repo (mmBERT base)
        te = _token_encoder_repo(path)
        if te:
            try:
                te_path = _snapshot(te, check=check)
                row["token_encoder"] = {"repo_id": te, "present": True, "size": _dir_size_bytes(te_path)}
                row["size"] += row["token_encoder"]["size"]
            except Exception as e:  # record the transitive miss; OTTER is optional
                row["ok"] = row["present"] = False
                row["token_encoder"] = {"repo_id": te, "present": False, "detail": str(e)[:120]}
    return row


def _print_report(rows: list[dict[str, Any]], *, check: bool, total: int) -> None:
    verb = "Verifying (cache only)" if check else "Downloading"
    print(f"{verb} {len(rows)} model(s) → HF cache\n{'=' * 72}")
    for r in rows:
        ok = r.get("present") if check else r.get("ok")
        mark = "OK " if ok else " ! "
        status = ("present" if r.get("present") else "MISSING") if check else ("ok" if r.get("ok") else "FAILED")
        print(f"  [{mark}] {r['name']:<12} {status:<8} {_human(r['size']):>10}  {r['repo_id']}")
        if not ok:
            print(f"          {r['detail']}")
        te = r.get("token_encoder")
        if te and not te.get("present"):
            print(f"          token_encoder {te['repo_id']}: {te.get('detail', 'missing')}")
        print(f"          ({r['reason']})")
    ok_count = sum(1 for r in rows if (r.get("present") if check else r.get("ok")))
    print(f"{'=' * 72}\n  {ok_count}/{len(rows)} OK · total cached {_human(total)}")


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

    settings = MemexSettings()  # type: ignore[call-arg]  # no bootstrap() — downloads need no CUDA
    set_settings(settings)

    targets = resolve_model_targets(settings, include_all=args.all)
    if args.only:
        wanted = {n.lower() for n in args.only}
        targets = [t for t in targets if t.name.lower() in wanted]
        if not targets:
            known = [t.name for t in resolve_model_targets(settings, include_all=True)]
            print(f"--only matched no models; known: {known}", file=sys.stderr)
            return 2

    rows = [process_target(t, check=args.check) for t in targets]
    total = sum(r["size"] for r in rows)

    if args.json:
        print(json.dumps({"check": args.check, "rows": rows, "total_bytes": total}, indent=1, default=str))
    else:
        _print_report(rows, check=args.check, total=total)

    if any(r.get("setup_error") for r in rows):
        return 2
    failed = [r for r in rows if not (r.get("present") if args.check else r.get("ok"))]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
