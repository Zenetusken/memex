"""Model-cache bootstrap — resolve the configured model ids from `MemexSettings` and fetch /
verify them in the HF cache. The shared core behind the `download-models` CLI command, the
`scripts/download-models.py` shim, and the webui `/resources` model-cache status panel.

Local-first / air-gapped (VISION.md Principle 1): run this once ONLINE on a fresh machine and the
runtime then loads every model OFFLINE. `huggingface_hub.snapshot_download` fetches a whole repo to
the cache WITHOUT loading it (no torch, no GPU) and verifies etags — so this is a CUDA-free bootstrap
that must run even on a GPU-less box (callers therefore do NOT `bootstrap()`).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import structlog

from memex.core.config import MemexSettings

logger = structlog.get_logger(__name__)

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


class _RowBase(TypedDict):
    """The keys `process_target` ALWAYS sets at construction — required, so report rendering can
    index them directly."""

    name: str
    repo_id: str
    reason: str


class DownloadRow(_RowBase, total=False):
    """A per-target result row (report + view-model). The `total=False` extras are added once a
    target is fetched/verified; a failed/missing row omits them, so consumers use `.get(...)`."""

    ok: bool
    present: bool
    size: int
    detail: str
    setup_error: bool
    token_encoder: dict[str, object]


def resolve_model_targets(settings: MemexSettings, *, include_all: bool) -> list[ModelTarget]:
    """Resolve the model set the CONFIGURED system needs, from `MemexSettings`.

    CORE (always — the `/ask` spine + the orchestrator vLLM serves from the HF cache, so it must be
    cached for offline serving): orchestrator, embedder, reranker.
    CAPABILITY (gated by config, or forced by `--all` — the full offline kit): VLM, chart-OCR,
    summarizer (if set), OTTER NER (if the backend is otter), ASR (if an id is set — `--all` can't
    invent one when it's None). `reasoner` is a reserved hook that never auto-serves → skip. Deduped
    by repo_id (preserve first-seen order)."""
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
    """OTTER loads a SECOND repo via `config.token_encoder` (enrich/ner_otter.py — the mmBERT base).
    Read it out of the downloaded `config.json` so it can be cached too. Best-effort + defensive."""
    cfg = Path(snapshot_dir) / "config.json"
    if not cfg.is_file():
        return None
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    te = data.get("token_encoder")
    if isinstance(te, str) and "/" in te and not te.startswith((".", "/")):
        return te
    return None


def _snapshot(repo_id: str, *, check: bool) -> str:
    """Fetch (or, with check=True, locate-without-network) a repo; return its local snapshot dir."""
    from huggingface_hub import (
        snapshot_download,  # type: ignore[reportUnknownVariableType]  # partial py.typed: Dict[Unknown] params
    )

    # Cast to the surface we use so the call site is fully typed (the documented convention for
    # partial-py.typed libs; mirrors registry._from_pretrained / ner_otter's HF-loader handling).
    snap = cast("Callable[..., str]", snapshot_download)
    return snap(repo_id, local_files_only=check)


def _asr_download(repo_id: str, *, check: bool) -> str:
    """faster-whisper resolves a size name → its CT2 repo and snapshot_downloads it (same HF cache)."""
    from faster_whisper.utils import download_model

    return download_model(repo_id, local_files_only=check)


def process_target(t: ModelTarget, *, check: bool) -> DownloadRow:
    """Fetch or verify one target. Returns a result row: ok/present, size, detail, extra. Never raises
    for an expected miss / HF error (the row records the failure); a missing optional dependency surfaces
    via the `setup_error` flag so callers can exit 2."""
    row: DownloadRow = {"name": t.name, "repo_id": t.repo_id, "reason": t.reason}
    try:
        path = _asr_download(t.repo_id, check=check) if t.kind == "asr" else _snapshot(t.repo_id, check=check)
    except ImportError as e:
        # ASR configured but `faster_whisper` (the `audio` extra) isn't installed → setup error.
        row.update(ok=False, present=False, size=0, detail=f"dependency missing: {e}", setup_error=True)
        return row
    except Exception as e:  # an expected miss (check) or HF/network error: record, don't abort the batch
        verb = "not cached" if check else "download failed"
        row.update(ok=False, present=False, size=0, detail=f"{verb}: {type(e).__name__}: {str(e)[:160]}")
        return row

    row.update(ok=True, present=True, size=_dir_size_bytes(path), detail=str(path))

    if t.kind == "otter":  # also handle the transitive token_encoder repo (mmBERT base)
        te = _token_encoder_repo(path)
        if te:
            try:
                te_path = _snapshot(te, check=check)
                row["token_encoder"] = {"repo_id": te, "present": True, "size": _dir_size_bytes(te_path)}
                row["size"] = row.get("size", 0) + _dir_size_bytes(te_path)
            except Exception as e:  # record the transitive miss; OTTER is optional
                row["ok"] = row["present"] = False
                row["token_encoder"] = {"repo_id": te, "present": False, "detail": str(e)[:120]}
    return row


def run_download(
    settings: MemexSettings,
    *,
    check: bool,
    include_all: bool,
    only: list[str] | None = None,
) -> tuple[list[DownloadRow], int]:
    """Resolve the configured targets (optionally restricted by `only` names), fetch/verify each, and
    compute the exit code: 0 = all present/fetched · 1 = any missing (check) or failed · 2 = setup error
    (a missing optional dependency, or `only` matched nothing). The shared orchestration for the CLI +
    the script shim."""
    targets = resolve_model_targets(settings, include_all=include_all)
    if only:
        wanted = {n.lower() for n in only}
        targets = [t for t in targets if t.name.lower() in wanted]
    if not targets:
        return [], 2  # empty config (impossible) or `only` matched nothing → usage/setup error
    rows = [process_target(t, check=check) for t in targets]
    if any(r.get("setup_error") for r in rows):
        return rows, 2
    failed = [r for r in rows if not (r.get("present") if check else r.get("ok"))]
    return rows, (1 if failed else 0)


def format_report(rows: list[DownloadRow], *, check: bool) -> str:
    """Render the per-model report as plain text (shared by the CLI + the script)."""
    verb = "Verifying (cache only)" if check else "Downloading"
    lines = [f"{verb} {len(rows)} model(s) → HF cache", "=" * 72]
    for r in rows:
        ok = r.get("present") if check else r.get("ok")
        mark = "OK " if ok else " ! "
        status = ("present" if r.get("present") else "MISSING") if check else ("ok" if r.get("ok") else "FAILED")
        lines.append(f"  [{mark}] {r['name']:<12} {status:<8} {_human(r.get('size', 0)):>10}  {r['repo_id']}")
        if not ok:
            lines.append(f"          {r.get('detail', '')}")
        te = r.get("token_encoder")
        if te and not te.get("present"):
            lines.append(f"          token_encoder {te['repo_id']}: {te.get('detail', 'missing')}")
        lines.append(f"          ({r.get('reason', '')})")
    ok_count = sum(1 for r in rows if (r.get("present") if check else r.get("ok")))
    total = sum(r.get("size", 0) for r in rows)
    lines.append("=" * 72)
    lines.append(f"  {ok_count}/{len(rows)} OK · total cached {_human(total)}")
    return "\n".join(lines)


def model_cache_status(settings: MemexSettings) -> dict[str, object] | None:
    """Read-only model-cache view-model for the webui `/resources` panel: each CONFIGURED model's
    presence + on-disk size (no network — `local_files_only`). Returns `None` on any fatal error so a
    status panel never 500s the page (the `_vram_panel` fail-safe convention). No side effects."""
    try:
        configured: list[dict[str, object]] = []
        for t in resolve_model_targets(settings, include_all=False):
            row = process_target(t, check=True)
            present = bool(row.get("present"))
            configured.append(
                {
                    "name": t.name,
                    "repo_id": t.repo_id,
                    "present": present,
                    "size_gb": round(row.get("size", 0) / (1024**3), 2),
                }
            )
        missing = sum(1 for c in configured if not c["present"])
        return {
            "configured": configured,
            "missing": missing,
            "action_hint": (
                "Run `memex download-models` (online) to cache the missing model(s)." if missing else None
            ),
        }
    except Exception:  # fail-safe: a status panel must never 500 /resources
        logger.warning("models.cache_status_failed")
        return None
