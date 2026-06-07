"""Unit tests for the model-cache bootstrap (`memex.models.download`).

The shared logic behind the `memex download-models` CLI command, the
`scripts/download-models.py` shim, and the webui `/resources` model-cache panel. All
HF / faster-whisper fetches are monkeypatched, so NOTHING here touches the network or
the real cache.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import memex.models.download as dl

_REPO = Path(__file__).resolve().parents[2]


def _fake_settings(**over: Any) -> SimpleNamespace:
    models = SimpleNamespace(
        orchestrator=over.get("orchestrator", "org/orch"),
        embedder=over.get("embedder", "org/emb"),
        reranker=over.get("reranker", "org/rer"),
        reranker_backend=over.get("reranker_backend", "cross_encoder"),
        vlm=over.get("vlm", "org/vlm"),
        chart_ocr=over.get("chart_ocr", "org/chart"),
        summarizer=over.get("summarizer", None),
        asr=over.get("asr", None),
        reasoner=over.get("reasoner", None),
    )
    parse = SimpleNamespace(
        disable_vlm=over.get("disable_vlm", True),
        disable_chart_ocr=over.get("disable_chart_ocr", False),
    )
    agents = SimpleNamespace(
        enrich_ner_backend=over.get("enrich_ner_backend", "llm"),
        enrich_ner_model=over.get("enrich_ner_model", "org/otter"),
    )
    return SimpleNamespace(models=models, parse=parse, agents=agents)


# ----- resolver -----


def test_resolver_default_is_core_plus_chart_ocr() -> None:
    """Defaults: disable_vlm=True (no VLM), disable_chart_ocr=False (chart-OCR in), backend=llm
    (no OTTER), asr/summarizer None. → core trio + chart-ocr."""
    names = [t.name for t in dl.resolve_model_targets(_fake_settings(), include_all=False)]
    assert names == ["orchestrator", "embedder", "reranker", "chart-ocr"]


def test_resolver_all_forces_capability_models() -> None:
    """--all pulls in VLM + OTTER even though disable_vlm=True / backend=llm; ASR stays out
    (None — --all can't invent an id)."""
    names = [t.name for t in dl.resolve_model_targets(_fake_settings(), include_all=True)]
    assert names == ["orchestrator", "embedder", "reranker", "vlm", "chart-ocr", "otter"]
    assert "asr" not in names


def test_resolver_honors_each_gate() -> None:
    s = _fake_settings(
        disable_chart_ocr=True,  # → no chart-ocr
        enrich_ner_backend="otter",  # → otter WITHOUT --all
        summarizer="org/sum",  # → summarizer
        asr="large-v3-turbo",  # → asr (the size name)
    )
    targets = dl.resolve_model_targets(s, include_all=False)
    names = [t.name for t in targets]
    assert names == ["orchestrator", "embedder", "reranker", "summarizer", "otter", "asr"]
    assert "chart-ocr" not in names
    asr = next(t for t in targets if t.name == "asr")
    assert asr.kind == "asr" and asr.repo_id == "large-v3-turbo"
    assert next(t for t in targets if t.name == "otter").kind == "otter"


def test_resolver_dedups_by_repo_id() -> None:
    s = _fake_settings(reranker="org/emb")  # reranker shares the embedder's repo
    repos = [t.repo_id for t in dl.resolve_model_targets(s, include_all=False)]
    assert repos.count("org/emb") == 1  # deduped, first-seen (embedder) wins
    assert "reranker" not in [t.name for t in dl.resolve_model_targets(s, include_all=False)]


def test_resolver_skips_reasoner() -> None:
    s = _fake_settings(reasoner="org/reasoner-RESERVED")
    repos = [t.repo_id for t in dl.resolve_model_targets(s, include_all=True)]
    assert "org/reasoner-RESERVED" not in repos  # reserved hook, never auto-served → never fetched


# ----- process_target -----


def test_process_target_check_missing_records_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(repo_id: str, *, check: bool) -> str:
        raise FileNotFoundError("not in local cache")  # what snapshot_download(local_files_only) raises

    monkeypatch.setattr(dl, "_snapshot", _raise)
    row = dl.process_target(dl.ModelTarget("embedder", "org/emb", "hf", "core"), check=True)
    assert row["present"] is False and row["ok"] is False
    assert "not cached" in row["detail"]
    assert not row.get("setup_error")


def test_process_target_download_ok_reports_size(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "model.bin").write_bytes(b"x" * 4096)
    monkeypatch.setattr(dl, "_snapshot", lambda repo_id, *, check: str(tmp_path))
    row = dl.process_target(dl.ModelTarget("embedder", "org/emb", "hf", "core"), check=False)
    assert row["ok"] is True and row["size"] == 4096


def test_process_target_asr_missing_dep_is_setup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_dep(repo_id: str, *, check: bool) -> str:
        raise ImportError("No module named 'faster_whisper'")

    monkeypatch.setattr(dl, "_asr_download", _no_dep)
    row = dl.process_target(dl.ModelTarget("asr", "large-v3-turbo", "asr", "asr set"), check=True)
    assert row["setup_error"] is True and row["ok"] is False


def test_process_target_otter_fetches_token_encoder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """OTTER's main repo carries a `config.json` with a `token_encoder` repo → a SECOND fetch."""
    main_dir = tmp_path / "otter"
    main_dir.mkdir()
    (main_dir / "config.json").write_text(json.dumps({"token_encoder": "org/mmbert-base"}))
    te_dir = tmp_path / "mmbert"
    te_dir.mkdir()
    (te_dir / "model.safetensors").write_bytes(b"y" * 2048)

    calls: list[str] = []

    def _snap(repo_id: str, *, check: bool) -> str:
        calls.append(repo_id)
        return str(main_dir) if repo_id == "org/otter" else str(te_dir)

    monkeypatch.setattr(dl, "_snapshot", _snap)
    row = dl.process_target(dl.ModelTarget("otter", "org/otter", "otter", "ner"), check=False)
    assert calls == ["org/otter", "org/mmbert-base"]  # the transitive repo was fetched too
    te = row["token_encoder"]
    assert te["repo_id"] == "org/mmbert-base" and te["present"]


# ----- run_download (exit-code orchestration) -----


def test_run_download_check_all_present_exits_0(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"z" * 10)
    monkeypatch.setattr(dl, "_snapshot", lambda repo_id, *, check: str(tmp_path))
    rows, code = dl.run_download(_fake_settings(), check=True, include_all=False)
    assert code == 0 and len(rows) == 4 and all(r["present"] for r in rows)


def test_run_download_check_one_missing_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"z" * 10)

    def _snap(repo_id: str, *, check: bool) -> str:
        if repo_id == "org/rer":  # the reranker isn't cached
            raise FileNotFoundError("missing")
        return str(tmp_path)

    monkeypatch.setattr(dl, "_snapshot", _snap)
    _rows, code = dl.run_download(_fake_settings(), check=True, include_all=False)
    assert code == 1


def test_run_download_asr_dep_missing_exits_2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"z" * 10)
    monkeypatch.setattr(dl, "_snapshot", lambda repo_id, *, check: str(tmp_path))

    def _no_dep(repo_id: str, *, check: bool) -> str:
        raise ImportError("No module named 'faster_whisper'")

    monkeypatch.setattr(dl, "_asr_download", _no_dep)
    _rows, code = dl.run_download(_fake_settings(asr="large-v3-turbo"), check=True, include_all=False)
    assert code == 2  # setup error dominates the exit code


def test_run_download_only_filters_to_named(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"z" * 10)
    monkeypatch.setattr(dl, "_snapshot", lambda repo_id, *, check: str(tmp_path))
    rows, code = dl.run_download(
        _fake_settings(), check=True, include_all=False, only=["embedder", "reranker"]
    )
    assert code == 0 and [r["name"] for r in rows] == ["embedder", "reranker"]


def test_run_download_only_no_match_exits_2() -> None:
    rows, code = dl.run_download(_fake_settings(), check=True, include_all=False, only=["nope"])
    assert rows == [] and code == 2


# ----- format_report -----


def test_format_report_renders_status_and_total(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"z" * 2048)
    monkeypatch.setattr(dl, "_snapshot", lambda repo_id, *, check: str(tmp_path))
    rows, _code = dl.run_download(_fake_settings(), check=True, include_all=False)
    out = dl.format_report(rows, check=True)
    assert "embedder" in out and "org/emb" in out
    assert "present" in out  # the per-row status verb (check mode)
    assert "4/4 OK" in out  # the footer tally


def test_format_report_shows_missing_detail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _snap(repo_id: str, *, check: bool) -> str:
        if repo_id == "org/rer":
            raise FileNotFoundError("missing")
        return str(tmp_path)

    (tmp_path / "f").write_bytes(b"z" * 10)
    monkeypatch.setattr(dl, "_snapshot", _snap)
    rows, _code = dl.run_download(_fake_settings(), check=True, include_all=False)
    out = dl.format_report(rows, check=True)
    assert "MISSING" in out and "not cached" in out


# ----- model_cache_status (webui view-model) -----


def test_model_cache_status_all_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"z" * 10)
    monkeypatch.setattr(dl, "_snapshot", lambda repo_id, *, check: str(tmp_path))
    status = dl.model_cache_status(_fake_settings())
    assert status is not None
    configured = status["configured"]
    assert isinstance(configured, list) and len(configured) == 4
    assert all(c["present"] for c in configured)
    assert status["missing"] == 0 and status["action_hint"] is None


def test_model_cache_status_missing_sets_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _snap(repo_id: str, *, check: bool) -> str:
        if repo_id == "org/chart":
            raise FileNotFoundError("missing")
        return str(tmp_path)

    (tmp_path / "f").write_bytes(b"z" * 10)
    monkeypatch.setattr(dl, "_snapshot", _snap)
    status = dl.model_cache_status(_fake_settings())
    assert status is not None
    assert status["missing"] == 1
    hint = status["action_hint"]
    assert isinstance(hint, str) and "download-models" in hint


def test_model_cache_status_fail_safe_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A probe failure must NEVER 500 /resources — the panel returns None."""

    def _boom(t: dl.ModelTarget, *, check: bool) -> dl.DownloadRow:
        raise RuntimeError("cache is corrupt")

    monkeypatch.setattr(dl, "process_target", _boom)
    assert dl.model_cache_status(_fake_settings()) is None


# ----- back-compat: the raw script shim still works -----


def _load_shim() -> ModuleType:
    path = _REPO / "scripts" / "download-models.py"
    spec = importlib.util.spec_from_file_location("download_models_cli", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_script_shim_main_returns_run_download_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"z" * 10)
    monkeypatch.setattr("memex.core.config.MemexSettings", lambda: _fake_settings())
    monkeypatch.setattr("memex.core.config.set_settings", lambda _s: None)
    monkeypatch.setattr(dl, "_snapshot", lambda repo_id, *, check: str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["download-models.py", "--check"])
    shim = _load_shim()
    assert shim.main() == 0  # delegates to run_download → all present → 0
