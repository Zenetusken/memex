"""Unit tests for the model-bootstrap CLI (`scripts/download-models.py`).

A `scripts/` dev tool → loaded via importlib (not a package import). All HF/faster-whisper
fetches are monkeypatched, so NOTHING here touches the network or the real cache.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    path = _REPO / "scripts" / "download-models.py"
    spec = importlib.util.spec_from_file_location("download_models_cli", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # @dataclass needs the module registered before exec
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cli() -> ModuleType:
    return _load()


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


def test_resolver_default_is_core_plus_chart_ocr(cli: ModuleType) -> None:
    """Defaults: disable_vlm=True (no VLM), disable_chart_ocr=False (chart-OCR in), backend=llm
    (no OTTER), asr/summarizer None. → core trio + chart-ocr."""
    names = [t.name for t in cli.resolve_model_targets(_fake_settings(), include_all=False)]
    assert names == ["orchestrator", "embedder", "reranker", "chart-ocr"]


def test_resolver_all_forces_capability_models(cli: ModuleType) -> None:
    """--all pulls in VLM + OTTER even though disable_vlm=True / backend=llm; ASR stays out
    (None — --all can't invent an id)."""
    names = [t.name for t in cli.resolve_model_targets(_fake_settings(), include_all=True)]
    assert names == ["orchestrator", "embedder", "reranker", "vlm", "chart-ocr", "otter"]
    assert "asr" not in names


def test_resolver_honors_each_gate(cli: ModuleType) -> None:
    s = _fake_settings(
        disable_chart_ocr=True,  # → no chart-ocr
        enrich_ner_backend="otter",  # → otter WITHOUT --all
        summarizer="org/sum",  # → summarizer
        asr="large-v3-turbo",  # → asr (the size name)
    )
    targets = cli.resolve_model_targets(s, include_all=False)
    names = [t.name for t in targets]
    assert names == ["orchestrator", "embedder", "reranker", "summarizer", "otter", "asr"]
    assert "chart-ocr" not in names
    asr = next(t for t in targets if t.name == "asr")
    assert asr.kind == "asr" and asr.repo_id == "large-v3-turbo"
    assert next(t for t in targets if t.name == "otter").kind == "otter"


def test_resolver_dedups_by_repo_id(cli: ModuleType) -> None:
    s = _fake_settings(reranker="org/emb")  # reranker shares the embedder's repo
    repos = [t.repo_id for t in cli.resolve_model_targets(s, include_all=False)]
    assert repos.count("org/emb") == 1  # deduped, first-seen (embedder) wins
    assert "reranker" not in [t.name for t in cli.resolve_model_targets(s, include_all=False)]


def test_resolver_skips_reasoner(cli: ModuleType) -> None:
    s = _fake_settings(reasoner="org/reasoner-RESERVED")
    repos = [t.repo_id for t in cli.resolve_model_targets(s, include_all=True)]
    assert "org/reasoner-RESERVED" not in repos  # reserved hook, never auto-served → never fetched


# ----- process_target -----


def test_process_target_check_missing_records_not_cached(cli: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(repo_id: str, *, check: bool) -> str:
        raise FileNotFoundError("not in local cache")  # what snapshot_download(local_files_only) raises

    monkeypatch.setattr(cli, "_snapshot", _raise)
    row = cli.process_target(cli.ModelTarget("embedder", "org/emb", "hf", "core"), check=True)
    assert row["present"] is False and row["ok"] is False
    assert "not cached" in row["detail"]
    assert not row.get("setup_error")


def test_process_target_download_ok_reports_size(cli: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "model.bin").write_bytes(b"x" * 4096)
    monkeypatch.setattr(cli, "_snapshot", lambda repo_id, *, check: str(tmp_path))
    row = cli.process_target(cli.ModelTarget("embedder", "org/emb", "hf", "core"), check=False)
    assert row["ok"] is True and row["size"] == 4096


def test_process_target_asr_missing_dep_is_setup_error(cli: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_dep(repo_id: str, *, check: bool) -> str:
        raise ImportError("No module named 'faster_whisper'")

    monkeypatch.setattr(cli, "_asr_download", _no_dep)
    row = cli.process_target(cli.ModelTarget("asr", "large-v3-turbo", "asr", "asr set"), check=True)
    assert row["setup_error"] is True and row["ok"] is False


def test_process_target_otter_fetches_token_encoder(cli: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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

    monkeypatch.setattr(cli, "_snapshot", _snap)
    row = cli.process_target(cli.ModelTarget("otter", "org/otter", "otter", "ner"), check=False)
    assert calls == ["org/otter", "org/mmbert-base"]  # the transitive repo was fetched too
    assert row["token_encoder"]["repo_id"] == "org/mmbert-base" and row["token_encoder"]["present"]


# ----- main (exit codes) -----


def _patch_settings(cli: ModuleType, monkeypatch: pytest.MonkeyPatch, settings: SimpleNamespace) -> None:
    monkeypatch.setattr("memex.core.config.MemexSettings", lambda: settings)
    monkeypatch.setattr("memex.core.config.set_settings", lambda _s: None)


def test_main_check_all_present_exits_0(cli: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"z" * 10)
    _patch_settings(cli, monkeypatch, _fake_settings())
    monkeypatch.setattr(cli, "_snapshot", lambda repo_id, *, check: str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["download-models.py", "--check"])
    assert cli.main() == 0


def test_main_check_one_missing_exits_1(cli: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"z" * 10)

    def _snap(repo_id: str, *, check: bool) -> str:
        if repo_id == "org/rer":  # the reranker isn't cached
            raise FileNotFoundError("missing")
        return str(tmp_path)

    _patch_settings(cli, monkeypatch, _fake_settings())
    monkeypatch.setattr(cli, "_snapshot", _snap)
    monkeypatch.setattr(sys, "argv", ["download-models.py", "--check"])
    assert cli.main() == 1


def test_main_asr_dep_missing_exits_2(cli: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"z" * 10)
    _patch_settings(cli, monkeypatch, _fake_settings(asr="large-v3-turbo"))
    monkeypatch.setattr(cli, "_snapshot", lambda repo_id, *, check: str(tmp_path))

    def _no_dep(repo_id: str, *, check: bool) -> str:
        raise ImportError("No module named 'faster_whisper'")

    monkeypatch.setattr(cli, "_asr_download", _no_dep)
    monkeypatch.setattr(sys, "argv", ["download-models.py", "--check"])
    assert cli.main() == 2  # setup error dominates the exit code
