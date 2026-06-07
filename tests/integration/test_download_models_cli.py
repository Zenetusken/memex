"""The `memex download-models` CLI command (registered in `cli/commands.register`).

`run_download` is faked, so these drive the command's option parsing, output rendering,
and exit-code wiring — no network, no HF cache, no CUDA (the command deliberately skips
`bootstrap()` so it runs on a GPU-less box). Invoked via Typer's CliRunner against a fresh
app the way the production entry point builds it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from memex.cli.commands import register
from memex.models.download import DownloadRow


def _app() -> typer.Typer:
    app = typer.Typer()
    register(app)
    return app


@pytest.fixture(autouse=True)
def _env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # MemexSettings() (no bootstrap) reads these; a tmp vault keeps it CUDA-free + offline.
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")


def _rows(present: bool = True) -> list[DownloadRow]:
    return [
        {"name": "embedder", "repo_id": "org/emb", "present": present, "ok": present,
         "size": 2048, "reason": "core"},
    ]


def test_download_models_check_exit_0_renders_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("memex.models.download.run_download", lambda *a, **k: (_rows(), 0))
    result = CliRunner().invoke(_app(), ["download-models", "--check"])
    assert result.exit_code == 0
    assert "embedder" in result.stdout and "org/emb" in result.stdout  # the text report


def test_download_models_propagates_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    # A missing model → run_download returns code 1 → the command exits 1 (scriptable).
    monkeypatch.setattr("memex.models.download.run_download", lambda *a, **k: (_rows(present=False), 1))
    result = CliRunner().invoke(_app(), ["download-models", "--check"])
    assert result.exit_code == 1


def test_download_models_json_emits_machine_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("memex.models.download.run_download", lambda *a, **k: (_rows(), 0))
    result = CliRunner().invoke(_app(), ["download-models", "--check", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["check"] is True
    assert [r["name"] for r in payload["rows"]] == ["embedder"]
    assert payload["total_bytes"] == 2048


def test_download_models_only_no_match_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    # run_download returns ([], 2) when --only matches nothing; the command surfaces the known
    # names and exits 2 (a setup/usage error, distinct from a missing-model 1).
    monkeypatch.setattr("memex.models.download.run_download", lambda *a, **k: ([], 2))
    result = CliRunner().invoke(_app(), ["download-models", "--check", "--only", "nope"])
    assert result.exit_code == 2
