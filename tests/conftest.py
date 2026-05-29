"""Shared test configuration.

Tests run with Langfuse disabled by default to avoid any chance of an
outbound network call from a unit run. Tests that explicitly exercise
the tracing path can re-enable it with `monkeypatch.delenv(...)`.

The PyMuPDF pre-filter is also disabled by default so existing parse
tests are deterministic. Tests that exercise the pre-filter opt in
explicitly via the `patch_pymupdf` fixture which monkey-patches
`pymupdf_convert` to a fake conversion.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_config_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Make the suite HERMETIC w.r.t. the developer's `~/.config/memex/config.toml`.

    `MemexSettings` bakes `toml_file = ~/.config/memex/config.toml` into `model_config` and
    reads it on every construction, so a real production config (e.g. an activated
    `enrich_ner_backend = "otter"`) would silently bleed into tests and change behaviour.
    Point the TOML source at an empty temp file → pure defaults + the test's own env
    overrides. (Tests that want a TOML value set it explicitly via init kwargs / env.)
    """
    from memex.core.config import MemexSettings

    empty = tmp_path_factory.mktemp("memex_cfg") / "config.toml"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setitem(MemexSettings.model_config, "toml_file", str(empty))


@pytest.fixture(autouse=True)
def _disable_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")


@pytest.fixture(autouse=True)
def _disable_pymupdf_prefilter_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing parse tests written before the PyMuPDF pre-filter assume
    Docling runs unconditionally. To keep them deterministic without
    rewriting them all, autouse a fake that immediately raises
    `PyMuPDFUnavailable`, which the pipeline catches and falls through
    to Docling silently.

    Tests that exercise the pre-filter override this with `patch_pymupdf`
    (or by directly re-patching `memex.parse.pipeline.pymupdf_convert`).
    """

    async def _unavailable(source: Path, *, timeout_s: int, **_kw: object) -> object:
        from memex.parse.pymupdf_backend import PyMuPDFUnavailable

        raise PyMuPDFUnavailable(
            "pymupdf disabled in test by default",
            context={"source": str(source)},
        )

    monkeypatch.setattr(
        "memex.parse.pipeline.pymupdf_convert",
        _unavailable,
        raising=False,
    )
