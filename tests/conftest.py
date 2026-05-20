"""Shared test configuration.

Tests run with Langfuse disabled by default to avoid any chance of an
outbound network call from a unit run. Tests that explicitly exercise
the tracing path can re-enable it with `monkeypatch.delenv(...)`.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
