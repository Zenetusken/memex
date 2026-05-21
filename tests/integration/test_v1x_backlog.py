"""v1.x backlog regression net.

Covers:
- Daemon supervisor: start refuses to overwrite a live PID, stop is a
  no-op on a fresh vault, status reports the layout shape correctly.
- Watcher → event-bus integration: a confirmed edit publishes a
  `vault.edited` event to the bus when one is configured; the bus
  being absent doesn't break the edit flow.
- Subprocess-sandboxed Docling: a worker that returns non-zero exit
  raises `DoclingCrashed`; a worker that returns valid JSON deserialises
  cleanly via `DoclingConversion.model_validate`.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from memex.core.bus import EventBus, set_bus
from memex.core.config import MemexSettings, set_settings


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    (tmp_path / "documents").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def settings(
    tmp_vault: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[MemexSettings]:
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_vault))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    s = MemexSettings()  # type: ignore[call-arg]
    set_settings(s)
    yield s
    set_settings(None)


# ----- Daemon supervisor -----


@pytest.mark.asyncio
async def test_daemon_status_reports_no_pid_on_fresh_vault(
    settings: MemexSettings,
) -> None:
    from memex.daemon import status

    result = await status(settings)
    assert result.pid is None
    assert result.alive is False
    # `reachable` depends on whether anything is listening on the
    # configured base_url. We can't assert it deterministically.


def test_daemon_stop_is_noop_without_pid_file(
    settings: MemexSettings,
) -> None:
    from memex.daemon import stop

    result = stop(settings)
    assert result.pid is None
    assert result.alive is False
    assert (result.error or "").startswith("no PID file")


def test_daemon_start_refuses_when_existing_pid_is_alive(
    settings: MemexSettings,
) -> None:
    """If the PID file points at a live process, `start` must error
    rather than spawn a duplicate."""
    from memex.daemon import DaemonAlreadyRunning, start

    pid_file = settings.vault_path / ".memex" / "daemon" / "vllm.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Our own PID is guaranteed alive.
    pid_file.write_text(str(os.getpid()), encoding="utf-8")

    with pytest.raises(DaemonAlreadyRunning):
        asyncio.run(start(settings))

    # PID file is preserved — we didn't disturb the "live" daemon.
    assert pid_file.exists()


def test_daemon_start_cleans_up_stale_pid_then_errors_on_missing_script(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stale PID (process not alive) is cleaned up before the next
    spawn attempt. We can verify the cleanup path by pointing
    `serve_script` at a missing path — `start` then raises
    `MemexError(serve-vllm script not found)`."""
    from memex.core.errors import MemexError
    from memex.daemon import start

    pid_file = settings.vault_path / ".memex" / "daemon" / "vllm.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # PID that almost certainly doesn't exist on this system.
    pid_file.write_text("999999", encoding="utf-8")

    # Point at a non-existent script so we don't actually spawn anything.
    monkeypatch.setenv(
        "MEMEX_INFERENCE__SERVE_SCRIPT",
        "/nonexistent/serve-vllm.sh",
    )
    new_settings = MemexSettings()  # type: ignore[call-arg]
    set_settings(new_settings)
    try:
        with pytest.raises(MemexError, match="serve-vllm script not found"):
            asyncio.run(start(new_settings))
        # The stale PID file was cleaned up.
        assert not pid_file.exists()
    finally:
        set_settings(settings)


# ----- Watcher → event bus -----


@pytest.mark.asyncio
async def test_watcher_confirm_publishes_to_bus_when_configured(
    settings: MemexSettings, tmp_path: Path
) -> None:
    """When a bus is set, the watcher's confirm-edit path publishes a
    `vault.edited` event to it. Stand up an in-memory bus (no SQLite),
    subscribe, drive an edit, assert the event is delivered."""
    from memex.ingest.pipeline import ingest_markdown_passthrough
    from memex.ingest.watcher import _confirm_user_edit, publish_edit_notice

    bus = EventBus(db_path=None, queue_size=16).start()
    set_bus(bus)
    try:
        ref = await ingest_markdown_passthrough(
            "# Bus test\n\nbefore.\n", source_stem="bus_test"
        )
        md_path = settings.vault_path / "documents" / f"{ref.doc_id}.md"
        # Make sure _confirm_user_edit will surface a notice.
        with md_path.open("a", encoding="utf-8") as f:
            f.write("\n\nedit from outside.\n")

        # Drive the watcher's actual publish path: `_drain_one` calls
        # `publish_edit_notice(notice)`, so we call the same helper
        # rather than re-implementing what we're testing.
        notice = await _confirm_user_edit(settings.vault_path, md_path)
        assert notice is not None

        captured: list[Any] = []

        async def _collect() -> None:
            async with bus.subscribe(event_type="vault.edited") as q:
                event = await asyncio.wait_for(q.get(), timeout=2.0)
                captured.append(event)

        # Subscribe first, then publish via the watcher's own helper.
        collector = asyncio.create_task(_collect())
        await asyncio.sleep(0.01)  # let the subscriber register
        await publish_edit_notice(notice)
        await asyncio.wait_for(collector, timeout=2.0)
        assert len(captured) == 1
        assert captured[0].event_type == "vault.edited"
        assert captured[0].payload["doc_id"] == ref.doc_id
    finally:
        set_bus(None)
        bus.stop()


@pytest.mark.asyncio
async def test_watcher_publish_failure_does_not_break_edit_flow(
    settings: MemexSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the bus is set but publishing throws, the watcher's publish
    helper swallows the exception. Best-effort observability, never
    blocks the pipeline.

    Drives the actual publish helper that `_drain_one` calls — earlier
    versions of this test invoked `bus.publish` directly inside their
    own try/except, which validated the test author's understanding
    rather than the code under test.
    """
    from memex.core.events import MemexEvent
    from memex.ingest.pipeline import ingest_markdown_passthrough
    from memex.ingest.watcher import (
        EditNotice,
        _confirm_user_edit,
        publish_edit_notice,
    )

    class _ExplodingBus(EventBus):
        async def publish(self, event: MemexEvent) -> None:  # type: ignore[override]
            raise RuntimeError("bus dead — should not block on_edit")

    bus = _ExplodingBus(db_path=None).start()
    set_bus(bus)
    try:
        ref = await ingest_markdown_passthrough(
            "# Bus failure\n\nbefore.\n", source_stem="bus_fail"
        )
        md = settings.vault_path / "documents" / f"{ref.doc_id}.md"
        with md.open("a", encoding="utf-8") as f:
            f.write("\n\nuser edit.\n")

        notice = await _confirm_user_edit(settings.vault_path, md)
        assert notice is not None
        # The actual integration point: `publish_edit_notice` is what
        # `_drain_one` calls in production. It must not propagate the
        # bus exception even though the bus's `publish` raises.
        await publish_edit_notice(notice)
        # If we got here, the helper swallowed the RuntimeError as
        # required. Sanity-check the structured type is what we expect.
        assert isinstance(notice, EditNotice)
    finally:
        set_bus(None)
        bus.stop()


# ----- Subprocess-sandboxed Docling -----


@pytest.mark.asyncio
async def test_docling_worker_subprocess_crash_surfaces_docling_crashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point the subprocess at a worker that exits non-zero. The parent
    must surface `DoclingCrashed` with the stderr in the context."""
    from memex.parse.docling_backend import DoclingCrashed, convert

    # Patch the worker-module name to a tiny inline script that exits 7
    # with a stderr message — simulates a Docling segfault.
    crash_script = tmp_path / "crash.py"
    crash_script.write_text(
        'import sys; print("boom", file=sys.stderr); sys.exit(7)\n',
        encoding="utf-8",
    )

    # Spawn `python crash.py <source>` instead of `python -m memex.parse.docling_worker`.
    import memex.parse.docling_backend as backend

    real_create_subprocess_exec = asyncio.create_subprocess_exec

    async def _spawn(*args: Any, **kwargs: Any):
        # Replace ['python', '-m', 'memex.parse.docling_worker', source]
        # with ['python', str(crash_script), source].
        new_args = (sys.executable, str(crash_script), args[-1])
        return await real_create_subprocess_exec(*new_args, **kwargs)

    monkeypatch.setattr(
        backend.asyncio,
        "create_subprocess_exec",
        _spawn,
    )

    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n%%EOF\n")

    with pytest.raises(DoclingCrashed) as excinfo:
        await convert(source, timeout_s=30)
    assert excinfo.value.context["exit_code"] == 7
    assert "boom" in str(excinfo.value.context["stderr"])


@pytest.mark.asyncio
async def test_docling_worker_subprocess_success_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker that prints valid `DoclingConversion` JSON must
    deserialise cleanly in the parent."""
    from memex.parse.docling_backend import convert

    ok_script = tmp_path / "ok.py"
    ok_script.write_text(
        'import json, sys\n'
        'payload = {\n'
        '    "markdown": "# Faked\\n\\nbody.\\n",\n'
        '    "pages": [{"page": 1, "markdown": "# Faked", "confidence": 0.92}],\n'
        '    "docling_version": "fake-1.0",\n'
        '    "figure_count": 0,\n'
        '    "table_count": 0,\n'
        '    "equation_count": 0,\n'
        '}\n'
        'json.dump(payload, sys.stdout)\n',
        encoding="utf-8",
    )

    import memex.parse.docling_backend as backend

    real_spawn = asyncio.create_subprocess_exec

    async def _spawn(*args: Any, **kwargs: Any):
        new_args = (sys.executable, str(ok_script), args[-1])
        return await real_spawn(*new_args, **kwargs)

    monkeypatch.setattr(
        backend.asyncio, "create_subprocess_exec", _spawn
    )

    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n%%EOF\n")

    conversion = await convert(source, timeout_s=30)
    assert conversion.markdown.startswith("# Faked")
    assert len(conversion.pages) == 1
    assert conversion.pages[0].confidence == 0.92
    assert conversion.docling_version == "fake-1.0"
    # Backward-compatibility: a payload without the P3.3 `figures`
    # field deserialises with an empty default. Older workers + newer
    # parents stay compatible.
    assert conversion.figures == []


@pytest.mark.asyncio
async def test_docling_worker_figures_metadata_roundtrips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P3.3 Session 2: a worker that emits `figures: list[...]` in its
    payload must deserialise into `DoclingConversion.figures` as a
    list of `FigureMetadata` objects. The bbox carries through as a
    4-tuple; caption is optional.
    """
    from memex.parse.docling_backend import convert

    ok_script = tmp_path / "ok_with_figures.py"
    ok_script.write_text(
        'import json, sys\n'
        'payload = {\n'
        '    "markdown": "# Doc\\n\\n<!-- image -->\\n",\n'
        '    "pages": [{"page": 1, "markdown": "# Doc", "confidence": 0.9}],\n'
        '    "docling_version": "fake-2.0",\n'
        '    "figure_count": 2,\n'
        '    "table_count": 0,\n'
        '    "equation_count": 0,\n'
        '    "figures": [\n'
        '        {"page_no": 1, "bbox": [10.0, 20.0, 100.0, 80.0],\n'
        '         "caption": "Figure 1: Chart"},\n'
        '        {"page_no": 3, "bbox": [50.5, 60.0, 200.0, 150.0],\n'
        '         "caption": None},\n'
        '    ],\n'
        '}\n'
        'json.dump(payload, sys.stdout)\n',
        encoding="utf-8",
    )

    import memex.parse.docling_backend as backend

    real_spawn = asyncio.create_subprocess_exec

    async def _spawn(*args: Any, **kwargs: Any):
        new_args = (sys.executable, str(ok_script), args[-1])
        return await real_spawn(*new_args, **kwargs)

    monkeypatch.setattr(
        backend.asyncio, "create_subprocess_exec", _spawn
    )

    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n%%EOF\n")

    conversion = await convert(source, timeout_s=30)
    assert len(conversion.figures) == 2
    assert conversion.figures[0].page_no == 1
    assert conversion.figures[0].bbox == (10.0, 20.0, 100.0, 80.0)
    assert conversion.figures[0].caption == "Figure 1: Chart"
    assert conversion.figures[1].page_no == 3
    assert conversion.figures[1].caption is None
