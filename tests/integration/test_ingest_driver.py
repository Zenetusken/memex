"""Integration tests for `webui.ingest_driver` — the subprocess driver for browser
document ingestion. Uses the `spawn` seam to inject a fake process (canned stderr
structlog + a stdout `IngestResult`); NO real subprocess, NO GPU.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from memex.webui.ingest_driver import Spawn, run_enrich, run_ingest


def _reader(lines: list[dict[str, object] | bytes]) -> asyncio.StreamReader:
    """A StreamReader pre-fed with newline-terminated lines (dicts → JSON), EOF'd.
    Created inside the running loop (the fake spawn is async)."""
    r = asyncio.StreamReader()
    for ln in lines:
        data = ln if isinstance(ln, bytes) else json.dumps(ln).encode()
        r.feed_data(data + b"\n")
    r.feed_eof()
    return r


class _FakeProc:
    def __init__(
        self,
        *,
        stdout: list[dict[str, object] | bytes],
        stderr: list[dict[str, object] | bytes],
        returncode: int,
    ) -> None:
        self.pid = 0  # satisfies the _Process protocol; the cleanup path never runs for a fake
        self.stdout: asyncio.StreamReader | None = _reader(stdout)
        self.stderr: asyncio.StreamReader | None = _reader(stderr)
        self._rc = returncode

    async def wait(self) -> int:
        return self._rc


@dataclass
class _Captured:
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


def _fake_spawn(
    *,
    stdout: list[dict[str, object] | bytes],
    stderr: list[dict[str, object] | bytes],
    returncode: int,
) -> tuple[Spawn, _Captured]:
    cap = _Captured()

    async def spawn(*args: str, env: dict[str, str]) -> _FakeProc:
        cap.args = list(args)
        cap.env = dict(env)
        return _FakeProc(stdout=stdout, stderr=stderr, returncode=returncode)

    return spawn, cap


async def test_run_ingest_streams_phases_and_extracts_doc_id() -> None:
    stderr: list[dict[str, object] | bytes] = [
        {"event": "ingest.accepted", "doc_id": "abc12345-foo"},
        {"event": "parse.docling.start"},
        {"event": "vlm.start", "page": 7},
        {"event": "chart_ocr.start"},
        {"event": "index.start"},
        b"a stray non-json line",  # skipped defensively, never raises
    ]
    stdout: list[dict[str, object] | bytes] = [
        {"accepted": True, "doc_id": "abc12345-foo", "rejection_reason": None}
    ]
    phases: list[str] = []
    spawn, cap = _fake_spawn(stdout=stdout, stderr=stderr, returncode=0)

    outcome = await run_ingest(Path("sample-doc.pdf"), on_phase=phases.append, spawn=spawn)

    assert outcome.accepted
    assert outcome.succeeded
    assert outcome.doc_id == "abc12345-foo"
    assert outcome.rejection_reason is None
    # The deep-read detail rides as the eyebrow under "Transcribing"; the parse engine + index
    # map to their step. (No "Enriching" — that is the separate run_enrich subprocess.)
    assert phases == [
        "Parsing",
        "Transcribing · page 7",
        "Transcribing · chart OCR",
        "Indexing",
    ]
    assert cap.args[1:] == ["ingest", "sample-doc.pdf"]
    assert cap.env["MEMEX_OBSERVABILITY__LOG_JSON"] == "true"  # JSON structlog (parseable)
    assert cap.env["PYTHONUNBUFFERED"] == "1"  # live streaming, not one burst at exit
    assert cap.env["MEMEX_PARSE__DISABLE_VLM"] == "false"  # diagram/scan docs need the VLM


async def test_run_ingest_negative_exit_code_reports_signal() -> None:
    # A negative exit code is `-signum` — the OOM killer reaping the parse-time VLM is the common
    # case on the 12 GB rig. Surface that clearly instead of a cryptic "exited with code -9".
    spawn, _cap = _fake_spawn(stdout=[], stderr=[], returncode=-9)
    outcome = await run_ingest(Path("doc.pdf"), on_phase=lambda _p: None, spawn=spawn)
    assert not outcome.succeeded
    assert outcome.rejection_reason is not None
    assert "signal 9" in outcome.rejection_reason
    assert "memory" in outcome.rejection_reason


async def test_run_ingest_rejected_file_surfaces_reason_no_docid() -> None:
    stderr: list[dict[str, object] | bytes] = [
        {"event": "ingest.rejected", "reason": "unsupported file type"}
    ]
    stdout: list[dict[str, object] | bytes] = [
        {"accepted": False, "doc_id": None, "rejection_reason": "unsupported file type"}
    ]
    spawn, _cap = _fake_spawn(stdout=stdout, stderr=stderr, returncode=0)

    outcome = await run_ingest(Path("sample.bin"), on_phase=lambda _p: None, spawn=spawn)

    assert not outcome.accepted
    assert not outcome.succeeded
    assert outcome.doc_id is None
    assert outcome.rejection_reason == "unsupported file type"


async def test_run_ingest_half_doc_accepted_but_nonzero_exit() -> None:
    # Accepted + parsed (a doc_id) but the chain exited non-zero (e.g. an index OOM, the
    # Cours-10 case this session): a half-doc — browsable but NOT askable, not "succeeded".
    stderr: list[dict[str, object] | bytes] = [
        {"event": "ingest.accepted", "doc_id": "def67890-bar"},
        {"event": "parse.docling.start"},
    ]
    stdout: list[dict[str, object] | bytes] = [
        {"accepted": True, "doc_id": "def67890-bar", "rejection_reason": None}
    ]
    spawn, _cap = _fake_spawn(stdout=stdout, stderr=stderr, returncode=1)

    outcome = await run_ingest(Path("sample-doc.pdf"), on_phase=lambda _p: None, spawn=spawn)

    assert outcome.accepted
    assert outcome.exit_code == 1
    assert outcome.doc_id == "def67890-bar"  # the half-doc id is preserved
    assert not outcome.succeeded
    assert "exited with code 1" in (outcome.rejection_reason or "")


async def test_run_ingest_doc_id_falls_back_to_accepted_event() -> None:
    # If stdout carries no parseable IngestResult, doc_id comes from the stderr accepted event.
    stderr: list[dict[str, object] | bytes] = [
        {"event": "ingest.accepted", "doc_id": "early123-baz"},
        {"event": "index.start"},
    ]
    stdout: list[dict[str, object] | bytes] = [b"garbage not json"]
    spawn, _cap = _fake_spawn(stdout=stdout, stderr=stderr, returncode=0)

    outcome = await run_ingest(Path("sample-doc.pdf"), on_phase=lambda _p: None, spawn=spawn)

    assert outcome.doc_id == "early123-baz"
    assert outcome.accepted  # `accepted` falls back to the ingest.accepted stderr event too


async def test_run_enrich_streams_phase_and_returns_exit_code() -> None:
    stderr: list[dict[str, object] | bytes] = [
        {"event": "enrich.start"},
        {"event": "enrich.done"},
    ]
    phases: list[str] = []
    spawn, cap = _fake_spawn(stdout=[], stderr=stderr, returncode=0)

    rc = await run_enrich("abc12345-foo", on_phase=phases.append, spawn=spawn)

    assert rc == 0
    assert "Enriching" in phases
    assert cap.args[1:] == ["enrich", "abc12345-foo"]


async def test_run_enrich_nonzero_exit_returns_code() -> None:
    spawn, _cap = _fake_spawn(stdout=[], stderr=[{"event": "enrich.start"}], returncode=2)
    rc = await run_enrich("abc12345-foo", on_phase=lambda _p: None, spawn=spawn)
    assert rc == 2


async def test_run_ingest_concurrent_drain_survives_large_interleaved_output() -> None:
    # THE load-bearing contract (a fake can't fail it — it pre-buffers + EOFs both pipes): a
    # REAL child writes >64 KiB to BOTH stdout and stderr INTERLEAVED, so a sequential drain
    # (one pipe fully before the other) would deadlock on the OS pipe buffer. Also exercises
    # the raised StreamReader limit (each line exceeds the default 64 KiB). `wait_for` fails
    # the test if `run_ingest` hangs.
    script = (
        "import sys, json\n"
        "big = 'x' * 70000\n"
        "for i in range(4):\n"
        "    sys.stderr.write(json.dumps({'event': 'vlm.start', 'page': i, 'pad': big}) + '\\n')\n"
        "    sys.stderr.flush()\n"
        "    sys.stdout.write(big + '\\n'); sys.stdout.flush()\n"
        "sys.stdout.write(json.dumps({'accepted': True, 'doc_id': 'real1234-doc'}) + '\\n')\n"
    )

    async def spawn(*args: str, env: dict[str, str]) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=1024 * 1024,
        )

    phases: list[str] = []
    outcome = await asyncio.wait_for(
        run_ingest(Path("big.pdf"), on_phase=phases.append, spawn=spawn), timeout=20
    )
    assert outcome.doc_id == "real1234-doc"
    assert outcome.accepted
    assert any(p.startswith("Transcribing · page") for p in phases)


async def test_run_ingest_watchdog_kills_silent_hung_child(monkeypatch: pytest.MonkeyPatch) -> None:
    # A hung child that emits NO output must be SIGKILLed by the SILENCE watchdog — else the webui's
    # RAG lock would never release (answering permanently paused). Spawn a REAL silent sleeper, give
    # it a tiny silence budget, and confirm it's signal-killed with the "hung" rejection (distinct
    # from the OOM-signal message). The fast poll keeps the test sub-second; the outer `wait_for`
    # fails the test if the watchdog does NOT fire.
    monkeypatch.setattr("memex.webui.ingest_driver._WATCHDOG_POLL_S", 0.05)

    async def spawn(*_args: str, env: dict[str, str]) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(30)",  # no output, sleeps well past the budget
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # so _terminate's killpg reaps the whole group
        )

    outcome = await asyncio.wait_for(
        run_ingest(Path("hang.pdf"), on_phase=lambda _p: None, spawn=spawn, silence_timeout_s=0.1),
        timeout=10,
    )
    assert not outcome.accepted
    assert outcome.exit_code < 0  # killed by a signal (SIGKILL from the watchdog)
    assert "hung" in (outcome.rejection_reason or "")


async def test_run_ingest_watchdog_does_not_kill_active_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A child that keeps STREAMING output (even slowly) must NOT be killed: every line resets the
    # idle timer, so an active-but-slow ingest survives a tight silence budget. Three lines spaced
    # below the budget, then a clean result.
    monkeypatch.setattr("memex.webui.ingest_driver._WATCHDOG_POLL_S", 0.02)
    script = (
        "import sys, json, time\n"
        "for i in range(3):\n"
        "    sys.stderr.write(json.dumps({'event': 'vlm.start', 'page': i}) + '\\n')\n"
        "    sys.stderr.flush(); time.sleep(0.05)\n"
        "sys.stdout.write(json.dumps({'accepted': True, 'doc_id': 'alive123-doc'}) + '\\n')\n"
    )

    async def spawn(*_args: str, env: dict[str, str]) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

    outcome = await asyncio.wait_for(
        run_ingest(Path("slow.pdf"), on_phase=lambda _p: None, spawn=spawn, silence_timeout_s=0.15),
        timeout=10,
    )
    assert outcome.accepted and outcome.doc_id == "alive123-doc"  # streamed → never tripped
