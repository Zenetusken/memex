"""Subprocess driver for browser-initiated document ingestion (the exclusive-GPU
ingestion mode — `docs/ROADMAP.md` "frictionless UI ingestion").

The module boundary forbids ``webui/`` from importing ``parse``/``index``/``enrich``/
``ingest`` (see `src/memex/CLAUDE.md`). The CLI ingest is already a fresh-process model
that pauses the orchestrator + serves the VLM, so the webui drives it as a CHILD
SUBPROCESS — one ``memex ingest <file>`` per file (also the fix for the multi-doc VLM
``VRAMExhausted``: the per-doc chart-OCR pass restarts the orchestrator between docs),
then a separate ``memex enrich <doc_id>``. This stays strictly inside the ``webui → core``
edge (only ``asyncio.create_subprocess_exec`` + ``json`` + ``core``/``webui.progress``),
adds zero new boundary exceptions, and gives the fresh-process CUDA semantics the
pipeline assumes.

**Stream contract (load-bearing):** structlog milestones go to **stderr**
(`observability/logging.py` — `PrintLoggerFactory(file=sys.stderr)`), while the
`IngestResult` JSON lands on **stdout** only at exit (`cli/commands._print` on a pipe).
Both pipes are drained CONCURRENTLY (`asyncio.gather`) — awaiting one while the other
fills its OS pipe buffer would deadlock the child. The subprocess env forces JSON
structlog + unbuffered output so progress streams live.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from pydantic import BaseModel

from memex.webui.progress import ingest_phase_for

# A phase sink — `progress.set_phase(cid, label)` bound to a cid by the caller.
OnPhase = Callable[[str], None]

# 2 GiB — the media-ingest ceiling (ADR-0017); passed to the subprocess since the real
# magic-based validation + cap live in `ingest/` (boundary-blocked from the webui).
_MAX_BYTES = 2 * 1024 * 1024 * 1024

# The subprocess pipe line-buffer cap. structlog serializes a full exception traceback as ONE
# physical line (`format_exc_info`), so a deep CUDA/docling traceback on the failure path can
# exceed asyncio's default 64 KiB StreamReader limit and raise ValueError mid-drain. Raise it
# generously; `_pump` additionally SKIPS an over-limit line rather than crash the drain.
_PIPE_LIMIT = 1024 * 1024


class _Process(Protocol):
    """The subset of ``asyncio.subprocess.Process`` the driver uses (so a fake process
    can be injected via the ``spawn`` seam in tests)."""

    pid: int
    stdout: asyncio.StreamReader | None
    stderr: asyncio.StreamReader | None

    async def wait(self) -> int: ...


class Spawn(Protocol):
    """The subprocess-spawn seam. The default spawns the real CLI; tests inject a fake
    that returns a `_Process` with canned stdout/stderr."""

    async def __call__(self, *args: str, env: dict[str, str]) -> _Process: ...


class IngestOutcome(BaseModel):
    """The result of one ``memex ingest`` subprocess run."""

    accepted: bool  # the file passed validation (the ingest stage's verdict)
    exit_code: int  # the whole chain's exit (parse/index failure ⇒ non-zero even if accepted)
    doc_id: str | None = None
    rejection_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        """True iff the file was accepted, the chain exited cleanly, and a doc_id was
        assigned — i.e. the doc reached at least parsed+indexed. A `doc_id` WITHOUT
        `succeeded` is a half-doc (accepted+parsed but index failed — the Cours-10 case)."""
        return self.accepted and self.exit_code == 0 and self.doc_id is not None


async def _default_spawn(*args: str, env: dict[str, str]) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        *args,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=_PIPE_LIMIT,
        # Own session/process group, so a cancelled or cut-short ingest can kill the WHOLE tree —
        # the `memex ingest` child AND the parse-time vLLM / chart-OCR it spawned — not just the
        # launcher (which would orphan a GPU-holding process). See `_terminate`.
        start_new_session=True,
    )


async def _terminate(proc: _Process) -> None:
    """SIGKILL an unfinished child's whole process GROUP so a cancelled (webui shutdown) or
    cut-short (a sink raised mid-drain) ingest never orphans a GPU-holding tree. The child is its
    own group leader (`start_new_session=True`); `killpg` reaps the parse-time vLLM with it. Every
    step is guarded — a fake test process (no real pid) or an already-dead group is a clean no-op.
    `killpg` is synchronous so it lands even while the task is unwinding a `CancelledError`."""
    with suppress(Exception):
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    with suppress(Exception):
        await proc.wait()  # reap the zombie (best-effort)


def _memex_bin() -> str:
    """The same-venv ``memex`` console script (no `uv` re-resolution overhead — the
    webui already runs inside the venv)."""
    return str(Path(sys.executable).parent / "memex")


def _build_env(extra: dict[str, str] | None) -> dict[str, str]:
    env: dict[str, str] = {
        **os.environ,
        "MEMEX_OBSERVABILITY__LOG_JSON": "true",  # JSON structlog on stderr (parseable)
        "PYTHONUNBUFFERED": "1",  # else lines buffer → progress arrives in one burst at exit
        "MEMEX_INGEST__MAX_BYTES": str(_MAX_BYTES),
        "MEMEX_PARSE__DISABLE_VLM": "false",  # diagram/scan docs need VLM escalation (default True)
    }
    if extra:
        env.update(extra)
    return env


def _parse_json_line(line: str) -> dict[str, object] | None:
    """A structlog JSON line → dict, or None for a non-JSON / non-object line (skip
    defensively — a stray print or a format change degrades to less detail, never a crash)."""
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None
    return cast("dict[str, object]", rec) if isinstance(rec, dict) else None


def _str(value: object) -> str:
    return value if isinstance(value, str) else ""


async def _pump(reader: asyncio.StreamReader | None, sink: Callable[[str], None]) -> None:
    """Drain a pipe line-by-line, calling ``sink`` on each non-empty decoded line. An
    over-limit line (a serialized traceback beyond `_PIPE_LIMIT`) is SKIPPED, not raised —
    `readline()` raises ValueError and resets its buffer, so the drain continues."""
    if reader is None:
        return
    while True:
        try:
            raw = await reader.readline()
        except ValueError:
            continue  # over-limit line — buffer was reset; skip it rather than crash the drain
        if not raw:
            return  # EOF
        line = raw.decode("utf-8", "replace").strip()
        if line:
            sink(line)


@dataclass
class _Seen:
    """Mutable holder for facts captured off the stderr stream (the `ingest.accepted` event)
    — the fallback when the stdout `IngestResult` is missing/unparseable."""

    doc_id: str | None = None
    accepted: bool = False


def _make_stderr_sink(on_phase: OnPhase, seen: _Seen) -> Callable[[str], None]:
    """Build the stderr line handler: map each structlog event → a phase (push to
    `on_phase`) and capture the `ingest.accepted` doc_id + accepted flag early."""

    def sink(line: str) -> None:
        rec = _parse_json_line(line)
        if rec is None:
            return
        event = _str(rec.get("event"))
        if event == "ingest.accepted":
            seen.accepted = True
            did = rec.get("doc_id")
            if isinstance(did, str):
                seen.doc_id = did
        phase = ingest_phase_for(event, rec.get("page"))
        if phase:
            on_phase(phase)

    return sink


async def run_ingest(
    file_path: Path,
    *,
    on_phase: OnPhase,
    extra_env: dict[str, str] | None = None,
    spawn: Spawn = _default_spawn,
) -> IngestOutcome:
    """Run ``memex ingest <file>`` as a child subprocess, streaming its structlog phases
    to ``on_phase`` and returning the verdict. ``extra_env`` (e.g. `orchestrator_serve_env`)
    is merged over the base env so the subprocess's post-parse vLLM restart brings up the
    configured orchestrator, not the serve-script default (see ROADMAP decision D3)."""
    proc = await spawn(_memex_bin(), "ingest", str(file_path), env=_build_env(extra_env))
    stdout_lines: list[str] = []
    seen = _Seen()
    completed = False
    try:
        await asyncio.gather(
            _pump(proc.stderr, _make_stderr_sink(on_phase, seen)),
            _pump(proc.stdout, stdout_lines.append),
        )
        exit_code = await proc.wait()
        completed = True
    finally:
        if not completed:  # cancelled (shutdown) or a sink raised — don't orphan the GPU-holding tree
            await _terminate(proc)

    result = _last_ingest_result(stdout_lines)
    doc_id = (_str(result.get("doc_id")) or None) if result else None
    doc_id = doc_id or seen.doc_id
    # `accepted` falls back to the `ingest.accepted` stderr event when the stdout IngestResult
    # is missing/unparseable (symmetric with the doc_id fallback above).
    accepted = bool(result.get("accepted")) if result else seen.accepted
    rejection = (_str(result.get("rejection_reason")) or None) if result else None
    if rejection is None and exit_code != 0:
        # A negative exit code is `-signum` (the child was killed by a signal). The common case on
        # this 12 GB rig is the OOM killer reaping a parse-time VLM serve — surface that clearly
        # instead of a cryptic "exited with code -9".
        if exit_code < 0:
            rejection = (
                f"the ingest process was terminated by signal {-exit_code} "
                "(it may have run out of memory)"
            )
        else:
            rejection = f"the ingest process exited with code {exit_code}"
    return IngestOutcome(
        accepted=accepted, exit_code=exit_code, doc_id=doc_id, rejection_reason=rejection
    )


async def run_enrich(
    doc_id: str,
    *,
    on_phase: OnPhase,
    extra_env: dict[str, str] | None = None,
    spawn: Spawn = _default_spawn,
) -> int:
    """Run ``memex enrich <doc_id>`` as a child subprocess (entities + citations), streaming
    its phases. Returns the exit code. NB enrich needs the orchestrator UP for citations, so
    the caller gates the spawn on orchestrator reachability (ROADMAP Inc 4)."""
    proc = await spawn(_memex_bin(), "enrich", doc_id, env=_build_env(extra_env))

    def on_stderr(line: str) -> None:
        rec = _parse_json_line(line)
        if rec is not None:
            phase = ingest_phase_for(_str(rec.get("event")), rec.get("page"))
            if phase:
                on_phase(phase)

    completed = False
    try:
        await asyncio.gather(_pump(proc.stderr, on_stderr), _pump(proc.stdout, lambda _line: None))
        rc = await proc.wait()
        completed = True
        return rc
    finally:
        if not completed:  # cancelled / cut short — don't orphan the subprocess tree
            await _terminate(proc)


def _last_ingest_result(stdout_lines: list[str]) -> dict[str, object] | None:
    """The last stdout JSON object carrying an ``accepted`` key — the `IngestResult`
    `_print`ed on a pipe. (One file ⇒ one such line; the loop is robust to extra output.)"""
    found: dict[str, object] | None = None
    for line in stdout_lines:
        rec = _parse_json_line(line)
        if rec is not None and "accepted" in rec:
            found = rec
    return found
