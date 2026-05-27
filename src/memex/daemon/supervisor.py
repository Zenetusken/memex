"""vLLM subprocess supervisor.

Reasonable defaults; no third-party process manager (systemd, supervisord,
etc.) needed. Idempotent: starting a daemon that's already running
errors loudly rather than silently spawning a duplicate.
"""

from __future__ import annotations

import asyncio
import errno
import os
import signal
import subprocess
import time
from pathlib import Path

import structlog
from pydantic import BaseModel

from memex.core.config import MemexSettings
from memex.core.errors import ConfigurationError, MemexError

logger = structlog.get_logger(__name__)

_STOP_GRACE_S = 10.0
_REACHABILITY_POLL_S = 1.0


class DaemonAlreadyRunning(MemexError):
    """A daemon PID file exists and the process is still alive."""


class DaemonStartTimeout(MemexError):
    """The vLLM endpoint didn't become reachable within the timeout."""


class DaemonStatus(BaseModel):
    """Snapshot of the vLLM daemon's process + HTTP reachability state,
    used by `memex daemon status` and the doctor report."""

    pid: int | None
    alive: bool
    reachable: bool
    base_url: str
    pid_file: str
    log_file: str
    error: str | None = None


def _daemon_dir(settings: MemexSettings) -> Path:
    d = settings.vault_path / ".memex" / "daemon"
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    return d


def _pid_file(settings: MemexSettings) -> Path:
    return _daemon_dir(settings) / "vllm.pid"


def _log_file(settings: MemexSettings) -> Path:
    return _daemon_dir(settings) / "vllm.log"


def _pid_alive(pid: int) -> bool:
    """True if `pid` names a running process. POSIX-only — `kill(pid, 0)`.

    N5 (audit 2026-05-20): the EPERM branch is **intentional** and
    correct. `os.kill(pid, 0)` raises:
    - `ESRCH` — no such process. We return False.
    - `EPERM` — process exists but is owned by another uid (or is in
      a different PID namespace). The process IS alive; we just can't
      signal it. Returning True is correct: `_pid_alive` answers
      "does the kernel show a process with this PID", not "can we
      kill it". The downstream `stop()` path acquires its own EPERM
      from the real signal call and bails out cleanly.
    - Anything else — defensive False; an unrecognised errno from
      kill(pid, 0) is more likely a bad PID than a live process.

    When EPERM fires we emit a debug-level structlog event so an
    operator chasing "memex says vLLM is running but I don't see it"
    can spot the cross-uid mismatch. Not a warning — EPERM is a
    legitimate state when the user re-installs Memex under a
    different account or runs inside a container with its own uid.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as e:
        if e.errno == errno.ESRCH:  # no such process
            return False
        if e.errno == errno.EPERM:
            logger.debug(
                "pid_alive.eperm",
                pid=pid,
                note="process exists but is owned by another uid",
            )
            return True
        # Unrecognised errno — be conservative: report not-alive so
        # the supervisor doesn't deadlock against a phantom PID.
        logger.warning(
            "pid_alive.unknown_errno",
            pid=pid,
            errno=e.errno,
        )
        return False
    return True


def _pid_owns_script(pid: int, script: Path) -> bool:
    """Verify the running PID's command line still references our script.

    Guards against PID-reuse: the daemon may have crashed and a new
    process may have inherited the PID. `stop()` reads `/proc/<pid>/cmdline`
    (Linux) and confirms the script path appears there before signalling.
    On non-Linux systems (or if /proc isn't mounted) we skip the check
    and return True — preserving prior behaviour for those platforms.
    """
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not cmdline_path.exists():
        return True  # non-Linux or no procfs; can't verify
    try:
        raw = cmdline_path.read_bytes()
    except OSError:
        return True
    # /proc/PID/cmdline is NUL-separated argv. Look for the script name.
    args = raw.split(b"\x00")
    needle = str(script).encode("utf-8")
    needle_basename = script.name.encode("utf-8")
    return any(needle in a or needle_basename in a for a in args)


def _resolve_script(settings: MemexSettings) -> Path:
    """Resolve `inference.serve_script` to an absolute path."""
    p = settings.inference.serve_script
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


async def _reachable(settings: MemexSettings) -> bool:
    """Probe the configured base_url for an OpenAI-compatible endpoint.

    "Not reachable" can mean many things: connect refused (daemon not
    up yet), DNS failure (bad base_url), HTTP 401 (bad api_key). We
    log the type/message at debug so `--verbose` start runs can tell a
    misconfiguration ("connection refused on every poll") from a slow
    warm-up ("got HTTPError on first probe, then ok").
    """
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url=settings.inference.base_url,
            api_key=settings.inference.api_key,
            timeout=2.0,
        )
        await client.models.list()
        return True
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.debug(
            "daemon.reachable.failed",
            error_type=type(e).__name__,
            error=str(e)[:200],
        )
        return False


async def start(settings: MemexSettings) -> DaemonStatus:
    """Spawn vLLM and wait for the endpoint to become reachable.

    Raises `DaemonAlreadyRunning` if the existing PID is still alive.
    Raises `DaemonStartTimeout` if `inference.daemon_startup_timeout_s`
    elapses without reachability.

    The script's stdout + stderr are redirected to a log under
    `vault/.memex/daemon/vllm.log`. The process is started in a new
    session so it survives the calling CLI's exit.
    """
    pid_file = _pid_file(settings)
    log_file = _log_file(settings)

    # Refuse to spawn over an existing live daemon.
    if pid_file.exists():
        try:
            existing = int(pid_file.read_text(encoding="utf-8").strip())
        except ValueError:
            existing = 0
        if existing > 0 and _pid_alive(existing):
            raise DaemonAlreadyRunning(
                f"vllm daemon already running with pid {existing}",
                context={
                    "pid": existing,
                    "pid_file": str(pid_file),
                    "fix": "stop the existing daemon with `memex daemon stop`",
                },
            )
        # Stale PID file — clean up and continue.
        pid_file.unlink()

    script = _resolve_script(settings)
    if not script.exists():
        raise ConfigurationError(
            f"serve-vllm script not found at {script}",
            context={
                "fix": "set inference.serve_script in MemexSettings or place "
                "scripts/serve-vllm.sh in the working directory",
                "script": str(script),
            },
        )

    log = logger.bind(script=str(script), pid_file=str(pid_file))
    log.info("daemon.start.spawning")

    # Open the log file for the child's stdout/stderr. Append so the
    # log persists across restarts; the user can rotate by hand.
    # Blocking open() + Popen are deliberate here: the fd is handed to a
    # DETACHED long-lived child (not loop-managed I/O), so the async
    # subprocess/aiofiles machinery would be wrong, not just unnecessary.
    log_fp = open(log_file, "ab")  # noqa: ASYNC230
    try:
        proc = subprocess.Popen(  # noqa: S603, ASYNC220 — config-sourced script path; detached child
            ["/usr/bin/env", "bash", str(script)],
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach so memex CLI can exit cleanly
            close_fds=True,
        )
    except Exception:
        log_fp.close()
        raise

    # Don't keep the file handle open in the parent; the child inherited
    # the fd via fork.
    log_fp.close()

    pid_file.write_text(str(proc.pid), encoding="utf-8")
    log.info("daemon.start.pid_written", pid=proc.pid)

    # Wait for reachability OR child exit OR timeout.
    deadline = time.monotonic() + settings.inference.daemon_startup_timeout_s
    while time.monotonic() < deadline:
        # Did the child exit early? (likely a config error in serve-vllm.sh)
        if proc.poll() is not None:
            pid_file.unlink(missing_ok=True)
            raise MemexError(
                f"vllm daemon exited with code {proc.returncode} before becoming reachable",
                context={
                    "exit_code": proc.returncode,
                    "log_file": str(log_file),
                    "hint": "inspect the log for the underlying error",
                },
            )
        if await _reachable(settings):
            log.info("daemon.start.reachable", pid=proc.pid)
            return DaemonStatus(
                pid=proc.pid,
                alive=True,
                reachable=True,
                base_url=settings.inference.base_url,
                pid_file=str(pid_file),
                log_file=str(log_file),
            )
        await asyncio.sleep(_REACHABILITY_POLL_S)

    # Timed out. Kill and clean up. Mirror stop() with SIGTERM → grace
    # → SIGKILL escalation so a daemon that ignores SIGTERM during torch
    # init (rare but documented in vLLM issues) doesn't leak.
    log.warning("daemon.start.timeout", pid=proc.pid)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    sigkill_deadline = time.monotonic() + _STOP_GRACE_S
    while time.monotonic() < sigkill_deadline:
        if not _pid_alive(proc.pid):
            break
        await asyncio.sleep(0.2)
    if _pid_alive(proc.pid):
        log.warning("daemon.start.escalating_to_sigkill", pid=proc.pid)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    pid_file.unlink(missing_ok=True)
    raise DaemonStartTimeout(
        f"vllm daemon did not become reachable within "
        f"{settings.inference.daemon_startup_timeout_s}s",
        context={
            "pid": proc.pid,
            "base_url": settings.inference.base_url,
            "log_file": str(log_file),
        },
    )


def stop(settings: MemexSettings) -> DaemonStatus:
    """Tear down the daemon. Idempotent — no-op if not running."""
    pid_file = _pid_file(settings)
    log_file = _log_file(settings)

    if not pid_file.exists():
        return DaemonStatus(
            pid=None,
            alive=False,
            reachable=False,
            base_url=settings.inference.base_url,
            pid_file=str(pid_file),
            log_file=str(log_file),
            error="no PID file; nothing to stop",
        )

    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        pid_file.unlink(missing_ok=True)
        return DaemonStatus(
            pid=None,
            alive=False,
            reachable=False,
            base_url=settings.inference.base_url,
            pid_file=str(pid_file),
            log_file=str(log_file),
            error="malformed PID file",
        )

    log = logger.bind(pid=pid)

    # PID-reuse guard: confirm the live PID still belongs to *our*
    # daemon before signalling. Without this, a crashed daemon whose
    # PID has been recycled (common with default Linux pid_max=32768
    # on long-uptime hosts) would have us SIGKILL an unrelated process.
    script = _resolve_script(settings)
    if not _pid_owns_script(pid, script):
        log.warning(
            "daemon.stop.pid_reused_clearing_file",
            pid=pid,
            script=str(script),
        )
        pid_file.unlink(missing_ok=True)
        return DaemonStatus(
            pid=None,
            alive=False,
            reachable=False,
            base_url=settings.inference.base_url,
            pid_file=str(pid_file),
            log_file=str(log_file),
            error=f"PID {pid} no longer references this daemon's script",
        )

    log.info("daemon.stop.signaling")

    # SIGTERM the whole process group (the script spawned children too).
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        # Process already gone; treat as success.
        pid_file.unlink(missing_ok=True)
        return DaemonStatus(
            pid=None,
            alive=False,
            reachable=False,
            base_url=settings.inference.base_url,
            pid_file=str(pid_file),
            log_file=str(log_file),
        )

    # Wait up to `_STOP_GRACE_S` for graceful exit.
    deadline = time.monotonic() + _STOP_GRACE_S
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            break
        time.sleep(0.2)
    else:
        # Still alive — escalate to SIGKILL.
        log.warning("daemon.stop.escalating_to_sigkill")
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    pid_file.unlink(missing_ok=True)
    return DaemonStatus(
        pid=None,
        alive=False,
        reachable=False,
        base_url=settings.inference.base_url,
        pid_file=str(pid_file),
        log_file=str(log_file),
    )


async def status(settings: MemexSettings) -> DaemonStatus:
    """Report current daemon state without side-effects."""
    pid_file = _pid_file(settings)
    log_file = _log_file(settings)
    pid: int | None = None
    alive = False

    if pid_file.exists():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            alive = _pid_alive(pid)
        except ValueError:
            pid = None

    return DaemonStatus(
        pid=pid,
        alive=alive,
        reachable=await _reachable(settings),
        base_url=settings.inference.base_url,
        pid_file=str(pid_file),
        log_file=str(log_file),
    )


async def restart(
    settings: MemexSettings,
    *,
    gpu_fraction: float | None = None,
    max_model_len: int | None = None,
) -> DaemonStatus:
    """Stop the daemon (if running), optionally re-point the orchestrator's
    GPU-utilization / context window, then start a fresh one.

    `gpu_fraction` → `MEMEX_VLLM_GPU_FRACTION`, `max_model_len` →
    `MEMEX_VLLM_MAX_MODEL_LEN`; both are read by `scripts/serve-vllm.sh` at
    launch and the spawned child inherits THIS process's environment. This is
    how a co-residence mode's orchestrator posture is applied (ADR-0007).

    Precondition: the daemon must be ours (a PID file under
    `vault/.memex/daemon/`). A hand-launched `serve-vllm.sh` has no PID file,
    so `stop` is a no-op and `start` would then refuse on the still-bound port
    — launch the orchestrator via `memex daemon start` for restart to work.
    `stop` runs in a thread (it blocks up to the SIGTERM grace window).
    """
    await asyncio.to_thread(stop, settings)
    if gpu_fraction is not None:
        os.environ["MEMEX_VLLM_GPU_FRACTION"] = f"{gpu_fraction:g}"
    if max_model_len is not None:
        os.environ["MEMEX_VLLM_MAX_MODEL_LEN"] = str(max_model_len)
    return await start(settings)
