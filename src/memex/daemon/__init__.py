"""vLLM daemon supervisor — see ADR-0001.

Spawns and tears down the out-of-process vLLM server that ADR-0001
commits to. State (PID, log) lives under `vault/.memex/daemon/`. The
supervisor is intentionally minimal:

- `start()` — runs `scripts/serve-vllm.sh` (configurable via
  `inference.serve_script`) in a detached subprocess, redirects stdout
  + stderr to a log file, writes the PID, then polls the configured
  `inference.base_url` until the endpoint becomes reachable or the
  startup timeout fires.
- `stop()` — reads the PID file, SIGTERMs the process group, waits up
  to 10 seconds, SIGKILLs on timeout, cleans up the PID file.
- `status()` — reports `{pid, alive, reachable, base_url, log_path}`.
"""

from memex.daemon.supervisor import (
    DaemonAlreadyRunning,
    DaemonStartTimeout,
    DaemonStatus,
    start,
    status,
    stop,
)

__all__ = [
    "DaemonAlreadyRunning",
    "DaemonStartTimeout",
    "DaemonStatus",
    "start",
    "status",
    "stop",
]
