"""Process sandboxing — see GUIDELINES.md Part VI 'Sandboxed parsing'.

Applies a seccomp-bpf filter that blocks every network syscall in the
current process. The Docling worker (`memex.parse.docling_worker`)
calls this *before* importing the docling library so any attempt by
docling — or anything it transitively imports — to phone home (DNS
lookup, model download, telemetry beacon) fails at the kernel level
with `EPERM`. Stacks the existing out-of-process Docling isolation
that ships via `docling_backend.convert`.

Linux-only. On macOS / Windows / BSD this module returns `SKIPPED`
with a reason and the worker continues unsandboxed. The user opts
out via `ParseSettings.docling_sandbox_network=False` if they have a
specialised setup that genuinely needs network during parse.

Once a seccomp filter is loaded into a process, it cannot be
removed — that's the kernel's contract. Children inherit the filter
on fork+exec.
"""

from __future__ import annotations

import errno
import sys
from typing import Literal

import structlog

logger = structlog.get_logger(__name__)

SandboxStatus = Literal["applied", "skipped", "failed"]


# Network-creation primitives. Blocking `socket` alone is enough to
# stop all egress (without a socket, no `connect`/`sendto`/...), but
# we list the others as defence in depth and to make the policy
# explicit in the audit trail.
_BLOCKED_SYSCALLS = (
    "socket",
    "socketpair",
    "connect",
    "bind",
    "sendto",
    "sendmsg",
    "sendmmsg",
)


def enable_network_block() -> tuple[SandboxStatus, str]:
    """Install a seccomp-bpf filter blocking network syscalls.

    Returns `(status, reason)`. On `"applied"`, the calling process
    can no longer create or connect sockets. On `"skipped"`, the
    sandbox isn't available on this platform / install. On
    `"failed"`, the filter library is present but the kernel
    refused the filter (rare — typically means seccomp is
    disabled in the kernel config).
    """
    if sys.platform != "linux":
        return ("skipped", f"non-linux platform: {sys.platform}")

    try:
        import pyseccomp as seccomp
    except ImportError as e:
        return ("skipped", f"pyseccomp not installed: {e}")

    try:
        f = seccomp.SyscallFilter(defaction=seccomp.ALLOW)
        applied_to: list[str] = []
        for name in _BLOCKED_SYSCALLS:
            try:
                f.add_rule(seccomp.ERRNO(errno.EPERM), name)
                applied_to.append(name)
            except (KeyError, OSError, ValueError):
                # Some syscalls don't exist on every architecture; the
                # filter rejects unknown names. Skip silently — `socket`
                # exists everywhere and is the load-bearing block.
                pass
        f.load()
    except Exception as e:
        logger.warning("sandbox.failed", reason=str(e))
        return ("failed", f"seccomp load failed: {e}")

    logger.info("sandbox.applied", blocked=applied_to)
    return ("applied", f"blocked: {','.join(applied_to)}")
