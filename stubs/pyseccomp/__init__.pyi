"""Minimal type stubs for the surface of `pyseccomp` used by Memex.

Only the symbols referenced by `memex.parse.sandbox` are declared. The
library is otherwise untyped (no `py.typed`); see ADR / GUIDELINES Part VI
for why the sandbox is network-syscall-blocking only.
"""

from typing import Final

# Default actions exposed as module-level constants by libseccomp.
ALLOW: Final[int]
KILL: Final[int]
KILL_PROCESS: Final[int]
TRAP: Final[int]
LOG: Final[int]
NOTIFY: Final[int]

class ERRNO:
    """Action that makes a matched syscall fail with the given errno."""

    def __init__(self, errno: int) -> None: ...

class SyscallFilter:
    """A seccomp-bpf syscall filter."""

    def __init__(self, defaction: int) -> None: ...
    def add_rule(self, action: ERRNO | int, syscall: int | str) -> None: ...
    def load(self) -> None: ...
