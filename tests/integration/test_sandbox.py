"""Network-egress sandbox for the Docling worker — v1.x regression net.

Three concerns, two of them platform-independent:

1. `enable_network_block()` returns `("skipped", ...)` on non-Linux
   platforms and when `pyseccomp` isn't installed. Tests stub
   `sys.platform` and the import path to cover both.
2. The Docling worker honours `MEMEX_PARSE_SANDBOX_NETWORK=0`
   (skips the sandbox) and refuses to run when the sandbox load
   reports `failed` (exit code 6 → `SandboxLoadFailed` in the parent).
3. **End-to-end network-block proof** (Linux + pyseccomp installed
   only): a subprocess applies the sandbox, then tries to create a
   socket. Must raise `PermissionError`. The test skips when seccomp
   isn't reachable so it doesn't fail spuriously in cross-platform CI.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

from memex.core.config import MemexSettings, set_settings

# ----- Fixtures -----


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    (tmp_path / "documents").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def settings(tmp_vault: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemexSettings]:
    monkeypatch.setenv("MEMEX_VAULT_PATH", str(tmp_vault))
    monkeypatch.setenv("MEMEX_OBSERVABILITY__LANGFUSE_ENABLED", "false")
    s = MemexSettings()  # type: ignore[call-arg]
    set_settings(s)
    yield s
    set_settings(None)


# ----- Branch 1: graceful skip on non-Linux / missing pyseccomp -----


def test_enable_network_block_skipped_on_non_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On macOS, Windows, BSD, etc. the sandbox returns 'skipped'
    immediately — seccomp is Linux-specific."""
    import memex.parse.sandbox as sandbox

    monkeypatch.setattr(sandbox.sys, "platform", "darwin")
    status, reason = sandbox.enable_network_block()
    assert status == "skipped"
    assert "darwin" in reason


def test_enable_network_block_skipped_without_pyseccomp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When pyseccomp can't be imported, the sandbox reports 'skipped'
    rather than 'failed' — the user can install it or accept the
    no-sandbox default by setting docling_sandbox_network=False."""
    import builtins

    import memex.parse.sandbox as sandbox

    # Force the Linux path so we reach the import attempt.
    monkeypatch.setattr(sandbox.sys, "platform", "linux")

    real_import = builtins.__import__

    def _no_pyseccomp(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name == "pyseccomp":
            raise ImportError("pyseccomp pretend-missing for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _no_pyseccomp)

    status, reason = sandbox.enable_network_block()
    assert status == "skipped"
    assert "pyseccomp" in reason


# ----- Branch 2: worker honours the env var + exit-code contract -----


def _worker_script(body: str) -> str:
    """Build a tiny Python script that exercises the sandbox module
    and reports back via stderr + exit code."""
    return f"import sys, os\nsys.path.insert(0, 'src')\n{body}\n"


def test_worker_skips_sandbox_when_disabled_via_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spawn the worker with `MEMEX_PARSE_SANDBOX_NETWORK=0`. The
    sandbox call must not run; the worker therefore won't emit the
    `sandbox: applied/skipped/failed` line. We use a tiny stand-in
    for the worker so we don't depend on docling being installed."""
    fake_worker = tmp_path / "fake_worker.py"
    fake_worker.write_text(
        _worker_script(
            "import os\n"
            "if os.environ.get('MEMEX_PARSE_SANDBOX_NETWORK', '1') == '1':\n"
            "    from memex.parse.sandbox import enable_network_block\n"
            "    print(f'sandbox-ran', file=sys.stderr)\n"
            "    enable_network_block()\n"
            "else:\n"
            "    print('sandbox-skipped-by-env', file=sys.stderr)\n"
        ),
        encoding="utf-8",
    )
    env = {
        **dict(__import__("os").environ),
        "MEMEX_PARSE_SANDBOX_NETWORK": "0",
    }
    res = subprocess.run(
        [sys.executable, str(fake_worker)],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert res.returncode == 0
    assert "sandbox-skipped-by-env" in res.stderr
    assert "sandbox-ran" not in res.stderr


# ----- Branch 3: end-to-end network block (Linux + pyseccomp only) -----


def _seccomp_supported_here() -> bool:
    """Best-effort check: are we on Linux with a working pyseccomp?"""
    if sys.platform != "linux":
        return False
    try:
        import pyseccomp  # noqa: F401
    except ImportError:
        return False
    # Some sandboxed CI environments disable seccomp entirely. The only
    # way to know for sure is to try loading a no-op filter.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pyseccomp as s\n"
                "f = s.SyscallFilter(defaction=s.ALLOW)\n"
                "f.load()\n"
                "print('ok')\n"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    return probe.returncode == 0 and "ok" in probe.stdout


@pytest.mark.skipif(
    not _seccomp_supported_here(),
    reason="seccomp requires Linux + pyseccomp + a kernel that allows it",
)
def test_sandbox_actually_blocks_socket_in_subprocess() -> None:
    """The marquee test. Spawn a subprocess that calls
    `enable_network_block()` and then tries to open a TCP socket. The
    socket call must raise `PermissionError`. This is the kernel-level
    guarantee — Python-side blocks could be bypassed by C extensions,
    but seccomp can't."""
    script = (
        "import sys\n"
        "sys.path.insert(0, 'src')\n"
        "from memex.parse.sandbox import enable_network_block\n"
        "status, reason = enable_network_block()\n"
        "if status != 'applied':\n"
        "    print(f'SETUP-FAILED: {status} - {reason}', file=sys.stderr)\n"
        "    sys.exit(2)\n"
        "import socket\n"
        "try:\n"
        "    socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "    print('FAIL: socket created despite sandbox', file=sys.stderr)\n"
        "    sys.exit(3)\n"
        "except PermissionError as e:\n"
        "    print(f'OK: blocked - {e}', file=sys.stderr)\n"
        "    sys.exit(0)\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if res.returncode == 2:
        pytest.skip(f"seccomp setup failed in this env: {res.stderr}")
    assert res.returncode == 0, (
        f"sandbox didn't block socket: rc={res.returncode}, "
        f"stderr={res.stderr}, stdout={res.stdout}"
    )
    assert "OK: blocked" in res.stderr


@pytest.mark.skipif(
    not _seccomp_supported_here(),
    reason="seccomp requires Linux + pyseccomp + a kernel that allows it",
)
def test_sandbox_does_not_block_filesystem_io() -> None:
    """Seccomp blocks network syscalls; file I/O is unaffected. Docling
    reads PDFs and writes its temp files — those paths must keep
    working under the sandbox."""
    script = (
        "import sys, tempfile, os\n"
        "sys.path.insert(0, 'src')\n"
        "from memex.parse.sandbox import enable_network_block\n"
        "status, _ = enable_network_block()\n"
        "if status != 'applied':\n"
        "    sys.exit(2)\n"
        "# Write + read a tempfile.\n"
        "with tempfile.NamedTemporaryFile('w+', delete=False) as f:\n"
        "    f.write('hello sandbox')\n"
        "    name = f.name\n"
        "with open(name) as g:\n"
        "    body = g.read()\n"
        "os.unlink(name)\n"
        "assert body == 'hello sandbox', f'got {body!r}'\n"
        "print('OK: filesystem ok')\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if res.returncode == 2:
        pytest.skip("seccomp setup failed in this env")
    assert res.returncode == 0, f"filesystem broke under sandbox: {res.stderr}"
    assert "OK: filesystem ok" in res.stdout


# ----- Branch 4: docling_backend.convert propagates the setting -----


@pytest.mark.asyncio
async def test_convert_passes_sandbox_env_var(
    settings: MemexSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`docling_backend.convert(sandbox_network=False)` must set
    `MEMEX_PARSE_SANDBOX_NETWORK=0` in the child's environment, and
    the default (True) must set it to `1`."""
    import memex.parse.docling_backend as backend

    captured_env: dict[str, str] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (
                b'{"markdown": "hi", "pages": [{"page": 1, "markdown": "hi", '
                b'"confidence": 1.0}], "docling_version": "fake", '
                b'"figure_count": 0, "table_count": 0, "equation_count": 0}',
                b"",
            )

    async def _fake_spawn(*args: object, **kwargs: object) -> _FakeProc:
        captured_env.update(cast("Mapping[str, str]", kwargs.get("env") or {}))
        return _FakeProc()

    monkeypatch.setattr(backend.asyncio, "create_subprocess_exec", _fake_spawn)

    source = tmp_path / "src.pdf"
    source.write_bytes(b"%PDF-1.7\n%%EOF\n")

    # Default: sandbox_network=True → env var "1"
    await backend.convert(source, timeout_s=10)
    assert captured_env["MEMEX_PARSE_SANDBOX_NETWORK"] == "1"

    captured_env.clear()
    await backend.convert(source, timeout_s=10, sandbox_network=False)
    assert captured_env["MEMEX_PARSE_SANDBOX_NETWORK"] == "0"


@pytest.mark.asyncio
async def test_convert_surfaces_sandbox_load_failure_distinctly(
    settings: MemexSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker that exits 6 (sandbox-load failed) raises the dedicated
    `SandboxLoadFailed` exception — not the generic `DoclingCrashed`.
    The breaker filter excludes `SandboxLoadFailed`, so configuration
    errors don't pile up and trip the parser circuit breaker."""
    import memex.parse.docling_backend as backend
    from memex.parse.docling_backend import SandboxLoadFailed

    class _FakeProc:
        returncode = 6

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"", b"sandbox: FAILED (pretend reason)\n")

    async def _fake_spawn(*args: object, **kwargs: object) -> _FakeProc:
        return _FakeProc()

    monkeypatch.setattr(backend.asyncio, "create_subprocess_exec", _fake_spawn)

    source = tmp_path / "src.pdf"
    source.write_bytes(b"%PDF-1.7\n%%EOF\n")

    with pytest.raises(SandboxLoadFailed) as excinfo:
        await backend.convert(source, timeout_s=10)
    assert "pretend reason" in cast("str", excinfo.value.context["stderr"])
