"""Short-lived vLLM server for the summarizer swap-in (ADR-0010).

When `ModelSettings.summarizer` is set, a `report`-detail summary serves a STRONGER model
(e.g. Gemma-3-12B-it-AWQ) briefly on the GPU freed by `pause_vllm_for_gpu` (orchestrator
down), routes the whole map-reduce there via `models.client.inference_override`, then tears
it down and the orchestrator is restored. The motivation: the 8B re-states a doc's thesis
in every report paragraph and won't dedup via prompting (a behavioral limit, ADR-0010
§Refinements); a 12B follows the "don't repeat" instruction far better.

This is the TEXT-ONLY twin of the parse-time VLM lifecycle
(`parse/vlm_backend.py::_serve_vlm_vllm`). The load-bearing spawn discipline is copied
faithfully (rather than shared, to avoid coupling to the parse path): the process-group id
is captured AT SPAWN, and teardown `killpg`s THAT captured gid — never a re-`getpgid` of a
possibly-dead launcher (the bug that orphaned a GPU-holding EngineCore, 2026-05-26) — then
waits for the group to empty so the EngineCore's CUDA context is released before the GPU is
handed back to the orchestrator.
"""

from __future__ import annotations

import asyncio
import os
import signal
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import structlog

from memex.core.config import SummarizerServeSettings, get_settings
from memex.core.errors import ModelCallError

logger = structlog.get_logger(__name__)


async def _reachable(url: str, timeout_s: float = 2.0) -> bool:
    """True iff a GET to `url` returns 2xx — the readiness probe."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(url)
            return 200 <= resp.status_code < 300
    except Exception:
        return False


def _tail(path: Path, n_chars: int) -> str:
    """Last `n_chars` of a file (surfaces a failed subprocess's stderr)."""
    try:
        return path.read_bytes()[-n_chars:].decode("utf-8", errors="replace")
    except OSError:
        return ""


async def _reap(proc: asyncio.subprocess.Process, gid: int) -> None:
    """SIGTERM→SIGKILL the captured process group, then wait until the group is EMPTY
    (`killpg(gid, 0)` raises `ProcessLookupError`) so the EngineCore's CUDA context is fully
    released before the GPU is handed back. Port-quiet does NOT guarantee VRAM release —
    group-emptiness does."""
    with suppress(ProcessLookupError):
        os.killpg(gid, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=30.0)
    except TimeoutError:
        with suppress(ProcessLookupError):
            os.killpg(gid, signal.SIGKILL)
    for _ in range(45):
        try:
            os.killpg(gid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(1.0)
    with suppress(ProcessLookupError):
        os.killpg(gid, signal.SIGKILL)


@asynccontextmanager
async def serve_summarizer_vllm(model_id: str) -> AsyncGenerator[str]:
    """Start a short-lived TEXT vLLM serving `model_id`, yield its base_url, tear down.

    The caller MUST already hold `pause_vllm_for_gpu()` (orchestrator down → GPU free) so the
    12B has room. Retries startup once after a GPU settle (a transient
    "free memory < desired" under desktop-graphics peak). Raises `ModelCallError` if it can't
    start after both attempts (the caller decides whether to fail or fall back)."""
    serve: SummarizerServeSettings = get_settings().models.summarizer_serve
    base_url = f"http://{serve.host}:{serve.port}/v1"
    log = logger.bind(component="summarizer.vllm", model=model_id, port=serve.port)
    cmd = [
        "uv",
        "run",
        "--extra",
        "serve",
        "vllm",
        "serve",
        model_id,
        "--host",
        serve.host,
        "--port",
        str(serve.port),
        "--gpu-memory-utilization",
        str(serve.gpu_memory_utilization),
        "--max-model-len",
        str(serve.max_model_len),
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        str(serve.max_model_len),
        "--enforce-eager",
        "--kv-cache-dtype",
        "auto",
    ]
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    errlog_path = Path(tempfile.gettempdir()) / "memex_summarizer_vllm_serve.log"
    proc: asyncio.subprocess.Process | None = None
    gid = 0
    attempts = 2
    for attempt in range(1, attempts + 1):
        errlog = errlog_path.open("wb")
        log.info("summarizer.vllm.start", attempt=attempt)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=errlog,
            stderr=errlog,
            stdin=asyncio.subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
        # Capture the gid NOW, while the launcher is alive (a failed startup exits it,
        # after which os.getpgid raises and the EngineCore child would orphan).
        gid = os.getpgid(proc.pid)
        ready = False
        exited = False
        for _ in range(serve.startup_timeout_s):
            if await _reachable(f"{base_url}/models"):
                ready = True
                break
            if proc.returncode is not None:
                exited = True
                break
            await asyncio.sleep(1.0)
        errlog.close()
        if ready:
            break
        tail = _tail(errlog_path, 1500)
        await _reap(proc, gid)
        proc = None
        log.warning(
            "summarizer.vllm.start_failed", attempt=attempt, exited_early=exited, stderr_tail=tail
        )
        if attempt < attempts:
            await asyncio.sleep(5.0)  # let the GPU settle, then retry once
            continue
        raise ModelCallError(
            "summarizer vLLM failed to start",
            context={"model": model_id, "attempts": attempts, "stderr_tail": tail},
        )
    if proc is None:  # for the type checker: the loop either broke ready or raised
        raise ModelCallError("summarizer vLLM failed to start", context={"model": model_id})
    try:
        log.info("summarizer.vllm.ready")
        yield base_url
    finally:
        log.info("summarizer.vllm.stop")
        await _reap(proc, gid)
