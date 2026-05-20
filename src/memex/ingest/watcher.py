"""Filesystem watcher — debounced re-enrich + re-index on markdown edits.

Watchdog observes `vault/documents/*.md`. Events are debounced on a
per-path 500 ms timer (editors emit multiple events per save). After
debounce we compare the on-disk sha256 to the manifest's
`content_sha256`; matching means Memex wrote it (skip), differing
means a user edit (trigger).

Conflict policy (IMPLEMENTATION-PLAN §2.3): if a watcher event fires
during a Memex write to the same path, the post-write hash will match
and the event is swallowed. If a user edit lands during a re-enrich,
the re-enrich notices on save and aborts with `vault.edit_conflict`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import structlog
import ulid
from pydantic import BaseModel
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from memex.core.bus import get_bus
from memex.core.errors import ConfigurationError
from memex.core.events import MemexEvent
from memex.core.manifest import read_manifest
from memex.vault.store import hash_bytes

logger = structlog.get_logger(__name__)

_DEBOUNCE_S = 0.5


class EditNotice(BaseModel):
    """Surfaced to subscribers when a vault edit is confirmed."""

    doc_id: str
    markdown_path: str
    prev_sha256: str | None
    new_sha256: str


class _Handler(FileSystemEventHandler):
    """Forwards filesystem events into an asyncio queue."""

    def __init__(
        self, queue: asyncio.Queue[Path], loop: asyncio.AbstractEventLoop
    ) -> None:
        super().__init__()
        self._queue = queue
        self._loop = loop

    def _enqueue(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() != ".md":
            return
        # Only documents *directly* under `vault/documents/` count;
        # nested `vault/documents/{doc_id}/figures/foo.md` should not
        # masquerade as a vault edit.
        if path.parent.name != "documents":
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, path)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._enqueue(event)

    def on_created(self, event: FileSystemEvent) -> None:
        self._enqueue(event)


async def publish_edit_notice(notice: EditNotice) -> None:
    """Best-effort publish of a `vault.edited` event for a confirmed edit.

    Tries to fetch the process event bus and emit a `vault.edited`
    event. The publish is wrapped in a try/except so a misconfigured
    or failing bus never propagates into the watcher's `on_edit`
    callback — observability must not block the pipeline.

    Exposed as a module-level helper (rather than nested inside
    `run_watcher._drain_one`) so the integration test can drive it
    directly without spinning up watchdog.
    """
    log = logger.bind(doc_id=notice.doc_id)
    try:
        bus = get_bus()
    except ConfigurationError:
        return
    try:
        await bus.publish(
            MemexEvent(
                correlation_id=str(ulid.ULID()),
                stage="ingest",
                event_type="vault.edited",
                payload={
                    "doc_id": notice.doc_id,
                    "markdown_path": notice.markdown_path,
                    "prev_sha256": notice.prev_sha256 or "",
                    "new_sha256": notice.new_sha256,
                },
            )
        )
    except Exception as e:
        log.warning(
            "watcher.bus_publish_failed",
            doc_id=notice.doc_id,
            error=str(e),
        )


async def _confirm_user_edit(
    vault_path: Path, markdown_path: Path
) -> EditNotice | None:
    """Return an EditNotice iff the on-disk hash differs from the manifest's.

    Returns None when the file no longer exists, or when the hash
    matches (Memex's own write).
    """
    if not markdown_path.exists():
        return None
    doc_id = markdown_path.stem
    text = markdown_path.read_text(encoding="utf-8")
    new_sha = hash_bytes(text.encode("utf-8"))
    manifest = await read_manifest(vault_path, doc_id)
    if manifest is None:
        # Brand-new file dropped directly into vault/documents/ — treat as
        # a user edit so the index picks it up.
        return EditNotice(
            doc_id=doc_id,
            markdown_path=str(markdown_path),
            prev_sha256=None,
            new_sha256=new_sha,
        )
    if manifest.content_sha256 == new_sha:
        return None
    return EditNotice(
        doc_id=doc_id,
        markdown_path=str(markdown_path),
        prev_sha256=manifest.content_sha256,
        new_sha256=new_sha,
    )


async def run_watcher(
    vault_path: Path,
    *,
    on_edit: Callable[[EditNotice], Awaitable[None]],
    stop: asyncio.Event | None = None,
) -> None:
    """Run the watcher loop until `stop` is set (or forever).

    `on_edit` is awaited per confirmed user edit; production wiring
    runs `enrich_document` + `index_document` from it.
    """
    docs_dir = vault_path / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    queue: asyncio.Queue[Path] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    handler = _Handler(queue, loop)
    observer = Observer()
    observer.schedule(handler, str(docs_dir), recursive=False)
    stop_event = stop or asyncio.Event()
    log = logger.bind(vault_path=str(vault_path))

    # Wrap `observer.start()` in the try/finally so a watchdog start
    # failure (e.g. inotify watch limit hit) doesn't leak the thread
    # the underlying loop already spawned.
    try:
        observer.start()
    except Exception:
        try:
            observer.stop()
        except Exception:
            pass
        raise
    log.info("watcher.start")

    # Per-path debounce: latest-event timestamp for the path.
    last_seen: dict[Path, float] = {}
    # Strong refs to in-flight drain tasks so a) the GC doesn't collect
    # them (a CPython anti-pattern), and b) we can await + drain them
    # on shutdown rather than silently cancelling user-edit handling.
    drain_tasks: set[asyncio.Task[None]] = set()

    async def _drain_one(path: Path) -> None:
        # Wait the debounce window from the last seen time; if more events
        # arrive during this wait, the timer effectively resets because
        # we re-read `last_seen[path]` after sleeping.
        while True:
            seen_at = last_seen.get(path)
            if seen_at is None:
                return
            wait = _DEBOUNCE_S - (time.monotonic() - seen_at)
            if wait <= 0:
                break
            await asyncio.sleep(wait)
        last_seen.pop(path, None)
        notice = await _confirm_user_edit(vault_path, path)
        if notice is None:
            log.debug("watcher.no_change", path=str(path))
            return
        log.info("watcher.edit_confirmed", doc_id=notice.doc_id)

        # Bus publish is best-effort + exception-isolated (see
        # `publish_edit_notice`). User-edit handling proceeds even when
        # the bus is misconfigured or throwing.
        await publish_edit_notice(notice)

        try:
            await on_edit(notice)
        except Exception as e:
            log.warning(
                "watcher.handler_failed",
                doc_id=notice.doc_id,
                error=str(e),
            )

    try:
        while not stop_event.is_set():
            get_task = asyncio.create_task(queue.get())
            stop_task = asyncio.create_task(stop_event.wait())
            done, pending = await asyncio.wait(
                {get_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            if stop_task in done:
                break
            path = get_task.result()
            now = time.monotonic()
            already_pending = path in last_seen
            last_seen[path] = now
            if not already_pending:
                task = asyncio.create_task(_drain_one(path))
                drain_tasks.add(task)
                task.add_done_callback(drain_tasks.discard)
    finally:
        # Drain pending debounce tasks so a clean shutdown doesn't silently
        # drop a user edit that was mid-debounce. Bound the wait — a stuck
        # drain shouldn't block shutdown indefinitely.
        if drain_tasks:
            log.info("watcher.draining", pending=len(drain_tasks))
            try:
                await asyncio.wait_for(
                    asyncio.gather(*drain_tasks, return_exceptions=True),
                    timeout=2.0,
                )
            except TimeoutError:
                log.warning(
                    "watcher.drain_timeout",
                    still_pending=len(drain_tasks),
                )
                for t in list(drain_tasks):
                    t.cancel()
        observer.stop()
        observer.join(timeout=2.0)
        log.info("watcher.stop")


async def default_reaction(notice: EditNotice) -> None:
    """The default on_edit handler — re-enrich + re-index the edited doc.

    Imported lazily to avoid a parse-stage import cycle: enrich/index
    pull in heavy chunker + model deps that we don't need just to
    start the watcher.
    """
    from memex.enrich.pipeline import enrich_document
    from memex.index.pipeline import index_document

    await enrich_document(notice.doc_id)
    await index_document(notice.doc_id)
