"""In-process progress registry for the webui's live ``/ask`` indicator.

The answering agent runs in a background task; this registry holds the CURRENT
phase per request (keyed by ``correlation_id``) so the long-poll status endpoint
can report it. **Event-driven** so the indicator mimics real-time SSE:
``set_phase`` / ``finish`` wake a held poll immediately (sub-100 ms), with a
monotonic ``version`` + a clear-before-await loop so the poll never misses a
change or busy-loops. Single-worker uvicorn → a plain in-process dict is safe
(no locking; all access is on the one event loop).

Lives in ``webui/`` (not ``core/``) because phase labels are a presentation
concern. The agent stays oblivious — it only emits node names via the opt-in
``answer_query(on_node=…)`` sink; the node→phase mapping is here.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from memex.agents.answering import FinalResponse
from memex.agents.bridge import BridgeAnswer
from memex.agents.expert import ExpertAnswer

# The ordered user-facing phases — the step list the UI renders. Every agent
# node maps to exactly one of these via `_NODE_PHASES`.
PHASES: tuple[str, ...] = (
    "Retrieving",
    "Reranking",
    "Assessing",
    "Drafting",
    "Grounding",
    "Checking relevance",
    "Composing",
)

# Agent node name → user-facing phase. Several fast nodes fold into one phase
# (e.g. retrieve/resolve_artifact_scope/expand_graph → "Retrieving") so the UI
# shows meaningful steps, not internal plumbing.
_NODE_PHASES: dict[str, str] = {
    "retrieve": "Retrieving",
    "resolve_artifact_scope": "Retrieving",
    "expand_graph": "Retrieving",
    "rerank": "Reranking",
    "query_tables": "Reranking",
    "assess": "Assessing",
    "answer": "Drafting",
    "regenerate": "Drafting",
    "verify": "Grounding",
    "assess_relevance": "Checking relevance",
    "compose": "Composing",
    "refuse": "Composing",
}

_TTL_SECONDS = 300.0  # reap abandoned (browser-closed) entries after this
_MAX_ENTRIES = 256  # hard cap — a leak-stop (single-user localhost)
_KEEPALIVE_S = 1.0  # long-poll hold before a heartbeat return (ticks the elapsed timer)


def phase_for(node: str) -> str:
    """Map an agent node name → its phase label. An unknown node (a future graph
    change) returns ``""`` so the caller keeps the current phase rather than
    blanking it."""
    return _NODE_PHASES.get(node, "")


# The summarizer's phases (it's linear, not a graph — `summarize_document` calls
# its `on_phase` sink directly). "Key figures" + the per-section counter ride under
# "Summarizing" as the eyebrow detail (see `summary_phase_view`), so the dominant
# phase shows live progress without a step that mostly sits still.
SUMMARY_PHASES: tuple[str, ...] = ("Reading", "Summarizing", "Reducing", "Composing")


def summary_phase_view(label: str) -> tuple[int, str]:
    """Split a summarizer phase label into ``(active-step index, eyebrow detail)``.
    `summarize_document` emits e.g. ``"Summarizing · section 3 of 9"`` or
    ``"Reducing"``; the base maps to a `SUMMARY_PHASES` step, the ``" · "`` suffix
    (if any) is the live detail (``"section 3 of 9"`` / ``"key figures"`` / ``""``).
    An unknown base falls back to step 0."""
    base, _, detail = label.partition(" · ")
    try:
        return SUMMARY_PHASES.index(base), detail
    except ValueError:
        return 0, detail


# The expert surface's phases (Surface B, ADR-0013 — linear, like the summarizer:
# `expert_answer` calls its `on_phase` sink directly with these exact labels).
EXPERT_PHASES: tuple[str, ...] = ("Retrieving evidence", "Reasoning")


def expert_phase_index(label: str) -> int:
    """Map an expert phase label → its index in ``EXPERT_PHASES`` (unknown → 0)."""
    try:
        return EXPERT_PHASES.index(label)
    except ValueError:
        return 0


# The reason-then-ground bridge's phases (Surface §11 — linear, like expert mode plus a
# trailing grounding step: `reason_then_ground` emits these exact labels via its `on_phase`).
BRIDGE_PHASES: tuple[str, ...] = ("Retrieving evidence", "Reasoning", "Grounding claims")


def bridge_phase_index(label: str) -> int:
    """Map a bridge phase label → its index in ``BRIDGE_PHASES`` (unknown → 0)."""
    try:
        return BRIDGE_PHASES.index(label)
    except ValueError:
        return 0


@dataclass
class ProgressEntry:
    """The live state of one in-flight ``/ask``, keyed by ``correlation_id``."""

    scope_doc_ids: list[str]  # always supplied by ProgressRegistry.new()
    scope_source: str
    question: str = ""  # the original /ask question — carried for the consented A→B escalation (§11)
    phase: str = PHASES[0]
    version: int = 0
    started_at: float = field(default_factory=time.monotonic)
    phase_started_at: float = field(default_factory=time.monotonic)
    done: bool = False
    # FinalResponse for /ask, summarize, and chat; ExpertAnswer for the ungrounded expert
    # surface (Surface B); BridgeAnswer for the reason-then-ground bridge (§11). The status
    # route knows which it launched.
    response: FinalResponse | ExpertAnswer | BridgeAnswer | None = None
    error: str | None = None
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None

    def phase_elapsed_s(self) -> int:
        """Whole seconds the CURRENT phase has been active (the live timer)."""
        return int(time.monotonic() - self.phase_started_at)

    def active_index(self) -> int:
        """Index of the current phase in ``PHASES`` (for done/active styling).
        Falls back to 0 if the phase isn't a known label."""
        try:
            return PHASES.index(self.phase)
        except ValueError:
            return 0


class ProgressRegistry:
    """``correlation_id → ProgressEntry`` for in-flight asks. Lazy cleanup (TTL
    sweep + size cap on ``new``); the status route evicts on delivery. One
    instance per app (a ``create_app`` local, like ``mode_switch_lock``)."""

    def __init__(self) -> None:
        self._entries: dict[str, ProgressEntry] = {}

    def new(
        self, cid: str, *, scope_doc_ids: list[str], scope_source: str, question: str = ""
    ) -> ProgressEntry:
        self._sweep()
        entry = ProgressEntry(
            scope_doc_ids=list(scope_doc_ids), scope_source=scope_source, question=question
        )
        self._entries[cid] = entry
        return entry

    def set_phase(self, cid: str, phase: str) -> None:
        """Advance the entry's phase (no-op if the cid is gone, the phase is
        empty, or it's unchanged — so same-phase node runs don't churn the UI)."""
        entry = self._entries.get(cid)
        if entry is None or not phase or phase == entry.phase:
            return
        entry.phase = phase
        entry.phase_started_at = time.monotonic()
        self._bump(entry)

    def finish(
        self,
        cid: str,
        *,
        response: FinalResponse | ExpertAnswer | BridgeAnswer | None = None,
        error: str | None = None,
    ) -> None:
        entry = self._entries.get(cid)
        if entry is None:
            return
        entry.done = True
        entry.response = response
        entry.error = error
        self._bump(entry)

    def attach_task(self, cid: str, task: asyncio.Task[None]) -> None:
        """Hold a strong ref to the background task so the event loop doesn't GC
        it mid-run (the documented fire-and-forget pitfall)."""
        entry = self._entries.get(cid)
        if entry is not None:
            entry.task = task

    def get(self, cid: str) -> ProgressEntry | None:
        return self._entries.get(cid)

    def evict(self, cid: str) -> None:
        self._entries.pop(cid, None)

    async def wait_for_change(
        self, cid: str, last_seen: int, *, keepalive: float = _KEEPALIVE_S
    ) -> ProgressEntry | None:
        """Block until the entry advances past ``last_seen``, finishes, or
        ``keepalive`` seconds elapse (a heartbeat). Returns the entry (``None``
        if the cid is unknown). Lost-wakeup-safe: the version is re-checked after
        ``changed.clear()`` and again would be caught by a set() during the await
        (set-after-clear leaves the event set → ``wait`` returns at once)."""
        entry = self._entries.get(cid)
        if entry is None:
            return None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + keepalive
        while not entry.done and entry.version <= last_seen:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            entry.changed.clear()
            if entry.done or entry.version > last_seen:  # a bump during clear()
                break
            try:
                await asyncio.wait_for(entry.changed.wait(), timeout=remaining)
            except TimeoutError:
                break
        return entry

    @staticmethod
    def _bump(entry: ProgressEntry) -> None:
        # Monotonic version + wake any waiter. NO clear() here — the waiter
        # clears before re-awaiting, so a set() that lands between its clear and
        # its await is not lost (the event stays set → wait returns immediately).
        entry.version += 1
        entry.changed.set()

    def _sweep(self) -> None:
        now = time.monotonic()
        for cid in [c for c, e in self._entries.items() if now - e.started_at > _TTL_SECONDS]:
            self._entries.pop(cid, None)
        overflow = len(self._entries) - _MAX_ENTRIES
        if overflow > 0:
            # Drop oldest-first, done entries before still-running ones.
            ordered = sorted(
                self._entries.items(), key=lambda kv: (not kv[1].done, kv[1].started_at)
            )
            for cid, _ in ordered[:overflow]:
                self._entries.pop(cid, None)
