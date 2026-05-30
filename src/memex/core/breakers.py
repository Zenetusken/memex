"""Lightweight circuit breakers for local resources — see GUIDELINES.md Part VI.

These aren't service-mesh breakers; they're a small counter-based
guard for predictable local failure modes (VRAM OOM, Docling
segfault) that we don't want to retry into oblivion.

A breaker is in one of three states:

- **closed** (healthy): requests pass through, failures increment the count.
- **open** (tripped): requests are rejected fast; no work attempted.
- **half_open** (probing): after `reset_after_s`, one request is allowed
  through; if it succeeds the breaker closes, if it fails it re-opens.

Trip threshold and reset window are configured per call site.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Generic, Literal, TypeVar

import structlog

from memex.core.errors import MemexError

logger = structlog.get_logger(__name__)

CircuitState = Literal["closed", "open", "half_open"]
T = TypeVar("T")


class CircuitBreakerOpen(MemexError):
    """The breaker is tripped; the call was rejected without being attempted."""


class CircuitBreaker(Generic[T]):
    """Counter + state machine.

    Use `await breaker.run(func)` to invoke a callable under the
    breaker. Failures are counted; threshold trips the breaker open.
    """

    def __init__(
        self,
        name: str,
        *,
        threshold: int = 3,
        reset_after_s: float = 60.0,
    ):
        self.name = name
        self.threshold = threshold
        self.reset_after_s = reset_after_s
        self._state: CircuitState = "closed"
        self._failures = 0
        self._opened_at: float | None = None
        self._probing = False  # True while the ONE half-open probe is in flight
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state — `"closed"` (healthy), `"open"`
        (tripped, fast-rejecting), or `"half_open"` (probing)."""
        return self._state

    @property
    def failures(self) -> int:
        """Number of consecutive failures since the breaker last
        closed. Resets to 0 on `_record_success()`."""
        return self._failures

    async def run(
        self,
        func: Callable[[], Awaitable[T]],
        *,
        is_failure: Callable[[BaseException], bool] | None = None,
    ) -> T:
        """Run `func` under the breaker.

        `is_failure` decides whether a raised exception should count as
        a breaker-relevant failure. Defaults to "every exception
        counts." Use it to ignore caller-level errors (validation,
        etc.) that don't reflect resource health.
        """
        is_prober = False
        async with self._lock:
            self._maybe_half_open()
            if self._state == "open":
                raise CircuitBreakerOpen(
                    f"circuit breaker {self.name!r} is open; refusing call",
                    context={
                        "breaker": self.name,
                        "failures": self._failures,
                        "opened_at": self._opened_at,
                    },
                )
            if self._state == "half_open":
                # Only ONE request probes the recovered resource; fast-reject the
                # rest (else N concurrent callers at the reset boundary all probe,
                # which can re-trigger the very OOM/segfault the breaker guards).
                if self._probing:
                    raise CircuitBreakerOpen(
                        f"circuit breaker {self.name!r} is half-open; a probe is already in flight",
                        context={
                            "breaker": self.name,
                            "failures": self._failures,
                            "opened_at": self._opened_at,
                        },
                    )
                self._probing = True
                is_prober = True

        try:
            try:
                result = await func()
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                # Cancellation, interrupt, or shutdown are not infrastructure
                # failures — never count them toward the trip threshold.
                raise
            except Exception as e:
                if is_failure is None or is_failure(e):
                    await self._record_failure()
                raise
            await self._record_success()
            return result
        finally:
            # Release the single-probe slot on EVERY exit (success / failure /
            # cancellation), so a cancelled probe can't wedge the breaker half-open.
            # `_probing` is only consulted in the half_open branch above, so a brief
            # stale-True window after a state change here is harmless.
            if is_prober:
                async with self._lock:
                    self._probing = False

    def _maybe_half_open(self) -> None:
        if (
            self._state == "open"
            and self._opened_at is not None
            and time.monotonic() - self._opened_at >= self.reset_after_s
        ):
            self._state = "half_open"
            logger.info("breaker.half_open", breaker=self.name)

    async def _record_failure(self) -> None:
        async with self._lock:
            self._failures += 1
            if self._state == "half_open" or self._failures >= self.threshold:
                self._state = "open"
                self._opened_at = time.monotonic()
                logger.warning(
                    "breaker.open",
                    breaker=self.name,
                    failures=self._failures,
                )

    async def _record_success(self) -> None:
        async with self._lock:
            if self._state == "half_open":
                logger.info("breaker.closed_after_probe", breaker=self.name)
            self._state = "closed"
            self._failures = 0
            self._opened_at = None

    def reset(self) -> None:
        """Force the breaker closed. For tests and `memex doctor`."""
        self._state = "closed"
        self._failures = 0
        self._opened_at = None
        self._probing = False
