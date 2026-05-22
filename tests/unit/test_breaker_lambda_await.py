"""Filler N8 — verify `CircuitBreaker.run` correctly awaits the
coroutine returned by a lambda wrapper.

The Docling parser path uses this pattern:

    breaker.run(
        lambda: docling_convert(source, timeout_s=...),
        is_failure=_is_docling_failure,
    )

The audit (2026-05-20) flagged that this returns a coroutine from
the lambda and asked to verify the breaker's `await func()` semantics
work correctly. The implementation in `core/breakers.py:94` is:

    result = await func()

When `func()` is called, the lambda runs, which calls
`docling_convert(...)`. Since `docling_convert` is `async def`,
calling it returns a coroutine object (not a value). `await` on that
coroutine awaits it. This is the correct pattern.

These tests pin the contract so a future refactor of the breaker or
the call sites doesn't accidentally regress to the broken form
(`breaker.run(docling_convert(...))` — calling once before passing,
which raises the coroutine outside the breaker's exception handling).
"""

from __future__ import annotations

import asyncio

import pytest

from memex.core.breakers import CircuitBreaker, CircuitBreakerOpen


@pytest.mark.asyncio
async def test_breaker_awaits_lambda_returning_coroutine() -> None:
    """The lambda-returns-coroutine pattern used by docling_backend
    must produce the coroutine's awaited value, not the coroutine
    object itself."""

    async def _async_work(value: int) -> int:
        await asyncio.sleep(0)
        return value * 2

    breaker: CircuitBreaker[int] = CircuitBreaker(name="test", threshold=3)

    # Production pattern: lambda wraps the async call so the breaker
    # owns the coroutine creation + await.
    result = await breaker.run(lambda: _async_work(21))

    assert result == 42, (
        "breaker returned coroutine or wrong value — `await func()` "
        "in CircuitBreaker.run did not unwrap the lambda's awaitable"
    )


@pytest.mark.asyncio
async def test_breaker_records_failure_when_lambda_async_call_raises() -> None:
    """If the async callable raises, the breaker must record it as a
    failure and re-raise. The lambda + await pattern must propagate
    the original exception, not swallow it.
    """

    class _Boom(Exception):
        pass

    async def _async_raiser() -> int:
        await asyncio.sleep(0)
        raise _Boom("expected")

    breaker: CircuitBreaker[int] = CircuitBreaker(name="test", threshold=2)
    assert breaker.failures == 0

    with pytest.raises(_Boom):
        await breaker.run(lambda: _async_raiser())

    assert breaker.failures == 1
    assert breaker.state == "closed"  # one failure, not yet at threshold


@pytest.mark.asyncio
async def test_breaker_trips_open_after_threshold_lambda_failures() -> None:
    """N consecutive failures via the lambda pattern must trip the
    breaker open at the configured threshold. Subsequent calls
    short-circuit with CircuitBreakerOpen without awaiting the
    lambda's coroutine.
    """

    class _Boom(Exception):
        pass

    async def _async_raiser() -> None:
        raise _Boom()

    breaker: CircuitBreaker[None] = CircuitBreaker(
        name="test", threshold=3
    )

    for _ in range(3):
        with pytest.raises(_Boom):
            await breaker.run(lambda: _async_raiser())

    assert breaker.state == "open"

    # Next call must short-circuit — the lambda should NOT be invoked.
    call_count = 0

    async def _tracker() -> None:
        nonlocal call_count
        call_count += 1

    with pytest.raises(CircuitBreakerOpen):
        await breaker.run(lambda: _tracker())
    assert call_count == 0, (
        "breaker open but still invoked the lambda — short-circuit broken"
    )


@pytest.mark.asyncio
async def test_breaker_lambda_is_called_per_run_not_once() -> None:
    """Each `breaker.run(lambda: _async_call())` must invoke the lambda
    afresh, producing a new coroutine. Reusing a coroutine across two
    awaits would raise RuntimeError: cannot reuse already awaited
    coroutine. The lambda pattern is what defers coroutine creation
    until inside the breaker.
    """
    call_count = 0

    async def _work() -> int:
        nonlocal call_count
        call_count += 1
        return call_count

    breaker: CircuitBreaker[int] = CircuitBreaker(name="test", threshold=10)

    for expected in range(1, 6):
        result = await breaker.run(lambda: _work())
        assert result == expected
    assert call_count == 5


@pytest.mark.asyncio
async def test_breaker_is_failure_predicate_filters_lambda_exceptions() -> None:
    """The `is_failure` predicate must see the actual exception type
    raised by the async callable (not a coroutine wrapper). Used in
    docling_backend to distinguish DoclingCrashed/DoclingTimeout
    (count as breaker failure) from caller-validation errors (don't
    count).
    """

    class _CallerError(Exception):
        pass

    class _InfraError(Exception):
        pass

    async def _raiser(exc: Exception) -> None:
        raise exc

    breaker: CircuitBreaker[None] = CircuitBreaker(
        name="test", threshold=3
    )

    def _is_infra(e: BaseException) -> bool:
        return isinstance(e, _InfraError)

    # Caller error — predicate says "not infra" — failure count stays 0.
    with pytest.raises(_CallerError):
        await breaker.run(
            lambda: _raiser(_CallerError()),
            is_failure=_is_infra,
        )
    assert breaker.failures == 0

    # Infra error — predicate says "yes infra" — failure count goes up.
    with pytest.raises(_InfraError):
        await breaker.run(
            lambda: _raiser(_InfraError()),
            is_failure=_is_infra,
        )
    assert breaker.failures == 1
