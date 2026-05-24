"""Unit tests for the MCP bearer-token auth middleware + bind validator."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from memex.core.errors import ConfigurationError
from memex.mcp.auth import (
    BearerAuthMiddleware,
    is_loopback,
    validate_bind,
)

# ----- is_loopback -----


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "::1", "localhost", "Localhost", "127.0.0.7"],
)
def test_is_loopback_recognises_loopback_forms(host: str) -> None:
    assert is_loopback(host) is True


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "192.168.1.5", "203.0.113.1", "10.0.0.1", "::"],
)
def test_is_loopback_rejects_non_loopback(host: str) -> None:
    assert is_loopback(host) is False


@pytest.mark.parametrize("host", ["", "not-an-ip", "example.com"])
def test_is_loopback_rejects_hostnames_and_garbage(host: str) -> None:
    assert is_loopback(host) is False


# ----- validate_bind -----


def test_validate_bind_allows_loopback_without_token() -> None:
    # No raise on loopback even when has_token=False.
    for host in ("127.0.0.1", "::1", "localhost"):
        validate_bind(host, has_token=False)


def test_validate_bind_refuses_non_loopback_without_token() -> None:
    for host in ("0.0.0.0", "192.168.1.5", "203.0.113.1"):
        with pytest.raises(ConfigurationError) as exc:
            validate_bind(host, has_token=False)
        assert host in str(exc.value.context["host"])
        assert "MEMEX_MCP__AUTH_TOKEN" in exc.value.context["fix"]


def test_validate_bind_allows_any_host_with_token() -> None:
    for host in ("0.0.0.0", "192.168.1.5", "203.0.113.1"):
        validate_bind(host, has_token=True)


# ----- BearerAuthMiddleware -----


def _ok(_request: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


@pytest.fixture
def app() -> Iterator[TestClient]:
    """A tiny Starlette app guarded by BearerAuthMiddleware."""
    base = Starlette(routes=[Route("/probe", _ok)])
    base.add_middleware(BearerAuthMiddleware, expected_token="s3cret-test-token")
    with TestClient(base) as client:
        yield client


def test_middleware_accepts_correct_token(app: TestClient) -> None:
    resp = app.get(
        "/probe",
        headers={"Authorization": "Bearer s3cret-test-token"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_middleware_rejects_missing_header(app: TestClient) -> None:
    resp = app.get("/probe")
    assert resp.status_code == 401
    assert resp.headers["WWW-Authenticate"].startswith("Bearer")
    body = resp.json()
    assert body["error"] == "unauthorized"
    assert body["reason"] == "missing_or_malformed"


def test_middleware_rejects_wrong_token(app: TestClient) -> None:
    resp = app.get(
        "/probe",
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401
    assert resp.json()["reason"] == "invalid_token"


def test_middleware_rejects_wrong_scheme(app: TestClient) -> None:
    """Authorization: Basic <…> should not be accepted as a bearer token."""
    resp = app.get(
        "/probe",
        headers={"Authorization": "Basic s3cret-test-token"},
    )
    assert resp.status_code == 401
    assert resp.json()["reason"] == "missing_or_malformed"


def test_middleware_constructor_rejects_empty_token() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        BearerAuthMiddleware(app=None, expected_token="")


def test_middleware_never_logs_the_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failed auth attempt must not leak either the expected or
    submitted token into log records. The expected token sits in
    middleware state; the submitted token comes from the request.
    """
    base = Starlette(routes=[Route("/probe", _ok)])
    expected = "expected-secret-do-not-log"
    base.add_middleware(BearerAuthMiddleware, expected_token=expected)
    submitted = "submitted-secret-do-not-log"
    with caplog.at_level(logging.DEBUG), TestClient(base) as client:
        client.get("/probe", headers={"Authorization": f"Bearer {submitted}"})
        client.get("/probe")  # missing header
        client.get(
            "/probe",
            headers={"Authorization": f"Bearer {expected}"},
        )  # success
    for record in caplog.records:
        msg = record.getMessage()
        assert expected not in msg, f"expected token leaked: {msg!r}"
        assert submitted not in msg, f"submitted token leaked: {msg!r}"
