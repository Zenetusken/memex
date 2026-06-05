"""Unit tests for the headless web UI uploader (`scripts/webui_ingest.py`).

The helper drives the real `POST /ingest` route and follows `_progress.html` → `_ingest_done.html`.
To stay faithful (and catch template drift), the fragment parsers are exercised against the ACTUAL
webui templates rendered by jinja2 — the same HTML the route emits — not hand-written look-alikes.
The script is a `scripts/` dev tool, so it's loaded via importlib (not a package import).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import jinja2
import pytest

_REPO = Path(__file__).resolve().parents[2]
_TEMPLATES = _REPO / "src" / "memex" / "webui" / "templates"
_INGEST_PHASES = ["Parsing", "Transcribing", "Indexing", "Enriching"]


def _load_helper() -> ModuleType:
    path = _REPO / "scripts" / "webui_ingest.py"
    spec = importlib.util.spec_from_file_location("webui_ingest_helper", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: the @dataclass decorator resolves annotations (under `from __future__
    # import annotations`) via sys.modules[cls.__module__], which must exist first.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def helper() -> ModuleType:
    return _load_helper()


@pytest.fixture(scope="module")
def env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATES)), autoescape=jinja2.select_autoescape()
    )


def _progress(env: jinja2.Environment, *, cid: str, version: int, active: int, detail: str) -> str:
    return env.get_template("_progress.html").render(
        poll_url=f"/ingest/{cid}/status?v={version}",
        phases=_INGEST_PHASES,
        active_index=active,
        elapsed=14,
        detail=detail,
    )


def _done(env: jinja2.Environment, *, doc_id: str | None, error: str | None) -> str:
    return env.get_template("_ingest_done.html").render(doc_id=doc_id, error=error)


# --- extract_status_url: the load-bearing progress-vs-terminal discriminator ---


def test_extract_status_url_from_progress(helper: ModuleType, env: jinja2.Environment) -> None:
    html = _progress(env, cid="01TESTCID", version=3, active=1, detail="page 2")
    assert helper.extract_status_url(html) == "/ingest/01TESTCID/status?v=3"


def test_extract_status_url_none_on_done(helper: ModuleType, env: jinja2.Environment) -> None:
    assert helper.extract_status_url(_done(env, doc_id="abc123-doc", error=None)) is None


def test_extract_status_url_none_on_ingesting_lock(
    helper: ModuleType, env: jinja2.Environment
) -> None:
    # _ingesting.html polls /ingest/lock — NOT a /status poll. Must read as terminal (busy), else
    # the helper would loop on the lock endpoint forever.
    html = env.get_template("_ingesting.html").render()
    assert "/ingest/lock" in html
    assert helper.extract_status_url(html) is None


def test_extract_status_url_none_on_expired(helper: ModuleType, env: jinja2.Environment) -> None:
    assert helper.extract_status_url(env.get_template("_progress_expired.html").render()) is None


# --- classify_terminal: every terminal branch the route can return ---


def test_classify_ingested(helper: ModuleType, env: jinja2.Environment) -> None:
    out = helper.classify_terminal(_done(env, doc_id="ae210f22-cisco-cyberops", error=None))
    assert out.status == "ingested"
    assert out.doc_id == "ae210f22-cisco-cyberops"
    assert out.ok is True


def test_classify_partial(helper: ModuleType, env: jinja2.Environment) -> None:
    out = helper.classify_terminal(_done(env, doc_id="abc123-doc", error="enrich failed: boom"))
    assert out.status == "partial"
    assert out.doc_id == "abc123-doc"
    assert out.ok is True  # browsable → still a success


def test_classify_failed_no_doc(helper: ModuleType, env: jinja2.Environment) -> None:
    out = helper.classify_terminal(_done(env, doc_id=None, error="file is empty"))
    assert out.status == "failed"
    assert out.doc_id is None
    assert out.ok is False
    assert "empty" in out.message


def test_classify_no_file_error(helper: ModuleType, env: jinja2.Environment) -> None:
    # The route's missing-file path renders _ingest_done.html with doc_id=None + an error.
    out = helper.classify_terminal(
        _done(env, doc_id=None, error="No file was provided — choose a file to upload.")
    )
    assert out.status == "failed"
    assert out.ok is False


def test_classify_busy(helper: ModuleType, env: jinja2.Environment) -> None:
    out = helper.classify_terminal(env.get_template("_ingesting.html").render())
    assert out.status == "busy"
    assert out.doc_id is None
    assert out.ok is False


def test_classify_expired(helper: ModuleType, env: jinja2.Environment) -> None:
    out = helper.classify_terminal(env.get_template("_progress_expired.html").render())
    assert out.status == "expired"
    assert out.ok is False


# --- progress_label: the human phase line ---


def test_progress_label_active_step_and_detail(helper: ModuleType, env: jinja2.Environment) -> None:
    html = _progress(env, cid="c", version=1, active=1, detail="page 2")
    assert helper.progress_label(html) == "Transcribing · page 2"


def test_progress_label_no_detail_is_just_step(helper: ModuleType, env: jinja2.Environment) -> None:
    html = _progress(env, cid="c", version=0, active=0, detail="")
    assert helper.progress_label(html) == "Parsing"


# --- multipart_body: a real multipart/form-data part ---


def test_multipart_body_framing(helper: ModuleType) -> None:
    body = helper.multipart_body("file", "note.md", b"hello\nworld", "BOUND123")
    assert body.startswith(b"--BOUND123\r\n")
    assert b'Content-Disposition: form-data; name="file"; filename="note.md"' in body
    assert b"\r\n\r\nhello\nworld\r\n" in body  # blank line then the verbatim content
    assert body.endswith(b"--BOUND123--\r\n")


def test_outcome_ok_matrix(helper: ModuleType) -> None:
    assert helper.Outcome("ingested", "d", "").ok is True
    assert helper.Outcome("partial", "d", "").ok is True
    for bad in ("failed", "busy", "expired", "unexpected", "timeout", "unreachable"):
        assert helper.Outcome(bad, None, "").ok is False


def test_run_unreachable_webui_is_clean(helper: ModuleType, tmp_path: Path) -> None:
    # No server on this port → a clean 'unreachable' Outcome, not a traceback.
    f = tmp_path / "x.txt"
    f.write_text("hi")
    out = helper.run("http://127.0.0.1:9", f, timeout=2.0)
    assert out.status == "unreachable"
    assert out.ok is False
