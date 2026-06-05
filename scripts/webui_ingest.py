#!/usr/bin/env python3
"""Headless uploader for the Memex web UI ingestion route.

Drives the REAL `POST /ingest` multipart route exactly as the browser's HTMX form does — no
browser, no CORS static server, no in-browser XHR — then follows the long-poll progress
(`/ingest/{cid}/status`) to completion, printing each phase. This is the supported substitute
for the `claude-in-chrome` `file_upload` tool, which (as of Claude Code 2.1.x) deliberately
"no longer accepts host filesystem paths" — a closed-source contract change, not patchable here.

    uv run python3 scripts/webui_ingest.py <file_path> [--url URL] [--timeout S] [--json]

Exits 0 when the document is ingested (or partially ingested → still browsable), 1 on a
failure / rejection / timeout / unreachable web UI.

For the PURE-PIPELINE case (no web UI at all), use the CLI instead — it ingests a host file
directly with no browser and no server:

    uv run memex ingest <file_path>

Stdlib-only (no third-party deps) so it runs under a bare Python or `uv run`. The HTML-fragment
parsers are pure module-level functions, unit-tested in tests/unit/test_webui_ingest_helper.py.

NB the file is buffered in memory for the multipart body, so for a multi-GB upload (e.g. a long
video) prefer `memex ingest <file_path>` — the web UI route itself streams to disk, but this
client does not.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_URL = "http://127.0.0.1:7423"

# The progress fragment's self-poll target — specifically `/ingest/{cid}/status`. NB the
# RAG-paused `_ingesting.html` fragment polls `/ingest/lock` instead; that must NOT match (it is
# a terminal "another ingest is running" signal, not a status poll to follow).
_STATUS_URL_RE = re.compile(r'hx-get="(/ingest/[^"/]+/status[^"]*)"')
_EYEBROW_RE = re.compile(r'class="ans-eyebrow"\s*>\s*(.*?)\s*<', re.DOTALL)
_DOC_ID_RE = re.compile(r'href="/documents/([^"/?#]+)"')
# The terminal message line(s): `_ingest_done.html` (.ingest-done-note), `_progress_expired.html`
# (.ans-flash-msg), `_ingesting.html` (.ingesting-notice span).
_MESSAGE_RE = re.compile(
    r'class="(?:ingest-done-note|ans-flash-msg)"\s*>\s*(.*?)\s*</p>', re.DOTALL
)
# The progress fragment's active step (the `<li ... aria-current="step"> …</span>StepName</li>`).
_ACTIVE_STEP_RE = re.compile(r'aria-current="step"[^>]*>.*?</span>\s*(.*?)\s*</li>', re.DOTALL)


@dataclass
class Outcome:
    """The terminal result of an upload run."""

    status: str  # ingested | partial | failed | busy | expired | unexpected | timeout | unreachable
    doc_id: str | None
    message: str

    @property
    def ok(self) -> bool:
        # A partial ingest left a browsable doc — treat as success (the parse/index ran).
        return self.status in ("ingested", "partial")


# ---------------------------------------------------------------------------
# Pure parsers (unit-tested)
# ---------------------------------------------------------------------------


def extract_status_url(html: str) -> str | None:
    """Return the `/ingest/{cid}/status?v=N` poll path from a `_progress.html` fragment, else
    None for any terminal fragment (done / expired / ingesting / error)."""
    m = _STATUS_URL_RE.search(html)
    return m.group(1) if m else None


def eyebrow(html: str) -> str | None:
    """The `.ans-eyebrow` label (e.g. 'Ingested', 'Working · Transcribing · page 3 · 12s')."""
    m = _EYEBROW_RE.search(html)
    return _collapse_ws(m.group(1)) if m else None


def doc_id(html: str) -> str | None:
    m = _DOC_ID_RE.search(html)
    return m.group(1) if m else None


def message(html: str) -> str | None:
    m = _MESSAGE_RE.search(html)
    if m:
        return _collapse_ws(_strip_tags(m.group(1)))
    # `_ingesting.html` has its message in a bare <span> inside `.ingesting-notice`.
    if "ingesting-notice" in html:
        spans = re.findall(r"<span[^>]*>\s*(.*?)\s*</span>", html, re.DOTALL)
        for s in spans:
            text = _collapse_ws(_strip_tags(s))
            if text:
                return text
    return None


def active_step(html: str) -> str | None:
    """The currently-active high-level phase label of a progress fragment (e.g. 'Transcribing')."""
    m = _ACTIVE_STEP_RE.search(html)
    return _collapse_ws(_strip_tags(m.group(1))) if m else None


def progress_label(html: str) -> str:
    """A one-line human phase label for a progress fragment: the active step + the eyebrow detail
    (the '· …' sub-phase), e.g. 'Transcribing · page 3'. The trailing '· Ns' elapsed is dropped."""
    step = active_step(html) or ""
    eb = eyebrow(html) or ""
    detail = ""
    # The eyebrow is 'Working · <detail> · <N>s'. Pull the middle detail, drop 'Working' + elapsed.
    body = eb[len("Working") :].strip() if eb.startswith("Working") else eb
    body = re.sub(r"·\s*\d+s\s*$", "", body).strip().lstrip("·").strip()
    if body:
        detail = body
    if step and detail and detail != step:
        return f"{step} · {detail}"
    return step or detail or "working"


def classify_terminal(html: str) -> Outcome:
    """Classify a terminal (non-progress) fragment into an Outcome."""
    label = (eyebrow(html) or "").lower()
    did = doc_id(html)
    msg = message(html) or ""
    if did and "partial" in label:
        return Outcome(
            "partial", did, msg or "Partially ingested — browsable but not fully consumed."
        )
    if did:
        return Outcome("ingested", did, msg or "Ingested — searchable and browsable.")
    if "still ingesting" in html.lower() or "ingesting-notice" in html:
        return Outcome("busy", None, msg or "Another document is already being ingested.")
    if "expired" in label:
        return Outcome(
            "expired", None, msg or "The ingest progress entry expired (web UI restarted?)."
        )
    if "failed" in label or "ans-flash-error" in html:
        return Outcome("failed", None, msg or "The file could not be ingested.")
    return Outcome("unexpected", None, msg or _collapse_ws(_strip_tags(html))[:200])


def multipart_body(field_name: str, filename: str, content: bytes, boundary: str) -> bytes:
    """Encode a single-file `multipart/form-data` body (matches what the browser file input sends)."""
    dash = f"--{boundary}".encode()
    head = (
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    return dash + b"\r\n" + head + content + b"\r\n" + dash + b"--\r\n"


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# HTTP (stdlib)
# ---------------------------------------------------------------------------


def _post_ingest(base_url: str, file_path: Path, *, timeout: float) -> str:
    boundary = uuid.uuid4().hex
    body = multipart_body("file", file_path.name, file_path.read_bytes(), boundary)
    req = urllib.request.Request(  # noqa: S310 (a localhost web UI the user runs)
        base_url.rstrip("/") + "/ingest",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read().decode("utf-8", "replace")


def _get(url: str, *, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        return resp.read().decode("utf-8", "replace")


def run(
    base_url: str,
    file_path: Path,
    *,
    timeout: float,
    on_phase: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Outcome:
    """Upload `file_path` to the web UI at `base_url` and follow progress to a terminal Outcome.

    `timeout` is a wall-clock ceiling (seconds) for the whole run. Each status GET long-holds
    ~1s, so the loop ticks about once per second while the pipeline works. Transient connection
    drops on a held poll are retried briefly (the web UI restarts the orchestrator mid-run).
    """
    try:
        html = _post_ingest(base_url, file_path, timeout=min(timeout, 120.0))
    except (urllib.error.URLError, OSError) as e:
        return Outcome("unreachable", None, f"web UI not reachable at {base_url}: {e}")

    deadline = clock() + timeout
    last_label: str | None = None
    consecutive_errors = 0
    while True:
        status_url = extract_status_url(html)
        if status_url is None:
            return classify_terminal(html)
        if on_phase is not None:
            label = progress_label(html)
            if label != last_label:
                on_phase(label)
                last_label = label
        if clock() > deadline:
            return Outcome("timeout", None, f"exceeded the {timeout:.0f}s timeout while ingesting")
        try:
            html = _get(base_url.rstrip("/") + status_url, timeout=30.0)
            consecutive_errors = 0
        except (urllib.error.URLError, OSError) as e:
            consecutive_errors += 1
            if consecutive_errors > 10:
                return Outcome("unreachable", None, f"lost the web UI mid-ingest: {e}")
            # A held poll can drop while the orchestrator restarts; re-poll the same URL.
            continue


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Upload a host file to the Memex web UI /ingest route (no browser needed).",
        epilog="For the pure-pipeline case (no web UI), use:  uv run memex ingest <file_path>",
    )
    parser.add_argument("file", type=Path, help="The host file to ingest.")
    parser.add_argument(
        "--url",
        default=os.environ.get("MEMEX_WEBUI_URL", _DEFAULT_URL),
        help=f"Web UI base URL (default: $MEMEX_WEBUI_URL or {_DEFAULT_URL}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Wall-clock ceiling in seconds (default 1800).",
    )
    parser.add_argument("--json", action="store_true", help="Print the final outcome as JSON.")
    args = parser.parse_args(argv)

    if not args.file.is_file():
        print(f"error: not a file: {args.file}", file=sys.stderr)
        return 1

    def _say(phase: str) -> None:
        if not args.json:
            print(f"  · {phase}", file=sys.stderr)

    if not args.json:
        print(f"Uploading {args.file.name} → {args.url}/ingest", file=sys.stderr)

    outcome = run(args.url, args.file, timeout=args.timeout, on_phase=_say)

    if args.json:
        print(
            json.dumps(
                {"status": outcome.status, "doc_id": outcome.doc_id, "message": outcome.message}
            )
        )
    else:
        tag = "✓" if outcome.ok else "✗"
        print(f"{tag} {outcome.status}: {outcome.message}")
        if outcome.doc_id:
            print(f"  doc_id: {outcome.doc_id}")
            print(f"  open:   {args.url.rstrip('/')}/documents/{outcome.doc_id}")
    return 0 if outcome.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
