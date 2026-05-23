"""Memex CLI commands.

The top-level surface: `ingest`, `parse`, `index`, `reindex`, `ask`,
`enrich`, `graph`, `search`, `watch`, `eval`, `doctor`, and the
`daemon` and `serve` subcommand groups. Every command bootstraps once
before doing any work (loads settings, configures observability,
asserts CUDA availability, registers the event bus).
"""

# pyright: reportUnusedFunction=false
# Typer command handlers are decorated with `@app.command()` which
# registers them in typer's command table. Pyright can't introspect
# the decorator's side effect and flags every command function as
# "not accessed." All 12 commands in this module are reached via the
# Typer CLI entry point.

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from memex.agents.answering import answer_query
from memex.cli.bootstrap import bootstrap
from memex.enrich.pipeline import enrich_document
from memex.eval.runner import run_eval
from memex.index.graph_store import GraphStore
from memex.index.pipeline import index_document, reindex_vault
from memex.ingest.pipeline import (
    IngestRequest,
    IngestResult,
    ingest_directory,
    ingest_file,
    ingest_markdown_passthrough,
)
from memex.ingest.watcher import default_reaction, run_watcher
from memex.parse.pipeline import parse_document

console = Console(stderr=False)  # stdout is the data channel
err = Console(stderr=True)

daemon_app = typer.Typer(help="Manage the local vLLM inference daemon.")
serve_app = typer.Typer(help="Serve Memex as an MCP server or local web UI.")
mcp_app = typer.Typer(help="MCP server utilities (token generation, ...).")


def _print(payload: object) -> None:
    """JSON on a pipe, rich on a TTY (GUIDELINES.md Part V)."""
    if sys.stdout.isatty():
        if isinstance(payload, dict):
            table = Table(show_header=False)
            for k, v in payload.items():
                table.add_row(str(k), str(v))
            console.print(table)
        else:
            console.print(payload)
    else:
        if hasattr(payload, "model_dump_json"):
            sys.stdout.write(payload.model_dump_json())  # type: ignore[attr-defined]
        else:
            json.dump(payload, sys.stdout, default=str)
        sys.stdout.write("\n")


async def _ingest_path_chain(
    p: Path,
    *,
    skip_parse: bool,
    ingest_only: bool,
    do_index: bool,
) -> list[IngestResult]:
    """Run a single file (or every file in a directory) through the
    full ingest → parse → index chain.
    """
    results: list[IngestResult] = []

    async def _process_one(file_path: Path) -> IngestResult:
        # Markdown shortcut: skip-parse drops directly into the vault.
        if skip_parse and file_path.suffix.lower() in {".md", ".markdown"}:
            ref = await ingest_markdown_passthrough(
                file_path.read_text(encoding="utf-8"),
                source_stem=file_path.stem,
            )
            result = IngestResult(
                correlation_id=ref.doc_id,
                source_path=str(file_path),
                accepted=True,
                doc_id=ref.doc_id,
                detected_kind="markdown",
                detected_mime="text/markdown",
                size_bytes=file_path.stat().st_size,
                is_markdown=True,
            )
        else:
            result = await ingest_file(IngestRequest(source_path=file_path))

        if not result.accepted or ingest_only or result.doc_id is None:
            return result

        await parse_document(result.doc_id)
        if do_index:
            await index_document(result.doc_id)
        return result

    if p.is_dir():
        # `ingest_directory` is a streaming iterator; route each accepted
        # file back through `_process_one` so `--skip-parse` and the
        # markdown-passthrough fast-path apply uniformly to directory
        # inputs (per-file shape matches the single-file branch).
        async for r in ingest_directory(p):
            if not r.accepted or r.doc_id is None:
                results.append(r)
                continue
            if ingest_only:
                results.append(r)
                continue
            # `ingest_directory` has already done the file-level work;
            # we just need parse + (optional) index for each accepted doc.
            # Markdown sources that came in via `ingest_directory` were
            # processed as bytes — when `--skip-parse` is set we still
            # need to honour it.
            if skip_parse and r.is_markdown:
                # Honour --skip-parse for directory items too.
                results.append(r)
                continue
            await parse_document(r.doc_id)
            if do_index:
                await index_document(r.doc_id)
            results.append(r)
    else:
        results.append(await _process_one(p))

    return results


def register(app: typer.Typer) -> None:
    """Attach commands to `app`."""

    @app.command()
    def ingest(
        paths: list[Path] = typer.Argument(  # noqa: B008
            ..., exists=True, readable=True, help="Files or directories to ingest."
        ),
        ingest_only: bool = typer.Option(
            False,
            "--ingest-only",
            help="Stop after validation + copy + manifest. Don't parse or index.",
        ),
        skip_parse: bool = typer.Option(
            False,
            "--skip-parse",
            help="For markdown sources, write straight to the vault without parsing.",
        ),
        no_index: bool = typer.Option(
            False, "--no-index", help="Parse but don't index."
        ),
    ) -> None:
        """Validate, copy, parse, and index input files."""

        async def _run() -> list[IngestResult]:
            bootstrap()
            out: list[IngestResult] = []
            for p in paths:
                out.extend(
                    await _ingest_path_chain(
                        p,
                        skip_parse=skip_parse,
                        ingest_only=ingest_only,
                        do_index=not no_index,
                    )
                )
            return out

        results = asyncio.run(_run())
        for r in results:
            _print(r)

    @app.command(name="parse")
    def parse_cmd(doc_id: str) -> None:
        """Re-parse a document already in the vault."""
        async def _run():
            bootstrap()
            return await parse_document(doc_id)

        _print(asyncio.run(_run()))

    @app.command(name="index")
    def index_cmd(doc_id: str) -> None:
        """Chunk, embed, and write derived state for one document."""
        async def _run():
            bootstrap()
            return await index_document(doc_id)

        _print(asyncio.run(_run()))

    @app.command()
    def reindex(
        force: bool = typer.Option(
            False, "--force", help="Delete .memex/{embeddings,search,graph} first."
        ),
    ) -> None:
        """Rebuild derived state from the canonical Markdown vault."""
        async def _run():
            bootstrap()
            return await reindex_vault(force=force)

        _print(asyncio.run(_run()))

    @app.command()
    def ask(
        query: str = typer.Argument(..., help="The question to answer."),
        token_budget: int = typer.Option(8000, help="Max tokens across the agent loop."),
    ) -> None:
        """Run the answering agent over the vault."""
        async def _run():
            bootstrap()
            return await answer_query(query, token_budget=token_budget)

        _print(asyncio.run(_run()))

    @app.command(name="enrich")
    def enrich_cmd(doc_id: str) -> None:
        """Run entity extraction + graph linking for one document."""
        async def _run():
            bootstrap()
            return await enrich_document(doc_id)

        _print(asyncio.run(_run()))

    @app.command()
    def graph(
        document: str = typer.Option(..., "--document", "-d", help="doc_id."),
        limit: int = typer.Option(50, help="Max neighbors to print."),
    ) -> None:
        """Print one-hop graph neighbors (shared entities) for a document."""
        async def _run():
            from memex.core.config import get_settings

            bootstrap()
            store = await GraphStore.open(get_settings().vault_path)
            try:
                return await store.neighbors(document, limit=limit)
            finally:
                await store.close()

        results = asyncio.run(_run())
        for r in results:
            _print(r)

    @app.command()
    def doctor() -> None:
        """Health check: vault integrity, daemon reachability, breaker state."""
        async def _run() -> dict[str, object]:
            bootstrap()
            return await _doctor_report()

        _print(asyncio.run(_run()))

    @app.command()
    def search(
        query: str = typer.Argument(..., help="The search query."),
        k: int = typer.Option(10, help="Number of top chunks to return."),
    ) -> None:
        """Hybrid search over the vault — BM25 ⊕ dense → RRF → rerank."""
        async def _run():
            from memex.retrieve import cross_encoder_rerank, hybrid_search

            bootstrap()
            candidates = await hybrid_search(query, k=max(50, k * 5))
            return await cross_encoder_rerank(query, candidates, top_k=k)

        results = asyncio.run(_run())
        for r in results:
            _print(r)

    @app.command()
    def watch() -> None:
        """Watch the vault for markdown edits and re-enrich + re-index live."""
        async def _run():
            from memex.core.config import get_settings

            bootstrap()
            await run_watcher(
                get_settings().vault_path,
                on_edit=default_reaction,
            )

        try:
            asyncio.run(_run())
        except KeyboardInterrupt:
            err.print("[yellow]watcher stopped[/yellow]")

    @app.command(name="eval")
    def eval_cmd(
        query_set: Path = typer.Argument(  # noqa: B008
            ..., help="Path to a JSON query set (see docs/eval-corpus-plan.md)."
        ),
        quick: bool = typer.Option(False, "--quick", help="Sample ~20% of queries."),
    ) -> None:
        """Run the eval harness against a query set."""
        async def _run():
            bootstrap()
            return await run_eval(query_set, quick=quick)

        _print(asyncio.run(_run()))

    @app.command()
    def upgrade(
        no_restart: bool = typer.Option(
            False,
            "--no-restart",
            help="Skip the systemd restart step (Pattern B / Pattern C boxes).",
        ),
        skip_sync: bool = typer.Option(
            False,
            "--skip-sync",
            help="Skip `uv sync` (git pull + restart only).",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Print steps without running them.",
        ),
    ) -> None:
        """Pull, sync, restart — apply updates to a systemd-deployed Memex.

        Runs three steps in order:

          1. git pull --ff-only           (refuses if your tree is dirty)
          2. uv sync --extra models --extra parse --extra serve
          3. systemctl --user restart memex-{vllm,web,mcp,watch}.service
                                          (only units actually installed)

        Pattern B (manual) / Pattern C (`memex daemon start`) users pass
        --no-restart since they don't have systemd units. Use --dry-run
        to preview without executing.
        """
        import shutil

        repo_root = _find_repo_root()
        err.print(f"[dim]Memex upgrade — repo: {repo_root}[/dim]")

        # ── Step 1: git pull ─────────────────────────────────────────
        _upgrade_step(
            "git pull --ff-only",
            ["git", "-C", str(repo_root), "pull", "--ff-only"],
            dry_run=dry_run,
        )

        # ── Step 2: uv sync ──────────────────────────────────────────
        if not skip_sync:
            _upgrade_step(
                "uv sync --extra models --extra parse --extra serve",
                [
                    "uv", "sync",
                    "--extra", "models",
                    "--extra", "parse",
                    "--extra", "serve",
                ],
                dry_run=dry_run,
                cwd=repo_root,
            )
        else:
            err.print("[dim]→ skip-sync: not running uv sync[/dim]")

        # ── Step 3: systemctl restart ────────────────────────────────
        if no_restart:
            err.print("[dim]→ no-restart: not touching systemd units[/dim]")
            err.print("[green]✓ upgrade complete[/green]")
            return
        if shutil.which("systemctl") is None:
            err.print(
                "[yellow]systemctl not found; skipping restart step. "
                "Restart your daemons manually if needed.[/yellow]"
            )
            err.print("[green]✓ upgrade complete[/green]")
            return
        units = _installed_memex_units()
        if not units:
            err.print(
                "[yellow]no memex user units installed; see "
                "docs/deploy/systemd.md to enable systemd supervision.[/yellow]"
            )
            err.print("[green]✓ upgrade complete[/green]")
            return
        err.print(
            f"[dim]→ restarting {len(units)} unit"
            f"{'s' if len(units) != 1 else ''}; "
            "vLLM's Type=notify gate adds ~30 s for the readiness check[/dim]"
        )
        _upgrade_step(
            f"systemctl --user restart {' '.join(units)}",
            ["systemctl", "--user", "restart", *units],
            dry_run=dry_run,
        )
        err.print("[green]✓ upgrade complete[/green]")

    app.add_typer(daemon_app, name="daemon")
    app.add_typer(serve_app, name="serve")
    app.add_typer(mcp_app, name="mcp")


def _find_repo_root() -> Path:
    """Locate the Memex git repo by walking up from this file.

    Falls back to the current working directory if no `.git` is
    found (e.g. when Memex is installed via `uv tool install` rather
    than cloned). `memex upgrade` is only useful for a git checkout,
    so callers should expect the git step to surface that mismatch.
    """
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def _installed_memex_units() -> list[str]:
    """Return the list of `memex-*.service` user units present on disk.

    Empty list when `systemctl --user` isn't usable — no session bus,
    inside a container, on macOS, etc. Caller treats an empty list as
    "nothing to restart" and reports it to the user.
    """
    import subprocess

    try:
        result = subprocess.run(
            [
                "systemctl", "--user", "list-unit-files", "memex-*.service",
                "--no-legend", "--no-pager",
            ],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    out: list[str] = []
    for line in result.stdout.strip().splitlines():
        unit = line.split()[0] if line.split() else ""
        if unit.startswith("memex-") and unit.endswith(".service"):
            out.append(unit)
    return out


def _upgrade_step(
    label: str,
    argv: list[str],
    *,
    dry_run: bool = False,
    cwd: Path | None = None,
) -> None:
    """Announce + run one step of `memex upgrade`.

    Streams the subprocess's stdout + stderr directly to the user's
    terminal so git / uv / systemctl output is visible in real time.
    Non-zero exit raises `typer.Exit(code)` so the overall command
    bubbles the failure up.
    """
    import subprocess

    err.print(f"[bold blue]→[/bold blue] {label}")
    if dry_run:
        err.print(f"  [dim]would run: {' '.join(argv)}[/dim]")
        return
    completed = subprocess.run(argv, cwd=cwd, check=False)
    if completed.returncode != 0:
        err.print(
            f"[red]✗ {label} failed (exit {completed.returncode})[/red]"
        )
        raise typer.Exit(code=completed.returncode)


@mcp_app.command("generate-token")
def mcp_generate_token(
    length: int = typer.Option(
        32,
        "--length",
        min=16,
        max=128,
        help="Token length in bytes (urlsafe-encoded; the printed string "
        "is ~33% longer).",
    ),
) -> None:
    """Print a fresh urlsafe bearer token to stdout.

    Save the output into `MEMEX_MCP__AUTH_TOKEN` (env or
    `~/.config/memex/config.toml`) and restart `memex serve mcp` to
    require it on incoming requests. See docs/deploy/mcp-http.md.

    The token is printed *without* a trailing newline-noise prompt so
    it can be piped into a secrets manager: `memex mcp generate-token
    | pbcopy` / `... | wl-copy`.
    """
    import secrets

    typer.echo(secrets.token_urlsafe(length))


@serve_app.command("mcp")
def serve_mcp(
    transport: str = typer.Option(
        "stdio",
        "--transport",
        help="`stdio` for desktop clients (Claude Code, Cursor) or "
        "`http` for network-local agents (binds 127.0.0.1 by default).",
    ),
    host: str = typer.Option("127.0.0.1", help="Bind host (http transport only)."),
    port: int = typer.Option(7424, help="Bind port (http transport only)."),
) -> None:
    """Run the Memex MCP server."""
    from memex.cli.bootstrap import bootstrap as _boot
    from memex.mcp.server import serve_http, serve_stdio

    _boot()
    if transport == "stdio":
        # Don't print to stdout on stdio transport — that channel IS the
        # MCP protocol. stderr is fine.
        err.print(
            "[green]Memex MCP server ready[/green] [dim](stdio transport, "
            "Ctrl-C to stop)[/dim]"
        )
        asyncio.run(serve_stdio())
    elif transport == "http":
        err.print(
            f"[green]Memex MCP server ready →[/green] [bold]http://{host}:{port}[/bold]   "
            f"[dim](Ctrl-C to stop)[/dim]"
        )
        asyncio.run(serve_http(host=host, port=port))
    else:
        err.print(
            f"[red]unknown transport {transport!r}; expected stdio or http[/red]"
        )
        raise typer.Exit(code=2)


@serve_app.command("web")
def serve_web(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(7423, help="Bind port."),
) -> None:
    """Run the local FastAPI + HTMX web UI."""
    import uvicorn

    from memex.cli.bootstrap import bootstrap as _boot
    from memex.webui.app import create_app

    _boot()
    # `log_config=None` keeps uvicorn out of structlog's way, but it
    # also silences uvicorn's own "Uvicorn running on ..." startup
    # banner. Print a one-line ready signal to stderr so the operator
    # can see the server is listening without having to curl /healthz.
    err.print(
        f"[green]Memex web UI ready →[/green] [bold]http://{host}:{port}[/bold]   "
        f"[dim](Ctrl-C to stop)[/dim]"
    )
    uvicorn.run(create_app(), host=host, port=port, log_config=None)


async def _doctor_report() -> dict[str, object]:
    """Build the `memex doctor` payload."""
    from memex.core.config import get_settings
    from memex.core.manifest import read_manifest
    from memex.models.client import get_client
    from memex.parse.pipeline import (
        get_docling_breaker_state,
        get_pymupdf_breaker_state,
    )
    from memex.prompts.loader import list_prompts
    from memex.vault.store import hash_bytes, list_documents

    settings = get_settings()

    # Vault integrity walk.
    issues: list[str] = []
    docs = 0
    stale_index = 0
    async for ref in list_documents(settings.vault_path):
        docs += 1
        manifest = await read_manifest(settings.vault_path, ref.doc_id)
        if manifest is None:
            issues.append(f"{ref.doc_id}: no manifest")
            continue
        actual = hash_bytes(ref.markdown_path.read_bytes())
        if manifest.content_sha256 != actual:
            issues.append(
                f"{ref.doc_id}: content_sha256 drifted (user edit?)"
            )
            stale_index += 1
        if manifest.index is None:
            stale_index += 1

    # Daemon probe.
    try:
        client = get_client()
        models = await client.models.list()
        daemon = {
            "reachable": True,
            "base_url": settings.inference.base_url,
            "models": [m.id for m in getattr(models, "data", [])],
        }
    except Exception as e:
        daemon = {
            "reachable": False,
            "base_url": settings.inference.base_url,
            "error": str(e),
        }

    # Breaker state — both parser breakers surfaced so an operator
    # can spot the asymmetric tripped case (e.g. PyMuPDF crashed but
    # Docling is still healthy).
    docling_state, docling_failures = get_docling_breaker_state()
    pymupdf_state, pymupdf_failures = get_pymupdf_breaker_state()
    breakers = {
        "docling": {
            "state": docling_state,
            "failures": docling_failures,
        },
        "pymupdf": {
            "state": pymupdf_state,
            "failures": pymupdf_failures,
        },
    }

    # Prompt versions — surfaces what's actually loaded so an
    # operator can confirm e.g. `answer/v2.md` is the active answer
    # template (vs an accidental rollback to v1).
    prompts = [
        {"name": p.name, "version": p.version} for p in list_prompts()
    ]

    return {
        "vault_path": str(settings.vault_path),
        "documents": docs,
        "stale_index_count": stale_index,
        "issues": issues,
        "daemon": daemon,
        "breakers": breakers,
        "prompts": prompts,
    }


@daemon_app.command("start")
def daemon_start() -> None:
    """Spawn the vLLM server in the background, wait for reachability."""
    from memex.cli.bootstrap import bootstrap as _boot
    from memex.daemon import (
        DaemonAlreadyRunning,
        DaemonStartTimeout,
    )
    from memex.daemon import (
        start as _start,
    )

    settings = _boot()
    try:
        status = asyncio.run(_start(settings))
    except DaemonAlreadyRunning as e:
        err.print(f"[yellow]{e}[/yellow]")
        raise typer.Exit(code=1) from e
    except DaemonStartTimeout as e:
        err.print(f"[red]{e}[/red]")
        err.print(
            f"[red]see log: {e.context.get('log_file')}[/red]"
        )
        raise typer.Exit(code=1) from e
    _print(status)


@daemon_app.command("stop")
def daemon_stop() -> None:
    """SIGTERM the running vLLM daemon (SIGKILL after 10 s grace)."""
    from memex.cli.bootstrap import bootstrap as _boot
    from memex.daemon import stop as _stop

    settings = _boot()
    _print(_stop(settings))


@daemon_app.command("status")
def daemon_status() -> None:
    """Report PID + reachability of the configured vLLM endpoint."""
    from memex.cli.bootstrap import bootstrap as _boot
    from memex.daemon import status as _status

    settings = _boot()
    _print(asyncio.run(_status(settings)))
