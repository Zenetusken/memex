"""Memex CLI — typer entry points.

The CLI is the canonical interface; everything the web UI does, the
CLI does, often more directly. Output is rich (tables, syntax
highlighting, progress bars) when stdout is a TTY, plain JSON when
piped. See GUIDELINES.md Part V "CLI".
"""

from __future__ import annotations

import typer

from memex import __version__
from memex.cli import commands

app = typer.Typer(
    name="memex",
    help="Local-first, fully agentic document understanding system.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the installed Memex version."""
    typer.echo(__version__)


commands.register(app)


if __name__ == "__main__":
    app()
