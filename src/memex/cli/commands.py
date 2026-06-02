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
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import typer
from rich.console import Console
from rich.table import Table

from memex.agents.answering import FinalResponse, answer_query
from memex.agents.chat import answer_turn
from memex.agents.expert import ExpertAnswer
from memex.cli.bootstrap import bootstrap
from memex.core.conversation_store import ConversationStore
from memex.enrich.pipeline import enrich_document
from memex.eval.runner import run_chat_eval, run_eval, run_parse_eval, run_summary_eval
from memex.index.graph_store import GraphStore
from memex.index.pipeline import index_document, reindex_vault, retitle_document
from memex.ingest.pipeline import (
    IngestRequest,
    IngestResult,
    ingest_directory,
    ingest_file,
    ingest_markdown_passthrough,
)
from memex.ingest.watcher import default_reaction, run_watcher
from memex.parse.pipeline import derive_title, parse_document, pause_vllm_for_gpu

# `typer.Option` / `typer.Argument` are typed as overloads whose signatures
# embed `click.ParamType[Unknown]`, so a bare member access trips pyright's
# `reportUnknownMemberType` under strict. They return `Any` at runtime (the
# value is a typer parameter sentinel slotted into the annotated default), so
# we re-expose them through `Any`-typed aliases. Behaviour is identical.
_Option: Callable[..., Any] = typer.Option  # type: ignore[reportUnknownMemberType]  # typer stub leaks click.ParamType[Unknown]
_Argument: Callable[..., Any] = typer.Argument  # type: ignore[reportUnknownMemberType]  # typer stub leaks click.ParamType[Unknown]

console = Console(stderr=False)  # stdout is the data channel
err = Console(stderr=True)

daemon_app = typer.Typer(help="Manage the local vLLM inference daemon.")
serve_app = typer.Typer(help="Serve Memex as an MCP server or local web UI.")
mcp_app = typer.Typer(help="MCP server utilities (token generation, ...).")
scope_set_app = typer.Typer(
    help="Manage saved document scope sets — named collections you reapply with "
    "`memex ask --scope-set NAME`."
)
mode_app = typer.Typer(
    help="Co-residence resource modes (ADR-0007): the VRAM tradeoff between the "
    "orchestrator's context window and GPU-resident retrieval."
)


def _print(payload: object) -> None:
    """JSON on a pipe, rich on a TTY (GUIDELINES.md Part V)."""
    if sys.stdout.isatty():
        if isinstance(payload, dict):
            mapping = cast("dict[object, object]", payload)
            table = Table(show_header=False)
            for k, v in mapping.items():
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
    force_docling: bool = False,
) -> list[IngestResult]:
    """Run a single file (or every file in a directory) through the
    full ingest → parse → index chain.

    `force_docling` is forwarded to `parse_document` per call; when
    True, the parse stage bypasses the PyMuPDF classifier and goes
    straight to Docling (useful for chart-OCR validation on otherwise-
    PyMuPDF-routed docs).
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

        await parse_document(result.doc_id, force_docling=force_docling)
        if do_index:
            await index_document(result.doc_id)
        return result

    # Hold vLLM paused across the WHOLE ingest (parse + index). On the 12 GB
    # rig the embedder can't run co-resident with vLLM (~8.5 GB), and the
    # parse's own per-VLM pause restarts vLLM *before* the index embed runs,
    # which then OOMs. With vLLM down for the duration the inner pause is a
    # no-op and the embed has the GPU; vLLM restarts once at the end. No-op
    # when vLLM isn't running (e.g. ingest with the daemon already stopped).
    async with pause_vllm_for_gpu():
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
                # `ingest_directory` has already done the file-level work
                # (copied `source.md` + wrote the manifest) but NOT the canonical
                # `{doc_id}.md`. Always run `parse_document`: for a markdown source
                # it dispatches to the passthrough (which materializes `{doc_id}.md`
                # WITHOUT a real parse — that IS the skip-parse behaviour), so the
                # doc becomes visible to list/search/index. Matches the single-file
                # branch, which also calls `parse_document` for skip-parse markdown.
                await parse_document(r.doc_id, force_docling=force_docling)
                if do_index:
                    await index_document(r.doc_id)
                results.append(r)
        else:
            results.append(await _process_one(p))

    return results


def _render_chat_response(response: FinalResponse) -> str:
    """Concise console rendering of one grounded chat turn — the answer + cited claims,
    or a refusal with related-doc suggestions (the CLI analogue of `_answer.html`)."""
    if not response.answered:
        lines = [f"⊘ {response.refusal_reason or 'No grounded answer from your vault.'}"]
        if response.related_documents:
            titles = ", ".join(d.title for d in response.related_documents[:3])
            lines.append(f"  You might look at: {titles}")
        return "\n".join(lines)
    lines = [response.summary or ""]
    for c in response.claims:
        lines.append(f"  • {c.claim}")
    if response.wikilinks:
        lines.append(f"  sources: {', '.join(response.wikilinks)}")
    return "\n".join(line for line in lines if line)


def _render_expert_answer(answer: ExpertAnswer) -> str:
    """Console rendering of an ungrounded expert answer (Surface B): the reasoned prose,
    the evidence it drew on, and the standing provenance caveat (model knowledge, unverified)."""
    lines = [answer.answer.strip(), ""]
    if answer.evidence:
        titles = ", ".join(dict.fromkeys(e.title for e in answer.evidence))
        lines.append(f"evidence consulted: {titles}")
    lines.append(f"⚠ {answer.provenance_note}")
    return "\n".join(lines)


async def run_chat_repl(
    read_line: Callable[[], str | None],
    emit: Callable[[str], None],
    *,
    scope_doc_ids: list[str] | None = None,
    resume_id: str | None = None,
) -> str:
    """Drive a grounded multi-turn chat loop over `answer_turn`, surface-agnostic and
    testable: `read_line()` returns the next user line (or `None` on EOF), `emit()` writes
    a line back. Returns the conversation id. `/exit` or `/quit` (or EOF) ends the loop.

    Each line runs the unchanged grounded pipeline (per-turn HARD gates intact); the
    conversation persists in the sqlite sidecar so `--resume` rehydrates it.
    """
    from memex.core.config import get_settings
    from memex.core.errors import ConfigurationError

    store = await ConversationStore.open(get_settings().vault_path)
    try:
        if resume_id:
            convo = await store.load(resume_id)
            if convo is None:
                raise ConfigurationError(
                    "no such conversation", context={"conversation_id": resume_id}
                )
            conversation_id = resume_id
            emit(f"Resumed conversation {conversation_id} ({convo.turn_count} prior turns).")
            for t in convo.turns:
                emit(f"you › {t.user_text}")
                emit(f"  {t.answer_summary}")
        else:
            convo = await store.create_conversation(scope_doc_ids=scope_doc_ids)
            conversation_id = convo.conversation_id
            scope_note = (
                f" (scoped to {len(convo.scope_doc_ids)} doc[s])" if convo.scope_doc_ids else ""
            )
            emit(
                f"New conversation {conversation_id}{scope_note}. "
                "Grounded in your vault — type /exit to quit."
            )
    finally:
        await store.close()

    while True:
        line = read_line()
        if line is None:
            break
        line = line.strip()
        if not line:
            continue
        if line in ("/exit", "/quit"):
            break
        result = await answer_turn(conversation_id, line)
        emit(_render_chat_response(result.response))
    return conversation_id


def register(app: typer.Typer) -> None:
    """Attach commands to `app`."""

    @app.command()
    def ingest(
        paths: list[Path] = _Argument(  # noqa: B008
            ..., exists=True, readable=True, help="Files or directories to ingest."
        ),
        ingest_only: bool = _Option(
            False,
            "--ingest-only",
            help="Stop after validation + copy + manifest. Don't parse or index.",
        ),
        skip_parse: bool = _Option(
            False,
            "--skip-parse",
            help="For markdown sources, write straight to the vault without parsing.",
        ),
        no_index: bool = _Option(False, "--no-index", help="Parse but don't index."),
        force_docling: bool = _Option(
            False,
            "--force-docling",
            help=(
                "Bypass the PyMuPDF classifier and route directly to Docling. "
                "Use to enable chart-OCR on born-digital text-heavy PDFs the "
                "classifier would normally route to PyMuPDF."
            ),
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
                        force_docling=force_docling,
                    )
                )
            return out

        results = asyncio.run(_run())
        for r in results:
            _print(r)

    @app.command(name="parse")
    def parse_cmd(
        doc_id: str,
        force_docling: bool = _Option(
            False,
            "--force-docling",
            help=(
                "Bypass the PyMuPDF classifier and route directly to Docling. "
                "Use to enable chart-OCR on born-digital text-heavy PDFs the "
                "classifier would normally route to PyMuPDF."
            ),
        ),
        refresh_vlm: bool = _Option(
            False,
            "--refresh-vlm",
            help=(
                "Bust this document's cached VLM transcriptions and re-run them "
                "(the VLM is non-deterministic; transcriptions are cached by "
                "default for reproducibility)."
            ),
        ),
    ) -> None:
        """Re-parse a document already in the vault."""

        async def _run():
            bootstrap()
            return await parse_document(
                doc_id, force_docling=force_docling, refresh_vlm=refresh_vlm
            )

        _print(asyncio.run(_run()))

    @app.command(name="index")
    def index_cmd(
        doc_id: str,
        force: bool = _Option(
            False,
            "--force",
            help="Re-chunk + re-embed every chunk (not just the diff). Use to "
            "backfill metadata that the content-addressed partial index skips "
            "on a same-content re-parse — e.g. source-page attribution "
            "(`Chunk.page`) after re-parsing a doc that predates it.",
        ),
    ) -> None:
        """Chunk, embed, and write derived state for one document."""

        async def _run():
            bootstrap()
            # Pause vLLM around the embed — the embedder OOMs co-resident with
            # vLLM (~8.5 GB) on the 12 GB rig; no-op when vLLM isn't running.
            async with pause_vllm_for_gpu():
                return await index_document(doc_id, force=force)

        _print(asyncio.run(_run()))

    @app.command(name="retitle")
    def retitle_cmd(
        doc_id: str,
        title: str = _Argument(
            "",
            help="The new title. Omit and pass --derive to pull it from "
            "the original source filename instead.",
        ),
        derive: bool = _Option(
            False,
            "--derive",
            help="Derive the title from the manifest's source filename "
            "(for docs ingested before the title-derivation fix).",
        ),
    ) -> None:
        """Rename a document's title everywhere it's stored.

        A title is pure metadata — it isn't part of the embedded text or
        the chunk id — so this updates the frontmatter plus the FTS,
        vector, and graph copies without re-chunking or re-embedding.
        """
        if derive == bool(title):
            raise typer.BadParameter("pass exactly one of TITLE or --derive")

        async def _run():
            bootstrap()
            from memex.core.config import get_settings

            new_title = await derive_title(get_settings().vault_path, doc_id) if derive else title
            return await retitle_document(doc_id, new_title)

        _print(asyncio.run(_run()))

    @app.command(name="remove")
    def remove_cmd(
        doc_id: str,
        yes: bool = _Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    ) -> None:
        """Remove a document entirely — its Markdown, asset dir, manifest, and
        ALL derived index state (vector, FTS, tables, graph).

        Irreversible: the canonical Markdown is the source of truth, so once
        removed the document is gone until you re-add the original source.
        """
        if not yes:
            typer.confirm(
                f"Permanently remove '{doc_id}' (Markdown + all derived index "
                "state)? This cannot be undone.",
                abort=True,
            )

        async def _run() -> dict[str, object]:
            bootstrap()
            from memex.core.config import get_settings
            from memex.core.manifest import delete_manifest, read_manifest
            from memex.index import remove_document
            from memex.vault.store import delete_document

            vault_path = get_settings().vault_path
            manifest = await read_manifest(vault_path, doc_id)
            md_exists = (vault_path / "documents" / f"{doc_id}.md").exists()
            if manifest is None and not md_exists:
                return {"doc_id": doc_id, "removed": False, "reason": "not found"}
            # Derived index state FIRST, then the canonical Markdown + assets,
            # then the manifest — so a crash mid-remove leaves the least-
            # confusing remnant (orphaned Markdown re-indexes cleanly on the
            # next reindex; orphaned index rows would surface a phantom doc).
            await remove_document(doc_id)
            await delete_document(vault_path, doc_id)
            await delete_manifest(vault_path, doc_id)
            return {"doc_id": doc_id, "removed": True}

        _print(asyncio.run(_run()))

    @app.command()
    def reindex(
        force: bool = _Option(
            False, "--force", help="Delete .memex/{embeddings,search,graph} first."
        ),
    ) -> None:
        """Rebuild derived state from the canonical Markdown vault."""

        async def _run():
            bootstrap()
            async with pause_vllm_for_gpu():
                return await reindex_vault(force=force)

        _print(asyncio.run(_run()))

    @app.command()
    def ask(
        query: str = _Argument(..., help="The question to answer."),
        token_budget: int = _Option(8000, help="Max tokens across the agent loop."),
        doc: list[str] = _Option(  # noqa: B008  # typer Option default sentinel
            [],
            "--doc",
            help="Scope retrieval to this document id (repeatable). Omit = whole vault.",
        ),
        scope_set: str = _Option(
            "",
            "--scope-set",
            help="Scope retrieval to a saved scope set by name (see `memex scope-set`). "
            "Combines with any --doc ids.",
        ),
    ) -> None:
        """Run the answering agent over the vault.

        `--doc <id>` (repeatable) scopes the answer to only those documents — it
        is grounded in them or refuses. `--scope-set NAME` adds a saved set's
        documents to that scope. Omit both to search the whole vault.
        """

        async def _run():
            bootstrap()
            scope_ids = list(doc)
            if scope_set.strip():
                from memex.core.config import get_settings
                from memex.core.scope_sets import get_scope_set, list_scope_sets

                vault_path = get_settings().vault_path
                found = await get_scope_set(vault_path, scope_set)
                if found is None:
                    available = [s.name for s in await list_scope_sets(vault_path)]
                    hint = ", ".join(available) if available else "(none saved yet)"
                    err.print(f"[red]No scope set named {scope_set!r}.[/red] Available: {hint}")
                    raise typer.Exit(code=2)
                scope_ids.extend(found.doc_ids)
            # Ordered de-dup so --doc + --scope-set overlap collapses cleanly.
            merged = list(dict.fromkeys(scope_ids))
            response = await answer_query(
                query, token_budget=token_budget, scope_doc_ids=merged or None
            )
            # "Explore connections" parity with the webui /ask panel + MCP: enrich the
            # response with the graph neighbours of the cited docs (read-only, post-grounding,
            # fail-open → []). Off the agent path, so the eval/answer contract is unchanged.
            from memex.core.config import get_settings
            from memex.retrieve import related_documents_for_answer

            response.related_documents = await related_documents_for_answer(
                get_settings().vault_path,
                [c.document_id for c in response.used_chunks],
                answered=response.answered,
            )
            return response

        _print(asyncio.run(_run()))

    @app.command()
    def chat(
        resume: str = _Option(
            "", "--resume", help="Resume a conversation by its id (printed when a chat starts)."
        ),
        doc: list[str] = _Option(  # noqa: B008  # typer Option default sentinel
            [],
            "--doc",
            help="Scope the whole conversation to this document id (repeatable).",
        ),
        scope_set: str = _Option(
            "", "--scope-set", help="Scope the conversation to a saved scope set by name."
        ),
    ) -> None:
        """Grounded multi-turn chat over the vault (Surface A).

        A conversational `ask`: each turn is answered ONLY from your documents — it
        refuses rather than invent — with memory of the conversation. State persists,
        so `--resume <id>` continues a thread. Type /exit (or Ctrl-D) to quit.
        """

        async def _run() -> str:
            bootstrap()
            scope_ids = list(doc)
            if scope_set.strip():
                from memex.core.config import get_settings
                from memex.core.scope_sets import get_scope_set, list_scope_sets

                vault_path = get_settings().vault_path
                found = await get_scope_set(vault_path, scope_set)
                if found is None:
                    available = [s.name for s in await list_scope_sets(vault_path)]
                    hint = ", ".join(available) if available else "(none saved yet)"
                    err.print(f"[red]No scope set named {scope_set!r}.[/red] Available: {hint}")
                    raise typer.Exit(code=2)
                scope_ids.extend(found.doc_ids)
            merged = list(dict.fromkeys(scope_ids))

            def _read() -> str | None:
                try:
                    return console.input("[bold blue]you ›[/bold blue] ")
                except EOFError:
                    return None

            def _emit(text: str) -> None:
                console.print(text)

            return await run_chat_repl(
                _read,
                _emit,
                scope_doc_ids=merged or None,
                resume_id=resume.strip() or None,
            )

        asyncio.run(_run())

    @app.command()
    def summarize(
        doc_id: str = _Argument(..., help="Document id to summarise."),
        instruction: str = _Option("", "--instruction", "-i", help="Optionally focus the summary."),
        detail: str = _Option(
            "standard",
            "--detail",
            help="Length/detail: brief | standard | detailed | report (multi-paragraph).",
        ),
        max_tokens: int = _Option(
            2048, "--max-tokens", help="Max output tokens per map/reduce call."
        ),
        token_budget: int = _Option(
            120_000,
            "--token-budget",
            help="Total token budget across the whole map-reduce (a long doc stops early).",
        ),
    ) -> None:
        """Summarise a document — a structured, GROUNDED summary (ADR-0008).

        Doc-type-aware map-reduce over the document's sections: an abstract +
        cited key-points + per-section digests, every point grounded to a source
        chunk (refuses rather than fabricate). `--detail` tunes length
        (brief/standard/detailed); `report` builds a multi-paragraph body via the
        hierarchical reducer (ADR-0010). Quality is identical in `fast` or `full` mode.
        """
        valid = ("brief", "standard", "detailed", "report")
        if detail not in valid:
            err.print(f"[red]Unknown --detail {detail!r}.[/red] Choose: {', '.join(valid)}")
            raise typer.Exit(code=2)

        async def _run():
            bootstrap()
            from memex.agents.document_summarizer import summarize_document

            # `detail` is narrowed to the SummaryDetail literal by the guard above.
            return await summarize_document(
                doc_id,
                instruction=instruction or None,
                detail=detail,
                max_output_tokens=max_tokens,
                token_budget=token_budget,
            )

        _print(asyncio.run(_run()))

    @app.command()
    def expert(
        question: str = _Argument(..., help="An analytical / advisory question to reason about."),
        doc: list[str] = _Option(  # noqa: B008  # typer Option default sentinel
            [], "--doc", help="Limit the consulted evidence to this document id (repeatable)."
        ),
        scope_set: str = _Option(
            "", "--scope-set", help="Limit the consulted evidence to a saved scope set by name."
        ),
    ) -> None:
        """Ungrounded EXPERT analysis (Surface B, ADR-0013) — reasoning, NOT a vault lookup.

        Answers an analytical / advisory / synthesis question from the model's own
        knowledge, reasoned OVER evidence retrieved from your vault. This INVERTS the
        grounding contract: unlike `ask`, it MAY go beyond your documents and is NOT
        verified — every answer is labelled as model knowledge to check. Disabled unless
        `MEMEX_AGENTS__EXPERT_MODE_ENABLED=true` (or `agents.expert_mode_enabled` in config).
        """

        async def _run() -> str:
            bootstrap()
            from memex.agents.expert import expert_answer
            from memex.core.config import get_settings

            if not get_settings().agents.expert_mode_enabled:
                err.print(
                    "[yellow]Expert mode is disabled.[/yellow] It is an UNGROUNDED reasoning "
                    "surface (ADR-0013) that can go beyond your vault. Enable it with "
                    "MEMEX_AGENTS__EXPERT_MODE_ENABLED=true (or agents.expert_mode_enabled in "
                    "config.toml)."
                )
                raise typer.Exit(code=2)

            scope_ids = list(doc)
            if scope_set.strip():
                from memex.core.scope_sets import get_scope_set, list_scope_sets

                vault_path = get_settings().vault_path
                found = await get_scope_set(vault_path, scope_set)
                if found is None:
                    available = [s.name for s in await list_scope_sets(vault_path)]
                    hint = ", ".join(available) if available else "(none saved yet)"
                    err.print(f"[red]No scope set named {scope_set!r}.[/red] Available: {hint}")
                    raise typer.Exit(code=2)
                scope_ids.extend(found.doc_ids)
            merged = list(dict.fromkeys(scope_ids))

            answer = await expert_answer(question, scope_doc_ids=merged or None)
            return _render_expert_answer(answer)

        console.print(asyncio.run(_run()))

    @app.command(name="enrich")
    def enrich_cmd(doc_id: str) -> None:
        """Run entity extraction + graph linking for one document."""

        async def _run():
            bootstrap()
            return await enrich_document(doc_id)

        _print(asyncio.run(_run()))

    @app.command()
    def graph(
        document: str = _Option(..., "--document", "-d", help="doc_id."),
        limit: int = _Option(50, help="Max neighbors to print."),
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
    def related(
        document: str = _Option(..., "--document", "-d", help="doc_id."),
        limit: int = _Option(10, help="Max related documents to print."),
    ) -> None:
        """Explore connections: documents related to this one via SHARED ENTITIES,
        ranked by entity SPECIFICITY (a rare shared concept outranks a generic one).
        The on-mission discovery surface over the entity graph."""

        async def _run():
            from memex.core.config import get_settings

            bootstrap()
            store = await GraphStore.open(get_settings().vault_path)
            try:
                return await store.related_documents(document, limit=limit)
            finally:
                await store.close()

        results = asyncio.run(_run())
        for r in results:
            _print(r)

    @app.command()
    def cites(
        document: str = _Option(..., "--document", "-d", help="doc_id."),
    ) -> None:
        """References: the document's 1-hop CITES neighbourhood — what it cites + what
        cites it (the resolved IN-VAULT citations). Transitive chain-following is
        data-gated; this is the honest 1-hop surface."""

        async def _run():
            from memex.core.config import get_settings

            bootstrap()
            store = await GraphStore.open(get_settings().vault_path)
            try:
                return await store.citations(document)
            finally:
                await store.close()

        _print(asyncio.run(_run()))

    @app.command()
    def entity(
        name: str = _Argument(..., help="Entity name (case-insensitive)."),
        max_docs: int = _Option(50, help="Max mentioning documents to list."),
        cooccurring: int = _Option(15, help="Max co-occurring entities to surface."),
        k: int = _Option(10, "--passages", "-k", help="Max passages (full-text) to return."),
    ) -> None:
        """Everything about an entity: its graph profile (kind(s), the documents that
        mention it, the co-occurring concept neighbourhood) + representative passages.
        Documents + co-occurring concepts come from the entity graph; the passages come
        from full-text search of those documents. An unknown name falls back to a
        whole-corpus text search (`resolved=False`)."""

        async def _run():
            from memex.retrieve import entity_overview

            bootstrap()
            return await entity_overview(
                name, max_docs=max_docs, max_cooccurring=cooccurring, passages_k=k
            )

        _print(asyncio.run(_run()))

    @app.command()
    def doctor() -> None:
        """Health check: vault integrity, daemon reachability, breaker state."""

        async def _run() -> dict[str, object]:
            bootstrap()
            return await _doctor_report()

        _print(asyncio.run(_run()))

    @app.command()
    def search(
        query: str = _Argument(..., help="The search query."),
        k: int = _Option(10, help="Number of top chunks to return."),
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
        query_set: Path = _Argument(  # noqa: B008
            ..., help="Path to a JSON query set (see docs/eval-corpus-plan.md)."
        ),
        quick: bool = _Option(False, "--quick", help="Sample ~20% of queries."),
    ) -> None:
        """Run the eval harness against a query set."""

        async def _run():
            bootstrap()
            return await run_eval(query_set, quick=quick)

        _print(asyncio.run(_run()))

    @app.command(name="eval-parse")
    def eval_parse_cmd(
        corpus_dir: Path = _Argument(  # noqa: B008
            ...,
            exists=True,
            file_okay=False,
            help="Corpus dir of <doc>/ground-truth.md (+ manifest.json). "
            "Predicted markdown is read from the vault by doc_id, or a "
            "predicted.md sibling. See docs/eval-corpus-plan.md.",
        ),
    ) -> None:
        """Score parse fidelity (CER / WER / structural-F1) against
        hand-curated ground truth."""

        async def _run():
            bootstrap()
            return await run_parse_eval(corpus_dir)

        _print(asyncio.run(_run()))

    @app.command(name="eval-summary")
    def eval_summary_cmd(
        query_set: Path = _Argument(  # noqa: B008
            ...,
            exists=True,
            dir_okay=False,
            help="JSON of summary-eval cases: per-doc must_mention / must_not_assert "
            "/ should_summarize. Runs `summarize_document` and scores recall + the "
            "no-leak HARD gate. See docs/eval-corpus-plan.md.",
        ),
    ) -> None:
        """Score grounded document summaries (ADR-0008): mention-recall + the
        no-hallucination (must_not_assert) gate + summarize/refuse correctness."""

        async def _run():
            bootstrap()
            return await run_summary_eval(query_set)

        _print(asyncio.run(_run()))

    @app.command(name="eval-chat")
    def eval_chat_cmd(
        query_set: Path = _Argument(  # noqa: B008
            ...,
            exists=True,
            dir_okay=False,
            help="JSON of multi-turn chat cases: per-case history + follow_up + the gold "
            "relevant_chunk_ids the rewritten query should retrieve. Measures the "
            "query-rewrite's gold-chunk recall (the A-1.5 follow-up-resolution metric). "
            "See docs/specs/grounded-agentic-chat.md §9.1.",
        ),
    ) -> None:
        """Score the grounded-chat query-rewrite: gold-chunk recall@k of `hybrid_search`
        on each follow-up's rewritten query (retrieval-isolated, no rerank/LLM)."""

        async def _run():
            bootstrap()
            return await run_chat_eval(query_set)

        _print(asyncio.run(_run()))

    @app.command()
    def upgrade(
        no_restart: bool = _Option(
            False,
            "--no-restart",
            help="Skip the systemd restart step (Pattern B / Pattern C boxes).",
        ),
        skip_sync: bool = _Option(
            False,
            "--skip-sync",
            help="Skip `uv sync` (git pull + restart only).",
        ),
        dry_run: bool = _Option(
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
                    "uv",
                    "sync",
                    "--extra",
                    "models",
                    "--extra",
                    "parse",
                    "--extra",
                    "serve",
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
    app.add_typer(scope_set_app, name="scope-set")
    app.add_typer(mode_app, name="mode")


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
        # systemctl is PATH-resolved (its location varies across distros);
        # argv is a fixed literal list, no shell — S607 is a false positive.
        result = subprocess.run(
            [  # noqa: S607
                "systemctl",
                "--user",
                "list-unit-files",
                "memex-*.service",
                "--no-legend",
                "--no-pager",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
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
    completed = subprocess.run(argv, cwd=cwd, check=False)  # noqa: S603  # caller-supplied literal argv, no shell
    if completed.returncode != 0:
        err.print(f"[red]✗ {label} failed (exit {completed.returncode})[/red]")
        raise typer.Exit(code=completed.returncode)


@mcp_app.command("generate-token")
def mcp_generate_token(
    length: int = _Option(
        32,
        "--length",
        min=16,
        max=128,
        help="Token length in bytes (urlsafe-encoded; the printed string is ~33% longer).",
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


@scope_set_app.command("create")
def scope_set_create(
    name: str = _Argument(..., help="A human label for the set (case-insensitive)."),
    doc: list[str] = _Option(  # noqa: B008  # typer Option default sentinel
        [],
        "--doc",
        help="A document id to include (repeatable). Run `memex search` or open the web "
        "UI to find ids.",
    ),
) -> None:
    """Create or update a saved scope set.

    Reapply it later with `memex ask --scope-set NAME`. Document ids are
    validated against the vault — an unknown id is rejected so a typo can't
    create a set that silently scopes to nothing.
    """

    async def _run() -> dict[str, object]:
        bootstrap()
        from memex.core.config import get_settings
        from memex.core.scope_sets import save_scope_set
        from memex.vault.store import list_documents

        vault_path = get_settings().vault_path
        known: set[str] = set()
        async for ref in list_documents(vault_path):
            known.add(ref.doc_id)
        requested = [d.strip() for d in doc if d.strip()]
        unknown = [d for d in requested if d not in known]
        if unknown:
            err.print(f"[red]Unknown document id(s):[/red] {', '.join(unknown)}")
            raise typer.Exit(code=2)
        record = await save_scope_set(vault_path, name, requested)
        return {"name": record.name, "doc_ids": record.doc_ids, "count": len(record.doc_ids)}

    _print(asyncio.run(_run()))


@scope_set_app.command("list")
def scope_set_list() -> None:
    """List every saved scope set (name + document count)."""

    async def _run() -> list[dict[str, object]]:
        bootstrap()
        from memex.core.config import get_settings
        from memex.core.scope_sets import list_scope_sets

        vault_path = get_settings().vault_path
        return [
            {"name": s.name, "count": len(s.doc_ids), "updated_at": s.updated_at.isoformat()}
            for s in await list_scope_sets(vault_path)
        ]

    _print(asyncio.run(_run()))


@scope_set_app.command("show")
def scope_set_show(
    name: str = _Argument(..., help="The set name (case-insensitive)."),
) -> None:
    """Show a scope set's documents (id + title)."""

    async def _run() -> dict[str, object]:
        bootstrap()
        from memex.core.config import get_settings
        from memex.core.scope_sets import get_scope_set
        from memex.vault.store import read_document_title

        vault_path = get_settings().vault_path
        found = await get_scope_set(vault_path, name)
        if found is None:
            return {"name": name, "found": False}
        docs = [
            {"doc_id": d, "title": await read_document_title(vault_path, d)} for d in found.doc_ids
        ]
        return {"name": found.name, "found": True, "documents": docs}

    _print(asyncio.run(_run()))


@scope_set_app.command("delete")
def scope_set_delete(
    name: str = _Argument(..., help="The set name (case-insensitive)."),
    yes: bool = _Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Delete a saved scope set. Removes only the named collection — no document
    is touched."""
    if not yes:
        typer.confirm(f"Delete the saved scope set {name!r}?", abort=True)

    async def _run() -> dict[str, object]:
        bootstrap()
        from memex.core.config import get_settings
        from memex.core.scope_sets import delete_scope_set

        vault_path = get_settings().vault_path
        removed = await delete_scope_set(vault_path, name)
        return {"name": name, "deleted": removed}

    _print(asyncio.run(_run()))


@serve_app.command("mcp")
def serve_mcp(
    transport: str = _Option(
        "stdio",
        "--transport",
        help="`stdio` for desktop clients (Claude Code, Cursor) or "
        "`http` for network-local agents (binds 127.0.0.1 by default).",
    ),
    host: str = _Option("127.0.0.1", help="Bind host (http transport only)."),
    port: int = _Option(7424, help="Bind port (http transport only)."),
) -> None:
    """Run the Memex MCP server."""
    from memex.cli.bootstrap import bootstrap as _boot
    from memex.mcp.server import serve_http, serve_stdio

    _boot()
    if transport == "stdio":
        # Don't print to stdout on stdio transport — that channel IS the
        # MCP protocol. stderr is fine.
        err.print(
            "[green]Memex MCP server ready[/green] [dim](stdio transport, Ctrl-C to stop)[/dim]"
        )
        asyncio.run(serve_stdio())
    elif transport == "http":
        err.print(
            f"[green]Memex MCP server ready →[/green] [bold]http://{host}:{port}[/bold]   "
            f"[dim](Ctrl-C to stop)[/dim]"
        )
        asyncio.run(serve_http(host=host, port=port))
    else:
        err.print(f"[red]unknown transport {transport!r}; expected stdio or http[/red]")
        raise typer.Exit(code=2)


@serve_app.command("web")
def serve_web(
    host: str = _Option("127.0.0.1", help="Bind host."),
    port: int = _Option(7423, help="Bind port."),
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
            issues.append(f"{ref.doc_id}: content_sha256 drifted (user edit?)")
        # Count a doc as stale-index at most ONCE even if it has BOTH a drifted
        # content hash AND a missing index stage (the two used to double-count).
        if manifest.content_sha256 != actual or manifest.index is None:
            stale_index += 1

    # Daemon probe.
    try:
        client = get_client()
        models = await client.models.list()
        served = [m.id for m in getattr(models, "data", [])]
        # Surface a model-id desync: the client sends settings.models.orchestrator,
        # so if the daemon is serving a DIFFERENT model every /ask 404s silently.
        # `orchestrator_match=False` is the loud signal that the serve-env bridge
        # (daemon/supervisor.orchestrator_serve_env) didn't take.
        daemon = {
            "reachable": True,
            "base_url": settings.inference.base_url,
            "models": served,
            "orchestrator_expected": settings.models.orchestrator,
            "orchestrator_match": settings.models.orchestrator in served,
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
    prompts = [{"name": p.name, "version": p.version} for p in list_prompts()]

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
        err.print(f"[red]see log: {e.context.get('log_file')}[/red]")
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


@daemon_app.command("restart")
def daemon_restart(
    gpu_fraction: float | None = _Option(
        None, "--gpu-fraction", min=0.1, max=1.0, help="New vLLM gpu-memory-utilization."
    ),
    max_model_len: int | None = _Option(
        None, "--max-model-len", min=512, help="New vLLM max-model-len (context window)."
    ),
) -> None:
    """Stop + restart the vLLM daemon, optionally re-pointing the GPU fraction
    and context window (the orchestrator half of a co-residence mode)."""
    from memex.cli.bootstrap import bootstrap as _boot
    from memex.daemon import DaemonStartTimeout
    from memex.daemon import restart as _restart

    settings = _boot()
    err.print("[dim]restarting vLLM daemon…[/dim]")
    try:
        status = asyncio.run(
            _restart(settings, gpu_fraction=gpu_fraction, max_model_len=max_model_len)
        )
    except DaemonStartTimeout as e:
        err.print(f"[red]{e}[/red]")
        err.print(f"[red]see log: {e.context.get('log_file')}[/red]")
        raise typer.Exit(code=1) from e
    _print(status)


@mode_app.command("show")
def mode_show() -> None:
    """Print the active co-residence resource profile (ADR-0007)."""
    from memex.cli.bootstrap import bootstrap as _boot
    from memex.core.config import get_settings
    from memex.core.resources import resolve_profile

    _boot()
    s = get_settings()
    profile = resolve_profile(
        s.models.co_residence_mode,
        embedder_device=s.models.embedder_device,
        reranker_device=s.models.reranker_device,
    )
    _print(profile.model_dump())


@mode_app.command("set")
def mode_set(
    mode: str = _Argument(..., help="fast | full | gpu_only | manual"),
) -> None:
    """Apply a co-residence mode (ADR-0007).

    If the orchestrator daemon is running and the mode prescribes a posture,
    restarts the daemon with the mode's GPU fraction + context window (applies
    NOW). Prints the env/config line to make the mode durable — set
    `MEMEX_MODELS__CO_RESIDENCE_MODE` and restart `memex serve web` for the
    retrieval-device change to take effect.
    """
    valid = ("fast", "full", "gpu_only", "manual")
    if mode not in valid:
        err.print(f"[red]Unknown mode {mode!r}.[/red] Choose one of: {', '.join(valid)}")
        raise typer.Exit(code=2)

    async def _run() -> dict[str, object]:
        bootstrap()
        from memex.core.config import config_toml_path, get_settings
        from memex.core.resources import resolve_profile
        from memex.daemon import restart as _restart
        from memex.daemon import status as _status

        s = get_settings()
        # `mode` is narrowed to the CoResidenceMode literal union by the guard above.
        profile = resolve_profile(
            mode,
            embedder_device=s.models.embedder_device,
            reranker_device=s.models.reranker_device,
        )
        result: dict[str, object] = {
            "mode": mode,
            "embedder_device": profile.embedder_device,
            "reranker_device": profile.reranker_device,
            "orchestrator_gpu_fraction": profile.orchestrator_gpu_fraction,
            "orchestrator_max_model_len": profile.orchestrator_max_model_len,
        }
        daemon_state = await _status(s)
        if profile.orchestrator_gpu_fraction is not None and daemon_state.alive:
            err.print("[dim]applying orchestrator posture (restarting daemon)…[/dim]")
            new_state = await _restart(
                s,
                gpu_fraction=profile.orchestrator_gpu_fraction,
                max_model_len=profile.orchestrator_max_model_len,
            )
            result["orchestrator_restarted"] = new_state.reachable
        else:
            result["orchestrator_restarted"] = False
            if profile.orchestrator_gpu_fraction is not None:
                err.print(
                    "[dim]orchestrator not daemon-managed/running; `memex daemon start` "
                    "to apply its util/context.[/dim]"
                )
        # Durable: this CLI applies the posture transiently; persist the mode so
        # the next `memex serve web` picks up the retrieval-device placement.
        result["persist"] = (
            f"set MEMEX_MODELS__CO_RESIDENCE_MODE={mode} (env) or add "
            f'`co_residence_mode = "{mode}"` under [models] in {config_toml_path()}, '
            "then restart `memex serve web`"
        )
        return result

    _print(asyncio.run(_run()))
