"""EmbeddingGemma native `task:`/`title:` prompt helpers (pure-sync).

EmbeddingGemma is a prompt-trained asymmetric retriever; its model card
prescribes wrapping the embedding INPUT (never the stored text):

- queries   → ``task: search result | query: {q}``  (the model's built-in
  ``query`` prompt, applied via sentence-transformers ``prompt_name=``)
- documents → ``title: {title|"none"} | text: {chunk}``  (built MANUALLY —
  the built-in ``document`` prompt is hardcoded ``title: none | text: ``
  and would discard the heading)

The wrapper touches ONLY the transient string handed to ``encode``;
``chunk.text``, ``chunk_id``, and the stored/retrieved chunk text are
unchanged, so the prompt never reaches assess/answer/verify — it only
changes which chunks retrieve. See ``docs/specs/embedding-native-prompts.md``.

If a future embedder's ST config lacks the ``query`` prompt, fall back to a
manual ``task: search result | query: `` prefix string at the call site.
"""

from __future__ import annotations

import os

from memex.core.types import Chunk

# EmbeddingGemma's built-in query prompt (`task: search result | query: `),
# applied via sentence-transformers `prompt_name=`. `search result` is the
# retrieval task — correct for RAG (not `question answering`/`fact checking`).
EMBED_QUERY_PROMPT_NAME = "query"

# The literal text of EmbeddingGemma's `query` prompt. Used as a MANUAL prefix
# when a future embedder's ST config lacks the registered `query` prompt (so the
# `prompt_name=` path raises). The fallback prepends this string verbatim so the
# query side still embeds in the model's trained query distribution. Kept here
# (not duplicated at the call site) so the docstring promise and the fallback
# agree on one source of truth.
EMBED_QUERY_PROMPT_TEXT = "task: search result | query: "

# Max length of the chosen title before the `title: X | text:` delimiter, so a
# pathological heading can't dominate the embedding input.
_TITLE_MAX_CHARS = 80


def document_input(title: str, text: str) -> str:
    """The document-side embedding input: ``title: {title} | text: {text}``.

    Built manually (not ``prompt_name="document"``, whose built-in is the
    hardcoded ``title: none | text: `` that would discard the heading).
    """
    return f"title: {title} | text: {text}"


def _sanitize_title(title: str) -> str:
    """Make *title* safe for the ``title: X | text:`` slot.

    A heading containing the ``" | "`` delimiter (or a newline) would corrupt
    EmbeddingGemma's title/text split, so we replace ``" | "`` with ``" / "``
    and collapse newlines to spaces, then clamp to ``_TITLE_MAX_CHARS``.
    """
    cleaned = title.replace(" | ", " / ").replace("\r", " ").replace("\n", " ").strip()
    if len(cleaned) > _TITLE_MAX_CHARS:
        cleaned = cleaned[:_TITLE_MAX_CHARS].rstrip()
    return cleaned


def chunk_title(chunk: Chunk) -> str:
    """The most-specific locating signal for a chunk's title slot.

    The deepest NON-EMPTY (stripped) ``heading_path`` entry if any, else the
    ``document_title`` if non-empty, else ``"none"`` (EmbeddingGemma's trained
    no-title sentinel). A malformed ``heading_path == [""]`` (e.g. from a
    double-space ``##  `` heading) must NOT yield an empty title (which would
    render ``title:  | text:``), so we pick the deepest non-empty entry rather
    than trust list-truthiness. The chosen title is sanitized so a heading
    containing the ``" | "`` delimiter or a newline can't corrupt the
    title/text split, and clamped to a reasonable length.
    """
    for entry in reversed(chunk.heading_path):
        if entry.strip():
            return _sanitize_title(entry)
    if chunk.document_title.strip():
        return _sanitize_title(chunk.document_title)
    return "none"


def native_prompts_enabled() -> bool:
    """Whether to wrap embedding inputs in EmbeddingGemma's native prompts.

    Default ON; ``MEMEX_EMBED_NATIVE_PROMPTS=0`` reverts to bare (the A/B /
    revert path). Read per-call (mirrors ``MEMEX_INDEX_EMBED_BATCH``).
    """
    return os.environ.get("MEMEX_EMBED_NATIVE_PROMPTS", "1") != "0"
