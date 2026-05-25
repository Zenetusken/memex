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


def document_input(title: str, text: str) -> str:
    """The document-side embedding input: ``title: {title} | text: {text}``.

    Built manually (not ``prompt_name="document"``, whose built-in is the
    hardcoded ``title: none | text: `` that would discard the heading).
    """
    return f"title: {title} | text: {text}"


def chunk_title(chunk: Chunk) -> str:
    """The most-specific locating signal for a chunk's title slot.

    Deepest ``heading_path[-1]`` if present, else the ``document_title``
    if truthy, else ``"none"`` (EmbeddingGemma's trained no-title sentinel).
    Short either way → no homogenization risk.
    """
    if chunk.heading_path:
        return chunk.heading_path[-1]
    if chunk.document_title:
        return chunk.document_title
    return "none"


def native_prompts_enabled() -> bool:
    """Whether to wrap embedding inputs in EmbeddingGemma's native prompts.

    Default ON; ``MEMEX_EMBED_NATIVE_PROMPTS=0`` reverts to bare (the A/B /
    revert path). Read per-call (mirrors ``MEMEX_INDEX_EMBED_BATCH``).
    """
    return os.environ.get("MEMEX_EMBED_NATIVE_PROMPTS", "1") != "0"
