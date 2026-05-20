"""Vault — Markdown read/write, frontmatter, atomic writes.

The vault is the source of truth (ADR-0003). This module owns reading
and writing `vault/documents/{id}.md`, parsing YAML frontmatter,
handling wikilinks, and the atomic-write semantics that keep partial
writes out of the canonical store.
"""

from memex.vault.store import (
    DocumentRef,
    Frontmatter,
    VaultDocument,
    assign_doc_id,
    create_document,
    delete_document,
    hash_bytes,
    list_documents,
    make_ref,
    read_document,
    write_document,
)

__all__ = [
    "DocumentRef",
    "Frontmatter",
    "VaultDocument",
    "assign_doc_id",
    "create_document",
    "delete_document",
    "hash_bytes",
    "list_documents",
    "make_ref",
    "read_document",
    "write_document",
]
