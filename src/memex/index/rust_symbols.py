"""Lightweight, line-scannable Rust symbol splitter (codebase-corpus arc, Phase 2).

Phase 1 ingests a `.rs` file VERBATIM (canonical `.md` = raw source). This module finds
TOP-LEVEL and impl/trait/mod-METHOD symbol boundaries so the index pipeline can inject
synthetic `## <symbol>` / `### <Type::method>` ATX headings into the TRANSIENT body the
chunker sees (NEVER the canonical `.md`) — driving the existing markdown section-splitter to
produce one chunk per symbol with `heading_path = [impl Foo, Foo::bar]`. That symbol becomes
the EmbeddingGemma doc-prompt title (`chunk_title` = deepest `heading_path`) and the citation
label — the razor-sharp-retrieval lever.

Design constraints:
- **No tree-sitter / no new dependency** (air-gap mandate). Pure stdlib: a small masking lexer
  + brace-depth line scan + regex.
- **Bounded failure**: a missed boundary only yields a COARSER chunk; content is NEVER dropped
  or altered. Any unexpected error → `split_rust_symbols` returns `[]` (the doc chunks as one
  prose blob, exactly today's behaviour). The masking lexer exists so a `{`/`}`/`fn` inside a
  string (`format!("{}")`), comment, char-literal (`'{'`), or raw string (`r#"..."#`) can't
  corrupt the brace depth or trigger a false symbol.
- **Pure + sync + no I/O.**
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Symbol:
    """A detected Rust symbol boundary.

    `start_line` is the 0-based line index to inject the heading BEFORE (already backed up over
    leading `#[...]` attributes + `///`/`//!` doc-comments so they travel with the symbol).
    `level` is the ATX heading level (2 = top-level `##`, 3 = impl/trait/mod method `###`).
    """

    start_line: int
    label: str
    level: int


# --- masking lexer: blank out string/comment/char-literal content (keep newlines + length) ---


def _mask_code(body: str) -> str:
    """Return a same-length copy of `body` with the CONTENTS of string literals, char literals,
    line/block comments, and raw strings replaced by spaces (newlines preserved). Real code —
    crucially the structural braces `{`/`}` and the symbol keywords — survives; a `{` inside
    `"{}"`, `// {`, `/* { */`, `'{'`, or `r#"{"#` does not. Rust block comments NEST."""
    out: list[str] = []
    i = 0
    n = len(body)
    state = "normal"
    block_depth = 0
    raw_hashes = 0

    def blank(ch: str) -> str:
        return "\n" if ch == "\n" else " "

    while i < n:
        c = body[i]
        nxt = body[i + 1] if i + 1 < n else ""
        if state == "normal":
            # raw string: r"..." / r#"..."# / br#"..."# (variable '#' count)
            if c in ("r", "b"):
                j = i
                if c == "b" and nxt == "r":
                    j = i + 1
                if body[j] == "r":
                    k = j + 1
                    h = 0
                    while k < n and body[k] == "#":
                        h += 1
                        k += 1
                    if k < n and body[k] == '"':
                        raw_hashes = h
                        state = "raw"
                        for p in range(i, k + 1):
                            out.append(blank(body[p]))
                        i = k + 1
                        continue
            if c == "/" and nxt == "/":
                state = "line_comment"
                out.append(" ")
                out.append(" ")
                i += 2
                continue
            if c == "/" and nxt == "*":
                state = "block_comment"
                block_depth = 1
                out.append(" ")
                out.append(" ")
                i += 2
                continue
            if c == '"':
                state = "string"
                out.append(" ")
                i += 1
                continue
            if c == "'":
                # char literal ('x' / '\n' / '\u{..}') vs lifetime ('a). Only char literals can
                # contain a brace; mask them. A lifetime has no closing ' nearby → leave normal.
                close = body.find("'", i + 1)
                seg = body[i : close + 1] if close != -1 else ""
                is_char = close != -1 and (
                    (body[i + 1 : i + 2] == "\\" and len(seg) <= 12) or (close == i + 2)
                )
                if is_char:
                    for p in range(i, close + 1):
                        out.append(blank(body[p]))
                    i = close + 1
                    continue
                out.append(c)
                i += 1
                continue
            out.append(c)
            i += 1
            continue
        if state == "line_comment":
            if c == "\n":
                state = "normal"
                out.append("\n")
            else:
                out.append(" ")
            i += 1
            continue
        if state == "block_comment":
            if c == "/" and nxt == "*":
                block_depth += 1
                out.append(" ")
                out.append(" ")
                i += 2
                continue
            if c == "*" and nxt == "/":
                block_depth -= 1
                out.append(" ")
                out.append(" ")
                i += 2
                if block_depth == 0:
                    state = "normal"
                continue
            out.append(blank(c))
            i += 1
            continue
        if state == "string":
            if c == "\\":
                out.append(" ")
                if nxt:
                    out.append(blank(nxt))
                i += 2
                continue
            if c == '"':
                state = "normal"
                out.append(" ")
                i += 1
                continue
            out.append(blank(c))
            i += 1
            continue
        if state == "raw":
            if c == '"' and body[i + 1 : i + 1 + raw_hashes] == "#" * raw_hashes:
                out.append(" ")
                for _ in range(raw_hashes):
                    out.append(" ")
                i += 1 + raw_hashes
                state = "normal"
                continue
            out.append(blank(c))
            i += 1
            continue
    return "".join(out)


# --- symbol keyword patterns (applied to the lstripped MASKED line) ---

_VIS = r"(?:pub(?:\([^)]*\))?\s+)?"
_FN_RE = re.compile(_VIS + r"(?:(?:async|const|unsafe|extern)\b[^{};]*?\s)?fn\s+([A-Za-z_]\w*)")
_STRUCT_RE = re.compile(_VIS + r"struct\s+([A-Za-z_]\w*)")
_ENUM_RE = re.compile(_VIS + r"enum\s+([A-Za-z_]\w*)")
_UNION_RE = re.compile(_VIS + r"union\s+([A-Za-z_]\w*)")
_TRAIT_RE = re.compile(_VIS + r"(?:unsafe\s+)?trait\s+([A-Za-z_]\w*)")
_TYPE_RE = re.compile(_VIS + r"type\s+([A-Za-z_]\w*)")
_MOD_RE = re.compile(_VIS + r"mod\s+([A-Za-z_]\w*)")
_CONST_RE = re.compile(_VIS + r"const\s+([A-Za-z_]\w*)")
_STATIC_RE = re.compile(_VIS + r"static\s+(?:mut\s+)?([A-Za-z_]\w*)")
_MACRO_RE = re.compile(r"macro_rules!\s+([A-Za-z_]\w*)")
# impl header: capture everything up to the body `{` or a `where` clause.
_IMPL_RE = re.compile(r"(?:unsafe\s+)?impl(?:<[^>]*>)?\s+(.+?)\s*(?:\{|\bwhere\b|$)")

_CONTAINER_KINDS = ("impl", "trait", "mod")
_PREAMBLE_PREFIXES = ("///", "//!", "#[", "#![")


def _match_symbol(line: str) -> tuple[str, str] | None:
    """Match a Rust item at the start of an lstripped, masked line. Returns `(kind, payload)`
    where payload is the symbol NAME (or, for `impl`, the raw header text). `fn` is tried before
    `const`/`type` so `const fn` / `async fn` classify as functions."""
    for kind, rx in (
        ("fn", _FN_RE),
        ("struct", _STRUCT_RE),
        ("enum", _ENUM_RE),
        ("union", _UNION_RE),
        ("trait", _TRAIT_RE),
        ("mod", _MOD_RE),
        ("type", _TYPE_RE),
        ("const", _CONST_RE),
        ("static", _STATIC_RE),
        ("macro_rules", _MACRO_RE),
    ):
        m = rx.match(line)
        if m:
            return kind, m.group(1)
    m = _IMPL_RE.match(line)
    if m:
        return "impl", m.group(1).strip()
    return None


def _top_label(kind: str, payload: str) -> str:
    if kind == "impl":
        return f"impl {payload}"
    if kind == "macro_rules":
        return f"macro_rules! {payload}"
    return f"{kind} {payload}"


def _qual_type(kind: str, payload: str) -> str:
    """The type used to qualify a container's methods (`Type::method`)."""
    if kind in ("trait", "mod"):
        return payload
    # impl: the type after ` for ` (the impl-ed type) else the type after `impl`; strip generics.
    header = payload
    tail = header.split(" for ", 1)[1] if " for " in header else header
    tail = re.sub(r"<.*$", "", tail).strip()
    seg = tail.split("::")[-1].strip()
    return seg or payload


def _preamble_start(raw_lines: list[str], masked_lines: list[str], sym_line: int) -> int:
    """Back up from `sym_line` over contiguous leading `#[...]` attributes (incl. multi-line,
    tracked by bracket balance) and `///`/`//!` doc-comments, stopping at a blank line or a
    non-preamble code line — so the injected heading lands ABOVE the docs/attrs."""
    start = sym_line
    balance = 0
    j = sym_line - 1
    while j >= 0:
        stripped = raw_lines[j].strip()
        ml = masked_lines[j]
        net = ml.count("(") + ml.count("[") - ml.count(")") - ml.count("]")
        is_doc_or_attr = stripped.startswith(_PREAMBLE_PREFIXES)
        attr_closer = net < 0 and "]" in ml
        if balance < 0 or attr_closer or is_doc_or_attr:
            balance += net
            start = j
            j -= 1
            continue
        break
    return start


def split_rust_symbols(body: str) -> list[Symbol]:
    """Find Rust symbol boundaries in `body`. Returns symbols in source order; `[]` on any
    failure (the caller then chunks the doc as prose — coarser, never wrong)."""
    try:
        return _split_rust_symbols(body)
    except Exception as e:  # bounded failure: never break indexing on a parser quirk
        logger.warning("rust_symbols.split_failed", error=str(e), error_type=type(e).__name__)
        return []


def _split_rust_symbols(body: str) -> list[Symbol]:
    masked = _mask_code(body)
    masked_lines = masked.split("\n")
    raw_lines = body.split("\n")
    # Brace depth at the START of each line (computed on the masked body).
    depth_at: list[int] = []
    d = 0
    for ml in masked_lines:
        depth_at.append(d)
        d += ml.count("{") - ml.count("}")

    symbols: list[Symbol] = []
    pending_container_type: str | None = None
    active_container_type: str | None = None
    prev_depth = 0
    for i, ml in enumerate(masked_lines):
        cur = depth_at[i]
        if cur == 0:
            active_container_type = None
        elif cur == 1 and prev_depth == 0:
            active_container_type = pending_container_type
        m = _match_symbol(ml.lstrip())
        if m:
            kind, payload = m
            if cur == 0:
                start = _preamble_start(raw_lines, masked_lines, i)
                symbols.append(Symbol(start_line=start, label=_top_label(kind, payload), level=2))
                pending_container_type = (
                    _qual_type(kind, payload) if kind in _CONTAINER_KINDS else None
                )
            elif cur == 1 and active_container_type and kind in ("fn", "const", "type"):
                start = _preamble_start(raw_lines, masked_lines, i)
                symbols.append(
                    Symbol(start_line=start, label=f"{active_container_type}::{payload}", level=3)
                )
        prev_depth = cur
    return symbols


def inject_symbol_headings(body: str, symbols: list[Symbol]) -> str:
    """Prepend each symbol's `("#" * level) + " " + label` line immediately before its
    (preamble-adjusted) start line. The TRANSIENT result feeds the chunker only; the canonical
    `.md` is untouched. The heading line BECOMES part of the chunk text → the content-addressed
    `chunk_id` → so the label/framing is LOAD-BEARING (golden-tested)."""
    if not symbols:
        return body
    lines = body.split("\n")
    inserts: dict[int, list[Symbol]] = {}
    for sym in symbols:
        inserts.setdefault(sym.start_line, []).append(sym)
    out: list[str] = []
    for i, line in enumerate(lines):
        for sym in sorted(inserts.get(i, []), key=lambda s: s.level):
            out.append(("#" * sym.level) + " " + sym.label)
        out.append(line)
    return "\n".join(out)
