#!/usr/bin/env python3
"""Verify every CSS class used in the webui templates (+ the runtime HTML built in
`graph.js`) has a matching rule in the vendored CSS (`tailwind.css` / `style.css` /
`graph.css`).

The webui ships a HAND-CURATED Tailwind subset (no JIT at runtime), so a class a template
uses but the subset omits silently NO-OPS — the component renders unstyled with no error.
This scanner is the safety net the `tailwind.css` header promises; run it in CI / pre-commit.

Exit 1 and list the misses if any used class is undefined; exit 0 if clean.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

WEBUI = Path(__file__).resolve().parent.parent / "src" / "memex" / "webui"
TEMPLATES = WEBUI / "templates"
STATIC = WEBUI / "static"
CSS_FILES = [STATIC / "tailwind.css", STATIC / "style.css", STATIC / "graph.css"]

_CLASS_ATTR = re.compile(r'class="([^"]*)"')
# Strip Jinja `{{ }}` / `{% %}` control+expression blocks and JS `${ }` interpolation from a
# class attribute BEFORE tokenizing, so `{% if x == 'high' %}conf-high{% endif %}` contributes
# only `conf-high`, not the control-flow fragments `if`/`=='/`endif`.
_INTERP = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|\$\{.*?\}", re.DOTALL)
# Conventional markers that intentionally have no CSS rule of their own.
_EXEMPT = {"dark"}  # the Tailwind dark-mode root class on <html> (no `dark:` utilities yet)


def _tailwind_escape(cls: str) -> str:
    """The CSS selector Tailwind emits for a class token: each of : / . [ ] is backslash-escaped
    and the whole thing prefixed with a dot (e.g. `hover:bg-red-800` → `.hover\\:bg-red-800`)."""
    return "." + re.sub(r"([:/.\[\]])", r"\\\1", cls)


def used_classes() -> set[str]:
    out: set[str] = set()
    for f in [*sorted(TEMPLATES.glob("*.html")), STATIC / "graph.js"]:
        if not f.exists():
            continue
        for m in _CLASS_ATTR.finditer(f.read_text(encoding="utf-8")):
            for tok in _INTERP.sub(" ", m.group(1)).split():
                if tok and not any(c in tok for c in "{}$"):
                    out.add(tok)
    return out


def main() -> int:
    css = "\n".join(f.read_text(encoding="utf-8") for f in CSS_FILES if f.exists())
    missing = sorted(c for c in used_classes() if c not in _EXEMPT and _tailwind_escape(c) not in css)
    if missing:
        print(f"[check-tailwind] {len(missing)} class(es) used in templates/graph.js with NO rule:")
        for c in missing:
            print(f"  - {c}   (expected selector {_tailwind_escape(c)} in tailwind.css/style.css/graph.css)")
        print("[check-tailwind] add the rule(s) to static/tailwind.css (or use a defined class).")
        return 1
    print(f"[check-tailwind] OK — all {len(used_classes())} used classes have a rule.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
