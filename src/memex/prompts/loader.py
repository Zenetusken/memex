"""Prompt loader — see IMPLEMENTATION-PLAN.md §1.7.

Resolves a prompt name to its highest-version on-disk template,
validates kwargs against the declared input schema, renders via
Jinja2, and returns the tagged string. The version-pinning
environment variable (`MEMEX_PROMPTS__PIN__<name>=<version>`) lets
eval runs lock a baseline.

On-disk layout under `src/memex/prompts/<name>/v<N>.md`:

    ---
    name: answer
    version: v1
    role: user
    target_model: Qwen/Qwen3-8B-AWQ
    ---
    {{ Jinja template body }}

The loader walks `src/memex/prompts/<name>/` for files matching
`v<int>.md` and picks the highest `<int>` unless pinned. The
rendered output is suffixed with a comment line carrying the tag
(`<!-- prompt: name@v1 -->`) — the model ignores it, but it lands
in Langfuse's recorded prompt and lets us trace which version
produced which output.
"""

from __future__ import annotations

import os
import re
from functools import cache
from pathlib import Path
from typing import Any, Literal

import frontmatter
from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel, Field

from memex.core.errors import MemexError

_PROMPTS_ROOT = Path(__file__).parent
_VERSION_RE = re.compile(r"^v(\d+)\.md$")

_jinja = Environment(
    undefined=StrictUndefined,
    autoescape=False,  # prompts are not HTML
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)


class PromptSpec(BaseModel):
    name: str
    version: str
    role: Literal["system", "user"] = "user"
    target_model: str | None = None
    input_schema: str | None = None
    output_schema: str | None = None
    eval_suite: str | None = None
    template: str = Field(exclude=True)  # not serialised


class PromptNotFoundError(MemexError):
    """No on-disk template matches the requested name (and pinned version)."""


def _candidate_dir(name: str) -> Path:
    return _PROMPTS_ROOT / name


def _versions(dir_path: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    if not dir_path.exists():
        return out
    for f in dir_path.iterdir():
        m = _VERSION_RE.match(f.name)
        if m:
            out.append((int(m.group(1)), f))
    return sorted(out)


def _pinned_version(name: str) -> str | None:
    key = f"MEMEX_PROMPTS__PIN__{name.upper().replace('-', '_')}"
    return os.environ.get(key)


@cache
def load_prompt_spec(name: str) -> PromptSpec:
    """Load and parse the active version of `name`."""
    dir_path = _candidate_dir(name)
    versions = _versions(dir_path)
    if not versions:
        raise PromptNotFoundError(
            f"no prompt template found for name={name!r}",
            context={"search_path": str(dir_path)},
        )

    pinned = _pinned_version(name)
    selected: Path
    if pinned:
        match = next((p for (n, p) in versions if f"v{n}" == pinned), None)
        if match is None:
            raise PromptNotFoundError(
                f"pinned version {pinned!r} of prompt {name!r} not found",
                context={"available": [f"v{n}" for (n, _) in versions]},
            )
        selected = match
    else:
        selected = versions[-1][1]

    post = frontmatter.loads(selected.read_text(encoding="utf-8"))
    meta = dict(post.metadata)
    meta.setdefault("name", name)
    meta.setdefault("version", selected.stem)
    return PromptSpec(template=post.content, **meta)


def render_prompt(name: str, **kwargs: Any) -> str:
    """Render the active version of `name` with `kwargs`.

    Appends a tag comment for trace correlation. Raises if a referenced
    variable is missing (StrictUndefined) so template/caller drift
    fails loudly.
    """
    spec = load_prompt_spec(name)
    template = _jinja.from_string(spec.template)
    rendered = template.render(**kwargs)
    tag = f"<!-- prompt: {spec.name}@{spec.version} -->"
    return f"{rendered.rstrip()}\n\n{tag}\n"


def list_prompts() -> list[PromptSpec]:
    """Discover every prompt under `prompts/`. For `memex doctor`."""
    out: list[PromptSpec] = []
    for child in sorted(_PROMPTS_ROOT.iterdir()):
        if child.is_dir() and _versions(child):
            out.append(load_prompt_spec(child.name))
    return out
