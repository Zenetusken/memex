"""The prompt_tag auto-derive helpers (2026-06-06) KILL the version-drift class:
a hardcoded `prompt_tag="<name>@vN"` in code drifts the moment `render_prompt`
auto-selects a higher on-disk version (or a PIN is set), so the Langfuse trace
label silently lies about which prompt actually ran. `prompt_tag_for(name)` /
`active_version(name)` derive the label from the loaded `PromptSpec.version`
instead, so it can never be wrong again.

The load-bearing test here is `test_no_hardcoded_version_tags_in_source` — a
source scan that fails if anyone reintroduces the literal antipattern. That is
what makes the fix PERMANENT rather than a one-off cleanup of the 3 instances we
found by hand.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import memex
from memex.prompts import active_version, prompt_tag_for
from memex.prompts.loader import PromptNotFoundError, load_prompt_spec

# The antipattern the auto-derive eliminates: a string-literal `name@vN` passed
# as `prompt_tag=`. Mirrors the regex in the plan; whitespace-tolerant. It must
# NOT match `prompt_tag=prompt_tag_for(...)` (no quote after `=`) nor the
# `prompt_tag: str | None = None` parameter default.
_HARDCODED_TAG_RE = re.compile(r"""prompt_tag\s*=\s*["'][^"']*@v\d""")

_SRC_ROOT = Path(memex.__file__).parent
_PROMPTS_DIR = _SRC_ROOT / "prompts"
_VERSION_FILE_RE = re.compile(r"^v(\d+)\.md$")


def _highest_on_disk(name: str) -> str:
    """Independently compute the highest `vN` template present for `name`."""
    nums = [
        int(m.group(1))
        for f in (_PROMPTS_DIR / name).iterdir()
        if (m := _VERSION_FILE_RE.match(f.name))
    ]
    assert nums, f"{name} has no versioned templates"
    return f"v{max(nums)}"


def test_prompt_tag_for_derives_name_and_version() -> None:
    load_prompt_spec.cache_clear()
    assert active_version("assess_sufficiency") == "v2"
    assert prompt_tag_for("assess_sufficiency") == "assess_sufficiency@v2"


def test_prompt_tag_for_is_pin_aware(monkeypatch: pytest.MonkeyPatch) -> None:
    """The derived tag reflects what ACTUALLY loaded — including a PIN. A
    hardcoded literal could not honor this: pinned to v1 while the string still
    said `@v2`."""
    monkeypatch.setenv("MEMEX_PROMPTS__PIN__ASSESS_SUFFICIENCY", "v1")
    load_prompt_spec.cache_clear()
    assert active_version("assess_sufficiency") == "v1"
    assert prompt_tag_for("assess_sufficiency") == "assess_sufficiency@v1"
    load_prompt_spec.cache_clear()  # don't leak the pin into other tests


def test_prompt_tag_for_unknown_name_raises() -> None:
    """Fail LOUD, not silently mislabel: an unknown prompt is a real bug, not a
    missing observability label."""
    load_prompt_spec.cache_clear()
    with pytest.raises(PromptNotFoundError):
        prompt_tag_for("no_such_prompt_xyz")


def test_derived_tag_matches_highest_on_disk_for_every_prompt() -> None:
    """Sweep the whole library, keyed by DIRECTORY name (what call sites pass):
    the derived version equals the highest `vN.md` actually present, and the tag
    is `<spec.name>@<highest>`. Recomputes the max independently of the loader."""
    load_prompt_spec.cache_clear()
    dirs = sorted(
        child.name
        for child in _PROMPTS_DIR.iterdir()
        if child.is_dir() and any(_VERSION_FILE_RE.match(f.name) for f in child.iterdir())
    )
    assert dirs, "no prompt directories discovered"
    for name in dirs:
        highest = _highest_on_disk(name)
        assert active_version(name) == highest
        assert prompt_tag_for(name) == f"{load_prompt_spec(name).name}@{highest}"


def test_no_hardcoded_version_tags_in_source() -> None:
    """THE PERMANENCE GUARD. No `prompt_tag="<name>@vN"` literal may exist in
    `src/memex` — every producer must derive via `prompt_tag_for`. Reintroducing
    a literal (the version-drift class) fails this test."""
    offenders: list[str] = []
    for py in _SRC_ROOT.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _HARDCODED_TAG_RE.search(line):
                rel = py.relative_to(_SRC_ROOT.parent)
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, (
        "Hardcoded prompt_tag version literals reintroduce the version-drift class — "
        "use prompt_tag_for(name) instead:\n" + "\n".join(offenders)
    )


def test_permanence_guard_regex_actually_fires() -> None:
    """Guard against the guard rotting: the regex must catch a reintroduced
    literal (built at runtime so it isn't itself a scanned source literal)."""
    reintroduced = "prompt_tag=" + '"answer@v3"'
    assert _HARDCODED_TAG_RE.search(reintroduced)
    assert not _HARDCODED_TAG_RE.search('prompt_tag=prompt_tag_for("answer")')
    assert not _HARDCODED_TAG_RE.search("prompt_tag: str | None = None")
