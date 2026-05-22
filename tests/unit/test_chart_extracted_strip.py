"""P3.3 v3 — verify `strip_chart_extracted_for_index` removes the
`[chart-extracted]...[/chart-extracted]` blocks emitted by the
chart-OCR stitch step, preserving everything else.

This is the indexing-side defence against retrieval perturbation.
The chart-OCR output (DePlot's linearised table data) gets injected
into chunk text, and the BM25 + dense embedding signal those dense
numerical tokens too strongly — boosting chart-bearing chunks for
unrelated queries that happen to mention years or percentages.

These tests pin the strip helper's contract.
"""

from __future__ import annotations

from memex.index.chunker import strip_chart_extracted_for_index


def test_strip_removes_single_chart_extracted_block() -> None:
    text = """## NVIDIA Datacenter Inventory

A bar chart showing yearly capacity in megawatts.

<!-- image -->

[chart-extracted]
TITLE |
| North America Primary Market
1 | 1330
2 | 1519
3 | 1822
[/chart-extracted]

Source: CBRE Data Center Trends.
"""
    stripped = strip_chart_extracted_for_index(text)

    # Block content gone.
    assert "1330" not in stripped
    assert "TITLE" not in stripped
    assert "chart-extracted" not in stripped
    # Prose preserved.
    assert "NVIDIA Datacenter Inventory" in stripped
    assert "Source: CBRE Data Center Trends." in stripped
    # The `<!-- image -->` placeholder is intentionally preserved —
    # it's a Docling artifact, not chart-OCR output, and removing it
    # would lose evidence that a figure exists at that point in the
    # markdown.
    assert "<!-- image -->" in stripped


def test_strip_removes_multiple_blocks_in_same_text() -> None:
    """A chunk may span multiple figures (e.g., a multi-figure slide).
    Every block should be stripped."""
    text = """Slide 1.

<!-- image -->

[chart-extracted]
A | 1
B | 2
[/chart-extracted]

Slide 2.

<!-- image -->

[chart-extracted]
X | 100
Y | 200
[/chart-extracted]

End of slides.
"""
    stripped = strip_chart_extracted_for_index(text)

    assert "1" not in stripped or "Slide 1" in stripped  # only digit from slide labels survives
    assert "A | 1" not in stripped
    assert "X | 100" not in stripped
    assert "Slide 1." in stripped
    assert "Slide 2." in stripped
    assert "End of slides." in stripped
    assert stripped.count("chart-extracted") == 0


def test_strip_passes_through_when_no_block_present() -> None:
    """Plain markdown without any chart-extracted blocks must be
    returned unchanged. The function is idempotent for the common
    case (chart-OCR disabled OR a chunk that has no figures).
    """
    text = """## Why CUDA Matters

CUDA's runtime API lets developers target NVIDIA GPUs with a C-like
language. The 2024 release added FP8 support and improved cooperative
groups.

```cuda
__global__ void kernel() { ... }
```

End.
"""
    assert strip_chart_extracted_for_index(text) == text


def test_strip_is_idempotent() -> None:
    """Stripping twice produces the same result as stripping once.
    Defensive — the helper might be called multiple times in a
    re-indexing flow.
    """
    text = """Body.

[chart-extracted]
Data
[/chart-extracted]

More body."""
    once = strip_chart_extracted_for_index(text)
    twice = strip_chart_extracted_for_index(once)
    assert once == twice


def test_strip_handles_multiline_chart_block() -> None:
    """The block may span many lines (DePlot's table is multi-row).
    The dot-all regex must consume newlines inside the block."""
    inner = "\n".join(f"row{i} | value{i}" for i in range(50))
    text = f"Prose before.\n\n[chart-extracted]\n{inner}\n[/chart-extracted]\n\nProse after."
    stripped = strip_chart_extracted_for_index(text)

    assert "Prose before." in stripped
    assert "Prose after." in stripped
    assert "row5" not in stripped
    assert "value49" not in stripped


def test_strip_leaves_other_bracketed_content_alone() -> None:
    """Markdown link syntax `[text](url)` and other square-bracket
    constructs must NOT be touched — the regex is anchored to the
    specific `[chart-extracted]` / `[/chart-extracted]` markers."""
    text = """Markdown link [click here](https://example.com).

A list:
- [x] done
- [ ] todo

[citation-needed]

[chart-extracted]
gone | yes
[/chart-extracted]

End.
"""
    stripped = strip_chart_extracted_for_index(text)
    assert "[click here]" in stripped
    assert "[x]" in stripped
    assert "[ ]" in stripped
    assert "[citation-needed]" in stripped
    assert "gone | yes" not in stripped
    assert "End." in stripped


def test_strip_handles_empty_block() -> None:
    """A chart-OCR refusal (UNREADABLE → empty extraction) might
    leave an empty `[chart-extracted][/chart-extracted]` pair.
    Still strips cleanly."""
    text = "Before.\n\n[chart-extracted][/chart-extracted]\n\nAfter."
    stripped = strip_chart_extracted_for_index(text)
    assert "Before." in stripped
    assert "After." in stripped
    assert "chart-extracted" not in stripped
