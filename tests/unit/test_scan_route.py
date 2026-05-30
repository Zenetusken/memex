"""Unit tests for the scan→VLM route's page assembly (spec: scan-vlm-parse.md).

`_assemble_scan_pages` is pure: VLM per-page results → ordered `PageDecision`s
(engine="scan") + the markdown parts to stitch. A failed page is recorded (conf 0,
error in rationale) but contributes no markdown — no content silently dropped.
"""

from __future__ import annotations

from memex.parse.docling_backend import DoclingPageOutput
from memex.parse.pipeline import _assemble_scan_pages, _strip_markdown_fence_wrapper


def _out(page: int, md: str) -> DoclingPageOutput:
    return DoclingPageOutput(page=page, markdown=md, confidence=1.0)


def test_assemble_all_pages_transcribed() -> None:
    results = {1: _out(1, "# A\n\nalpha"), 2: _out(2, "# B\n\nbeta")}
    pages, parts = _assemble_scan_pages(results, 2)
    assert [p.engine for p in pages] == ["scan", "scan"]
    assert all(p.confidence == 1.0 for p in pages)
    assert parts == ["# A\n\nalpha", "# B\n\nbeta"]  # reading order preserved


def test_assemble_failed_page_recorded_but_skipped_from_markdown() -> None:
    results: dict[int, DoclingPageOutput | Exception] = {
        1: _out(1, "alpha"),
        2: ValueError("vlm boom"),
    }
    pages, parts = _assemble_scan_pages(results, 2)
    assert pages[0].confidence == 1.0
    assert pages[1].engine == "scan" and pages[1].confidence == 0.0
    assert "vlm boom" in pages[1].rationale  # the failure is recorded, not silent
    assert parts == ["alpha"]  # only the readable page contributes content


def test_assemble_missing_page_is_a_failure() -> None:
    # A page absent from the results dict (e.g. the VLM dropped it) → conf 0, no markdown.
    pages, parts = _assemble_scan_pages({1: _out(1, "alpha")}, 2)
    assert pages[1].confidence == 0.0
    assert parts == ["alpha"]


def test_assemble_blank_page_transcribed_but_no_part() -> None:
    # A genuinely blank page transcribes to whitespace → counted (conf 1) but no content.
    pages, parts = _assemble_scan_pages({1: _out(1, "   \n  ")}, 1)
    assert pages[0].confidence == 1.0
    assert parts == []


def test_strip_fence_unwraps_markdown_labelled_wrapper() -> None:
    md = "```markdown\n# Title\n\nbody text\n```"
    assert _strip_markdown_fence_wrapper(md) == "# Title\n\nbody text"


def test_strip_fence_preserves_nested_code_block() -> None:
    # The VLM wraps the whole page AND the page itself contains a fenced ASCII diagram.
    md = "```markdown\n# Notes\n\n```\n[A] -> [B]\n```\n\ndone\n```"
    out = _strip_markdown_fence_wrapper(md)
    assert out.startswith("# Notes")
    assert "```\n[A] -> [B]\n```" in out  # the inner diagram fence survives, balanced
    assert "done" in out


def test_strip_fence_leaves_bare_fence_alone() -> None:
    # A bare ``` wrapper is ambiguous (could be a real code block) → not unwrapped.
    md = "```\nint main() {}\n```"
    assert _strip_markdown_fence_wrapper(md) == md


def test_strip_fence_noop_on_plain_markdown() -> None:
    md = "# Heading\n\nsome prose with no wrapper"
    assert _strip_markdown_fence_wrapper(md) == md


def test_strip_fence_md_variant_and_applied_in_assembly() -> None:
    assert _strip_markdown_fence_wrapper("```md\nhello\n```") == "hello"
    # And the assembly applies it: a fence-wrapped page → unwrapped part.
    _, parts = _assemble_scan_pages({1: _out(1, "```markdown\n# P\n\nalpha\n```")}, 1)
    assert parts == ["# P\n\nalpha"]


def test_strip_fence_noop_when_close_missing() -> None:
    # The conservative branch the W5 vault sweep found in 3 real docs (e.g. cr350-semaine-5):
    # a ```markdown opener whose closing fence was lost (EOF). With no matching close we CANNOT
    # know the boundary → return the input unchanged rather than risk swallowing real content.
    md = "```markdown\n# Intro\n\nbody with no closing fence"
    assert _strip_markdown_fence_wrapper(md) == md


def test_strip_fence_unwraps_title_slide_with_decorative_narration() -> None:
    # The canonical vault W5 case (21bc279d-ensa-module-10@7): a ```markdown wrapper around a
    # real title heading AND a W6 decorative-narration line. Unwrapping recovers the heading as
    # a heading; the decorative line is left for the (separate) W6 prompt fix to suppress.
    md = (
        "```markdown\n# Module 10: Network Management\n\n"
        '*Image Description: A man in a suit ... reads "OBEY" ... solid orange color.*\n```'
    )
    out = _strip_markdown_fence_wrapper(md)
    assert out.startswith("# Module 10: Network Management")
    assert not out.startswith("```") and not out.endswith("```")
