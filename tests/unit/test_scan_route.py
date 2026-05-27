"""Unit tests for the scan→VLM route's page assembly (spec: scan-vlm-parse.md).

`_assemble_scan_pages` is pure: VLM per-page results → ordered `PageDecision`s
(engine="scan") + the markdown parts to stitch. A failed page is recorded (conf 0,
error in rationale) but contributes no markdown — no content silently dropped.
"""

from __future__ import annotations

from memex.parse.docling_backend import DoclingPageOutput
from memex.parse.pipeline import _assemble_scan_pages


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
