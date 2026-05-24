"""Parse-quality eval wiring — CER/WER/structural-F1 against ground truth.

Covers the heading extractor (fence- + chart-block-aware), the combined
`score_parse_quality`, and `run_parse_eval` over the documented corpus
layout (`<doc>/ground-truth.md` + `manifest.json`, predicted markdown
from a `predicted.md` override or the vault). The runner is async but
needs no GPU — predicted markdown is read from disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from memex.eval.runner import run_parse_eval
from memex.eval.scoring import (
    extract_markdown_headings,
    score_parse_quality,
    strip_frontmatter,
)


def test_strip_frontmatter() -> None:
    md = "---\ntitle: X\ndate: 2026-01-01\n---\n# Heading\n\nBody.\n"
    assert strip_frontmatter(md) == "# Heading\n\nBody.\n"
    assert strip_frontmatter("# No frontmatter\n") == "# No frontmatter\n"


def test_extract_headings_basic_and_atx_closing() -> None:
    md = "# Title\n\n## Section A ##\n\ntext\n\n### Sub\n"
    assert extract_markdown_headings(md) == [
        (1, "Title"),
        (2, "Section A"),
        (3, "Sub"),
    ]


def test_extract_headings_skips_code_fences() -> None:
    md = "# Real\n\n```\n# not a heading (in fence)\n```\n\n## Also real\n"
    assert extract_markdown_headings(md) == [(1, "Real"), (2, "Also real")]


def test_extract_headings_strips_inline_markdown() -> None:
    # pymupdf4llm wraps headings in **bold**; structural F1 compares the
    # heading text, so the markers must be stripped (the `## **Overview**`
    # case from the tidewater synthetic doc).
    md = "## **Overview**\n\n### `code` ref\n\n## [Tips](http://x)\n"
    assert extract_markdown_headings(md) == [
        (2, "Overview"),
        (3, "code ref"),
        (2, "Tips"),
    ]


def test_extract_headings_skips_chart_extracted_blocks() -> None:
    md = (
        "# Real Heading\n\n"
        "[chart-extracted]\n# Minimal lead-time\n# Inter-operability\n[/chart-extracted]\n\n"
        "## After Chart\n"
    )
    # The inert chart-figure H1 labels must not count as document headings.
    assert extract_markdown_headings(md) == [(1, "Real Heading"), (2, "After Chart")]


def test_score_parse_quality_identical_is_perfect() -> None:
    md = "---\ntitle: T\n---\n# H1\n\nThe quick brown fox.\n"
    s = score_parse_quality(md, md)
    assert s.cer == 0.0
    assert s.wer == 0.0
    assert s.structural_f1_headings == 1.0


def test_score_parse_quality_perturbed() -> None:
    ref = "# Heading\n\nThe quick brown fox jumps.\n"
    pred = "# Heading\n\nThe quick brown cat jumps.\n"  # one word substituted
    s = score_parse_quality(pred, ref)
    assert 0.0 < s.cer < 1.0
    assert 0.0 < s.wer < 1.0
    assert s.structural_f1_headings == 1.0  # headings unchanged


def _write_doc(
    root: Path, name: str, ground_truth: str, *, predicted: str | None, scoring: dict | None
) -> None:
    d = root / "modern-printed" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "ground-truth.md").write_text(ground_truth, encoding="utf-8")
    if predicted is not None:
        (d / "predicted.md").write_text(predicted, encoding="utf-8")
    manifest: dict[str, object] = {"doc_id": name}
    if scoring is not None:
        manifest["scoring"] = scoring
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


async def test_run_parse_eval_scores_and_thresholds(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    gt = "---\ntitle: A\n---\n# Alpha\n\nThe quick brown fox jumps over the lazy dog.\n"
    # doc-a: predicted == ground truth → perfect, passes a tight threshold.
    _write_doc(
        corpus,
        "doc-a",
        gt,
        predicted=gt,
        scoring={"cer_threshold": 0.02, "wer_threshold": 0.05, "structural_f1_threshold": 0.95},
    )
    # doc-b: predicted drops two words + a heading → fails the tight threshold.
    _write_doc(
        corpus,
        "doc-b",
        gt,
        predicted="---\ntitle: A\n---\n## Wrong Level\n\nThe quick fox jumps the lazy dog.\n",
        scoring={"cer_threshold": 0.02, "wer_threshold": 0.05, "structural_f1_threshold": 0.95},
    )
    # doc-c: no predicted.md and an empty vault → recorded as an error, not fatal.
    _write_doc(corpus, "doc-c", gt, predicted=None, scoring=None)

    report = await run_parse_eval(corpus, vault_path=tmp_path / "empty-vault")

    assert report.doc_count == 2  # a + b scored; c errored
    by_id = {d.doc_id: d for d in report.per_doc}
    assert by_id["doc-a"].cer == 0.0
    assert by_id["doc-a"].passed is True
    assert by_id["doc-b"].cer > 0.0
    assert by_id["doc-b"].passed is False  # wrong heading level + dropped words
    assert by_id["doc-b"].structural_f1_threshold == 0.95  # threshold carried through
    assert report.passed_count == 1
    assert any("doc-c" in e for e in report.errors)


async def test_run_parse_eval_reads_predicted_from_vault(tmp_path: Path) -> None:
    """With no predicted.md, the runner reads the vault's parsed markdown."""
    corpus = tmp_path / "corpus"
    vault = tmp_path / "vault"
    (vault / "documents").mkdir(parents=True)
    gt = "# Heading\n\nGround truth body.\n"
    _write_doc(corpus, "vault-doc", gt, predicted=None, scoring=None)
    # The "predicted" output lives in the vault as documents/<doc_id>.md.
    (vault / "documents" / "vault-doc.md").write_text(
        "---\ntitle: V\n---\n# Heading\n\nGround truth body.\n", encoding="utf-8"
    )

    report = await run_parse_eval(corpus, vault_path=vault)
    assert report.doc_count == 1
    assert report.per_doc[0].doc_id == "vault-doc"
    assert report.per_doc[0].cer == 0.0  # vault body matches ground truth (frontmatter stripped)
    assert report.errors == []
