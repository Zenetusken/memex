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
    extract_markdown_equations,
    extract_markdown_headings,
    extract_markdown_tables,
    normalize_equation,
    score_parse_quality,
    strip_frontmatter,
    structural_f1_equations,
    structural_f1_tables,
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


# --- Tables ----------------------------------------------------------------

_GFM_TABLE = "| Col A | Col B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n"


def test_extract_tables_basic() -> None:
    tables = extract_markdown_tables(f"# H\n\n{_GFM_TABLE}\ntext after\n")
    assert tables == [[["Col A", "Col B"], ["1", "2"], ["3", "4"]]]  # delimiter row dropped


def test_extract_tables_skips_fences_and_chart_blocks() -> None:
    md = (
        f"{_GFM_TABLE}\n"
        "```\n| fenced | not a table |\n|---|---|\n| x | y |\n```\n"
        "[chart-extracted]\n| chart | row |\n|---|---|\n| a | b |\n[/chart-extracted]\n"
    )
    # Only the real GFM table counts — the fenced + chart-block ones are skipped.
    assert extract_markdown_tables(md) == [[["Col A", "Col B"], ["1", "2"], ["3", "4"]]]


def test_structural_f1_tables_perfect_and_empty() -> None:
    t = extract_markdown_tables(_GFM_TABLE)
    assert structural_f1_tables(t, t) == 1.0
    assert structural_f1_tables([], []) == 1.0  # no tables either side → not penalized
    assert structural_f1_tables(t, []) == 0.0  # predicted a table that isn't in the truth


def test_structural_f1_tables_cell_mismatch_and_shape() -> None:
    ref = [[["Col A", "Col B"], ["1", "2"], ["3", "4"]]]
    one_cell_wrong = [[["Col A", "Col B"], ["1", "9"], ["3", "4"]]]  # 5/6 cells match
    f1 = structural_f1_tables(one_cell_wrong, ref)
    assert 0.0 < f1 < 1.0
    # A dropped column lowers recall (ref has cells the prediction lacks).
    missing_col = [[["Col A"], ["1"], ["3"]]]
    assert structural_f1_tables(missing_col, ref) < 1.0


# --- Equations --------------------------------------------------------------


def test_normalize_equation() -> None:
    assert normalize_equation(r"\dfrac{a}{b}") == r"\frac{a}{b}"
    assert normalize_equation(r"\tfrac{a}{b}") == r"\frac{a}{b}"
    assert normalize_equation(r"a  +   b") == "a + b"  # whitespace collapsed
    assert normalize_equation(r"\left( x \right)") == "( x )"  # \left/\right dropped


def test_extract_equations_display_inline_and_fence() -> None:
    md = (
        "Inline $E = mc^2$ here.\n\n"
        "$$\\int_0^1 x\\,dx$$\n\n"
        "Bracket \\[a^2 + b^2\\] and paren \\(y = x\\).\n\n"
        "```\necho $PATH  # shell var, not math\n```\n"
    )
    eqs = {normalize_equation(e) for e in extract_markdown_equations(md)}
    assert "E = mc^2" in eqs
    assert r"\int_0^1 x dx" in eqs  # \, spacing macro dropped
    assert "a^2 + b^2" in eqs
    assert "y = x" in eqs
    assert not any("PATH" in e for e in eqs)  # fenced shell $VAR excluded


def test_structural_f1_equations() -> None:
    ref = [r"E = mc^2", r"\dfrac{a}{b}"]
    assert structural_f1_equations(ref, ref) == 1.0
    # \dfrac vs \frac normalize to the same equation → still perfect.
    assert structural_f1_equations([r"E = mc^2", r"\frac{a}{b}"], ref) == 1.0
    assert structural_f1_equations([], []) == 1.0  # no math either side
    assert structural_f1_equations([r"E = mc^2"], ref) < 1.0  # missing one (recall)


def test_score_parse_quality_includes_table_and_equation_facets() -> None:
    doc = f"# H\n\nText with $E=mc^2$.\n\n{_GFM_TABLE}"
    s = score_parse_quality(doc, doc)
    assert s.structural_f1_tables == 1.0
    assert s.structural_f1_equations == 1.0
    # A doc with no tables/equations is not penalized on those facets.
    plain = score_parse_quality("# H\n\nJust prose.\n", "# H\n\nJust prose.\n")
    assert plain.structural_f1_tables == 1.0
    assert plain.structural_f1_equations == 1.0


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
