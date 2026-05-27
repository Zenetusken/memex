"""Eval harness — the engine that scores Memex against a query set.

See docs/eval-corpus-plan.md for the full corpus design and the
per-category accuracy bars; this module ships the engine, the metric
implementations, and the report shape.
"""

from memex.eval.runner import (
    EvalQuery,
    EvalQueryResult,
    EvalReport,
    ParseEvalDoc,
    ParseEvalReport,
    SummaryEvalCase,
    SummaryEvalReport,
    SummaryEvalResult,
    run_eval,
    run_parse_eval,
    run_summary_eval,
)
from memex.eval.scoring import (
    CitationPrecisionInput,
    ParseQualityScores,
    absent_assertion_violations,
    character_error_rate,
    citation_precision,
    extract_markdown_headings,
    mention_recall,
    score_parse_quality,
    strip_frontmatter,
    structural_f1_headings,
    word_error_rate,
)

__all__ = [
    "CitationPrecisionInput",
    "EvalQuery",
    "EvalQueryResult",
    "EvalReport",
    "ParseEvalDoc",
    "ParseEvalReport",
    "ParseQualityScores",
    "SummaryEvalCase",
    "SummaryEvalReport",
    "SummaryEvalResult",
    "absent_assertion_violations",
    "character_error_rate",
    "citation_precision",
    "extract_markdown_headings",
    "mention_recall",
    "run_eval",
    "run_parse_eval",
    "run_summary_eval",
    "score_parse_quality",
    "strip_frontmatter",
    "structural_f1_headings",
    "word_error_rate",
]
