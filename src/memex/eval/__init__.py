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
    run_eval,
    run_parse_eval,
)
from memex.eval.scoring import (
    CitationPrecisionInput,
    ParseQualityScores,
    character_error_rate,
    citation_precision,
    extract_markdown_headings,
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
    "character_error_rate",
    "citation_precision",
    "extract_markdown_headings",
    "run_eval",
    "run_parse_eval",
    "score_parse_quality",
    "strip_frontmatter",
    "structural_f1_headings",
    "word_error_rate",
]
