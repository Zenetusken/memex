"""Eval harness — the engine that scores Memex against a query set.

See docs/eval-corpus-plan.md for the full corpus design and the
per-category accuracy bars; this module ships the engine, the metric
implementations, and the report shape.
"""

from memex.eval.runner import (
    EvalQuery,
    EvalQueryResult,
    EvalReport,
    run_eval,
)
from memex.eval.scoring import (
    CitationPrecisionInput,
    character_error_rate,
    citation_precision,
    structural_f1_headings,
    word_error_rate,
)

__all__ = [
    "CitationPrecisionInput",
    "EvalQuery",
    "EvalQueryResult",
    "EvalReport",
    "character_error_rate",
    "citation_precision",
    "run_eval",
    "structural_f1_headings",
    "word_error_rate",
]
