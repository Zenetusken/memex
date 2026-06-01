"""LangGraph state machines — see GUIDELINES.md Part III "Agent design".

Agents are explicit graphs with budgets and a mandatory `refuse` outcome,
not free-form ReAct loops. Every node returns a typed pydantic update.

`Chunk` is not re-exported from here even though it appears in node
signatures — it's a shared type and lives in `memex.core.types`. Import
it from there, not via `memex.agents`.
"""

from memex.agents.answering import (
    AnswerState,
    CitedClaim,
    DraftAnswer,
    FinalResponse,
    SufficiencyAssessment,
    VerificationResult,
    answer_query,
    build_answering_graph,
)
from memex.agents.chat import ChatTurnResult, answer_turn
from memex.agents.table_sql import coerce_number, query_doc_tables

__all__ = [
    "AnswerState",
    "ChatTurnResult",
    "CitedClaim",
    "DraftAnswer",
    "FinalResponse",
    "SufficiencyAssessment",
    "VerificationResult",
    "answer_query",
    "answer_turn",
    "build_answering_graph",
    "coerce_number",
    "query_doc_tables",
]
