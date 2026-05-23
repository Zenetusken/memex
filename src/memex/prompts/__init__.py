"""Versioned prompt library — see GUIDELINES.md Part III "Prompt management".

Prompts are code: versioned, tested, evaluated, reviewed. They live as
`.md` files with YAML frontmatter declaring name, version, role, target
model, input/output schema, and eval suite.
"""

from memex.prompts.loader import render_messages, render_prompt

__all__ = ["render_messages", "render_prompt"]
