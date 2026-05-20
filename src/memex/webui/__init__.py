"""Local web UI — FastAPI + HTMX, no SPA, no build step.

Serves on localhost only (default port 7423). The UI's job is the
visual parts of the workflow the CLI can't do well: ask + show
grounded answer with citations, document list, markdown render, PDF
side-by-side, Cytoscape graph view, and annotation correction.

See GUIDELINES.md Part V "Local web UI" and IMPLEMENTATION-PLAN §1.10.
"""

from memex.webui.app import create_app

__all__ = ["create_app"]
