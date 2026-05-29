"""Guard: every CSS class used in the webui templates / graph.js must have a rule in the
vendored CSS subset (else it silently no-ops). Runs `scripts/check-tailwind-coverage.py`.
This is the net that would have caught the 2026-05-29 audit's 21 missing utilities."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_every_used_class_has_a_rule() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "check-tailwind-coverage.py"
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
