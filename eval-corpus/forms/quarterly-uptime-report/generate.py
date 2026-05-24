#!/usr/bin/env python3
"""Render `source.pdf` from `source.fodt` for the quarterly-uptime-report
parse-eval fixture.

`source.fodt` (Flat ODF Text) is the human-authored canonical source — a
native LibreOffice Writer document, so its table renders with real cell
borders (unlike an HTML table through the writerweb filter, which
flattens). That's the point of this fixture: exercise GFM-table parse
fidelity. The ground truth (`ground-truth.md`) is independent of Memex
by construction (see CITATION.md).

    uv run python eval-corpus/forms/quarterly-uptime-report/generate.py

Requires `libreoffice`/`soffice`. Sets LD_LIBRARY_PATH to LibreOffice's
program dir for the subprocess (soffice.bin's `$ORIGIN` RUNPATH isn't
honored on this host). Regenerate after editing source.fodt, then re-run
`memex eval-parse` and refresh predicted.md + the manifest thresholds.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

_LO_PROGRAM_DIR = "/usr/lib/libreoffice/program"


def main() -> None:
    here = Path(__file__).parent
    src = here / "source.fodt"
    out = here / "source.pdf"
    env = os.environ.copy()
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{_LO_PROGRAM_DIR}:{existing}" if existing else _LO_PROGRAM_DIR
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(  # noqa: S603  # fixed argv, no shell
            ["soffice", "--headless", "--convert-to", "pdf", "--outdir", tmp, str(src)],  # noqa: S607
            check=True,
            env=env,
            capture_output=True,
        )
        shutil.copyfile(Path(tmp) / f"{src.stem}.pdf", out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
