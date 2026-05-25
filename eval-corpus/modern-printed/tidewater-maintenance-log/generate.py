#!/usr/bin/env python3
"""Render `source.pdf` from `source.fodt` for the tidewater-maintenance-log
parse-eval fixture.

`source.fodt` (Flat ODF Text) is the human-authored canonical source — a
native LibreOffice Writer document. **Migrated 2026-05-25** from an earlier
HTML→ODT→PDF render: soffice's HTML import dropped the table's cell borders,
so the rendered PDF had no detectable table grid (pymupdf4llm emitted the
header as a heading and the rows as one prose line → structural_f1_tables =
0.0). A native `.fodt` table with explicit cell borders (like the
forms/quarterly-uptime-report fixture) renders a real grid, so it parses as a
proper GFM table. The ground truth (`ground-truth.md`) is independent of Memex
by construction (see CITATION.md) and is unchanged by the migration.

    uv run python eval-corpus/modern-printed/tidewater-maintenance-log/generate.py

Requires `libreoffice`/`soffice`. Sets LD_LIBRARY_PATH to LibreOffice's
program dir for the subprocess (soffice.bin's `$ORIGIN` RUNPATH isn't honored
on this host). Regenerate after editing source.fodt, then re-run
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
