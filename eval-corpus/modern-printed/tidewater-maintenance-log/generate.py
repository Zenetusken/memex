#!/usr/bin/env python3
"""Render `source.pdf` for the tidewater-maintenance-log parse-eval doc.

Synthetic, copyright-clean content authored for the Memex eval corpus
(see CITATION.md). The HTML below mirrors `ground-truth.md`; the CSS
gives the headings distinct font sizes. Rendered with **LibreOffice**
(a born-digital producer Memex's classifier recognizes), so the PDF's
internal structure resembles real office output far better than a
low-level PDF writer does — an earlier PyMuPDF `Story` render produced
a non-representative PDF where pymupdf4llm dropped the first paragraph
after some headings (a Story artifact, confirmed 2026-05-24: the
LibreOffice render does not drop them).

    uv run python eval-corpus/modern-printed/tidewater-maintenance-log/generate.py

Requires `libreoffice`/`soffice` installed. This script sets
`LD_LIBRARY_PATH` to LibreOffice's program dir for the soffice
subprocess — on this host the `$ORIGIN` RUNPATH in `soffice.bin` isn't
honored, so the libs (e.g. `libreglo.so`) don't resolve without it.

The ground truth is independent of Memex by construction: this script
defines the canonical document; Memex never touches it. Regenerate
`source.pdf` after editing the content, then re-run `memex eval-parse`
and refresh `predicted.md` + the manifest thresholds.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

HTML = """
<h1>Tidewater Lighthouse Maintenance Log</h1>

<h2>Overview</h2>
<p>This log records the quarterly inspection of the Tidewater lighthouse and
its supporting equipment. The structure has operated continuously since its
last refit and remains within expected tolerances. Routine maintenance keeps
the optic, the rotation drive, and the backup power chain in service.</p>

<h2>Inspection Schedule</h2>
<p>Inspections run on a fixed quarterly cadence. Each visit covers the optic
assembly, the structural envelope, and the electrical systems in that order.</p>
<ul>
<li>Clean the optic and verify the lamp changer rotates freely.</li>
<li>Check the gallery railing and the access ladder for corrosion.</li>
<li>Test the backup generator under load for thirty minutes.</li>
</ul>

<h3>Equipment Checklist</h3>
<p>The following items must be confirmed present and serviceable before the
inspector signs off:</p>
<ul>
<li>Spare lamp set, sealed and dated.</li>
<li>Hand winch with an intact safety pawl.</li>
<li>First aid kit within its expiry window.</li>
</ul>

<h2>Observed Conditions</h2>
<p>Visibility readings were taken at dawn from the gallery deck.</p>
<table>
<tr><th>Date</th><th>Visibility</th><th>Notes</th></tr>
<tr><td>2026-04-10</td><td>Good</td><td>Light haze offshore, cleared by midday.</td></tr>
<tr><td>2026-04-11</td><td>Poor</td><td>Dense fog; foghorn engaged for six hours.</td></tr>
<tr><td>2026-04-12</td><td>Good</td><td>Clear horizon, calm sea state.</td></tr>
</table>

<h2>Recommendations</h2>
<p>The inspector recommends the following actions before the next quarter:</p>
<ol>
<li>Replace the gallery railing fasteners showing surface rust.</li>
<li>Schedule a load test of the secondary battery bank.</li>
<li>Restock the spare lamp set to the full complement of four.</li>
</ol>
"""

CSS = """
body { font-family: sans-serif; font-size: 11pt; line-height: 1.4; }
p { margin: 0 0 10pt 0; }
h1 { font-size: 22pt; font-weight: bold; margin: 0 0 12pt 0; }
h2 { font-size: 16pt; font-weight: bold; margin: 16pt 0 10pt 0; }
h3 { font-size: 13pt; font-weight: bold; margin: 12pt 0 8pt 0; }
ul, ol { margin: 0 0 10pt 0; }
li { margin: 0 0 4pt 0; }
table { border-collapse: collapse; margin: 4pt 0 10pt 0; }
th, td { border: 1px solid #444; padding: 4px 8px; text-align: left; }
th { font-weight: bold; }
"""

_LO_PROGRAM_DIR = "/usr/lib/libreoffice/program"


def _soffice_env() -> dict[str, str]:
    env = os.environ.copy()
    # soffice.bin's RUNPATH is `$ORIGIN`, which isn't honored on this host;
    # point the loader at the program dir so libreglo.so etc. resolve.
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{_LO_PROGRAM_DIR}:{existing}" if existing else _LO_PROGRAM_DIR
    )
    return env


def _convert(src: Path, to: str, outdir: Path) -> Path:
    subprocess.run(  # noqa: S603  # fixed argv, no shell; soffice is the documented tool
        ["soffice", "--headless", "--convert-to", to, "--outdir", str(outdir), str(src)],  # noqa: S607
        check=True,
        env=_soffice_env(),
        capture_output=True,
    )
    return outdir / f"{src.stem}.{to}"


def main() -> None:
    out = Path(__file__).with_name("source.pdf")
    full_html = (
        f"<html><head><meta charset='utf-8'><style>{CSS}</style></head>"
        f"<body>{HTML}</body></html>"
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        html_path = tmpdir / "doc.html"
        html_path.write_text(full_html, encoding="utf-8")
        # HTML → ODT → PDF: the two-step uses Writer's PDF export, which
        # lays out the document closer to real word-processor output than
        # the one-step writer_web filter (it recovers the H1 title).
        odt = _convert(html_path, "odt", tmpdir)
        pdf = _convert(odt, "pdf", tmpdir)
        shutil.copyfile(pdf, out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
