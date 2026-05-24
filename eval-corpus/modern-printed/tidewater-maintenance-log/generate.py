#!/usr/bin/env python3
"""Render `source.pdf` for the tidewater-maintenance-log parse-eval doc.

Synthetic, copyright-clean content authored for the Memex eval corpus
(see CITATION.md). The HTML below mirrors `ground-truth.md`; the CSS
gives the headings distinct font sizes so a parser can recover the
H1/H2/H3 hierarchy from a born-digital PDF. Rendered with PyMuPDF's
`Story` API — no extra dependency.

    uv run python eval-corpus/modern-printed/tidewater-maintenance-log/generate.py

The ground truth is independent of Memex by construction: this script
defines the canonical document; Memex never touches it. Regenerate
`source.pdf` after editing the content here, then re-run
`memex eval-parse` and refresh `predicted.md` + the manifest thresholds.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

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


def main() -> None:
    out = Path(__file__).with_name("source.pdf")
    story = fitz.Story(html=HTML, user_css=CSS)
    writer = fitz.DocumentWriter(str(out))
    mediabox = fitz.paper_rect("letter")
    where = mediabox + (54, 54, -54, -54)  # 0.75" margins
    more = 1
    while more:
        dev = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
