#!/usr/bin/env python3
"""Docling heading-height histogram — the Stage-0 calibration diagnostic for the
mode-anchored re-tiering redesign (deferred-backlog "Docling mode-anchored re-tiering").

For each doc it runs the REAL Docling convert (mirroring `docling_worker`'s converter
options) and prints the `bucketed_header_heights` distribution — the SAME bucketing
production's `_recover_heading_levels` sees — plus the dominance-gate analysis. The point:
discriminate the 10-K (a populous small-height section tier UNDER a thin scatter of rarer
taller singletons → mode-anchor should fire) from a slide deck (one uniform bucket → flat)
and a real hierarchical doc (a pyramid: populous tiers ABOVE the mode → must NOT fire).

Usage:
    uv run python scripts/docling_heading_histogram.py <doc_id|source.pdf> [...]
    uv run python scripts/docling_heading_histogram.py --json <doc_id> > hist.json

NB this calls Docling DIRECTLY (no seccomp sandbox) — a read-only diagnostic; models must
already be cached (production uses them). Resolves a doc_id to its vault source PDF.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Starting-guess gate constants (Stage 0 calibrates these from the real histograms).
DOMINANCE_MIN_COUNT = 8
DOMINANCE_FRACTION = 0.25
DOMINANCE_MARGIN = 2.0
TALLER_RARE_MAX = 2


def _resolve_source(arg: str) -> Path:
    """A path → itself; else treat as a vault doc_id and find its source PDF."""
    p = Path(arg)
    if p.exists() and p.is_file():
        return p
    from memex.core.config import get_settings

    docdir = get_settings().vault_path / "documents" / arg
    for name in ("source.pdf", "converted.pdf"):
        cand = docdir / name
        if cand.exists():
            return cand
    matches = sorted(docdir.glob("source.*"))
    if matches:
        return matches[0]
    raise SystemExit(f"no source PDF for {arg!r} (looked in {docdir})")


def _convert(source: Path) -> Any:
    """Mirror docling_worker's converter (do_ocr off, default pipeline) → DoclingDocument."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    return converter.convert(str(source)).document


def _gate_analysis(headers: list[tuple[Any, float]]) -> dict[str, Any]:
    """Compute the dominance-gate inputs + per-clause verdict for the starting constants."""
    n = len(headers)
    freq: Counter[float] = Counter(b for _, b in headers)
    mode_bucket, mode_count = freq.most_common(1)[0]
    others = [c for b, c in freq.items() if b != mode_bucket]
    second_freq = max(others) if others else 0
    taller = {b: c for b, c in freq.items() if b > mode_bucket}
    taller_max_freq = max(taller.values()) if taller else 0
    clauses = {
        "min_count": mode_count >= DOMINANCE_MIN_COUNT,
        "fraction": n > 0 and mode_count >= DOMINANCE_FRACTION * n,
        "margin": mode_count >= DOMINANCE_MARGIN * second_freq,
        "pyramid_guard (taller buckets are rare)": taller_max_freq <= TALLER_RARE_MAX,
    }
    return {
        "n_headers": n,
        "distinct_buckets": len(freq),
        "mode_bucket": mode_bucket,
        "mode_count": mode_count,
        "second_freq": second_freq,
        "taller_than_mode": dict(sorted(taller.items(), reverse=True)),
        "taller_max_freq": taller_max_freq,
        "clauses": clauses,
        "gate_fires": all(clauses.values()),
    }


def _heading_text(item: Any) -> str:
    t = getattr(item, "text", "") or ""
    return " ".join(t.split())[:70]


def report(arg: str, *, post_demote: bool = False) -> dict[str, Any]:
    from memex.parse.docling_worker import bucketed_header_heights

    source = _resolve_source(arg)
    doc = _convert(source)
    demoted = 0
    if post_demote:
        # Mirror production: `_recover_heading_levels` runs BEFORE `_demote_misdetected_headers`,
        # but the demoted prose headers drop OUT of the final .md headings — so the production-
        # faithful tier distribution is the POST-demotion one. (Pre-demotion over-counts prose.)
        from docling_core.types.doc.document import TextItem

        from memex.parse.docling_worker import _demote_misdetected_headers

        demoted = _demote_misdetected_headers(doc, text_item_cls=TextItem)
    headers = bucketed_header_heights(doc)
    if post_demote:
        # The demotion swaps __class__→TextItem but LEAVES label=="section_header", and
        # bucketed_header_heights filters by LABEL — so exclude the class-swapped items here to
        # get the distribution of headings that actually survive into the .md (production-faithful).
        headers = [(it, b) for (it, b) in headers if type(it).__name__ != "TextItem"]
    freq: Counter[float] = Counter(b for _, b in headers)
    # per-bucket sample texts (the eyeball: is the buried tier real section titles?)
    samples: dict[float, list[str]] = {}
    for item, b in headers:
        samples.setdefault(b, [])
        if len(samples[b]) < 4:
            samples[b].append(_heading_text(item))
    g = _gate_analysis(headers) if headers else {"n_headers": 0, "gate_fires": False}

    print(f"\n{'=' * 78}\n{arg}  ({source.name})  post_demote={post_demote} demoted={demoted}\n{'=' * 78}")
    if not headers:
        print("  NO section-header items with provenance heights (no-op).")
        return {"doc": arg, **g}
    print(f"  headers={g['n_headers']}  distinct_buckets={g['distinct_buckets']}  "
          f"mode={g['mode_bucket']}pt×{g['mode_count']}  2nd_freq={g['second_freq']}")
    print(f"  taller-than-mode buckets: {g['taller_than_mode']}  (max_freq={g['taller_max_freq']})")
    print("  histogram (height pt → count): tallest first, * = mode")
    for b in sorted(freq, reverse=True):
        mark = " *" if b == g["mode_bucket"] else "  "
        bar = "#" * min(freq[b], 60)
        print(f"   {mark} {b:6.1f}  {freq[b]:4d}  {bar}")
        for s in samples[b]:
            print(f"          · {s!r}")
    print(f"  CLAUSES {g['clauses']}")
    print(f"  >>> mode-anchor gate FIRES: {g['gate_fires']}  "
          f"(constants: min={DOMINANCE_MIN_COUNT} frac={DOMINANCE_FRACTION} "
          f"margin={DOMINANCE_MARGIN} taller_rare_max={TALLER_RARE_MAX})")
    return {"doc": arg, "histogram": {str(k): v for k, v in sorted(freq.items(), reverse=True)},
            "samples": {str(k): v for k, v in samples.items()}, **g}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("docs", nargs="+", help="vault doc_id(s) or source PDF path(s)")
    ap.add_argument("--json", action="store_true", help="emit JSON (the per-doc analyses)")
    ap.add_argument("--post-demote", action="store_true",
                    help="apply _demote_misdetected_headers before bucketing (production-faithful tiers)")
    args = ap.parse_args()
    # Initialise settings so `get_settings()` (doc_id → vault source resolution) works.
    from memex.core.config import MemexSettings, set_settings

    set_settings(MemexSettings())  # type: ignore[call-arg]
    out = [report(d, post_demote=args.post_demote) for d in args.docs]
    if args.json:
        print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
