# Spec: scan→VLM parse route

Status: **shipped 2026-05-27**. Related: ADR-0006 §4 (VLM via parse-time vLLM), specs
`vlm-vllm-serving.md` + `vlm-transcription-cache.md` + `office-pdf-conversion.md`.

**Validated end-to-end (2026-05-27):** a CC-BY handwritten C++ note (image-only PDF,
0 text layer) → `memex ingest` (VLM on, no force-docling) → the scan route fired
(`parse.scan.start`/`done`), the VLM transcribed every fact faithfully, 4 chunks
indexed → `memex ask "who developed C++ and when?"` → grounded answer *"developed by
Bjørne[sic] Stroustrup in 1979 at Bell Labs."* (The pre-change path crashed Docling on
the same PDF.) Detection signal corrected during validation — see "Detection".

## Problem

A scanned / handwritten PDF is **image-only** (no text layer). The parse classifier
(`parse/pipeline.py::_classify`) detects this two ways — a known scan/OCR producer
(`doc_type="scan"`, Tier 1.B) or image-heavy-with-little-text (`doc_type="image-heavy"`,
`image_heavy_page_fraction > 0.5 and chars_per_page_avg < 100`) — and in **both** sets
`needs_ocr=True`, which falls through to **Docling with `force_ocr`**. That is wrong for
this content:

- Docling's OCR is **printed-text** OCR (EasyOCR/Tesseract) — it cannot read cursive /
  handwriting, only typeset glyphs.
- On a real image-only handwritten PDF it **crashed** (`DoclingCrashed`, worker exit 5,
  under the seccomp sandbox — validated 2026-05-27 on the GNHK/CS-Notes corpus).

Meanwhile the **VLM** (Qwen3-VL-8B, via `parse/vlm_backend.convert_pages`) transcribes
handwriting **faithfully** — validated 2026-05-27 (a handwritten C++ note → clean
markdown with every fact + the diagram). But the VLM is only reached via Docling's
*per-page* escalation (`_route_and_escalate`, image_fraction ≥ 0.20), which requires a
**successful Docling parse first** — and Docling crashes before any page escalates. So a
full scan never reaches the model that can read it.

## Decision

Add a **scan route**: when the classifier flags a scan-type doc AND the VLM is enabled,
transcribe **every page** with the VLM (`convert_pages`) and assemble the page markdown —
**bypassing Docling entirely**. This is the document-level analogue of the existing
per-page VLM escalation; it reuses the same VLM serving + cache + prompt.

This is the correct route (not a workaround): a scanned/handwritten doc *should* go to
the handwriting-capable model, not to printed-text OCR.

## Detection & routing

`_classify` already produces the signal — no new heuristic. The scan-type doc types are
`{"scan", "image-heavy"}` (both carry `needs_ocr=True`). Today `_parse_with_pymupdf`
computes the `_Classification` locally and returns only a `_PreFilterDecision`
(`result | None`, `force_ocr_on_fallthrough`). We thread the scan signal out:

```python
@dataclass(frozen=True)
class _PreFilterDecision:
    result: ParseResult | None
    force_ocr_on_fallthrough: bool = False
    is_scan: bool = False   # NEW — classification.doc_type in {"scan","image-heavy"}
```

`_parse_pdf`, after the PyMuPDF pre-filter punts (`decision.result is None`):

```python
if decision.is_scan and not settings.parse.disable_vlm and source.suffix.lower() == ".pdf":
    return await _parse_scan_with_vlm(vault_path, doc_id, source, refresh_vlm=refresh_vlm)
# else: Docling fallthrough (unchanged), force_ocr=decision.force_ocr_on_fallthrough
```

**Gating** (matches the escalation's VLM gate):
- `disable_vlm=True` → NO scan route; fall through to Docling-OCR (current best-effort
  behaviour, unchanged — so this is purely additive and backward-compatible).
- `force_docling=True` (explicit operator override) bypasses the PyMuPDF pre-filter
  entirely → goes straight to Docling; the scan route is the AUTO path only. (A scan
  should simply not be force-docling'd.)

## `_parse_scan_with_vlm`

Modelled on `_parse_with_docling`'s write/manifest/return tail; the body comes from the
VLM instead of Docling.

```python
async def _parse_scan_with_vlm(
    vault_path: Path, doc_id: str, source: Path, *, refresh_vlm: bool = False
) -> ParseResult:
```

1. **Enumerate pages** — `pypdfium2.PdfDocument(str(source))`; `n = len(doc)`. Page
   numbers are **1-based** for `convert_pages` (`[1..n]`).
2. **Open the VLM cache** — `VLMTranscriptionCache.open(vault_path)` (try/finally close),
   same as `_parse_with_docling`.
3. **Transcribe all pages, orchestrator paused**:
   ```python
   async with pause_vllm_for_gpu():            # nestable; the CLI already holds it
       results = await convert_pages(
           source_pdf=source, page_numbers=list(range(1, n + 1)),
           cache=vlm_cache, refresh_vlm=refresh_vlm,
       )                                        # dict[int, DoclingPageOutput | Exception]
   ```
4. **Per-page decisions + stitch** (reading order): a `DoclingPageOutput` → its markdown +
   `PageDecision(engine="scan", confidence=1.0)`; an `Exception` → skip its markdown +
   `PageDecision(engine="scan", confidence=0.0, rationale="VLM scan failed: …")`.
   `markdown = "\n\n".join(p.markdown for p in ordered if p.markdown)`.
5. **Finalize + write** — `_finalize_body(markdown)` (the same table-linearization /
   cleanup the other routes use) → `write_document` → `update_manifest(parse=ParseStage(
   …, pages=decisions, …))`.
6. **Return** `ParseResult(doc_id, correlation_id, engine="scan", pages=decisions,
   markdown_bytes=…)`.

`PageDecision.engine` Literal gains `"scan"` (`manifest.py`); `ParseResult.engine` is a
free `str`.

**No chart-OCR pass** on the scan route — the VLM already transcribes any diagrams/charts
inline (its prompt does), so there are no `<!-- image -->` placeholders to stitch.

## GPU lifecycle

`convert_pages` (vLLM path) starts a short-lived VLM vLLM on the GPU, so the orchestrator
must be down. `_parse_scan_with_vlm` wraps the call in `pause_vllm_for_gpu()` exactly like
`_route_and_escalate`; it's nestable, so the CLI `ingest`/`index`/`reindex` outer pause
makes the inner one a no-op (one pause for the whole run, one restart at the end).

## Reproducibility & re-parse

The VLM cache (`vlm_cache.sqlite`, content+model+prompt-addressed) makes a re-parse of the
same scan **replay** the cached transcriptions (zero VLM forward passes on a full hit) —
the determinism guarantee from `vlm-transcription-cache.md` extends to the scan route.
`memex parse --refresh-vlm <doc>` busts it; the cache is in the `reindex --force` teardown.

## Edge cases

- **All pages fail VLM** → empty markdown → the doc indexes to 0 chunks → answer/summarize
  **refuses** (HARD-gate-safe; never fabricates from an unreadable scan).
- **A page fails** → that page's content is absent (logged); the rest stitch normally.
- **Mixed doc** (substantial text + some scanned pages) → `chars_per_page_avg ≥ 100` →
  NOT image-heavy → goes to Docling with per-page escalation (existing path), not the
  whole-doc scan route. The scan route is for predominantly-image docs.
- **VLM disabled** → Docling-OCR fallback (unchanged) — a scan still parses (poorly), no
  regression.

## Testing

- **Unit** (`test_scan_route.py` or `test_pipeline_*`): `_classify` flags
  image-heavy/scan → `is_scan` threaded through `_PreFilterDecision`; the page-decision +
  markdown-stitch assembly from a faked `convert_pages` dict (incl. a failed page →
  skipped + `engine="scan"`/conf 0). The `disable_vlm` gate (→ no scan route).
- **Integration** (`test_office_routing.py` sibling / `test_scan_routing.py`): a faked
  `convert_pages` + a no-text PDF signal → `parse_document` routes to the scan path,
  writes the stitched markdown, records `engine="scan"` pages in the manifest. Fakes the
  VLM (no GPU), per the test conventions.
- **Live (GPU)**: ingest the CC-BY CS-Notes handwritten PDFs (no `--force-docling`, VLM
  enabled) → the scan route transcribes → vault markdown carries the handwriting → chunks
  index → `memex ask`/`summarize` answer from them. (Then the handwritten eval corpus.)

## Out of scope (deferred)

- A dedicated handwriting HTR model (the VLM suffices; revisit only if accuracy gaps show).
- Scan-specific summary route tuning (the `scan` SUMMARY route in ADR-0008 — separate;
  this is the PARSE route that makes scans ingestable in the first place).
