# Spec — Standalone image → 1-page PDF at parse, then the scan→VLM route

**Status:** Shipped 2026-06-05 (branch `feat/image-ingestion`; ADR-0020). Inc-1 validation
`457ec72`, Inc-2 parse route `a548abd`.

## Problem

The ingest pipeline rejected standalone image files (`.png`, `.jpg`, …) — `ingest/validation.py`
had no image kind. A screenshot, infographic, exported diagram, or photographed page is common
drop-in material and had no path in. An **image is a one-page scan**: the scan→VLM route
(`docs/specs/scan-vlm-parse.md`) already transcribes whole image-only documents with the VLM,
bypassing Docling-OCR (printed-text-only). The build is therefore **maximal reuse** — accept the
image at validation, wrap it into a cached 1-page PDF (the Office→PDF precedent,
`docs/specs/office-pdf-conversion.md`), and run the unchanged scan route.

## Design

### 1. Validation accepts image magics — `ingest/validation.py` (Inc-1)
- `DetectedKind` gains `"image"`. `_detect_image(head) -> (kind, mime, has_macros) | None` matches:
  PNG `\x89PNG\r\n\x1a\n`, JPEG `\xff\xd8\xff`, WebP `RIFF`@0 + `WEBP`@8 (distinct from WAV's
  `RIFF`+`WAVE`), TIFF `II*\x00`/`MM\x00*`, and the **ASCII-startable** BMP `BM` / GIF `GIF87a|GIF89a`
  **gated on `not _looks_like_text(head)`** (so prose "BMW…" / "GIF89a is a format…" stays text).
  Called in `_detect()` **after** `_detect_audio`/`_detect_video`, **before** the text fallback.
- HEIC/AVIF `ftyp` **image** containers stay **rejected** (not matched by `_detect_image`, excluded
  from `_VIDEO_FTYP_BRANDS`) — they need a separate decode dependency (ADR-0020 Revisit).
- `image` is deliberately **absent** from `ingest/pipeline.py::_EXTENSION_FOR_KIND`, so the ingest copy
  falls back to the **original suffix** — `.png` vs `.jpg` vs `.webp` are preserved (the audio
  precedent), which the suffix-based parse route keys on. Size/empty/macro gates are unchanged (an
  image is never `has_macros`).

### 2. Conversion module — `parse/image_convert.py` (Inc-2)
- `IMAGE_SUFFIXES = {.png, .jpg, .jpeg, .webp, .bmp, .tif, .tiff, .gif}` drives parse routing.
- `convert_image_to_pdf(source, out_dir) -> Path` — PIL is **lazy-imported** (a `[parse]` dep, the
  `office_convert`/`pdf_render` discipline) and the decode+save runs under `asyncio.to_thread`. Opens
  the first frame; a non-`RGB`/`L` mode (RGBA / palette / CMYK / LA) is flattened to **RGB** (a PDF
  image is rendered without an alpha channel → a deterministic raster); saves
  `out_dir/{stem}.pdf` at `_PDF_RESOLUTION_DPI = 144` (so page_points = px/2 → the scan route's
  scale-2.0 `_render_page_to_image` reproduces the image at its **native** pixel resolution; the VLM
  processor's `max_pixels` still caps a very large image). Raises `ImageConversionError` (a
  `MemexError` with `context`) on a corrupt/truncated/over-large image (`OSError` —
  `UnidentifiedImageError` subclasses it — / `ValueError` / `DecompressionBombError`).

### 3. Routing — `parse/pipeline.py::parse_document`
After the Office branch and before `.pdf`:
```python
if source.suffix.lower() in IMAGE_SUFFIXES:
    source = await _ensure_converted_image_pdf(settings.vault_path, doc_id, source)
    return await _parse_scan_with_vlm(settings.vault_path, doc_id, source,
                                      engine="image", refresh_vlm=refresh_vlm)
```
Routing **direct to `_parse_scan_with_vlm`** (not `_parse_pdf`) is the mechanism that makes the VLM
**mandatory**: the scan route **never consults `parse.disable_vlm`**, so the default `disable_vlm=True`
is bypassed for images with **zero extra code** (the audio-route precedent — an image has no non-VLM
extraction path). `_parse_scan_with_vlm` + the pure `_assemble_scan_pages` were parameterized with
`engine: Literal["scan","image"]="scan"` — the default keeps the scanned-PDF path **byte-identical**
(same `PageDecision`/`ParseResult`/log/rationale); `engine="image"` tags the manifest so the route is
auditable. The manifest `PageDecision.engine` Literal gains `"image"`.

### 4. Cache stability — `_ensure_converted_image_pdf` (why `converted.pdf` is cached, not transient)
The `_ensure_converted_pdf` analogue: return `documents/{doc_id}/converted.pdf` if it exists (no
re-conversion), else convert in a tempdir and `shutil.move` it into place. `PIL.Image.save(path,
"PDF")` stamps a fresh `time.gmtime()` CreationDate/ModDate into the PDF Info trailer, so re-converting
each parse would churn the PDF bytes — and the **content-addressed VLM-transcription cache key is
keyed on `sha256(pdf_bytes)`**. Caching the converted PDF in the doc dir and reusing it on re-parse
keeps the bytes byte-stable, so the VLM cache replays correctly across re-parses (the original
LibreOffice-CreationDate problem). `converted.pdf` is deliberately **not** a `source.*` name, so
`_source_file` still resolves the original image as provenance, and the webui `_find_preview_pdf`
returns it for the doc-view page-image preview (both for free, like Office). `memex remove` drops the
doc dir → a fresh ingest re-converts.

### 5. HARD-gate neutrality is STRUCTURAL
Image → VLM → markdown flows through the **unchanged** index/grounding gate (identical posture to a
scanned PDF). A blank/unreadable image transcribes to nothing → `_parse_scan_with_vlm`'s
`if not parts: raise ParseConfidenceTooLow(recoverable=True)` fires **before `write_document`** → **no
junk 0-chunk doc** → an honest refuse. Parse-stage only ⇒ `/ask`/`summarize`/chat/bridge/MCP and their
HARD gates are byte-untouched.

### 6. Surfaces (Inc-3)
- **CLI** — `memex ingest photo.png` works out of the box (validation accepts it; the parse route runs
  the VLM regardless of `disable_vlm`).
- **WebUI** — `POST /ingest` + `ingest_driver` already forward any file to `memex ingest`, so only the
  `ingest.html` accepted-types copy changes ("image (PNG/JPEG/WebP/BMP/TIFF/GIF)"). The doc-view
  page-image preview + the original-image download link work for free (the Office reuse).

## Files
| File | Change |
|---|---|
| `ingest/validation.py` (Inc-1) | `_detect_image` + `"image"` `DetectedKind`; docstring |
| `parse/image_convert.py` (NEW, Inc-2) | `convert_image_to_pdf` + `IMAGE_SUFFIXES` + `ImageConversionError` |
| `parse/pipeline.py` (Inc-2) | `_ensure_converted_image_pdf`; the image branch; `_parse_scan_with_vlm`/`_assemble_scan_pages` `engine` param |
| `core/manifest.py` (Inc-2) | `PageDecision.engine` Literal gains `"image"` |
| `parse/__init__.py` (Inc-2) | re-export `IMAGE_SUFFIXES` / `ImageConversionError` / `convert_image_to_pdf` |
| `pyproject.toml` (Inc-2) | pin `pillow>=10` in `[parse]` |
| `webui/templates/ingest.html` (Inc-3) | accepted-types copy mentions images |

## Tests
- `tests/unit/test_validation_image.py` — per-format magic detection; `validate_file` accepts each;
  HEIC `ftyp` still rejected; WAV `RIFF` stays audio (no WebP collision); BMP/GIF text-gating;
  truncated-head no crash; suffix preserved via the `_EXTENSION_FOR_KIND` fallback.
- `tests/unit/test_image_convert.py` — real PIL on tiny fixtures: every mode (RGB/L direct;
  RGBA/P/CMYK/LA → RGB) + every format (PNG/JPEG/WebP/BMP/TIFF/GIF) → a valid single-page PDF;
  multi-frame GIF → first frame; corrupt bytes → typed `ImageConversionError`; out-dir created; the
  **DPI↔scale coupling guard** (a 288×144 image → a 144×72 pt page = px/2, so the scan route's
  scale-2.0 render reproduces native pixels — pins `_PDF_RESOLUTION_DPI` against a silent regression).
- `tests/integration/test_image_routing.py` — fakes the PIL convert + the VLM serving/cache/page-count:
  a `.png` routes through `parse_document` to `engine="image"` **ignoring `disable_vlm`**; converts
  ONCE then reuses the cached `converted.pdf`; preserves the original `.png` (provenance); a blank
  image → `ParseConfidenceTooLow`, **no doc written**.
- `tests/integration/test_webui.py` — the ingest page copy advertises images.

## Verification (live, end-to-end)
1. **CLI** — `memex ingest diagram.png` (default `disable_vlm=True`, ignored by the image route) →
   `memex ask "what does the diagram show?"` → a grounded, cited answer transcribed from the image.
   Re-parse → the VLM does **not** re-run (cache hit via the stable `converted.pdf`); `--refresh-vlm` →
   a fresh draw.
2. **WebUI** — `/ingest` upload a `.png` → progress Parsing→Transcribing→Indexing→Enriching → done →
   the doc is browsable + askable; the source-preview pane renders the page image (from
   `converted.pdf`); the download link serves the original `.png`.
3. **HARD-gate honesty** — (a) a blank/all-white image → `ParseConfidenceTooLow`, no 0-chunk doc → any
   query refuses; (b) a non-document photo + an unsupported question → honest refusal. (Do **not**
   assert refusal on "what is shown?" — a faithful caption there is *correct* grounded behavior.)
