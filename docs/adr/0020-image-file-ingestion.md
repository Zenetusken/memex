# ADR-0020: Standalone Image-File Ingestion via the Scan→VLM Route

- **Status**: Accepted (v1 shipped 2026-06-05, branch `feat/image-ingestion`)
- **Date**: 2026-06-05
- **Deciders**: Memex core team
- **Tags**: parse, ingest, vlm, images, architecture

## Context

Memex ingests PDF / Office / scanned-PDF / audio / video, but a **standalone image file**
(`.png`, `.jpg`, …) was **rejected at `ingest/validation.py`** — its magic-detection had no image
kind. Yet a screenshot, an infographic, an exported topology diagram, or a photographed page is
common drop-in class material (the user's own `CISCO CyberOps.png`). The gap was tabled during the
UI-ingestion live test (2026-06-05) as the one clear, self-contained code item next.

**An image is a one-page scan.** The scan→VLM route already exists for exactly this shape —
whole-document VLM transcription that bypasses Docling-OCR (`parse/pipeline.py::_parse_scan_with_vlm`
→ `vlm_backend.convert_pages` → the VLM cache → `_assemble_scan_pages`; ADR-0006 §4,
`docs/specs/scan-vlm-parse.md`). So the build is **maximal reuse**: detect the image magics → render
the image to a **cached** 1-page PDF → run the *unchanged* scan route. This mirrors the
**Office→PDF precedent** (ADR — `docs/specs/office-pdf-conversion.md`): pypdfium2 (the VLM page
rasteriser) is PDF-only, so the non-PDF source is wrapped into a PDF up front and the existing PDF
pipeline does the rest.

## Decision Drivers

- **Maximal reuse / no new grounding surface** — an image must flow through the *unchanged* VLM →
  markdown → index → grounding path. No new fabrication surface; a faithful caption of a photo is
  *correct* grounded content, not a hallucination.
- **Grounded HARD gate preserved** — `refusal_cf=1.0` / 0-hallucination is untouchable. Images are a
  parse-stage perception input, never a new grounding path.
- **Reproducible re-parse** — content-addressed `chunk_id`s require deterministic-or-cached output.
- **Local-first / 12 GB single GPU** — no cloud OCR; reuse the existing parse-time VLM serve.
- **Works out of the box** — `memex ingest photo.png` must Just Work with the default config (where
  `disable_vlm=True`), exactly as `memex ingest lecture.mp3` does.

## Decision

**Accept standalone images as a new parse route that wraps the image into a cached 1-page PDF and
runs the existing scan→VLM transcription**, `engine="image"`. Two user decisions, both confirmed:

1. **The full PIL raster set** — `IMAGE_SUFFIXES = {.png, .jpg, .jpeg, .webp, .bmp, .tif, .tiff,
   .gif}`. HEIC/AVIF stay deferred (they need a separate decode dependency; their `ftyp` brands also
   stay rejected at validation — see Revisit).
2. **The VLM is mandatory for images.** An image has **no non-VLM extraction path** (there is no text
   layer to read, no Docling-OCR fallback worth running on an arbitrary raster) — exactly like the
   audio route always runs ASR (ADR-0017). This falls out **by construction**: `parse_document` routes
   an image **direct to `_parse_scan_with_vlm`** (not `_parse_pdf`), and that route **never consults
   `parse.disable_vlm`**. So the default `disable_vlm=True` is bypassed for images with **zero extra
   code** — no gate, no special-case flag.

### Mechanism (verified against the source)

- **Validation is orthogonal to parsing.** `ingest/validation.py` accepts the image magics (a new
  `_detect_image`; `DetectedKind` gains `"image"`) — distinct offset-0 magic that **cannot collide**
  with the excluded HEIC/AVIF `ftyp` brands (`_VIDEO_FTYP_BRANDS`) or WAV's `RIFF`+`WAVE`. The parse
  pipeline then routes **purely on file suffix**. `image` is deliberately **absent** from
  `_EXTENSION_FOR_KIND`, so the ingest copy preserves the original `.png`/`.jpg`/`.webp` suffix (the
  audio precedent) — which the suffix-based parse route keys on. (Inc-1, `457ec72`.)
- **Render → 1-page PDF → reuse the scan route** (`parse/image_convert.py::convert_image_to_pdf`,
  Inc-2). PIL is lazy-imported (a `[parse]` dep) and the decode+save runs under `asyncio.to_thread`.
  A non-`RGB`/`L` mode (RGBA / palette / CMYK / LA) is flattened to **RGB** so the rendered raster is
  deterministic (a PDF image is rendered without an alpha channel); a multi-frame source (multi-page
  TIFF / animated GIF) takes its **first frame** in v1. `parse_document` adds an `IMAGE_SUFFIXES`
  branch **after** the Office block and **before** `.pdf`, calling `_ensure_converted_image_pdf` →
  `_parse_scan_with_vlm(engine="image")`. The 1-page PDF renders back to a bitmap inside
  `convert_pages` (`_render_page_to_image`) → the VLM. No new VLM/grounding/index/cache path.
- **Caching the converted PDF is MANDATORY for determinism** (`_ensure_converted_image_pdf`, the
  `_ensure_converted_pdf` analogue). `PIL.Image.save(path, "PDF")` stamps a fresh `time.gmtime()`
  CreationDate/ModDate into the PDF Info trailer (`PdfImagePlugin.py`), so re-converting every parse
  would churn `sha256(pdf_bytes)` → the content-addressed VLM-transcription cache key → re-transcribe
  → `chunk_id` churn (re-embed, eval-anchor drift). Caching as `documents/{doc_id}/converted.pdf`
  (returned early on re-parse, before any re-conversion) keeps the bytes — hence the cache key —
  byte-stable. **Identical to the Office LibreOffice-CreationDate fix.**
- **`engine="image"`** tags the `ParseResult` + every `PageDecision` (the manifest `engine` Literal
  gains `"image"`) so the route is auditable; the scan route's `engine` default stays `"scan"`, so the
  existing scan corpus is **byte-identical**.
- **Free reuse, no webui-route change.** `converted.pdf` is **not** a `source.*` name, so `_source_file`
  still resolves the original `.png` as downloadable provenance, and the webui `_find_preview_pdf`
  already returns `converted.pdf` → the doc-view page-image preview works for free (both like Office).
  The webui `POST /ingest` + `ingest_driver` already forward any file to `memex ingest` (validation
  decides), so only the `ingest.html` accepted-types copy changes.

## HARD-gate neutrality is STRUCTURAL

Not "images yield clean text." Image → VLM → markdown flows through the **unchanged** index/grounding
gate, so the gate posture is identical to any scanned PDF. A **blank/unreadable image** transcribes
to nothing → `_parse_scan_with_vlm`'s `if not parts: raise ParseConfidenceTooLow(recoverable=True)`
fires **before `write_document`** → **no junk 0-chunk doc** is ever written → any query against it
refuses honestly. The VLM cache extends the determinism guarantee to images (a re-parse replays the
transcription). Parse-stage only ⇒ `/ask`, `summarize`, chat, the bridge, MCP, and their HARD gates
are byte-untouched.

## Considered Options

1. **Render → 1-page PDF → scan→VLM route (CHOSEN).** Maximal reuse; the Office precedent. Gets the
   VLM cache, the page-image preview, and provenance **for free**; VLM-mandatory falls out of routing
   direct to the scan route.
2. **Direct image → VLM (no PDF wrap).** Rejected: it forks the VLM serving + cache path (which is
   `sha256(pdf_bytes)`-keyed and PDF-page-oriented) and **loses the free page-image preview** and the
   uniform provenance handling, for no benefit — `convert_pages` already rasterises the 1-page PDF
   back to the native pixels.
3. **Docling image OCR.** Rejected as the route: Docling's OCR is **printed-text-only** (EasyOCR/
   Tesseract) — it can't read an arbitrary infographic/diagram/photo, which is the whole point. (NB:
   at the Inc-1 boundary, *before* the Inc-2 route landed, a now-accepted image fell through
   `parse_document` to the Docling fallthrough — closed by the Inc-2 routing branch.)

## Consequences

### Positive

- A screenshot / diagram / photographed page becomes a first-class, searchable,
  **grounded-answerable** vault document with **no change to the answering graph or its HARD gates**.
- Reuses the proven parse machinery: the suffix-dispatch precedent (`office`/`scan`/`audio`), the
  cached-`converted.pdf` determinism pattern, the parse-time VLM serve, the VLM transcription cache,
  the page-image preview, and source provenance.
- VLM-mandatory is **structural, not a flag** — `memex ingest photo.png` works out of the box under
  the default `disable_vlm=True`.

### Negative / Trade-offs

- Adds **`pillow>=10`** as an explicit `[parse]` dependency (it was already transitive via
  docling/pymupdf4llm + `parse/pdf_render`; now pinned because the route depends on it directly).
- **Multi-frame sources take the first frame only** in v1 (multi-page TIFF / animated GIF). The scan
  route already handles multi-page PDFs, so full multi-frame transcription is a converter-only
  extension (Revisit).
- **HEIC/AVIF are not accepted** (decode dependency deferred).

### Neutral

- VISION's "Markdown as source of truth" holds — the canonical transcript `.md` is content-only; the
  `converted.pdf` is regenerable derived state in the doc dir (`memex remove` drops it; a fresh ingest
  re-converts). The VLM transcription sidecar/cache follows the `chart_extractions` precedent
  (ADR-0003).

## Revisit When

- **HEIC/AVIF** are requested often enough to justify a decode dependency (`pillow-heif` or similar) —
  then un-exclude their `ftyp` brands at validation and add the suffixes to `IMAGE_SUFFIXES`.
- **Full multi-page TIFF / animated GIF transcription** is wanted — a converter-only change
  (`convert_image_to_pdf` emits an N-page PDF; the scan route already transcribes every page).
- **A non-VLM image path** ever becomes worthwhile (e.g. a fast printed-text screenshot OCR) — it
  would be an *additional* route, not a change to this one (images stay VLM-mandatory by default).
- **Off-list-but-magic-valid extensions** (e.g. a JPEG saved as `.jfif`/`.jpe`) — validation accepts
  by magic but the parse route keys on the fixed `IMAGE_SUFFIXES` set, so such a file misses the image
  branch and falls through to Docling (a clean *typed* parse failure, **not** a crash or a junk doc).
  The fixed set covers all common extensions; widen `IMAGE_SUFFIXES` if a real off-list extension shows
  up. (A future option: route by the validator's detected `kind=="image"` instead of by suffix.)

## References

- **Spec:** [`image-ingestion.md`](../specs/image-ingestion.md) — the implementation design.
- [ADR-0006](0006-cuda-dispatch-and-dtype.md) §4 — the parse-time VLM-via-vLLM serve this route reuses.
- [ADR-0017](0017-audio-asr-ingestion-route.md) — the audio route whose **always-run-the-perception-model**
  shape (ASR is mandatory for audio) this route mirrors (the VLM is mandatory for images).
- [ADR-0003](0003-markdown-vault-as-source-of-truth.md) — content-only `.md` + regenerable derived
  state (`converted.pdf` + the VLM cache follow this).
- Specs [`scan-vlm-parse.md`](../specs/scan-vlm-parse.md) (the route reused),
  [`office-pdf-conversion.md`](../specs/office-pdf-conversion.md) (the convert-then-cache precedent),
  [`vlm-transcription-cache.md`](../specs/vlm-transcription-cache.md) (the determinism cache).
