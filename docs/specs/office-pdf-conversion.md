# Spec — Office (pptx/docx/xlsx) → PDF at parse, then the full PDF pipeline

**Status:** Shipped 2026-05-26 (commit `11e80fb`). Live-validated on 31 CCNA `.pptx` decks.

## Problem

The ingest pipeline accepts Office formats (`OFFICE_SUFFIXES` = `.pptx/.ppt/.docx/.doc/.xlsx/.xls/.odp/.odt/.ods`), and Docling parses them to markdown. But the **VLM-escalation and chart-OCR passes render pages + figure crops via pypdfium2, which is PDF-only** — handed a `.pptx` it raises `PdfiumError: Data format error` and crashes the chart-OCR pass, so a diagram-heavy slide deck's figures are never transcribed. PowerPoint course decks (the motivating real use case) are exactly diagram-heavy slide decks.

Native pptx rendering was rejected: there is no robust pure-Python pptx→image renderer; LibreOffice is the standard tool (Docling itself shells out to it for some formats). Converting Office→PDF up front and running the **existing** PDF pipeline is the industry-standard, engine-agnostic approach — and it means figure bboxes (from Docling) align with the page raster (from pypdfium2) because both see the same PDF.

## Design

### 1. Conversion module — `parse/office_convert.py`
- `OFFICE_SUFFIXES` drives routing.
- `convert_to_pdf(source, out_dir, *, timeout_s=180) -> Path` runs `soffice --headless --convert-to pdf` in a subprocess via `asyncio.to_thread`. Raises `OfficeConversionError` (a `MemexError`) on missing LibreOffice / non-zero exit / timeout / no output.
- **Broken-`soffice` workaround (load-bearing in some envs):** the launcher sometimes fails with `libreglo.so: cannot open shared object file` *though the file is present* in the LibreOffice program dir — the wrapper just doesn't add its own libs to the loader path. `convert_to_pdf` sets `LD_LIBRARY_PATH` to the detected program dir (`_libreoffice_lib_dir` — the dir containing `soffice.bin` among the common locations). Defensive + harmless where the wrapper already works.
- A per-conversion throwaway `UserInstallation` profile avoids clashing with a desktop LibreOffice's profile lock.

### 2. Routing — `parse/pipeline.py::parse_document`
An Office source is converted to a cached **`documents/{doc_id}/converted.pdf`** (via `_ensure_converted_pdf`), then routed to `_parse_pdf` with **`force_docling=True`** — the converted PDF carries LibreOffice/Impress producer metadata, which the classifier would send to PyMuPDF (no figure-transcription stage), so we force the Docling VLM/chart-OCR path. The original Office file stays as `source.{ext}` (provenance); `converted.pdf` is deliberately **not** a `source.*` name so `_source_file` still resolves the original.

### 3. Cache stability (why `converted.pdf` is cached, not transient)
LibreOffice stamps a fresh `CreationDate` into every conversion, so re-converting each parse would churn the PDF bytes — and the **content-addressed VLM / chart-OCR cache keys are keyed on those bytes**. Caching the converted PDF in the doc dir and reusing it on re-parse keeps the bytes byte-stable, so the VLM/chart-OCR caches replay correctly across re-parses. (`memex remove` drops the doc dir, so a fresh ingest re-converts.)

### 4. GPU handoff — `pause_vllm_for_gpu` (renamed public, `parse/pipeline.py`)
Office decks are large (many diagram pages → many VLM calls → many chunks). On the 12 GB tier the **embedder OOMs co-resident with vLLM** (~8.5 GB resident — genuine over-allocation, not fragmentation). The parse's own per-VLM pause restarts vLLM *before* the index embed runs, which then OOMs. Fix: the CLI `ingest`/`index`/`reindex` chains wrap their parse+index work in `pause_vllm_for_gpu()` — vLLM stays down for the duration (the parse's inner pause then no-ops: sees vLLM unreachable, yields without pausing, skips the restart) and restarts once at the end. `async with`-nestable; no-op when vLLM isn't running. (The reranker's OOM is handled separately by `retrieve/rerank.py::_score_with_oom_fallback` — a batch-1 retry, since that one is recoverable in-process.)

## Files
| File | Change |
|---|---|
| `parse/office_convert.py` (NEW) | `convert_to_pdf` + `OFFICE_SUFFIXES` + `OfficeConversionError` + the lib-path detection |
| `parse/pipeline.py` | `_ensure_converted_pdf`; the Office branch in `parse_document`; `_pause_vllm_for_gpu_parse` → public `pause_vllm_for_gpu` |
| `parse/__init__.py` | re-export `convert_to_pdf`, `OFFICE_SUFFIXES`, `OfficeConversionError`, `pause_vllm_for_gpu` |
| `cli/commands.py` | `ingest`/`index`/`reindex` wrap their work in `pause_vllm_for_gpu()` |

## Tests
- `tests/integration/test_office_routing.py` — fakes `convert_to_pdf` + `_parse_pdf`: an Office source converts ONCE to `converted.pdf`, routes to `_parse_pdf(force_docling=True)`, preserves the original `source.pptx`, and reuses the cached PDF on re-parse (no second conversion). The real LibreOffice conversion + Docling pass are validated on the rig, not in pytest.

## Verification (the live proof)
`SRWE_Module_1.pptx` → `converted.pdf` (926 KB) → Docling (force_docling) → **19 diagram pages VLM-transcribed** → chart-OCR 2 figs / 0 `stitch_count_mismatch` → **88 chunks indexed, no OOM**, 0 broken `![]()` image links. Then the full library: **all 31 CCNA decks** (16 SRWE + 15 ENSA) ingested, 0 failures.

## Anti-scope
- NOT a native pptx renderer (LibreOffice is the dependency; clearly errored if absent).
- NOT converting on every parse (cached for byte-stability).
- NOT pausing vLLM per-deck in a bulk (the CLI holds one pause across the whole `ingest` chain).
