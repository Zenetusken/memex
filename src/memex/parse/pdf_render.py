"""Render PDF pages to PNG bytes for the webui source-preview pane.

pypdfium2 + PIL ONLY — no ML/Docling imports — so the webui can import this
without pulling in the heavy VLM/chart-OCR stack. The parse pipeline rasterises
pages the same way (`vlm_backend._render_page_to_image`); this is the light,
dependency-minimal twin the preview pane uses.

Why server-side rasterisation instead of embedding the PDF in an ``<iframe>``:
native in-browser PDF rendering is defeated by the browser's "download PDFs
instead of opening" setting (a per-user preference no ``Content-Disposition``
header can override) and is flaky inside iframes generally. Rendering each page
to a PNG the pane shows as a plain ``<img>`` works in every browser/setting, is
air-gap-safe (no client-side PDF library), and is the right affordance for
scanned / handwritten docs — the original page sits beside its transcription.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import cast

from memex.core.errors import MemexError

# pypdfium2 / PDFium is NOT thread-safe — concurrent PdfDocument open+render across
# threads can SEGFAULT. The webui preview pane emits many `<img>` requests that each
# render a page via `asyncio.to_thread`, so serialize ALL pdfium access in this
# process behind one lock. A render is ~100-200 ms; for a single-user localhost
# preview, serializing is imperceptible — and crash-free (a deck's ~30 concurrent
# page renders segfaulted the webui without it).
_PDFIUM_LOCK = threading.Lock()

# ~144 DPI for a Letter/A4 page — crisp enough to read scanned handwriting
# without ballooning the PNG. Matches vlm_backend's page-render scale.
_PREVIEW_SCALE = 2.0


class PDFPreviewError(MemexError):
    """A source PDF page could not be rasterised for the preview pane
    (pypdfium2 missing, page out of range, or a render failure)."""


def _wrap_pdfium(e: Exception, message: str, **context: object) -> PDFPreviewError:
    """Re-tag a pypdfium2 / IO failure as a `PDFPreviewError` so callers (the
    webui route guards) catch ONE error type. Identifies the SDK by module name
    rather than catching its exception base directly (the convention in
    `src/memex/CLAUDE.md` for third-party SDK errors)."""
    return PDFPreviewError(message, context={**context, "underlying": str(e)})


def _open(pdf_path: Path):  # type: ignore[no-untyped-def]  # -> pdfium.PdfDocument
    try:
        import pypdfium2 as pdfium
    except ImportError as e:  # pragma: no cover - exercised only without [parse]
        raise PDFPreviewError(
            "pypdfium2 is required to render PDF previews; install with `uv sync --extra parse`",
            context={"underlying": str(e)},
        ) from e
    try:
        return pdfium.PdfDocument(str(pdf_path))
    except Exception as e:  # a corrupt / truncated PDF — pdfium raises PdfiumError
        if type(e).__module__.split(".")[0] == "pypdfium2" or isinstance(e, OSError | ValueError):
            raise _wrap_pdfium(e, "could not open PDF for preview", path=str(pdf_path)) from e
        raise


def pdf_page_count(pdf_path: Path) -> int:
    """Number of pages in a PDF — the preview pane emits one ``<img>`` per page.
    Raises `PDFPreviewError` on an unreadable PDF (the route degrades to no-pane)."""
    with _PDFIUM_LOCK:
        doc = _open(pdf_path)
        try:
            return len(doc)
        except Exception as e:
            if type(e).__module__.split(".")[0] == "pypdfium2":
                raise _wrap_pdfium(e, "could not read PDF page count", path=str(pdf_path)) from e
            raise
        finally:
            doc.close()


def pdf_page_size(pdf_path: Path, page_index: int = 0) -> tuple[float, float]:
    """Native width/height **in points** for one page — the preview pane uses it
    as the CSS `aspect-ratio` on placeholder `<img>`s, so the browser-native
    `loading="lazy"` correctly defers offscreen pages (a 0-height placeholder is
    "near viewport" and fires immediately, defeating the lazy attribute).

    Raises `PDFPreviewError` on the same conditions as the page renderer.
    """
    with _PDFIUM_LOCK:
        doc = _open(pdf_path)
        try:
            page_count = len(doc)
            if not 0 <= page_index < page_count:
                raise PDFPreviewError(
                    f"page index {page_index} out of range",
                    context={"page_index": page_index, "page_count": page_count},
                )
            # `get_size()` is a public pypdfium2 PdfPage method but isn't on the
            # type stubs — `cast` pins the (width, height) shape locally so the
            # callers see a strict `tuple[float, float]`.
            size = cast(tuple[float, float], doc[page_index].get_size())  # type: ignore[attr-defined]
            return float(size[0]), float(size[1])
        except PDFPreviewError:
            raise
        except Exception as e:
            if type(e).__module__.split(".")[0] == "pypdfium2":
                raise _wrap_pdfium(e, "could not read PDF page size", page_index=page_index) from e
            raise
        finally:
            doc.close()


def render_pdf_page_png(pdf_path: Path, page_index: int, *, scale: float = _PREVIEW_SCALE) -> bytes:
    """Rasterise a **0-based** page to PNG bytes.

    Raises `PDFPreviewError` when the page is out of range, the PDF is
    unreadable, or rendering fails. The PNG is encoded **before** the document
    closes, so PIL never reads pypdfium2's freed C-owned bitmap buffer (cf. the
    N7 audit note in `vlm_backend._render_page_to_image`).
    """
    with _PDFIUM_LOCK:
        doc = _open(pdf_path)
        try:
            page_count = len(doc)
            if not 0 <= page_index < page_count:
                raise PDFPreviewError(
                    f"page index {page_index} out of range",
                    context={"page_index": page_index, "page_count": page_count},
                )
            bitmap = doc[page_index].render(scale=scale)
            buf = io.BytesIO()
            bitmap.to_pil().save(buf, format="PNG")  # encoded while the doc is still open
            return buf.getvalue()
        except PDFPreviewError:
            raise
        except Exception as e:  # a render failure on an otherwise-openable PDF
            if type(e).__module__.split(".")[0] == "pypdfium2":
                raise _wrap_pdfium(e, "could not render PDF page", page_index=page_index) from e
            raise
        finally:
            doc.close()
