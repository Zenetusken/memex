"""Unit tests for the PyMuPDF routing classifier.

The classifier is a pure function over `PdfSignals` so each tier
can be exercised in isolation with synthesised signals. No vault,
no subprocess, no I/O.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from memex.core.config import MemexSettings, set_settings
from memex.parse.pipeline import _classify
from memex.parse.pymupdf_backend import PdfSignals


@pytest.fixture(autouse=True)
def _settings() -> Iterator[None]:
    """The classifier reads mixed-content thresholds from settings; give
    it a deterministic default rather than relying on whatever ambient
    config is loaded."""
    set_settings(MemexSettings())  # type: ignore[call-arg]
    yield
    set_settings(None)


def _signals(**overrides: object) -> PdfSignals:
    defaults: dict[str, object] = {
        "creator": None,
        "producer": None,
        "is_tagged": False,
        "page_count": 10,
        "avg_aspect_ratio": 1.0,
        "embedded_font_count": 0,
        "image_count_total": 0,
        "image_heavy_page_fraction": 0.0,
        "image_area_fraction": 0.0,
        "total_chars": 0,
        "chars_per_page_avg": 0.0,
        "chars_per_page_median": 0.0,
        "chars_per_page_p10": 0.0,
        "chars_per_page_p90": 0.0,
        "empty_page_fraction": 0.0,
        "replacement_char_fraction": 0.0,
        "word_like_token_fraction": 0.6,
        "unique_char_variety": 50,
        "whitespace_fraction": 0.15,
        "has_headings": False,
        "has_tables": False,
        "has_lists": False,
        "has_code_blocks": False,
    }
    defaults.update(overrides)
    return PdfSignals(**defaults)  # type: ignore[arg-type]


def test_powerpoint_high_confidence() -> None:
    """Born-digital producer + reasonable text → Tier 1.A → use PyMuPDF."""
    sig = _signals(
        producer="Microsoft PowerPoint 2023",
        chars_per_page_avg=200.0,
    )
    result = _classify(sig)
    assert result.doc_type == "born-digital"
    assert result.confidence >= 0.95
    assert result.needs_ocr is False
    assert result.attribution["tier"] == "1.A"


def test_powerpoint_with_mixed_content_routes_to_docling_ocr() -> None:
    """PowerPoint + native text + heavy image area → mixed-content → force OCR."""
    sig = _signals(
        producer="Microsoft PowerPoint 2023",
        chars_per_page_avg=200.0,
        image_area_fraction=0.40,
        image_heavy_page_fraction=0.50,
    )
    result = _classify(sig)
    assert result.doc_type == "mixed-content"
    assert result.confidence < 0.5
    assert result.needs_ocr is True


def test_powerpoint_with_decorative_images_still_uses_pymupdf() -> None:
    """PowerPoint + native text + small image area → still PyMuPDF.
    Defends against false-positive mixed-content triggers on docs with
    just a logo or front-cover image.
    """
    sig = _signals(
        producer="Microsoft PowerPoint 2023",
        chars_per_page_avg=200.0,
        image_area_fraction=0.05,
        image_heavy_page_fraction=0.10,
    )
    result = _classify(sig)
    assert result.doc_type == "born-digital"
    assert result.needs_ocr is False


def test_scanner_producer_forces_docling_ocr() -> None:
    """ABBYY-produced PDF → Tier 1.B → confidence 0.0 + needs_ocr=True."""
    sig = _signals(producer="ABBYY FineReader OCR", chars_per_page_avg=600.0)
    result = _classify(sig)
    assert result.doc_type == "scan"
    assert result.confidence == 0.0
    assert result.needs_ocr is True


def test_powerpoint_rasterised_forces_ocr() -> None:
    """PowerPoint producer + near-zero text = export-as-images → force OCR."""
    sig = _signals(producer="Microsoft PowerPoint 2023", chars_per_page_avg=10.0)
    result = _classify(sig)
    assert result.doc_type == "born-digital-but-rasterised"
    assert result.needs_ocr is True


def test_tagged_pdf_promotes_to_pymupdf() -> None:
    """Tagged PDF + text → Tier 2 → use PyMuPDF at 0.90 confidence."""
    sig = _signals(is_tagged=True, chars_per_page_avg=120.0)
    result = _classify(sig)
    assert result.doc_type == "tagged-pdf"
    assert result.confidence == 0.90
    assert result.needs_ocr is False


def test_image_heavy_routes_to_docling_ocr() -> None:
    """Generic producer, low text, many images per page → force OCR."""
    sig = _signals(
        chars_per_page_avg=50.0,
        image_heavy_page_fraction=0.70,
    )
    result = _classify(sig)
    assert result.doc_type == "image-heavy"
    assert result.needs_ocr is True


def test_mojibake_demotes_without_ocr_force() -> None:
    """U+FFFD > 5% → broken encoding → fall through but don't force OCR.
    OCR can't fix a font-mapping bug; let Docling re-extract.
    """
    sig = _signals(
        chars_per_page_avg=400.0,
        replacement_char_fraction=0.10,
    )
    result = _classify(sig)
    assert result.doc_type == "mojibake"
    assert result.confidence < 0.5
    assert result.needs_ocr is False


def test_near_empty_doc_forces_ocr() -> None:
    """No text anywhere → scan-like → force OCR."""
    sig = _signals(
        chars_per_page_avg=2.0,
        chars_per_page_p90=15.0,
    )
    result = _classify(sig)
    assert result.doc_type == "scan-like"
    assert result.confidence == 0.0
    assert result.needs_ocr is True


def test_landscape_paper_fallback_uses_density() -> None:
    """Generic landscape PDF in Tier 4 → density-based confidence.

    Uses chars-per-page above the slide-deck threshold (Tier 0.5)
    so this test exercises the Tier 4 fallback specifically. Wide-
    format landscape documents (folded brochures, two-column
    landscape technical docs) are "slide-shaped" but text-dense
    enough to not be slide decks.
    """
    sig = _signals(
        avg_aspect_ratio=1.78,
        chars_per_page_avg=1000.0,  # above slide-deck threshold (800)
        has_headings=True,
    )
    result = _classify(sig)
    assert result.doc_type == "slide"
    assert result.confidence >= 0.5


def test_portrait_paper_fallback_uses_density() -> None:
    sig = _signals(
        avg_aspect_ratio=0.77,
        chars_per_page_avg=900.0,
        has_headings=True,
        has_tables=True,
    )
    result = _classify(sig)
    assert result.doc_type == "paper"
    assert result.confidence >= 0.5


def test_structure_bonus_boosts_confidence() -> None:
    """Two identical landscape docs in Tier 4 — one with markdown
    structure clues, one without — should diverge in confidence.

    Uses chars-per-page above the slide-deck threshold so both
    signals fall through to Tier 4 (where the structure bonus
    actually applies). Below the threshold both would hit Tier 0.5
    and return identical confidence.
    """
    base = _signals(avg_aspect_ratio=1.78, chars_per_page_avg=900.0)
    enriched = _signals(
        avg_aspect_ratio=1.78,
        chars_per_page_avg=900.0,
        has_headings=True,
        has_tables=True,
        has_lists=True,
        has_code_blocks=True,
    )
    base_result = _classify(base)
    enriched_result = _classify(enriched)
    assert enriched_result.confidence > base_result.confidence


def test_slide_deck_landscape_and_low_density_routes_to_docling() -> None:
    """16:9 aspect + slide-typical text density → Tier 0.5 → Docling.

    Preempts any Tier 1.A win that PowerPoint-produced decks would
    otherwise score, because PyMuPDF text extraction loses chart
    structure on slide-shaped content. Verified on the GTC 2024 CUDA
    deck — see docs/ROADMAP.md parser-investigation section.
    """
    sig = _signals(
        producer="Microsoft PowerPoint 2023",
        avg_aspect_ratio=1.78,  # 16:9
        chars_per_page_avg=400.0,
    )
    result = _classify(sig)
    assert result.doc_type == "slide-deck"
    assert result.confidence < 0.5  # below pymupdf_min_confidence default
    assert result.attribution["tier"] == "0.5-slide-deck"


def test_landscape_but_text_dense_stays_pymupdf() -> None:
    """Landscape + dense text → not a slide deck (e.g., wide-format
    legal doc, two-column landscape report). Aspect alone is not
    enough; density gate prevents false-positive routing.
    """
    sig = _signals(
        producer="LibreOffice",
        avg_aspect_ratio=1.5,
        chars_per_page_avg=1500.0,  # well above the 800 threshold
    )
    result = _classify(sig)
    assert result.doc_type != "slide-deck"


def test_portrait_low_density_stays_pymupdf() -> None:
    """Portrait + low text → not a slide deck (e.g., sparse cover sheet,
    image-heavy portrait report). Density alone is not enough; aspect
    gate prevents false-positive routing.
    """
    sig = _signals(
        producer="Microsoft Word",
        avg_aspect_ratio=0.77,  # standard letter portrait
        chars_per_page_avg=300.0,
    )
    result = _classify(sig)
    assert result.doc_type != "slide-deck"
