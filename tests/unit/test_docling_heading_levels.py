"""Unit tests for Docling heading-level recovery in the worker.

Docling's PDF reading-order model leaves every `SectionHeaderItem` at
`level=1`, so `export_to_markdown` collapses all headings to `## `.
`_recover_heading_levels` re-derives the level from provenance bbox
height — ranking headers among themselves, writing the level back in
place, touching only section-headers, never adding/removing items.

The fakes are duck-typed so these run without docling installed (the
worker imports docling lazily, inside `_convert_to_payload`).
"""

from __future__ import annotations

from memex.parse.docling_worker import (
    _demote_misdetected_headers,
    _demote_prose_headings,
    _looks_like_prose_heading,
    _recover_heading_levels,
)


class _FakeBBox:
    def __init__(self, height: float) -> None:
        self.height = height


class _FakeProv:
    def __init__(self, height: float) -> None:
        self.bbox = _FakeBBox(height)


class _FakeHeader:
    """A `SectionHeaderItem` stand-in: mutable `.level`, `.text`, `prov` bboxes."""

    def __init__(
        self, *heights: float, level: int = 1, label: str = "section_header", text: str = ""
    ) -> None:
        self.label = label
        self.level = level
        self.text = text
        self.prov = [_FakeProv(h) for h in heights]


class _FakeTextItem:
    """Stand-in for docling's `TextItem` — the reclassification target."""

    label = "text"


class _FakeText:
    """A non-header text item — has no meaningful height, never touched."""

    def __init__(self) -> None:
        self.label = "text"
        self.prov: list[_FakeProv] = []


class _FakeTitle:
    """A `TitleItem` stand-in: `label='title'` and deliberately NO `.level`."""

    def __init__(self, height: float) -> None:
        self.label = "title"
        self.prov = [_FakeProv(height)]


class _FakeDoc:
    def __init__(self, texts: list[object]) -> None:
        self.texts = texts


def test_distinct_heights_rank_descending() -> None:
    h1, h2, h3 = _FakeHeader(24.0), _FakeHeader(18.0), _FakeHeader(14.0)
    n = _recover_heading_levels(_FakeDoc([h1, h2, h3]))
    assert n == 3
    assert (h1.level, h2.level, h3.level) == (1, 2, 3)


def test_uniform_heights_all_level_1() -> None:
    # Slide-deck case: uniform title heights stay peers, no spurious tiers.
    hs = [_FakeHeader(18.0) for _ in range(4)]
    n = _recover_heading_levels(_FakeDoc(list(hs)))
    assert n == 4
    assert all(h.level == 1 for h in hs)


def test_caps_at_level_5() -> None:
    hs = [_FakeHeader(float(s)) for s in (30, 26, 22, 18, 14, 10)]
    _recover_heading_levels(_FakeDoc(list(hs)))
    assert [h.level for h in hs] == [1, 2, 3, 4, 5, 5]  # 6th tier capped


def test_header_without_prov_is_skipped() -> None:
    bare = _FakeHeader(level=1)  # no prov → left at default, not counted
    big = _FakeHeader(24.0)
    n = _recover_heading_levels(_FakeDoc([bare, big]))
    assert n == 1
    assert bare.level == 1
    assert big.level == 1  # only one tier among the headers WITH a bbox


def test_non_headers_ignored() -> None:
    title = _FakeTitle(40.0)  # bigger than any header, but no .level
    text = _FakeText()
    header = _FakeHeader(18.0)
    n = _recover_heading_levels(_FakeDoc([title, text, header]))
    assert n == 1
    assert not hasattr(title, "level")
    assert header.level == 1


def test_tolerance_bucketing_groups_near_heights() -> None:
    big = _FakeHeader(24.0)
    a, b = _FakeHeader(11.98), _FakeHeader(12.03)  # same 12.0 bucket
    _recover_heading_levels(_FakeDoc([big, a, b]))
    assert big.level == 1
    assert a.level == b.level == 2  # one shared tier, not two


def test_multi_prov_uses_max_height() -> None:
    wrapped = _FakeHeader(12.0, 24.0)  # max(prov) = 24 → top tier
    other = _FakeHeader(18.0)
    _recover_heading_levels(_FakeDoc([wrapped, other]))
    assert wrapped.level == 1
    assert other.level == 2


def test_multiline_over_rank_is_a_known_limitation() -> None:
    # A heading wrapping two lines reports a ~2x bbox and over-ranks as a
    # bigger heading. Pinned so the behavior can't change silently.
    single = _FakeHeader(12.0)
    two_line = _FakeHeader(24.0)  # same font, wrapped → taller bbox
    _recover_heading_levels(_FakeDoc([single, two_line]))
    assert two_line.level == 1
    assert single.level == 2


def test_empty_doc_is_noop() -> None:
    assert _recover_heading_levels(_FakeDoc([])) == 0
    assert _recover_heading_levels(_FakeDoc([_FakeText()])) == 0


# ----- Prose-heading demotion -----


def test_looks_like_prose_sentence_with_period() -> None:
    assert _looks_like_prose_heading("Data centers are becoming AI factories.")
    assert _looks_like_prose_heading("AI now perceives, reasons, plans, and acts.")


def test_looks_like_prose_long_without_punct() -> None:
    long = "this heading just keeps going on and on well past any real title length you would ever expect"
    assert _looks_like_prose_heading(long)  # >15 words


def test_real_short_headings_are_not_prose() -> None:
    for h in ("AI Is a Five-Layer Cake", "A Global AI Ecosystem", "Forward-Looking Statements"):
        assert not _looks_like_prose_heading(h)


def test_short_label_with_period_is_not_prose() -> None:
    # "Item 1." / "Note 5." legitimately end in a period — guarded by min-words.
    assert not _looks_like_prose_heading("Item 1.")
    assert not _looks_like_prose_heading("Note 5.")


def test_heading_with_colon_or_comma_kept() -> None:
    assert not _looks_like_prose_heading("Fiscal 2026: A Defining Year")
    assert not _looks_like_prose_heading("Dear NVIDIANs and Stakeholders,")


def test_demote_strips_prefix_from_prose() -> None:
    md = "## AI Is a Five-Layer Cake\n\n###### Data centers are becoming AI factories.\n"
    out, n = _demote_prose_headings(md)
    assert n == 1
    assert "## AI Is a Five-Layer Cake" in out  # real heading kept
    assert "Data centers are becoming AI factories." in out
    assert "###### Data centers" not in out  # prefix gone


def test_demote_skips_code_fences() -> None:
    md = "## Real Heading\n\n```\n# A full sentence inside code that ends.\n```\n"
    out, n = _demote_prose_headings(md)
    assert n == 0
    assert "# A full sentence inside code that ends." in out  # untouched in fence


def test_demote_preserves_emphasis_markers_in_output() -> None:
    md = "###### **The buildout has only just begun in earnest.**\n"
    out, n = _demote_prose_headings(md)
    assert n == 1
    # text (incl. bold markers) preserved; only the `#` prefix removed
    assert out.strip() == "**The buildout has only just begun in earnest.**"


def test_demote_noop_on_clean_headings() -> None:
    md = "# Title\n\n## Section\n\n### Subsection\n"
    out, n = _demote_prose_headings(md)
    assert n == 0
    assert out == md


# ----- Root-level reclassification (mis-detection fix) -----


def test_reclassify_demotes_prose_header() -> None:
    real = _FakeHeader(18.0, text="AI Is a Five-Layer Cake")
    prose = _FakeHeader(12.0, text="Data centers are becoming AI factories.")
    n = _demote_misdetected_headers(_FakeDoc([real, prose]), text_item_cls=_FakeTextItem)
    assert n == 1
    assert isinstance(prose, _FakeTextItem)  # reclassified → serialises as paragraph
    assert type(real).__name__ == "_FakeHeader"  # real heading untouched


def test_reclassify_keeps_real_headers() -> None:
    hs = [
        _FakeHeader(18.0, text=t)
        for t in ("Bar chart", "A Global AI Ecosystem", "Forward-Looking Statements")
    ]
    n = _demote_misdetected_headers(_FakeDoc(list(hs)), text_item_cls=_FakeTextItem)
    assert n == 0
    assert all(type(h).__name__ == "_FakeHeader" for h in hs)


def test_reclassify_skips_titles_and_non_headers() -> None:
    title = _FakeTitle(40.0)  # no `.level` → not a section header
    text = _FakeText()  # label 'text'
    prose_hdr = _FakeHeader(12.0, text="This sentence was wrongly tagged as a heading.")
    n = _demote_misdetected_headers(_FakeDoc([title, text, prose_hdr]), text_item_cls=_FakeTextItem)
    assert n == 1
    assert isinstance(prose_hdr, _FakeTextItem)
    assert type(title).__name__ == "_FakeTitle"


def test_reclassify_long_prose_without_punct() -> None:
    long = _FakeHeader(
        12.0,
        text="this heading just keeps going on and on well past any real title length you would expect",
    )
    n = _demote_misdetected_headers(_FakeDoc([long]), text_item_cls=_FakeTextItem)
    assert n == 1
    assert isinstance(long, _FakeTextItem)


def test_reclassify_empty_doc() -> None:
    assert _demote_misdetected_headers(_FakeDoc([]), text_item_cls=_FakeTextItem) == 0
