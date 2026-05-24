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

from memex.parse.docling_worker import _recover_heading_levels


class _FakeBBox:
    def __init__(self, height: float) -> None:
        self.height = height


class _FakeProv:
    def __init__(self, height: float) -> None:
        self.bbox = _FakeBBox(height)


class _FakeHeader:
    """A `SectionHeaderItem` stand-in: mutable `.level`, `prov` bboxes."""

    def __init__(self, *heights: float, level: int = 1, label: str = "section_header") -> None:
        self.label = label
        self.level = level
        self.prov = [_FakeProv(h) for h in heights]


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
