"""Unit tests for `core/text.py` helpers — the chart-block-aware
text-manipulation primitives shared between `index/` and `agents/`.

Pinned post-v7 verification audit (2026-05-23) since the helpers had
been exercised only transitively via the chunker. The truncated-block
defense (orphan opener / closer) is new behavior; the other helpers
were already in production.
"""

from __future__ import annotations

from memex.core.text import (
    chart_extracted_spans,
    is_inside_any_span,
    is_name_only_chunk,
    strip_chart_extracted_for_index,
)

# ----------------------------------------------------------------------
# chart_extracted_spans — happy path
# ----------------------------------------------------------------------


def test_chart_extracted_spans_balanced_block() -> None:
    """Single balanced `[chart-extracted]...[/chart-extracted]` block →
    one span covering the entire match."""
    text = "before [chart-extracted]inner[/chart-extracted] after"
    spans = chart_extracted_spans(text)
    assert len(spans) == 1
    start, end = spans[0]
    assert text[start:end] == "[chart-extracted]inner[/chart-extracted]"


def test_chart_extracted_spans_multiple_blocks() -> None:
    """Two balanced blocks → two spans, ordered by start offset."""
    text = "[chart-extracted]a[/chart-extracted] mid [chart-extracted]b[/chart-extracted]"
    spans = chart_extracted_spans(text)
    assert len(spans) == 2
    assert spans[0][0] < spans[1][0]


def test_chart_extracted_spans_no_blocks_returns_empty() -> None:
    """Text without chart-block tags returns empty list."""
    assert chart_extracted_spans("just prose, no chart blocks") == []
    assert chart_extracted_spans("") == []


# ----------------------------------------------------------------------
# chart_extracted_spans — truncation defense (post-v7 audit fix)
# ----------------------------------------------------------------------


def test_chart_extracted_spans_orphan_opener() -> None:
    """An opener with no matching closer (e.g. mid-chunk truncation,
    user-edited vault) extends to end-of-text. Without this defense,
    a `# H1` inside the orphan would silently split sections — exactly
    the regression v7 was built to fix.
    """
    text = "before [chart-extracted]inner content # H1 inside"
    spans = chart_extracted_spans(text)
    assert len(spans) == 1
    start, end = spans[0]
    # Span starts at the opener and extends to end-of-text
    assert text[start : start + len("[chart-extracted]")] == "[chart-extracted]"
    assert end == len(text)


def test_chart_extracted_spans_orphan_closer() -> None:
    """A closer with no matching opener extends from start-of-text to
    the closer position. Defensive — protects the H1 filter from inert
    chart labels above an orphan close tag."""
    text = "# H1 inside more content[/chart-extracted] after"
    spans = chart_extracted_spans(text)
    assert len(spans) == 1
    start, end = spans[0]
    assert start == 0
    assert text[end - len("[/chart-extracted]") : end] == "[/chart-extracted]"


def test_chart_extracted_spans_mixed_balanced_and_orphan() -> None:
    """One balanced block + one orphan opener → 2 spans (both H1-filter-
    protected). Order-stable."""
    text = "[chart-extracted]ok[/chart-extracted] middle [chart-extracted]orphan with # inside"
    spans = chart_extracted_spans(text)
    assert len(spans) == 2
    # First span is the balanced one
    assert text[spans[0][0] : spans[0][1]] == "[chart-extracted]ok[/chart-extracted]"
    # Second span is the orphan — extends to end-of-text
    assert spans[1][1] == len(text)


# ----------------------------------------------------------------------
# is_inside_any_span
# ----------------------------------------------------------------------


def test_is_inside_any_span_inside_a_span() -> None:
    spans = [(10, 20), (30, 40)]
    assert is_inside_any_span(15, spans) is True
    assert is_inside_any_span(35, spans) is True


def test_is_inside_any_span_outside_all_spans() -> None:
    spans = [(10, 20), (30, 40)]
    assert is_inside_any_span(5, spans) is False
    assert is_inside_any_span(25, spans) is False
    assert is_inside_any_span(50, spans) is False


def test_is_inside_any_span_boundary_is_exclusive_at_end() -> None:
    """`end` is exclusive: offset == end means OUTSIDE the span."""
    spans = [(10, 20)]
    assert is_inside_any_span(20, spans) is False
    # Start IS inclusive
    assert is_inside_any_span(10, spans) is True


def test_is_inside_any_span_empty_spans_returns_false() -> None:
    """No spans → any offset is "outside" all of them (no protection)."""
    assert is_inside_any_span(0, []) is False
    assert is_inside_any_span(100, []) is False


# ----------------------------------------------------------------------
# strip_chart_extracted_for_index (already tested indirectly; pin
# happy + edge here too)
# ----------------------------------------------------------------------


def test_strip_idempotent_on_no_chart_blocks() -> None:
    text = "plain prose with no chart blocks"
    assert strip_chart_extracted_for_index(text) == text


def test_strip_removes_balanced_block() -> None:
    text = "before [chart-extracted]middle[/chart-extracted] after"
    assert strip_chart_extracted_for_index(text) == "before  after"


def test_strip_does_not_remove_orphan_opener() -> None:
    """Defense: the strip uses the balanced-only regex, so an orphan
    opener falls through (no closer → not removed). The FTS layer's
    upsert sees the raw orphan-prefixed content; harmless because the
    text after the orphan opener is still just text."""
    text = "before [chart-extracted]orphan content"
    assert strip_chart_extracted_for_index(text) == text


# ----------------------------------------------------------------------
# is_name_only_chunk — the present-as-answer guard detector (ADR-0016 audit rec 1)
# ----------------------------------------------------------------------

_NAME_LIST_SLIDE = (
    "### Contrôle d'accès\n"
    "- Role-Based Access Control (RBAC)\n"
    "- Attribute-Based Access Control (ABAC)\n"
    "- Mandatory Access Control (MAC)\n"
    "- Discretionary Access Control (DAC)\n\n"
    "<!-- image: kind=icon -->"
)


def test_name_only_flags_the_name_list_slide() -> None:
    """The audit pathology: a heading + ≥2 bare short name bullets, no substantive sentence."""
    assert is_name_only_chunk(_NAME_LIST_SLIDE) is True


def test_name_only_keeps_prose_bullets() -> None:
    text = (
        "### OSPF Features\n"
        "- OSPF is a link-state routing protocol that was developed as an alternative for RIP.\n"
        "- OSPF offers faster convergence and scales to much larger network implementations."
    )
    assert is_name_only_chunk(text) is False


def test_name_only_keeps_plain_prose() -> None:
    text = (
        "This definition focuses on preventing unauthorized access to data and services "
        "coupled with making the access control enforcement as granular as possible."
    )
    assert is_name_only_chunk(text) is False


def test_name_only_short_circuits_on_chart_block() -> None:
    """A `[chart-extracted]` block is structured data — substantive support, never name-only."""
    text = "### Config\n[chart-extracted]\nR1(config)# ip domain-name example.com\n[/chart-extracted]"
    assert is_name_only_chunk(text) is False


def test_name_only_short_circuits_on_table_rows() -> None:
    text = "### T\n[table-rows]\ncol=Revenue value=215.9\n[/table-rows]"
    assert is_name_only_chunk(text) is False


def test_name_only_short_circuits_on_gfm_table() -> None:
    text = "| Metric | Value |\n|---|---|\n| Revenue | 215.9 |\n| Margin | 71.1 |"
    assert is_name_only_chunk(text) is False


def test_name_only_keeps_single_terse_sentence() -> None:
    """The FLOOR: a 1-line chunk is NOT confidently a name list — keep it (safe direction)."""
    assert is_name_only_chunk("OSPF is link-state.") is False


def test_name_only_keeps_short_fake_chunk_text() -> None:
    """The bridge/webui test fakes use this 4-word single-line chunk — the floor keeps it."""
    assert is_name_only_chunk("some grounded body text") is False


def test_name_only_keeps_heading_only_chunk() -> None:
    assert is_name_only_chunk("### OSPF Features and Characteristics") is False


def test_name_only_keeps_eight_word_bullet() -> None:
    """An ≥8-word bullet is a substantive sentence, not a bare name."""
    text = "### Access\n- Role-Based Access Control assigns permissions to enterprise roles directly"
    assert is_name_only_chunk(text) is False


def test_name_only_flags_two_short_bullets_with_heading() -> None:
    text = "### Models\n- Role-Based Access Control\n- Mandatory Access Control"
    assert is_name_only_chunk(text) is True


def test_name_only_empty_text_is_not_name_only() -> None:
    assert is_name_only_chunk("") is False
    assert is_name_only_chunk("\n\n   \n") is False
