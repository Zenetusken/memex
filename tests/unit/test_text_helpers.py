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
    claim_grounded_only_by_name,
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
    text = (
        "### Config\n[chart-extracted]\nR1(config)# ip domain-name example.com\n[/chart-extracted]"
    )
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
    text = (
        "### Access\n- Role-Based Access Control assigns permissions to enterprise roles directly"
    )
    assert is_name_only_chunk(text) is False


def test_name_only_flags_two_short_bullets_with_heading() -> None:
    text = "### Models\n- Role-Based Access Control\n- Mandatory Access Control"
    assert is_name_only_chunk(text) is True


def test_name_only_empty_text_is_not_name_only() -> None:
    assert is_name_only_chunk("") is False
    assert is_name_only_chunk("\n\n   \n") is False


# ----------------------------------------------------------------------
# is_name_only_chunk — leading-enumerator harden (the numbered sub-heading gap)
# ----------------------------------------------------------------------


def test_name_only_flags_numbered_subheadings() -> None:
    """The closed gap: a slide of numbered SUB-HEADINGS — each exactly 8 tokens ONLY because the
    `N.` enumerator counts — is now name-only (the enumerator is stripped before the word count)."""
    text = (
        "### Sécurité du réseau\n"
        "3. Protection du plan de contrôle (Control Plane)\n"
        "4. Protection du plan de données (Data Plane)"
    )
    assert is_name_only_chunk(text) is True


def test_name_only_enumerator_strip_keeps_long_numbered_prose() -> None:
    """FP guard: a numbered line that is still ≥8 CONTENT words after the strip stays substantive."""
    text = (
        "### Plan\n"
        "1. The control plane processes all administrative requests and stays fully isolated\n"
        "2. Segmentation"
    )
    assert is_name_only_chunk(text) is False


def test_name_only_flags_paren_enumerator_form() -> None:
    """The `N)` enumerator form is stripped too (regex alternation `[.)]`)."""
    text = "### Étapes\n1) Configuration du plan de contrôle local\n2) Vérification du plan"
    assert is_name_only_chunk(text) is True


def test_name_only_numbered_floor_keeps_single_short_line() -> None:
    """The ≥2-short-line floor still holds: one numbered heading + a real heading is NOT name-only."""
    text = "### Sécurité\n3. Protection du plan de contrôle (Control Plane)"
    assert is_name_only_chunk(text) is False


# ----------------------------------------------------------------------
# claim_grounded_only_by_name — the shared bridge/verify demotion rule
# ----------------------------------------------------------------------

_NAME_LIST = "### Contrôle d'accès\n- Role-Based Access Control (RBAC)\n- Attribute-Based (ABAC)"
_PROSE = (
    "### OSPF\n- OSPF assigns a cost to each link and floods link-state advertisements to peers."
)


def test_grounded_only_by_name_behavioral_on_name_list_is_true() -> None:
    assert claim_grounded_only_by_name("RBAC assigns permissions by job role.", _NAME_LIST) is True


def test_grounded_only_by_name_membership_on_name_list_is_false() -> None:
    """Membership-first KEEP: a name-list DOES ground a membership claim."""
    assert (
        claim_grounded_only_by_name("RBAC is one of the listed access-control models.", _NAME_LIST)
        is False
    )


def test_grounded_only_by_name_behavioral_on_prose_is_false() -> None:
    """A substantive prose chunk is not name-only → keep (short-circuits at the chunk test)."""
    assert claim_grounded_only_by_name("OSPF assigns a cost to each link.", _PROSE) is False


def test_grounded_only_by_name_fr_chunk_en_claim_trap() -> None:
    """The common shape: a FR name-list chunk + an EN claim. Behavioral demotes, membership keeps."""
    assert claim_grounded_only_by_name("ABAC evaluates attributes dynamically.", _NAME_LIST) is True
    assert (
        claim_grounded_only_by_name("ABAC is included in the access-control list.", _NAME_LIST)
        is False
    )


# ---- relevance world-knowledge comparison detector (audit-15 M3) ----


from memex.core.text import (  # noqa: E402 — section-local import, file convention
    relevance_reason_cites_world_knowledge as _wk,
)


def test_wk_matches_the_observed_handwritten_case() -> None:
    r = ("The answer incorrectly lists 'Compile' and 'Machine Code' as stages instead of the "
         "standard three stages (Preprocessing, Compilation, Assembly, Linking) or the "
         "two-stage model (Compilation, Linking) described in C++ documentation")
    assert _wk(r) is True


def test_wk_matches_textbook_and_widely_accepted() -> None:
    assert _wk("This contradicts the textbook account of TCP handshakes.") is True
    assert _wk("The widely-accepted model has four phases, not three.") is True
    assert _wk("The answer omits the conventional linking stage.") is True
    assert _wk("These are not the correct stages of compilation.") is True


def test_wk_does_not_match_topic_mismatch_reasons() -> None:
    # Legit non-responsive reasons must NOT be overridden — including ones where
    # "standard" is the TOPIC (a document/spec name), not an external-knowledge comparison.
    assert _wk("The question asks about kernels but the answer describes libraries.") is False
    assert _wk("The answer is about X's ecosystem, not X itself.") is False
    assert _wk("The answer fails to define the Pareto chart as requested.") is False
    assert _wk("The question asks which port is the trunk; the answer discusses VLAN ranges.") is False


def test_wk_nist_standard_as_topic_is_borderline_accepted_tradeoff() -> None:
    # DOCUMENTED TRADEOFF: "the NIST standard" as a topic noun DOES match the regex; the
    # override would ship a GROUNDED answer despite a topic-mismatch vote. Advisory-gate
    # worst case (a grounded slightly-off-topic answer), accepted and pinned here.
    assert _wk("The answer describes the NIST standard rather than the asked protocol.") is True


# ---- denial-framed summary detector (audit-15 M2) ----


from memex.core.text import (  # noqa: E402 — section-local import, file convention
    is_denial_framed_summary as _denial,
)


def test_denial_matches_the_three_observed_drafts() -> None:
    assert _denial("The chunks do not state which specific GPUs were used to train GTE, "
                   "only that training was conducted on up to 8 NVIDIA A100 GPUs") is True
    assert _denial("The chunks do not state which base language model GTE-large is initialized "
                   "from. They mention that GTEbase and GTElarge are initialized") is True
    assert _denial("The chunks do not contain a literal definition of what a Pareto chart is. "
                   "They mention 'Pareto analysis' in the context of bar charts") is True


def test_denial_does_not_match_true_refusals() -> None:
    assert _denial("No literal answer in chunks.") is False
    assert _denial("The chunks do not contain any information about quantum schedulers.") is False
    assert _denial("") is False


def test_denial_does_not_match_affirmative_summaries() -> None:
    assert _denial("GTE was trained on up to 8 NVIDIA A100 GPUs.") is False
    assert _denial("The stages are source code, compile, and machine code.") is False


# ---- summary-scope guard v2 detector (audit-17; v1 reverted for -107 over-refusal) ----


from memex.core.text import summary_subject_unsupported as _scope2

AR12_CHUNK = "Gross margin was 71.1% for fiscal year 2026, up from 69.8%. Revenue grew across segments."
TG13_CHUNK = ("## const _DEFAULT_MAX_LEN\n/// Maximum characters forwarded to the TUI. "
              "const _DEFAULT_MAX_LEN: usize = 120;")


def test_scope2_fires_on_ar12_with_evidence() -> None:
    # The subject 'Graphics segment' is absent from claim AND chunk -> fire.
    assert _scope2(
        "What was the gross margin of NVIDIA's Graphics segment in fiscal 2026?",
        "The gross margin for NVIDIA's Graphics segment in fiscal 2026 was 71.1%.",
        ["Gross margin was 71.1% in fiscal year 2026"],
        [AR12_CHUNK],
    ) is True


def test_scope2_fires_on_tg13_with_evidence() -> None:
    assert _scope2(
        "According to the developer guidelines, what is the exact maximum line length in characters?",
        "The developer guidelines specify a maximum line length of 120 characters.",
        ["The maximum line length enforced is 120 characters."],
        [TG13_CHUNK],
    ) is True


def test_scope2_paraphrase_with_subject_in_evidence_does_not_fire() -> None:
    # THE v1 KILLER: the claim paraphrases, but the EVIDENCE carries the subject -> no fire.
    assert _scope2(
        "What are the stages of the C++ compilation process shown in the notes?",
        "The compilation process has three stages: source code, compile, machine code.",
        ["The stages are source code, compile, and machine code."],
        ["My notes on the C++ compilation process: source code -> compile -> machine code."],
    ) is False


def test_scope2_subject_in_claim_does_not_fire() -> None:
    assert _scope2(
        "What was the gross margin of the Graphics segment?",
        "The Graphics segment gross margin was 65%.",
        ["The Graphics segment gross margin was 65%."],
        ["irrelevant chunk"],
    ) is False


def test_scope2_no_claims_inert() -> None:
    assert _scope2("q?", "s", [], ["chunk"]) is False
