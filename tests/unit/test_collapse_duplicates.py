"""audit-10 step 6a (W13) — consecutive exact-duplicate block collapse.

`collapse_consecutive_duplicates` drops a block that is an EXACT (whitespace-
normalized) re-emission of the immediately-preceding KEPT block — animation
slide-build re-emission + figure/page-seam double-transcription. RAW EQUALITY +
strict adjacency (window 1) is the only false-positive-free setting (a ratio
threshold collapses parallel data that shares a template). Excluded blocks
(image markers, PictureClassifier labels, box-art) are kept but don't count
toward adjacency; headings compare level-insensitively (keep the shallower).
"""

from __future__ import annotations

from memex.parse.pipeline import _finalize_body
from memex.parse.pipeline import collapse_consecutive_duplicates as cc


def test_exact_heading_reemission_collapses() -> None:
    # The CUDA-deck shape: an identical slide-title heading re-emitted (animation frame),
    # separated only by an image marker — still adjacent, so the 2nd collapses.
    md = (
        "###### But Transistors Are Getting Less Power-Efficient\n\n"
        "<!-- image -->\n\n"
        "###### But Transistors Are Getting Less Power-Efficient\n\n"
        "Body prose for the slide.\n"
    )
    out = cc(md)
    assert out.count("###### But Transistors Are Getting Less Power-Efficient") == 1
    assert "<!-- image -->" in out  # the marker is preserved (kept, not counted for adjacency)
    assert "Body prose for the slide." in out


def test_verbatim_prose_bullet_seam_collapses_keeps_one() -> None:
    md = "- Partager les vidéos avec une autre personne\n\n- Partager les vidéos avec une autre personne\n"
    out = cc(md)
    assert out.count("- Partager les vidéos avec une autre personne") == 1


def test_figure_retranscription_seam_collapses() -> None:
    # A VLM diagram transcription re-emitted verbatim across a page seam (one block each).
    block = "Internet then Router then Firewall then Private Network reached at the end."
    md = f"{block}\n\n{block}\n"
    assert cc(md).count(block) == 1


def test_more_than_two_run_collapses_to_single() -> None:
    # A 4-step animation: the same title 4×. Exactly one survives.
    md = "\n\n".join(["###### Conditional Nodes"] * 4) + "\n"
    assert cc(md).count("###### Conditional Nodes") == 1


def test_same_title_different_level_keeps_shallower() -> None:
    # `#### DHCP` then `###### DHCP` (same text, different level) → keep the SHALLOWER (####).
    md = "#### DHCP Protocol\n\n###### DHCP Protocol\n"
    out = cc(md).strip()
    assert out == "#### DHCP Protocol"
    # And the reverse order also keeps the shallower.
    md2 = "###### DHCP Protocol\n\n#### DHCP Protocol\n"
    assert cc(md2).strip() == "#### DHCP Protocol"


def test_boxart_connector_rows_preserved_NEGATIVE() -> None:
    # The guidelines flowchart: identical box-drawing connector rows each link a DIFFERENT node
    # pair — they are diagram units, never dup targets, so both survive.
    md = "[A]\n\n│                 │\n\n[B]\n\n│                 │\n\n[C]\n"
    out = cc(md)
    assert out.count("│                 │") == 2


def test_scattered_legitimate_repeats_untouched_NEGATIVE() -> None:
    # The 10-K shape: the SAME label repeated at DIFFERENT positions (different entities),
    # NOT consecutive → never collapsed.
    md = (
        "Vote required: a majority.\n\nProposal 1 detail.\n\n"
        "Vote required: a majority.\n\nProposal 2 detail.\n\n"
        "Vote required: a majority.\n\nProposal 3 detail.\n"
    )
    assert cc(md) == md  # byte-identical — every repeat is separated by distinct content


def test_distinct_code_bodies_under_same_heading_preserved_NEGATIVE() -> None:
    # `single_program.cu` re-emitted, but each time followed by a DIFFERENT code block. The
    # distinct code block between the two headings BREAKS adjacency (window 1), so NOTHING
    # collapses — both titles AND both code bodies survive. This is the conservative-by-design
    # behaviour: only headings separated by EXCLUDED noise (markers/labels) are adjacent. (A
    # ratio/section-level collapse that drops the 2nd title here is the deferred FP-risky lever.)
    md = (
        "###### single_program.cu\n\n```\nkernelA();\n```\n\n"
        "###### single_program.cu\n\n```\nkernelB();\n```\n"
    )
    out = cc(md)
    assert out == md  # byte-identical — distinct code breaks adjacency, nothing collapses
    assert "kernelA();" in out and "kernelB();" in out


def test_fenced_code_only_collapses_on_exact_match() -> None:
    same = "```\nx = compute()\n```"
    assert cc(f"{same}\n\n{same}\n").count("x = compute()") == 1  # adjacent identical fences → one
    diff = "```\nx = a()\n```\n\n```\nx = b()\n```\n"
    assert cc(diff) == diff  # differing fenced blocks both kept


def test_idempotent_and_noop_on_clean_doc() -> None:
    clean = "# Title\n\nUnique prose one.\n\n## Section\n\nUnique prose two.\n"
    assert cc(clean) == clean  # no consecutive dups → byte-identical
    dup = "###### T\n\n###### T\n\nbody\n"
    once = cc(dup)
    assert cc(once) == once  # idempotent


def test_image_marker_and_label_preserved_across_collapse() -> None:
    # Two identical titles separated by an image marker + a PictureClassifier label: the title
    # collapses (markers don't count for adjacency) but the marker + label are preserved.
    md = "###### Slide\n\n<!-- image -->\n\nLogo\n\n###### Slide\n\nReal content.\n"
    out = cc(md)
    assert out.count("###### Slide") == 1
    assert "<!-- image -->" in out and "Logo" in out and "Real content." in out


def test_wired_into_finalize_body() -> None:
    # End-to-end: _finalize_body composes the dedup with the other scrubbers; a duplicate slide
    # title collapses and a real data table is left intact.
    body = (
        "###### Slide\n\n###### Slide\n\nUnique body.\n\n"
        "| API | Application Programming Interface |\n|---|---|\n| BYOD | Bring Your Own Device |\n"
    )
    out = _finalize_body(body)
    assert out.count("Slide") == 1  # one survives (also re-levelled by the heading normalizer)
    assert "| API | Application Programming Interface |" in out  # real table preserved
