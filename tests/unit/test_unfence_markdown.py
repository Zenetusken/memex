"""audit-10 step 6b (W14) — un-fence mid-document ```markdown / ```md blocks.

`unfence_markdown_blocks` drops the fences around a block the parser (or the VLM,
beyond W5's whole-page wrapper) wrongly wrapped in a ```markdown / ```md code
fence, so the trapped headings + prose become top-level. A markdown/md language
tag is by construction a prose signal (false-positive-free); a block containing
a NESTED fence is left alone (un-fencing would expose an unbalanced inner code
fence); a real-code fence (any other / no language tag) is never considered. The
riskier bare-tagged-heading and pull-quote→blockquote un-fences are DEFERRED
(8/10 real-code-corruption risk in the FP analysis).
"""

from __future__ import annotations

from memex.parse.pipeline import _finalize_body
from memex.parse.pipeline import unfence_markdown_blocks as uf

FENCE = "```"


def test_unfences_clean_markdown_block() -> None:
    md = f"{FENCE}markdown\n# Our Lead Director\n\n## Stephen C. Neal\n\nDirector since 2019.\n{FENCE}\n"
    out = uf(md)
    assert FENCE not in out  # both fences gone
    assert "# Our Lead Director" in out
    assert "## Stephen C. Neal" in out
    assert "Director since 2019." in out


def test_md_variant_tag() -> None:
    assert uf(f"{FENCE}md\n# Heading\n\nprose\n{FENCE}\n") == "# Heading\n\nprose\n"


def test_abstains_on_nested_fence_keeps_block() -> None:
    # The LLDP-class case: a ```markdown wrapper that EMBEDS a real ```text CLI block. Un-fencing
    # would splice the inner ```text out unbalanced — so leave the whole thing fenced.
    md = f"{FENCE}markdown\n# Device Discovery\n\n{FENCE}text\nS1# show lldp neighbors\n{FENCE}\n\nmore\n{FENCE}\n"
    assert uf(md) == md  # byte-identical — abstained


def test_leaves_language_tagged_code_fenced_NEGATIVE() -> None:
    # A real code block (language-tagged) is never a W14 target.
    py = f"{FENCE}python\n# src/memex/core/events.py\nfrom pydantic import BaseModel\n{FENCE}\n"
    assert uf(py) == py
    cli = f"{FENCE}text\nR1# show ip route\nR1# show interfaces\n{FENCE}\n"
    assert uf(cli) == cli


def test_leaves_bare_fence_block_alone_NEGATIVE() -> None:
    # A bare-tagged (no language) fence — even one containing a heading — is DEFERRED (8/10 risk),
    # so the conservative markdown-only un-fence leaves it untouched.
    md = f"{FENCE}\n# Troubleshooting\nThe IEEE 802.3 standard...\n{FENCE}\n"
    assert uf(md) == md


def test_abstains_when_no_closing_fence() -> None:
    md = f"{FENCE}markdown\n# Heading\n\nbody with no closing fence\n"
    assert uf(md) == md  # no bare close → conservative no-op


def test_idempotent_and_noop() -> None:
    md = f"{FENCE}markdown\n# H\n\nprose\n{FENCE}\n\n{FENCE}markdown\n## H2\n\nmore\n{FENCE}\n"
    once = uf(md)
    assert uf(once) == once  # idempotent (no outer fence left)
    clean = "# Title\n\nJust prose, no fences.\n"
    assert uf(clean) == clean  # fast-path no-op


def test_multiple_blocks_and_real_code_between() -> None:
    # Two markdown mis-fences with a REAL code block between → both un-fence, the code stays.
    md = (
        f"{FENCE}markdown\n# A\n{FENCE}\n\n"
        f"{FENCE}python\nx = 1\n{FENCE}\n\n"
        f"{FENCE}markdown\n# B\n{FENCE}\n"
    )
    out = uf(md)
    assert out.count("# A") == 1 and out.count("# B") == 1
    assert f"{FENCE}python\nx = 1\n{FENCE}" in out  # real code untouched
    assert "```markdown" not in out


def test_wired_into_finalize_body_headings_become_visible() -> None:
    # End-to-end: a markdown-fenced region's headings are un-fenced FIRST, so the heading
    # normalizer (fence-aware) then re-levels them by their section number.
    body = f"{FENCE}markdown\n## 1 Introduction\n\n## 1.1 History\n{FENCE}\n"
    out = _finalize_body(body)
    assert "```" not in out  # un-fenced
    assert "## 1 Introduction" in out  # 1 -> H2 (kept)
    assert "### 1.1 History" in out  # 1.1 -> H3 (the normalizer now SEES the recovered heading)
