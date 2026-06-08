"""Unit tests for the Rust symbol splitter + injector (codebase-corpus arc, Phase 2).

`split_rust_symbols` finds top-level + impl/trait/mod-method boundaries via a masking lexer
(strings/comments/char-literals/raw-strings blanked so their braces + `fn`-like tokens can't
corrupt the scan) + brace-depth line scan. `inject_symbol_headings` prepends `## <symbol>` /
`### <Type::method>` ATX lines so the existing markdown chunker splits per symbol. Bounded
failure: a miss yields a coarser chunk, never dropped/wrong content.
"""

from __future__ import annotations

from memex.index.rust_symbols import (
    Symbol,
    inject_symbol_headings,
    split_rust_symbols,
)


def _labels(body: str) -> list[tuple[int, str]]:
    return [(s.level, s.label) for s in split_rust_symbols(body)]


def test_top_level_items_detected() -> None:
    body = (
        "use std::io;\n\n"
        "pub fn run() -> i32 { 1 }\n\n"
        "struct Foo { a: i32 }\n\n"
        "pub enum Bar { A, B }\n\n"
        "trait Baz { fn z(&self); }\n\n"
        "type Alias = u32;\n\n"
        "const MAX: u32 = 9;\n\n"
        "static NAME: &str = \"x\";\n\n"
        "macro_rules! mac { () => {}; }\n"
    )
    labels = _labels(body)
    assert (2, "fn run") in labels
    assert (2, "struct Foo") in labels
    assert (2, "enum Bar") in labels
    assert (2, "trait Baz") in labels
    assert (2, "type Alias") in labels
    assert (2, "const MAX") in labels
    assert (2, "static NAME") in labels
    assert (2, "macro_rules! mac") in labels


def test_visibility_and_qualifiers_on_fn() -> None:
    body = (
        "pub(crate) fn a() {}\n\n"
        "pub async fn b() {}\n\n"
        "const fn c() -> u8 { 0 }\n\n"
        "pub unsafe fn d() {}\n\n"
        'pub extern "C" fn e() {}\n'
    )
    labels = {label for _, label in _labels(body)}
    assert {"fn a", "fn b", "fn c", "fn d", "fn e"} <= labels


def test_impl_methods_are_nested_and_fully_qualified() -> None:
    body = (
        "pub struct Foo;\n\n"
        "impl Foo {\n"
        "    pub fn new() -> Self { Foo }\n"
        "    fn helper(&self) -> u32 { 1 }\n"
        "}\n"
    )
    labels = _labels(body)
    assert (2, "struct Foo") in labels
    assert (2, "impl Foo") in labels
    assert (3, "Foo::new") in labels
    assert (3, "Foo::helper") in labels


def test_impl_trait_for_type_qualifies_by_the_impl_target() -> None:
    body = (
        "impl From<Vec<u8>> for ResponseItem {\n"
        "    fn from(v: Vec<u8>) -> Self { todo!() }\n"
        "}\n\n"
        "impl std::fmt::Display for Payload {\n"
        "    fn fmt(&self, f: &mut Formatter) -> Result { todo!() }\n"
        "}\n"
    )
    labels = _labels(body)
    # The qualifying type is the type AFTER `for`, last path segment, generics stripped.
    assert (2, "impl From<Vec<u8>> for ResponseItem") in labels
    assert (3, "ResponseItem::from") in labels
    assert (2, "impl std::fmt::Display for Payload") in labels
    assert (3, "Payload::fmt") in labels


def test_cfg_test_mod_nests_its_test_functions() -> None:
    body = (
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    use super::*;\n\n"
        "    #[test]\n"
        "    fn it_works() { assert!(true); }\n"
        "}\n"
    )
    labels = _labels(body)
    assert (2, "mod tests") in labels
    assert (3, "tests::it_works") in labels


def test_preamble_backs_up_over_doc_comments_and_attributes() -> None:
    body = (
        "/// Doc line one.\n"
        "/// Doc line two.\n"
        "#[derive(Debug, Clone)]\n"
        "pub struct Widget {\n"
        "    a: i32,\n"
        "}\n"
    )
    syms = split_rust_symbols(body)
    [s] = [s for s in syms if s.label == "struct Widget"]
    # start_line is the FIRST doc-comment line (0-based 0), not the `struct` line (3).
    assert s.start_line == 0


def test_preamble_handles_multiline_attribute() -> None:
    body = (
        "#[derive(\n"
        "    Debug,\n"
        "    Clone,\n"
        ")]\n"
        "pub enum E { A }\n"
    )
    syms = split_rust_symbols(body)
    [s] = [s for s in syms if s.label == "enum E"]
    assert s.start_line == 0  # backed up over the whole multi-line #[derive(...)]


def test_brace_inside_string_does_not_corrupt_depth() -> None:
    """A `{}`-bearing format string must NOT shift brace depth — else the symbol AFTER it would
    be mis-nested. The masking lexer blanks the string content."""
    body = (
        "impl A {\n"
        '    fn f(&self) { let _ = format!("data:{};x={}", 1, 2); }\n'
        "}\n\n"
        "pub fn after() {}\n"
    )
    labels = _labels(body)
    assert (3, "A::f") in labels
    # `after` is top-level (depth 0) — would be depth>0 if the string braces had leaked.
    assert (2, "fn after") in labels


def test_fn_like_token_in_comment_is_not_a_symbol() -> None:
    body = "// fn ghost() {}\n/* fn also_ghost() {} */\npub fn real() {}\n"
    labels = {label for _, label in _labels(body)}
    assert "fn real" in labels
    assert "fn ghost" not in labels
    assert "fn also_ghost" not in labels


def test_char_literal_brace_is_masked() -> None:
    body = "pub fn open() -> char { '{' }\n\npub fn close() -> char { '}' }\n"
    labels = _labels(body)
    # Both are top-level; the char-literal braces must not unbalance depth.
    assert (2, "fn open") in labels
    assert (2, "fn close") in labels


def test_raw_string_with_hashes_and_braces_is_masked() -> None:
    body = (
        "pub fn a() {\n"
        '    let _ = r#"a { brace } and "quote""#;\n'
        "}\n\n"
        "pub fn b() {}\n"
    )
    labels = _labels(body)
    assert (2, "fn a") in labels
    assert (2, "fn b") in labels  # depth stayed correct through the raw string


def test_multiline_signature_with_where_clause() -> None:
    body = (
        "impl Payload {\n"
        "    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>\n"
        "    where\n"
        "        S: Serializer,\n"
        "    {\n"
        "        s.done()\n"
        "    }\n"
        "}\n"
    )
    labels = _labels(body)
    assert (3, "Payload::serialize") in labels


def test_lifetime_is_not_mistaken_for_a_char_literal() -> None:
    """A lifetime `'a` (no closing quote) must not start a char-literal mask that would swallow
    the following `{`. `fn g` after must still be detected at the right depth."""
    body = (
        "impl<'a> Holder<'a> {\n"
        "    fn get(&'a self) -> &'a str { self.0 }\n"
        "}\n\n"
        "pub fn standalone() {}\n"
    )
    labels = _labels(body)
    assert (3, "Holder::get") in labels
    assert (2, "fn standalone") in labels


def test_unbalanced_braces_degrade_gracefully_no_exception() -> None:
    body = "pub fn broken() {\n    let x = {{{ ;\n"  # intentionally unbalanced
    # Must not raise; returns whatever it found (bounded failure → coarser chunk).
    syms = split_rust_symbols(body)
    assert isinstance(syms, list)


def test_injection_prepends_headings_before_symbols() -> None:
    body = "pub fn one() {}\n\npub fn two() {}\n"
    syms = split_rust_symbols(body)
    out = inject_symbol_headings(body, syms)
    lines = out.split("\n")
    assert "## fn one" in lines
    assert "## fn two" in lines
    # The original code lines survive verbatim (the heading is ADDED above them).
    assert "pub fn one() {}" in lines
    assert lines.index("## fn one") < lines.index("pub fn one() {}")


def test_injection_no_symbols_returns_body_unchanged() -> None:
    body = "// just a comment\nlet x = 1;\n"
    assert inject_symbol_headings(body, []) == body
    assert inject_symbol_headings(body, split_rust_symbols(body)) == body


def test_injected_heading_levels_drive_nested_heading_path() -> None:
    """The injected `##`/`###` levels are what the chunker turns into heading_path — pin that a
    method gets level 3 under its impl's level 2 (so heading_path = [impl Foo, Foo::m])."""
    body = "impl Foo {\n    fn m(&self) {}\n}\n"
    syms = split_rust_symbols(body)
    impl_sym = next(s for s in syms if s.label == "impl Foo")
    method_sym = next(s for s in syms if s.label == "Foo::m")
    assert impl_sym.level == 2
    assert method_sym.level == 3
    out = inject_symbol_headings(body, syms)
    assert "## impl Foo" in out
    assert "### Foo::m" in out


def test_frozen_symbol_dataclass() -> None:
    s = Symbol(start_line=3, label="fn x", level=2)
    assert (s.start_line, s.label, s.level) == (3, "fn x", 2)
