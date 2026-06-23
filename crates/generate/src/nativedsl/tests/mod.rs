use std::path::{Path, PathBuf};

use crate::grammars::InputGrammar;
use crate::nativedsl::lexer::TokenKind;
use crate::rules::{Precedence, Rule};

use super::{
    Constraint, ContainerKind, DataTy, DisallowedItemKind, DslError, ElemTy, ExpandErrorKind,
    InnerTy, LexErrorKind, LowerErrorKind, NativeDslError, NoteMessage, ParseErrorKind,
    ResolveErrorKind, ScalarTy, TupleSig, Ty, TypeErrorKind, parse_native_dsl,
};

/// Generate tests that parse a grammar and assert on the first variable's rule.
macro_rules! rule_tests {
    ($($name:ident { $input:expr, $expected:expr })*) => {
        $(#[test] fn $name() {
            let g = dsl($input);
            assert_eq!(g.variables[0].rule, $expected);
        })*
    };
}

/// Generate tests that just verify a grammar compiles without error.
macro_rules! compile_tests {
    ($($name:ident { $input:expr })*) => {
        $(#[test] fn $name() { dsl($input); })*
    };
}

macro_rules! assert_err {
    ($err:expr, $variant:ident) => {
        match $err {
            DslError::$variant(e) => e,
            other => panic!("expected {} error, got {other:?}", stringify!($variant)),
        }
    };
}

mod bindings;
mod cfg;
mod combinators;
mod config;
mod errors;
mod imports;
mod inheritance;
mod rule_set_macros;
mod types;

/// Directory containing DSL-specific helper fixtures (inherit bases, import helpers).
pub(super) fn test_fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/nativedsl/tests/fixtures")
}

/// Convert a path to a forward-slash string for embedding in DSL source.
/// Ensures import/inherit paths work on Windows where `display()` produces `\`.
pub(super) fn dsl_path(p: &std::path::Path) -> String {
    p.to_str().unwrap().replace('\\', "/")
}

pub(super) fn dsl(input: &str) -> InputGrammar {
    parse_native_dsl(input, &test_fixtures_dir().join("grammar.tsg")).unwrap()
}

pub(super) fn dsl_err(input: &str) -> DslError {
    parse_native_dsl(input, &test_fixtures_dir().join("grammar.tsg")).unwrap_err()
}

/// Raw pipeline result, for tests that assert only success-or-failure without
/// committing to a grammar or a specific error kind.
pub(super) fn try_dsl(input: &str) -> Result<InputGrammar, DslError> {
    parse_native_dsl(input, &test_fixtures_dir().join("grammar.tsg"))
}

/// Write `modules` (filename -> source) into a fresh tempdir and parse `root`
/// against it, so imports/inherits resolve by relative filename - e.g.
/// `import("helper.tsg")`. The tempdir drops after parsing, which has already
/// read every file. Returns the raw result so callers `.unwrap()` for the
/// grammar or `.unwrap_err()` for the error.
pub(super) fn parse_with_modules(
    modules: &[(&str, &str)],
    root: &str,
) -> Result<InputGrammar, DslError> {
    let dir = tempfile::tempdir().unwrap();
    for (name, src) in modules {
        std::fs::write(dir.path().join(name), src).unwrap();
    }
    // The root path must exist on disk - it's canonicalized to seed cycle
    // detection - even though parsing reads `root` directly.
    let root_path = dir.path().join("grammar.tsg");
    std::fs::write(&root_path, root).unwrap();
    parse_native_dsl(root, &root_path)
}

/// Build the Rule tree for `comma_sep1(item)`:
/// `seq(item, choice(repeat(seq(",", item)), blank))`
pub(super) fn comma_sep1_rule(item: &str) -> Rule {
    sep_by1_rule(",", item)
}

/// Build the Rule tree for `comma_sep(item)`:
/// `choice(comma_sep1(item), blank)`
pub(super) fn comma_sep_rule(item: &str) -> Rule {
    Rule::choice(vec![comma_sep1_rule(item), Rule::Blank])
}

/// Build the Rule tree for `sep_by1(sep, item)`:
/// `seq(item, choice(repeat(seq(sep, item)), blank))`
pub(super) fn sep_by1_rule(sep: &str, item: &str) -> Rule {
    Rule::seq(vec![
        Rule::NamedSymbol(item.into()),
        Rule::choice(vec![
            Rule::repeat(Rule::seq(vec![
                Rule::String(sep.into()),
                Rule::NamedSymbol(item.into()),
            ])),
            Rule::Blank,
        ]),
    ])
}

/// Parse a grammar that inherits from a tempfile base, returning the error.
/// `base_content` is written to `base.tsg`, and the parent grammar is:
/// `let base = inherit("base.tsg") grammar { language: "derived", inherits: base } rule extra { "hello" }`
pub(super) fn inherit_err(base_content: &str) -> (DslError, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.tsg");
    std::fs::write(&base_path, base_content).unwrap();
    let parent_path = dir.path().join("parent.tsg");
    let parent_src = "let base = inherit(\"base.tsg\")\n\
                      grammar { language: \"derived\", inherits: base }\n\
                      rule extra { \"hello\" }\n";
    std::fs::write(&parent_path, parent_src).unwrap();
    let err = parse_native_dsl(parent_src, &parent_path).unwrap_err();
    (err, base_path)
}

pub(super) fn rule_names(g: &InputGrammar) -> Vec<&str> {
    g.variables.iter().map(|v| v.name.as_str()).collect()
}

pub(super) fn find_rule<'a>(g: &'a InputGrammar, name: &str) -> &'a Rule {
    &g.variables
        .iter()
        .find(|v| v.name == name)
        .unwrap_or_else(|| panic!("rule '{name}' not found"))
        .rule
}
