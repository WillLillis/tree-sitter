use std::path::{Path, PathBuf};

use crate::grammars::InputGrammar;
use crate::nativedsl::lexer::TokenKind;
use crate::rules::{Precedence, Rule};

use super::{
    DisallowedItemKind, DslError, InnerTy, LexErrorKind, LowerErrorKind, NoteMessage,
    ParseErrorKind, Ty, TypeErrorKind, parse_native_dsl,
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

macro_rules! assert_err {
    ($err:expr, $variant:ident) => {
        match $err {
            DslError::$variant(e) => e,
            other => panic!("expected {} error, got {other:?}", stringify!($variant)),
        }
    };
}

mod bindings;
mod combinators;
mod config;
mod errors;
mod imports;
mod inheritance;
mod types;

/// Directory containing DSL-specific helper fixtures (inherit bases, import helpers).
pub(super) fn test_fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/nativedsl/test_fixtures")
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
