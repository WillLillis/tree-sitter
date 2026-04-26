use std::path::Path;

use crate::grammars::{InputGrammar, VariableType};
use crate::nativedsl::lexer::TokenKind;
use crate::rules::{Precedence, Rule};

use super::{
    DisallowedItemKind, DslError, InnerTy, LexErrorKind, LowerErrorKind, NoteMessage,
    ParseErrorKind, Ty, TypeErrorKind, parse_native_dsl,
};

macro_rules! assert_err {
    ($err:expr, $variant:ident) => {
        match $err {
            DslError::$variant(e) => e,
            other => panic!("expected {} error, got {other:?}", stringify!($variant)),
        }
    };
}

mod ast_tests;
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
