use std::path::{Path, PathBuf};

use crate::grammars::InputGrammar;
use crate::nativedsl::lexer::TokenKind;
use crate::rules::{Associativity, Rule as RuleNode, RuleId};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) enum Precedence {
    #[default]
    None,
    Integer(i32),
    Name(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Alias {
    value: String,
    is_named: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct MetadataParams {
    precedence: Precedence,
    dynamic_precedence: i32,
    associativity: Option<Associativity>,
    is_token: bool,
    is_main_token: bool,
    alias: Option<Alias>,
    field_name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum RuleExpectation {
    Blank,
    String(String),
    Pattern(String, String),
    NamedSymbol(String),
    Choice(Vec<Self>),
    Metadata {
        params: MetadataParams,
        rule: Box<Self>,
    },
    Repeat(Box<Self>),
    Seq(Vec<Self>),
    Reserved {
        rule: Box<Self>,
        context_name: String,
    },
}

// Keep expected-rule expressions compact throughout the test suite.
pub(super) type Rule = RuleExpectation;

impl RuleExpectation {
    fn with_metadata(mut content: Self, update: impl FnOnce(&mut MetadataParams)) -> Self {
        if let Self::Metadata { params, .. } = &mut content
            && !params.is_token
        {
            update(params);
            content
        } else {
            let mut params = MetadataParams::default();
            update(&mut params);
            Self::Metadata {
                params,
                rule: Box::new(content),
            }
        }
    }
    pub(super) fn field(name: String, content: Self) -> Self {
        Self::with_metadata(content, |p| p.field_name = Some(name))
    }
    pub(super) fn alias(content: Self, value: String, is_named: bool) -> Self {
        Self::with_metadata(content, |p| p.alias = Some(Alias { value, is_named }))
    }
    pub(super) fn token(content: Self) -> Self {
        Self::with_metadata(content, |p| p.is_token = true)
    }
    pub(super) fn immediate_token(content: Self) -> Self {
        Self::with_metadata(content, |p| {
            p.is_token = true;
            p.is_main_token = true;
        })
    }
    pub(super) fn prec(value: Precedence, content: Self) -> Self {
        Self::with_metadata(content, |p| p.precedence = value)
    }
    pub(super) fn prec_left(value: Precedence, content: Self) -> Self {
        Self::with_metadata(content, |p| {
            p.precedence = value;
            p.associativity = Some(Associativity::Left);
        })
    }
    pub(super) fn prec_right(value: Precedence, content: Self) -> Self {
        Self::with_metadata(content, |p| {
            p.precedence = value;
            p.associativity = Some(Associativity::Right);
        })
    }
    pub(super) fn prec_dynamic(value: i32, content: Self) -> Self {
        Self::with_metadata(content, |p| p.dynamic_precedence = value)
    }
    pub(super) fn repeat(rule: Self) -> Self {
        Self::Repeat(Box::new(rule))
    }
    pub(super) fn seq(rules: Vec<Self>) -> Self {
        Self::Seq(rules)
    }
    pub(super) fn choice(rules: Vec<Self>) -> Self {
        let mut result = Vec::with_capacity(rules.len());
        for rule in rules {
            if let Self::Choice(children) = rule {
                result.extend(children);
            } else if !result.contains(&rule) {
                result.push(rule);
            }
        }
        Self::Choice(result)
    }
    pub(super) fn pattern(value: &'static str, flags: &'static str) -> Self {
        Self::Pattern(value.into(), flags.into())
    }
}

use super::{
    Constraint, ContainerKind, DataTy, DisallowedItemKind, DslError, ElemTy, ExpandErrorKind,
    InnerTy, LexErrorKind, LowerErrorKind, NativeDslError, NoteMessage, ParseErrorKind,
    ResolveErrorKind, ScalarTy, TupleSig, Ty, TypeErrorKind, parse_native_dsl,
};

/// Parse a grammar and match the first variable's rule directly in its pool.
macro_rules! rule_tests {
    ($($name:ident { $input:expr, $expected:expr })*) => {
        $(#[test] fn $name() {
            let g = dsl($input);
            assert_rule(&g.pool, g.variables[0].root, &$expected);
        })*
    };
}

/// Generate tests that just verify a grammar compiles without error.
macro_rules! compile_tests {
    ($($name:ident { $input:expr })*) => {
        $(#[test] fn $name() { dsl($input); })*
    };
}

macro_rules! find_rule_tests {
    ($($name:ident { $input:expr, $rule:expr, $expected:expr })*) => {
        $(#[test] fn $name() {
            let g = dsl($input);
            assert_named_rule(&g, $rule, &$expected);
        })*
    };
}

macro_rules! rule_names_tests {
    ($($name:ident { $input:expr, $expected:expr })*) => {
        $(#[test] fn $name() {
            let g = dsl($input);
            assert_eq!(
                g.variables
                    .iter()
                    .map(|variable| g.pool.resolve(variable.name))
                    .collect::<Vec<_>>(),
                $expected,
            );
        })*
    };
}

macro_rules! externals_tests {
    ($($name:ident { $input:expr, $expected:expr })*) => {
        $(#[test] fn $name() {
            let g = dsl($input);
            assert_rules(&g.pool, &g.external_roots, &$expected);
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

/// Generate test functions asserting a specific error kind for a given input.
/// `$variant` is the `DslError` variant to match (e.g. Parse, Resolve, Lower).
///
/// Two forms:
/// - `error_tests! { Variant { name { input, EXPECTED } ... } }` compares the
///   kind by value (`assert_eq!`); for kinds that derive `Eq`.
/// - `error_tests! { match Variant { name { input, PATTERN [if GUARD] } ... } }`
///   pattern-matches the kind (`matches!`); for kinds that can't derive `Eq`
///   (e.g. holding a non-`Eq` `io::Error`). The optional guard keeps the payload
///   assertion.
macro_rules! error_tests {
    ($variant:ident { $($name:ident { $input:expr, $expected:expr })* }) => {
        $(#[test] fn $name() {
            let e = assert_err!(dsl_err($input), $variant);
            assert_eq!(e.kind, $expected);
        })*
    };
    (match $variant:ident { $($name:ident { $input:expr, $expected:pat $(if $guard:expr)? })* }) => {
        $(#[test] fn $name() {
            let e = assert_err!(dsl_err($input), $variant);
            assert!(
                matches!(&e.kind, $expected $(if $guard)?),
                "unexpected error kind: {:?}", e.kind
            );
        })*
    };
}

mod bindings;
mod cfg;
mod combinators;
mod config;
mod errors;
mod imports;
mod inheritance;
mod iterative;
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

pub(super) fn assert_rule(
    pool: &crate::rules::RulePool,
    actual: RuleId,
    expected: &RuleExpectation,
) {
    fn mismatch(
        pool: &crate::rules::RulePool,
        actual: RuleId,
        expected: &RuleExpectation,
        path: &str,
    ) -> Result<(), String> {
        match (pool.node(actual), expected) {
            (RuleNode::Blank, Rule::Blank) => Ok(()),
            (RuleNode::String(a), Rule::String(e))
            | (RuleNode::NamedSymbol(a), Rule::NamedSymbol(e))
                if pool.resolve(a) == e =>
            {
                Ok(())
            }
            (RuleNode::Pattern(a, af), Rule::Pattern(e, ef))
                if pool.resolve(a) == e && pool.resolve(af) == ef =>
            {
                Ok(())
            }
            (RuleNode::Seq(range), Rule::Seq(expected))
            | (RuleNode::Choice(range), Rule::Choice(expected)) => {
                let actual = pool.child_slice(range);
                if actual.len() != expected.len() {
                    return Err(format!(
                        "{path}: child count {} != {}",
                        actual.len(),
                        expected.len()
                    ));
                }
                for (index, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
                    mismatch(pool, actual, expected, &format!("{path}[{index}]"))?;
                }
                Ok(())
            }
            (RuleNode::Repeat(actual), Rule::Repeat(expected)) => {
                mismatch(pool, actual, expected, &format!("{path}.repeat"))
            }
            (
                RuleNode::Reserved { rule, ctx },
                Rule::Reserved {
                    rule: expected,
                    context_name,
                },
            ) if pool.resolve(ctx) == context_name => {
                mismatch(pool, rule, expected, &format!("{path}.reserved"))
            }
            (
                RuleNode::Metadata { params, rule },
                Rule::Metadata {
                    params: expected_params,
                    rule: expected_rule,
                },
            ) => {
                let params = pool.params(params);
                let precedence = match params.precedence {
                    crate::rules::Precedence::None => Precedence::None,
                    crate::rules::Precedence::Integer(n) => Precedence::Integer(n),
                    crate::rules::Precedence::Name(s) => Precedence::Name(pool.resolve(s).into()),
                };
                let equal = precedence == expected_params.precedence
                    && params.dynamic_precedence == expected_params.dynamic_precedence
                    && params.associativity == expected_params.associativity
                    && params.is_token == expected_params.is_token
                    && params.is_main_token == expected_params.is_main_token
                    && params.alias.map(|a| (pool.resolve(a.value), a.is_named))
                        == expected_params
                            .alias
                            .as_ref()
                            .map(|a| (a.value.as_str(), a.is_named))
                    && params.field.map(|s| pool.resolve(s))
                        == expected_params.field_name.as_deref();
                if !equal {
                    return Err(format!("{path}.metadata: parameters differ"));
                }
                mismatch(pool, rule, expected_rule, &format!("{path}.metadata.rule"))
            }
            (actual, expected) => Err(format!("{path}: {actual:?} != {expected:?}")),
        }
    }

    if let Err(reason) = mismatch(pool, actual, expected, "root") {
        panic!("pooled rule did not match expectation: {reason}\nexpected: {expected:#?}");
    }
}

pub(super) fn assert_rules(
    pool: &crate::rules::RulePool,
    actual: &[RuleId],
    expected: &[RuleExpectation],
) {
    assert_eq!(actual.len(), expected.len(), "rule-list lengths differ");
    for (&actual, expected) in actual.iter().zip(expected) {
        assert_rule(pool, actual, expected);
    }
}

pub(super) fn assert_named_rule(grammar: &InputGrammar, name: &str, expected: &RuleExpectation) {
    let root = rule_id(grammar, name);
    assert_rule(&grammar.pool, root, expected);
}

pub(super) fn rule_id(grammar: &InputGrammar, name: &str) -> RuleId {
    grammar
        .variables
        .iter()
        .find(|variable| grammar.pool.resolve(variable.name) == name)
        .unwrap_or_else(|| panic!("rule {name:?} not found"))
        .root
}

pub(super) fn dsl_err(input: &str) -> DslError {
    parse_native_dsl(input, &test_fixtures_dir().join("grammar.tsg")).unwrap_err()
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
    let root_path = dir.path().join("grammar.tsg");
    std::fs::write(&root_path, root).unwrap();
    parse_native_dsl(root, &root_path)
}

pub(super) fn rule_names(g: &InputGrammar) -> Vec<&str> {
    g.variables
        .iter()
        .map(|variable| g.pool.resolve(variable.name))
        .collect()
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
