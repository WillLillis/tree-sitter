use std::path::{Path, PathBuf};

use crate::grammars::InputGrammar as PooledGrammar;
use crate::nativedsl::lexer::TokenKind;
use crate::rules::{Associativity, Rule as PooledRule, RuleId};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct InputGrammar {
    name: String,
    variables: Vec<Variable>,
    external_tokens: Vec<Rule>,
    extra_symbols: Vec<Rule>,
    reserved_words: Vec<ReservedWordContext>,
    supertype_symbols: Vec<String>,
    expected_conflicts: Vec<Vec<String>>,
    variables_to_inline: Vec<String>,
    word_token: Option<String>,
    precedence_orderings: Vec<Vec<PrecedenceEntry>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Variable {
    name: String,
    rule: Rule,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ReservedWordContext {
    name: String,
    reserved_words: Vec<Rule>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum PrecedenceEntry {
    Name(String),
    Symbol(String),
}

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

// Temporary migration alias for test files that still use the legacy name.
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

fn normalize_rule(grammar: &PooledGrammar, id: RuleId) -> Rule {
    let strings = &grammar.pool;
    match strings.node(id) {
        PooledRule::Blank => Rule::Blank,
        PooledRule::String(value) => Rule::String(strings.resolve(value).into()),
        PooledRule::Pattern(value, flags) => {
            Rule::Pattern(strings.resolve(value).into(), strings.resolve(flags).into())
        }
        PooledRule::NamedSymbol(name) => Rule::NamedSymbol(strings.resolve(name).into()),
        PooledRule::Sym { .. } => panic!("native DSL lowering emitted an interned symbol"),
        PooledRule::Seq(range) => Rule::Seq(
            strings
                .child_slice(range)
                .iter()
                .map(|&child| normalize_rule(grammar, child))
                .collect(),
        ),
        PooledRule::Choice(range) => Rule::Choice(
            strings
                .child_slice(range)
                .iter()
                .map(|&child| normalize_rule(grammar, child))
                .collect(),
        ),
        PooledRule::Repeat(rule) => Rule::Repeat(Box::new(normalize_rule(grammar, rule))),
        PooledRule::Reserved { rule, ctx } => Rule::Reserved {
            rule: Box::new(normalize_rule(grammar, rule)),
            context_name: strings.resolve(ctx).into(),
        },
        PooledRule::Metadata { params, rule } => {
            let params = strings.params(params);
            Rule::Metadata {
                params: MetadataParams {
                    precedence: match params.precedence {
                        crate::rules::Precedence::None => Precedence::None,
                        crate::rules::Precedence::Integer(n) => Precedence::Integer(n),
                        crate::rules::Precedence::Name(s) => {
                            Precedence::Name(strings.resolve(s).into())
                        }
                    },
                    dynamic_precedence: params.dynamic_precedence,
                    associativity: params.associativity,
                    is_token: params.is_token,
                    is_main_token: params.is_main_token,
                    alias: params.alias.map(|a| Alias {
                        value: strings.resolve(a.value).into(),
                        is_named: a.is_named,
                    }),
                    field_name: params.field.map(|s| strings.resolve(s).into()),
                },
                rule: Box::new(normalize_rule(grammar, rule)),
            }
        }
    }
}

impl From<PooledGrammar> for InputGrammar {
    fn from(grammar: PooledGrammar) -> Self {
        let resolve = |id| grammar.pool.resolve(id).to_owned();
        Self {
            name: resolve(grammar.name),
            variables: grammar
                .variables
                .iter()
                .map(|v| Variable {
                    name: resolve(v.name),
                    rule: normalize_rule(&grammar, v.root),
                })
                .collect(),
            external_tokens: grammar
                .external_roots
                .iter()
                .map(|&r| normalize_rule(&grammar, r))
                .collect(),
            extra_symbols: grammar
                .extra_roots
                .iter()
                .map(|&r| normalize_rule(&grammar, r))
                .collect(),
            reserved_words: grammar
                .reserved_sets
                .iter()
                .map(|set| ReservedWordContext {
                    name: resolve(set.name),
                    reserved_words: set
                        .roots
                        .iter()
                        .map(|&r| normalize_rule(&grammar, r))
                        .collect(),
                })
                .collect(),
            supertype_symbols: grammar
                .supertype_names
                .iter()
                .map(|&s| resolve(s))
                .collect(),
            expected_conflicts: grammar
                .conflict_names
                .iter()
                .map(|set| set.iter().map(|&s| resolve(s)).collect())
                .collect(),
            variables_to_inline: grammar.inline_names.iter().map(|&s| resolve(s)).collect(),
            word_token: grammar.word_name.map(resolve),
            precedence_orderings: grammar
                .precedence_orderings
                .iter()
                .map(|ordering| {
                    ordering
                        .iter()
                        .map(|entry| match entry {
                            crate::grammars::PrecedenceEntry::Name(s) => {
                                PrecedenceEntry::Name(resolve(*s))
                            }
                            crate::grammars::PrecedenceEntry::Symbol(s) => {
                                PrecedenceEntry::Symbol(resolve(*s))
                            }
                        })
                        .collect()
                })
                .collect(),
        }
    }
}

use super::{
    Constraint, ContainerKind, DataTy, DisallowedItemKind, DslError, ElemTy, ExpandErrorKind,
    InnerTy, LexErrorKind, LowerErrorKind, NativeDslError, NoteMessage, ParseErrorKind,
    ResolveErrorKind, ScalarTy, TupleSig, Ty, TypeErrorKind, parse_native_dsl,
};

/// Parse a grammar and match the first variable's pooled rule.
macro_rules! pooled_rule_tests {
    ($($name:ident { $input:expr, $expected:expr })*) => {
        $(#[test] fn $name() {
            let g = pooled_dsl($input);
            assert_rule(&g.pool, g.variables[0].root, &$expected);
        })*
    };
}

/// Generate tests that just verify a grammar compiles without error.
macro_rules! compile_tests {
    ($($name:ident { $input:expr })*) => {
        $(#[test] fn $name() { pooled_dsl($input); })*
    };
}

/// Generate tests that parse a grammar and assert on a named rule's body.
macro_rules! find_rule_tests {
    ($($name:ident { $input:expr, $rule:expr, $expected:expr })*) => {
        $(#[test] fn $name() {
            let g = dsl($input);
            assert_eq!(*find_rule(&g, $rule), $expected);
        })*
    };
}

macro_rules! pooled_find_rule_tests {
    ($($name:ident { $input:expr, $rule:expr, $expected:expr })*) => {
        $(#[test] fn $name() {
            let g = pooled_dsl($input);
            assert_named_rule(&g, $rule, &$expected);
        })*
    };
}

macro_rules! pooled_rule_names_tests {
    ($($name:ident { $input:expr, $expected:expr })*) => {
        $(#[test] fn $name() {
            let g = pooled_dsl($input);
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

macro_rules! pooled_externals_tests {
    ($($name:ident { $input:expr, $expected:expr })*) => {
        $(#[test] fn $name() {
            let g = pooled_dsl($input);
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
    parse_native_dsl(input, &test_fixtures_dir().join("grammar.tsg"))
        .unwrap()
        .into()
}

pub(super) fn pooled_dsl(input: &str) -> PooledGrammar {
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
            (PooledRule::Blank, Rule::Blank) => Ok(()),
            (PooledRule::String(a), Rule::String(e))
            | (PooledRule::NamedSymbol(a), Rule::NamedSymbol(e))
                if pool.resolve(a) == e =>
            {
                Ok(())
            }
            (PooledRule::Pattern(a, af), Rule::Pattern(e, ef))
                if pool.resolve(a) == e && pool.resolve(af) == ef =>
            {
                Ok(())
            }
            (PooledRule::Seq(range), Rule::Seq(expected))
            | (PooledRule::Choice(range), Rule::Choice(expected)) => {
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
            (PooledRule::Repeat(actual), Rule::Repeat(expected)) => {
                mismatch(pool, actual, expected, &format!("{path}.repeat"))
            }
            (
                PooledRule::Reserved { rule, ctx },
                Rule::Reserved {
                    rule: expected,
                    context_name,
                },
            ) if pool.resolve(ctx) == context_name => {
                mismatch(pool, rule, expected, &format!("{path}.reserved"))
            }
            (
                PooledRule::Metadata { params, rule },
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

pub(super) fn assert_named_rule(grammar: &PooledGrammar, name: &str, expected: &RuleExpectation) {
    let variable = grammar
        .variables
        .iter()
        .find(|variable| grammar.pool.resolve(variable.name) == name)
        .unwrap_or_else(|| panic!("rule {name:?} not found"));
    assert_rule(&grammar.pool, variable.root, expected);
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
    // The root path must exist on disk - it's canonicalized to seed cycle
    // detection - even though parsing reads `root` directly.
    let root_path = dir.path().join("grammar.tsg");
    std::fs::write(&root_path, root).unwrap();
    parse_native_dsl(root, &root_path).map(Into::into)
}

pub(super) fn pooled_parse_with_modules(
    modules: &[(&str, &str)],
    root: &str,
) -> Result<PooledGrammar, DslError> {
    let dir = tempfile::tempdir().unwrap();
    for (name, src) in modules {
        std::fs::write(dir.path().join(name), src).unwrap();
    }
    let root_path = dir.path().join("grammar.tsg");
    std::fs::write(&root_path, root).unwrap();
    parse_native_dsl(root, &root_path)
}

pub(super) fn pooled_rule_names(g: &PooledGrammar) -> Vec<&str> {
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
