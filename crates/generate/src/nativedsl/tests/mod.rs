use std::path::Path;

use crate::grammars::{InputGrammar, VariableType};
use crate::nativedsl::lexer::TokenKind;
use crate::rules::{Precedence, Rule};

use super::{
    DslError, InnerTy, LexErrorKind, LowerErrorKind, ParseErrorKind, ResolveErrorKind, Ty,
    TypeErrorKind, ast, parse_native_dsl,
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
mod inheritance;
mod print;
mod roundtrip;
mod types;

pub(super) fn test_grammars_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test/fixtures/test_grammars")
}

pub(super) fn dsl(input: &str) -> InputGrammar {
    parse_native_dsl(input, &test_grammars_dir().join("grammar.tsg")).unwrap()
}

pub(super) fn dsl_err(input: &str) -> DslError {
    parse_native_dsl(input, &test_grammars_dir().join("grammar.tsg")).unwrap_err()
}

/// Follows the same path as the CLI: `parse_native_dsl` -> normalize -> `grammar_to_json`.
pub(super) fn dsl_to_json(input: &str, path: &std::path::Path) -> (InputGrammar, String) {
    let mut grammar = parse_native_dsl(input, path).unwrap();
    crate::parse_grammar::normalize_grammar(&mut grammar);
    let json = serde_json::to_string_pretty(&super::lower::grammar_to_json(&grammar))
        .expect("grammar JSON serialization should not fail");
    (grammar, json)
}

pub(super) fn native_grammar_path(name: &str) -> std::path::PathBuf {
    test_grammars_dir().join(name).join("grammar.tsg")
}

pub(super) fn read_native_grammar(name: &str) -> String {
    let path = native_grammar_path(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
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

pub(super) fn assert_grammar_eq(native: &InputGrammar, js: &InputGrammar) {
    assert_eq!(native.name, js.name, "name mismatch");
    let native_rules: rustc_hash::FxHashMap<&str, &Rule> = native
        .variables
        .iter()
        .map(|v| (v.name.as_str(), &v.rule))
        .collect();
    let js_rules: rustc_hash::FxHashMap<&str, &Rule> = js
        .variables
        .iter()
        .map(|v| (v.name.as_str(), &v.rule))
        .collect();
    let mut mismatches = Vec::new();
    for (name, js_rule) in &js_rules {
        match native_rules.get(name) {
            None => mismatches.push(format!("MISSING: {name}")),
            Some(native_rule) if native_rule != js_rule => {
                mismatches.push(format!("MISMATCH: {name}"));
            }
            _ => {}
        }
    }
    for name in native_rules.keys() {
        if !js_rules.contains_key(name) {
            mismatches.push(format!("EXTRA: {name}"));
        }
    }
    if native.extra_symbols != js.extra_symbols {
        mismatches.push("CONFIG: extras".into());
    }
    if native.expected_conflicts != js.expected_conflicts {
        mismatches.push("CONFIG: conflicts".into());
    }
    if native.external_tokens != js.external_tokens {
        mismatches.push("CONFIG: externals".into());
    }
    if native.variables_to_inline != js.variables_to_inline {
        mismatches.push("CONFIG: inline".into());
    }
    if native.supertype_symbols != js.supertype_symbols {
        mismatches.push("CONFIG: supertypes".into());
    }
    if native.word_token != js.word_token {
        mismatches.push("CONFIG: word".into());
    }
    if native.precedence_orderings != js.precedence_orderings {
        mismatches.push("CONFIG: precedences".into());
    }
    if !mismatches.is_empty() {
        mismatches.sort();
        panic!("{} issues:\n{}", mismatches.len(), mismatches.join("\n"));
    }
}

// ===== Benchmarks =====

fn run_bench(label: &str, f: impl Fn()) {
    for _ in 0..10 {
        #[expect(clippy::unit_arg, reason = "bench")]
        std::hint::black_box(f());
    }
    let n = 1000;
    let start = std::time::Instant::now();
    for _ in 0..n {
        #[expect(clippy::unit_arg, reason = "bench")]
        std::hint::black_box(f());
    }
    eprintln!("{label}: {:?}/iter", start.elapsed() / n);
}

#[test]
#[ignore = "benchmark"]
fn bench_native_dsl_c() {
    let path = native_grammar_path("c_native");
    let source = read_native_grammar("c_native");
    run_bench("Native DSL C", || {
        parse_native_dsl(&source, &path).unwrap();
    });
}

#[test]
#[ignore = "benchmark"]
fn bench_native_dsl_javascript() {
    let path = native_grammar_path("javascript_native");
    let source = read_native_grammar("javascript_native");
    run_bench("Native DSL JavaScript", || {
        parse_native_dsl(&source, &path).unwrap();
    });
}

#[test]
#[ignore = "benchmark"]
fn bench_native_dsl_cpp() {
    let path = native_grammar_path("cpp_native");
    let source = read_native_grammar("cpp_native");
    run_bench("Native DSL C++", || {
        parse_native_dsl(&source, &path).unwrap();
    });
}

#[test]
#[ignore = "benchmark"]
fn bench_json_parse_c() {
    let json = std::fs::read_to_string(test_grammars_dir().join("../grammars/c/src/grammar.json"))
        .unwrap();
    run_bench("JSON parse_grammar C", || {
        crate::parse_grammar::parse_grammar(&json).unwrap();
    });
}
