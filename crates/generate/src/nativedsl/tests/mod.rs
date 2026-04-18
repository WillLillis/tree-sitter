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
mod imports;
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
    let json = serde_json::to_string_pretty(&super::serialize::grammar_to_json(&grammar))
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
    if native.reserved_words != js.reserved_words {
        mismatches.push("CONFIG: reserved".into());
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
fn bench_native_dsl_vim() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../grammars/tree-sitter-vim/grammar.tsg");
    let source = std::fs::read_to_string(&path).unwrap();
    run_bench("Native DSL Vim", || {
        parse_native_dsl(&source, &path).unwrap();
    });
}

#[test]
#[ignore = "benchmark"]
fn bench_native_dsl_python() {
    let path =
        std::path::PathBuf::from("/home/lillis/projects/grammars/tree-sitter-python/grammar.tsg");
    let source = std::fs::read_to_string(&path).unwrap();
    run_bench("Native DSL Python", || {
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

#[test]
#[ignore = "benchmark"]
fn bench_lexer_only() {
    let source = read_native_grammar("c_native");
    let n = 20000u32;
    // Warmup
    for _ in 0..10 {
        std::hint::black_box(super::lexer::Lexer::new(&source).tokenize().unwrap());
    }
    let start = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(super::lexer::Lexer::new(&source).tokenize().unwrap());
    }
    let elapsed = start.elapsed();
    eprintln!("Lexer only: {:?}/iter", elapsed / n);

    // Also measure: how many tokens?
    let tokens = super::lexer::Lexer::new(&source).tokenize().unwrap();
    eprintln!("Token count: {}", tokens.len());
    let mut ident_count = 0u32;
    let mut kw_count = 0u32;
    let mut str_count = 0u32;
    let mut punct_count = 0u32;
    let mut comment_count = 0u32;
    for t in &tokens {
        match t.kind {
            super::lexer::TokenKind::Ident => ident_count += 1,
            super::lexer::TokenKind::StringLit | super::lexer::TokenKind::RawStringLit { .. } => {
                str_count += 1;
            }
            super::lexer::TokenKind::Comment => comment_count += 1,
            k if k.is_keyword() => kw_count += 1,
            _ => punct_count += 1,
        }
    }
    eprintln!(
        "  Idents: {ident_count}, Keywords: {kw_count}, Strings: {str_count}, Punct: {punct_count}, Comments: {comment_count}"
    );
}

#[test]
#[ignore = "benchmark"]
fn bench_lexer_breakdown() {
    use super::lexer::{Lexer, Token, TokenKind};
    let source = read_native_grammar("c_native");

    eprintln!("Token size: {} bytes", std::mem::size_of::<Token>());
    eprintln!("TokenKind size: {} bytes", std::mem::size_of::<TokenKind>());

    // Measure: tokenize fully
    let n = 20000u32;
    for _ in 0..10 {
        std::hint::black_box(Lexer::new(&source).tokenize().unwrap());
    }
    let start = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(Lexer::new(&source).tokenize().unwrap());
    }
    let full = start.elapsed();
    eprintln!("Full tokenize: {:?}/iter", full / n);

    // Measure: just scanning (skip_whitespace + classify first byte) without building tokens
    // We approximate by counting bytes processed
    let bytes = source.as_bytes();
    let start = std::time::Instant::now();
    for _ in 0..n {
        let mut pos = 0usize;
        let mut count = 0u32;
        while pos < bytes.len() {
            // Skip whitespace
            while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            if pos >= bytes.len() {
                break;
            }
            // Just advance past one "token" worth of bytes
            match bytes[pos] {
                b'{' | b'}' | b'(' | b')' | b'[' | b']' | b',' | b'.' | b'=' | b'<' | b'>' => {
                    pos += 1;
                }
                b':' | b'-' => {
                    pos += if pos + 1 < bytes.len() && matches!(bytes[pos + 1], b':' | b'>') {
                        2
                    } else {
                        1
                    }
                }
                b'"' => {
                    pos += 1;
                    while pos < bytes.len() && bytes[pos] != b'"' {
                        if bytes[pos] == b'\\' {
                            pos += 1;
                        }
                        pos += 1;
                    }
                    pos += 1;
                }
                b'/' if pos + 1 < bytes.len() && bytes[pos + 1] == b'/' => {
                    while pos < bytes.len() && bytes[pos] != b'\n' {
                        pos += 1;
                    }
                }
                b'0'..=b'9' => {
                    while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                        pos += 1;
                    }
                }
                b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                    while pos < bytes.len()
                        && matches!(bytes[pos], b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')
                    {
                        pos += 1;
                    }
                }
                _ => pos += 1,
            }
            count += 1;
        }
        std::hint::black_box(count);
    }
    let scan_only = start.elapsed();
    eprintln!("Scan only (no Token/Vec): {:?}/iter", scan_only / n);
    eprintln!(
        "Overhead (Token construction + Vec + keywords): {:?}/iter",
        full.checked_sub(scan_only).unwrap()
    );
}

#[test]
#[ignore = "benchmark"]
fn bench_per_stage() {
    let path = native_grammar_path("c_native");
    let source = read_native_grammar("c_native");
    let n = 10000u32;

    // Lex
    let start = std::time::Instant::now();
    let mut tokens_store = None;
    for _ in 0..n {
        let t = super::lexer::Lexer::new(&source).tokenize().unwrap();
        tokens_store = Some(t);
    }
    let lex_time = start.elapsed() / n;
    let tokens = tokens_store.unwrap();

    // Parse
    let start = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(
            super::parser::Parser::new(&tokens, source.clone(), &path)
                .parse()
                .unwrap(),
        );
    }
    let parse_time = start.elapsed() / n;

    let mut ast = super::parser::Parser::new(&tokens, source.clone(), &path)
        .parse()
        .unwrap();
    super::validate_grammar(&ast).unwrap();
    // C grammar has no inheritance or imports
    let base_names: Vec<String> = Vec::new();
    let inherit_span = None;

    // Resolve
    let start = std::time::Instant::now();
    for _ in 0..n {
        let mut a = super::parser::Parser::new(&tokens, source.clone(), &path)
            .parse()
            .unwrap();
        super::resolve::resolve(&mut a, &base_names, inherit_span, &path).unwrap();
    }
    let resolve_time = start.elapsed() / n;

    super::resolve::resolve(&mut ast, &base_names, inherit_span, &path).unwrap();

    // Typecheck
    let start = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(super::typecheck::check(&ast, Vec::new()).unwrap());
    }
    let typecheck_time = start.elapsed() / n;

    // Lower (C grammar has no base, so base is always None)
    // This includes: eval all expressions + build_rule (ARule→Rule conversion)
    // + config field evaluation. build_rule dominates due to String allocations.
    let start = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(super::lower::lower_with_base(&ast, &path, &[]).unwrap());
    }
    let lower_time = start.elapsed() / n;

    let total = lex_time + parse_time + resolve_time + typecheck_time + lower_time;
    eprintln!(
        "Lex:       {:>8?} ({:.1}%)",
        lex_time,
        lex_time.as_nanos() as f64 / total.as_nanos() as f64 * 100.0
    );
    eprintln!(
        "Parse:     {:>8?} ({:.1}%)",
        parse_time,
        parse_time.as_nanos() as f64 / total.as_nanos() as f64 * 100.0
    );
    eprintln!(
        "Resolve:   {:>8?} ({:.1}%)",
        resolve_time,
        resolve_time.as_nanos() as f64 / total.as_nanos() as f64 * 100.0
    );
    eprintln!(
        "Typecheck: {:>8?} ({:.1}%)",
        typecheck_time,
        typecheck_time.as_nanos() as f64 / total.as_nanos() as f64 * 100.0
    );
    eprintln!(
        "Lower:     {:>8?} ({:.1}%)",
        lower_time,
        lower_time.as_nanos() as f64 / total.as_nanos() as f64 * 100.0
    );
    eprintln!("Total:     {:>8?}", total);
}
