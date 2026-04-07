//! Native DSL front-end for tree-sitter grammar definitions.
//!
//! This module implements a five-stage pipeline that compiles `.tsg` source
//! files into the same [`InputGrammar`] representation produced by parsing
//! `grammar.json`:
//!
//! 1. **Lexing** ([`lexer`]) - tokenizes source text into a flat `Vec<Token>`.
//! 2. **Parsing** ([`parser`]) - builds an arena-based AST from the token stream.
//! 3. **Name resolution** ([`resolve`]) - resolves identifiers to rule/variable references.
//! 4. **Type checking** ([`typecheck`]) - validates types across expressions and function calls.
//! 5. **Lowering** ([`lower`]) - evaluates the AST into an [`InputGrammar`].
//!
//! Each stage produces structured, serializable errors that carry source spans.

pub mod ast;
mod lexer;
pub mod lower;
mod parser;
mod resolve;
mod typecheck;

use std::path::{Path, PathBuf};

use crate::grammars::InputGrammar;

use ast::Span;
use lexer::LexError;
use lower::LowerError;
use parser::ParseError;
use resolve::ResolveError;
use serde::Serialize;
use thiserror::Error;
use typecheck::TypeError;

/// Parse a native DSL source file into an [`InputGrammar`].
///
/// Runs the full pipeline: lex, parse, resolve, typecheck, lower.
///
/// # Errors
///
/// Returns [`DslError`] if any pipeline stage fails. The error carries the
/// source [`Span`] of the offending construct.
pub fn parse_native_dsl(input: &str, grammar_dir: &Path) -> Result<InputGrammar, DslError> {
    let tokens = lexer::Lexer::new(input).tokenize()?;
    let mut ast = parser::Parser::new(tokens, input).parse()?;
    resolve::resolve(&mut ast)?;
    typecheck::check(&ast)?;
    Ok(lower::lower_with_path(&ast, grammar_dir, &[])?)
}

/// Parse a native DSL source file and serialize to `grammar.json` format.
///
/// This is a convenience wrapper around [`parse_native_dsl`] followed by
/// JSON serialization. The output is intended to be compatible with the
/// existing `grammar.json` format consumed by the tree-sitter generator.
///
/// # Errors
///
/// Returns [`DslError`] if parsing fails.
///
/// # Panics
///
/// Panics if JSON serialization of the grammar fails (should not happen with
/// valid grammar data).
pub fn parse_native_dsl_to_json(input: &str, grammar_dir: &Path) -> Result<String, DslError> {
    let grammar = parse_native_dsl(input, grammar_dir)?;
    Ok(
        serde_json::to_string_pretty(&lower::grammar_to_json(&grammar))
            .expect("grammar JSON serialization should not fail"),
    )
}

/// Unified error type for all pipeline stages.
///
/// Each variant wraps the stage-specific error and can be converted from it
/// via `From`/`?`. Use [`DslError::span`] to get the source location.
#[derive(Debug, Error, Serialize)]
pub enum DslError {
    #[error(transparent)]
    Lex(#[from] LexError),
    #[error(transparent)]
    Parse(#[from] ParseError),
    #[error(transparent)]
    Resolve(#[from] ResolveError),
    #[error(transparent)]
    Type(#[from] TypeError),
    #[error(transparent)]
    Lower(#[from] LowerError),
}

impl DslError {
    /// Returns the source span where the error occurred.
    #[must_use]
    pub const fn span(&self) -> Span {
        match self {
            Self::Lex(e) => e.span,
            Self::Parse(e) => e.span,
            Self::Resolve(e) => e.span,
            Self::Type(e) => e.span,
            Self::Lower(e) => e.span,
        }
    }
}

/// A DSL pipeline error bundled with the source text and file path needed
/// for rich diagnostic rendering.
///
/// This is the error type produced by [`crate::LoadGrammarError::NativeDsl`].
/// Its [`Display`](std::fmt::Display) impl renders a Rust-style diagnostic
/// with source context, line numbers, and underline pointers.
#[derive(Debug, Error, Serialize)]
#[error("{}", render_diagnostic(error, source_text, path))]
pub struct NativeDslError {
    pub error: DslError,
    #[serde(skip)]
    pub source_text: String,
    #[serde(skip)]
    pub path: PathBuf,
}

fn paint(color: anstyle::AnsiColor, text: &str) -> String {
    let style = anstyle::Style::new().fg_color(Some(color.into())).bold();
    format!("{style}{text}{style:#}")
}

fn render_diagnostic(error: &DslError, source_text: &str, path: &Path) -> String {
    use std::fmt::Write;

    use anstyle::AnsiColor;

    let span = error.span();
    let offset = span.start as usize;
    let bytes = source_text.as_bytes();

    // Find line boundaries and line number
    let mut line_start = 0;
    let mut prev_line_start = None;
    let mut line_num = 1;
    for (i, &b) in bytes.iter().enumerate().take(offset) {
        if b == b'\n' {
            prev_line_start = Some(line_start);
            line_start = i + 1;
            line_num += 1;
        }
    }
    let line_end =
        memchr::memchr(b'\n', &bytes[line_start..]).map_or(bytes.len(), |pos| line_start + pos);
    let col = offset - line_start + 1;

    let line_text = &source_text[line_start..line_end];
    let span_start_in_line = col.saturating_sub(1);
    let span_len = (span.end - span.start) as usize;
    let underline_len = span_len
        .max(1)
        .min(line_text.len().saturating_sub(span_start_in_line));

    let path = path.display();
    let gutter_width = digit_count(line_num);
    let pipe = paint(AnsiColor::Cyan, "|");
    let underline = paint(AnsiColor::Red, &"^".repeat(underline_len));

    let mut out = String::new();
    writeln!(out, "{}: {error}", paint(AnsiColor::Red, "error")).unwrap();
    writeln!(
        out,
        " {} {path}:{line_num}:{col}",
        paint(AnsiColor::Cyan, "-->")
    )
    .unwrap();
    writeln!(out, " {:>gutter_width$} {pipe}", "").unwrap();
    if let Some(prev_start) = prev_line_start {
        let prev_end =
            memchr::memchr(b'\n', &bytes[prev_start..]).map_or(bytes.len(), |pos| prev_start + pos);
        let prev_num = line_num - 1;
        writeln!(
            out,
            " {} {pipe} {}",
            paint(AnsiColor::Cyan, &prev_num.to_string()),
            &source_text[prev_start..prev_end],
        )
        .unwrap();
    }
    writeln!(
        out,
        " {} {pipe} {line_text}",
        paint(AnsiColor::Cyan, &line_num.to_string()),
    )
    .unwrap();
    write!(
        out,
        " {:>gutter_width$} {pipe} {:>span_start_in_line$}{underline}",
        "", "",
    )
    .unwrap();
    out
}

fn digit_count(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    ((n as f64).log10().floor() as usize) + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammars::VariableType;
    use crate::rules::Rule;

    #[test]
    fn test_minimal_grammar() {
        // Largest payload: NodeId(4) + Option<NodeId>(4) + NodeId(4) = 12, plus 4-byte tag = 16
        // 4 nodes per 64-byte cache line
        assert_eq!(std::mem::size_of::<ast::Node>(), 16);
        let input = r#"
            grammar {
                language: "test",
            }

            rule program {
                repeat("hello")
            }
        "#;
        let grammar = parse_native_dsl(input, Path::new(".")).unwrap();
        assert_eq!(grammar.name, "test");
        assert_eq!(grammar.variables.len(), 1);
        assert_eq!(grammar.variables[0].name, "program");
        assert_eq!(grammar.variables[0].kind, VariableType::Named);
        assert_eq!(
            grammar.variables[0].rule,
            Rule::choice(vec![
                Rule::repeat(Rule::String("hello".into())),
                Rule::Blank,
            ])
        );
    }

    #[test]
    fn test_function_expansion() {
        let input = r#"
            grammar {
                language: "test",
            }

            fn comma_sep(item: rule) -> rule {
                seq(item, repeat(seq(",", item)))
            }

            rule program {
                comma_sep(identifier)
            }

            rule identifier {
                regexp("[a-z]+")
            }
        "#;
        let grammar = parse_native_dsl(input, Path::new(".")).unwrap();
        assert_eq!(grammar.variables.len(), 2);
        assert_eq!(
            grammar.variables[0].rule,
            Rule::seq(vec![
                Rule::NamedSymbol("identifier".into()),
                Rule::choice(vec![
                    Rule::repeat(Rule::seq(vec![
                        Rule::String(",".into()),
                        Rule::NamedSymbol("identifier".into()),
                    ])),
                    Rule::Blank,
                ]),
            ])
        );
    }

    #[test]
    fn test_let_and_prec() {
        let input = r#"
            grammar {
                language: "test",
            }

            let PREC: int = 5

            rule expr {
                prec.left(PREC, seq(expr, "+", expr))
            }
        "#;
        let grammar = parse_native_dsl(input, Path::new(".")).unwrap();
        assert_eq!(
            grammar.variables[0].rule,
            Rule::prec_left(
                crate::rules::Precedence::Integer(5),
                Rule::seq(vec![
                    Rule::NamedSymbol("expr".into()),
                    Rule::String("+".into()),
                    Rule::NamedSymbol("expr".into()),
                ])
            )
        );
    }

    #[test]
    fn test_object_field_access() {
        let input = r#"
            grammar {
                language: "test",
            }

            let PREC: int = { ADD: 1, MUL: 2 }

            rule add {
                prec.left(PREC.ADD, seq(expr, "+", expr))
            }

            rule mul {
                prec.left(PREC.MUL, seq(expr, "*", expr))
            }

            rule expr {
                choice(add, mul)
            }
        "#;
        let grammar = parse_native_dsl(input, Path::new(".")).unwrap();
        assert_eq!(grammar.variables.len(), 3);
        assert_eq!(
            grammar.variables[0].rule,
            Rule::prec_left(
                crate::rules::Precedence::Integer(1),
                Rule::seq(vec![
                    Rule::NamedSymbol("expr".into()),
                    Rule::String("+".into()),
                    Rule::NamedSymbol("expr".into()),
                ])
            )
        );
    }

    #[test]
    fn test_grammar_config() {
        let input = r#"
            grammar {
                language: "test",
                extras: [regexp(r"\s"), comment],
                externals: [heredoc],
                supertypes: [_expression],
                inline: [_statement],
                conflicts: [[primary, arrow]],
                word: identifier,
            }

            rule program { _expression }
            rule _expression { "x" }
            rule comment { regexp(r"\/\/.*") }
            rule identifier { regexp("[a-z]+") }
        "#;
        let grammar = parse_native_dsl(input, Path::new(".")).unwrap();
        assert_eq!(grammar.extra_symbols.len(), 2);
        assert_eq!(grammar.external_tokens.len(), 1);
        assert_eq!(grammar.supertype_symbols, vec!["_expression"]);
        assert_eq!(grammar.variables_to_inline, vec!["_statement"]);
        assert_eq!(
            grammar.expected_conflicts,
            vec![vec!["primary".to_string(), "arrow".to_string()]]
        );
        assert_eq!(grammar.word_token, Some("identifier".into()));
    }

    #[test]
    fn test_for_expression() {
        let input = r#"
            grammar {
                language: "test",
            }

            rule binary {
                choice(
                    for (op: str, p: int) in [
                        ("+", 1),
                        ("*", 2),
                    ] {
                        prec.left(p, seq(expr, op, expr))
                    }
                )
            }

            rule expr { "x" }
        "#;
        let grammar = parse_native_dsl(input, Path::new(".")).unwrap();
        assert_eq!(
            grammar.variables[0].rule,
            Rule::choice(vec![
                Rule::prec_left(
                    crate::rules::Precedence::Integer(1),
                    Rule::seq(vec![
                        Rule::NamedSymbol("expr".into()),
                        Rule::String("+".into()),
                        Rule::NamedSymbol("expr".into()),
                    ])
                ),
                Rule::prec_left(
                    crate::rules::Precedence::Integer(2),
                    Rule::seq(vec![
                        Rule::NamedSymbol("expr".into()),
                        Rule::String("*".into()),
                        Rule::NamedSymbol("expr".into()),
                    ])
                ),
            ])
        );
    }

    #[test]
    fn test_for_single_binding() {
        let input = r#"
            grammar {
                language: "test",
            }

            rule keywords {
                choice(
                    for (kw: str) in ["if", "else", "while"] {
                        kw
                    }
                )
            }
        "#;
        let grammar = parse_native_dsl(input, Path::new(".")).unwrap();
        assert_eq!(
            grammar.variables[0].rule,
            Rule::choice(vec![
                Rule::String("if".into()),
                Rule::String("else".into()),
                Rule::String("while".into()),
            ])
        );
    }

    #[test]
    fn test_for_single_element_tuple_destructure() {
        let input = r#"
            grammar {
                language: "test",
            }

            rule keywords {
                choice(
                    for (kw: str) in [("if"), ("else"), ("while")] {
                        kw
                    }
                )
            }
        "#;
        let grammar = parse_native_dsl(input, Path::new(".")).unwrap();
        assert_eq!(
            grammar.variables[0].rule,
            Rule::choice(vec![
                Rule::String("if".into()),
                Rule::String("else".into()),
                Rule::String("while".into()),
            ])
        );
    }

    #[test]
    fn test_for_binding_count_mismatch() {
        let input = r#"
            grammar {
                language: "test",
            }

            rule bad {
                choice(
                    for (a: str, b: int) in [("x")] {
                        a
                    }
                )
            }
        "#;
        let err = parse_native_dsl(input, Path::new(".")).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("bindings"),
            "error should mention binding count mismatch: {msg}",
        );
    }

    #[test]
    fn test_json_roundtrip() {
        #[expect(clippy::needless_raw_string_hashes, reason = "false positive")]
        let input = r##"
            grammar {
                language: "test",
                extras: [regexp(r"\s")],
            }

            rule program { repeat(pair) }
            rule pair { seq(field(key, string), ":", field(value, string)) }
            rule string { regexp(r#""[^"]*""#) }
        "##;
        let grammar = parse_native_dsl(input, Path::new(".")).unwrap();
        let json_str = parse_native_dsl_to_json(input, Path::new(".")).unwrap();
        let reparsed = crate::parse_grammar::parse_grammar(&json_str).unwrap();
        assert_eq!(grammar.name, reparsed.name);
        assert_eq!(grammar.variables.len(), reparsed.variables.len());
        for (orig, re) in grammar.variables.iter().zip(reparsed.variables.iter()) {
            assert_eq!(orig.name, re.name);
        }
    }

    #[test]
    fn test_expression_fn_helper() {
        let input = r##"
            grammar {
                language: "test",
            }

            fn make_if(content: rule) -> rule {
                seq("#if", content, "#endif")
            }

            rule preproc_if { make_if(_statement) }
            rule preproc_in_block_if { make_if(_block_item) }

            rule _statement { "stmt" }
            rule _block_item { "item" }
        "##;
        let grammar = parse_native_dsl(input, Path::new(".")).unwrap();
        let names: Vec<&str> = grammar.variables.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "preproc_if",
                "preproc_in_block_if",
                "_statement",
                "_block_item"
            ]
        );
        assert_eq!(
            grammar.variables[0].rule,
            Rule::seq(vec![
                Rule::String("#if".into()),
                Rule::NamedSymbol("_statement".into()),
                Rule::String("#endif".into()),
            ])
        );
        assert_eq!(
            grammar.variables[1].rule,
            Rule::seq(vec![
                Rule::String("#if".into()),
                Rule::NamedSymbol("_block_item".into()),
                Rule::String("#endif".into()),
            ])
        );
    }

    #[test]
    fn test_type_check_fn_args() {
        let input = r#"
            grammar {
                language: "test",
            }

            fn needs_int(x: int) -> rule {
                prec(x, "a")
            }

            rule program {
                needs_int("not_an_int")
            }
        "#;
        let err = parse_native_dsl(input, Path::new(".")).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("expected int"), "error was: {msg}",);
    }

    #[test]
    fn test_prec_inside_token_immediate() {
        let source = read_native_grammar("json_native");
        let grammar_dir = native_grammar_dir("json_native");
        let g = parse_native_dsl(&source, &grammar_dir).unwrap();
        let string_content = g
            .variables
            .iter()
            .find(|v| v.name == "string_content")
            .unwrap();
        eprintln!("string_content rule: {:?}", string_content.rule);
        if let Rule::Metadata { params, .. } = &string_content.rule {
            assert_eq!(
                params.precedence,
                crate::rules::Precedence::Integer(1),
                "precedence lost in token.immediate(prec(1, ...))"
            );
            assert!(params.is_token);
            assert!(params.is_main_token);
        } else {
            panic!("expected Metadata, got {:?}", string_content.rule);
        }
    }

    fn test_grammars_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test/fixtures/test_grammars")
    }

    fn native_grammar_dir(name: &str) -> std::path::PathBuf {
        test_grammars_dir().join(name)
    }

    fn read_native_grammar(name: &str) -> String {
        let path = native_grammar_dir(name).join("grammar.tsg");
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
    }

    #[test]
    fn test_json_native_grammar() {
        let grammar_dir = native_grammar_dir("json_native");
        let native_source = read_native_grammar("json_native");
        let native_json = parse_native_dsl_to_json(&native_source, &grammar_dir).unwrap();

        let js_path = test_grammars_dir().join("../grammars/json/src/grammar.json");
        let js_json = std::fs::read_to_string(&js_path).unwrap();

        let native_grammar = crate::parse_grammar::parse_grammar(&native_json).unwrap();
        let js_grammar = crate::parse_grammar::parse_grammar(&js_json).unwrap();
        assert_eq!(native_grammar, js_grammar);
    }

    #[test]
    fn test_c_native_grammar() {
        let grammar_dir = native_grammar_dir("c_native");
        let native_source = read_native_grammar("c_native");
        let native_json = parse_native_dsl_to_json(&native_source, &grammar_dir).unwrap();
        let js_path = test_grammars_dir().join("../grammars/c/src/grammar.json");
        let js_json = std::fs::read_to_string(&js_path).unwrap();

        let native_grammar = crate::parse_grammar::parse_grammar(&native_json).unwrap();
        let js_grammar = crate::parse_grammar::parse_grammar(&js_json).unwrap();
        assert_eq!(native_grammar, js_grammar);
    }

    #[test]
    fn test_javascript_native_grammar() {
        let grammar_dir = native_grammar_dir("javascript_native");
        let native_source = read_native_grammar("javascript_native");
        let native_json = parse_native_dsl_to_json(&native_source, &grammar_dir).unwrap();
        let js_path = test_grammars_dir().join("../grammars/javascript/src/grammar.json");
        let js_json = std::fs::read_to_string(&js_path).unwrap();

        let native_grammar = crate::parse_grammar::parse_grammar(&native_json).unwrap();
        let js_grammar = crate::parse_grammar::parse_grammar(&js_json).unwrap();

        assert_eq!(native_grammar.name, js_grammar.name, "name mismatch");

        let native_rules: rustc_hash::FxHashMap<&str, &crate::rules::Rule> = native_grammar
            .variables
            .iter()
            .map(|v| (v.name.as_str(), &v.rule))
            .collect();
        let js_rules: rustc_hash::FxHashMap<&str, &crate::rules::Rule> = js_grammar
            .variables
            .iter()
            .map(|v| (v.name.as_str(), &v.rule))
            .collect();

        for (name, js_rule) in &js_rules {
            let native_rule = native_rules
                .get(name)
                .unwrap_or_else(|| panic!("missing rule '{name}' in native grammar"));
            assert_eq!(native_rule, js_rule, "rule mismatch for '{name}'");
        }
        for name in native_rules.keys() {
            assert!(
                js_rules.contains_key(name),
                "extra rule '{name}' in native grammar"
            );
        }
        assert_eq!(native_rules.len(), js_rules.len(), "rule count mismatch");

        assert_eq!(
            native_grammar.extra_symbols, js_grammar.extra_symbols,
            "extras mismatch"
        );
        assert_eq!(
            native_grammar.expected_conflicts, js_grammar.expected_conflicts,
            "conflicts mismatch"
        );
        assert_eq!(
            native_grammar.external_tokens, js_grammar.external_tokens,
            "externals mismatch"
        );
        assert_eq!(
            native_grammar.variables_to_inline, js_grammar.variables_to_inline,
            "inline mismatch"
        );
        assert_eq!(
            native_grammar.supertype_symbols, js_grammar.supertype_symbols,
            "supertypes mismatch"
        );
        assert_eq!(
            native_grammar.word_token, js_grammar.word_token,
            "word mismatch"
        );
        assert_eq!(
            native_grammar.precedence_orderings, js_grammar.precedence_orderings,
            "precedences mismatch"
        );
    }

    #[test]
    #[ignore = "benchmark"] // run with --ignored for benchmarking
    fn bench_native_dsl_c() {
        let grammar_dir = native_grammar_dir("c_native");
        let source = read_native_grammar("c_native");
        for _ in 0..10 {
            std::hint::black_box(parse_native_dsl(&source, &grammar_dir).unwrap());
        }
        let n = 1000;
        let start = std::time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(parse_native_dsl(&source, &grammar_dir).unwrap());
        }
        eprintln!("Native DSL C: {:?}/iter", start.elapsed() / n);
    }

    #[test]
    #[ignore = "benchmark"]
    fn bench_native_dsl_javascript() {
        let grammar_dir = native_grammar_dir("javascript_native");
        let source = read_native_grammar("javascript_native");
        for _ in 0..10 {
            std::hint::black_box(parse_native_dsl(&source, &grammar_dir).unwrap());
        }
        let n = 1000;
        let start = std::time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(parse_native_dsl(&source, &grammar_dir).unwrap());
        }
        eprintln!("Native DSL JavaScript: {:?}/iter", start.elapsed() / n);
    }

    #[test]
    #[ignore = "benchmark"]
    fn bench_json_parse_c() {
        let js_json =
            std::fs::read_to_string(test_grammars_dir().join("../grammars/c/src/grammar.json"))
                .unwrap();
        for _ in 0..10 {
            std::hint::black_box(crate::parse_grammar::parse_grammar(&js_json).unwrap());
        }
        let n = 1000;
        let start = std::time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(crate::parse_grammar::parse_grammar(&js_json).unwrap());
        }
        eprintln!("JSON parse_grammar C: {:?}/iter", start.elapsed() / n);
    }
}
