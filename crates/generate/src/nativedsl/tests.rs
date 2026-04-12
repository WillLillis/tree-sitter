use std::path::Path;

use crate::grammars::{InputGrammar, VariableType};
use crate::nativedsl::lexer::TokenKind;
use crate::rules::{Precedence, Rule};

use super::{
    DslError, LexErrorKind, LowerErrorKind, ParseErrorKind, ResolveErrorKind, Ty, TypeErrorKind,
    ast, parse_native_dsl,
};

macro_rules! assert_err {
    ($err:expr, $variant:ident) => {
        match $err {
            DslError::$variant(e) => e,
            other => panic!("expected {} error, got {other:?}", stringify!($variant)),
        }
    };
}

fn test_grammars_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test/fixtures/test_grammars")
}

fn dsl(input: &str) -> InputGrammar {
    parse_native_dsl(input, &test_grammars_dir().join("grammar.tsg")).unwrap()
}

fn dsl_err(input: &str) -> DslError {
    parse_native_dsl(input, &test_grammars_dir().join("grammar.tsg")).unwrap_err()
}

/// Follows the same path as the CLI: `parse_native_dsl` -> normalize -> `grammar_to_json`.
fn dsl_to_json(input: &str, path: &std::path::Path) -> (InputGrammar, String) {
    let mut grammar = parse_native_dsl(input, path).unwrap();
    crate::parse_grammar::normalize_grammar(&mut grammar);
    let json = serde_json::to_string_pretty(&super::lower::grammar_to_json(&grammar))
        .expect("grammar JSON serialization should not fail");
    (grammar, json)
}

fn native_grammar_path(name: &str) -> std::path::PathBuf {
    test_grammars_dir().join(name).join("grammar.tsg")
}

fn read_native_grammar(name: &str) -> String {
    let path = native_grammar_path(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

fn rule_names(g: &InputGrammar) -> Vec<&str> {
    g.variables.iter().map(|v| v.name.as_str()).collect()
}

fn find_rule<'a>(g: &'a InputGrammar, name: &str) -> &'a Rule {
    &g.variables
        .iter()
        .find(|v| v.name == name)
        .unwrap_or_else(|| panic!("rule '{name}' not found"))
        .rule
}

fn assert_grammar_eq(native: &InputGrammar, js: &InputGrammar) {
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

// ===== Node size =====

#[test]
fn node_enum_is_16_bytes() {
    // 4 nodes per 64-byte cache line
    assert_eq!(std::mem::size_of::<ast::Node>(), 16);
}

// ===== Minimal grammar =====

#[test]
fn minimal_grammar() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { repeat("hello") }
    "#);
    assert_eq!(g.name, "test");
    assert_eq!(g.variables.len(), 1);
    assert_eq!(g.variables[0].name, "program");
    assert_eq!(g.variables[0].kind, VariableType::Named);
    assert_eq!(
        g.variables[0].rule,
        Rule::choice(vec![
            Rule::repeat(Rule::String("hello".into())),
            Rule::Blank
        ])
    );
}

// ===== Combinators =====

#[test]
fn seq_and_choice() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { seq(choice("a", "b"), "c") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::seq(vec![
            Rule::choice(vec![Rule::String("a".into()), Rule::String("b".into())]),
            Rule::String("c".into()),
        ])
    );
}

#[test]
fn repeat1() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { repeat1("x") }
    "#);
    assert_eq!(g.variables[0].rule, Rule::repeat(Rule::String("x".into())));
}

#[test]
fn optional_combinator() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { optional("x") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::choice(vec![Rule::String("x".into()), Rule::Blank])
    );
}

#[test]
fn blank_combinator() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { blank() }
    "#);
    assert_eq!(g.variables[0].rule, Rule::Blank);
}

#[test]
fn token_combinator() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { token(seq("a", "b")) }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::token(Rule::seq(vec![
            Rule::String("a".into()),
            Rule::String("b".into()),
        ]))
    );
}

#[test]
fn token_immediate_combinator() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { token_immediate("x") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::immediate_token(Rule::String("x".into()))
    );
}

#[test]
fn field_combinator() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { field(name, "x") }
        rule name { "n" }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::field("name".to_string(), Rule::String("x".into()))
    );
}

#[test]
fn alias_with_string() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { alias(identifier, "id") }
        rule identifier { regexp("[a-z]+") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::alias(
            Rule::NamedSymbol("identifier".into()),
            "id".to_string(),
            false
        )
    );
}

#[test]
fn regexp_combinator() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { regexp("[a-z]+") }
    "#);
    assert!(matches!(&g.variables[0].rule, Rule::Pattern(p, _) if p == "[a-z]+"));
}

#[test]
fn concat_combinator() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { regexp(concat("[", "a-z", "]+")) }
    "#);
    assert!(matches!(&g.variables[0].rule, Rule::Pattern(p, _) if p == "[a-z]+"));
}

// ===== Prec variants =====

#[test]
fn prec_default() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { prec(1, "x") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::prec(Precedence::Integer(1), Rule::String("x".into()))
    );
}

#[test]
fn prec_left() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { prec_left(2, "x") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::prec_left(Precedence::Integer(2), Rule::String("x".into()))
    );
}

#[test]
fn prec_right() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { prec_right(3, "x") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::prec_right(Precedence::Integer(3), Rule::String("x".into()))
    );
}

#[test]
fn prec_dynamic() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { prec_dynamic(4, "x") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::prec_dynamic(4, Rule::String("x".into()))
    );
}

#[test]
fn prec_with_string_name() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { prec_left("assign", "x") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::prec_left(Precedence::Name("assign".into()), Rule::String("x".into()))
    );
}

// ===== Let bindings =====

#[test]
fn let_binding_int() {
    let g = dsl(r#"
        grammar { language: "test" }
        let P: int = 5
        rule program { prec(P, "x") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::prec(Precedence::Integer(5), Rule::String("x".into()))
    );
}

#[test]
fn let_binding_object_field_access() {
    let g = dsl(r#"
        grammar { language: "test" }
        let PREC = { ADD: 1, MUL: 2 }
        rule add { prec_left(PREC.ADD, seq(expr, "+", expr)) }
        rule mul { prec_left(PREC.MUL, seq(expr, "*", expr)) }
        rule expr { choice(add, mul) }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::prec_left(
            Precedence::Integer(1),
            Rule::seq(vec![
                Rule::NamedSymbol("expr".into()),
                Rule::String("+".into()),
                Rule::NamedSymbol("expr".into()),
            ])
        )
    );
}

#[test]
fn let_binding_negative_int() {
    let g = dsl(r#"
        grammar { language: "test" }
        let P: int = -1
        rule program { prec(P, "x") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::prec(Precedence::Integer(-1), Rule::String("x".into()))
    );
}

// ===== Functions =====

#[test]
fn function_expansion() {
    let g = dsl(r#"
        grammar { language: "test" }
        fn comma_sep(item: rule) -> rule {
            seq(item, repeat(seq(",", item)))
        }
        rule program { comma_sep(identifier) }
        rule identifier { regexp("[a-z]+") }
    "#);
    assert_eq!(g.variables.len(), 2);
    assert_eq!(
        g.variables[0].rule,
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
fn function_multiple_calls() {
    let g = dsl(r##"
        grammar { language: "test" }
        fn make_if(content: rule) -> rule { seq("#if", content, "#endif") }
        rule preproc_if { make_if(_statement) }
        rule preproc_block_if { make_if(_block_item) }
        rule _statement { "stmt" }
        rule _block_item { "item" }
    "##);
    assert_eq!(
        g.variables[0].rule,
        Rule::seq(vec![
            Rule::String("#if".into()),
            Rule::NamedSymbol("_statement".into()),
            Rule::String("#endif".into()),
        ])
    );
    assert_eq!(
        g.variables[1].rule,
        Rule::seq(vec![
            Rule::String("#if".into()),
            Rule::NamedSymbol("_block_item".into()),
            Rule::String("#endif".into()),
        ])
    );
}

// ===== For expressions =====

#[test]
fn for_tuple_destructure() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule binary {
            choice(
                for (op: str, p: int) in [("+", 1), ("*", 2)] {
                    prec_left(p, seq(expr, op, expr))
                }
            )
        }
        rule expr { "x" }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::choice(vec![
            Rule::prec_left(
                Precedence::Integer(1),
                Rule::seq(vec![
                    Rule::NamedSymbol("expr".into()),
                    Rule::String("+".into()),
                    Rule::NamedSymbol("expr".into()),
                ])
            ),
            Rule::prec_left(
                Precedence::Integer(2),
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
fn for_single_binding() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule keywords {
            choice(for (kw: str) in ["if", "else", "while"] { kw })
        }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::choice(vec![
            Rule::String("if".into()),
            Rule::String("else".into()),
            Rule::String("while".into()),
        ])
    );
}

#[test]
fn for_single_element_tuple_destructure() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule keywords {
            choice(for (kw: str) in [("if"), ("else"), ("while")] { kw })
        }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::choice(vec![
            Rule::String("if".into()),
            Rule::String("else".into()),
            Rule::String("while".into()),
        ])
    );
}

// ===== Grammar config =====

#[test]
fn grammar_config_all_fields() {
    let g = dsl(r#"
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
    "#);
    assert_eq!(g.extra_symbols.len(), 2);
    assert_eq!(g.external_tokens.len(), 1);
    assert_eq!(g.supertype_symbols, vec!["_expression"]);
    assert_eq!(g.variables_to_inline, vec!["_statement"]);
    assert_eq!(
        g.expected_conflicts,
        vec![vec!["primary".to_string(), "arrow".to_string()]]
    );
    assert_eq!(g.word_token, Some("identifier".into()));
}

// ===== Raw strings =====

#[test]
fn raw_string_literal() {
    #[expect(clippy::needless_raw_string_hashes, reason = "false positive")]
    let g = dsl(r##"
        grammar { language: "test" }
        rule program { regexp(r#""[^"]*""#) }
    "##);
    assert!(matches!(&g.variables[0].rule, Rule::Pattern(p, _) if p == r#""[^"]*""#));
}

// ===== String escapes =====

#[test]
fn string_with_escapes() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { "\n\t\\" }
    "#);
    assert_eq!(g.variables[0].rule, Rule::String("\n\t\\".into()));
}

// ===== JSON roundtrip =====

#[test]
fn json_roundtrip() {
    #[expect(clippy::needless_raw_string_hashes, reason = "false positive")]
    let input = r##"
        grammar { language: "test", extras: [regexp(r"\s")] }
        rule program { repeat(pair) }
        rule pair { seq(field(key, string), ":", field(value, string)) }
        rule string { regexp(r#""[^"]*""#) }
    "##;
    let path = test_grammars_dir().join("grammar.tsg");
    let (grammar, json_str) = dsl_to_json(input, &path);
    let reparsed = crate::parse_grammar::parse_grammar(&json_str).unwrap();
    assert_eq!(grammar.name, reparsed.name);
    assert_eq!(grammar.variables.len(), reparsed.variables.len());
    for (orig, re) in grammar.variables.iter().zip(reparsed.variables.iter()) {
        assert_eq!(orig.name, re.name);
    }
}

// ===== Additional combinator coverage =====

#[test]
fn alias_with_named_rule() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { alias(_impl, block) }
        rule _impl { "impl" }
        rule block { "{}" }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::alias(Rule::NamedSymbol("_impl".into()), "block".to_string(), true)
    );
}

#[test]
fn reserved_combinator() {
    let g = dsl(r#"
        grammar {
            language: "test",
            reserved: { default: [identifier] },
        }
        rule program { reserved("default", identifier) }
        rule identifier { regexp("[a-z]+") }
    "#);
    assert!(matches!(
        &g.variables[0].rule,
        Rule::Reserved { context_name, .. } if context_name == "default"
    ));
}

#[test]
fn reserved_multiple_sets() {
    let g = dsl(r#"
        grammar {
            language: "test",
            reserved: {
                global: ["if", "else", "for"],
                properties: ["get", "set"],
            },
        }
        rule program { reserved("global", regexp("[a-z]+")) }
    "#);
    assert_eq!(g.reserved_words.len(), 2);
    assert_eq!(g.reserved_words[0].name, "global");
    assert_eq!(g.reserved_words[0].reserved_words.len(), 3);
    assert_eq!(g.reserved_words[1].name, "properties");
    assert_eq!(g.reserved_words[1].reserved_words.len(), 2);
}

#[test]
fn reserved_inherited() {
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar_with_reserved.tsg")
        grammar { language: "derived", inherits: base, reserved: base.reserved }
    "#);
    assert_eq!(g.reserved_words.len(), 2);
    assert_eq!(g.reserved_words[0].name, "global");
    assert_eq!(
        g.reserved_words[0].reserved_words,
        vec![
            Rule::String("if".into()),
            Rule::String("else".into()),
            Rule::String("for".into()),
        ]
    );
    assert_eq!(g.reserved_words[1].name, "properties");
    assert_eq!(
        g.reserved_words[1].reserved_words,
        vec![Rule::String("get".into()), Rule::String("set".into()),]
    );
}

#[test]
fn reserved_empty_inherited() {
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base, reserved: base.reserved }
    "#);
    assert!(g.reserved_words.is_empty());
}

#[test]
fn for_inline_in_seq() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program {
            seq(
                "start",
                for (kw: str) in ["a", "b"] { kw },
                "end",
            )
        }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::seq(vec![
            Rule::String("start".into()),
            Rule::String("a".into()),
            Rule::String("b".into()),
            Rule::String("end".into()),
        ])
    );
}

#[test]
fn regexp_with_flags() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { regexp("[a-z]+", "i") }
    "#);
    assert!(matches!(
        &g.variables[0].rule,
        Rule::Pattern(p, f) if p == "[a-z]+" && f == "i"
    ));
}

#[test]
fn rule_reference_in_body() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { other }
        rule other { "x" }
    "#);
    assert_eq!(g.variables[0].rule, Rule::NamedSymbol("other".into()));
}

#[test]
fn let_binding_typed_str() {
    let g = dsl(r#"
        grammar { language: "test" }
        let SEP: str = ","
        rule program { seq("a", SEP, "b") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::seq(vec![
            Rule::String("a".into()),
            Rule::String(",".into()),
            Rule::String("b".into()),
        ])
    );
}

#[test]
fn function_multi_params() {
    let g = dsl(r#"
        grammar { language: "test" }
        fn wrap(before: str, content: rule, after: str) -> rule {
            seq(before, content, after)
        }
        rule program { wrap("(", identifier, ")") }
        rule identifier { regexp("[a-z]+") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::seq(vec![
            Rule::String("(".into()),
            Rule::NamedSymbol("identifier".into()),
            Rule::String(")".into()),
        ])
    );
}

#[test]
fn nested_function_calls() {
    let g = dsl(r#"
        grammar { language: "test" }
        fn comma_sep1(item: rule) -> rule { seq(item, repeat(seq(",", item))) }
        fn comma_sep(item: rule) -> rule { optional(comma_sep1(item)) }
        rule program { comma_sep(identifier) }
        rule identifier { regexp("[a-z]+") }
    "#);
    // comma_sep calls comma_sep1 internally
    assert_eq!(g.variables.len(), 2);
    // Just verify it doesn't error - the structure is complex
    assert!(matches!(&g.variables[0].rule, Rule::Choice(_)));
}

#[test]
fn config_precedences() {
    let g = dsl(r#"
        grammar {
            language: "test",
            precedences: [["add", multiply]],
        }
        rule program { "x" }
        rule multiply { "y" }
    "#);
    assert_eq!(g.precedence_orderings.len(), 1);
    assert_eq!(g.precedence_orderings[0].len(), 2);
}

// ===== Error cases =====

#[test]
fn error_type_mismatch_fn_args() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        fn needs_int(x: int) -> rule { prec(x, "a") }
        rule program { needs_int("not_an_int") }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::TypeMismatch {
            expected: Ty::Int,
            got: Ty::Str
        }
    );
}

#[test]
fn error_for_binding_count_mismatch() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        rule bad { choice(for (a: str, b: int) in [("x")] { a }) }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::ForBindingCountMismatch {
            bindings: 2,
            tuple_elements: 1
        }
    );
}

#[test]
fn error_unknown_identifier() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        rule program { nonexistent }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert_eq!(
        e.kind,
        ResolveErrorKind::UnknownIdentifier("nonexistent".into())
    );
}

#[test]
fn error_duplicate_rule() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        rule program { "a" }
        rule program { "b" }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert_eq!(
        e.kind,
        ResolveErrorKind::DuplicateDeclaration("program".into())
    );
}

#[test]
fn error_forward_reference_let() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        rule program { MY_VAR }
        let MY_VAR: str = "x"
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert_eq!(e.kind, ResolveErrorKind::UnknownIdentifier("MY_VAR".into()));
}

#[test]
fn error_unknown_grammar_field() {
    let err = dsl_err(
        r#"
        grammar { language: "test", bogus: "x" }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::UnknownGrammarField("bogus".into()));
}

// ===== Prec inside token_immediate =====

#[test]
fn prec_inside_token_immediate() {
    let source = read_native_grammar("json_native");
    let g = parse_native_dsl(&source, &native_grammar_path("json_native")).unwrap();
    let string_content = g
        .variables
        .iter()
        .find(|v| v.name == "string_content")
        .unwrap();
    if let Rule::Metadata { params, .. } = &string_content.rule {
        assert_eq!(params.precedence, Precedence::Integer(1));
        assert!(params.is_token);
        assert!(params.is_main_token);
    } else {
        panic!("expected Metadata, got {:?}", string_content.rule);
    }
}

// ===== Full grammar roundtrips =====

#[test]
fn json_native_grammar_roundtrip() {
    let dir = native_grammar_path("json_native");
    let (_, native_json) = dsl_to_json(&read_native_grammar("json_native"), &dir);
    let js_json =
        std::fs::read_to_string(test_grammars_dir().join("../grammars/json/src/grammar.json"))
            .unwrap();
    let native = crate::parse_grammar::parse_grammar(&native_json).unwrap();
    let js = crate::parse_grammar::parse_grammar(&js_json).unwrap();
    assert_eq!(native, js);
}

#[test]
fn c_native_grammar_roundtrip() {
    let dir = native_grammar_path("c_native");
    let (_, native_json) = dsl_to_json(&read_native_grammar("c_native"), &dir);
    let js_json =
        std::fs::read_to_string(test_grammars_dir().join("../grammars/c/src/grammar.json"))
            .unwrap();
    let native = crate::parse_grammar::parse_grammar(&native_json).unwrap();
    let js = crate::parse_grammar::parse_grammar(&js_json).unwrap();
    assert_eq!(native, js);
}

#[test]
fn javascript_native_grammar_roundtrip() {
    let dir = native_grammar_path("javascript_native");
    let (_, native_json) = dsl_to_json(&read_native_grammar("javascript_native"), &dir);
    let js_json = std::fs::read_to_string(
        test_grammars_dir().join("../grammars/javascript/src/grammar.json"),
    )
    .unwrap();
    let native = crate::parse_grammar::parse_grammar(&native_json).unwrap();
    let js = crate::parse_grammar::parse_grammar(&js_json).unwrap();
    assert_grammar_eq(&native, &js);
}

// ===== C++ grammar roundtrip =====

#[test]
fn cpp_native_grammar_roundtrip() {
    let dir = native_grammar_path("cpp_native");
    let (_, native_json) = dsl_to_json(&read_native_grammar("cpp_native"), &dir);
    let js_json =
        std::fs::read_to_string(test_grammars_dir().join("../grammars/cpp/src/grammar.json"))
            .unwrap();
    let native = crate::parse_grammar::parse_grammar(&native_json).unwrap();
    let js = crate::parse_grammar::parse_grammar(&js_json).unwrap();
    assert_grammar_eq(&native, &js);
}

/// Verify the direct path (no JSON roundtrip) works through `prepare_grammar`.
/// This is the same path the CLI takes via `GrammarSource::Grammar`.
#[test]
fn cpp_direct_prepare_grammar() {
    let dir = native_grammar_path("cpp_native");
    let mut grammar = parse_native_dsl(&read_native_grammar("cpp_native"), &dir).unwrap();
    crate::parse_grammar::normalize_grammar(&mut grammar);
    crate::prepare_grammar::prepare_grammar(&grammar).unwrap();
}

// ===== Inheritance =====

#[test]
fn inherit_all_rules() {
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
    "#);
    assert_eq!(g.name, "derived");
    assert_eq!(
        rule_names(&g),
        vec![
            "program",
            "statement",
            "expression",
            "identifier",
            "_inline_rule"
        ]
    );
}

#[test]
fn inherit_config() {
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
    "#);
    assert_eq!(g.word_token.as_deref(), Some("identifier"));
    assert_eq!(g.variables_to_inline, vec!["_inline_rule"]);
    assert_eq!(g.extra_symbols.len(), 1);
}

#[test]
fn inherit_config_append_inline() {
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar {
            language: "derived",
            inherits: base,
            inline: append(base.inline, [_extra_inline]),
        }
        rule _extra_inline { "extra" }
    "#);
    assert_eq!(g.variables_to_inline, vec!["_inline_rule", "_extra_inline"]);
}

#[test]
fn inherit_config_append_extras() {
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar {
            language: "derived",
            inherits: base,
            extras: append(base.extras, [comment]),
        }
        rule comment { regexp(r"//.*") }
    "#);
    assert_eq!(g.extra_symbols.len(), 2);
}

#[test]
fn config_expr_let_binding() {
    let g = dsl(r#"
        let my_extras: list<rule> = [regexp(r"\s"), comment]
        grammar {
            language: "test",
            extras: my_extras,
        }
        rule program { "x" }
        rule comment { regexp(r"//.*") }
    "#);
    assert_eq!(g.extra_symbols.len(), 2);
}

#[test]
fn config_expr_word() {
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base, word: base.word }
    "#);
    assert_eq!(g.word_token.as_deref(), Some("identifier"));
}

#[test]
fn override_rule_replaces_body() {
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
        override rule expression { choice(identifier, "42") }
    "#);
    assert_eq!(
        *find_rule(&g, "expression"),
        Rule::choice(vec![
            Rule::NamedSymbol("identifier".into()),
            Rule::String("42".into()),
        ])
    );
    assert_eq!(
        *find_rule(&g, "_inline_rule"),
        Rule::String("inline".into())
    );
}

#[test]
fn override_preserves_rule_order() {
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
        override rule statement { "overridden" }
    "#);
    assert_eq!(g.variables[1].name, "statement");
    assert_eq!(g.variables[1].rule, Rule::String("overridden".into()));
}

#[test]
fn new_rules_appended() {
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
        rule new_rule { "hello" }
    "#);
    assert_eq!(g.variables.last().unwrap().name, "new_rule");
    assert_eq!(
        g.variables.last().unwrap().rule,
        Rule::String("hello".into())
    );
    assert_eq!(g.variables.len(), 6);
}

#[test]
fn override_and_new_combined() {
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
        override rule expression { choice(identifier, number) }
        rule number { regexp("[0-9]+") }
    "#);
    assert_eq!(g.variables.len(), 6);
    assert_eq!(
        *find_rule(&g, "expression"),
        Rule::choice(vec![
            Rule::NamedSymbol("identifier".into()),
            Rule::NamedSymbol("number".into()),
        ])
    );
}

#[test]
fn config_override_replaces() {
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar {
            language: "derived",
            inherits: base,
            extras: [regexp(r"\s"), regexp(r"//.*")],
        }
    "#);
    assert_eq!(g.extra_symbols.len(), 2);
}

#[test]
fn config_word_inherited() {
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
    "#);
    assert_eq!(g.word_token.as_deref(), Some("identifier"));
}

#[test]
fn config_word_overridden() {
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base, word: expression }
    "#);
    assert_eq!(g.word_token.as_deref(), Some("expression"));
}

#[test]
fn rule_inline_expands_base_rule_body() {
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
        override rule expression { choice(base::expression, "extended") }
    "#);
    // c::rule_name inlines the body of the base rule (which is `identifier`,
    // i.e. NamedSymbol("identifier"))
    assert_eq!(
        *find_rule(&g, "expression"),
        Rule::choice(vec![
            Rule::NamedSymbol("identifier".into()),
            Rule::String("extended".into()),
        ])
    );
}

#[test]
fn config_access_extras() {
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base, extras: base.extras }
        override rule program { "x" }
    "#);
    assert_eq!(g.name, "derived");
    // base has extras: [regexp("\\s")], verify it was inherited
    assert_eq!(g.extra_symbols.len(), 1);
    assert_eq!(
        g.extra_symbols[0],
        Rule::Pattern("\\s".into(), String::new())
    );
}

#[test]
fn inherit_from_json() {
    let g = dsl(r#"
        let base = inherit("../grammars/json/src/grammar.json")
        grammar { language: "json_extended", inherits: base }
        rule new_type { "null" }
    "#);
    assert_eq!(g.name, "json_extended");
    assert!(g.variables.iter().any(|v| v.name == "document"));
    assert!(g.variables.iter().any(|v| v.name == "new_type"));
}

#[test]
fn append_concatenates_lists() {
    let g = dsl(r#"
        grammar { language: "test" }
        let a: list<str> = ["x", "y"]
        let b: list<str> = ["z"]
        let c: list<str> = append(a, b)
        rule program { choice(for (s: str) in c { s }) }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::choice(vec![
            Rule::String("x".into()),
            Rule::String("y".into()),
            Rule::String("z".into()),
        ])
    );
}

// ===== Inheritance errors =====

#[test]
fn error_override_without_inherit() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        override rule foo { "bar" }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert_eq!(e.kind, LowerErrorKind::OverrideWithoutInherit);
}

#[test]
fn error_override_rule_not_found() {
    let err = dsl_err(
        r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
        override rule nonexistent { "oops" }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert_eq!(
        e.kind,
        LowerErrorKind::OverrideRuleNotFound("nonexistent".into())
    );
}

#[test]
fn error_inherit_child_error() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.tsg");
    std::fs::write(
        &base_path,
        r#"grammar { language: "base" }
rule program { bogus_ref }
"#,
    )
    .unwrap();

    let parent_path = dir.path().join("parent.tsg");
    let parent_src = r#"let base = inherit("base.tsg")
grammar { language: "derived", inherits: base }
rule extra { "hello" }
"#
    .to_string();
    let err = parse_native_dsl(&parent_src, &parent_path).unwrap_err();

    let DslError::Inherited(inherited) = &err else {
        panic!("expected Inherited error, got {err:?}")
    };
    // Inner error is a resolve error from the base grammar
    let DslError::Resolve(resolve_err) = inherited.inner.as_ref() else {
        panic!("expected Resolve error, got {:?}", inherited.inner)
    };
    assert_eq!(
        resolve_err.kind,
        ResolveErrorKind::UnknownIdentifier("bogus_ref".to_string())
    );
    // InheritedError carries the base file's context
    assert_eq!(inherited.path, base_path);
    assert!(inherited.source_text.contains("bogus_ref"));
    // inherit_span points to the base.tsg path in the parent (quotes stripped)
    assert_eq!(
        &parent_src[inherited.inherit_span.start as usize..inherited.inherit_span.end as usize],
        "base.tsg"
    );
}

#[test]
fn error_inherit_bad_path() {
    let err = dsl_err(
        r#"
        let base = inherit("nonexistent/grammar.tsg")
        grammar { language: "derived", inherits: base }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert!(
        matches!(e.kind, LowerErrorKind::InheritLoadError(ref msg) if msg.contains("nonexistent"))
    );
}

#[test]
fn error_inherit_bad_extension() {
    let dir = tempfile::tempdir().unwrap();
    let bad_file = dir.path().join("grammar.txt");
    std::fs::write(&bad_file, "not a grammar").unwrap();

    let input = format!(
        r#"
        let base = inherit("{}")
        grammar {{ language: "derived", inherits: base }}
    "#,
        bad_file.display()
    );
    let err = parse_native_dsl(&input, Path::new(".")).unwrap_err();
    let e = assert_err!(err, Lower);
    assert!(
        matches!(e.kind, LowerErrorKind::InheritLoadError(ref msg) if msg.contains("unsupported file extension"))
    );
}

#[test]
fn error_inherit_cycle() {
    let dir = tempfile::tempdir().unwrap();
    let a_path = dir.path().join("a.tsg");
    let b_path = dir.path().join("b.tsg");

    std::fs::write(
        &a_path,
        format!(
            r#"
        let base = inherit("{}")
        grammar {{ language: "a", inherits: base }}
    "#,
            b_path.display()
        ),
    )
    .unwrap();

    std::fs::write(
        &b_path,
        format!(
            r#"
        let base = inherit("{}")
        grammar {{ language: "b", inherits: base }}
    "#,
            a_path.display()
        ),
    )
    .unwrap();

    let source = std::fs::read_to_string(&a_path).unwrap();
    let err = parse_native_dsl(&source, dir.path()).unwrap_err();
    // a inherits b, b inherits a -> cycle detected when a is loaded recursively from b
    let outer = assert_err!(err, Inherited);
    let DslError::Inherited(inner) = outer.inner.as_ref() else {
        panic!("expected nested Inherited error, got {:?}", outer.inner)
    };
    let DslError::Lower(e) = inner.inner.as_ref() else {
        panic!("expected Lower error, got {:?}", inner.inner)
    };
    let LowerErrorKind::InheritCycle(chain) = &e.kind else {
        panic!("expected InheritCycle, got {:?}", e.kind)
    };
    let filenames: Vec<_> = chain
        .iter()
        .map(|p| p.file_name().unwrap().to_str().unwrap())
        .collect();
    assert_eq!(filenames, ["b.tsg", "a.tsg", "b.tsg"]);
}

#[test]
fn error_append_mismatched_types() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let a: list<str> = ["x"]
        let b: list<int> = [1]
        let c = append(a, b)
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::TypeMismatch {
            expected: Ty::List(Box::new(Ty::Str)),
            got: Ty::List(Box::new(Ty::Int)),
        }
    );
}

#[test]
fn error_append_non_list() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let a: str = "x"
        let b: list<str> = ["y"]
        let c = append(a, b)
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::AppendRequiresList(Ty::Str));
}

// ===== More error cases =====

#[test]
fn error_missing_grammar_block() {
    let err = dsl_err(
        r#"
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert_eq!(e.kind, LowerErrorKind::MissingGrammarBlock);
}

#[test]
fn error_missing_language_field() {
    let err = dsl_err(
        r#"
        grammar { extras: [] }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert_eq!(e.kind, LowerErrorKind::MissingLanguageField);
}

#[test]
fn error_let_type_annotation_mismatch() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let X: int = "not_an_int"
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::TypeMismatch {
            expected: Ty::Int,
            got: Ty::Str
        }
    );
}

#[test]
fn error_undefined_function() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        rule program { nonexistent_fn(identifier) }
        rule identifier { "x" }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert_eq!(
        e.kind,
        ResolveErrorKind::UnknownIdentifier("nonexistent_fn".into())
    );
}

#[test]
fn error_rule_called_as_function() {
    let err = dsl_err(
        r#"
        grammar { language: "test", word: ident }
        rule program { ident("default", regexp("[a-z]+")) }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::UndefinedFunction("ident".into()));
}

#[test]
fn error_self_referential_let() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let P = { a: P.a }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::UnresolvedVariable("P".into()));
}

#[test]
fn error_wrong_arg_count() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        fn one_arg(x: rule) -> rule { x }
        rule program { one_arg("a", "b") }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::ArgCountMismatch {
            fn_name: "one_arg".into(),
            expected: 1,
            got: 2,
        }
    );
}

#[test]
fn error_rule_inline_on_non_grammar() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let obj = { x: 1 }
        rule program { obj::x }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::ConfigAccessOnNonGrammar(Ty::Object(vec![("x".into(), Ty::Int)]))
    );
}

#[test]
fn error_field_access_on_non_object() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x: int = 5
        rule program { prec(x.foo, "a") }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::FieldAccessOnNonObject(Ty::Int));
}

#[test]
fn error_field_not_found() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x = { a: 1 }
        rule program { prec(x.b, "a") }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::FieldNotFound {
            field: "b".into(),
            on_type: Ty::Object(vec![("a".into(), Ty::Int)]),
        }
    );
}

#[test]
fn error_word_non_rule_var() {
    let err = dsl_err(
        r#"
        let my_word = 42
        grammar { language: "test", word: my_word }
        rule program { "y" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::TypeMismatch {
            expected: Ty::Rule,
            got: Ty::Int,
        }
    );
}

#[test]
fn error_config_access_unknown_field() {
    let err = dsl_err(
        r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
        let x = base.bogus
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::UnknownConfigField("bogus".into()));
}

#[test]
fn error_duplicate_let() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let X: int = 1
        let X: int = 2
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert_eq!(e.kind, ResolveErrorKind::DuplicateDeclaration("X".into()));
}

#[test]
fn error_duplicate_fn() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        fn f(x: rule) -> rule { x }
        fn f(x: rule) -> rule { x }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert_eq!(e.kind, ResolveErrorKind::DuplicateDeclaration("f".into()));
}

#[test]
fn error_for_requires_list() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x: int = 5
        rule program { choice(for (v: int) in x { prec(v, "a") }) }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::ForRequiresList(Ty::Int));
}

#[test]
fn error_rule_body_not_rule_typed() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        rule program { 42 }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::TypeMismatch {
            expected: Ty::Rule,
            got: Ty::Int
        }
    );
}

#[test]
fn error_fn_return_type_mismatch() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        fn bad(x: rule) -> int { x }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::TypeMismatch {
            expected: Ty::Int,
            got: Ty::Rule
        }
    );
}

#[test]
fn error_prec_value_not_int_or_str() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        rule program { prec(identifier, "x") }
        rule identifier { "id" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::TypeMismatch {
            expected: Ty::Int,
            got: Ty::Rule
        }
    );
}

#[test]
fn error_list_inconsistent_types() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x = ["a", 1]
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::ListElementTypeMismatch {
            first: Ty::Str,
            got: Ty::Int
        }
    );
}

#[test]
fn error_multiple_grammar_blocks() {
    let err = dsl_err(
        r#"
        grammar { language: "first" }
        grammar { language: "second" }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::DuplicateGrammarBlock);
}

#[test]
fn multiple_inherit_bindings_rejected() {
    let err = dsl_err(
        r#"
        let a = inherit("inherit_base/grammar.tsg")
        let b = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: a }
        rule extra { "x" }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert_eq!(e.kind, LowerErrorKind::MultipleInherits);
}

// ===== Lexer error tests =====

#[test]
fn error_unterminated_string() {
    let err = dsl_err(r#"grammar { language: "test }rule program { "x" }"#);
    let e = assert_err!(err, Lex);
    assert_eq!(e.kind, LexErrorKind::UnterminatedString);
}

#[test]
fn error_newline_in_string() {
    let err = dsl_err("grammar { language: \"test\n\" }");
    let e = assert_err!(err, Lex);
    assert_eq!(e.kind, LexErrorKind::NewlineInString);
}

#[test]
fn error_invalid_escape() {
    let err = dsl_err(r#"grammar { language: "te\qst" } rule program { "x" }"#);
    let e = assert_err!(err, Lex);
    assert_eq!(e.kind, LexErrorKind::InvalidEscape('q'));
}

#[test]
fn error_unterminated_escape() {
    let err = dsl_err(r#"grammar { language: "test\"#);
    let e = assert_err!(err, Lex);
    assert_eq!(e.kind, LexErrorKind::UnterminatedEscape);
}

#[test]
fn error_unterminated_raw_string() {
    let err = dsl_err(r#"grammar { language: r#"test } rule program { "x" }"#);
    let e = assert_err!(err, Lex);
    assert_eq!(e.kind, LexErrorKind::UnterminatedRawString);
}

#[test]
fn error_expected_raw_string_quote() {
    let err = dsl_err(r#"grammar { language: r## } rule program { "x" }"#);
    let e = assert_err!(err, Lex);
    assert_eq!(e.kind, LexErrorKind::ExpectedRawStringQuote);
}

#[test]
fn error_integer_overflow() {
    let err = dsl_err(r#"grammar { language: "test" } rule program { prec(99999999999, "x") }"#);
    let e = assert_err!(err, Lex);
    assert_eq!(e.kind, LexErrorKind::IntegerOverflow);
}

// ===== Parser error tests =====

#[test]
fn error_expected_token() {
    let err = dsl_err(r#"grammar { language: "test" } rule program { seq("a" "b") }"#);
    let e = assert_err!(err, Parse);
    assert_eq!(
        e.kind,
        ParseErrorKind::ExpectedToken {
            expected: TokenKind::RParen,
            got: TokenKind::StringLit
        }
    );
}

#[test]
fn error_expected_expression() {
    let err = dsl_err(r#"grammar { language: "test" } rule program { seq(,) }"#);
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::ExpectedExpression);
}

#[test]
fn error_expected_item() {
    let err = dsl_err(r#"grammar { language: "test" } "stray_string""#);
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::ExpectedItem);
}

#[test]
fn error_unknown_type() {
    let err = dsl_err(r#"grammar { language: "test" } let x: foo = "y" rule program { "x" }"#);
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::UnknownType("foo".into()));
}

#[test]
fn error_missing_return_type() {
    let err = dsl_err(r#"grammar { language: "test" } fn f(x: rule) { x } rule program { "x" }"#);
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::MissingReturnType);
}

#[test]
fn error_builtin_arg_count_messages() {
    macro_rules! g {
        ($expr:expr) => {
            concat!(r#"grammar { language: "test" } rule foo { "#, $expr, " }")
        };
    }
    let cases: &[(&str, &str, u8, usize)] = &[
        // 0-arg: blank
        (g!("blank(\"a\")"), "blank", 0, 1),
        (g!("blank(\"a\", \"b\")"), "blank", 0, 2),
        // 1-arg: too few
        (g!("repeat()"), "repeat", 1, 0),
        (g!("repeat1()"), "repeat1", 1, 0),
        (g!("optional()"), "optional", 1, 0),
        (g!("token()"), "token", 1, 0),
        (g!("token_immediate()"), "token_immediate", 1, 0),
        (g!("inherit()"), "inherit", 1, 0),
        // 1-arg: too many
        (g!(r#"repeat("a", "b")"#), "repeat", 1, 2),
        (g!(r#"repeat1("a", "b")"#), "repeat1", 1, 2),
        (g!(r#"optional("a", "b")"#), "optional", 1, 2),
        (g!(r#"token("a", "b")"#), "token", 1, 2),
        (g!(r#"token_immediate("a", "b")"#), "token_immediate", 1, 2),
        (g!(r#"inherit("a", "b")"#), "inherit", 1, 2),
        // 2-arg: too few
        (g!("prec()"), "prec", 2, 0),
        (g!("prec(1)"), "prec", 2, 1),
        (g!("prec_left()"), "prec_left", 2, 0),
        (g!("prec_left(1)"), "prec_left", 2, 1),
        (g!("prec_right()"), "prec_right", 2, 0),
        (g!("prec_right(1)"), "prec_right", 2, 1),
        (g!("prec_dynamic()"), "prec_dynamic", 2, 0),
        (g!("prec_dynamic(1)"), "prec_dynamic", 2, 1),
        (g!("field()"), "field", 2, 0),
        (g!("field(name)"), "field", 2, 1),
        (g!("alias()"), "alias", 2, 0),
        (g!(r#"alias("a")"#), "alias", 2, 1),
        (g!("append()"), "append", 2, 0),
        (g!(r#"append("a")"#), "append", 2, 1),
        (g!("reserved()"), "reserved", 2, 0),
        (g!(r#"reserved("ctx")"#), "reserved", 2, 1),
        // 2-arg: too many
        (g!(r#"prec(1, "a", "b")"#), "prec", 2, 3),
        (g!(r#"prec_left(1, "a", "b")"#), "prec_left", 2, 3),
        (g!(r#"prec_right(1, "a", "b")"#), "prec_right", 2, 3),
        (g!(r#"prec_dynamic(1, "a", "b")"#), "prec_dynamic", 2, 3),
        (g!(r#"field(name, "a", "b")"#), "field", 2, 3),
        (g!(r#"alias("a", "b", "c")"#), "alias", 2, 3),
        (g!(r#"append([1], [2], [3])"#), "append", 2, 3),
        (g!(r#"reserved("ctx", "a", "b")"#), "reserved", 2, 3),
        // regexp (1-2 args)
        (g!("regexp()"), "regexp", 1, 0),
        (g!(r#"regexp("a", "b", "c")"#), "regexp", 2, 3),
    ];
    for &(src, name, expected, got) in cases {
        let err = dsl_err(src);
        let e = assert_err!(err, Parse);
        assert_eq!(
            e.kind,
            ParseErrorKind::WrongArgumentCount {
                name,
                expected,
                got
            },
            "wrong error for: {src}"
        );
    }
}

#[test]
fn error_builtin_arg_count_display() {
    let cases = [
        (
            r#"grammar { language: "test" } rule foo { prec(1) }"#,
            "'prec' takes 2 arguments, got 1",
        ),
        (
            r#"grammar { language: "test" } rule foo { repeat() }"#,
            "'repeat' takes 1 argument, got 0",
        ),
        (
            r#"grammar { language: "test" } rule foo { blank("x") }"#,
            "'blank' takes no arguments, got 1",
        ),
    ];
    for (src, expected_msg) in cases {
        let err = dsl_err(src);
        let e = assert_err!(err, Parse);
        assert_eq!(e.to_string(), expected_msg, "wrong message for: {src}");
    }
}

#[test]
fn trailing_comma_in_builtins() {
    macro_rules! g {
        ($expr:expr) => {
            concat!(r#"grammar { language: "test" } rule foo { "#, $expr, " }")
        };
    }
    // All of these have a trailing comma after the correct number of args.
    // They should parse successfully, not produce WrongArgumentCount.
    for src in [
        // 1-arg
        g!(r#"repeat("x",)"#),
        g!(r#"repeat1("x",)"#),
        g!(r#"optional("x",)"#),
        g!(r#"token("x",)"#),
        g!(r#"token_immediate("x",)"#),
        // 2-arg
        g!(r#"prec(1, "x",)"#),
        g!(r#"prec_left(1, "x",)"#),
        g!(r#"prec_right(1, "x",)"#),
        g!(r#"prec_dynamic(1, "x",)"#),
        g!(r#"field(name, "x",)"#),
        g!(r#"alias("x", foo,)"#),
        g!(r#"reserved("ctx", "x",)"#),
        // regexp (1 and 2 arg)
        g!(r#"regexp("pat",)"#),
        g!(r#"regexp("pat", "flags",)"#),
    ] {
        dsl(src);
    }
    // append needs a list context
    dsl(r#"grammar { language: "test", extras: append(["x"], ["y"],) } rule foo { "x" }"#);
}

#[test]
fn error_keyword_as_ident() {
    for src in [
        r#"grammar { language: "test" } rule seq { "x" }"#,
        r#"grammar { language: "test" } let token = "x" rule foo { "x" }"#,
        r#"grammar { language: "test" } fn repeat(x: rule) -> rule { x } rule foo { "x" }"#,
    ] {
        let err = dsl_err(src);
        let e = assert_err!(err, Parse);
        assert_eq!(
            e.kind,
            ParseErrorKind::KeywordAsIdent,
            "wrong error for: {src}"
        );
    }
}

#[test]
fn error_inherit_without_config() {
    let err = dsl_err(
        r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "test" }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert_eq!(e.kind, LowerErrorKind::InheritWithoutConfig);
}

#[test]
fn error_inherits_without_inherit() {
    let err = dsl_err(
        r#"
        grammar { language: "test", inherits: "not_inherit" }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert_eq!(e.kind, LowerErrorKind::InheritsWithoutInherit);
}

#[test]
fn error_config_field_unset() {
    // inherit_base_no_word has no word set, so base.word produces ConfigFieldUnset
    let err = dsl_err(
        r#"
        let base = inherit("inherit_base_no_word/grammar.tsg")
        grammar { language: "test", inherits: base, word: base.word }
        override rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert_eq!(e.kind, LowerErrorKind::ConfigFieldUnset);
}

#[test]
fn error_expected_rule_name_in_word() {
    let err = dsl_err(
        r#"
        grammar { language: "test", word: "not_a_name" }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::ExpectedRuleName);
}

#[test]
fn error_duplicate_grammar_field() {
    let err = dsl_err(
        r#"
        grammar { language: "test", word: foo, word: bar }
        rule foo { "x" }
        rule bar { "y" }
    "#,
    );
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::DuplicateGrammarField("word".into()));
    assert!(e.note.is_some());
}

#[test]
fn error_duplicate_object_key() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x = { a: 1, b: 2, a: 3 }
        rule foo { "x" }
    "#,
    );
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::DuplicateObjectKey("a".into()));
    assert!(e.note.is_some());
}

#[test]
fn error_too_many_raw_string_hashes() {
    let src = format!("let x = r{}\"test\"", "#".repeat(256));
    let err = dsl_err(&src);
    let e = assert_err!(err, Lex);
    assert_eq!(e.kind, LexErrorKind::TooManyHashes);
}

#[test]
fn error_nesting_too_deep() {
    let deep = "(-".repeat(300) + "1" + &")".repeat(300);
    let src = format!("grammar {{ language: \"test\" }} rule foo {{ {deep} }}");
    let err = dsl_err(&src);
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::NestingTooDeep);
}

#[test]
fn empty_list_compatible_with_any_list_type() {
    // Empty list in extras (list<rule> context)
    dsl(r#"
        grammar { language: "test", extras: [] }
        rule foo { "x" }
    "#);
    // Empty list in a let with explicit list<str> annotation
    dsl(r#"
        grammar { language: "test" }
        let x: list<str> = []
        rule foo { "x" }
    "#);
}

#[test]
fn empty_list_append() {
    // append([], concrete_list) should return the concrete list type
    let g = dsl(r#"
        grammar { language: "test", extras: append([], [regexp("\\s")]) }
        rule foo { "x" }
    "#);
    assert_eq!(g.extra_symbols.len(), 1);
}

#[test]
fn recursive_fn_depth_limit() {
    let err = dsl_err(
        r#"grammar { language: "test" } fn f(x: rule) -> rule { f(x) } rule program { f("x") }"#,
    );
    let e = assert_err!(err, Lower);
    assert!(matches!(e.kind, LowerErrorKind::CallDepthExceeded(_)));
}

/// Iterate over all `.tsg` files in `fuzz_regressions/` and verify none panic.
/// To add a new regression, just drop a `.tsg` file in that directory.
#[test]
fn fuzz_regressions() {
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/nativedsl/fuzz_regressions");
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", dir.display()))
        .filter_map(|e| {
            let path = e.ok()?.path();
            (path.extension()?.to_str()? == "tsg").then_some(path)
        })
        .collect();
    entries.sort();
    assert!(!entries.is_empty(), "no .tsg files in {}", dir.display());
    let dummy_path = dir.join("grammar.tsg");
    for path in &entries {
        let input = std::fs::read_to_string(path).unwrap();
        let result = std::panic::catch_unwind(|| {
            let _ = parse_native_dsl(&input, &dummy_path);
        });
        assert!(
            result.is_ok(),
            "panic on regression file: {}",
            path.display()
        );
    }
}

#[test]
fn error_expected_function_name() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x = { a: 1 }
        rule program { x.a("y") }
    "#,
    );
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::ExpectedFunctionName);
}

// ===== Typecheck error tests =====

#[test]
fn error_for_bindings_not_tuple() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let items: list<rule> = [identifier]
        rule identifier { regexp("[a-z]+") }
        rule program { seq(for (a: rule, b: rule) in items { a }) }
    "#,
    );
    let e = assert_err!(err, Type);
    matches!(
        e.kind,
        TypeErrorKind::ForBindingsNotTuple { bindings: 2, .. }
    );
}

// ===== Lower error tests =====

#[test]
fn error_inherit_rule_not_found() {
    let err = dsl_err(
        r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
        rule extra { base::nonexistent_rule }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert_eq!(
        e.kind,
        LowerErrorKind::InheritRuleNotFound("nonexistent_rule".into())
    );
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
