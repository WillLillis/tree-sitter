use super::*;

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
        rule _statement { "s" }
        rule primary { "p" }
        rule arrow { "->" }
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

#[test]
fn conflicts_accepts_appended_list() {
    let g = dsl(r#"
        grammar {
            language: "test",
            conflicts: append([[a, b]], [[a, c]]),
        }
        rule program { "x" }
        rule a { "a" }
        rule b { "b" }
        rule c { "c" }
    "#);
    assert_eq!(
        g.expected_conflicts,
        vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["a".to_string(), "c".to_string()],
        ]
    );
}

#[test]
fn precedences_mixed_names_and_idents() {
    let g = dsl(r#"
        grammar {
            language: "test",
            precedences: [["member", call, "binary"]],
        }
        rule program { "x" }
        rule call { "c" }
    "#);
    assert_eq!(g.precedence_orderings.len(), 1);
    assert_eq!(g.precedence_orderings[0].len(), 3);
}

#[test]
fn empty_list_compatible_with_any_list_type() {
    // Empty list in extras (list_t<rule_t> context)
    dsl(r#"
        grammar { language: "test", extras: [] }
        rule foo { "x" }
    "#);
    // Empty list in a let with explicit list_t<str_t> annotation
    dsl(r#"
        grammar { language: "test" }
        let x: list_t<str_t> = []
        rule foo { "x" }
    "#);
}

#[test]
fn empty_list_append() {
    let g = dsl(r#"
        grammar { language: "test", extras: append([], [regexp("\\s")]) }
        rule foo { "x" }
    "#);
    assert_eq!(g.extra_symbols.len(), 1);
}

#[test]
fn raw_string_literal() {
    #[expect(clippy::needless_raw_string_hashes, reason = "false positive")]
    let g = dsl(r##"
        grammar { language: "test" }
        rule program { regexp(r#""[^"]*""#) }
    "##);
    assert!(matches!(&g.variables[0].rule, Rule::Pattern(p, _) if p == r#""[^"]*""#));
}

#[test]
fn string_with_escapes() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { "\n\t\\" }
    "#);
    assert_eq!(g.variables[0].rule, Rule::String("\n\t\\".into()));
}

#[test]
fn json_roundtrip() {
    #[expect(clippy::needless_raw_string_hashes, reason = "false positive")]
    let input = r##"
        grammar { language: "test", extras: [regexp(r"\s")] }
        rule program { repeat(pair) }
        rule pair { seq(field(key, string), ":", field(value, string)) }
        rule string { regexp(r#""[^"]*""#) }
    "##;
    let path = test_fixtures_dir().join("grammar.tsg");
    let mut grammar = parse_native_dsl(input, &path).unwrap();
    crate::parse_grammar::normalize_grammar(&mut grammar);
    let json_str =
        serde_json::to_string_pretty(&crate::nativedsl::serialize::grammar_to_json(&grammar))
            .expect("grammar JSON serialization should not fail");
    let reparsed = crate::parse_grammar::parse_grammar(&json_str).unwrap();
    assert_eq!(grammar.name, reparsed.name);
    assert_eq!(grammar.variables.len(), reparsed.variables.len());
    for (orig, re) in grammar.variables.iter().zip(reparsed.variables.iter()) {
        assert_eq!(orig.name, re.name);
    }
}

#[test]
fn empty_grammar() {
    let g = dsl(r#"grammar { language: "empty" }"#);
    assert_eq!(g.name, "empty");
    assert!(g.variables.is_empty());
}

#[test]
fn config_word_with_conflicts() {
    let g = dsl(r#"
        grammar {
            language: "test",
            word: identifier,
            conflicts: [[identifier, keyword]],
        }
        rule program { choice(identifier, keyword) }
        rule identifier { regexp(r"[a-z]+") }
        rule keyword { "if" }
    "#);
    assert_eq!(g.word_token.as_deref(), Some("identifier"));
    assert_eq!(g.expected_conflicts, vec![vec!["identifier", "keyword"]]);
}

#[test]
fn config_extras_with_inline() {
    let g = dsl(r#"
        grammar {
            language: "test",
            extras: [_ws],
            inline: [_ws],
        }
        rule program { "x" }
        rule _ws { regexp(r"\s") }
    "#);
    assert_eq!(g.extra_symbols.len(), 1);
    assert_eq!(g.variables_to_inline, vec!["_ws"]);
}

#[test]
fn config_all_fields_at_once() {
    let g = dsl(r#"
        grammar {
            language: "test",
            extras: [regexp(r"\s")],
            externals: [heredoc],
            inline: [_inline],
            supertypes: [_expr],
            word: ident,
            conflicts: [[ident, kw]],
            precedences: [["+" , ident]],
        }
        rule program { choice(_expr, heredoc) }
        rule _expr { choice(ident, kw) }
        rule ident { regexp(r"[a-z]+") }
        rule kw { "if" }
        rule _inline { "x" }
        rule heredoc { "<<" }
    "#);
    assert_eq!(g.extra_symbols.len(), 1);
    assert_eq!(g.external_tokens.len(), 1);
    assert_eq!(g.variables_to_inline, vec!["_inline"]);
    assert_eq!(g.supertype_symbols, vec!["_expr"]);
    assert_eq!(g.word_token.as_deref(), Some("ident"));
    assert_eq!(g.expected_conflicts.len(), 1);
    assert_eq!(g.precedence_orderings.len(), 1);
}

#[test]
fn externals_inline_list() {
    let g = dsl(r#"
        grammar { language: "test", externals: [heredoc, _eof] }
        rule program { "x" }
    "#);
    assert_eq!(g.external_tokens.len(), 2);
}

#[test]
fn externals_with_string_literals() {
    // String literals in externals (anonymous tokens) don't need pre-registration
    let g = dsl(r#"
        grammar { language: "test", externals: [heredoc, "||"] }
        rule program { "x" }
    "#);
    assert_eq!(g.external_tokens.len(), 2);
}

#[test]
fn externals_mixed_with_declared_rules() {
    let g = dsl(r#"
        grammar { language: "test", externals: [heredoc, comment] }
        rule program { "x" }
        rule comment { regexp("//.*") }
    "#);
    assert_eq!(g.external_tokens.len(), 2);
}

#[test]
fn externals_used_in_extras() {
    let g = dsl(r#"
        grammar {
            language: "test",
            externals: [_newline],
            extras: [regexp(r"\s"), _newline],
        }
        rule program { "x" }
    "#);
    assert_eq!(g.external_tokens.len(), 1);
    assert_eq!(g.extra_symbols.len(), 2);
}

#[test]
fn externals_used_in_rule_body() {
    let g = dsl(r#"
        grammar { language: "test", externals: [heredoc] }
        rule program { choice("x", heredoc) }
    "#);
    assert_eq!(g.external_tokens.len(), 1);
}

#[test]
fn externals_used_in_conflicts() {
    let g = dsl(r#"
        grammar {
            language: "test",
            externals: [heredoc],
            conflicts: [[program, heredoc]],
        }
        rule program { "x" }
    "#);
    assert_eq!(g.expected_conflicts.len(), 1);
}

#[test]
fn externals_append_inline_lists() {
    // append() of two inline lists
    let g = dsl(r#"
        grammar { language: "test", externals: append([heredoc], [_eof]) }
        rule program { "x" }
    "#);
    assert_eq!(g.external_tokens.len(), 2);
}

#[test]
fn externals_empty_list() {
    let g = dsl(r#"
        grammar { language: "test", externals: [] }
        rule program { "x" }
    "#);
    assert!(g.external_tokens.is_empty());
}

#[test]
fn externals_via_let_bindings() {
    // Direct and chained let indirection both resolve external names.
    for src in [
        r#"let ext: list_t<rule_t> = [heredoc]
        grammar { language: "test", externals: ext }
        rule program { "x" }"#,
        r#"let a: list_t<rule_t> = [heredoc]
        let b: list_t<rule_t> = a
        grammar { language: "test", externals: b }
        rule program { "x" }"#,
    ] {
        assert_eq!(
            dsl(src).external_tokens,
            vec![Rule::NamedSymbol("heredoc".into())]
        );
    }
}

#[test]
fn externals_via_let_with_append() {
    let g = dsl(r#"
        let ext: list_t<rule_t> = append([heredoc], [_eof])
        grammar { language: "test", externals: ext }
        rule program { "x" }
    "#);
    assert_eq!(g.external_tokens.len(), 2);
}

#[test]
fn externals_via_let_mixed_declared_and_undeclared() {
    let g = dsl(r#"
        let ext: list_t<rule_t> = [heredoc, comment]
        grammar { language: "test", externals: ext }
        rule program { "x" }
        rule comment { regexp("//.*") }
    "#);
    assert_eq!(g.external_tokens.len(), 2);
}

#[test]
fn externals_via_append_let_and_inline() {
    // append(let_binding, inline_list)
    let g = dsl(r#"
        let base_ext: list_t<rule_t> = [heredoc]
        grammar { language: "test", externals: append(base_ext, [_eof]) }
        rule program { "x" }
    "#);
    assert_eq!(g.external_tokens.len(), 2);
}

#[test]
fn externals_used_in_extras_via_let() {
    let g = dsl(r#"
        let ext: list_t<rule_t> = [_newline]
        grammar {
            language: "test",
            externals: ext,
            extras: [regexp(r"\s"), _newline],
        }
        rule program { "x" }
    "#);
    assert_eq!(g.external_tokens.len(), 1);
    assert_eq!(g.extra_symbols.len(), 2);
}

#[test]
fn externals_via_chained_let_with_append() {
    let g = dsl(r#"
        let a: list_t<rule_t> = [heredoc]
        let b: list_t<rule_t> = append(a, [_eof])
        grammar { language: "test", externals: b }
        rule program { "x" }
    "#);
    assert_eq!(g.external_tokens.len(), 2);
}

#[test]
fn error_externals_via_function_call() {
    let e = assert_err!(
        dsl_err(
            r#"
            fn mk_ext() list_t<rule_t> { [heredoc] }
            grammar { language: "test", externals: mk_ext() }
            rule program { "x" }
        "#
        ),
        Type
    );
    assert_eq!(e.kind, TypeErrorKind::InvalidExternalsExpression);
}
