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
    // conflicts expression is no longer special-cased, so `append` works.
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
    // precedences inner list accepts string literals and rule references together.
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
    // Empty list in extras (list_rule_t context)
    dsl(r#"
        grammar { language: "test", extras: [] }
        rule foo { "x" }
    "#);
    // Empty list in a let with explicit list_str_t annotation
    dsl(r#"
        grammar { language: "test" }
        let x: list_str_t = []
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
    let path = test_fixtures_dir().join("grammar.tsg");
    let (grammar, json_str) = dsl_to_json(input, &path);
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

// ===== Externals resolution =====

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
    // Mix of undeclared externals and declared rules
    let g = dsl(r#"
        grammar { language: "test", externals: [heredoc, comment] }
        rule program { "x" }
        rule comment { regexp("//.*") }
    "#);
    assert_eq!(g.external_tokens.len(), 2);
}

#[test]
fn externals_used_in_extras() {
    // External token referenced in extras
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
    // External token referenced in a rule
    let g = dsl(r#"
        grammar { language: "test", externals: [heredoc] }
        rule program { choice("x", heredoc) }
    "#);
    assert_eq!(g.external_tokens.len(), 1);
}

#[test]
fn externals_used_in_conflicts() {
    // External token referenced in conflicts
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
fn externals_via_let_binding() {
    // External names in let bindings are resolved by following the variable
    // to its value expression during external name pre-registration.
    let g = dsl(r#"
        let ext: list_rule_t = [heredoc]
        grammar { language: "test", externals: ext }
        rule program { "x" }
    "#);
    assert_eq!(g.external_tokens, vec![Rule::NamedSymbol("heredoc".into())]);
}
