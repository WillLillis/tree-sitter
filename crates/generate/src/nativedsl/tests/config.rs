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
    let path = test_grammars_dir().join("grammar.tsg");
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
