use super::*;
use crate::grammars::PrecedenceEntry;

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
            inline: append(grammar_config(base).inline, [_extra_inline]),
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
            extras: append(grammar_config(base).extras, [comment]),
        }
        rule comment { regexp(r"//.*") }
    "#);
    assert_eq!(g.extra_symbols.len(), 2);
}

#[test]
fn config_expr_let_binding() {
    let g = dsl(r#"
        let my_extras: list_rule_t = [regexp(r"\s"), comment]
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
        grammar { language: "derived", inherits: base, word: grammar_config(base).word }
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
        grammar { language: "derived", inherits: base, extras: grammar_config(base).extras }
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
fn config_all_fields_access_and_append() {
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar_full_config.tsg")
        grammar {
            language: "derived",
            inherits: base,
            extras: grammar_config(base).extras,
            externals: append(grammar_config(base).externals, [eof_marker]),
            inline: grammar_config(base).inline,
            supertypes: append(grammar_config(base).supertypes, [_statement]),
            word: grammar_config(base).word,
            conflicts: append(grammar_config(base).conflicts, [[keyword, _statement]]),
            precedences: append(grammar_config(base).precedences, [["unary", "binary"]]),
        }
        rule _statement { "stmt" }
        rule eof_marker { "EOF" }
    "#);
    assert_eq!(
        g.extra_symbols,
        vec![Rule::Pattern("\\s".into(), String::new())]
    );
    assert_eq!(
        g.external_tokens,
        vec![
            Rule::NamedSymbol("heredoc".into()),
            Rule::NamedSymbol("eof_marker".into()),
        ]
    );
    assert_eq!(g.variables_to_inline, vec!["_inline_rule"]);
    assert_eq!(g.supertype_symbols, vec!["_expression", "_statement"]);
    assert_eq!(g.word_token.as_deref(), Some("identifier"));
    assert_eq!(
        g.expected_conflicts,
        vec![
            vec!["identifier".to_string(), "keyword".to_string()],
            vec!["keyword".to_string(), "_statement".to_string()],
        ]
    );
    assert_eq!(
        g.precedence_orderings,
        vec![
            vec![
                PrecedenceEntry::Name("member".into()),
                PrecedenceEntry::Name("call".into()),
            ],
            vec![
                PrecedenceEntry::Name("unary".into()),
                PrecedenceEntry::Name("binary".into()),
            ],
        ]
    );
}

#[test]
fn externals_append_inherited_with_new_undeclared() {
    // Append a new undeclared external to inherited externals
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar_full_config.tsg")
        grammar {
            language: "derived",
            inherits: base,
            externals: append(grammar_config(base).externals, [_eof_marker]),
        }
    "#);
    // base has [heredoc], we add [_eof_marker]
    assert_eq!(g.external_tokens.len(), 2);
}

#[test]
fn externals_inherited_directly() {
    // grammar_config(base).externals used directly (no append)
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar_full_config.tsg")
        grammar {
            language: "derived",
            inherits: base,
            externals: grammar_config(base).externals,
        }
    "#);
    assert_eq!(g.external_tokens.len(), 1);
}

#[test]
fn inherit_from_json() {
    let dir = tempfile::tempdir().unwrap();
    let json_path = dir.path().join("base.json");
    std::fs::write(
        &json_path,
        r#"{"name":"base_json","rules":{"program":{"type":"STRING","value":"x"}},"extras":[],"conflicts":[],"precedences":[],"externals":[],"inline":[],"supertypes":[]}"#,
    )
    .unwrap();
    let input = format!(
        "let base = inherit(\"{}\")\ngrammar {{ language: \"extended\", inherits: base }}\nrule new_rule {{ \"y\" }}",
        json_path.display()
    );
    let g = parse_native_dsl(&input, Path::new(".")).unwrap();
    assert_eq!(g.name, "extended");
    assert!(g.variables.iter().any(|v| v.name == "program"));
    assert!(g.variables.iter().any(|v| v.name == "new_rule"));
}

#[test]
fn append_concatenates_lists() {
    let g = dsl(r#"
        grammar { language: "test" }
        let a: list_str_t = ["x", "y"]
        let b: list_str_t = ["z"]
        let c: list_str_t = append(a, b)
        rule program { choice(for (s: str_t) in c { s }) }
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

#[test]
fn inherit_from_grammar_that_imports() {
    // The base grammar itself imports a helper. This exercises the offset-based
    // module table: the base has global_id > 0 and its import has global_id > 1,
    // so the base's evaluator must use base_id to offset table indices.
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar_with_import.tsg")
        grammar { language: "derived", inherits: base }
        rule extra { "extra" }
    "#);
    assert_eq!(g.name, "derived");
    // Inherited rule from base uses the imported helper function
    assert!(rule_names(&g).contains(&"program"));
    assert!(rule_names(&g).contains(&"extra"));
}

#[test]
fn import_before_inherit_in_source_order() {
    // Import appears before inherit in source order. The import gets a
    // higher module index (assigned by node position during loading) but
    // is evaluated first. This tests that module eval order doesn't
    // corrupt the module values table.
    let g = dsl(r#"
        let h = import("import_helpers/helpers.tsg")
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
        rule new_rule { h::comma_sep1(identifier) }
    "#);
    assert_eq!(g.name, "derived");
    assert_eq!(
        *find_rule(&g, "new_rule"),
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
