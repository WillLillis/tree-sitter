use super::*;
use crate::grammars::PrecedenceEntry;

#[test]
fn inherit_rules_and_config() {
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
            inline: append(grammar_config(base, inline), [_extra_inline]),
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
            extras: append(grammar_config(base, extras), [comment]),
        }
        rule comment { regexp(r"//.*") }
    "#);
    assert_eq!(g.extra_symbols.len(), 2);
}

#[test]
fn config_expr_let_binding() {
    let g = dsl(r#"
        let my_extras: list_t<rule_t> = [regexp(r"\s"), comment]
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
        grammar { language: "derived", inherits: base, word: grammar_config(base, word) }
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
        grammar { language: "derived", inherits: base, extras: grammar_config(base, extras) }
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
            extras: grammar_config(base, extras),
            externals: append(grammar_config(base, externals), [eof_marker]),
            inline: grammar_config(base, inline),
            supertypes: append(grammar_config(base, supertypes), [_statement]),
            word: grammar_config(base, word),
            conflicts: append(grammar_config(base, conflicts), [[keyword, _statement]]),
            precedences: append(grammar_config(base, precedences), [["unary", "binary"]]),
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
            externals: append(grammar_config(base, externals), [_eof_marker]),
        }
    "#);
    // base has [heredoc], we add [_eof_marker]
    assert_eq!(g.external_tokens.len(), 2);
}

#[test]
fn externals_inherited_directly() {
    // grammar_config(base, externals) used directly (no append)
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar_full_config.tsg")
        grammar {
            language: "derived",
            inherits: base,
            externals: grammar_config(base, externals),
        }
    "#);
    assert_eq!(g.external_tokens.len(), 1);
}

#[test]
fn append_concatenates_lists() {
    let g = dsl(r#"
        grammar { language: "test" }
        let a: list_t<str_t> = ["x", "y"]
        let b: list_t<str_t> = ["z"]
        let c: list_t<str_t> = append(a, b)
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
    // Inherited rule from base uses the imported helper macro
    assert!(rule_names(&g).contains(&"program"));
    assert!(rule_names(&g).contains(&"extra"));
}

#[test]
fn override_rule_can_access_base_let() {
    // override rule body referencing a let defined in the inherited grammar.
    // This exercises eager evaluation of the base's let bindings before any
    // override rule body that references them.
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base.tsg");
    std::fs::write(
        &base,
        r#"
        let GREETING: str_t = "hello"
        grammar { language: "base" }
        rule program { GREETING }
        "#,
    )
    .unwrap();

    let parent = dir.path().join("parent.tsg");
    let src = format!(
        r#"
        let base = inherit("{}")
        grammar {{ language: "derived", inherits: base }}
        override rule program {{ seq(base::GREETING, "!") }}
        "#,
        dsl_path(&base)
    );
    std::fs::write(&parent, &src).unwrap();
    let g = parse_native_dsl(&src, &parent).unwrap();
    assert_eq!(
        *find_rule(&g, "program"),
        Rule::seq(vec![Rule::String("hello".into()), Rule::String("!".into())])
    );
}

#[test]
fn nested_inheritance_merges_all_rules() {
    // child inherits parent inherits grandparent. All three levels' rules
    // should appear in the child's grammar.
    let g = dsl(r#"
        let parent = inherit("inherit_base/nested_parent.tsg")
        grammar { language: "child", inherits: parent }
        rule child_only { "child" }
    "#);
    assert_eq!(g.name, "child");
    let names = rule_names(&g);
    // Grandparent's rules
    assert!(names.contains(&"program"));
    assert!(names.contains(&"identifier"));
    assert!(names.contains(&"_gp_inline"));
    // Parent's added rules
    assert!(names.contains(&"parent_only"));
    assert!(names.contains(&"_parent_inline"));
    // Parent's override of statement should be present (not grandparent's)
    let statement = find_rule(&g, "statement");
    assert_eq!(
        *statement,
        Rule::choice(vec![
            Rule::NamedSymbol("identifier".into()),
            Rule::String("!".into())
        ])
    );
    // Child's own rule
    assert!(names.contains(&"child_only"));
}

#[test]
fn nested_inheritance_qualified_access_to_grandparent_via_parent() {
    // child references parent::identifier where identifier is defined in
    // grandparent. The merge in parent's lowering puts identifier into
    // parent.lowered.variables, so resolution succeeds without traversing.
    let g = dsl(r#"
        let parent = inherit("inherit_base/nested_parent.tsg")
        grammar { language: "child", inherits: parent }
        rule wrapper { parent::identifier }
    "#);
    let wrapper = find_rule(&g, "wrapper");
    assert!(matches!(wrapper, Rule::Pattern(p, _) if p == "[a-z]+"));
}

#[test]
fn nested_inheritance_explicit_chain_access() {
    // child reaches grandparent via parent::gp::identifier, traversing
    // parent's own inherit binding (`gp`).
    let g = dsl(r#"
        let parent = inherit("inherit_base/nested_parent.tsg")
        grammar { language: "child", inherits: parent }
        rule wrapper { parent::gp::identifier }
    "#);
    let wrapper = find_rule(&g, "wrapper");
    assert!(matches!(wrapper, Rule::Pattern(p, _) if p == "[a-z]+"));
}

#[test]
fn nested_inheritance_child_override_of_grandparent_rule() {
    // child overrides identifier (defined in grandparent, untouched by parent).
    // The override should win.
    let g = dsl(r#"
        let parent = inherit("inherit_base/nested_parent.tsg")
        grammar { language: "child", inherits: parent }
        override rule identifier { regexp("[A-Z]+") }
    "#);
    let identifier = find_rule(&g, "identifier");
    assert!(matches!(identifier, Rule::Pattern(p, _) if p == "[A-Z]+"));
}

#[test]
fn nested_inheritance_child_override_of_parent_override() {
    // parent overrides statement (originally from grandparent).
    // child overrides statement again. Child's version wins.
    let g = dsl(r#"
        let parent = inherit("inherit_base/nested_parent.tsg")
        grammar { language: "child", inherits: parent }
        override rule statement { "child_statement" }
    "#);
    let statement = find_rule(&g, "statement");
    assert_eq!(*statement, Rule::String("child_statement".into()));
}

#[test]
fn nested_inheritance_grammar_config_transitive() {
    // parent appended _parent_inline to gp's inline. child reads
    // grammar_config(parent, inline) which should return [_gp_inline, _parent_inline].
    let g = dsl(r#"
        let parent = inherit("inherit_base/nested_parent.tsg")
        grammar {
            language: "child",
            inherits: parent,
            inline: append(grammar_config(parent, inline), [_child_inline]),
        }
        rule _child_inline { "c" }
    "#);
    assert_eq!(
        g.variables_to_inline,
        vec!["_gp_inline", "_parent_inline", "_child_inline"]
    );
}

#[test]
fn nested_inheritance_chain_accesses_pre_override_rule() {
    // parent overrides `statement`; grandparent's original is `identifier`.
    // From the child, `parent::statement` should yield parent's override,
    // while `parent::gp::statement` should yield grandparent's original.
    let g = dsl(r#"
        let parent = inherit("inherit_base/nested_parent.tsg")
        grammar { language: "child", inherits: parent }
        rule via_parent { parent::statement }
        rule via_chain { parent::gp::statement }
    "#);
    let via_parent = find_rule(&g, "via_parent");
    let via_chain = find_rule(&g, "via_chain");
    // parent::statement -> parent's override: choice(identifier, "!")
    assert_eq!(
        *via_parent,
        Rule::choice(vec![
            Rule::NamedSymbol("identifier".into()),
            Rule::String("!".into())
        ])
    );
    // parent::gp::statement -> grandparent's original: identifier
    assert_eq!(*via_chain, Rule::NamedSymbol("identifier".into()));
}

#[test]
fn nested_inheritance_word_token_propagates() {
    // grandparent set word: identifier. parent didn't override. child should
    // see the inherited word_token.
    let g = dsl(r#"
        let parent = inherit("inherit_base/nested_parent.tsg")
        grammar { language: "child", inherits: parent }
    "#);
    assert_eq!(g.word_token.as_deref(), Some("identifier"));
}

#[test]
fn import_before_inherit_in_source_order() {
    // Import before inherit in source order. Tests that eval order doesn't
    // corrupt the module values table.
    let g = dsl(r#"
        let h = import("import_helpers/helpers.tsg")
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
        rule new_rule { h::comma_sep1(identifier) }
    "#);
    assert_eq!(g.name, "derived");
    assert_eq!(*find_rule(&g, "new_rule"), comma_sep1_rule("identifier"));
}
