use super::*;

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
