use super::*;

#[test]
fn rule_set_macro_single_rule() {
    let g = dsl(r#"
        rules make_program() {
            rule program { "x" }
        }
        grammar { language: "test" }
        @make_program()
        "#);
    assert_eq!(g.variables.len(), 1);
    assert_eq!(g.variables[0].name, "program");
    assert_eq!(g.variables[0].rule, Rule::String("x".into()));
}

#[test]
fn rule_set_macro_multiple_rules() {
    let g = dsl(r#"
        rules pair() {
            rule a { "x" }
            rule b { "y" }
        }
        grammar { language: "test", start: a }
        @pair()
        "#);
    let names: Vec<&str> = g.variables.iter().map(|v| v.name.as_str()).collect();
    assert_eq!(names, vec!["a", "b"]);
}

#[test]
fn rule_set_macro_str_param_substitution() {
    let g = dsl(r#"
        rules lit(s: str_t) {
            rule program { s }
        }
        grammar { language: "test" }
        @lit("hello")
        "#);
    assert_eq!(g.variables[0].name, "program");
    assert_eq!(g.variables[0].rule, Rule::String("hello".into()));
}

#[test]
fn rule_set_macro_rule_param_substitution() {
    let g = dsl(r#"
        rules wrap(inner: rule_t) {
            rule program { seq("(", inner, ")") }
        }
        rule digit { "1" }
        grammar { language: "test", start: program }
        @wrap(digit)
        "#);
    let program = g.variables.iter().find(|v| v.name == "program").unwrap();
    assert_eq!(
        program.rule,
        Rule::seq(vec![
            Rule::String("(".into()),
            Rule::NamedSymbol("digit".into()),
            Rule::String(")".into()),
        ])
    );
}

#[test]
fn rule_set_macro_computed_name_via_concat() {
    let g = dsl(r#"
        rules suffixed(s: str_t) {
            rule @concat("foo_", s) { "x" }
        }
        grammar { language: "test", start: foo_bar }
        @suffixed("bar")
        "#);
    let names: Vec<&str> = g.variables.iter().map(|v| v.name.as_str()).collect();
    assert_eq!(names, vec!["foo_bar"]);
}

#[test]
fn rule_set_macro_multiple_invocations() {
    let g = dsl(r#"
        rules one(s: str_t) {
            rule @concat("k_", s) { s }
        }
        grammar { language: "test", start: k_a }
        @one("a")
        @one("b")
        @one("c")
        "#);
    let names: Vec<&str> = g.variables.iter().map(|v| v.name.as_str()).collect();
    assert_eq!(names, vec!["k_a", "k_b", "k_c"]);
}

#[test]
fn rule_set_macro_symref_in_expr_position() {
    // pair("foo") produces a_foo and b_foo; b_foo's body references
    // a_foo by computed name via @concat(...) in expression position.
    let g = dsl(r#"
        rules pair(s: str_t) {
            rule @concat("a_", s) { "x" }
            rule @concat("b_", s) { seq(@concat("a_", s), "y") }
        }
        grammar { language: "test", start: a_foo }
        @pair("foo")
        "#);
    let names: Vec<&str> = g.variables.iter().map(|v| v.name.as_str()).collect();
    assert_eq!(names, vec!["a_foo", "b_foo"]);
    let b = g.variables.iter().find(|v| v.name == "b_foo").unwrap();
    assert_eq!(
        b.rule,
        Rule::seq(vec![
            Rule::NamedSymbol("a_foo".into()),
            Rule::String("y".into()),
        ])
    );
}

#[test]
fn rule_set_macro_with_for_loop_over_param() {
    // For-loop inside a rule-set body iterates over a list_t<rule_t> macro
    // param. The iterable lives in ForConfig (indexed by ForId), separate from
    // the cloned body subtree - so substituting the param requires either a
    // fresh ForId per expansion or some other plumbing. Without that, `items`
    // is read from the macro template (where it's a MacroParam) instead of
    // the substituted arg.
    let g = dsl(r#"
        rules with_choices(items: list_t<rule_t>) {
            rule program {
                choice(for (x: rule_t) in items { x })
            }
        }
        rule a { "a" }
        rule b { "b" }
        grammar { language: "test", start: program }
        @with_choices([a, b])
    "#);
    let program = g.variables.iter().find(|v| v.name == "program").unwrap();
    assert_eq!(
        program.rule,
        Rule::choice(vec![
            Rule::NamedSymbol("a".into()),
            Rule::NamedSymbol("b".into()),
        ])
    );
}

#[test]
fn rule_set_macro_with_regular_rules_around() {
    let g = dsl(r#"
        rules extras() {
            rule helper_a { "ha" }
            rule helper_b { "hb" }
        }
        grammar { language: "test", start: program }
        rule program { seq(helper_a, helper_b) }
        @extras()
        "#);
    let names: Vec<&str> = g.variables.iter().map(|v| v.name.as_str()).collect();
    // start: program rotates program to position 0; remaining order preserved.
    assert_eq!(names, vec!["program", "helper_a", "helper_b"]);
}
