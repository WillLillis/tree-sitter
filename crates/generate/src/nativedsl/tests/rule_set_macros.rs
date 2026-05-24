use super::*;

#[test]
fn rule_set_macro_single_rule() {
    let g = dsl(r#"
        macro make_program() {
            rule program { "x" }
        }
        grammar { language: "test" }
        make_program()
        "#);
    assert_eq!(g.variables.len(), 1);
    assert_eq!(g.variables[0].name, "program");
    assert_eq!(g.variables[0].rule, Rule::String("x".into()));
}

#[test]
fn rule_set_macro_multiple_rules() {
    let g = dsl(r#"
        macro pair() {
            rule a { "x" }
            rule b { "y" }
        }
        grammar { language: "test", start: a }
        pair()
        "#);
    let names: Vec<&str> = g.variables.iter().map(|v| v.name.as_str()).collect();
    assert_eq!(names, vec!["a", "b"]);
}

#[test]
fn rule_set_macro_str_param_substitution() {
    let g = dsl(r#"
        macro lit(s: str_t) {
            rule program { s }
        }
        grammar { language: "test" }
        lit("hello")
        "#);
    assert_eq!(g.variables[0].name, "program");
    assert_eq!(g.variables[0].rule, Rule::String("hello".into()));
}

#[test]
fn rule_set_macro_rule_param_substitution() {
    let g = dsl(r#"
        macro wrap(inner: rule_t) {
            rule program { seq("(", inner, ")") }
        }
        rule digit { "1" }
        grammar { language: "test", start: program }
        wrap(digit)
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
        macro suffixed(s: str_t) {
            rule @concat("foo_", s) { "x" }
        }
        grammar { language: "test", start: foo_bar }
        suffixed("bar")
        "#);
    let names: Vec<&str> = g.variables.iter().map(|v| v.name.as_str()).collect();
    assert_eq!(names, vec!["foo_bar"]);
}

#[test]
fn rule_set_macro_multiple_invocations() {
    let g = dsl(r#"
        macro one(s: str_t) {
            rule @concat("k_", s) { s }
        }
        grammar { language: "test", start: k_a }
        one("a")
        one("b")
        one("c")
        "#);
    let names: Vec<&str> = g.variables.iter().map(|v| v.name.as_str()).collect();
    assert_eq!(names, vec!["k_a", "k_b", "k_c"]);
}

#[test]
fn rule_set_macro_symref_in_expr_position() {
    // pair("foo") produces a_foo and b_foo; b_foo's body references
    // a_foo by computed name via @concat(...) in expression position.
    let g = dsl(r#"
        macro pair(s: str_t) {
            rule @concat("a_", s) { "x" }
            rule @concat("b_", s) { seq(@concat("a_", s), "y") }
        }
        grammar { language: "test", start: a_foo }
        pair("foo")
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
fn rule_set_macro_with_regular_rules_around() {
    let g = dsl(r#"
        macro extras() {
            rule helper_a { "ha" }
            rule helper_b { "hb" }
        }
        grammar { language: "test", start: program }
        rule program { seq(helper_a, helper_b) }
        extras()
        "#);
    let names: Vec<&str> = g.variables.iter().map(|v| v.name.as_str()).collect();
    // start: program rotates program to position 0; remaining order preserved.
    assert_eq!(names, vec!["program", "helper_a", "helper_b"]);
}
