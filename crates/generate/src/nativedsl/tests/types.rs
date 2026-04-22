use super::*;

// ===== str_t -> rule_t coercion in lists =====

#[test]
fn list_mixed_rule_and_str_coerces_to_list_rule() {
    // String literals are str_t, rule refs are rule_t. A mixed list should
    // coerce to list_t<rule_t> since str_t is a subtype of rule_t.
    let g = dsl(r#"
        grammar { language: "test" }
        let items: list_t<rule_t> = [program, "literal"]
        rule program { choice(for (r: rule_t) in items { r }) }
    "#);
    assert_eq!(g.variables.len(), 1);
}

#[test]
fn list_str_first_then_rule_coerces_to_list_rule() {
    // First element str, second rule - should widen to list_t<rule_t>
    let g = dsl(r#"
        grammar { language: "test" }
        let items: list_t<rule_t> = ["literal", program]
        rule program { "x" }
    "#);
    assert_eq!(g.variables.len(), 1);
}

#[test]
fn extras_mixed_rule_and_str() {
    // extras is list_t<rule_t> - should accept a mix of rule refs and strings
    let g = dsl(r#"
        grammar { language: "test", extras: [ws, "\n"] }
        rule program { "x" }
        rule ws { regexp(r"\s") }
    "#);
    assert_eq!(g.extra_symbols.len(), 2);
}

#[test]
fn object_mixed_str_and_rule_values_coerces() {
    // Object values with mixed str/rule should coerce field types
    dsl(r#"
        grammar { language: "test" }
        let o = { a: program, b: "literal" }
        rule program { o.a }
    "#);
}

#[test]
fn object_str_first_then_rule_coerces() {
    dsl(r#"
        grammar { language: "test" }
        let o = { a: "literal", b: program }
        rule program { o.b }
    "#);
}

// ===== Integer and nested list types =====

#[test]
fn list_int_literal_infers_list_int() {
    dsl(r#"
        grammar { language: "test" }
        let nums: list_t<int_t> = [1, 2, 3]
        rule program { "x" }
    "#);
}

#[test]
fn list_list_rule_literal_infers_list_list_rule() {
    dsl(r#"
        grammar { language: "test" }
        let pairs: list_t<list_t<rule_t>> = [[a, b], [c]]
        rule program { "x" }
        rule a { "a" }
        rule b { "b" }
        rule c { "c" }
    "#);
}

#[test]
fn list_list_str_subtype_of_list_list_rule() {
    dsl(r#"
        grammar { language: "test" }
        let groups: list_t<list_t<rule_t>> = [["a", "b"], ["c"]]
        rule program { "x" }
    "#);
}

#[test]
fn list_list_int_literal_infers_list_list_int() {
    dsl(r#"
        grammar { language: "test" }
        let nested: list_t<list_t<int_t>> = [[1], [2, 3]]
        rule program { "x" }
    "#);
}

#[test]
fn for_over_list_list_rule_yields_list_rule() {
    dsl(r#"
        grammar { language: "test" }
        let groups: list_t<list_t<rule_t>> = [[a], [b]]
        rule program { choice(for (inner: list_t<rule_t>) in groups { seq(for (r: rule_t) in inner { r }) }) }
        rule a { "a" }
        rule b { "b" }
    "#);
}
