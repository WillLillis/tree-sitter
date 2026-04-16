use super::*;

#[test]
fn list_int_literal_infers_list_int() {
    dsl(r#"
        grammar { language: "test" }
        let nums: list_int_t = [1, 2, 3]
        rule program { "x" }
    "#);
}

#[test]
fn list_list_rule_literal_infers_list_list_rule() {
    dsl(r#"
        grammar { language: "test" }
        let pairs: list_list_rule_t = [[a, b], [c]]
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
        let groups: list_list_rule_t = [["a", "b"], ["c"]]
        rule program { "x" }
    "#);
}

#[test]
fn list_list_int_literal_infers_list_list_int() {
    dsl(r#"
        grammar { language: "test" }
        let nested: list_list_int_t = [[1], [2, 3]]
        rule program { "x" }
    "#);
}

#[test]
fn for_over_list_list_rule_yields_list_rule() {
    dsl(r#"
        grammar { language: "test" }
        let groups: list_list_rule_t = [[a], [b]]
        rule program { choice(for (inner: list_rule_t) in groups { seq(for (r: rule_t) in inner { r }) }) }
        rule a { "a" }
        rule b { "b" }
    "#);
}
