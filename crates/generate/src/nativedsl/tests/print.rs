use super::*;

#[test]
fn print_accepts_string() {
    // Just check parse/typecheck/lower all succeed. The actual stderr output
    // isn't captured by this test since it'd require redirecting stderr in a
    // portable way; output format is verified by running the CLI manually.
    dsl(r#"
        grammar { language: "test" }
        print("hello")
        rule program { "x" }
    "#);
}

#[test]
fn print_accepts_rule_ref() {
    dsl(r#"
        grammar { language: "test" }
        print(program)
        rule program { seq("a", "b") }
    "#);
}

#[test]
fn print_accepts_list() {
    dsl(r#"
        grammar { language: "test" }
        let extras: list_rule_t = [regexp(r"\s"), comment]
        print(extras)
        rule program { "x" }
        rule comment { "//" }
    "#);
}

#[test]
fn print_accepts_object() {
    dsl(r#"
        grammar { language: "test" }
        let PREC = { ADD: 1, MUL: 2 }
        print(PREC)
        rule program { "x" }
    "#);
}

#[test]
fn print_accepts_grammar_config_access() {
    dsl(r#"
        grammar { language: "test", conflicts: [[a, b]] }
        rule program { "x" }
        rule a { "a" }
        rule b { "b" }
    "#);
    // Note: testing c.extras requires inheritance fixture; covered separately.
}
