use super::*;

// ===== Let bindings =====

#[test]
fn let_binding_int() {
    let g = dsl(r#"
        grammar { language: "test" }
        let P: int_t = 5
        rule program { prec(P, "x") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::prec(Precedence::Integer(5), Rule::String("x".into()))
    );
}

#[test]
fn let_binding_negative_int() {
    let g = dsl(r#"
        grammar { language: "test" }
        let P: int_t = -1
        rule program { prec(P, "x") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::prec(Precedence::Integer(-1), Rule::String("x".into()))
    );
}

#[test]
fn let_binding_typed_str() {
    let g = dsl(r#"
        grammar { language: "test" }
        let SEP: str_t = ","
        rule program { seq("a", SEP, "b") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::seq(vec![
            Rule::String("a".into()),
            Rule::String(",".into()),
            Rule::String("b".into()),
        ])
    );
}

#[test]
fn let_binding_object_field_access() {
    let g = dsl(r#"
        grammar { language: "test" }
        let PREC = { ADD: 1, MUL: 2 }
        rule add { prec_left(PREC.ADD, seq(expr, "+", expr)) }
        rule mul { prec_left(PREC.MUL, seq(expr, "*", expr)) }
        rule expr { choice(add, mul) }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::prec_left(
            Precedence::Integer(1),
            Rule::seq(vec![
                Rule::NamedSymbol("expr".into()),
                Rule::String("+".into()),
                Rule::NamedSymbol("expr".into()),
            ])
        )
    );
}

#[test]
fn object_with_list_rule_values() {
    let g = dsl(r#"
        grammar {
            language: "test",
            reserved: { global: ["if", "else"], props: ["get", "set"] },
        }
        rule program { "x" }
    "#);
    assert_eq!(g.reserved_words.len(), 2);
    assert_eq!(g.reserved_words[0].name, "global");
    assert_eq!(g.reserved_words[1].name, "props");
}

#[test]
fn object_field_access_list_value() {
    // Access a field from an object with list values
    let g = dsl(r#"
        grammar { language: "test" }
        let GROUPS = { kw: ["if", "else"], ops: ["+", "-"] }
        rule program {
            choice(
                for (k: str_t) in GROUPS.kw { k },
                for (o: str_t) in GROUPS.ops { o },
            )
        }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::choice(vec![
            Rule::String("if".into()),
            Rule::String("else".into()),
            Rule::String("+".into()),
            Rule::String("-".into()),
        ])
    );
}

#[test]
fn object_with_str_values() {
    let g = dsl(r#"
        grammar { language: "test" }
        let ALIASES = { plus: "+", minus: "-" }
        rule program { seq(ALIASES.plus, ALIASES.minus) }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::seq(vec![Rule::String("+".into()), Rule::String("-".into()),])
    );
}

// ===== Functions =====

#[test]
fn function_expansion() {
    let g = dsl(r#"
        grammar { language: "test" }
        fn comma_sep(item: rule_t) -> rule_t {
            seq(item, repeat(seq(",", item)))
        }
        rule program { comma_sep(identifier) }
        rule identifier { regexp("[a-z]+") }
    "#);
    assert_eq!(g.variables.len(), 2);
    assert_eq!(
        g.variables[0].rule,
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

#[test]
fn function_multiple_calls() {
    let g = dsl(r##"
        grammar { language: "test" }
        fn make_if(content: rule_t) -> rule_t { seq("#if", content, "#endif") }
        rule preproc_if { make_if(_statement) }
        rule preproc_block_if { make_if(_block_item) }
        rule _statement { "stmt" }
        rule _block_item { "item" }
    "##);
    assert_eq!(
        g.variables[0].rule,
        Rule::seq(vec![
            Rule::String("#if".into()),
            Rule::NamedSymbol("_statement".into()),
            Rule::String("#endif".into()),
        ])
    );
    assert_eq!(
        g.variables[1].rule,
        Rule::seq(vec![
            Rule::String("#if".into()),
            Rule::NamedSymbol("_block_item".into()),
            Rule::String("#endif".into()),
        ])
    );
}

#[test]
fn function_multi_params() {
    let g = dsl(r#"
        grammar { language: "test" }
        fn wrap(before: str_t, content: rule_t, after: str_t) -> rule_t {
            seq(before, content, after)
        }
        rule program { wrap("(", identifier, ")") }
        rule identifier { regexp("[a-z]+") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::seq(vec![
            Rule::String("(".into()),
            Rule::NamedSymbol("identifier".into()),
            Rule::String(")".into()),
        ])
    );
}

#[test]
fn nested_function_calls() {
    let g = dsl(r#"
        grammar { language: "test" }
        fn comma_sep1(item: rule_t) -> rule_t { seq(item, repeat(seq(",", item))) }
        fn comma_sep(item: rule_t) -> rule_t { optional(comma_sep1(item)) }
        rule program { comma_sep(identifier) }
        rule identifier { regexp("[a-z]+") }
    "#);
    // comma_sep calls comma_sep1 internally
    assert_eq!(g.variables.len(), 2);
    // Just verify it doesn't error - the structure is complex
    assert!(matches!(&g.variables[0].rule, Rule::Choice(_)));
}

// ===== For expressions =====

#[test]
fn for_tuple_destructure() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule binary {
            choice(
                for (op: str_t, p: int_t) in [("+", 1), ("*", 2)] {
                    prec_left(p, seq(expr, op, expr))
                }
            )
        }
        rule expr { "x" }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::choice(vec![
            Rule::prec_left(
                Precedence::Integer(1),
                Rule::seq(vec![
                    Rule::NamedSymbol("expr".into()),
                    Rule::String("+".into()),
                    Rule::NamedSymbol("expr".into()),
                ])
            ),
            Rule::prec_left(
                Precedence::Integer(2),
                Rule::seq(vec![
                    Rule::NamedSymbol("expr".into()),
                    Rule::String("*".into()),
                    Rule::NamedSymbol("expr".into()),
                ])
            ),
        ])
    );
}

#[test]
fn for_single_binding() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule keywords {
            choice(for (kw: str_t) in ["if", "else", "while"] { kw })
        }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::choice(vec![
            Rule::String("if".into()),
            Rule::String("else".into()),
            Rule::String("while".into()),
        ])
    );
}

#[test]
fn for_single_element_tuple_destructure() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule keywords {
            choice(for (kw: str_t) in [("if"), ("else"), ("while")] { kw })
        }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::choice(vec![
            Rule::String("if".into()),
            Rule::String("else".into()),
            Rule::String("while".into()),
        ])
    );
}

#[test]
fn recursive_fn_depth_limit() {
    let err = dsl_err(
        r#"grammar { language: "test" } fn f(x: rule_t) -> rule_t { f(x) } rule program { f("x") }"#,
    );
    let e = assert_err!(err, Lower);
    assert!(matches!(e.kind, LowerErrorKind::CallDepthExceeded(_)));
}
