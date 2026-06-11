use super::*;

rule_tests! {
    let_binding_int {
        r#"grammar { language: "test" }
        let P: int_t = 5
        rule program { prec(P, "x") }"#,
        Rule::prec(Precedence::Integer(5), Rule::String("x".into()))
    }
    let_binding_negative_int {
        r#"grammar { language: "test" }
        let P: int_t = -1
        rule program { prec(P, "x") }"#,
        Rule::prec(Precedence::Integer(-1), Rule::String("x".into()))
    }
    let_binding_i32_min {
        r#"grammar { language: "test" }
        let P: int_t = -2147483648
        rule program { prec(P, "x") }"#,
        Rule::prec(Precedence::Integer(i32::MIN), Rule::String("x".into()))
    }
    let_binding_i32_max {
        r#"grammar { language: "test" }
        let P: int_t = 2147483647
        rule program { prec(P, "x") }"#,
        Rule::prec(Precedence::Integer(i32::MAX), Rule::String("x".into()))
    }
    int_double_negation {
        r#"grammar { language: "test" }
        let P: int_t = --5
        rule program { prec(P, "x") }"#,
        Rule::prec(Precedence::Integer(5), Rule::String("x".into()))
    }
    let_binding_typed_str {
        r#"grammar { language: "test" }
        let SEP: str_t = ","
        rule program { seq("a", SEP, "b") }"#,
        Rule::seq(vec![
            Rule::String("a".into()), Rule::String(",".into()), Rule::String("b".into()),
        ])
    }
    let_binding_object_field_access {
        r#"grammar { language: "test" }
        let PREC = { ADD: 1, MUL: 2 }
        rule add { prec_left(PREC.ADD, seq(expr, "+", expr)) }
        rule mul { prec_left(PREC.MUL, seq(expr, "*", expr)) }
        rule expr { choice(add, mul) }"#,
        Rule::prec_left(
            Precedence::Integer(1),
            Rule::seq(vec![
                Rule::NamedSymbol("expr".into()),
                Rule::String("+".into()),
                Rule::NamedSymbol("expr".into()),
            ])
        )
    }
    object_field_access_list_value {
        r#"grammar { language: "test" }
        let GROUPS = { kw: ["if", "else"], ops: ["+", "-"] }
        rule program {
            choice(
                for (k: str_t) in GROUPS.kw { k },
                for (o: str_t) in GROUPS.ops { o },
            )
        }"#,
        Rule::choice(vec![
            Rule::String("if".into()), Rule::String("else".into()),
            Rule::String("+".into()), Rule::String("-".into()),
        ])
    }
    object_with_str_values {
        r#"grammar { language: "test" }
        let ALIASES = { plus: "+", minus: "-" }
        rule program { seq(ALIASES.plus, ALIASES.minus) }"#,
        Rule::seq(vec![Rule::String("+".into()), Rule::String("-".into())])
    }
    function_expansion {
        r#"grammar { language: "test" }
        macro comma_sep(item: rule_t) rule_t { seq(item, repeat(seq(",", item))) }
        rule program { comma_sep(identifier) }
        rule identifier { regexp("[a-z]+") }"#,
        Rule::seq(vec![
            Rule::NamedSymbol("identifier".into()),
            Rule::choice(vec![
                Rule::repeat(Rule::seq(vec![
                    Rule::String(",".into()), Rule::NamedSymbol("identifier".into()),
                ])),
                Rule::Blank,
            ]),
        ])
    }
    function_multi_params {
        r#"grammar { language: "test" }
        macro wrap(before: str_t, content: rule_t, after: str_t) rule_t {
            seq(before, content, after)
        }
        rule program { wrap("(", identifier, ")") }
        rule identifier { regexp("[a-z]+") }"#,
        Rule::seq(vec![
            Rule::String("(".into()), Rule::NamedSymbol("identifier".into()), Rule::String(")".into()),
        ])
    }
    for_single_binding {
        r#"grammar { language: "test" }
        rule keywords { choice(for (kw: str_t) in ["if", "else", "while"] { kw }) }"#,
        Rule::choice(vec![
            Rule::String("if".into()), Rule::String("else".into()), Rule::String("while".into()),
        ])
    }
}

#[test]
fn for_in_list_str_body_via_let() {
    // For-loop spreads its body into a list literal under a let binding.
    let g = dsl(r#"
    let kws: list_t<str_t> = [for (kw: str_t) in ["if", "else", "while"] { kw }]
    grammar {
        language: "test",
        reserved: { global: kws },
    }
    rule program { "x" }"#);
    assert_eq!(g.reserved_words.len(), 1);
    assert_eq!(
        g.reserved_words[0].reserved_words,
        vec![
            Rule::String("if".into()),
            Rule::String("else".into()),
            Rule::String("while".into()),
        ],
    );
}

#[test]
fn for_in_list_rule_body() {
    // For-loop body builds rules (case-insensitive regexps), spread into the
    // grammar's extras. Mirrors the PHP keywords.tsg pattern.
    let g = dsl(r#"grammar {
        language: "test",
        extras: [for (kw: str_t) in ["if", "else"] { regexp(kw, "i") }],
    }
    rule program { "x" }"#);
    assert_eq!(
        g.extra_symbols,
        vec![Rule::pattern("if", "i"), Rule::pattern("else", "i"),],
    );
}

#[test]
fn for_in_list_tuple_destructure() {
    // Tuple destructuring inside a for-loop, spread into a list literal
    // consumed by reserved_words.
    let g = dsl(r#"grammar {
        language: "test",
        reserved: {
            global: [
                for (kw: str_t, _: int_t) in [("if", 1), ("else", 2)] { kw },
            ],
        },
    }
    rule program { "x" }"#);
    assert_eq!(
        g.reserved_words[0].reserved_words,
        vec![Rule::String("if".into()), Rule::String("else".into())],
    );
}

#[test]
fn for_in_list_mixed_with_concrete_elements() {
    // For-loop spread interleaved with concrete list elements; order is
    // preserved.
    let g = dsl(r#"grammar {
        language: "test",
        reserved: {
            global: ["alpha", for (kw: str_t) in ["mid1", "mid2"] { kw }, "omega"],
        },
    }
    rule program { "x" }"#);
    assert_eq!(
        g.reserved_words[0].reserved_words,
        vec![
            Rule::String("alpha".into()),
            Rule::String("mid1".into()),
            Rule::String("mid2".into()),
            Rule::String("omega".into()),
        ],
    );
}

#[test]
fn for_in_list_empty_iterable() {
    // Spreading a for-loop over an empty literal contributes zero elements.
    // The binding annotation makes the body type self-sufficient; no inference
    // from the iterable is needed.
    let g = dsl(r#"grammar {
        language: "test",
        reserved: {
            global: ["only", for (kw: str_t) in [] { kw }],
        },
    }
    rule program { "x" }"#);
    assert_eq!(
        g.reserved_words[0].reserved_words,
        vec![Rule::String("only".into())],
    );
}

#[test]
fn for_in_list_nested() {
    // A for-loop in spread position whose body contains another for-loop
    // spread expands the cartesian product into the surrounding list.
    let g = dsl(r#"
        let prefixes: list_t<str_t> = ["a", "b"]
        let suffixes: list_t<str_t> = ["1", "2", "3"]
        grammar {
            language: "test",
            reserved: {
                global: [for (p: str_t) in prefixes {
                    for (s: str_t) in suffixes { concat(p, s) }
                }],
            },
        }
        rule program { "x" }
    "#);
    assert_eq!(
        g.reserved_words[0].reserved_words,
        vec![
            Rule::String("a1".into()),
            Rule::String("a2".into()),
            Rule::String("a3".into()),
            Rule::String("b1".into()),
            Rule::String("b2".into()),
            Rule::String("b3".into()),
        ],
    );
}

#[test]
fn for_in_choice_nested() {
    // Nested for-loop spread inside choice() (seq/choice path), confirming
    // the same flatMap behavior as in list literals.
    let g = dsl(r#"
        let prefixes: list_t<str_t> = ["a", "b"]
        let suffixes: list_t<str_t> = ["1", "2"]
        grammar { language: "test" }
        rule program {
            choice(for (p: str_t) in prefixes {
                for (s: str_t) in suffixes { concat(p, s) }
            })
        }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::choice(vec![
            Rule::String("a1".into()),
            Rule::String("a2".into()),
            Rule::String("b1".into()),
            Rule::String("b2".into()),
        ]),
    );
}

#[test]
fn object_with_list_rule_values() {
    let g = dsl(r#"grammar {
        language: "test",
        reserved: { global: ["if", "else"], props: ["get", "set"] },
    }
    rule program { "x" }"#);
    assert_eq!(g.reserved_words.len(), 2);
    assert_eq!(g.reserved_words[0].name, "global");
    assert_eq!(g.reserved_words[1].name, "props");
}

#[test]
fn function_multiple_calls() {
    let g = dsl(r##"grammar { language: "test" }
        macro make_if(content: rule_t) rule_t { seq("#if", content, "#endif") }
        rule preproc_if { make_if(_statement) }
        rule preproc_block_if { make_if(_block_item) }
        rule _statement { "stmt" }
        rule _block_item { "item" }"##);
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
fn nested_function_calls() {
    let g = dsl(r#"grammar { language: "test" }
        macro comma_sep1(item: rule_t) rule_t { seq(item, repeat(seq(",", item))) }
        macro comma_sep(item: rule_t) rule_t { optional(comma_sep1(item)) }
        rule program { comma_sep(identifier) }
        rule identifier { regexp("[a-z]+") }"#);
    assert_eq!(g.variables[0].rule, comma_sep_rule("identifier"));
}

#[test]
fn for_tuple_destructure() {
    let g = dsl(r#"grammar { language: "test" }
        rule binary {
            choice(for (op: str_t, p: int_t) in [("+", 1), ("*", 2)] {
                prec_left(p, seq(expr, op, expr))
            })
        }
        rule expr { "x" }"#);
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
fn for_named_list_of_tuples() {
    // Tier-1 tuples: a list of tuples bound to a let (type inferred as
    // list_t<tuple_t<str_t, int_t>>), then destructured by a multi-binding
    // for-loop. Before tier-1 the table had to be written inline in the loop.
    let g = dsl(r#"grammar { language: "test" }
        let ops = [("+", 1), ("*", 2)]
        rule binary {
            choice(for (op: str_t, p: int_t) in ops {
                prec_left(p, seq(expr, op, expr))
            })
        }
        rule expr { "x" }"#);
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
fn tuple_annotations_check_against_values() {
    // Explicit `tuple_t<...>` and `list_t<tuple_t<...>>` annotations parse and
    // check against their values; the list mixes a tuple-typed variable with a
    // literal tuple, then destructures in a for-loop.
    let g = dsl(r#"grammar { language: "test" }
        let plus: tuple_t<str_t, int_t> = ("+", 1)
        let ops: list_t<tuple_t<str_t, int_t>> = [plus, ("*", 2)]
        rule binary {
            choice(for (op: str_t, p: int_t) in ops {
                prec_left(p, seq(expr, op, expr))
            })
        }
        rule expr { "x" }"#);
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
fn tuple_in_object_annotation_checks() {
    // obj_t<tuple_t<...>> parses and checks: objects hold tuples, preserving the
    // InnerTy == DataTy - Object invariant.
    dsl(r#"grammar { language: "test" }
        let m: obj_t<tuple_t<str_t, int_t>> = { plus: ("+", 1) }
        rule program { "x" }"#);
}

#[test]
fn tuple_value_bound_to_let() {
    // Each tuple is a first-class value bound to its own let, then collected
    // into a list literal and destructured - exercises tuples-as-values and
    // tuple-typed variables flowing into a list.
    let g = dsl(r#"grammar { language: "test" }
        let plus = ("+", 1)
        let times = ("*", 2)
        rule binary {
            choice(for (op: str_t, p: int_t) in [plus, times] {
                prec_left(p, seq(expr, op, expr))
            })
        }
        rule expr { "x" }"#);
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
fn for_with_rule_items() {
    let g = dsl(r#"grammar { language: "test" }
        rule program {
            choice(for (r: rule_t) in [seq("a", "b"), seq("c", "d")] { r })
        }"#);
    assert_eq!(
        g.variables[0].rule,
        Rule::choice(vec![
            Rule::seq(vec![Rule::String("a".into()), Rule::String("b".into())]),
            Rule::seq(vec![Rule::String("c".into()), Rule::String("d".into())]),
        ])
    );
}

#[test]
fn for_empty_list() {
    let g = dsl(r#"grammar { language: "test" }
        let ops: list_t<str_t> = []
        rule program {
            choice("fallback", for (s: str_t) in ops { s })
        }"#);
    assert_eq!(
        g.variables[0].rule,
        Rule::choice(vec![Rule::String("fallback".into())])
    );
}

#[test]
fn error_macro_param_shadows_let() {
    let e = assert_err!(
        dsl_err(
            r#"grammar { language: "test" }
            let X: str_t = "shadowed"
            macro wrap(X: rule_t) rule_t { seq("(", X, ")") }
            rule program { wrap(identifier) }
            rule identifier { regexp("[a-z]+") }"#
        ),
        Resolve
    );
    assert_eq!(e.kind, ResolveErrorKind::ShadowedBinding("X".into()));
}

#[test]
fn recursive_fn_depth_limit() {
    let err = dsl_err(
        r#"grammar { language: "test" } macro f(x: rule_t) rule_t { f(x) } rule program { f("x") }"#,
    );
    let e = assert_err!(err, Lower);
    assert!(matches!(e.kind, LowerErrorKind::CallDepthExceeded(_)));
}
