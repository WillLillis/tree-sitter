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
    // `.field` directly on an object literal (not a let-bound ident) - exercises
    // the parse_postfix `.field` loop, distinct from parse_ident_expr's path.
    object_literal_field_access {
        r#"grammar { language: "test" } rule program { prec({ P: 1 }.P, "x") }"#,
        Rule::prec(Precedence::Integer(1), Rule::String("x".into()))
    }
    // A single for-binding over a list of tuples binds the WHOLE tuple (regression:
    // it once took element 0); a later multi-binding loop then destructures it.
    for_single_tuple_binding_whole {
        r#"grammar { language: "test" }
        let copy = [for (p: tuple_t<str_t, int_t>) in [("kw_a", 1), ("kw_b", 2)] { p }]
        rule program { choice(for (name: str_t, n: int_t) in copy { prec(n, name) }) }"#,
        Rule::choice(vec![
            Rule::prec(Precedence::Integer(1), Rule::String("kw_a".into())),
            Rule::prec(Precedence::Integer(2), Rule::String("kw_b".into())),
        ])
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
    forward_reference_let {
        // A `let` referenced before its definition resolves to its value; let
        // bindings are visible regardless of source order.
        r#"grammar { language: "test" }
        rule program { MY_VAR }
        let MY_VAR: str_t = "x""#,
        Rule::String("x".into())
    }
    for_in_choice_nested {
        // Nested for-loop spread inside choice() (seq/choice path), confirming
        // the same flatMap behavior as in list literals.
        r#"
        let prefixes: list_t<str_t> = ["a", "b"]
        let suffixes: list_t<str_t> = ["1", "2"]
        grammar { language: "test" }
        rule program {
            choice(for (p: str_t) in prefixes {
                for (s: str_t) in suffixes { concat(p, s) }
            })
        }
    "#,
        Rule::choice(vec![
            Rule::String("a1".into()),
            Rule::String("a2".into()),
            Rule::String("b1".into()),
            Rule::String("b2".into()),
        ])
    }
    nested_function_calls {
        r#"grammar { language: "test" }
        macro comma_sep1(item: rule_t) rule_t { seq(item, repeat(seq(",", item))) }
        macro comma_sep(item: rule_t) rule_t { optional(comma_sep1(item)) }
        rule program { comma_sep(identifier) }
        rule identifier { regexp("[a-z]+") }"#,
        comma_sep_rule("identifier")
    }
    for_tuple_destructure {
        r#"grammar { language: "test" }
        rule binary {
            choice(for (op: str_t, p: int_t) in [("+", 1), ("*", 2)] {
                prec_left(p, seq(expr, op, expr))
            })
        }
        rule expr { "x" }"#,
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
    }
    tuple_elements_from_macro_calls {
        // A tuple element can be any scalar-typed expression, e.g. a macro call:
        // `tag("+")` returns its rule_t arg, lowering to the string rule "+".
        r#"grammar { language: "test" }
        macro tag(r: rule_t) rule_t { r }
        rule binary {
            choice(for (op: rule_t, p: int_t) in [(tag("+"), 1), (tag("*"), 2)] {
                prec_left(p, seq(expr, op, expr))
            })
        }
        rule expr { "x" }"#,
        plus_times_choice()
    }
    for_named_list_of_tuples {
        // A list of tuples bound to a let (inferred list_t<tuple_t<str_t, int_t>>),
        // then destructured by a multi-binding for-loop.
        r#"grammar { language: "test" }
        let ops = [("+", 1), ("*", 2)]
        rule binary {
            choice(for (op: str_t, p: int_t) in ops {
                prec_left(p, seq(expr, op, expr))
            })
        }
        rule expr { "x" }"#,
        plus_times_choice()
    }
    tuple_annotations_check_against_values {
        // Explicit tuple_t / list_t<tuple_t> annotations check against their values,
        // mixing a tuple-typed variable with a literal tuple.
        r#"grammar { language: "test" }
        let plus: tuple_t<str_t, int_t> = ("+", 1)
        let ops: list_t<tuple_t<str_t, int_t>> = [plus, ("*", 2)]
        rule binary {
            choice(for (op: str_t, p: int_t) in ops {
                prec_left(p, seq(expr, op, expr))
            })
        }
        rule expr { "x" }"#,
        plus_times_choice()
    }
    tuple_value_bound_to_let {
        // Each tuple is a first-class value bound to its own let, then collected into
        // a list and destructured.
        r#"grammar { language: "test" }
        let plus = ("+", 1)
        let times = ("*", 2)
        rule binary {
            choice(for (op: str_t, p: int_t) in [plus, times] {
                prec_left(p, seq(expr, op, expr))
            })
        }
        rule expr { "x" }"#,
        plus_times_choice()
    }
    macro_with_tuple_table_param {
        // A macro whose parameter is a list-of-tuples, iterated and destructured in
        // the body; the call passes a literal table.
        r#"grammar { language: "test" }
        macro binops(ops: list_t<tuple_t<str_t, int_t>>) rule_t {
            choice(for (op: str_t, p: int_t) in ops {
                prec_left(p, seq(expr, op, expr))
            })
        }
        rule binary { binops([("+", 1), ("*", 2)]) }
        rule expr { "x" }"#,
        plus_times_choice()
    }
    for_over_appended_tuple_lists {
        // append() of two tuple lists is itself a list-of-tuples; the for-loop
        // types and destructures the appended result with no special-casing.
        r#"grammar { language: "test" }
        let a = [("+", 1)]
        let b = [("*", 2)]
        rule binary {
            choice(for (op: str_t, p: int_t) in append(a, b) {
                prec_left(p, seq(expr, op, expr))
            })
        }
        rule expr { "x" }"#,
        plus_times_choice()
    }
    for_multi_binding_empty_literal {
        // A multi-binding for over an empty literal contributes zero iterations;
        // the binding annotations stand alone with no iterable type to infer.
        r#"grammar { language: "test" }
        rule program {
            choice("fallback", for (op: str_t, p: int_t) in [] { prec(p, op) })
        }"#,
        Rule::choice(vec![Rule::String("fallback".into())])
    }
    for_with_rule_items {
        r#"grammar { language: "test" }
        rule program {
            choice(for (r: rule_t) in [seq("a", "b"), seq("c", "d")] { r })
        }"#,
        Rule::choice(vec![
            Rule::seq(vec![Rule::String("a".into()), Rule::String("b".into())]),
            Rule::seq(vec![Rule::String("c".into()), Rule::String("d".into())]),
        ])
    }
    for_empty_list {
        r#"grammar { language: "test" }
        let ops: list_t<str_t> = []
        rule program {
            choice("fallback", for (s: str_t) in ops { s })
        }"#,
        Rule::choice(vec![Rule::String("fallback".into())])
    }
    macro_reads_let_defined_after_call_site {
        // A macro expanded at an early call site reads a let defined later, so lower
        // must evaluate that let on demand rather than assume source order (was a panic).
        r#"grammar { language: "test" }
        rule prog { m() }
        let b = "x"
        macro m() str_t { b }"#,
        Rule::String("x".into())
    }
    macro_for_loop_iterates_let_defined_after_call_site {
        // Same forward reference, but the macro body iterates the later let rather
        // than reading it as a scalar - exercises the for-loop iterable path.
        r#"grammar { language: "test" }
        rule prog { m() }
        let items: list_t<str_t> = ["a", "b"]
        macro m() rule_t { choice(for (x: str_t) in items { x }) }"#,
        Rule::choice(vec![Rule::String("a".into()), Rule::String("b".into())])
    }
}

// Keywords (seq/token/repeat) are contextual: usable as a rule, let, or macro
// name. Each context is its own case; the lone rule is the only variable.
rule_names_tests! {
    keyword_seq_as_rule_name {
        r#"grammar { language: "test" } rule seq { "x" }"#,
        vec!["seq"]
    }
    keyword_token_as_let_name {
        r#"grammar { language: "test" } let token = "x" rule foo { token }"#,
        vec!["foo"]
    }
    keyword_repeat_as_macro_name {
        r#"grammar { language: "test" } macro repeat(x: rule_t) rule_t { x } rule foo { repeat("a") }"#,
        vec!["foo"]
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
    // A for-loop spread over an empty literal contributes zero elements; the
    // binding annotation makes the body type self-sufficient.
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

/// The lowered `choice` of two `prec_left` binary rules for the `("+", 1)` /
/// `("*", 2)` table - the shared expected output for the tuple tests.
fn plus_times_choice() -> Rule {
    let binop = |op: &str, p: i32| {
        Rule::prec_left(
            Precedence::Integer(p),
            Rule::seq(vec![
                Rule::NamedSymbol("expr".into()),
                Rule::String(op.into()),
                Rule::NamedSymbol("expr".into()),
            ]),
        )
    };
    Rule::choice(vec![binop("+", 1), binop("*", 2)])
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

#[test]
fn macro_let_cycle_reported_as_circular() {
    // `b = m()` whose body reads `b` is a cycle resolve can't catch (macro resolves
    // at its def site); on-demand let eval detects the reentry as a circular binding.
    let err = dsl_err(
        r#"grammar { language: "test" }
        let b = m()
        macro m() str_t { b }
        rule prog { b }"#,
    );
    let e = assert_err!(err, Lower);
    assert_eq!(e.kind, LowerErrorKind::CircularLet("b".into()));
    assert_eq!(e.notes.len(), 1);
}

#[test]
fn mutual_let_cycle_reported_as_circular() {
    // Two lets defined in terms of each other: now that forward refs resolve, the
    // cycle is caught at typecheck with a note at the back-reference.
    let err = dsl_err(
        r#"grammar { language: "test" }
        let a = b
        let b = a
        rule program { "x" }"#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::CircularLet("a".into()));
    assert_eq!(e.notes.len(), 1);
}
