use super::*;
use crate::rules::{Precedence, Rule};
use std::fmt::Write as _;
use std::path::Path;

#[test]
fn import_function_expands_comma_sep1() {
    let g = dsl(r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::comma_sep1(identifier) }
        rule identifier { regexp(r"[a-z]+") }
    "#);
    assert_eq!(*find_rule(&g, "program"), comma_sep1_rule("identifier"));
}

#[test]
fn import_function_expands_comma_sep() {
    let g = dsl(r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::comma_sep(identifier) }
        rule identifier { regexp(r"[a-z]+") }
    "#);
    assert_eq!(*find_rule(&g, "program"), comma_sep_rule("identifier"));
}

#[test]
fn import_function_with_string_param() {
    let g = dsl(r#"
        let n = import("import_helpers/nested.tsg")
        grammar { language: "test" }
        rule program { n::sep_by1(";", identifier) }
        rule identifier { regexp(r"[a-z]+") }
    "#);
    assert_eq!(*find_rule(&g, "program"), sep_by1_rule(";", "identifier"));
}

#[test]
fn import_function_intra_module_call() {
    let g = dsl(r#"
        let c = import("import_helpers/chained.tsg")
        grammar { language: "test" }
        rule program { c::double_wrap(identifier) }
        rule identifier { regexp(r"[a-z]+") }
    "#);
    // double_wrap(x) = wrap(seq(x, ",", x))
    // wrap(y) = seq("(", y, ")")
    assert_eq!(
        *find_rule(&g, "program"),
        Rule::seq(vec![
            Rule::String("(".into()),
            Rule::seq(vec![
                Rule::NamedSymbol("identifier".into()),
                Rule::String(",".into()),
                Rule::NamedSymbol("identifier".into()),
            ]),
            Rule::String(")".into()),
        ])
    );
}

#[test]
fn import_string_value() {
    let g = dsl(r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::GREETING }
    "#);
    assert_eq!(*find_rule(&g, "program"), Rule::String("hello".into()));
}

#[test]
fn import_let_name_does_not_collide_with_local() {
    let g = dsl(r#"
        let h = import("import_helpers/helpers.tsg")
        let GREETING: str_t = "goodbye"
        grammar { language: "test" }
        rule program { seq(h::GREETING, GREETING) }
    "#);
    assert_eq!(
        *find_rule(&g, "program"),
        Rule::seq(vec![
            Rule::String("hello".into()),
            Rule::String("goodbye".into()),
        ])
    );
}

#[test]
fn import_int_value_in_prec() {
    let g = dsl(r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program {
            prec_left(h::PREC.ADD, seq(program, "+", program))
        }
    "#);
    assert_eq!(
        *find_rule(&g, "program"),
        Rule::prec_left(
            Precedence::Integer(1),
            Rule::seq(vec![
                Rule::NamedSymbol("program".into()),
                Rule::String("+".into()),
                Rule::NamedSymbol("program".into()),
            ])
        )
    );
}

#[test]
fn import_value_reassigned_to_local() {
    let g = dsl(r#"
        let h = import("import_helpers/helpers.tsg")
        let p = h::PREC
        grammar { language: "test" }
        rule program {
            prec_left(p.MUL, seq(program, "*", program))
        }
    "#);
    assert_eq!(
        *find_rule(&g, "program"),
        Rule::prec_left(
            Precedence::Integer(2),
            Rule::seq(vec![
                Rule::NamedSymbol("program".into()),
                Rule::String("*".into()),
                Rule::NamedSymbol("program".into()),
            ])
        )
    );
}

#[test]
fn import_transitive_function_body() {
    let g = dsl(r#"
        let n = import("import_helpers/nested.tsg")
        grammar { language: "test" }
        rule program { n::sep_by1(",", identifier) }
        rule identifier { regexp(r"[a-z]+") }
    "#);
    assert_eq!(*find_rule(&g, "program"), sep_by1_rule(",", "identifier"));
}

#[test]
fn import_transitive_value() {
    let g = dsl(r#"
        let n = import("import_helpers/nested.tsg")
        grammar { language: "test" }
        rule program { prec(n::NESTED_VAL, "x") }
    "#);
    assert_eq!(
        *find_rule(&g, "program"),
        Rule::prec(Precedence::Integer(42), Rule::String("x".into()))
    );
}

#[test]
fn import_multiple_modules_body() {
    let g = dsl(r#"
        let h = import("import_helpers/helpers.tsg")
        let n = import("import_helpers/nested.tsg")
        grammar { language: "test" }
        rule program {
            choice(
                h::comma_sep1(identifier),
                n::sep_by1(";", identifier),
            )
        }
        rule identifier { regexp(r"[a-z]+") }
    "#);
    assert_eq!(
        *find_rule(&g, "program"),
        Rule::choice(vec![
            sep_by1_rule(",", "identifier"),
            sep_by1_rule(";", "identifier")
        ])
    );
}

#[test]
fn import_function_result_in_seq() {
    let g = dsl(r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { seq("{", h::comma_sep1(identifier), "}") }
        rule identifier { regexp(r"[a-z]+") }
    "#);
    assert_eq!(
        *find_rule(&g, "program"),
        Rule::seq(vec![
            Rule::String("{".into()),
            comma_sep1_rule("identifier"),
            Rule::String("}".into()),
        ])
    );
}

#[test]
fn import_module_values_only() {
    let g = dsl(r#"
        let m = import("import_helpers/minimal.tsg")
        grammar { language: "test" }
        rule program { prec(m::X, "x") }
    "#);
    assert_eq!(
        *find_rule(&g, "program"),
        Rule::prec(Precedence::Integer(1), Rule::String("x".into()))
    );
}

#[test]
fn import_function_uses_own_let_binding() {
    let dir = tempfile::tempdir().unwrap();
    let helper = dir.path().join("self_ref.tsg");
    std::fs::write(
        &helper,
        r#"
let DELIM = ","
macro delimited(item: rule_t) rule_t { seq(item, repeat(seq(DELIM, item))) }
"#,
    )
    .unwrap();

    let input = format!(
        r#"
        let h = import("{}")
        grammar {{ language: "test" }}
        rule program {{ h::delimited(identifier) }}
        rule identifier {{ regexp(r"[a-z]+") }}
    "#,
        dsl_path(&helper)
    );
    let g = parse_native_dsl(&input, Path::new(".")).unwrap();
    assert_eq!(*find_rule(&g, "program"), sep_by1_rule(",", "identifier"));
}

#[test]
fn error_import_member_not_found() {
    let err = dsl_err(
        r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::nonexistent }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert_eq!(
        e.kind,
        ResolveErrorKind::ImportMemberNotFound("nonexistent".into())
    );
}

#[test]
fn error_import_macro_used_as_value() {
    let err = dsl_err(
        r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::comma_sep1 }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::MacroUsedAsValue("h::comma_sep1".into())
    );
}

#[test]
fn error_import_function_not_found() {
    let err = dsl_err(
        r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::nonexistent("x") }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::ImportMacroNotFound("nonexistent".into())
    );
}

#[test]
fn error_qualified_call_on_non_module() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x = { a: 1 }
        rule program { x::something("y") }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::TypeMismatch {
            expected: Ty::ANY_MODULE,
            got: Ty::Data(DataTy::Object(InnerTy::Scalar(ScalarTy::Int))),
        }
    );
}

#[test]
fn error_import_wrong_arg_count() {
    let err = dsl_err(
        r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::comma_sep1("a", "b") }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::ArgCountMismatch {
            macro_name: "comma_sep1".into(),
            expected: 1,
            got: 2,
        }
    );
}

#[test]
fn error_import_disallowed_items() {
    for (content, expected) in [
        (
            "override rule foo { \"x\" }",
            DisallowedItemKind::OverrideRule,
        ),
        (
            "grammar { language: \"bad\" }\nmacro f(x: rule_t) rule_t { x }",
            DisallowedItemKind::GrammarBlock,
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let bad_helper = dir.path().join("bad.tsg");
        std::fs::write(&bad_helper, content).unwrap();
        let input = format!(
            "let h = import(\"{}\")\ngrammar {{ language: \"test\" }}\nrule program {{ \"x\" }}",
            dsl_path(&bad_helper)
        );
        let err = parse_native_dsl(&input, Path::new(".")).unwrap_err();
        let outer = assert_err!(err, Module);
        let DslError::Lower(e) = outer.inner.as_ref() else {
            panic!("expected Lower error, got {:?}", outer.inner)
        };
        assert_eq!(e.kind, LowerErrorKind::ModuleDisallowedItem(expected));
    }
}

#[test]
fn error_import_cycle() {
    let dir = tempfile::tempdir().unwrap();
    let a_path = dir.path().join("a.tsg");
    let b_path = dir.path().join("b.tsg");

    std::fs::write(
        &a_path,
        format!("let b = import(\"{}\")", dsl_path(&b_path)),
    )
    .unwrap();
    std::fs::write(
        &b_path,
        format!("let a = import(\"{}\")", dsl_path(&a_path)),
    )
    .unwrap();

    let input = format!(
        r#"
        let h = import("{}")
        grammar {{ language: "test" }}
        rule program {{ "x" }}
    "#,
        dsl_path(&a_path)
    );
    let err = parse_native_dsl(&input, Path::new(".")).unwrap_err();
    // Should detect cycle somewhere in the chain
    let is_cycle = matches!(
        &err,
        DslError::Module(m) if {
            fn has_cycle(e: &DslError) -> bool {
                match e {
                    DslError::Lower(l) => matches!(l.kind, LowerErrorKind::ModuleCycle),
                    DslError::Module(m) => has_cycle(&m.inner),
                    _ => false,
                }
            }
            has_cycle(&m.inner)
        }
    );
    assert!(is_cycle, "expected cycle error, got {err:?}");
}

#[test]
fn error_import_cycle_three_levels() {
    let dir = tempfile::tempdir().unwrap();
    let a_path = dir.path().join("a.tsg");
    let b_path = dir.path().join("b.tsg");
    let c_path = dir.path().join("c.tsg");
    std::fs::write(
        &a_path,
        format!("let b = import(\"{}\")", dsl_path(&b_path)),
    )
    .unwrap();
    std::fs::write(
        &b_path,
        format!("let c = import(\"{}\")", dsl_path(&c_path)),
    )
    .unwrap();
    std::fs::write(
        &c_path,
        format!("let a = import(\"{}\")", dsl_path(&a_path)),
    )
    .unwrap();
    let input = format!(
        r#"let h = import("{}")
        grammar {{ language: "test" }}
        rule program {{ "x" }}"#,
        dsl_path(&a_path)
    );
    let err = parse_native_dsl(&input, Path::new(".")).unwrap_err();
    fn has_cycle(e: &DslError) -> bool {
        match e {
            DslError::Lower(l) => matches!(l.kind, LowerErrorKind::ModuleCycle),
            DslError::Module(m) => has_cycle(&m.inner),
            _ => false,
        }
    }
    assert!(has_cycle(&err), "expected cycle error, got {err:?}");
}

#[test]
fn error_too_many_modules() {
    // Loading 256 helpers + 1 root tips the loader's module-id counter past
    // u8::MAX, which should surface as `ModuleTooMany`.
    let dir = tempfile::tempdir().unwrap();
    let mut imports = String::new();
    for i in 0..256 {
        let path = dir.path().join(format!("h{i}.tsg"));
        std::fs::write(&path, format!("let v{i}: str_t = \"x\"")).unwrap();
        let _ = writeln!(imports, "let h{i} = import(\"{}\")", dsl_path(&path));
    }
    let root = format!("{imports}grammar {{ language: \"test\" }}\nrule program {{ \"x\" }}\n");
    let root_path = dir.path().join("root.tsg");
    std::fs::write(&root_path, &root).unwrap();
    let err = parse_native_dsl(&root, &root_path).unwrap_err();
    let e = assert_err!(err, Lower);
    assert!(matches!(e.kind, LowerErrorKind::ModuleTooMany));
}

#[test]
fn error_import_bad_path() {
    let err = dsl_err(
        r#"
        let h = import("nonexistent/helpers.tsg")
        grammar { language: "test" }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert!(matches!(e.kind, LowerErrorKind::ModuleResolveFailed { .. }));
}

#[test]
fn error_import_nested_error() {
    let dir = tempfile::tempdir().unwrap();
    let bad_helper = dir.path().join("bad_syntax.tsg");
    std::fs::write(&bad_helper, "let x = ~~~").unwrap();

    let input = format!(
        r#"
        let h = import("{}")
        grammar {{ language: "test" }}
        rule program {{ "x" }}
    "#,
        dsl_path(&bad_helper)
    );
    let err = parse_native_dsl(&input, Path::new(".")).unwrap_err();
    let outer = assert_err!(err, Module);
    assert!(matches!(outer.inner.as_ref(), DslError::Lex(_)));
}

#[test]
fn error_import_transitive_nested_error() {
    // root -> middle.tsg -> bad.tsg (lex error)
    // Error should be double-wrapped: Module(Module(Lex(...)))
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("bad.tsg");
    std::fs::write(&bad, "let x = ~~~").unwrap();
    let middle = dir.path().join("middle.tsg");
    std::fs::write(&middle, format!("let h = import(\"{}\")", dsl_path(&bad))).unwrap();

    let input = format!(
        r#"
        let m = import("{}")
        grammar {{ language: "test" }}
        rule program {{ "x" }}
    "#,
        dsl_path(&middle)
    );
    let err = parse_native_dsl(&input, Path::new(".")).unwrap_err();
    let outer = assert_err!(err, Module);
    let inner = assert_err!(*outer.inner, Module);
    assert!(matches!(inner.inner.as_ref(), DslError::Lex(_)));
}

#[test]
fn import_function_receives_complex_expr() {
    let g = dsl(r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::comma_sep1(seq(identifier, ":", identifier)) }
        rule identifier { regexp(r"[a-z]+") }
    "#);
    let id = || Rule::NamedSymbol("identifier".into());
    let pair = Rule::seq(vec![id(), Rule::String(":".into()), id()]);
    assert_eq!(
        *find_rule(&g, "program"),
        Rule::seq(vec![
            pair.clone(),
            Rule::choice(vec![
                Rule::repeat(Rule::seq(vec![Rule::String(",".into()), pair])),
                Rule::Blank,
            ]),
        ])
    );
}

#[test]
fn import_function_receives_caller_let_binding() {
    let g = dsl(r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        let SEP: str_t = ";"
        rule program { h::sep_by(SEP, identifier) }
        rule identifier { regexp(r"[a-z]+") }
    "#);
    assert_eq!(*find_rule(&g, "program"), sep_by1_rule(";", "identifier"));
}

#[test]
fn import_function_receives_object_field() {
    let g = dsl(r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        let SEPS = { list: ",", stmt: ";" }
        rule program { h::sep_by(SEPS.stmt, identifier) }
        rule identifier { regexp(r"[a-z]+") }
    "#);
    assert_eq!(*find_rule(&g, "program"), sep_by1_rule(";", "identifier"));
}

#[test]
fn import_same_module_twice() {
    let g = dsl(r#"
        let h1 = import("import_helpers/helpers.tsg")
        let h2 = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { choice(h1::GREETING, h2::GREETING) }
    "#);
    assert_eq!(
        *find_rule(&g, "program"),
        Rule::choice(vec![
            Rule::String("hello".into()),
            Rule::String("hello".into()),
        ])
    );
}

#[test]
fn import_diamond() {
    // A imports B and C. Both B and C import helpers.tsg.
    // Should work (each gets its own copy of helpers).
    let dir = tempfile::tempdir().unwrap();
    let b_path = dir.path().join("b.tsg");
    let c_path = dir.path().join("c.tsg");
    let helpers_path = dir.path().join("helpers.tsg");

    std::fs::write(&helpers_path, "let VAL = 10").unwrap();
    std::fs::write(
        &b_path,
        format!(
            "let h = import(\"{}\")\nmacro b_fn(x: rule_t) rule_t {{ prec(h::VAL, x) }}",
            dsl_path(&helpers_path)
        ),
    )
    .unwrap();
    std::fs::write(
        &c_path,
        format!(
            "let h = import(\"{}\")\nmacro c_fn(x: rule_t) rule_t {{ prec(h::VAL, x) }}",
            dsl_path(&helpers_path)
        ),
    )
    .unwrap();

    let input = format!(
        r#"
        let b = import("{}")
        let c = import("{}")
        grammar {{ language: "test" }}
        rule program {{ choice(b::b_fn("a"), c::c_fn("b")) }}
    "#,
        dsl_path(&b_path),
        dsl_path(&c_path)
    );
    let g = parse_native_dsl(&input, Path::new(".")).unwrap();
    assert_eq!(
        *find_rule(&g, "program"),
        Rule::choice(vec![
            Rule::prec(Precedence::Integer(10), Rule::String("a".into())),
            Rule::prec(Precedence::Integer(10), Rule::String("b".into())),
        ])
    );
}

#[test]
fn import_diamond_dedups_shared_leaf() {
    // 128 helpers each importing the same leaf. Without loader-wide dedup,
    // the leaf would be loaded once per helper (128 leaf instances + 128
    // helpers + 1 root = 257, tripping the u8 module limit). With dedup,
    // the leaf loads once total (128 + 1 + 1 = 130, well under 256).
    let dir = tempfile::tempdir().unwrap();
    let leaf = dir.path().join("leaf.tsg");
    std::fs::write(&leaf, "let X: int_t = 10").unwrap();

    let mut imports = String::new();
    for i in 0..128 {
        let h = dir.path().join(format!("h{i}.tsg"));
        std::fs::write(
            &h,
            format!(
                "let l = import(\"{}\")\nmacro f{i}(r: rule_t) rule_t {{ prec(l::X, r) }}",
                dsl_path(&leaf)
            ),
        )
        .unwrap();
        let _ = writeln!(imports, "let h{i} = import(\"{}\")", dsl_path(&h));
    }
    let root =
        format!("{imports}grammar {{ language: \"test\" }}\nrule program {{ h0::f0(\"x\") }}\n");
    parse_native_dsl(&root, Path::new(".")).unwrap();
}

#[test]
fn helper_rule_collision_errors() {
    // Two helpers each defining `expression` -> resolver collision error.
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.tsg");
    let b = dir.path().join("b.tsg");
    std::fs::write(&a, "rule expression { \"a\" }").unwrap();
    std::fs::write(&b, "rule expression { \"b\" }").unwrap();

    let input = format!(
        r#"
        let a = import("{}")
        let b = import("{}")
        grammar {{ language: "test" }}
        rule program {{ expression }}
    "#,
        dsl_path(&a),
        dsl_path(&b)
    );
    let err = parse_native_dsl(&input, Path::new(".")).unwrap_err();
    let outer = assert_err!(err, Resolve);
    assert_eq!(
        outer.kind,
        ResolveErrorKind::DuplicateDeclaration("expression".into())
    );
}

#[test]
fn override_helper_rule() {
    // Helper defines `digit`. Root overrides it. The override should appear
    // in the final grammar; the original is accessible via h::digit (which
    // inlines).
    let dir = tempfile::tempdir().unwrap();
    let helper = dir.path().join("h.tsg");
    std::fs::write(&helper, r#"rule digit { regexp(r"[0-9]") }"#).unwrap();

    let input = format!(
        r#"
        let h = import("{}")
        grammar {{ language: "test" }}
        rule program {{ digit }}
        override rule digit {{ choice(h::digit, "x") }}
    "#,
        dsl_path(&helper)
    );
    let g = parse_native_dsl(&input, Path::new(".")).unwrap();
    // `digit` in final grammar = the override body, with h::digit inlined to
    // the original pattern.
    assert_eq!(
        *find_rule(&g, "digit"),
        Rule::choice(vec![
            Rule::Pattern(r"[0-9]".into(), String::new()),
            Rule::String("x".into()),
        ])
    );
}

#[test]
fn helper_rule_transitive_promotion() {
    // root -> A -> B. B has rule `inner`. Root references `inner` by bare
    // name (no qualified access). Transitive promotion should make this
    // work.
    let dir = tempfile::tempdir().unwrap();
    let b = dir.path().join("b.tsg");
    let a = dir.path().join("a.tsg");
    std::fs::write(&b, r#"rule inner { "leaf" }"#).unwrap();
    std::fs::write(
        &a,
        format!(
            "let b = import(\"{}\")\nrule middle {{ inner }}",
            dsl_path(&b)
        ),
    )
    .unwrap();

    let input = format!(
        r#"
        let a = import("{}")
        grammar {{ language: "test" }}
        rule program {{ seq(middle, inner) }}
    "#,
        dsl_path(&a)
    );
    let g = parse_native_dsl(&input, Path::new(".")).unwrap();
    assert!(rule_names(&g).contains(&"inner"));
    assert!(rule_names(&g).contains(&"middle"));
    assert_eq!(*find_rule(&g, "inner"), Rule::String("leaf".into()));
}

#[test]
fn helper_rule_qualified_inlines() {
    // `h::rule_name` inlines the rule body, mirroring `base::rule_name` for
    // inherited grammars.
    let dir = tempfile::tempdir().unwrap();
    let helper = dir.path().join("h.tsg");
    std::fs::write(&helper, r#"rule greeting { "hello" }"#).unwrap();

    let input = format!(
        r#"
        let h = import("{}")
        grammar {{ language: "test" }}
        rule program {{ seq(h::greeting, "!") }}
    "#,
        dsl_path(&helper)
    );
    let g = parse_native_dsl(&input, Path::new(".")).unwrap();
    // h::greeting inlined - program body has the literal "hello", not a
    // NamedSymbol reference.
    assert_eq!(
        *find_rule(&g, "program"),
        Rule::seq(vec![Rule::String("hello".into()), Rule::String("!".into())])
    );
}

#[test]
fn helper_rule_materialized_into_grammar() {
    // Helper defines `digit`. Root references it by bare name. The rule
    // should materialize as a top-level Variable in the final grammar,
    // accessible as a NamedSymbol.
    let dir = tempfile::tempdir().unwrap();
    let helper = dir.path().join("h.tsg");
    std::fs::write(&helper, r#"rule digit { regexp(r"[0-9]") }"#).unwrap();

    let input = format!(
        r#"
        let h = import("{}")
        grammar {{ language: "test" }}
        rule program {{ repeat1(digit) }}
    "#,
        dsl_path(&helper)
    );
    let g = parse_native_dsl(&input, Path::new(".")).unwrap();
    assert_eq!(rule_names(&g), vec!["program", "digit"]);
    assert_eq!(
        *find_rule(&g, "program"),
        Rule::repeat(Rule::NamedSymbol("digit".into()))
    );
    assert_eq!(
        *find_rule(&g, "digit"),
        Rule::Pattern(r"[0-9]".into(), String::new())
    );
}

#[test]
fn error_helper_rule_collides_with_root_rule() {
    // Helper rules materialize bare-named into the root grammar. A name
    // collision between a helper rule and a root rule is a duplicate
    // declaration, not silent shadowing - tree-sitter's grammar JSON can't
    // hold two variables with the same name.
    let dir = tempfile::tempdir().unwrap();
    let helper = dir.path().join("h.tsg");
    std::fs::write(&helper, r#"rule expression { "from_helper" }"#).unwrap();

    let input = format!(
        r#"
        let h = import("{}")
        grammar {{ language: "test" }}
        rule program {{ expression }}
        rule expression {{ "from_root" }}
    "#,
        dsl_path(&helper)
    );
    let err = parse_native_dsl(&input, Path::new("."));
    let e = assert_err!(err.unwrap_err(), Resolve);
    assert_eq!(
        e.kind,
        ResolveErrorKind::DuplicateDeclaration("expression".into())
    );
}

#[test]
fn helper_can_define_rules() {
    // Mutually-recursive helper rules with intra-helper references plus a
    // reference to a helper-declared external. Rules materialize into the
    // root grammar's variables.
    let dir = tempfile::tempdir().unwrap();
    let helper = dir.path().join("exp.tsg");
    std::fs::write(
        &helper,
        r"
        external _paren_open
        external _paren_close
        rule expression { choice(application, seq(_paren_open, expression, _paren_close)) }
        rule application { seq(expression, expression) }
    ",
    )
    .unwrap();

    let input = format!(
        r#"
        let exp = import("{}")
        grammar {{ language: "test", externals: [exp::_paren_open, exp::_paren_close] }}
        rule program {{ expression }}
    "#,
        dsl_path(&helper)
    );
    let g = parse_native_dsl(&input, Path::new(".")).unwrap();
    assert_eq!(rule_names(&g), vec!["program", "expression", "application"]);
}

#[test]
fn helper_external_qualified_in_rule_body() {
    let dir = tempfile::tempdir().unwrap();
    let helper = dir.path().join("ext.tsg");
    std::fs::write(&helper, "external _foo\nexternal _bar\n").unwrap();

    let input = format!(
        r#"
        let e = import("{}")
        grammar {{ language: "test", externals: [e::_foo, e::_bar] }}
        rule program {{ seq(e::_foo, e::_bar) }}
    "#,
        dsl_path(&helper)
    );
    let g = parse_native_dsl(&input, Path::new(".")).unwrap();
    assert_eq!(g.external_tokens.len(), 2);
    assert_eq!(
        *find_rule(&g, "program"),
        Rule::seq(vec![
            Rule::NamedSymbol("_foo".into()),
            Rule::NamedSymbol("_bar".into()),
        ])
    );
}

#[test]
fn helper_external_member_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let helper = dir.path().join("ext.tsg");
    std::fs::write(&helper, "external _foo\n").unwrap();

    let input = format!(
        r#"
        let e = import("{}")
        grammar {{ language: "test", externals: [e::_missing] }}
        rule program {{ "x" }}
    "#,
        dsl_path(&helper)
    );
    let err = parse_native_dsl(&input, Path::new("."));
    let e = assert_err!(err.unwrap_err(), Resolve);
    assert_eq!(
        e.kind,
        ResolveErrorKind::ImportMemberNotFound("_missing".into())
    );
}

#[test]
fn import_empty_module() {
    let dir = tempfile::tempdir().unwrap();
    let empty = dir.path().join("empty.tsg");
    std::fs::write(&empty, "// nothing here\n").unwrap();

    let input = format!(
        r#"
        let e = import("{}")
        grammar {{ language: "test" }}
        rule program {{ "x" }}
    "#,
        dsl_path(&empty)
    );
    let g = parse_native_dsl(&input, Path::new(".")).unwrap();
    assert_eq!(*find_rule(&g, "program"), Rule::String("x".into()));
}

#[test]
fn import_value_in_config_extras() {
    let dir = tempfile::tempdir().unwrap();
    let helper = dir.path().join("ws.tsg");
    std::fs::write(&helper, r#"let WS = [regexp(r"\s"), regexp(r"//[^\n]*")]"#).unwrap();

    let input = format!(
        r#"
        let ws = import("{}")
        grammar {{ language: "test", extras: ws::WS }}
        rule program {{ "x" }}
    "#,
        dsl_path(&helper)
    );
    let g = parse_native_dsl(&input, Path::new(".")).unwrap();
    assert_eq!(g.extra_symbols.len(), 2);
    assert_eq!(
        g.extra_symbols[0],
        Rule::Pattern(r"\s".into(), String::new())
    );
    assert_eq!(
        g.extra_symbols[1],
        Rule::Pattern(r"//[^\n]*".into(), String::new())
    );
}

#[test]
fn import_keyword_as_rule_name() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { import }
        rule import { "import_stmt" }
    "#);
    assert_eq!(g.variables[1].name, "import");
}

#[test]
fn chained_let_alias_to_import_module() {
    let g = dsl(r#"
        let h = import("import_helpers/helpers.tsg")
        let h2 = h
        grammar { language: "test" }
        rule program { h2::GREETING }
    "#);
    assert_eq!(*find_rule(&g, "program"), Rule::String("hello".into()));
}

#[test]
fn import_helper_rule_set_macro_expands_locally() {
    // A helper that defines a `rules` macro and calls it via `@pair(...)` at
    // its own item position should expand into ExpandedRule items that flow
    // into the importing grammar's variable table.
    let g = dsl(r#"
        let h = import("import_helpers/rule_set_self.tsg")
        grammar { language: "test", start: program }
        rule program { seq(a_helper, b_helper) }
    "#);
    let names: Vec<&str> = g.variables.iter().map(|v| v.name.as_str()).collect();
    assert!(names.contains(&"a_helper"), "missing a_helper in {names:?}");
    assert!(names.contains(&"b_helper"), "missing b_helper in {names:?}");
    assert_eq!(*find_rule(&g, "a_helper"), Rule::String("x".into()));
    assert_eq!(*find_rule(&g, "b_helper"), Rule::String("y".into()));
}

#[test]
fn import_call_depth_shared_across_modules() {
    let err = dsl_err(
        r#"
        let h = import("import_helpers/recursive.tsg")
        grammar { language: "test" }
        rule program { h::recurse("x") }
    "#,
    );
    let e = assert_err!(err, Lower);
    let LowerErrorKind::CallDepthExceeded(trace) = &e.kind else {
        panic!("expected CallDepthExceeded");
    };
    let fixtures = test_fixtures_dir();
    let grammar_path = fixtures.join("grammar.tsg");
    let recursive_path = fixtures.join("import_helpers/recursive.tsg");
    // First frame: call site in root grammar
    assert_eq!(trace[0], ("recurse".into(), grammar_path, 4, 24));
    // Remaining frames: self-recursive calls within the imported helper
    for frame in &trace[1..] {
        assert_eq!(*frame, ("recurse".into(), recursive_path.clone(), 4, 5));
    }
}
