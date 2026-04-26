use super::*;
use crate::rules::{Precedence, Rule};
use std::path::Path;

// -- Imported function body assertions --

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

// -- Imported values --

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

// -- Transitive imports --

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

// -- Multiple imports --

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

// -- Import with only values, only functions --

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
fn delimited(item: rule_t) rule_t {
    seq(item, repeat(seq(DELIM, item)))
}
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

// -- Error cases --

#[test]
fn error_import_member_not_found() {
    let err = dsl_err(
        r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::nonexistent }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::ImportMemberNotFound("nonexistent".into())
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
        TypeErrorKind::ImportFunctionNotFound("nonexistent".into())
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
        TypeErrorKind::QualifiedCallOnNonModule(Ty::Object(InnerTy::Int))
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
            fn_name: "comma_sep1".into(),
            expected: 1,
            got: 2,
        }
    );
}

#[test]
fn error_import_disallowed_rule() {
    let dir = tempfile::tempdir().unwrap();
    let bad_helper = dir.path().join("bad.tsg");
    std::fs::write(&bad_helper, "rule foo { \"x\" }").unwrap();

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
    let DslError::Lower(e) = outer.inner.as_ref() else {
        panic!("expected Lower error, got {:?}", outer.inner)
    };
    assert_eq!(
        e.kind,
        LowerErrorKind::ModuleDisallowedItem(DisallowedItemKind::Rule)
    );
}

#[test]
fn error_import_disallowed_grammar_block() {
    let dir = tempfile::tempdir().unwrap();
    let bad_helper = dir.path().join("bad.tsg");
    std::fs::write(
        &bad_helper,
        "grammar { language: \"bad\" }\nfn f(x: rule_t) rule_t { x }",
    )
    .unwrap();

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
    let DslError::Lower(e) = outer.inner.as_ref() else {
        panic!("expected Lower error, got {:?}", outer.inner)
    };
    assert_eq!(
        e.kind,
        LowerErrorKind::ModuleDisallowedItem(DisallowedItemKind::GrammarBlock)
    );
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
    std::fs::write(&bad_helper, "let x = @@@").unwrap();

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
    std::fs::write(&bad, "let x = @@@").unwrap();
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
fn error_import_json_not_allowed() {
    let dir = tempfile::tempdir().unwrap();
    let json = dir.path().join("base.json");
    std::fs::write(
        &json,
        r#"{"name":"test","rules":{"program":{"type":"BLANK"}}}"#,
    )
    .unwrap();

    let input = format!(
        r#"
        let h = import("{}")
        grammar {{ language: "test" }}
        rule program {{ "x" }}
    "#,
        dsl_path(&json)
    );
    let err = parse_native_dsl(&input, Path::new(".")).unwrap_err();
    let e = assert_err!(err, Lower);
    assert_eq!(e.kind, LowerErrorKind::JsonImportNotAllowed);
}

// -- Edge cases --

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
            "let h = import(\"{}\")\nfn b_fn(x: rule_t) rule_t {{ prec(h::VAL, x) }}",
            dsl_path(&helpers_path)
        ),
    )
    .unwrap();
    std::fs::write(
        &c_path,
        format!(
            "let h = import(\"{}\")\nfn c_fn(x: rule_t) rule_t {{ prec(h::VAL, x) }}",
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

// -- Keyword as identifier in imported context --

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
